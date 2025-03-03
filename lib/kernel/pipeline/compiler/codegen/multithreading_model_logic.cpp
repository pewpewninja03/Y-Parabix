#include "../pipeline_compiler.hpp"
#include <pthread.h>
#include <llvm/IR/Verifier.h>

enum PipelineStateObjectField : unsigned {
    SHARED_STATE_PARAM
    , THREAD_LOCAL_PARAM
    , PIPELINE_PARAMS
    , PIPELINE_PARAM_PADDING
    , INITIAL_SEG_NUMBER
    , FIXED_NUMBER_OF_THREADS
    , ACCUMULATED_SEGMENT_TIME
    , ACCUMULATED_SYNCHRONIZATION_TIME
    , CURRENT_THREAD_ID
    , CURRENT_THREAD_STATUS_FLAG
    , TERMINATION_SIGNAL
    // -------------------
    , THREAD_STRUCT_SIZE
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concat
 ** ------------------------------------------------------------------------------------------------------------- */
StringRef concat(StringRef A, StringRef B, SmallVector<char, 256> & tmp) {
    Twine C = A + B;
    tmp.clear();
    C.toVector(tmp);
    return StringRef(tmp.data(), tmp.size());
}

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMultiThreadKernelMethod
 *
 * Given a computation expressed as a logical pipeline of K kernels k0, k_1, ...k_(K-1)
 * operating over an input stream set S, a segment-parallel implementation divides the input
 * into segments and coordinates a set of T <= K threads to each process one segment at a time.
 * Let S_0, S_1, ... S_N be the segments of S.   Segments are assigned to threads in a round-robin
 * fashion such that processing of segment S_i by the full pipeline is carried out by thread i mod T.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateMultiThreadKernelMethod(KernelBuilder & b) {

    Module * const m = b.getModule();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const boolTy = b.getInt1Ty();
    IntegerType * const sizeTy = b.getSizeTy();

    const auto storedState = storeDoSegmentState();

    StructType * const threadStructTy = getThreadStuctType(b, storedState);

    PointerType * const threadStructPtrTy = threadStructTy->getPointerTo();

    ConstantInt * const i32_ZERO = b.getInt32(0);
    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);
    ConstantInt * const sz_TWO = b.getSize(2);

    SmallVector<char, 256> tmp;
    const auto threadName = concat(mTarget->getName(), "_MultithreadedDoSegment", tmp);

    FunctionType * const threadFuncType = FunctionType::get(voidPtrTy, {voidPtrTy}, false);
    Function * const threadFunc = Function::Create(threadFuncType, Function::InternalLinkage, threadName, m);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        threadFunc->setHasUWTable();
        #else
        threadFunc->setUWTableKind(UWTableKind::Default);
        #endif
    }
    Value * const initialSharedState = getHandle();
    Value * const initialThreadLocal = getThreadLocalHandle();
    Value * const initialTerminationSignalPtr = getTerminationSignalPtr();

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER
    // -------------------------------------------------------------------------------------------------------------------------

    // use the process thread to handle the initial segment function after spawning
    // (n - 1) threads to handle the subsequent offsets

    Function * const pthreadCreateFn = m->getFunction("pthread_create");
    Function * const pthreadExitFn = m->getFunction("pthread_exit");
    Function * const pthreadJoinFn = m->getFunction("pthread_join");

    IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);

    const DataLayout & DL = m->getDataLayout();
    const auto pThreadAlign = DL.getABITypeAlign(pThreadTy).value();

    Value * minimumNumOfThreads = nullptr;

    size_t numOfComputeKernels = 0;
    if (AllowIOProcessThread) {
        const auto firstKernel = KernelPartitionId[FirstComputePartitionId];
        const auto lastKernel = KernelPartitionId[LastComputePartitionId + 1];
        numOfComputeKernels = (lastKernel - firstKernel + 1) + 1;
    } else {
        numOfComputeKernels = (LastKernel - FirstKernel + 1);
    }

    // TODO: redesign to avoid the unnecessary store?

    Value * maximumNumOfThreads = b.CreateUMin(b.getScalarField(MAXIMUM_NUM_OF_THREADS), b.getSize(numOfComputeKernels));
    if (mUseDynamicMultithreading) {
        minimumNumOfThreads = b.getScalarField(MINIMUM_NUM_OF_THREADS);
        if (AllowIOProcessThread) {
            minimumNumOfThreads = b.CreateUMax(minimumNumOfThreads, b.getSize(2));
            b.setScalarField(MINIMUM_NUM_OF_THREADS, minimumNumOfThreads);
        }
        maximumNumOfThreads = b.CreateUMax(maximumNumOfThreads, minimumNumOfThreads);
        b.setScalarField(MAXIMUM_NUM_OF_THREADS, maximumNumOfThreads);
    } else {
        minimumNumOfThreads = maximumNumOfThreads;
    }

    Value * const threadStateArray = b.CreateAlignedMalloc(threadStructTy, maximumNumOfThreads, 0, b.getCacheAlignment());
    assert (threadStateArray->getType() == threadStructTy->getPointerTo());
    IntegerType * const intPtrTy = b.getIntPtrTy(DL);

    BasicBlock * const constructThread = b.CreateBasicBlock("constructThread");
    BasicBlock * const constructedThreads = b.CreateBasicBlock("constructedThreads");

    BasicBlock * const constructThreadEntry = b.GetInsertBlock();
    Value * moreThanOneThread = nullptr;
    // construct and start the threads
    if (LLVM_UNLIKELY(mIsIOProcessThread)) {
        b.CreateBr(constructThread);
    } else {
        moreThanOneThread = b.CreateICmpNE(maximumNumOfThreads, sz_ONE);
        b.CreateCondBr(moreThanOneThread, constructThread, constructedThreads);
    }



    b.SetInsertPoint(constructThread);
    PHINode * const threadIndex = b.CreatePHI(sizeTy, 2);
    threadIndex->addIncoming(sz_ONE, constructThreadEntry);

    FixedArray<Value *, 2> fieldIndex;
    fieldIndex[0] = threadIndex;

    Value * const cThreadState = b.CreateGEP(threadStructTy, threadStateArray, threadIndex);
    Value * cThreadLocal = nullptr;
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        SmallVector<Value *, 2> args;
        if (initialSharedState) {
            args.push_back(initialSharedState);
        }
        args.push_back(ConstantPointerNull::get(cast<PointerType>(initialThreadLocal->getType())));
        cThreadLocal = mTarget->initializeThreadLocalInstance(b, args);
        if (LLVM_LIKELY(mTarget->allocatesInternalStreamSets())) {
            Function * const allocInternal = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b, false);
            SmallVector<Value *, 3> allocArgs;
            if (LLVM_LIKELY(mTarget->isStateful())) {
                allocArgs.push_back(initialSharedState);
            }
            allocArgs.push_back(cThreadLocal);
            allocArgs.push_back(sz_ONE);
            b.CreateCall(allocInternal->getFunctionType(), allocInternal, allocArgs);
        }
    }

    Value * initialSegNum = threadIndex;
    Value * maxComputeThreads = maximumNumOfThreads;
    if (AllowIOProcessThread) {
        initialSegNum = b.CreateSub(threadIndex, sz_ONE);
        maxComputeThreads = b.CreateSub(maxComputeThreads, sz_ONE);
    }
    writeThreadStructObject(b, threadStructTy, cThreadState, initialSharedState, cThreadLocal, storedState, initialSegNum, maxComputeThreads);
    Value * const nextThreadIndex = b.CreateAdd(threadIndex, sz_ONE);
    BasicBlock * constructNextThread = nullptr;
    if (mUseDynamicMultithreading) {
        BasicBlock * const startThread = b.CreateBasicBlock("startThread", constructedThreads);
        constructNextThread = b.CreateBasicBlock("constructNextThread", constructedThreads);
        Value * const start = b.CreateICmpULE(nextThreadIndex, minimumNumOfThreads);
        b.CreateCondBr(start, startThread, constructNextThread);

        b.SetInsertPoint(startThread);
    }
    FixedArray<Value *, 4> pthreadCreateArgs;
    if (mUseDynamicMultithreading) {
        fieldIndex[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
        Value * initThreadStateFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex);
        b.CreateAlignedStore(sz_ONE, initThreadStateFlagPtr, SizeTyABIAlignment);
    }
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);

    pthreadCreateArgs[0] = b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex);
    assert (pthreadCreateArgs[0]->getType() == pThreadTy->getPointerTo());
    pthreadCreateArgs[1] = ConstantPointerNull::get(voidPtrTy);
    pthreadCreateArgs[2] = b.CreatePointerCast(threadFunc, voidPtrTy);
    pthreadCreateArgs[3] = b.CreatePointerCast(cThreadState, voidPtrTy);
    b.CreateCall(pthreadCreateFn->getFunctionType(), pthreadCreateFn, pthreadCreateArgs);
    if (mUseDynamicMultithreading) {
        b.CreateBr(constructNextThread);

        b.SetInsertPoint(constructNextThread);
    }

    Value * const createMoreThreads = b.CreateICmpULT(nextThreadIndex, maximumNumOfThreads);
    threadIndex->addIncoming(nextThreadIndex, b.GetInsertBlock());
    b.CreateCondBr(createMoreThreads, constructThread, constructedThreads);

    b.SetInsertPoint(constructedThreads);

    // execute the process thread
    Value * const processState = threadStateArray;
    Value * const maxProcessThreads = AllowIOProcessThread ? sz_ONE : maximumNumOfThreads;
    writeThreadStructObject(b, threadStructTy, processState, initialSharedState, initialThreadLocal, storedState, sz_ZERO, maxProcessThreads);
    fieldIndex[0] = i32_ZERO;
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);
    b.CreateAlignedStore(ConstantInt::getNullValue(pThreadTy), b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex), pThreadAlign);
    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b.saveIP();

    const auto anyDebugOptionIsSet = codegen::AnyDebugOptionIsSet();

    const auto hasTermSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;

    SmallVector<Type *, 2> csRetValFields;
    csRetValFields.push_back(hasTermSignal ? sizeTy : boolTy);
    if (CheckAssertions) {
        csRetValFields.push_back(boolTy);
    }

    StructType * const csRetValType = StructType::get(b.getContext(), csRetValFields);

    FixedArray<Type *, 2> csParams;
    csParams[0] = threadStructPtrTy; // thread state
    csParams[1] = sizeTy; // segment number

    FunctionType * const csDoSegmentComputeFuncType = FunctionType::get(csRetValType, csParams, false);

    FunctionType * csDoSegmentProcessFuncType = nullptr;
    if (AllowIOProcessThread) {
        FixedArray<Type *, 3> csParams;
        csParams[0] = threadStructPtrTy; // thread state
        csParams[1] = sizeTy; // segment number
        csParams[2] = sizeTy; // number of threads
        csDoSegmentProcessFuncType = FunctionType::get(csRetValType, csParams, false);
    } else {
        csDoSegmentProcessFuncType = csDoSegmentComputeFuncType;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // GENERATE DO SEGMENT (KERNEL EXECUTION) FUNCTION CODE
    // -------------------------------------------------------------------------------------------------------------------------

    auto makeDoSegmentLogicFunction = [&](Function * csFunc, const bool generateProcessThread) -> void {

        csFunc->setCallingConv(CallingConv::C);

        if (LLVM_UNLIKELY(CheckAssertions)) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            csFunc->setHasUWTable();
            #else
            csFunc->setUWTableKind(UWTableKind::Default);
            #endif
        }
        b.SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", csFunc));
        mIsIOProcessThread = AllowIOProcessThread && generateProcessThread;
        auto args = csFunc->arg_begin();
        Value * const threadStruct = &*args++;
        assert (threadStruct->getType() == threadStructPtrTy);
        readThreadStructObject(b, threadStructTy, threadStruct);
        assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));

//        Value * baseFunctionPtrInt = b.CreatePtrToInt(csFunc, intPtrTy);
//        b.CallPrintInt(csFunc->getName().str() + "." + b.GetInsertBlock()->getName().str(), baseFunctionPtrInt);

        readDoSegmentState(b, threadStructTy, threadStruct);
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
        mSegNo = &*args++;
        Value * numOfActiveThreads = nullptr;
        if (generateProcessThread) {
            numOfActiveThreads = &*args++;
        }
        assert (args == csFunc->arg_end());
        #ifdef PRINT_DEBUG_MESSAGES
        debugInit(b);
        #endif
        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            setupPAPIOnCurrentThread(b);
        }
        #endif
        Value * segmentStartTime = nullptr;
        if (mUseDynamicMultithreading) {
            segmentStartTime = b.CreateReadCycleCounter();
        }
        // generate the pipeline logic for this thread
        start(b);

        branchToInitialPartition(b);
        const auto firstComputeKernel = FirstKernelInPartition[FirstComputePartitionId];
        assert (AllowIOProcessThread || firstComputeKernel == FirstKernel);
        if (mIsIOProcessThread) {
            assert (!mIsNestedPipeline);
            assert (FirstKernel != PipelineInput);
            if (FirstKernel < firstComputeKernel) {
                // Let C be the current segment number of the compute thread,
                // P be the current segment number of the process thread and
                // T be the total number of active threads.

                // If P < C + T, then the process thread is behind the compute thread.

                waitUntilCurrentSegmentNumberIsLessThan(b, firstComputeKernel, numOfActiveThreads);
                for (auto i = FirstKernel; i < firstComputeKernel; ++i) {
                    setActiveKernel(b, i, true);
                    writeProcessThreadMessage(b, std::to_string(i) + ".starting " + mKernel->getName(), threadStructTy, threadStruct);
                    executeKernel(b);
                    writeProcessThreadMessage(b, std::to_string(i) + ".exiting " + mKernel->getName(), threadStructTy, threadStruct);
                }
            }
            const auto afterLastComputeKernel = FirstKernelInPartition[LastComputePartitionId + 1];
            assert (firstComputeKernel < afterLastComputeKernel);
            if (LLVM_LIKELY(afterLastComputeKernel < PipelineOutput)) {
                for (auto i = afterLastComputeKernel; i <= LastKernel; ++i) {
                    setActiveKernel(b, i, true);
                    writeProcessThreadMessage(b, std::to_string(i) + ".starting " + mKernel->getName(), threadStructTy, threadStruct);
                    executeKernel(b);
                    writeProcessThreadMessage(b, std::to_string(i) + ".exiting " + mKernel->getName(), threadStructTy, threadStruct);
                }
            }
        } else if (firstComputeKernel != PipelineInput) {
            if (AllowIOProcessThread && firstComputeKernel != FirstKernel) {
                 // If C < P, then the compute thread is behind the process thread.
                 waitUntilCurrentSegmentNumberIsLessThan(b, firstComputeKernel - 1, nullptr);
            }
            const auto lastComputeKernel = FirstKernelInPartition[LastComputePartitionId + 1] - 1;
            assert (lastComputeKernel < PipelineOutput);
            assert (AllowIOProcessThread || lastComputeKernel == LastKernel);
            for (auto i = firstComputeKernel; i <= lastComputeKernel; ++i) {
                setActiveKernel(b, i, true);
                writeProcessThreadMessage(b, std::to_string(i) + ".starting " + mKernel->getName(), threadStructTy, threadStruct);
                executeKernel(b);
                writeProcessThreadMessage(b, std::to_string(i) + ".exiting " + mKernel->getName(), threadStructTy, threadStruct);
            }
        }
        mKernel = nullptr;
        mKernelId = 0;
        mSegNo = nullptr;
        if (mUseDynamicMultithreading) {
            Value * const segmentEndTime = b.CreateReadCycleCounter();
            Value * const totalSegmentTime = b.CreateSub(segmentEndTime, segmentStartTime);
            FixedArray<Value *, 2> indices2;
            indices2[0] = i32_ZERO;
            indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
            Value * const segPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const current = b.CreateAlignedLoad(b.getSizeTy(), segPtr, SizeTyABIAlignment);
            Value * const accum = b.CreateAdd(current, totalSegmentTime);
            b.CreateAlignedStore(accum, segPtr, SizeTyABIAlignment);
        }

        Value * const terminated = hasPipelineTerminated(b);
        SmallVector<Value *, 2> retVal;
        if (hasTermSignal) {
            retVal.push_back(terminated);
        } else {
            retVal.push_back(b.CreateIsNotNull(terminated));
        }
        if (LLVM_UNLIKELY(CheckAssertions)) {
            retVal.push_back(mPipelineProgress);
        }

//        Constant * ba = BlockAddress::get(csFunc, b.GetInsertBlock());
//        Value * ptr = b.CreateAdd(baseFunctionPtrInt, ConstantExpr::getPtrToInt(ba, intPtrTy));
//        b.CallPrintInt(csFunc->getName().str() + "." + b.GetInsertBlock()->getName().str(), ptr);

        b.CreateAggregateRet(retVal.data(), retVal.size());

        mIsIOProcessThread = false;
    };

    const auto outerFuncName = concat(mTarget->getName(), "_ComputeThread", tmp);
    Function * doSegmentComputeThreadFunc = Function::Create(csDoSegmentComputeFuncType, Function::InternalLinkage, outerFuncName, m);

    makeDoSegmentLogicFunction(doSegmentComputeThreadFunc, false);

    Function * doSegmentProcessThreadFunc = nullptr;

    if (AllowIOProcessThread) {
        assert (!mIsNestedPipeline);
        const auto outerFuncName = concat(mTarget->getName(), "_ProcessThread", tmp);
        doSegmentProcessThreadFunc = Function::Create(csDoSegmentProcessFuncType, Function::InternalLinkage, outerFuncName, m);
        makeDoSegmentLogicFunction(doSegmentProcessThreadFunc, true); 
        doSegmentProcessThreadFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
        doSegmentComputeThreadFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
    } else {
        doSegmentProcessThreadFunc = doSegmentComputeThreadFunc;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc, const bool generateProcessThread) {
        assert (threadFunc);
        threadFunc->setCallingConv(CallingConv::C);
        auto arg = threadFunc->arg_begin();
        arg->setName("threadStruct");

        b.SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", threadFunc));
        Value * const threadStruct = b.CreatePointerCast(arg, threadStructPtrTy);
        readThreadStructObject(b, threadStructTy, threadStruct);
        assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));
        assert (isFromCurrentFunction(b, getThreadLocalHandle(), !mTarget->hasThreadLocal()));
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);

        Value * minimumThreads = nullptr;
        Value * maximumThreads = nullptr;
        Value * synchronizationCostCheckPeriod = nullptr;
        Value * syncAddThreadThreadhold = nullptr;
        Value * syncRemoveThreadThreadhold = nullptr;

        if (mUseDynamicMultithreading && generateProcessThread) {
            minimumThreads = b.getScalarField(MINIMUM_NUM_OF_THREADS);
            maximumThreads = b.getScalarField(MAXIMUM_NUM_OF_THREADS);
            synchronizationCostCheckPeriod =
                b.getScalarField(DYNAMIC_MULTITHREADING_SEGMENT_PERIOD);
            Type * const floatTy = b.getFloatTy();
            Constant * const f100 = ConstantFP::get(floatTy, 100.0);
            syncAddThreadThreadhold =
                b.getScalarField(DYNAMIC_MULTITHREADING_ADDITIONAL_THREAD_SYNCHRONIZATION_THRESHOLD);
            syncAddThreadThreadhold = b.CreateFDiv(syncAddThreadThreadhold, f100);
            syncRemoveThreadThreadhold =
                b.getScalarField(DYNAMIC_MULTITHREADING_REMOVE_THREAD_SYNCHRONIZATION_THRESHOLD);
            syncRemoveThreadThreadhold = b.CreateFDiv(syncRemoveThreadThreadhold, f100);
        }

        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            initPAPIOnCurrentThread(b);
            setupPAPIOnCurrentThread(b);
        }
        #endif
        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);
        }
        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            startPAPIMeasurement(b, PAPIKernelCounter::PAPI_FULL_PIPELINE_TIME);
        }
        #endif

        #ifdef PRINT_DEBUG_MESSAGES
        debugInit(b);
        if (mIsNestedPipeline) {
            debugPrint(b, "------------------------------------------------- START %" PRIx64, getHandle());
        } else {
            debugPrint(b, "================================================= START %" PRIx64, getHandle());
        }
        #endif

        // generate the pipeline logic for this thread
        BasicBlock * const mPipelineLoop = b.CreateBasicBlock("PipelineLoop");
        BasicBlock * const mPipelineEnd = b.CreateBasicBlock("PipelineEnd");

        BasicBlock * const entryBlock = b.GetInsertBlock();
        b.CreateBr(mPipelineLoop);

        b.SetInsertPoint(mPipelineLoop);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            mMadeProgressInLastSegment = b.CreatePHI(b.getInt1Ty(), 2, "madeProgressInLastSegment");
            mMadeProgressInLastSegment->addIncoming(b.getTrue(), entryBlock);
        }
        PHINode * nextCheckSegmentPhi = nullptr;
        PHINode * activeThreadsPhi = nullptr;
        if (mUseDynamicMultithreading && generateProcessThread) {
            nextCheckSegmentPhi = b.CreatePHI(sizeTy, 2, "nextCheckPhi");
            nextCheckSegmentPhi->addIncoming(synchronizationCostCheckPeriod, entryBlock);
            activeThreadsPhi = b.CreatePHI(sizeTy, 2, "activeThreadsPhi");
            activeThreadsPhi->addIncoming(minimumThreads, entryBlock);
        }

        obtainCurrentSegmentNumber(b, entryBlock);

        SmallVector<Value *, 3> args(2);
        args[0] = threadStruct;
        args[1] = mSegNo; assert (mSegNo);
        if (generateProcessThread && AllowIOProcessThread) {
            assert (doSegmentProcessThreadFunc != doSegmentComputeThreadFunc);
            args.push_back(mUseDynamicMultithreading ? activeThreadsPhi : mNumOfFixedThreads);
        }

        Function * const doSegFunc = generateProcessThread ? doSegmentProcessThreadFunc : doSegmentComputeThreadFunc;

        Value * csRetVal = nullptr;
        if (CheckAssertions) {
            BasicBlock * rethrowException = b.WriteDefaultRethrowBlock();
            const auto prefix = makeKernelName(mKernelId);
            BasicBlock * const invokeOk = b.CreateBasicBlock(prefix + "_invokeOk");
            #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
            csRetVal = b.CreateInvoke(doSegFunc->getFunctionType(), doSegFunc, invokeOk, rethrowException, args);
            #else
            csRetVal = b.CreateInvoke(doSegFunc->getFunctionType(), invokeOk, mRethrowException, args);
            #endif
            b.SetInsertPoint(invokeOk);
        } else {
            csRetVal = b.CreateCall(doSegFunc->getFunctionType(), doSegFunc, args);
        }


        Value * const terminated = b.CreateExtractValue(csRetVal, {0});

        Value * done = b.CreateIsNotNull(terminated);
        Value * madeProgress = nullptr;

        if (LLVM_UNLIKELY(CheckAssertions)) {
            madeProgress = b.CreateExtractValue(csRetVal, {1});
//            if (LLVM_LIKELY(hasTermSignal)) {
                madeProgress = b.CreateOr(madeProgress, done);
//            }
            if (LLVM_LIKELY(!AllowIOProcessThread)) {
                Value * const live = b.CreateOr(mMadeProgressInLastSegment, madeProgress);
                b.CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
            }
        }

        PHINode * startOfNextPeriodPhi = nullptr;
        PHINode * currentNumOfThreadsPhi = nullptr;

        if (mUseDynamicMultithreading && generateProcessThread) {

            // If a thread got stalled or the period was set so low, we could reenter this check prior to
            // the thread itself stopping,

            BasicBlock * checkSynchronizationCostLoop = b.CreateBasicBlock("checkSynchronizationCostLoop", mPipelineEnd);
            BasicBlock * checkToAddThread = b.CreateBasicBlock("checkToAddThread", mPipelineEnd);

            BasicBlock * selectThreadStructToUseForAddThread = b.CreateBasicBlock("selectThreadStructToUseForAddThread", mPipelineEnd);
            BasicBlock * checkIfThreadIsCancelled = b.CreateBasicBlock("checkIfThreadIsCancelled", mPipelineEnd);
            BasicBlock * joinCancelledThread = b.CreateBasicBlock("joinCancelledThread", mPipelineEnd);
            BasicBlock * addThread = b.CreateBasicBlock("addThread", mPipelineEnd);

            BasicBlock * checkToRemoveThread = b.CreateBasicBlock("checkToRemoveThread", mPipelineEnd);
            BasicBlock * selectThreadToRemove = b.CreateBasicBlock("selectThreadToRemove", mPipelineEnd);

            BasicBlock * removeThread = b.CreateBasicBlock("removeThread", mPipelineEnd);
            BasicBlock * nextSegment = b.CreateBasicBlock("nextSegment", mPipelineEnd);
            BasicBlock * recordBeforeNextSegment = nextSegment;
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                recordBeforeNextSegment = b.CreateBasicBlock("recordBeforeNextSegment", nextSegment);
            }

            Value * const check = b.CreateICmpUGE(mSegNo, nextCheckSegmentPhi);
            FixedArray<Value *, 2> indices2;

            BasicBlock * const loopEntry = b.GetInsertBlock();
            b.CreateUnlikelyCondBr(check, checkSynchronizationCostLoop, nextSegment);

            b.SetInsertPoint(checkSynchronizationCostLoop);
            PHINode * const indexPhi = b.CreatePHI(sizeTy, 2);
            indexPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const segmentTimeAccumPhi = b.CreatePHI(sizeTy, 2);
            segmentTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const synchronizationTimeAccumPhi = b.CreatePHI(sizeTy, 2);
            synchronizationTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);

            indices2[0] = indexPhi;
            indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
            Value * const segTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const nextSegTime = b.CreateAdd(segmentTimeAccumPhi, b.CreateAlignedLoad(sizeTy, segTimePtr, SizeTyABIAlignment));

            segmentTimeAccumPhi->addIncoming(nextSegTime, checkSynchronizationCostLoop);

            indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
            Value * const syncTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const nextSyncTime = b.CreateAdd(synchronizationTimeAccumPhi, b.CreateAlignedLoad(sizeTy, syncTimePtr, SizeTyABIAlignment));
            synchronizationTimeAccumPhi->addIncoming(nextSyncTime, checkSynchronizationCostLoop);

            Value * const nextIndex = b.CreateAdd(indexPhi, b.getSize(1));
            indexPhi->addIncoming(nextIndex, checkSynchronizationCostLoop);
            Value * const hasMore = b.CreateICmpNE(nextIndex, activeThreadsPhi);
            b.CreateCondBr(hasMore, checkSynchronizationCostLoop, checkToAddThread);

            b.SetInsertPoint(checkToAddThread);
            Type * const floatTy = b.getFloatTy();
            Value * const fSegTime = b.CreateUIToFP(nextSegTime, floatTy);
            Value * const fSyncTime = b.CreateUIToFP(nextSyncTime, floatTy);
            Value * const fSyncOverhead = b.CreateFDiv(fSyncTime, fSegTime);


            // subtract out the values so that we can keep each check
            indices2[0] = sz_ZERO;
            indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);

            Value * const baseSegTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            b.CreateAlignedStore(sz_ZERO, baseSegTimePtr, SizeTyABIAlignment);
            indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
            Value * const baseSyncTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);

            b.CreateAlignedStore(sz_ZERO, baseSyncTimePtr, SizeTyABIAlignment);

            Value * const syncOverheadLow = b.CreateFCmpULT(fSyncOverhead, syncAddThreadThreadhold);
            Value * const canAddMoreThreads = b.CreateICmpULT(activeThreadsPhi, maximumThreads);
            Value * const canAdd = b.CreateAnd(syncOverheadLow, canAddMoreThreads);
            Value * const startOfNextPeriod = b.CreateAdd(mSegNo, synchronizationCostCheckPeriod);
            b.CreateCondBr(canAdd, selectThreadStructToUseForAddThread, checkToRemoveThread);

            b.SetInsertPoint(selectThreadStructToUseForAddThread);
            PHINode * const selectToAddPhi = b.CreatePHI(sizeTy, 2);
            selectToAddPhi->addIncoming(sz_ONE, checkToAddThread);
            indices2[0] = selectToAddPhi;
            indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
            Value * addThreadStateFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * addThreadStateFlag = b.CreateAlignedLoad(sizeTy, addThreadStateFlagPtr, SizeTyABIAlignment);
            Value * canUseThreadStruct = b.CreateICmpNE(addThreadStateFlag, sz_ONE);
            Value * const nextToCheckForAdd = b.CreateAdd(selectToAddPhi, sz_ONE);
            selectToAddPhi->addIncoming(nextToCheckForAdd, selectThreadStructToUseForAddThread);
            b.CreateCondBr(canUseThreadStruct, checkIfThreadIsCancelled, selectThreadStructToUseForAddThread);

            b.SetInsertPoint(checkIfThreadIsCancelled);
            Value * isCancelled = b.CreateICmpEQ(addThreadStateFlag, sz_TWO);
            indices2[1] = b.getInt32(CURRENT_THREAD_ID);
            Value * const threadIdPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            b.CreateCondBr(isCancelled, joinCancelledThread, addThread);

            b.SetInsertPoint(joinCancelledThread);
            FixedArray<Value *, 2> pthreadJoinArgs;
            Value * threadId = b.CreateAlignedLoad(pThreadTy, threadIdPtr, pThreadAlign);
            pthreadJoinArgs[0] = threadId;
            pthreadJoinArgs[1] = b.CreateAllocaAtEntryPoint(voidPtrTy);
            b.CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);
            b.CreateBr(addThread);

            b.SetInsertPoint(addThread);
            b.CreateAlignedStore(sz_ONE, addThreadStateFlagPtr, SizeTyABIAlignment);
            pthreadCreateArgs[0] = threadIdPtr;
            Value * const ts = b.CreateInBoundsGEP(threadStructTy, threadStruct, selectToAddPhi);
            pthreadCreateArgs[3] = b.CreatePointerCast(ts, voidPtrTy);
            b.CreateCall(pthreadCreateFn->getFunctionType(), pthreadCreateFn, pthreadCreateArgs);
            Value * numOfThreadsAfterAdd = b.CreateAdd(activeThreadsPhi, sz_ONE);
            b.CreateBr(recordBeforeNextSegment);

            b.SetInsertPoint(checkToRemoveThread);
            Value * const syncOverheadHigh = b.CreateFCmpUGT(fSyncOverhead, syncRemoveThreadThreadhold);
            Value * const canRemoveMoreThreads = b.CreateICmpUGT(activeThreadsPhi, minimumThreads);
            Value * const canRemove = b.CreateAnd(syncOverheadHigh, canRemoveMoreThreads);
            b.CreateCondBr(canRemove, selectThreadToRemove, recordBeforeNextSegment);

            b.SetInsertPoint(selectThreadToRemove);
            PHINode * const selectedThreadPhi = b.CreatePHI(sizeTy, 2);
            selectedThreadPhi->addIncoming(sz_ONE, checkToRemoveThread);
            indices2[0] = selectedThreadPhi;
            indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
            Value * const cancelFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const isActive = b.CreateICmpEQ(b.CreateAlignedLoad(sizeTy, cancelFlagPtr, SizeTyABIAlignment), sz_ONE);
            Value * const nextThreadToCheckForRemove = b.CreateAdd(selectedThreadPhi, sz_ONE);
            selectedThreadPhi->addIncoming(nextThreadToCheckForRemove, selectThreadToRemove);
            b.CreateCondBr(isActive, removeThread, selectThreadToRemove);

            b.SetInsertPoint(removeThread);
            // mark this thread to terminate when it reaches the end of a segment iteration
            b.CreateAlignedStore(sz_TWO, cancelFlagPtr, SizeTyABIAlignment);
            Value * const numOfThreadsAfterRemoval = b.CreateSub(activeThreadsPhi, sz_ONE);
            b.CreateBr(recordBeforeNextSegment);

            PHINode * numOfThreadsPhi = nullptr;
            BasicBlock * recordBeforeNextSegmentExit = nullptr;
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                b.SetInsertPoint(recordBeforeNextSegment);

                numOfThreadsPhi = b.CreatePHI(sizeTy, 2);
                numOfThreadsPhi->addIncoming(numOfThreadsAfterAdd, addThread);
                numOfThreadsPhi->addIncoming(activeThreadsPhi, checkToRemoveThread);
                numOfThreadsPhi->addIncoming(numOfThreadsAfterRemoval, removeThread);

                recordDynamicThreadingState(b, mSegNo, fSyncOverhead, numOfThreadsPhi);

                recordBeforeNextSegmentExit = b.GetInsertBlock();

                b.CreateBr(nextSegment);
            }

            b.SetInsertPoint(nextSegment);
            startOfNextPeriodPhi = b.CreatePHI(sizeTy, 3, "startOfNextPeriodPhi");
            startOfNextPeriodPhi->addIncoming(nextCheckSegmentPhi, loopEntry);
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, recordBeforeNextSegmentExit);
            } else {
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, addThread);
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, removeThread);
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, checkToRemoveThread);
            }

            currentNumOfThreadsPhi = b.CreatePHI(sizeTy, 3, "currentNumOfThreadsPhi");
            currentNumOfThreadsPhi->addIncoming(activeThreadsPhi, loopEntry);
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                currentNumOfThreadsPhi->addIncoming(numOfThreadsPhi, recordBeforeNextSegmentExit);
            } else {
                currentNumOfThreadsPhi->addIncoming(numOfThreadsAfterAdd, addThread);
                currentNumOfThreadsPhi->addIncoming(activeThreadsPhi, checkToRemoveThread);
                currentNumOfThreadsPhi->addIncoming(numOfThreadsAfterRemoval, removeThread);
            }
        }

        BasicBlock * const exitBlock = b.GetInsertBlock();
        incrementCurrentSegNo(b, exitBlock);

        if (mIsNestedPipeline) {
            b.CreateBr(mPipelineEnd);
        } else {
            if (LLVM_UNLIKELY(CheckAssertions)) {
                mMadeProgressInLastSegment->addIncoming(madeProgress, exitBlock);
            }
            if (mUseDynamicMultithreading && generateProcessThread) {
                nextCheckSegmentPhi->addIncoming(startOfNextPeriodPhi, exitBlock);
                activeThreadsPhi->addIncoming(currentNumOfThreadsPhi, exitBlock);
            } else if (mUseDynamicMultithreading) {
                FixedArray<Value *, 2> indices2;
                indices2[0] = sz_ZERO;
                indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
                Value * const statusFlagPtr = b.CreateGEP(threadStructTy, threadStruct, indices2);
                Value * const cancelled = b.CreateAlignedLoad(sizeTy, statusFlagPtr, SizeTyABIAlignment);
                done = b.CreateOr(done, b.CreateICmpEQ(cancelled, sz_TWO));
            }
            assert (hasTermSignal);
            b.CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
        }

        b.SetInsertPoint(mPipelineEnd);
        assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));
        assert (isFromCurrentFunction(b, getThreadLocalHandle(), !mTarget->hasThreadLocal()));
        if (AllowIOProcessThread && !generateProcessThread) {
            assert (!mIsNestedPipeline);
            const auto ref = b.getScalarFieldPtr(COMPUTE_THREAD_TERMINATION_STATE);
            DataLayout DL(b.getModule());
            #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(13, 0, 0)
            llvm::MaybeAlign align = Align(DL.getTypeStoreSize(b.getSizeTy()));
            #endif
            b.CreateAtomicCmpXchg(ref.first, b.getSize(0), terminated,
            #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(13, 0, 0)
                                                           *align,
            #endif
            AtomicOrdering::Release, AtomicOrdering::Acquire);
        }

        #ifdef PRINT_DEBUG_MESSAGES
        if (mIsNestedPipeline) {
            debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
        } else {
            debugPrint(b, "================================================= END %" PRIx64, getHandle());
        }
        #endif

        mExpectedNumOfStridesMultiplier = nullptr;
        mThreadLocalStreamSetBaseAddress = nullptr;

        if (LLVM_LIKELY(hasTermSignal)) {
            writeTerminationSignalToLocalState(b, threadStructTy, threadStruct, terminated);
        }

        // only call pthread_exit() within spawned threads; otherwise it'll be equivalent to calling exit() within the process
        Value * retVal = nullptr;
        if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
            retVal = b.CreateIntToPtr(b.CreateZExt(mSegNo, intPtrTy), voidPtrTy);
        } else {
            retVal = ConstantPointerNull::getNullValue(voidPtrTy);
        }

        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            updateTotalCycleCounterTime(b);
        }
        #ifdef ENABLE_PAPI
        if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
            recordTotalPAPIMeasurement(b);
            stopPAPIOnCurrentThread(b);
        }
        #endif
        if (AllowIOProcessThread) {
            assert (!mIsNestedPipeline);
            if (!generateProcessThread) {
                b.CreateCall(pthreadExitFn->getFunctionType(), pthreadExitFn, retVal);
            }
        } else {
            BasicBlock * const exitThread  = b.CreateBasicBlock("ExitThread");
            BasicBlock * const exitFunction  = b.CreateBasicBlock("ExitProcessFunction");
            b.CreateCondBr(isProcessThread(b, threadStructTy, threadStruct), exitFunction, exitThread);
            b.SetInsertPoint(exitThread);
            b.CreateCall(pthreadExitFn->getFunctionType(), pthreadExitFn, retVal);
            b.CreateBr(exitFunction);
            b.SetInsertPoint(exitFunction);
        }
        b.CreateRet(retVal);
    };

    makeThreadFunction(threadFunc, false);

    Function * processThreadFunc = nullptr;

    if (mUseDynamicMultithreading || AllowIOProcessThread) {
        const auto outerFuncName = concat(mTarget->getName(), "_MultithreadedProcessThread", tmp);
        processThreadFunc = Function::Create(threadFuncType, Function::InternalLinkage, outerFuncName, m);
        processThreadFunc->setCallingConv(CallingConv::C);
        processThreadFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
        mIsIOProcessThread = AllowIOProcessThread;
        makeThreadFunction(processThreadFunc, true);
        mIsIOProcessThread = false;
    } else {
        processThreadFunc = threadFunc;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER CONTINUED
    // -------------------------------------------------------------------------------------------------------------------------

    b.restoreIP(resumePoint);
    FixedArray<Value *, 1> processArgs;
    processArgs[0] = b.CreatePointerCast(processState, voidPtrTy);
    Value * const mainThreadRetVal =
        b.CreateCall(threadFuncType, processThreadFunc, processArgs);

    Value * firstSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        firstSegNo = b.CreatePtrToInt(mainThreadRetVal, intPtrTy);
    }

    assert (isFromCurrentFunction(b, processState));
    assert (isFromCurrentFunction(b, initialSharedState));
    assert (isFromCurrentFunction(b, initialThreadLocal));

    setHandle(initialSharedState);
    setThreadLocalHandle(initialThreadLocal);
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);

    Value * firstTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        firstTerminationSignal = readTerminationSignalFromLocalState(b, threadStructTy, processState);
        assert (firstTerminationSignal);
    }

    // wait for all other threads to complete
    Value * const status = b.CreateAlloca(voidPtrTy);
    BasicBlock * const checkStatusOfThread = b.CreateBasicBlock("checkStatusOfThread");
    BasicBlock * const joinThread = b.CreateBasicBlock("joinThread");
    BasicBlock * const finalizeAfterJoinThread = b.CreateBasicBlock("finalizeAfterJoinThread");
    BasicBlock * const joinedThreads = b.CreateBasicBlock("joinedThreads");

    BasicBlock * const joinThreadEntry = b.GetInsertBlock();

    // join the threads and destroy any state objects
    if (LLVM_UNLIKELY(mIsIOProcessThread)) {
        b.CreateBr(checkStatusOfThread);
    } else {
        b.CreateCondBr(moreThanOneThread, checkStatusOfThread, joinedThreads);
    }

    b.SetInsertPoint(checkStatusOfThread);
    PHINode * const joinThreadIndex = b.CreatePHI(sizeTy, 2);
    joinThreadIndex->addIncoming(sz_ONE, joinThreadEntry);
    PHINode * finalTerminationSignalPhi = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b.CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
    }
    PHINode * finalSegNoPhi = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi = b.CreatePHI(firstSegNo->getType(), 2);
        finalSegNoPhi->addIncoming(firstSegNo, joinThreadEntry);
    }

    fieldIndex[0] = joinThreadIndex;
    if (mUseDynamicMultithreading) {
        fieldIndex[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
        Value * const statusFlag = b.CreateAlignedLoad(sizeTy, b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex), SizeTyABIAlignment);
        b.CreateCondBr(b.CreateICmpNE(statusFlag, sz_ZERO), joinThread, finalizeAfterJoinThread);
    } else {
        b.CreateBr(joinThread);
    }

    b.SetInsertPoint(joinThread);
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);
    FixedArray<Value *, 2> pthreadJoinArgs;
    Value * threadId = b.CreateAlignedLoad(pThreadTy, b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex), pThreadAlign);
    pthreadJoinArgs[0] = threadId;
    pthreadJoinArgs[1] = status;
    b.CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);
    b.CreateBr(finalizeAfterJoinThread);

    b.SetInsertPoint(finalizeAfterJoinThread);

    // calculate the last segment # used by any kernel in case any reports require it.
    Value * finalSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        // Value * const retVal = b.CreatePointerCast(status, intPtrPtrTy);
        Value * const retVal = b.CreatePtrToInt(b.CreateAlignedLoad(voidPtrTy, status, PtrTyABIAlignment), intPtrTy);
        finalSegNo = b.CreateUMax(finalSegNoPhi, retVal);
    }

    Value * const jThreadState = b.CreateGEP(threadStructTy, threadStateArray, joinThreadIndex);
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        fieldIndex[1] = b.getInt32(THREAD_LOCAL_PARAM);
        Type * const handlePtrTy = getThreadLocalHandle()->getType();
        Value * const jThreadLocal = b.CreateAlignedLoad(handlePtrTy, b.CreateGEP(threadStructTy, threadStateArray, fieldIndex), PtrTyABIAlignment);
        SmallVector<Value *, 3> threadLocalArgs;
        if (LLVM_LIKELY(mTarget->isStateful())) {
            threadLocalArgs.push_back(initialSharedState);
        }
        threadLocalArgs.push_back(initialThreadLocal);
        threadLocalArgs.push_back(jThreadLocal);
        mTarget->finalizeThreadLocalInstance(b, threadLocalArgs);
        b.CreateFree(jThreadLocal);
        threadLocalArgs.pop_back();
    }
    Value * finalTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        Value * const terminatedSignal = readTerminationSignalFromLocalState(b, threadStructTy, jThreadState);
        assert (terminatedSignal);
        finalTerminationSignal = b.CreateUMax(finalTerminationSignalPhi, terminatedSignal);
    }

    Value * const nextJoinIndex = b.CreateAdd(joinThreadIndex, sz_ONE);
    BasicBlock * const joinThreadExit = b.GetInsertBlock();
    joinThreadIndex->addIncoming(nextJoinIndex, joinThreadExit);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi->addIncoming(finalSegNo, joinThreadExit);
    }
    Value * const joinMoreThreads = b.CreateICmpULT(nextJoinIndex, maximumNumOfThreads);
    b.CreateCondBr(joinMoreThreads, checkStatusOfThread, joinedThreads);

    b.SetInsertPoint(joinedThreads);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b.CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }

    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        PHINode * phi = b.CreatePHI(firstSegNo->getType(), 2);
        phi->addIncoming(firstSegNo, joinThreadEntry);
        phi->addIncoming(finalSegNo, joinThreadExit);
        mSegNo = phi;
    } else {
        mSegNo = nullptr;
    }
    b.CreateFree(threadStateArray);
    restoreDoSegmentState(storedState);
    if (PipelineHasTerminationSignal) {
        assert (initialTerminationSignalPtr);
        b.CreateAlignedStore(finalTerminationSignalPhi, initialTerminationSignalPtr, SizeTyABIAlignment);
    }

    assert (getHandle() == initialSharedState);
    assert (getThreadLocalHandle() == initialThreadLocal);
    assert (b.getCompiler() == this);

    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);

    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, FirstKernel, type);
        assert (isFromCurrentFunction(b, ptr));
        b.CreateAlignedStore(mSegNo, ptr, SizeTyABIAlignment);
        concludeStridesPerSegmentRecording(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief start
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::start(KernelBuilder & b) {

    mLocallyAvailableItems.reset(FirstStreamSet, LastStreamSet);
    mKernelTerminationSignal.reset(FirstKernel, LastKernel);

    makePartitionEntryPoints(b);

    if (LLVM_UNLIKELY(CheckAssertions)) {
        mRethrowException = b.WriteDefaultRethrowBlock();
    }

    mExpectedNumOfStridesMultiplier = b.getScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER);
    initializeFlowControl(b);
    readExternalConsumerItemCounts(b);
    loadInternalStreamSetHandles(b, true);
    loadInternalStreamSetHandles(b, false);

    mKernel = nullptr;
    mCurrentKernelName = nullptr;
    mKernelId = 0;
    mAddressableItemCountPtr.clear();
    mVirtualBaseAddressPtr.clear();
    mPipelineProgress = b.getFalse();


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * PipelineCompiler::getThreadStuctType(KernelBuilder & b, const std::vector<Value *> & props) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE + 1> fields; // +1 for the cache line padding
    LLVMContext & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    Type * const emptyTy = StructType::get(C);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    if (LLVM_LIKELY(mTarget->isStateful())) {
        fields[SHARED_STATE_PARAM] = getHandle()->getType();
        assert (fields[SHARED_STATE_PARAM]->isPointerTy());
    } else {
        fields[SHARED_STATE_PARAM] = emptyTy;
    }

    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        fields[THREAD_LOCAL_PARAM] = getThreadLocalHandle()->getType();
        assert (fields[THREAD_LOCAL_PARAM]->isPointerTy());
    } else {
        fields[THREAD_LOCAL_PARAM] = emptyTy;
    }

    const auto n = props.size();
    std::vector<Type *> paramType(n);

    DataLayout dl(b.getModule());
    for (unsigned i = 0; i < n; ++i) {
        paramType[i] = props[i]->getType();
    }
    fields[PIPELINE_PARAMS] = StructType::get(b.getContext(), paramType);
    const auto sizeTyWidth = sizeTy->getIntegerBitWidth();
    const auto paramPaddingBytes = sizeTyWidth - (b.getTypeSize(dl,fields[PIPELINE_PARAMS]) % sizeTyWidth);
    IntegerType * const int8Ty = b.getInt8Ty();
    fields[PIPELINE_PARAM_PADDING] = ArrayType::get(int8Ty, paramPaddingBytes);
    if (mUseDynamicMultithreading) {
        fields[INITIAL_SEG_NUMBER] = emptyTy;
        fields[FIXED_NUMBER_OF_THREADS] = emptyTy;
        fields[ACCUMULATED_SEGMENT_TIME] = sizeTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = sizeTy;
    } else {
        fields[INITIAL_SEG_NUMBER] = sizeTy;
        fields[FIXED_NUMBER_OF_THREADS] = sizeTy;
        fields[ACCUMULATED_SEGMENT_TIME] = emptyTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = emptyTy;
    }
    if (mUseDynamicMultithreading) {
        fields[CURRENT_THREAD_STATUS_FLAG] = sizeTy;
    } else {
        fields[CURRENT_THREAD_STATUS_FLAG] = emptyTy;
    }
    IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);
    assert (pThreadTy == b.getModule()->getFunction("pthread_self")->getReturnType());
    fields[CURRENT_THREAD_ID] = pThreadTy;
    if (LLVM_LIKELY(!mIsNestedPipeline || PipelineHasTerminationSignal)) {
        fields[TERMINATION_SIGNAL] = sizeTy;
    } else {
        fields[TERMINATION_SIGNAL] = emptyTy;
    }

    // add padding to force this struct to be cache-line-aligned
    uint64_t structSize = 0UL;
    for (unsigned i = 0; i < THREAD_STRUCT_SIZE; ++i) {
        assert (fields[i]);
        structSize += b.getTypeSize(dl, fields[i]);
    }
    const auto cl = b.getCacheAlignment();
    const auto paddingBytes = (2 * cl) - (structSize % cl);
    fields[THREAD_STRUCT_SIZE] = ArrayType::get(int8Ty, paddingBytes);
    StructType * const sty = StructType::get(C, fields, true);
    assert (b.getTypeSize(dl, sty) == (structSize + paddingBytes));
    return sty;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initThreadStructObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeThreadStructObject(KernelBuilder & b,
                                               StructType * const threadStateTy,
                                               Value * const threadState,
                                               Value * const shared, Value * const threadLocal,
                                               const std::vector<Value *> & props,
                                               Value * const threadNum, Value * const numOfThreads) const {

    FixedArray<Value *, 2> indices2;
    indices2[0] = b.getInt32(0);
    if (LLVM_LIKELY(mTarget->isStateful())) {
        indices2[1] = b.getInt32(SHARED_STATE_PARAM);
        b.CreateAlignedStore(shared, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), PtrTyABIAlignment);
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        indices2[1] = b.getInt32(THREAD_LOCAL_PARAM);
        b.CreateAlignedStore(threadLocal, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), PtrTyABIAlignment);
    }
    const auto n = props.size();
    assert (threadStateTy->getStructElementType(PIPELINE_PARAMS)->getStructNumElements() == n);
    FixedArray<Value *, 3> indices3;
    indices3[0] = indices2[0];
    indices3[1] = b.getInt32(PIPELINE_PARAMS);

    Type * const paramStructTy = threadStateTy->getStructElementType(PIPELINE_PARAMS);
    const DataLayout & DL = b.getModule()->getDataLayout();
    for (unsigned i = 0; i < n; ++i) {
        indices3[2] = b.getInt32(i);
        const auto align = DL.getABITypeAlign(paramStructTy->getStructElementType(i)).value();
        b.CreateAlignedStore(props[i], b.CreateInBoundsGEP(threadStateTy, threadState, indices3), align);
    }

    if (mUseDynamicMultithreading) {
        Constant * const sz_ZERO = b.getSize(0);
        indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
        b.CreateAlignedStore(sz_ZERO, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
        indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        b.CreateAlignedStore(sz_ZERO, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
    } else {
        indices2[1] = b.getInt32(INITIAL_SEG_NUMBER);
        b.CreateAlignedStore(threadNum, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
        indices2[1] = b.getInt32(FIXED_NUMBER_OF_THREADS);
        b.CreateAlignedStore(numOfThreads, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readThreadStructObject(KernelBuilder & b, StructType * const threadStateTy, Value * threadState) {
    Constant * const i32_ZERO = b.getInt32(0);
    IntegerType * const sizeTy = b.getSizeTy();
    FixedArray<Value *, 2> indices2;
    indices2[0] = i32_ZERO;
    if (mTarget->isStateful()) {
        indices2[1] = b.getInt32(SHARED_STATE_PARAM);
        Type * ty = mTarget->getSharedStateType()->getPointerTo();
        setHandle(b.CreateAlignedLoad(ty, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), PtrTyABIAlignment));
    }
    if (mTarget->hasThreadLocal()) {
        indices2[1] = b.getInt32(THREAD_LOCAL_PARAM);
        Type * ty = mTarget->getThreadLocalStateType()->getPointerTo();
        setThreadLocalHandle(b.CreateAlignedLoad(ty, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), PtrTyABIAlignment));
    }
    if (mUseDynamicMultithreading) {
        if (LLVM_UNLIKELY(mIsIOProcessThread)) {
            mSegNo = b.getSize(0);
            mNumOfFixedThreads = b.getSize(1);
        }
        indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        mAccumulatedSynchronizationTimePtr = b.CreateInBoundsGEP(threadStateTy, threadState, indices2);
    } else {
        indices2[1] = b.getInt32(INITIAL_SEG_NUMBER);
        mSegNo = b.CreateAlignedLoad(sizeTy, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
        mAccumulatedSynchronizationTimePtr = nullptr;
        indices2[1] = b.getInt32(FIXED_NUMBER_OF_THREADS);
        mNumOfFixedThreads = b.CreateAlignedLoad(sizeTy, b.CreateInBoundsGEP(threadStateTy, threadState, indices2), SizeTyABIAlignment);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isProcessThread
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeProcessThreadMessage(KernelBuilder & b, StringRef label, StructType * const threadStateTy, Value * const threadState) const {
//    FixedArray<Value *, 2> indices;
//    indices[0] = b.getInt32(0);
//    indices[1] = b.getInt32(CURRENT_THREAD_ID);
//    Value * const ptr = b.CreateInBoundsGEP(threadStateTy, threadState, indices);
//    IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);
//    Value * const threadId = b.CreateLoad(pThreadTy, ptr);
//    b.CallPrintInt(label, threadId);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isProcessThread
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::isProcessThread(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(CURRENT_THREAD_ID);
    Value * const ptr = b.CreateInBoundsGEP(threadStateTy, threadState, indices);
    IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);
    auto & DL = b.getModule()->getDataLayout();
    const auto pThreadAlign = DL.getABITypeAlign(pThreadTy).value();
    Value * const threadId = b.CreateAlignedLoad(pThreadTy, ptr, pThreadAlign);
//    b.CallPrintInt("exiting thread", threadId);
    return b.CreateIsNull(threadId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkPThreadLibrary
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkPThreadLibrary(KernelBuilder & b) {

    Type * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const intTy = IntegerType::getIntNTy(b.getContext(), sizeof(int) * CHAR_BIT);
    IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);

    BEGIN_SCOPED_REGION
    FunctionType * funTy = FunctionType::get(pThreadTy, false);
    b.LinkFunction("pthread_self", funTy, (void*)&pthread_self);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 4> params;
    params[0] = pThreadTy->getPointerTo();
    params[1] = voidPtrTy;
    params[2] = voidPtrTy;
    params[3] = voidPtrTy;
    FunctionType * funTy = FunctionType::get(intTy, params, false);
    b.LinkFunction("pthread_create", funTy, (void*)&pthread_create);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 2> params;
    params[0] = pThreadTy;
    params[1] = voidPtrTy->getPointerTo();
    FunctionType * funTy = FunctionType::get(intTy, params, false);
    b.LinkFunction("pthread_join", funTy, (void*)&pthread_join);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FunctionType * funTy = FunctionType::get(intTy, false);
    b.LinkFunction("sched_yield", funTy, (void*)&sched_yield);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 1> pthreadExitArgs;
    pthreadExitArgs[0] = voidPtrTy;
    FunctionType * pthreadExitFnTy = FunctionType::get(b.getVoidTy(), pthreadExitArgs, false);
    b.LinkFunction("pthread_exit", pthreadExitFnTy, (void*)pthread_exit); // ->addAttribute(0, llvm::Attribute::AttrKind::NoReturn);
    END_SCOPED_REGION
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSingleThreadKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateSingleThreadKernelMethod(KernelBuilder & b) {
    assert (!mUseDynamicMultithreading);
    if (LLVM_LIKELY(mIsNestedPipeline)) {
        mSegNo = mExternalSegNo; assert (mExternalSegNo);
    } else {
        mSegNo = b.getSize(0);
    }
    mNumOfFixedThreads = b.getSize(1);
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        if (LLVM_UNLIKELY(!mIsNestedPipeline)) {
            initPAPIOnCurrentThread(b);
        }
        setupPAPIOnCurrentThread(b);
    }
    #endif
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        startPAPIMeasurement(b, PAPIKernelCounter::PAPI_FULL_PIPELINE_TIME);
    }
    #endif
    start(b);

    BasicBlock * const mPipelineLoop = b.CreateBasicBlock("PipelineLoop");
    BasicBlock * const mPipelineEnd = b.CreateBasicBlock("PipelineEnd");
    BasicBlock * const entryBlock = b.GetInsertBlock();
    b.CreateBr(mPipelineLoop);

    b.SetInsertPoint(mPipelineLoop);
    mMadeProgressInLastSegment = b.CreatePHI(b.getInt1Ty(), 2, "madeProgressInLastSegment");
    mMadeProgressInLastSegment->addIncoming(b.getTrue(), entryBlock);
    obtainCurrentSegmentNumber(b, entryBlock);

    branchToInitialPartition(b);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        executeKernel(b);
    }

    Value * terminated = nullptr;
    if (mIsNestedPipeline || mUseDynamicMultithreading) {
        if (PipelineHasTerminationSignal) {
            terminated = hasPipelineTerminated(b);
        }
        b.CreateBr(mPipelineEnd);
    } else {
        terminated = hasPipelineTerminated(b);
        Value * const done = b.CreateIsNotNull(terminated);
        if (LLVM_UNLIKELY(CheckAssertions && !AllowIOProcessThread)) {
            Value * const progressedOrFinished = b.CreateOr(mPipelineProgress, done);
            Value * const live = b.CreateOr(mMadeProgressInLastSegment, progressedOrFinished);
            b.CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }
        BasicBlock * const exitBlock = b.GetInsertBlock();
        mMadeProgressInLastSegment->addIncoming(mPipelineProgress, exitBlock);
        if (!UseJumpGuidedSynchronization || mIsIOProcessThread) {
            incrementCurrentSegNo(b, exitBlock);
        }
        b.CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
    }
    b.SetInsertPoint(mPipelineEnd);

    if (PipelineHasTerminationSignal) {
        Value * const ptr = getTerminationSignalPtr();
        b.CreateAlignedStore(terminated, ptr, SizeTyABIAlignment);
    }

    #ifdef PRINT_DEBUG_MESSAGES
    if (mIsNestedPipeline) {
        debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
    } else {
        debugPrint(b, "================================================= END %" PRIx64, getHandle());
    }
    #endif

    mExpectedNumOfStridesMultiplier = nullptr;
    mThreadLocalStreamSetBaseAddress = nullptr;

    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);

    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        // TODO: this isn't fully correct when this is a nested pipeline
        concludeStridesPerSegmentRecording(b);
    }

    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        recordTotalPAPIMeasurement(b);
    }
    #endif
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        updateTotalCycleCounterTime(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readTerminationSignalFromLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readTerminationSignalFromLocalState(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(TERMINATION_SIGNAL);
    return b.CreateAlignedLoad(b.getSizeTy(), b.CreateInBoundsGEP(threadStateTy, threadState, indices), SizeTyABIAlignment);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeTerminationSignalToLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeTerminationSignalToLocalState(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState, Value * const terminated) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(TERMINATION_SIGNAL);
    b.CreateAlignedStore(b.CreateZExt(terminated, b.getSizeTy()), b.CreateInBoundsGEP(threadStateTy, threadState, indices), SizeTyABIAlignment);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief copyInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> PipelineCompiler::storeDoSegmentState() const {

    const auto numOfInputs = getNumOfStreamInputs();
    const auto numOfOutputs = getNumOfStreamOutputs();

    assert (!mTarget->hasAttribute(AttrId::InternallySynchronized));

    std::vector<Value *> S;
    S.reserve(4 + numOfInputs * 5 + numOfOutputs * 6);

    auto append = [&](Value * v) {
        if (v) S.push_back(v);
    };

    append(mIsFinal);
    append(mNumOfStrides);
    append(mFixedRateFactor);
    append(mExternalSegNo);

    auto copy = [&](const Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned i = 0; i < n; ++i) {
            append(V[i]);
        }
    };

    copy(mInputIsClosed, numOfInputs);
    copy(mProcessedInputItemPtr, numOfInputs);
    copy(mAccessibleInputItems, numOfInputs);
    copy(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        assert(getInputStreamSetBuffer(i)->getHandle());
        append(getInputStreamSetBuffer(i)->getHandle());
    }

    copy(mProducedOutputItemPtr, numOfOutputs);
    copy(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    copy(mInitiallyProducedOutputItems, numOfOutputs);
    copy(mWritableOutputItems, numOfOutputs);
    copy(mConsumedOutputItems, numOfOutputs);
    for (unsigned i = 0; i < numOfOutputs; ++i) {
        assert(getOutputStreamSetBuffer(i)->getHandle());
        append(getOutputStreamSetBuffer(i)->getHandle());
    }

    return S;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief restoreInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readDoSegmentState(KernelBuilder & b, StructType * const threadStructTy, Value * const propertyState) {
    FixedArray<Value *, 3> indices3;
    indices3[0] = b.getInt32(0);
    indices3[1] = b.getInt32(PIPELINE_PARAMS);

    StructType * const paramType = cast<StructType>(threadStructTy->getStructElementType(PIPELINE_PARAMS));

    unsigned i = 0;
    #ifndef NDEBUG
    const auto n = paramType->getStructNumElements();
    #endif

    auto & DL = b.getModule()->getDataLayout();

    auto revertOne = [&](Value *& v, const bool accept) {
        if (accept) {
            assert (i < n);
            indices3[2] = b.getInt32(i);
            Value * ptr = b.CreateInBoundsGEP(threadStructTy, propertyState, indices3);
            Type * const ty = paramType->getStructElementType(i);
            const auto align = DL.getABITypeAlign(ty).value();
            v = b.CreateAlignedLoad(paramType->getStructElementType(i), ptr, align);
            ++i;
        }
    };

    revertOne(mIsFinal, mIsFinal != nullptr);
    revertOne(mNumOfStrides, mNumOfStrides != nullptr);
    revertOne(mFixedRateFactor, mFixedRateFactor != nullptr);
    revertOne(mExternalSegNo, mExternalSegNo != nullptr);

    auto revert = [&](Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned j = 0; j < n; ++j) {
            revertOne(V[j], V[j] != nullptr);
        }
    };

    const auto numOfInputs = getNumOfStreamInputs();
    revert(mInputIsClosed, numOfInputs);
    revert(mProcessedInputItemPtr, numOfInputs);
    revert(mAccessibleInputItems, numOfInputs);
    revert(mAvailableInputItems, numOfInputs);
    for (unsigned j = 0; j < numOfInputs; ++j) {
        Value * handle = nullptr;
        revertOne(handle, true); assert (handle);
        getInputStreamSetBuffer(j)->setHandle(handle);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();
    revert(mProducedOutputItemPtr, numOfOutputs);
    revert(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    revert(mInitiallyProducedOutputItems, numOfOutputs);
    revert(mWritableOutputItems, numOfOutputs);
    revert(mConsumedOutputItems, numOfOutputs);

    for (unsigned j = 0; j < numOfOutputs; ++j) {
        Value * handle = nullptr;
        revertOne(handle, true); assert (handle);
        getOutputStreamSetBuffer(j)->setHandle(handle);
    }

    assert (i == n);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief restoreInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::restoreDoSegmentState(const std::vector<Value *> & S) {

    auto o = S.begin();

    auto revertOne = [&](Value *& v, const bool accept) {
        if (accept) {
            assert (o != S.end());
            v = *o++;
        }
    };

    revertOne(mIsFinal, mIsFinal != nullptr);
    revertOne(mNumOfStrides, mNumOfStrides != nullptr);
    revertOne(mFixedRateFactor, mFixedRateFactor != nullptr);
    revertOne(mExternalSegNo, mExternalSegNo != nullptr);

    auto revert = [&](Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned j = 0; j < n; ++j) {
            revertOne(V[j], V[j] != nullptr);
        }
    };

    const auto numOfInputs = getNumOfStreamInputs();
    revert(mInputIsClosed, numOfInputs);
    revert(mProcessedInputItemPtr, numOfInputs);
    revert(mAccessibleInputItems, numOfInputs);
    revert(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        Value * handle = nullptr;
        revertOne(handle, true); assert (handle);
        getInputStreamSetBuffer(i)->setHandle(handle);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();
    revert(mProducedOutputItemPtr, numOfOutputs);
    revert(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    revert(mInitiallyProducedOutputItems, numOfOutputs);
    revert(mWritableOutputItems, numOfOutputs);
    revert(mConsumedOutputItems, numOfOutputs);

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        Value * handle = nullptr;
        revertOne(handle, true); assert (handle);
        getOutputStreamSetBuffer(i)->setHandle(handle);
    }

    assert (o == S.end());

}


}
