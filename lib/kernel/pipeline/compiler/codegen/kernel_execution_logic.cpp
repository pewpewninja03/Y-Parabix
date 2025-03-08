#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeKernelCall
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeKernelCall(KernelBuilder & b) {

    // TODO: add MProtect to buffers and their handles.

    // TODO: send in the # of output items we want in the external buffers

    Value * doSegment;
    FunctionType * doSegFuncType;
    std::tie(doSegment, doSegFuncType) = getKernelDoSegmentFunction(b);

    #ifndef NDEBUG
    mKernelDoSegmentFunctionType = doSegFuncType;
    #endif

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b.CreateMProtect(mKernel->getSharedStateType(), mKernelSharedHandle, CBuilder::Protect::WRITE);
    }

    if (LLVM_UNLIKELY(mKernelIsInternallySynchronized || mUsesIllustrator)) {
        // TODO: only needed if its possible to loop back or if we are not guaranteed that this kernel will always fire.
        // even if it can loop back but will only loop back at the final block, we can relax the need for this by adding +1.
        const auto prefix = makeKernelName(mKernelId);
        Value * const intSegNoPtr = b.getScalarFieldPtr(prefix + INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX).first;
        mInternallySynchronizedSubsegmentNumber = b.CreateAlignedLoad(b.getSizeTy(), intSegNoPtr, SizeTyABIAlignment);
        Value * const nextSegNo = b.CreateAdd(mInternallySynchronizedSubsegmentNumber, b.getSize(1));
        b.CreateAlignedStore(nextSegNo, intSegNoPtr, SizeTyABIAlignment);
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "# " + prefix + " executing subsegment number %" PRIu64, mInternallySynchronizedSubsegmentNumber);
        #endif
    }

    mCurrentNumOfLinearStrides = mNumOfLinearStrides;

    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {

        assert (!mUsesIllustrator);
        assert (!mIsIOProcessThread);

        if (mCurrentKernelIsStateFree) {
            updateProcessedAndProducedItemCounts(b);
        }

        // If this is the final subsegment before termination, we do not release the
        // pre-invocation synchronization until *after* we've written the termination
        // status. Although most of the time, this would not be a severe issue ---
        // assuming the internal kernel can safely execute a 0-item segment --- when
        // inputs to a kernel have differing lengths, this could mean we might produce
        // output data that wouldn't be observed in a single-threaded run.

        mHasMoreInput = hasMoreInput(b);
        Value * waitToRelease = b.CreateOr(mHasMoreInput, b.CreateIsNotNull(mIsFinalInvocation));

        const auto prefix = makeKernelName(mKernelId);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "* " + prefix + "_finalInvoc = %" PRIu64, mIsFinalInvocation);
        debugPrint(b, "* " + prefix + "_waitToRelease = %" PRIu64, waitToRelease);
        #endif

        BasicBlock * const releaseSyncLock =
            b.CreateBasicBlock(prefix + "_releasePreInvocationLock", mKernelCompletionCheck);
        BasicBlock * const resumeKernelExecution =
            b.CreateBasicBlock(prefix + "_resumeKernelExecution", mKernelCompletionCheck);
        b.CreateUnlikelyCondBr(waitToRelease, resumeKernelExecution, releaseSyncLock);

        b.SetInsertPoint(releaseSyncLock);

        if (mCurrentKernelIsStateFree) {
            writeInternalProcessedAndProducedItemCounts(b, false);
        }
        releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);

        b.CreateBr(resumeKernelExecution);
        b.SetInsertPoint(resumeKernelExecution);
    }

    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeKernelName(mKernelId);
    debugPrint(b, "* " + prefix + "_isFinal = %" PRIu64, mIsFinalInvocation);
    debugPrint(b, "* " + prefix + "_executing = %" PRIu64, mNumOfLinearStrides);
    #endif

    BasicBlock * individualStrideLoop = nullptr;
    PHINode * currentIndividualStrideIndexPhi = nullptr;
    PHINode * nextIndividualStrideIndexPhi = nullptr;

    SmallVector<PHINode *, 0> outerProcessedPhis;
    SmallVector<PHINode *, 0> outerProcessedDeferredPhis;
    SmallVector<PHINode *, 0> outerProducedPhis;
    SmallVector<PHINode *, 0> outerProducedDeferredPhis;

    if (LLVM_UNLIKELY(mExecuteStridesIndividually)) {

        const auto prefix = makeKernelName(mKernelId);

        IntegerType * const sizeTy = b.getSizeTy();

        ConstantInt * const sz_ZERO = b.getSize(0);
        ConstantInt * const sz_ONE = b.getSize(1);

        Value * const isFinal = b.CreateICmpEQ(mNumOfLinearStrides, sz_ZERO);

        BasicBlock * const entry = b.GetInsertBlock();
        individualStrideLoop = b.CreateBasicBlock(prefix + "_determineNextSingleStrideArgs", mKernelCompletionCheck);
        BasicBlock * const individualStrideBody = b.CreateBasicBlock(prefix + "_individualStrideBody", mKernelCompletionCheck);

        b.CreateUnlikelyCondBr(isFinal, individualStrideBody, individualStrideLoop);

        b.SetInsertPoint(individualStrideLoop);
        currentIndividualStrideIndexPhi = b.CreatePHI(b.getSizeTy(), 2, prefix + "_currentStridePhi");
        currentIndividualStrideIndexPhi->addIncoming(sz_ZERO, entry);

        const auto indeg = in_degree(mKernelId, mBufferGraph);

        outerProcessedPhis.resize(indeg);
        outerProcessedDeferredPhis.resize(indeg);

        //bool hasRelativeInput = false;

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            PHINode * const phi = b.CreatePHI(sizeTy, 2);
            phi->addIncoming(mAlreadyProcessedPhi[br.Port], entry);
            outerProcessedPhis[br.Port.Number] = phi;
            mCurrentProcessedItemCountPhi[br.Port] = phi;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                PHINode * const phi = b.CreatePHI(sizeTy, 2);
                phi->addIncoming(mAlreadyProcessedDeferredPhi[br.Port], entry);
                outerProcessedDeferredPhis[br.Port.Number] = phi;
                mCurrentProcessedDeferredItemCountPhi[br.Port] = phi;
            }
        }

        const auto outdeg = out_degree(mKernelId, mBufferGraph);
        outerProducedPhis.resize(outdeg);
        outerProducedDeferredPhis.resize(outdeg);

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            PHINode * const phi = b.CreatePHI(sizeTy, 2);
            phi->addIncoming(mAlreadyProducedPhi[br.Port], entry);
            outerProducedPhis[br.Port.Number] = phi;
            mCurrentProducedItemCountPhi[br.Port] = phi;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                PHINode * const phi = b.CreatePHI(sizeTy, 2);
                phi->addIncoming(mAlreadyProducedDeferredPhi[br.Port], entry);
                outerProducedDeferredPhis[br.Port.Number] = phi;
                mCurrentProducedDeferredItemCountPhi[br.Port] = phi;
            }
        }

        SmallVector<Value *, 16> linearInputItems(indeg);
        SmallVector<Value *, 16> linearOutputItems(outdeg);

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            linearInputItems[br.Port.Number] = calculateNumOfLinearItems(b, br, sz_ONE, "writeKernelCall");
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            linearOutputItems[br.Port.Number] = calculateNumOfLinearItems(b, br, sz_ONE, "writeKernelCall");
        }

        Value * const nextStrideIndex = b.CreateAdd(currentIndividualStrideIndexPhi, b.getSize(1));
        BasicBlock * const oneStrideArgsExit = b.GetInsertBlock();
        b.CreateBr(individualStrideBody);

        b.SetInsertPoint(individualStrideBody);
        nextIndividualStrideIndexPhi = b.CreatePHI(sizeTy, 2, prefix + "_nextStridePhi");
        nextIndividualStrideIndexPhi->addIncoming(sz_ZERO, entry);
        nextIndividualStrideIndexPhi->addIncoming(nextStrideIndex, oneStrideArgsExit);
        PHINode * const numOfInvokedStrides = b.CreatePHI(sizeTy, 2, prefix + "_numOfInvokedStrides");
        numOfInvokedStrides->addIncoming(sz_ZERO, entry);
        numOfInvokedStrides->addIncoming(sz_ONE, oneStrideArgsExit);
        mCurrentNumOfLinearStrides = numOfInvokedStrides;

        if (mCurrentFixedRateFactor) {
            const auto stride = mKernel->getStride() * mFixedRateLCM;
            assert (stride.denominator() == 1);
            PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_fixedRateFactorPhi");
            phi->addIncoming(mCurrentFixedRateFactor, entry);
            phi->addIncoming(b.getSize(stride.numerator()), oneStrideArgsExit);
            mCurrentFixedRateFactor = phi;
        }

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            PHINode * const phi = b.CreatePHI(sizeTy, 2);
            phi->addIncoming(mAlreadyProcessedPhi[br.Port], entry);
            phi->addIncoming(outerProcessedPhis[br.Port.Number], oneStrideArgsExit);
            mCurrentProcessedItemCountPhi[br.Port] = phi;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                PHINode * const phi = b.CreatePHI(sizeTy, 2);
                phi->addIncoming(mAlreadyProcessedDeferredPhi[br.Port], entry);
                phi->addIncoming(outerProcessedDeferredPhis[br.Port.Number], oneStrideArgsExit);
                mCurrentProcessedDeferredItemCountPhi[br.Port] = phi;
            }
            PHINode * const phi2 = b.CreatePHI(sizeTy, 2);
            phi2->addIncoming(mCurrentLinearInputItems[br.Port], entry);
            phi2->addIncoming(linearInputItems[br.Port.Number], oneStrideArgsExit);
            mCurrentLinearInputItems[br.Port] = phi2;
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            PHINode * const phi = b.CreatePHI(sizeTy, 2);
            phi->addIncoming(mAlreadyProducedPhi[br.Port], entry);
            phi->addIncoming(outerProducedPhis[br.Port.Number], oneStrideArgsExit);
            mCurrentProducedItemCountPhi[br.Port] = phi;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                PHINode * const phi = b.CreatePHI(sizeTy, 2);
                phi->addIncoming(mAlreadyProducedDeferredPhi[br.Port], entry);
                phi->addIncoming(outerProducedDeferredPhis[br.Port.Number], oneStrideArgsExit);
                mCurrentProducedDeferredItemCountPhi[br.Port] = phi;
            }
            PHINode * const phi2 = b.CreatePHI(sizeTy, 2);
            phi2->addIncoming(mCurrentLinearOutputItems[br.Port], entry);
            phi2->addIncoming(linearOutputItems[br.Port.Number], oneStrideArgsExit);
            mCurrentLinearOutputItems[br.Port] = phi2;
        }

    }

    ArgVec args;

    buildKernelCallArgumentList(b, args);

    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        startPAPIMeasurement(b, PAPIKernelCounter::PAPI_KERNEL_EXECUTION);
    }
    #endif
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        startCycleCounter(b, CycleCounter::KERNEL_EXECUTION);
    }
    Value * doSegmentRetVal = nullptr;
    if (mRethrowException) {
        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const invokeOk = b.CreateBasicBlock(prefix + "_invokeOk", mKernelCompletionCheck);
        #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
        doSegmentRetVal = b.CreateInvoke(doSegFuncType, doSegment, invokeOk, mRethrowException, args);
        #else
        doSegmentRetVal = b.CreateInvoke(doSegment, invokeOk, mRethrowException, args);
        #endif
        b.SetInsertPoint(invokeOk);
    } else {
        doSegmentRetVal = b.CreateCall(doSegFuncType, doSegment, args);
    }
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        updateCycleCounter(b, mKernelId, CycleCounter::KERNEL_EXECUTION);
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_EXECUTION);
    }
    #endif
    if (mKernelCanTerminateEarly) {
        mTerminatedExplicitly = doSegmentRetVal;
        assert (doSegmentRetVal->getType()->isIntegerTy());
    } else {
        mTerminatedExplicitly = nullptr;
    }

    if (LLVM_LIKELY(!mCurrentKernelIsStateFree)) {
        updateProcessedAndProducedItemCounts(b);
        readReturnedOutputVirtualBaseAddresses(b);
    }

    if (LLVM_LIKELY(mRecordHistogramData)) {
        updateTransferredItemsForHistogramData(b);
    }

    if (LLVM_UNLIKELY(mExecuteStridesIndividually)) {

        BasicBlock * const individualStrideBodyExit = b.GetInsertBlock();
        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const individualStrideLoopExit = b.CreateBasicBlock(prefix + "_individualStrideExit", mKernelCompletionCheck);

        currentIndividualStrideIndexPhi->addIncoming(nextIndividualStrideIndexPhi, individualStrideBodyExit);

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            outerProcessedPhis[br.Port.Number]->addIncoming(mProcessedItemCount[br.Port], individualStrideBodyExit);
            if (LLVM_UNLIKELY(mCurrentProcessedDeferredItemCountPhi[br.Port] )) {
                outerProcessedDeferredPhis[br.Port.Number]->addIncoming(mProcessedDeferredItemCount[br.Port], individualStrideBodyExit);
            }
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            outerProducedPhis[br.Port.Number]->addIncoming(mProducedItemCount[br.Port], individualStrideBodyExit);
            if (LLVM_UNLIKELY(mCurrentProducedDeferredItemCountPhi[br.Port] )) {
                outerProducedDeferredPhis[br.Port.Number]->addIncoming(mProducedDeferredItemCount[br.Port], individualStrideBodyExit);
            }
        }

        if (LLVM_UNLIKELY(mUsesIllustrator)) {

            auto isCountableType = [](const Value * const ptr, const Binding & binding) {
                return ptr ? isCountable(binding) : false;
            };

            for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
                const BufferPort & br = mBufferGraph[e];
                if (LLVM_UNLIKELY(br.isIllustrated())) {
                    Value * initial = mCurrentProducedItemCountPhi[br.Port];
                    Value * produced = mProducedItemCount[br.Port];
                    if (LLVM_UNLIKELY(mCurrentProducedDeferredItemCountPhi[br.Port] )) {
                        initial = mCurrentProducedDeferredItemCountPhi[br.Port];
                        produced = mProducedDeferredItemCount[br.Port];
                    }
                    // Normally when a kernel terminates early, we just read the values in the termination block. Since we may need that
                    // value here, however, read it here.
                    if (mKernelCanTerminateEarly && isCountableType(mReturnedProducedItemCountPtr[br.Port], getOutputBinding(br.Port))) {
                        Value * finalItemCount = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProducedItemCountPtr[br.Port], SizeTyABIAlignment);
                        Value * const isFinal = b.CreateICmpNE(mTerminatedExplicitly, b.getSize(0));
                        produced = b.CreateSelect(isFinal, finalItemCount, produced);
                    }

                    illustrateStreamSet(b, target(e, mBufferGraph), initial, produced);
                }
            }
        }

        Value * done = b.CreateICmpEQ(nextIndividualStrideIndexPhi, mNumOfLinearStrides);
        if (mKernelCanTerminateEarly) {
            done = b.CreateOr(done, b.CreateICmpNE(doSegmentRetVal, b.getSize(0)));
        }
        b.CreateCondBr(done, individualStrideLoopExit, individualStrideLoop);

        b.SetInsertPoint(individualStrideLoopExit);
    }

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b.CreateMProtect(mKernel->getSharedStateType(), mKernelSharedHandle, CBuilder::Protect::NONE);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief buildKernelCallArgumentList
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::buildKernelCallArgumentList(KernelBuilder & b, ArgVec & args) {

    // WARNING: any change to this must be reflected in Kernel::addDoSegmentDeclaration, Kernel::getDoSegmentFields,
    // Kernel::setDoSegmentProperties, Kernel::getDoSegmentProperties and PipelineKernel::addOrDeclareMainFunction

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    unsigned numOfAddressableItemCount = 0;
    unsigned numOfVirtualBaseAddresses = 0;

    auto addNextArg = [&](Value * arg) {

        #ifndef NDEBUG
        assert ("null argument" && arg);
        const auto n = mKernelDoSegmentFunctionType->getNumParams();
        if (LLVM_UNLIKELY(args.size() >= n)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << mKernel->getName() << ": "
                   "was given too many arguments";
            throw std::runtime_error(out.str().str());
        }

        Type * const argTy = mKernelDoSegmentFunctionType->getParamType(args.size());
        if (LLVM_UNLIKELY(argTy != arg->getType())) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);

            Function * const func = mKernel->getDoSegmentFunction(b, true);

            out << mKernel->getName() << ": "
                "invalid argument type for ";

            auto argItr = func->arg_begin();
            std::advance(argItr, args.size());
            out << argItr->getName();

            out << " (#" << args.size()
                << "): expected ";
            argTy->print(out);
            out << " but got ";
            arg->getType()->print(out);
            assert (false);
            report_fatal_error(out.str());
        }
        #endif
        args.push_back(arg);
    };

    auto addItemCountArg = [&](const BufferPort & port,
                               const bool forceAddressability,
                               Value * const itemCount) -> Value * {

        if (LLVM_UNLIKELY(port.isRelative())) {
            return nullptr;
        }
        const Binding & binding = port.Binding;
        Value * ptr = nullptr;
        if (forceAddressability || isAddressable(binding)) {
            if (LLVM_UNLIKELY(numOfAddressableItemCount == mAddressableItemCountPtr.size())) {
                auto aic = b.CreateAllocaAtEntryPoint(b.getSizeTy());
                mAddressableItemCountPtr.push_back(aic);
            }
            ptr = mAddressableItemCountPtr[numOfAddressableItemCount++];
            b.CreateAlignedStore(itemCount, ptr, SizeTyABIAlignment);
            addNextArg(ptr);
        } else if (isCountable(binding)) {
            addNextArg(itemCount);
        }
        return ptr;
    };


    args.reserve(4 + (numOfInputs + numOfOutputs) * 4);
    if (LLVM_LIKELY(mKernelSharedHandle)) {
        addNextArg(mKernelSharedHandle);
    }
    assert (mKernelThreadLocalHandle == nullptr || !mKernelThreadLocalHandle->getType()->isEmptyTy());
    if (LLVM_UNLIKELY(mKernelThreadLocalHandle)) {
        if (LLVM_UNLIKELY(mIsOptimizationBranch)) {
            ConstantInt * i32_ZERO = b.getInt32(0);
            FixedArray<Value *, 3> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_ZERO;
            offset[2] = i32_ZERO;
            Value * const branchTypePtr = b.CreateGEP(mKernel->getThreadLocalStateType(), mKernelThreadLocalHandle, offset);
            b.CreateStore(mOptimizationBranchSelectedBranch, branchTypePtr);
        }
        addNextArg(mKernelThreadLocalHandle);
    }
    if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
        addNextArg(mInternallySynchronizedSubsegmentNumber);
    }
    addNextArg(mCurrentNumOfLinearStrides);
    if (mCurrentFixedRateFactor) {
        addNextArg(mCurrentFixedRateFactor);
    }
    #ifdef ENABLE_PAPI
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
        addNextArg(PAPIEventSetId);
    }
    #endif

    PointerType * const voidPtrTy = b.getVoidPtrTy();

    for (unsigned i = 0; i < numOfInputs; ++i) {
        const StreamSetPort inputPort{PortType::Input, i};
        const auto port = getInput(mKernelId, inputPort);
        const BufferPort & rt = mBufferGraph[port];

        assert (rt.Port.Number == inputPort.Number);
        if (LLVM_LIKELY(rt.Port.Reason == ReasonType::Explicit)) {
            Value * processed = nullptr;

            if (rt.isDeferred()) {
                processed = mCurrentProcessedDeferredItemCountPhi[rt.Port];
            } else {
                processed = mCurrentProcessedItemCountPhi[rt.Port];
            }
            assert (processed);

            Value * const addr = mInputVirtualBaseAddressPhi[inputPort]; assert (addr);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, makeBufferName(mKernelId, inputPort) + "_processed = %" PRIu64, processed);
            debugPrint(b, makeBufferName(mKernelId, inputPort) + "_addr = %" PRIx64, addr);
            #endif
            addNextArg(b.CreatePointerCast(addr, voidPtrTy));
            if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
                Value * isExhausted = nullptr;
                if (mExhaustedInputPortPhi[inputPort]) {
                    isExhausted = mExhaustedInputPortPhi[inputPort];
                } else {
                    isExhausted = mExhaustedInputPort[inputPort]; assert (isExhausted);
                }
                addNextArg(isExhausted);
            }

            mReturnedProcessedItemCountPtr[inputPort] = addItemCountArg(rt, rt.isDeferred(), processed);

            if (LLVM_UNLIKELY(requiresItemCount(rt.Binding))) {
                // calculate how many linear items are from the *deferred* position
                Value * inputItems = mLinearInputItemsPhi[inputPort]; assert (inputItems);
                if (rt.isDeferred()) {
                    const auto prefix = makeBufferName(mKernelId, inputPort);
                    Value * diff = b.CreateSub(mCurrentProcessedItemCountPhi[inputPort], mCurrentProcessedDeferredItemCountPhi[rt.Port], prefix + "_deferredItems");
                    inputItems = b.CreateAdd(inputItems, diff);
                }
                addNextArg(inputItems);
            }
        }
    }

    PointerType * const voidPtrPtrTy = voidPtrTy->getPointerTo();

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto port = getOutput(mKernelId, StreamSetPort(PortType::Output, i));
        const BufferPort & rt = mBufferGraph[port];

        assert (rt.Port.Reason == ReasonType::Explicit);
        assert (rt.Port.Type == PortType::Output);

        const auto streamSet = target(port, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        const StreamSetBuffer * const buffer = bn.Buffer;

        Value * produced = nullptr;
        if (rt.isDeferred()) {
            produced = mCurrentProducedDeferredItemCountPhi[rt.Port];
        } else {
            produced = mCurrentProducedItemCountPhi[rt.Port];
        }
        assert (produced);

        if (LLVM_UNLIKELY(rt.isShared())) {
            addNextArg(b.CreatePointerCast(buffer->getHandle(), voidPtrTy));
        } else if (LLVM_UNLIKELY(rt.isManaged())) {
            if (LLVM_UNLIKELY(numOfVirtualBaseAddresses == mVirtualBaseAddressPtr.size())) {
                auto vba = b.CreateAllocaAtEntryPoint(voidPtrTy);
                mVirtualBaseAddressPtr.push_back(vba);
            }
            Value * ptr = mVirtualBaseAddressPtr[numOfVirtualBaseAddresses++];
            ptr = b.CreatePointerCast(ptr, buffer->getPointerType()->getPointerTo());
            b.CreateAlignedStore(buffer->getBaseAddress(b), ptr, PtrTyABIAlignment);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, makeBufferName(mKernelId, rt.Port) + "_produced = %" PRIu64, produced);
            debugPrint(b, makeBufferName(mKernelId, rt.Port) + "_ba = %" PRIx64, buffer->getBaseAddress(b));
            #endif
            addNextArg(b.CreatePointerCast(ptr, voidPtrPtrTy));
            mReturnedOutputVirtualBaseAddressPtr[rt.Port] = ptr;
        } else {

            Value * const vba = getVirtualBaseAddress(b, rt, bn, produced, bn.isNonThreadLocal(), true);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, makeBufferName(mKernelId, rt.Port) + "_produced = %" PRIu64, produced);
            debugPrint(b, makeBufferName(mKernelId, rt.Port) + "_vba = %" PRIx64, vba);
            #endif
            addNextArg(b.CreatePointerCast(vba, voidPtrTy));
        }

        mReturnedProducedItemCountPtr[rt.Port] = addItemCountArg(rt, rt.isDeferred() || mKernelCanTerminateEarly, produced);

        if (LLVM_UNLIKELY(rt.isShared() || rt.isManaged())) {
            addNextArg(readConsumedItemCount(b, streamSet));
        } else if (requiresItemCount(rt.Binding)) {
            addNextArg(mLinearOutputItemsPhi[rt.Port]);
        }

    }
    assert (args.size() == mKernelDoSegmentFunctionType->getNumParams());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateProcessedAndProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateProcessedAndProducedItemCounts(KernelBuilder & b) {

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    const auto mustExplicitlyTerminate = mKernel->hasAttribute(AttrId::MustExplicitlyTerminate);

    Value * rejectedTermSignal = nullptr;
    if (mustExplicitlyTerminate) {
        assert (mTerminatedExplicitly && !mCurrentKernelIsStateFree);
        rejectedTermSignal = b.CreateAnd(b.CreateIsNull(mCurrentNumOfLinearStrides), b.CreateIsNull(mTerminatedExplicitly));
    }

    size_t principalProducerPartId = 0;

    if (LLVM_UNLIKELY(mHasPrincipalInput)) {
        for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & port = mBufferGraph[e];
            if (LLVM_UNLIKELY(port.isPrincipal())) {
                const auto streamSet = source(e, mBufferGraph);
                principalProducerPartId = KernelPartitionId[parent(streamSet, mBufferGraph)];
                break;
            }
        }
    }

    // calculate or read the item counts (assuming this kernel did not terminate)
    for (unsigned i = 0; i < numOfInputs; ++i) {
        Value * processed = nullptr;
        const auto inputPort = StreamSetPort{PortType::Input, i};
        const auto inputEdge = getInput(mKernelId, inputPort);
        const auto & port = mBufferGraph[inputEdge];
        const Binding & input = port.Binding;
        const ProcessingRate & rate = input.getRate();
        if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum() || rate.isGreedy())) {

            Value * inputItems = mCurrentLinearInputItems[inputPort];
            if (LLVM_UNLIKELY(mustExplicitlyTerminate && port.TransitiveAdd > 0 && rate.isFixed())) {
                inputItems = revertTransitiveAddCalculation(b, rate, inputItems, rejectedTermSignal);
            }
            processed = b.CreateAdd(mCurrentProcessedItemCountPhi[inputPort], inputItems);

            // If we have a principal input, that input may be longer than the other fixed rate
            // streamsets. We need to correct for this to ensure the processed item count is
            // always less than or equal to the available items.

            // TODO: can we determine whether an input is guaranteed to have more items than
            // the principal input?

            if (LLVM_UNLIKELY(mHasPrincipalInput && rate.isFixed() && !port.isPrincipal())) {
                const auto streamSet = source(inputEdge, mBufferGraph);
                const auto partId = KernelPartitionId[parent(streamSet, mBufferGraph)];
                if (partId != principalProducerPartId) {
                    processed = b.CreateUMin(processed, mLocallyAvailableItems[streamSet]);
                }
            }

            assert (input.isDeferred() ^ (mCurrentProcessedDeferredItemCountPhi[inputPort] == nullptr));
            if (mCurrentProcessedDeferredItemCountPhi[inputPort]) {
                assert (mReturnedProcessedItemCountPtr[inputPort]);
                mProcessedDeferredItemCount[inputPort] = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProcessedItemCountPtr[inputPort], SizeTyABIAlignment);
                #ifdef PRINT_DEBUG_MESSAGES
                const auto prefix = makeBufferName(mKernelId, inputPort);
                debugPrint(b, prefix + "_processed_deferred' = %" PRIu64, mProcessedDeferredItemCount[inputPort]);
                #endif
                if (LLVM_UNLIKELY(CheckAssertions)) {
                    Value * const deferred = mProcessedDeferredItemCount[inputPort];
                    Value * const isDeferred = b.CreateICmpULE(deferred, processed);
                    Value * const isFinal = mIsFinalInvocationPhi;
                    // TODO: workaround now for ScanMatch; if it ends with a match on a
                    // block-aligned boundary the start of the next match seems to be one
                    // after? Revise the logic to only perform a 0-item final block on
                    // kernels that may produce Add'ed data? Define the final/non-final
                    // contract first.
                    Value * const isDeferredOrFinal = b.CreateOr(isDeferred, b.CreateIsNotNull(isFinal));
                    b.CreateAssert(isDeferredOrFinal,
                                    "%s.%s: deferred processed item count (%" PRIu64 ") "
                                    "exceeds non-deferred (%" PRIu64 ")",
                                    mCurrentKernelName,
                                    b.GetString(input.getName()),
                                    deferred, processed);
                }
            }
        } else if (rate.isBounded() || rate.isUnknown()) {
            assert (mReturnedProcessedItemCountPtr[inputPort]);
            processed = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProcessedItemCountPtr[inputPort], SizeTyABIAlignment);
        } else {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Kernel " << mKernel->getName() << ":" << input.getName()
                << " has an " << "input" << " rate that is not properly handled by the PipelineKernel";
            report_fatal_error(StringRef(out.str()));
        }

        mProcessedItemCount[inputPort] = processed; assert (processed);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, inputPort);
        debugPrint(b, prefix + "_processed' = %" PRIu64, mProcessedItemCount[inputPort]);
        #endif
    }

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto outputPort = StreamSetPort{PortType::Output, i};
        Value * produced = nullptr;
        const Binding & output = getOutputBinding(outputPort);
        const ProcessingRate & rate = output.getRate();
        if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum())) {
            produced = b.CreateAdd(mCurrentProducedItemCountPhi[outputPort], mCurrentLinearOutputItems[outputPort]);
            assert (output.isDeferred() ^ (mCurrentProducedDeferredItemCountPhi[outputPort] == nullptr));
            if (mCurrentProducedDeferredItemCountPhi[outputPort]) {
                assert (mReturnedProducedItemCountPtr[outputPort]);
                mProducedDeferredItemCount[outputPort] = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProducedItemCountPtr[outputPort], SizeTyABIAlignment);
                #ifdef PRINT_DEBUG_MESSAGES
                const auto prefix = makeBufferName(mKernelId, outputPort);
                debugPrint(b, prefix + "_produced_deferred' = %" PRIu64, mProcessedDeferredItemCount[outputPort]);
                #endif
                if (LLVM_UNLIKELY(CheckAssertions)) {
                    Value * const deferred = mProducedDeferredItemCount[outputPort];
                    Value * const isDeferred = b.CreateICmpULE(deferred, produced);
                    Value * const isFinal = mIsFinalInvocationPhi;
                    // TODO: workaround now for ScanMatch; if it ends with a match on a
                    // block-aligned boundary the start of the next match seems to be one
                    // after? Revise the logic to only perform a 0-item final block on
                    // kernels that may produce Add'ed data? Define the final/non-final
                    // contract first.
                    Value * const isDeferredOrFinal = b.CreateOr(isDeferred, b.CreateIsNotNull(isFinal));
                    b.CreateAssert(isDeferredOrFinal,
                                    "%s.%s: deferred processed item count (%" PRIu64 ") "
                                    "exceeds non-deferred (%" PRIu64 ")",
                                    mCurrentKernelName,
                                    b.GetString(output.getName()),
                                    deferred, produced);
                }
            }
        } else if (rate.isBounded() || rate.isUnknown()) {
            assert (mReturnedProducedItemCountPtr[outputPort]);
            produced = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProducedItemCountPtr[outputPort], SizeTyABIAlignment);
        } else if (rate.isRelative()) {
            const auto refPort = getReference(outputPort);
            Value * itemCount = nullptr;
            Value * deferredItemCount = nullptr;
            if (LLVM_LIKELY(refPort.Type == PortType::Input)) {
                itemCount = mProcessedItemCount[refPort];
                if (mProcessedDeferredItemCount[refPort]) {
                    deferredItemCount = mProcessedDeferredItemCount[refPort];
                }
            } else {
                itemCount = mProducedItemCount[refPort];
                if (mProducedDeferredItemCount[refPort]) {
                    deferredItemCount = mProducedDeferredItemCount[refPort];
                }
            }
            if (deferredItemCount) {
                mProducedDeferredItemCount[outputPort] = b.CreateMulRational(deferredItemCount, rate.getRate());
            }
            produced = b.CreateMulRational(itemCount, rate.getRate());
        } else {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Kernel " << mKernel->getName() << ":" << output.getName()
                << " has an " << "output" << " rate that is not properly handled by the PipelineKernel";
            report_fatal_error(StringRef(out.str()));
        }

        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, StreamSetPort{PortType::Output, i});
        debugPrint(b, prefix + "_produced' = %" PRIu64, produced);
        #endif

        if (LLVM_UNLIKELY(CheckAssertions)) {
            if (mReturnedProducedItemCountPtr[outputPort]) {
                const auto port = getOutput(mKernelId, outputPort);
                const auto streamSet = target(port, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (LLVM_LIKELY((bn.isInternal() || bn.isReturned()) && bn.isOwned() && bn.isNonThreadLocal())) {
                    Value * const writable = getWritableOutputItems(b, mBufferGraph[port]);
                    Value * const delta = b.CreateSub(produced, mCurrentProducedItemCountPhi[outputPort]);
                    Value * const withinCapacity = b.CreateICmpULE(delta, writable);
                    const Binding & output = getOutputBinding(outputPort);
                    b.CreateAssert(withinCapacity,
                                    "%s.%s: reported produced item count delta (%" PRIu64 ") "
                                    "exceeds writable items (%" PRIu64 ")",
                                    mCurrentKernelName,
                                    b.GetString(output.getName()),
                                    delta, writable);
                }
            }
        }
        mProducedItemCount[outputPort] = produced;
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInternalProcessedAndProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInternalProcessedAndProducedItemCounts(KernelBuilder & b, const bool atTermination) {

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    // calculate or read the item counts (assuming this kernel did not terminate)


    for (unsigned i = 0; i < numOfInputs; ++i) {
        const auto inputPort = StreamSetPort{PortType::Input, i};
        Value * const ic = atTermination ? mProcessedItemCountAtTerminationPhi[inputPort] : mProcessedItemCount[inputPort];
        b.CreateAlignedStore(ic, mProcessedItemCountPtr[inputPort], SizeTyABIAlignment);
    }

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto outputPort = StreamSetPort{PortType::Output, i};
        Value * const ic = atTermination ? mProducedAtTermination[outputPort] : mProducedItemCount[outputPort];
        b.CreateAlignedStore(ic, mProducedItemCountPtr[outputPort], SizeTyABIAlignment);
    }

}

}
