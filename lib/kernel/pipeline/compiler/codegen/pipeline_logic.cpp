#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindAdditionalInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::bindAdditionalInitializationArguments(KernelBuilder & b, ArgIterator & arg, const ArgIterator & arg_end) {
    bindFamilyInitializationArguments(b, arg, arg_end);
    bindRepeatingStreamSetInitializationArguments(b, arg, arg_end);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateImplicitKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateImplicitKernels(KernelBuilder & b) {
    assert (b.getModule() == mTarget->getModule());
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const_cast<Kernel *>(getKernel(i))->generateOrLoadKernel(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getABIAlignments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getABIAlignments(KernelBuilder & b) {
    auto & DL = b.getModule()->getDataLayout();
    SizeTyABIAlignment = DL.getABITypeAlign(b.getSizeTy()).value();
    Int64TyABIAlignment = DL.getABITypeAlign(b.getInt64Ty()).value();
    PtrTyABIAlignment = DL.getABITypeAlign(b.getVoidPtrTy()).value();
    Int32TyABIAlignment = DL.getABITypeAlign(b.getInt32Ty()).value();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPipelineKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addPipelineKernelProperties(KernelBuilder & b) {
    // TODO: look into improving cache locality/false sharing of this struct

    // TODO: create a non-persistent / pass through input scalar type to allow the
    // pipeline to pass an input scalar to a kernel rather than recording it needlessly?
    // Non-family kernels can be contained within the shared state but family ones
    // must be allocated dynamically.

    IntegerType * const sizeTy = b.getSizeTy();

    #ifdef ENABLE_PAPI
    if (LLVM_LIKELY(NumOfPAPIEvents > 0)) {
        mTarget->addThreadLocalScalar(b.getInt32Ty(), STATISTICS_PAPI_EVENT_SET, 0);
    }
    #endif
    if (LLVM_LIKELY(num_edges(ThreadLocalPlacement) > 0)) {
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        mTarget->addThreadLocalScalar(int8PtrTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY, 0);
        mTarget->addThreadLocalScalar(sizeTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, 0);
    }
    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    if (HasZeroExtendedStream) {
        PointerType * const voidPtrTy = b.getVoidPtrTy();
        mTarget->addThreadLocalScalar(voidPtrTy, ZERO_EXTENDED_BUFFER);
        mTarget->addThreadLocalScalar(sizeTy, ZERO_EXTENDED_SPACE);
    }

    mKernelId = 0;
    mKernel = mTarget;
    auto currentPartitionId = -1U;
    addBufferHandlesToPipelineKernel(b, PipelineInput, 0);
    addConsumerKernelProperties(b, PipelineInput);

    const auto numOfPhases = PartitionPhaseBoundaries.size();

    for (size_t phase = 1; phase < numOfPhases; ++phase) {

        const auto firstPartition = PartitionPhaseBoundaries[phase - 1];
        const auto oneAfterLastPartition = PartitionPhaseBoundaries[phase];
        const auto firstKernelInCurrentPhase = std::max(FirstKernelInPartition[firstPartition], FirstKernel);
        const auto oneAfterLastKernelInCurrentPhase = FirstKernelInPartition[oneAfterLastPartition];
        assert (oneAfterLastKernelInCurrentPhase <= PipelineOutput);

        mTarget->addInternalScalar(sizeTy, PHASE_NEXT_LOGICAL_SEGMENT_NUMBER + std::to_string(phase), getCacheLineGroupId(firstKernelInCurrentPhase));

        for (auto i = firstKernelInCurrentPhase; i < oneAfterLastKernelInCurrentPhase; ++i) {
            const auto partitionId = KernelPartitionId[i];
            const bool isRoot = (partitionId != currentPartitionId);
            currentPartitionId = partitionId;
            addInternalKernelProperties(b, i, isRoot);
            #ifdef ENABLE_PAPI
            addPAPIEventCounterKernelProperties(b, i, isRoot);
            #endif
            addProducedItemCountDeltaProperties(b, i);
            addUnconsumedItemCountProperties(b, i);
            if (LLVM_UNLIKELY(mBufferGraph[i].startsNestedSynchronizationRegion())) {
                assert (isRoot);
                assert (UseJumpGuidedSynchronization);
                mTarget->addInternalScalar(sizeTy,
                    NEXT_LOGICAL_SEGMENT_NUMBER + std::to_string(i), getCacheLineGroupId(i));
            }
        }

    }



    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        auto currentPartitionId = -1U;
        constexpr auto L = 0U;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const auto partitionId = KernelPartitionId[i];
            const bool isRoot = (partitionId != currentPartitionId);
            currentPartitionId = partitionId;
            addCycleCounterProperties(b, i, isRoot, L + i);
        }
        addCycleCounterProperties(b, PipelineOutput, true, L + PipelineOutput);
        mTarget->addThreadLocalScalar(b.getInt64Ty(), STATISTICS_CYCLE_COUNT_TOTAL,
                                      getCacheLineGroupId(PipelineOutput), ThreadLocalScalarAccumulationRule::Sum);
    }

    addRepeatingStreamSetBufferProperties(b);
    generateMetaDataForRepeatingStreamSets(b);
    #ifdef ENABLE_PAPI
    addPAPIEventCounterPipelineProperties(b);
    #endif
    if (LLVM_UNLIKELY(TraceDynamicMultithreading && mUseDynamicMultithreading)) {
        addDynamicThreadingReportProperties(b, getCacheLineGroupId(PipelineOutput + 1));
    }
    addZeroInputStructProperties(b);
    addLocalDynamicBufferStructs(b);

}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addInternalKernelProperties(KernelBuilder & b, const unsigned kernelId, const bool isRoot) {

    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (LastKernel == PipelineInput);
        return;
    }

    assert (FirstKernel <= kernelId && kernelId <= LastKernel);
    mKernelId = kernelId;
    mKernel = getKernel(kernelId);
    assert (mKernel->isGenerated());

    bool allowDataParallelExecution = false;

    #ifndef DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION
    if (LLVM_UNLIKELY(isKernelStateFree(kernelId))) {
        if (LLVM_LIKELY((mKernel->getKernelFlags() & Kernel::KernelFlags::RequiresIllustratorObject) == 0)) {
            allowDataParallelExecution = true;
            mIsStatelessKernel.set(kernelId);
        }
    }
    #endif

    const auto isInternallySynchronized = mKernel->hasAttribute(AttrId::InternallySynchronized);
    if (LLVM_UNLIKELY(isInternallySynchronized)) {
        mIsInternallySynchronized.set(kernelId);
    }

    IntegerType * const sizeTy = b.getSizeTy();

    const auto groupId = getCacheLineGroupId(kernelId);

    addTerminationProperties(b, kernelId, groupId);

    const auto name = makeKernelName(kernelId);

    const auto syncLockType = allowDataParallelExecution ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
    mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[syncLockType], groupId);
    if (isRoot) {
        addSegmentLengthSlidingWindowKernelProperties(b, kernelId, groupId);
    }

    addConsumerKernelProperties(b, kernelId);

    const auto isStateFree = allowDataParallelExecution & !isInternallySynchronized;

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        // If this is relative, the processed and produced counts can be computed from the original stream
        if (LLVM_UNLIKELY(br.isRelative())) {
            continue;
        }
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateFree)) {
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (LLVM_UNLIKELY(br.isRelative())) {
            continue;
        }
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateFree)) {
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    addBufferHandlesToPipelineKernel(b, kernelId, groupId);

    addFamilyKernelProperties(b, kernelId, groupId);

    if (LLVM_UNLIKELY(isInternallySynchronized || (mKernel->getKernelFlags() & Kernel::KernelFlags::RequiresIllustratorObject) || kernelHasAnyPipelineIllustratedStreamSet(kernelId))) {
        // TODO: only needed if its possible to loop back or if we are not guaranteed that this kernel will always fire
        mTarget->addInternalScalar(sizeTy, name + INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX, groupId);
    }

    if (LLVM_LIKELY(mKernel->isStateful())) {
        Type * sharedStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            sharedStateTy = b.getVoidPtrTy();
        } else {
            sharedStateTy = mKernel->getSharedStateType();
            assert (!sharedStateTy->isEmptyTy());
        }
        mTarget->addInternalScalar(sharedStateTy, name, groupId);
    }

    if (mKernel->hasThreadLocal()) {
        // we cannot statically allocate a "family" thread local object.
        Type * localStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            localStateTy = b.getVoidPtrTy();
        } else {
            localStateTy = mKernel->getThreadLocalStateType();
            assert (!localStateTy->isEmptyTy());
        }
        mTarget->addThreadLocalScalar(localStateTy, name + KERNEL_THREAD_LOCAL_SUFFIX, groupId);
    }

    if (LLVM_UNLIKELY(allowDataParallelExecution)) {
        mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[SYNC_LOCK_POST_INVOCATION], groupId);
    }

    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        addHistogramProperties(b, kernelId, groupId);
    }

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
        for (const auto output : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & bp = mBufferGraph[output];
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bp.isManaged() || bn.Buffer->isDynamic()) {
                const BufferPort & rd = mBufferGraph[output];
                const auto prefix = makeBufferName(kernelId, rd.Port);
                LLVMContext & C = b.getContext();
                const auto numOfConsumers = std::max(out_degree(streamSet, mConsumerGraph), 1UL);

                // segment num  0
                // new capacity 1
                // produced item count 2
                // consumer processed item count [3,n)
                Type * const traceStructTy = ArrayType::get(sizeTy, numOfConsumers + 3);
                FixedArray<Type *, 2> traceStruct;
                traceStruct[0] = traceStructTy->getPointerTo(); // pointer to trace log
                traceStruct[1] = sizeTy; // length of trace log
                mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                                   prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX, groupId);
            }
        }
    }

    if (LLVM_UNLIKELY(isRoot && DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        LLVMContext & C = b.getContext();
//        FixedArray<Type *, 2> recordStruct;
//        recordStruct[0] = sizeTy; // segment num
//        recordStruct[1] = sizeTy; // # of strides
        Type * const recordStructTy = ArrayType::get(sizeTy, 2);

        FixedArray<Type *, 4> traceStruct;
        traceStruct[0] = sizeTy; // last num of strides (to avoid unnecessary loads of the trace
                                 // log and simplify the logic for first stride)
        traceStruct[1] = recordStructTy->getPointerTo(); // pointer to trace log
        traceStruct[2] = sizeTy; // trace length
        traceStruct[3] = sizeTy; // trace capacity (for realloc)

        mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                           name + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX, groupId);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeMethod(KernelBuilder & b) {

    // TODO: if we detect a fatal error at init, we should not execute
    // the pipeline loop.

    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (LastKernel == PipelineInput);
        return;
    }

    getABIAlignments(b);

    initializeScalarValues(b);

    initializeKernelAssertions(b);

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);

    Value * terminated = nullptr;
    auto partitionId = KernelPartitionId[PipelineInput];

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto curPartitionId = KernelPartitionId[i];
        const auto isRoot = (curPartitionId != partitionId);
        partitionId = curPartitionId;
        // Family kernels must be initialized in the "main" method.
        setActiveKernel(b, i, false);
        assert (mKernelId == i);
        assert (mKernel->isGenerated());
        if (isRoot) {
            initializeStridesPerSegment(b);
        }

        if (LLVM_LIKELY(!isKernelFamilyCall(i))) {
            ArgVec args;

            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {

                Constant * sharedStateTySize = nullptr;
                if (mKernel->isStateful()) {
                    sharedStateTySize = b.getTypeSize(mKernel->getSharedStateType());
                } else {
                    sharedStateTySize = ConstantInt::getAllOnesValue(b.getSizeTy());
                }
                args.push_back(sharedStateTySize);

                Constant * threadLocalTySize = nullptr;
                if (mKernel->hasThreadLocal()) {
                    threadLocalTySize = b.getTypeSize(mKernel->getThreadLocalStateType());
                } else {
                    threadLocalTySize = ConstantInt::getAllOnesValue(b.getSizeTy());
                }
                args.push_back(threadLocalTySize);
            }
            if (LLVM_LIKELY(mKernel->isStateful())) {
                args.push_back(mKernelSharedHandle);
            }
            #ifndef NDEBUG
            unsigned expected = 0;
            #endif
            for (const auto e : make_iterator_range(in_edges(i, mScalarGraph))) {
                assert (mScalarGraph[e].Type == PortType::Input);
                assert (expected++ == mScalarGraph[e].Number);
                const auto scalar = source(e, mScalarGraph);
                Value * const scalarVal = getScalar(b, scalar);

                args.push_back(scalarVal);
            }
            addFamilyCallInitializationArguments(b, i, args);
            addRepeatingStreamSetInitializationArguments(i, args);
            #ifndef NDEBUG
            for (unsigned j = 0; j != args.size(); ++j) {
                assert (isFromCurrentFunction(b, args[j], false));
            }
            #endif
            Value * const signal = callKernelInitializeFunction(b, args);
            Value * const terminatedOnInit = b.CreateICmpNE(signal, unterminated);

            if (terminated) {
                terminated = b.CreateOr(terminated, terminatedOnInit);
            } else {
                terminated = terminatedOnInit;
            }
        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (LLVM_UNLIKELY(br.isIllustrated())) {
                registerStreamSetIllustrator(b, target(e, mBufferGraph));
            }
        }

        // Is this the last kernel in a partition? If so, store the accumulated
        // termination signal.
        if (terminated && HasTerminationSignal.test(mKernelId)) {
            Value * const signal = b.CreateSelect(terminated, aborted, unterminated);
            writeTerminationSignal(b, mKernelId, signal);
            terminated = nullptr;
        }
    }



    if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
        initDynamicThreadingReportProperties(b);
    }
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * const segmentSize) {
    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (FirstKernel == LastKernel);
        return;
    }

    getABIAlignments(b);

    initializeInitialSlidingWindowSegmentLengths(b, segmentSize);

    assert (PartitionPhaseBoundaries.size() >= 2);

    bool getInputSize = false;

    if (PartitionPhaseBoundaries.size() > 2) {
        getInputSize = true;
    } else {
        for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
            const auto streamSet = source(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isReturned())) {
                if (getReturnedBufferScaleFactor(streamSet) > 0) {
                    getInputSize = true;
                    break;
                }
            }
        }
    }

    Value * expectedSourceOutputSize = nullptr;
    if (LLVM_UNLIKELY(getInputSize)) {
        Value * bufferScaling = nullptr;
        for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
            if (LLVM_UNLIKELY(in_degree(kernel, mBufferGraph) == 0)) {
                setActiveKernel(b, kernel, false);
                assert (mKernel->isGenerated());
                FixedArray<Value *, 1> args;
                args[0] = mKernelSharedHandle;
                Value * eosVal = callKernelExpectedSourceOutputSizeFunction(b, args);
                bufferScaling = b.CreateUMax(eosVal, bufferScaling);
            }
        }
        if (bufferScaling) {
            expectedSourceOutputSize = b.CreateCeilUDivRational(bufferScaling, b.getBitBlockWidth());
        }
    }

    const Rational T{mTarget->getStride(), b.getBitBlockWidth()};
    Value * allocScale = b.CreateCeilUMulRational(segmentSize, T);
    if (LLVM_LIKELY(!mIsNestedPipeline)) {
        allocScale = b.CreateMul(allocScale, b.getScalarField(MAXIMUM_NUM_OF_THREADS));
    }
    allocateOwnedBuffers(b, allocScale, expectedSourceOutputSize, true);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeThreadLocalMethod(KernelBuilder & b) {
    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (FirstKernel == LastKernel);
        return;
    }
    getABIAlignments(b);
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true);
            assert (mKernel == kernel);
            callKernelInitializeThreadLocalFunction(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * const segmentSize) {
    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (FirstKernel == LastKernel);
        return;
    }
    getABIAlignments(b);
    assert (LastStreamSet >= FirstStreamSet);
    assert (PartitionCount > 0);
    initializeThreadLocalMemory(b, segmentSize);
    const Rational T{mTarget->getStride(), b.getBitBlockWidth()};
    allocateOwnedBuffers(b, b.CreateCeilUMulRational(segmentSize, T), nullptr, false);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateKernelMethod(KernelBuilder & b) {
    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (FirstKernel == LastKernel);
        return;
    }
    getABIAlignments(b);
    initializeKernelAssertions(b);
    initializeScalarValues(b);
    if (mIsNestedPipeline) {
        generateSingleThreadKernelMethod(b);
    } else {
        generateMultiThreadKernelMethod(b);
    }
    resetInternalBufferHandles();
    SizeTyABIAlignment = 0;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeMethod(KernelBuilder & b) {
    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (LastKernel == PipelineInput);
        return;
    }
    getABIAlignments(b);
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet() || NumOfPAPIEvents > 0)) {
        printOptionalCycleCounter(b);
        #ifdef ENABLE_PAPI
        printPAPIReportIfRequested(b);
        #endif
        printOptionalBlockingIOStatistics(b);
        printOptionalBlockedIOPerSegment(b);
        printOptionalBufferExpansionHistory(b);
        printOptionalStridesPerSegment(b);
        printProducedItemCountDeltas(b);
        printUnconsumedItemCounts(b);
        if (mGenerateTransferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::TransferredItems);
        }
        if (mGenerateDeferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::DeferredItems);
        }
        if (TraceDynamicMultithreading) {
            printDynamicThreadingReport(b);
        }
    }

    initializeScalarValues(b);
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        SmallVector<Value *, 1> args;
        if (LLVM_LIKELY(mKernel->isStateful())) {
            assert (mTarget->isStateful());
            args.push_back(mKernelSharedHandle);
        }
        if (LLVM_UNLIKELY(mKernel->hasThreadLocal())) {
            assert (mTarget->hasThreadLocal());
            args.push_back(mKernelThreadLocalHandle);
        }
        mScalarValue[i] = callKernelFinalizeFunction(b, args);
    }

    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        freeHistogramProperties(b);
    }

    deallocateRepeatingBuffers(b);
    releaseOwnedBuffers(b);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeThreadLocalMethod(KernelBuilder & b) {

    if (LLVM_UNLIKELY(FirstKernel == PipelineInput)) {
        assert (FirstKernel == LastKernel);
        return;
    }

    getABIAlignments(b);
    assert (mTarget->hasThreadLocal());

    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        assert (kernel->hasThreadLocal() || !isa<PipelineKernel>(kernel));
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true, true);
            assert (mKernel == kernel);
            SmallVector<Value *, 2> args;
            if (LLVM_LIKELY(mKernelSharedHandle != nullptr)) {
                args.push_back(mKernelSharedHandle);
            }
            args.push_back(mKernelCommonThreadLocalHandle); assert (mKernelCommonThreadLocalHandle);
            args.push_back(mKernelThreadLocalHandle); assert (mKernelThreadLocalHandle);
            callKernelFinalizeThreadLocalFunction(b, args);
//            if (LLVM_UNLIKELY(isKernelFamilyCall(i))) {
//                b.CreateFree(mKernelThreadLocalHandle);
//            }
        }
    }

    // Since all of the nested kernels thread local state is contained within
    // this pipeline thread's thread local state, freeing the pipeline's will
    // also free the inner kernels.
    if (LLVM_LIKELY(num_edges(ThreadLocalPlacement) > 0)) {
        Value * tlptr = b.getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
        b.CreateFree(tlptr);
    }
    if (LLVM_UNLIKELY(HasZeroExtendedStream)) {
        b.CreateFree(b.getScalarField(ZERO_EXTENDED_BUFFER));
    }
    freeZeroedInputBuffers(b);
}

}
