#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionEntryPoints
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::makePartitionEntryPoints(KernelBuilder & b) {

    IntegerType * const boolTy = b.getInt1Ty();
    IntegerType * const sizeTy = b.getInt64Ty();


    assert (mCurrentPipelinePhase < PartitionPhaseBoundaries.size());
    const auto firstPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase - 1];
    assert (PartitionPhaseBoundaries[mCurrentPipelinePhase] > 0);
    const auto oneAfterLastPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase];
    assert (firstPartition < oneAfterLastPartition);
    assert (oneAfterLastPartition < PartitionCount);
    const auto count = oneAfterLastPartition - firstPartition + 1;

    for (unsigned i = firstPartition; i < oneAfterLastPartition; ++i) {
        mPartitionEntryPoint[i] = b.CreateBasicBlock("Partition" + std::to_string(i));
    }
    mPartitionEntryPoint[oneAfterLastPartition] = b.CreateBasicBlock("PipelineEnd");
    for (unsigned i = firstPartition + 1; i <= oneAfterLastPartition; ++i) {
        mPartitionPipelineProgressPhi[i] =
            PHINode::Create(boolTy, count, std::to_string(i) + ".pipelineProgress", mPartitionEntryPoint[i]);
    }

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        #ifndef NDEBUG
        mPartitionStartTimePhi.reset(0U, PartitionCount - 1U);
        #endif
        for (unsigned i = firstPartition + 1; i < oneAfterLastPartition; ++i) {
            mPartitionStartTimePhi[i] =
                PHINode::Create(sizeTy, count, std::to_string(i) + ".startTimeCycleCounter", mPartitionEntryPoint[i]);
        }
    }

    for (size_t i = 0; i < PartitionCount; ++i) {
        for (size_t j = 0; j < LastStreamSet - FirstStreamSet + 1; ++j) {
            mPartitionProducedItemCountPhi[i][j] = nullptr;
        }
    }
    for (size_t i = 0; i < PartitionCount; ++i) {
        for (size_t j = FirstKernel; j <= LastKernel; ++j) {
            mPartitionTerminationSignalPhi[i][j - FirstKernel] = nullptr;
        }
    }

    for (size_t kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        mKernelTerminationSignal[kernel] = nullptr;
    }

    // Create any PHI nodes we need to propogate the current produced/consumed item counts
    // of the kernels we jump over as well as the termination signals for any kernel we may
    // need to check if its closed or not.

    const auto firstComputeKernel = FirstKernelInPartition[firstPartition];
    assert (FirstKernel <= firstComputeKernel);
    const auto oneAfterLastComputeKernel = FirstKernelInPartition[oneAfterLastPartition];
    assert (firstComputeKernel < oneAfterLastComputeKernel);
    assert (oneAfterLastComputeKernel <= PipelineOutput);

    for (auto producer = firstComputeKernel; producer < oneAfterLastComputeKernel; ++producer) {
        for (const auto output : make_iterator_range(out_edges(producer, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];

            if (bn.isConstant() || bn.isThreadLocal()) {
                continue;
            }

            const BufferPort & outputPort = mBufferGraph[output];
            const auto prefix = makeBufferName(producer, outputPort.Port);

            const auto k = streamSet - FirstStreamSet;

            auto lastReader = producer;
            for (const auto input : make_iterator_range(out_edges(mConsumerGraph[streamSet], mConsumerGraph))) {
                const auto consumer = target(input, mBufferGraph);
                if (lastReader < consumer && consumer < oneAfterLastComputeKernel) {
                    lastReader = consumer;
                }
            }
            const auto readsPartId = KernelPartitionId[lastReader];
            assert (firstPartition <= readsPartId && readsPartId < oneAfterLastPartition);
            const auto prodPrefix = prefix + "_produced@partition";
            const auto prodPartId = KernelPartitionId[producer];
            for (auto partitionId = prodPartId + 1; partitionId <= readsPartId; ++partitionId) {
                auto entryPoint = mPartitionEntryPoint[partitionId]; assert (entryPoint);
                PHINode * const phi = PHINode::Create(sizeTy, PartitionCount, prodPrefix + std::to_string(partitionId), entryPoint);
                mPartitionProducedItemCountPhi[partitionId][k] = phi;
            }

        }


    }

    // any termination signal needs to be phi-ed out if it can be read by a descendent
    // or guards the loop condition at the end of the pipeline loop.

    assert (KernelPartitionId[PipelineInput] == 0);
    assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);

    if (LLVM_LIKELY(FirstKernel != PipelineInput)) {

        BitVector toCheck(LastKernel + 1);

        auto makeTerminationSignal = [&](const size_t partitionId) {
            auto entryPoint = mPartitionEntryPoint[partitionId]; assert (entryPoint);
            const auto prefix = "terminationSignalForPartition" + std::to_string(partitionId) + "@";
            const auto firstKernelInPartition = FirstKernelInPartition[partitionId];
            toCheck.reset();

            for (auto p = firstPartition; p < partitionId; ++p) {
                if (mTerminationCheck[p]) {
                    toCheck.set(FirstKernelInPartition[p]);
                }
            }

            for (auto k = firstComputeKernel; k < firstKernelInPartition; ) {
                for (const auto e : make_iterator_range(out_edges(k, mBufferGraph))) {
                    const auto streamSet = target(e, mBufferGraph);
                    for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                        const auto consumer = target(f, mBufferGraph);
                        const auto conPartId = KernelPartitionId[consumer];
                        if (conPartId >= partitionId && conPartId <= oneAfterLastPartition) {
                            const auto s = getTerminationSignalIndex(k);
                            toCheck.set(s);
                            goto added_to_set;
                        }
                    }
                }
    added_to_set: ++k;
            }

            for (auto k = firstComputeKernel; k < firstKernelInPartition; ++k) {
                if (toCheck.test(k)) {
                    assert (FirstKernel <= firstComputeKernel && firstComputeKernel <= LastKernel);
                    PHINode * const phi = PHINode::Create(sizeTy, 2, prefix + std::to_string(k), entryPoint);
                    mPartitionTerminationSignalPhi[partitionId][k - FirstKernel] = phi;
                }
            }
        };

        for (auto partitionId = firstPartition; partitionId <= oneAfterLastPartition; ++partitionId) {
            makeTerminationSignal(partitionId);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief branchToInitialPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::branchToInitialPartition(KernelBuilder & b) {

    mCurrentPartitionId = -1U;
    size_t firstPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase - 1U];
    const auto firstKernel = FirstKernelInPartition[firstPartition];
    if (LLVM_LIKELY(firstKernel != PipelineInput)) {
        BasicBlock * const entry = mPartitionEntryPoint[firstPartition]; assert (entry);
        b.CreateBr(entry);

        b.SetInsertPoint(entry);
        setActiveKernel(b, firstKernel, true);
        assert (mKernelId == firstKernel);
        if (isMultithreaded()) {
            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_KERNEL_TOTAL});
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter)) {
                startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::TOTAL_TIME});
            } else if (LLVM_UNLIKELY(mUseDynamicMultithreading)) {
                startCycleCounter(b, CycleCounter::KERNEL_SYNCHRONIZATION);
            }
            const auto type = isDataParallel(mKernelId) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
            acquireSynchronizationLock(b, mKernelId, type, mSegNo);
            if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
                updateCycleCounter(b, mKernelId, CycleCounter::KERNEL_SYNCHRONIZATION);
            }
            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
            }
            #endif
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPartitionExitPoint
 ** ------------------------------------------------------------------------------------------------------------- */
BasicBlock * PipelineCompiler::getPartitionExitPoint(KernelBuilder & /* b */) {
    assert (mKernelId >= FirstKernel && mKernelId <= PipelineOutput);
    assert (mCurrentPartitionId < (PartitionCount - 1));
    const auto nextPartitionId = mCurrentPartitionId + 1;
    assert (PartitionPhaseBoundaries[mCurrentPipelinePhase - 1U] < nextPartitionId);
    assert (nextPartitionId <= PartitionPhaseBoundaries[mCurrentPipelinePhase]);
    BasicBlock * bb = mPartitionEntryPoint[nextPartitionId]; assert (bb);
    return bb;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionEntry
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionEntry(KernelBuilder & b) {
    #ifndef NDEBUG
    const auto firstKernelOfCurrentPhase = FirstKernelInPartition[PartitionPhaseBoundaries[mCurrentPipelinePhase - 1]];
    const auto firstKernelInNextPhase = FirstKernelInPartition[PartitionPhaseBoundaries[mCurrentPipelinePhase]];
    assert (firstKernelOfCurrentPhase <= mKernelId && mKernelId < firstKernelInNextPhase);
    #endif
    mIsPartitionRoot = false;
    const auto partitionId = KernelPartitionId[mKernelId];
    if (partitionId != mCurrentPartitionId) {
        mIsPartitionRoot = true;
        assert (FirstKernelInPartition[partitionId] == mKernelId);
        mCurrentPartitionId = partitionId;
        mCurrentPartitionRoot = mKernelId;
        assert (FirstKernelInPartition[partitionId] == mKernelId);
        mNextPartitionEntryPoint = getPartitionExitPoint(b);
        determinePartitionStrideRateScalingFactor();
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** entering partition %" PRIu64, b.getSize(mCurrentPartitionId));
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determinePartitionStrideRateScalingFactor
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::determinePartitionStrideRateScalingFactor() {
    auto l = StrideStepLength[mCurrentPartitionRoot];
    auto g = StrideStepLength[mCurrentPartitionRoot];
    const auto firstKernelInNextPartition = FirstKernelInPartition[mCurrentPartitionId + 1];
    for (auto i = mCurrentPartitionRoot + 1U; i < firstKernelInNextPartition; ++i) {
        assert (KernelPartitionId[i] == mCurrentPartitionId);
        l = boost::lcm(l, StrideStepLength[i]);
        g = boost::gcd(g, StrideStepLength[i]);
    }
    assert (l > 0 && g > 0);
    // If a kernel within this partition has a min/max stride value that is greater
    // than the min/max stride of the partition root then when the root kernel
    // executes its final block, its partial stride may actually require the other
    // kernel executes N full strides and a final block. To accomidate this
    // possibility, the partition root scales the num of partition strides
    // full+partial strides by to the following ratio:
    mPartitionStrideRateScalingFactor = Rational{l,g};
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffersInCurrentPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionItemCounts(KernelBuilder & b, const unsigned kernel,
                                                 const unsigned targetPartitionId,
                                                 const bool fromKernelEntryBlock) {

    BasicBlock * const exitPoint = b.GetInsertBlock();

    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];

        const auto streamSet = target(e, mBufferGraph);

        // When jumping out of a partition to some subsequent one, we may have to
        // phi-out some of the produced item counts. We 3 cases to consider:

        // (1) if we've executed the kernel, we use the fully produced item count.
        // (2) if producer is the current kernel, we use the already produced phi node.
        // (3) if we have yet to execute (and will be jumping over) the kernel, load
        // the prior produced count.


        const auto k = streamSet - FirstStreamSet;

        // Select/load the appropriate produced item count
        PHINode * const prodPhi = mPartitionProducedItemCountPhi[targetPartitionId][k];
        if (prodPhi) {
            Value * produced = nullptr;
            if (kernel < mKernelId) {
                produced = mLocallyAvailableItems[streamSet];
            } else if (kernel == mKernelId && !mAllowDataParallelExecution) {
                assert (!mAllowDataParallelExecution);
                if (fromKernelEntryBlock) {
                    if (LLVM_UNLIKELY(br.isDeferred())) {
                        produced = mInitiallyProducedDeferredItemCount[streamSet];
                    } else {
                        produced = mInitiallyProducedItemCount[streamSet];
                    }
                } else if (mProducedAtJumpPhi[br.Port]) {
                    produced = mProducedAtJumpPhi[br.Port];
                } else {
                    if (LLVM_UNLIKELY(br.isDeferred())) {
                        produced = mAlreadyProducedDeferredPhi[br.Port];
                    } else {
                        produced = mAlreadyProducedPhi[br.Port];
                    }
                }
                mLocallyAvailableItems[streamSet] = produced;
            } else { // if (kernel > mKernelId) {
                StreamSetPort port;
                if (br.isRelative()) {
                    port = getReference(kernel, br.Port);
                } else {
                    port = br.Port;
                }
                const auto prefix = makeBufferName(kernel, port);
                Type * ty = nullptr;
                Value * ptr = nullptr;
                if (LLVM_UNLIKELY(br.isDeferred() && !fromKernelEntryBlock)) {
                    std::tie(ptr, ty) = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                } else {
                    std::tie(ptr, ty) = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX);
                }
                assert (ty == b.getSizeTy());
                produced = b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment, true);
                if (br.isRelative()) {
                    produced = b.CreateMulRational(produced, br.getRate().getRate());
                }
                mLocallyAvailableItems[streamSet] = produced;
            }

            assert (isFromCurrentFunction(b, produced, false));

            #ifdef PRINT_DEBUG_MESSAGES
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << makeKernelName(mKernelId) << " -> " <<
                   makeBufferName(kernel, br.Port) << "_avail = %" PRIu64;
            debugPrint(b, out.str(), produced);
            #endif

            prodPhi->addIncoming(produced, exitPoint);
        }

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutPartitionStatusFlags
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionStatusFlags(KernelBuilder & b, const unsigned targetPartitionId, const bool fromKernelEntryBlock) {

    BasicBlock * const exitPoint = b.GetInsertBlock();

    const auto firstPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase - 1];
    assert (firstPartition < targetPartitionId);
    const auto firstComputeKernel = FirstKernelInPartition[firstPartition];
    assert (targetPartitionId <= PartitionPhaseBoundaries[mCurrentPipelinePhase]);

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    const auto firstKernelOfTargetPartition = FirstKernelInPartition[targetPartitionId];
    assert (firstComputeKernel <= firstKernelOfTargetPartition);
    for (auto kernel = firstComputeKernel; kernel < firstKernelOfTargetPartition; ++kernel) {
        PHINode * const termPhi = mPartitionTerminationSignalPhi[targetPartitionId][kernel - FirstKernel];
        if (termPhi) {
            Value * const term = mKernelTerminationSignal[kernel];
            assert (isFromCurrentFunction(b, term));
            termPhi->addIncoming(term ? term : unterminated, exitPoint);
        }
    }

    PHINode * const progressPhi = mPartitionPipelineProgressPhi[targetPartitionId];
    assert (progressPhi);
    assert (isFromCurrentFunction(b, progressPhi, false));
    Value * const progress = mPipelineProgress;
    assert (isFromCurrentFunction(b, progress, false));
    progressPhi->addIncoming(progress, exitPoint);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutAllStateAndReleaseSynchronizationLocks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionStateAndReleaseSynchronizationLocks(KernelBuilder & b,
    const unsigned targetKernelId, const unsigned targetPartitionId, const bool fromKernelEntryBlock) {

    const auto firstPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase - 1];
    const auto firstComputeKernel = FirstKernelInPartition[firstPartition];
    const auto oneAfterLastPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase];
    const auto oneAfterLastComputeKernel = FirstKernelInPartition[oneAfterLastPartition];

    assert (firstComputeKernel <= mKernelId && mKernelId < oneAfterLastComputeKernel);

    const auto releaseSynchronizationUpTo = oneAfterLastComputeKernel;

    phiOutPartitionItemCounts(b, PipelineInput, targetPartitionId, fromKernelEntryBlock);
    for (auto kernel = firstComputeKernel; kernel < mKernelId; ++kernel) {
        phiOutPartitionItemCounts(b, kernel, targetPartitionId, fromKernelEntryBlock);
    }

    phiOutPartitionItemCounts(b, mKernelId, targetPartitionId, fromKernelEntryBlock);

    releaseAllSynchronizationLocksFor(b, mKernelId);

    for (auto kernel = mKernelId + 1; kernel < targetKernelId; ++kernel) {
        phiOutPartitionItemCounts(b, kernel, targetPartitionId, fromKernelEntryBlock);
        if (HasTerminationSignal.test(kernel)) {
            mKernelTerminationSignal[kernel] = readTerminationSignal(b, kernel);
        }
        if (firstComputeKernel <= kernel && kernel < releaseSynchronizationUpTo) {
            releaseAllSynchronizationLocksFor(b, kernel);
        }
    }

    phiOutPartitionStatusFlags(b, targetPartitionId, fromKernelEntryBlock);
    for (auto kernel = mKernelId + 1; kernel < targetKernelId; ++kernel) {
        mKernelTerminationSignal[kernel] = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief acquirePartitionSynchronizationLock
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::acquirePartitionSynchronizationLock(KernelBuilder & b, const unsigned firstKernelInTargetPartition, Value * const segNo) {

    const auto oneAfterLastPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase];
    const auto oneAfterLastKernelInCurrentPhase = FirstKernelInPartition[oneAfterLastPartition];
    assert (firstKernelInTargetPartition <= oneAfterLastKernelInCurrentPhase);

    if (firstKernelInTargetPartition == oneAfterLastKernelInCurrentPhase) {
        const auto lastKernelInCurrentPhase = oneAfterLastKernelInCurrentPhase - 1U;
        assert (mKernelId <= lastKernelInCurrentPhase);
        const auto type = isDataParallel(lastKernelInCurrentPhase) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;

        if (LLVM_LIKELY(mKernelId != lastKernelInCurrentPhase)) {
            #ifdef ENABLE_PAPI
            if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
                accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter)) {
                updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
            }
        }

        if (LLVM_LIKELY(mKernelId != lastKernelInCurrentPhase || type == SYNC_LOCK_POST_INVOCATION)) {


            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                startPAPIMeasurement(b, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
                startCycleCounter(b, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
            }
            acquireSynchronizationLock(b, lastKernelInCurrentPhase, type, segNo);
            if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
                updateCycleCounter(b, firstKernelInTargetPartition, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
            }
            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                accumPAPIMeasurementWithoutReset(b, firstKernelInTargetPartition, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
            }
            #endif
        }

        if (LLVM_UNLIKELY(mKernelId == lastKernelInCurrentPhase)) {
            #ifdef ENABLE_PAPI
            if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
                accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter)) {
                updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
            }
        }

    } else {

        assert (firstKernelInTargetPartition > mKernelId);
        assert (firstKernelInTargetPartition < oneAfterLastKernelInCurrentPhase);

        #ifdef ENABLE_PAPI
        if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
            accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
            startPAPIMeasurement(b, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
        }
        #endif
        if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
            if (EnableCycleCounter) {
                updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
            }
            startCycleCounter(b, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
        }

        const auto type = isDataParallel(firstKernelInTargetPartition) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, firstKernelInTargetPartition, type, segNo);

        if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
            if (EnableCycleCounter) {
                const auto partId = KernelPartitionId[firstKernelInTargetPartition];
                assert (partId < PartitionPhaseBoundaries[mCurrentPipelinePhase]);
                Value * const startTime = mCycleCounters[CycleCounter::PARTITION_JUMP_SYNCHRONIZATION];
                mPartitionStartTimePhi[partId]->addIncoming(startTime, b.GetInsertBlock());
            }
            // TODO: this fails because it could be summing the pipeline output but there isn't a slot for it
            updateCycleCounter(b, firstKernelInTargetPartition, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
        }

        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents > 0) {
            accumPAPIMeasurementWithoutReset(b, firstKernelInTargetPartition, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
        }
        #endif
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseAllSynchronizationLocksUntil
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseAllSynchronizationLocksFor(KernelBuilder & b, const unsigned kernel) {

    if (KernelPartitionId[kernel - 1] != KernelPartitionId[kernel]) {
        recordStridesPerSegment(b, kernel, b.getSize(0));
    }

    if (isDataParallel(kernel)) {
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_PRE_INVOCATION, mSegNo);
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_POST_INVOCATION, mSegNo);
    } else {
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_FULL, mSegNo);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInitiallyTerminatedPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInitiallyTerminatedPartitionExit(KernelBuilder & b) {

    b.SetInsertPoint(mKernelInitiallyTerminated);
    // NOTE: this branches to the next partition regardless of the jump target destination.

    const auto nextPartitionId = mCurrentPartitionId + 1U;
    const auto oneAfterLastPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase];
    assert (nextPartitionId <= oneAfterLastPartition);
    const auto jumpTargetId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (nextPartitionId <= jumpTargetId);
    assert (jumpTargetId <= oneAfterLastPartition);

    if (LLVM_LIKELY((nextPartitionId != jumpTargetId) && mIsPartitionRoot)) {

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (seqexit) = %" PRIu64, mSegNo);
        #endif

        const auto targetKernelId = FirstKernelInPartition[nextPartitionId];
        acquirePartitionSynchronizationLock(b, targetKernelId, mSegNo);
        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, nextPartitionId, true);
        updateLocalDynamicBufferStructsUntil(b, targetKernelId);
        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);
        b.CreateBr(mNextPartitionEntryPoint);

    } else if (mKernelJumpToNextUsefulPartition) {
        assert (mIsPartitionRoot);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (reusejump) = %" PRIu64, mSegNo);
        #endif

        mKernelInitiallyTerminatedExit = b.GetInsertBlock();
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            const auto & br = mBufferGraph[e];
            Value * produced = nullptr;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                produced = mInitiallyProducedDeferredItemCount[streamSet];
            } else {
                produced = mInitiallyProducedItemCount[streamSet];
            }
            const auto port = br.Port;
            assert (isFromCurrentFunction(b, produced, false));
            mProducedAtJumpPhi[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
        }

//        mMaximumNumOfStridesAtJumpPhi->addIncoming(b.getSize(0), mKernelInitiallyTerminatedExit);
        b.CreateBr(mKernelJumpToNextUsefulPartition);
    } else {
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (exitdirect) = %" PRIu64, mSegNo);
        #endif
        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        }
        mKernelInitiallyTerminatedExit = b.GetInsertBlock();
        updateKernelExitPhisAfterInitiallyTerminated(b);
        b.CreateBr(mKernelExit);
    }


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeJumpToNextPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeJumpToNextPartition(KernelBuilder & b) {

    assert (mIsPartitionRoot);

    b.SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto jumpPartitionId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (mCurrentPartitionId < jumpPartitionId);
    const auto targetKernelId = FirstKernelInPartition[jumpPartitionId];
    assert (targetKernelId > (mKernelId + 1U));
    const auto firstKernelInNextPhase = FirstKernelInPartition[PartitionPhaseBoundaries[mCurrentPipelinePhase]];
    assert (targetKernelId <= firstKernelInNextPhase);
    assert (firstKernelInNextPhase <= PipelineOutput);

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + makeKernelName(mKernelId) + ".jumping = %" PRIu64, mSegNo);
    #endif

    if (targetKernelId != firstKernelInNextPhase) {
        acquirePartitionSynchronizationLock(b, targetKernelId, mSegNo);
        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, jumpPartitionId, false);
        updateLocalDynamicBufferStructsUntil(b, targetKernelId);
        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);
    } else {

        if (mAllowDataParallelExecution) {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
        }
        if ((mKernelId + 1U) < targetKernelId) {
            acquirePartitionSynchronizationLock(b, targetKernelId, mSegNo);
        }
        const auto type = mAllowDataParallelExecution ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
        releaseSynchronizationLock(b, mKernelId, type, mSegNo);
        for (auto kernel = mKernelId + 1U; kernel < targetKernelId; ++kernel) {
            releaseAllSynchronizationLocksFor(b, kernel);
        }
        #ifdef ENABLE_PAPI
        if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
            accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
        }
        #endif
        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
        }

        phiOutPartitionStatusFlags(b, jumpPartitionId, false);
    }

    assert (mPartitionEntryPoint[jumpPartitionId]);

    b.CreateBr(mPartitionEntryPoint[jumpPartitionId]);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionExit(KernelBuilder & b) {

    // TODO: if any statefree kernel exists, swap counter accumulators to be thread local
    // and combine them at the end?

    const auto nextKernel = mKernelId + 1U;

    updateLocalDynamicBufferStructsUntil(b, nextKernel);

    const auto type = mAllowDataParallelExecution ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
    releaseSynchronizationLock(b, mKernelId, type, mSegNo);
    const auto oneAfterLastPartition = PartitionPhaseBoundaries[mCurrentPipelinePhase];
    const auto firstKernelInNextPhase = FirstKernelInPartition[oneAfterLastPartition];
    assert (nextKernel <= firstKernelInNextPhase);
    assert (firstKernelInNextPhase <= PipelineOutput);

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    }
    #endif

    if (LLVM_LIKELY(nextKernel != firstKernelInNextPhase)) {
        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_KERNEL_TOTAL});
        }
        #endif
        if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
            if (EnableCycleCounter) {
                startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::TOTAL_TIME});
            } else {
                startCycleCounter(b, CycleCounter::KERNEL_SYNCHRONIZATION);
            }
        }
        const auto type = isDataParallel(nextKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, nextKernel, type, mSegNo);
        if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
            updateCycleCounter(b, nextKernel, CycleCounter::KERNEL_SYNCHRONIZATION);
        }
        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            accumPAPIMeasurementWithoutReset(b, nextKernel, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
        }
        #endif
    }

    const auto nextPartitionId = KernelPartitionId[nextKernel];
    assert (mCurrentPartitionId <= nextPartitionId);
    assert (nextPartitionId <= oneAfterLastPartition);

    if (nextPartitionId != mCurrentPartitionId) {

        BasicBlock * const exitBlock = b.GetInsertBlock();
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** exiting partition %" PRIu64, b.getSize(mCurrentPartitionId));
        #endif
        b.CreateBr(mNextPartitionEntryPoint);

        b.SetInsertPoint(mNextPartitionEntryPoint);
        PHINode * const progressPhi = mPartitionPipelineProgressPhi[nextPartitionId]; assert (progressPhi);
        assert (mPipelineProgress);
        progressPhi->addIncoming(mPipelineProgress, exitBlock);
        mPipelineProgress = progressPhi;

        // Since there may be multiple paths into this kernel, phi out the start time
        // for each path.
        if (LLVM_UNLIKELY(EnableCycleCounter && nextPartitionId != oneAfterLastPartition)) {
            assert (mPartitionStartTimePhi[nextPartitionId]);
            assert (mCycleCounters[TOTAL_TIME]);
            mPartitionStartTimePhi[nextPartitionId]->addIncoming(mCycleCounters[TOTAL_TIME], exitBlock);
            mCycleCounters[TOTAL_TIME] = mPartitionStartTimePhi[nextPartitionId];
        }

        const auto n = LastStreamSet - FirstStreamSet + 1U;

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            PHINode * const phi = mPartitionProducedItemCountPhi[nextPartitionId][streamSet - FirstStreamSet];
            if (phi) {
                assert (isFromCurrentFunction(b, phi, false));
                assert (isFromCurrentFunction(b, mLocallyAvailableItems[streamSet], false));
                phi->addIncoming(mLocallyAvailableItems[streamSet], exitBlock);
                mLocallyAvailableItems[streamSet] = phi;
            }


        }

        const auto firstKernelOfNextPartition = FirstKernelInPartition[nextPartitionId];
        assert (firstKernelOfNextPartition <= PipelineOutput);

        for (auto kernel = FirstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
            PHINode * const termPhi = mPartitionTerminationSignalPhi[nextPartitionId][kernel - FirstKernel];
            if (termPhi) {
                assert (isFromCurrentFunction(b, termPhi, false));
                assert (isFromCurrentFunction(b, mKernelTerminationSignal[kernel], false));
                termPhi->addIncoming(mKernelTerminationSignal[kernel], exitBlock);
                mKernelTerminationSignal[kernel] = termPhi;
            }
        }
    }
}

} // end of namespace kernel
