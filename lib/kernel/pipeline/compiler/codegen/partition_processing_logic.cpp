#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionEntryPoints
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::makePartitionEntryPoints(KernelBuilder & b) {

    assert (LastComputePartitionId < (PartitionCount - 1));
    IntegerType * const boolTy = b.getInt1Ty();
    IntegerType * const sizeTy = b.getInt64Ty();

    #ifndef NDEBUG
    for (unsigned i = 0 + 1; i < PartitionCount; ++i) {
        mPartitionEntryPoint[i] = nullptr;
    }
    #endif

    assert (KernelPartitionId[FirstKernel] == 1 || FirstKernel == PipelineInput);
    assert (FirstComputePartitionId > 0 || FirstKernel == PipelineInput);
    assert (FirstComputePartitionId <= LastComputePartitionId);
    assert (KernelPartitionId[LastKernel] == (PartitionCount - 2));
    assert (LastComputePartitionId < (PartitionCount - 1));

    if (mIsIOProcessThread) {
        assert (!mIsNestedPipeline);
        assert (FirstKernel != PipelineInput);
        BasicBlock * const pipelineEnd =  b.CreateBasicBlock("PipelineEnd");
        for (unsigned i = 1; i < FirstComputePartitionId; ++i) {
            assert (mPartitionEntryPoint[i] == nullptr);
            mPartitionEntryPoint[i] = b.CreateBasicBlock("Partition" + std::to_string(i), pipelineEnd);
        }
        for (unsigned i = LastComputePartitionId + 1; i < (PartitionCount - 1); ++i) {
            assert (mPartitionEntryPoint[i] == nullptr);
            mPartitionEntryPoint[i] = b.CreateBasicBlock("Partition" + std::to_string(i), pipelineEnd);
        }
        assert (mPartitionEntryPoint[(PartitionCount - 1)] == nullptr);
        mPartitionEntryPoint[(PartitionCount - 1)] = pipelineEnd;
        size_t skip = 1;
        if (LLVM_UNLIKELY(FirstComputePartitionId == 1)) {
            skip = 2;
        } else {
            for (unsigned i = 2; i < FirstComputePartitionId; ++i) {
                mPartitionPipelineProgressPhi[i] =
                    PHINode::Create(boolTy, PartitionCount, std::to_string(i) + ".pipelineProgress", mPartitionEntryPoint[i]);
            }
        }
        for (unsigned i = (LastComputePartitionId + skip); i < PartitionCount; ++i) {
            mPartitionPipelineProgressPhi[i] =
                PHINode::Create(boolTy, PartitionCount, std::to_string(i) + ".pipelineProgress", mPartitionEntryPoint[i]);
        }
    } else if (LLVM_LIKELY(FirstKernel != PipelineInput)) {
        BasicBlock * const pipelineEnd =  b.CreateBasicBlock("PipelineEnd");
        for (unsigned i = FirstComputePartitionId; i <= LastComputePartitionId; ++i) {
            assert (mPartitionEntryPoint[i] == nullptr);
            mPartitionEntryPoint[i] = b.CreateBasicBlock("Partition" + std::to_string(i), pipelineEnd);
        }
        assert (mPartitionEntryPoint[(PartitionCount - 1)] == nullptr);
        mPartitionEntryPoint[PartitionCount - 1] = pipelineEnd;
        for (unsigned i = FirstComputePartitionId + 1; i <= LastComputePartitionId; ++i) {
            mPartitionPipelineProgressPhi[i] =
                PHINode::Create(boolTy, PartitionCount, std::to_string(i) + ".pipelineProgress", mPartitionEntryPoint[i]);
        }
        mPartitionPipelineProgressPhi[(PartitionCount - 1)] =
            PHINode::Create(boolTy, PartitionCount, ".pipelineProgress", mPartitionEntryPoint[(PartitionCount - 1)] );
    }

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        if (mIsIOProcessThread) {
            for (unsigned i = 2; i < FirstComputePartitionId; ++i) {
                mPartitionStartTimePhi[i] =
                    PHINode::Create(sizeTy, PartitionCount, std::to_string(i) + ".startTimeCycleCounter", mPartitionEntryPoint[i]);
            }
            for (unsigned i = FirstComputePartitionId; i <= LastComputePartitionId; ++i) {
                mPartitionStartTimePhi[i] = nullptr;
            }
            for (unsigned i = LastComputePartitionId + 1; i < (PartitionCount - 1); ++i) {
                mPartitionStartTimePhi[i] =
                    PHINode::Create(sizeTy, PartitionCount, std::to_string(i) + ".startTimeCycleCounter", mPartitionEntryPoint[i]);
            }
        } else {
            for (unsigned i = FirstComputePartitionId + 1; i <= LastComputePartitionId; ++i) {
                mPartitionStartTimePhi[i] =
                    PHINode::Create(sizeTy, PartitionCount, std::to_string(i) + ".startTimeCycleCounter", mPartitionEntryPoint[i]);
            }
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

    if (mIsIOProcessThread) {
        return;
    }

    // Create any PHI nodes we need to propogate the current produced/consumed item counts
    // of the kernels we jump over as well as the termination signals for any kernel we may
    // need to check if its closed or not.

    const auto firstComputeKernel = FirstKernelInPartition[FirstComputePartitionId];

    if (LLVM_LIKELY(FirstStreamSet > PipelineOutput)) {

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const BufferNode & bn = mBufferGraph[streamSet];

            if (LLVM_UNLIKELY(bn.isConstant())) {
                continue;
            }

            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            if (LLVM_UNLIKELY(producer < firstComputeKernel)) {
                continue;
            }

            const BufferPort & outputPort = mBufferGraph[output];
            const auto prefix = makeBufferName(producer, outputPort.Port);

            const auto k = streamSet - FirstStreamSet;

            auto lastReader = producer;
            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(input, mBufferGraph);
                lastReader = std::max(lastReader, consumer);
            }
            const auto readsPartId = std::min(KernelPartitionId[lastReader], LastComputePartitionId);
            assert (FirstComputePartitionId <= readsPartId && readsPartId <= LastComputePartitionId);
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
            const auto l = FirstKernelInPartition[partitionId];
            toCheck.reset();

            const auto final = std::min<size_t>(partitionId, (LastComputePartitionId + 1));
            for (auto p = FirstComputePartitionId; p < final; ++p) {
                if (mTerminationCheck[p]) {
                    toCheck.set(FirstKernelInPartition[p]);
                }
            }

            for (auto k = firstComputeKernel; k < l; ) {
                for (const auto e : make_iterator_range(out_edges(k, mBufferGraph))) {
                    const auto streamSet = target(e, mBufferGraph);
                    for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                        const auto consumer = target(f, mBufferGraph);
                        const auto conPartId = KernelPartitionId[consumer];
                        if (conPartId >= partitionId && (conPartId <= LastComputePartitionId || conPartId == (PartitionCount - 1))) {
                            const auto s = getTerminationSignalIndex(k);
                            toCheck.set(s);
                            goto added_to_set;
                        }
                    }
                }
    added_to_set: ++k;
            }

            for (auto k = firstComputeKernel; k < l; ++k) {
                if (toCheck.test(k)) {
                    PHINode * const phi = PHINode::Create(sizeTy, 2, prefix + std::to_string(k), entryPoint);
                    mPartitionTerminationSignalPhi[partitionId][k - FirstKernel] = phi;
                }
            }
        };

        for (auto partitionId = FirstComputePartitionId; partitionId <= LastComputePartitionId; ++partitionId) {
            makeTerminationSignal(partitionId);
        }
        assert (KernelPartitionId[PipelineOutput] == (PartitionCount - 1U));
        makeTerminationSignal(PartitionCount - 1U);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief branchToInitialPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::branchToInitialPartition(KernelBuilder & b) {

    mCurrentPartitionId = -1U;

    size_t firstPartition;
    if (mIsIOProcessThread) {
        firstPartition = KernelPartitionId[FirstKernel];
        if (LLVM_UNLIKELY(firstPartition == FirstComputePartitionId)) {
            firstPartition = LastComputePartitionId + 1;
        }
    } else {
        firstPartition = FirstComputePartitionId;
    }

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
            } else if (LLVM_UNLIKELY(mUseDynamicMultithreading && !mIsIOProcessThread)) {
                startCycleCounter(b, CycleCounter::KERNEL_SYNCHRONIZATION);
            }
            if (LLVM_LIKELY(!mIsIOProcessThread)) {
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

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPartitionExitPoint
 ** ------------------------------------------------------------------------------------------------------------- */
BasicBlock * PipelineCompiler::getPartitionExitPoint(KernelBuilder & /* b */) {
    assert (mKernelId >= FirstKernel && mKernelId <= PipelineOutput);
    assert (mCurrentPartitionId < (PartitionCount - 1));
    auto nextPartitionId = mCurrentPartitionId + 1;
    if (mIsIOProcessThread) {
        assert (mCurrentPartitionId < FirstComputePartitionId || mCurrentPartitionId > LastComputePartitionId);
        if (nextPartitionId == FirstComputePartitionId) {
            nextPartitionId = LastComputePartitionId + 1;
        }
    } else {
        assert (FirstComputePartitionId <= mCurrentPartitionId && mCurrentPartitionId <= LastComputePartitionId);
        if (nextPartitionId > LastComputePartitionId) {
            nextPartitionId = (PartitionCount - 1);
        }
    }
    BasicBlock * bb = mPartitionEntryPoint[nextPartitionId]; assert (bb);
    return bb;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionEntry
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionEntry(KernelBuilder & b) {
    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    mIsPartitionRoot = false;
    const auto partitionId = KernelPartitionId[mKernelId];
    assert (mIsIOProcessThread ^ (FirstComputePartitionId <= partitionId && partitionId <= LastComputePartitionId));
    mUsesNestedSynchronizationVariable = false;
    mNestedSegNum = mSegNo;
    if (partitionId != mCurrentPartitionId) {
        mIsPartitionRoot = true;
        assert (FirstKernelInPartition[partitionId] == mKernelId);
        mCurrentPartitionId = partitionId;
        mCurrentPartitionRoot = mKernelId;
        assert (FirstKernelInPartition[partitionId] == mKernelId);
        mNextPartitionEntryPoint = getPartitionExitPoint(b);
        determinePartitionStrideRateScalingFactor();
        mUsesNestedSynchronizationVariable = mBufferGraph[mKernelId].startsNestedSynchronizationRegion();
        assert (!mUsesNestedSynchronizationVariable || !mIsIOProcessThread);
        if (LLVM_UNLIKELY(mUsesNestedSynchronizationVariable && (partitionId < LastComputePartitionId))) {
            mPartitionExitSegNoPhi = PHINode::Create(b.getSizeTy(), 2, "regionedSegNo", mNextPartitionEntryPoint);
        } else {
            mPartitionExitSegNoPhi = nullptr;
        }
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** entering partition %" PRIu64, b.getSize(mCurrentPartitionId));
        #endif
    } else {
        assert (!mBufferGraph[mKernelId].startsNestedSynchronizationRegion());
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
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(KernelBuilder & b) const {
    for (auto i = mKernelId; i <= LastKernel; ++i) {
        if (KernelPartitionId[i] != mCurrentPartitionId) {
            break;
        }
        loadLastGoodVirtualBaseAddressesOfUnownedBuffers(b, i);
    }
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
                assert (!mCurrentKernelIsStateFree);
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
            } else { // if (kernel > mKernelId) {
                StreamSetPort port;
                if (br.isRelative()) {
                    port = getReference(kernel, br.Port);
                } else {
                    port = br.Port;
                }
                const auto prefix = makeBufferName(kernel, port);
                Value * ptr = nullptr;
                if (LLVM_UNLIKELY(br.isDeferred() && !fromKernelEntryBlock)) {
                    ptr = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX).first;
                } else {
                    ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
                }
                produced = b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment);
                if (br.isRelative()) {
                    produced = b.CreateMulRational(produced, br.getRate().getRate());
                }
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

    assert (!mIsIOProcessThread);

    BasicBlock * const exitPoint = b.GetInsertBlock();

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    const auto firstKernelOfTargetPartition = FirstKernelInPartition[targetPartitionId];
    assert (firstKernelOfTargetPartition <= PipelineOutput);
    const auto firstKernel = FirstKernelInPartition[FirstComputePartitionId];
    assert (firstKernel <= firstKernelOfTargetPartition);
    for (auto kernel = firstKernel; kernel < firstKernelOfTargetPartition; ++kernel) {
        PHINode * const termPhi = mPartitionTerminationSignalPhi[targetPartitionId][kernel - FirstKernel];
        if (termPhi) {
            Value * const term = mKernelTerminationSignal[kernel];
            assert (isFromCurrentFunction(b, term));
            termPhi->addIncoming(term ? term : unterminated, exitPoint);
        }
    }

    assert (!mIsIOProcessThread);
    assert ((FirstComputePartitionId <= targetPartitionId && targetPartitionId <= LastComputePartitionId) || targetPartitionId == KernelPartitionId[PipelineOutput]);

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

    assert (!mIsIOProcessThread);
    assert (KernelPartitionId[FirstKernel] == 1);
    const auto firstComputeKernel =
        FirstKernelInPartition[FirstComputePartitionId];
    const auto oneAfterLastComputeKernel = FirstKernelInPartition[LastComputePartitionId + 1];

    assert (firstComputeKernel > PipelineInput);
    assert (firstComputeKernel <= mKernelId && mKernelId < oneAfterLastComputeKernel);

    const auto releaseSynchronizationUpTo = mUsesNestedSynchronizationVariable ? PipelineInput : oneAfterLastComputeKernel;

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

    assert (!mIsIOProcessThread);

    const auto afterLastComputeKernel = FirstKernelInPartition[LastComputePartitionId + 1];
    if (firstKernelInTargetPartition >= afterLastComputeKernel) {
        const auto lastComputeKernel = afterLastComputeKernel - 1U;
        const auto type = isDataParallel(lastComputeKernel) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;

        assert (firstKernelInTargetPartition <= PipelineOutput);

        if (LLVM_LIKELY(mKernelId != lastComputeKernel)) {
            #ifdef ENABLE_PAPI
            if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
                accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter)) {
                updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
            }
        }

        if (LLVM_LIKELY(mKernelId != lastComputeKernel || type == SYNC_LOCK_POST_INVOCATION)) {
            assert (mKernelId <= lastComputeKernel);

            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                startPAPIMeasurement(b, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
            }
            #endif
            if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
                startCycleCounter(b, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
            }
            acquireSynchronizationLock(b, lastComputeKernel, type, segNo);
            if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
                updateCycleCounter(b, firstKernelInTargetPartition, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
            }
            #ifdef ENABLE_PAPI
            if (NumOfPAPIEvents) {
                accumPAPIMeasurementWithoutReset(b, firstKernelInTargetPartition, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION);
            }
            #endif
        }

        if (LLVM_UNLIKELY(mKernelId == lastComputeKernel)) {
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
        assert (firstKernelInTargetPartition < PipelineOutput);

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
                assert (partId < (PartitionCount - 1));
                BasicBlock * const exitBlock = b.GetInsertBlock();
                Value * const startTime = mCycleCounters[CycleCounter::PARTITION_JUMP_SYNCHRONIZATION];
                mPartitionStartTimePhi[partId]->addIncoming(startTime, exitBlock);
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

    if (LLVM_LIKELY(!mIsIOProcessThread)) {
        if (isDataParallel(kernel)) {
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_POST_INVOCATION, mSegNo);
        } else {
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_FULL, mSegNo);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInitiallyTerminatedPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInitiallyTerminatedPartitionExit(KernelBuilder & b) {

    b.SetInsertPoint(mKernelInitiallyTerminated);

    loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(b);

    // NOTE: this branches to the next partition regardless of the jump target destination.

    auto nextPartitionId = mCurrentPartitionId + 1U;
    if (mIsIOProcessThread) {
        if (nextPartitionId == FirstComputePartitionId) {
            nextPartitionId = LastComputePartitionId + 1;
        }
    } else {
        if (nextPartitionId == (LastComputePartitionId + 1)) {
            nextPartitionId = (PartitionCount - 1);
            assert (KernelPartitionId[PipelineOutput] == (PartitionCount - 1));
        }
    }
    assert (nextPartitionId == (PartitionCount - 1) || (mIsIOProcessThread ^ (FirstComputePartitionId <= nextPartitionId && nextPartitionId <= LastComputePartitionId)));

    const auto jumpTargetId = PartitionJumpTargetId[mCurrentPartitionId];

    assert (nextPartitionId <= jumpTargetId);

    assert (jumpTargetId == (PartitionCount - 1) || (mIsIOProcessThread ^ (FirstComputePartitionId <= jumpTargetId && jumpTargetId <= LastComputePartitionId)));

    if (LLVM_LIKELY((nextPartitionId != jumpTargetId) && mIsPartitionRoot) && (!mIsIOProcessThread)) {

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (seqexit) = %" PRIu64, mSegNo);
        #endif

        assert (nextPartitionId <= LastComputePartitionId);

        const auto targetKernelId = FirstKernelInPartition[nextPartitionId];

        Value * nextSegNo = mSegNo;

        if (mUsesNestedSynchronizationVariable) {
            assert (mIsPartitionRoot);
            if (LLVM_UNLIKELY(isDataParallel(mKernelId))) {
                acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
            }
            nextSegNo = obtainNextSegmentNumber(b);
        }

        acquirePartitionSynchronizationLock(b, targetKernelId, nextSegNo);
        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, nextPartitionId, true);
        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);

        mKernelInitiallyTerminatedExit = b.GetInsertBlock();
        if (LLVM_UNLIKELY(mPartitionExitSegNoPhi)) {
            mPartitionExitSegNoPhi->addIncoming(nextSegNo, mKernelInitiallyTerminatedExit);
        }
        b.CreateBr(mNextPartitionEntryPoint);

    } else if (mKernelJumpToNextUsefulPartition) {
        assert (mIsPartitionRoot);
        assert (!mIsIOProcessThread);
        assert (!mUsesNestedSynchronizationVariable);

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

        mMaximumNumOfStridesAtJumpPhi->addIncoming(b.getSize(0), mKernelInitiallyTerminatedExit);
        b.CreateBr(mKernelJumpToNextUsefulPartition);
    } else {
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (exitdirect) = %" PRIu64, mSegNo);
        #endif

        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            assert (!mIsIOProcessThread);
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
    assert (!mIsIOProcessThread);

    b.SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto jumpPartitionId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (mCurrentPartitionId < jumpPartitionId);
    const auto targetKernelId = FirstKernelInPartition[jumpPartitionId];
    assert (targetKernelId > (mKernelId + 1U));
    assert (targetKernelId < FirstKernelInPartition[LastComputePartitionId + 1] || targetKernelId == PipelineOutput);
    assert (!mUsesNestedSynchronizationVariable || targetKernelId == PipelineOutput);


    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + makeKernelName(mKernelId) + ".jumping = %" PRIu64, mSegNo);
    #endif

    updateNextSlidingWindowSize(b, mMaximumNumOfStridesAtJumpPhi, b.getSize(0));

    if (!mUsesNestedSynchronizationVariable || targetKernelId != PipelineOutput) {
        acquirePartitionSynchronizationLock(b, targetKernelId, mSegNo);
        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, jumpPartitionId, false);
        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);
    } else {
        if (LLVM_UNLIKELY(isDataParallel(mKernelId))) {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        } else {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_FULL, mSegNo);
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

    b.CreateBr(mPartitionEntryPoint[jumpPartitionId]);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionExit(KernelBuilder & b) {

    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);


    // TODO: if any statefree kernel exists, swap counter accumulators to be thread local
    // and combine them at the end?

    auto nextKernel = mKernelId + 1;

    if (mIsIOProcessThread) {
        assert (!mUsesNestedSynchronizationVariable);
        const auto firstComputeKernel = FirstKernelInPartition[FirstComputePartitionId];
        assert (mKernelId < firstComputeKernel || mKernelId >= FirstKernelInPartition[LastComputePartitionId + 1]);
        if (nextKernel == firstComputeKernel) {
            nextKernel = FirstKernelInPartition[LastComputePartitionId + 1];
            assert (!isDataParallel(mKernelId));
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_FULL, mSegNo);
        }
    } else {
        Value * nextSegNo = mSegNo;
        if (LLVM_UNLIKELY(mUsesNestedSynchronizationVariable)) {
            assert (mIsPartitionRoot);
            nextSegNo = obtainNextSegmentNumber(b);
        }
        const auto type = isDataParallel(mKernelId) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
        releaseSynchronizationLock(b, mKernelId, type, mSegNo);
        assert (FirstKernelInPartition[FirstComputePartitionId] <= mKernelId && mKernelId < FirstKernelInPartition[LastComputePartitionId + 1]);
        if (nextKernel == FirstKernelInPartition[LastComputePartitionId + 1]) {
            nextKernel = PipelineOutput;
        }
        mSegNo = nextSegNo;
    }

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    }
    #endif

    if (LLVM_LIKELY(nextKernel < PipelineOutput)) {
        #ifdef ENABLE_PAPI
        if (NumOfPAPIEvents) {
            startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_KERNEL_TOTAL});
        }
        #endif
        if (LLVM_UNLIKELY(EnableCycleCounter || (mUseDynamicMultithreading && !mIsIOProcessThread))) {
            if (EnableCycleCounter) {
                startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::TOTAL_TIME});
            } else {
                assert (!mIsIOProcessThread);
                startCycleCounter(b, CycleCounter::KERNEL_SYNCHRONIZATION);
            }
        }
        if (LLVM_LIKELY(!mIsIOProcessThread)) {
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
    }

    const auto nextPartitionId = KernelPartitionId[nextKernel];
    assert (nextKernel != PipelineOutput || (nextPartitionId != mCurrentPartitionId));

    if (nextPartitionId != mCurrentPartitionId) {
        assert (mCurrentPartitionId < nextPartitionId);
        assert (nextPartitionId <= PartitionCount);
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

        if (LLVM_UNLIKELY(mPartitionExitSegNoPhi)) {
            mPartitionExitSegNoPhi->addIncoming(mSegNo, exitBlock);
            mSegNo = mPartitionExitSegNoPhi;
            mPartitionExitSegNoPhi = nullptr;
        }

        // Since there may be multiple paths into this kernel, phi out the start time
        // for each path.
        if (LLVM_UNLIKELY(EnableCycleCounter && nextPartitionId < (PartitionCount - 1))) {
            mPartitionStartTimePhi[nextPartitionId]->addIncoming(mCycleCounters[TOTAL_TIME], exitBlock);
            mCycleCounters[TOTAL_TIME] = mPartitionStartTimePhi[nextPartitionId];
        }

        const auto n = LastStreamSet - FirstStreamSet + 1U;

        for (unsigned i = 0; i != n; ++i) {
            PHINode * const phi = mPartitionProducedItemCountPhi[nextPartitionId][i];
            if (phi) {
                assert (isFromCurrentFunction(b, phi, false));
                const auto streamSet = FirstStreamSet + i;
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
