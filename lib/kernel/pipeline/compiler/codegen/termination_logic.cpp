#include "../pipeline_compiler.hpp"
#include <llvm/Transforms/Utils/Local.h>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addTerminationProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addTerminationProperties(KernelBuilder & b, const size_t kernelId, const size_t groupId) {
    if (HasTerminationSignal.test(kernelId)) {

        IntegerType * const sizeTy = b.getSizeTy();
        mTarget->addInternalScalar(sizeTy, TERMINATION_PREFIX + std::to_string(kernelId), groupId);
        if (in_degree(kernelId, mTerminationPropagationGraph) != 0) {
            mTarget->addInternalScalar(sizeTy, CONSUMER_TERMINATION_COUNT_PREFIX + std::to_string(kernelId), groupId);
        }

        for (auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            if (LLVM_UNLIKELY(mBufferGraph[e].isCrossThreaded())) {
                mKernelProducesCrossThreadedData.set(kernelId);
                mTarget->addInternalScalar(sizeTy, CROSS_THREADED_TERMINATION_SEGMENT_NUMBER_PREFIX + std::to_string(kernelId), groupId);
                break;
            }
        }

    } else {
        assert (in_degree(kernelId, mTerminationPropagationGraph) == 0);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getTerminationSignalIndex
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getTerminationSignalIndex(const unsigned kernel) const {
    if (HasTerminationSignal.test(kernel)) {
        return kernel;
    } else {
        const auto root = FirstKernelInPartition[KernelPartitionId[kernel]];
        assert (HasTerminationSignal.test(root));
        return root;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasKernelTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::hasKernelTerminated(KernelBuilder & b, const size_t kernel, const bool normally) {
    Value * signal = mKernelIsClosed[kernel];
    if (signal == nullptr) {
        const auto partitionId = KernelPartitionId[kernel];
        const auto isComputeThreadPartition = (FirstComputePartitionId <= partitionId) && (partitionId <= LastComputePartitionId);
        if (mIsIOProcessThread != isComputeThreadPartition) {
            const auto idx = getTerminationSignalIndex(kernel);
            signal = mKernelTerminationSignal[idx];
            assert (isFromCurrentFunction(b, signal, false));
        } else {
            signal = readTerminationSignal(b, kernel);
        }
    }
    if (normally) {
        Constant * const completed = getTerminationSignal(b, TerminationSignal::Completed);
        return b.CreateICmpEQ(signal, completed);
    } else {
        Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
        return b.CreateICmpNE(signal, unterminated);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief pipelineTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::hasPipelineTerminated(KernelBuilder & b) {

    Value * hard = nullptr;
    Value * soft = nullptr;

    assert (KernelPartitionId[PipelineInput] == 0);
    assert (KernelPartitionId[PipelineOutput] == (PartitionCount - 1));

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);
    Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);

    for (auto partitionId = 1U; partitionId < (PartitionCount - 1); ++partitionId) {
        if (const auto type = mTerminationCheck[partitionId]) {
            const auto root = FirstKernelInPartition[partitionId];
            assert (HasTerminationSignal.test(root));
            const auto onSameThread =
                ((FirstComputePartitionId <= partitionId) && (partitionId <= LastComputePartitionId)) != mIsIOProcessThread;
            if (onSameThread || (type & TerminationCheckFlag::Hard)) {
                Value * signal = nullptr;
                if (onSameThread) {
                    signal = mKernelTerminationSignal[root];
                    assert (signal);
                } else {
                    signal = readTerminationSignal(b, root);
                }
                if (type & TerminationCheckFlag::Hard) {
                    Value * const final = b.CreateICmpEQ(signal, fatal);
                    if (hard) {
                        hard = b.CreateOr(hard, final);
                    } else {
                        hard = final;
                    }
                }
                if (type & TerminationCheckFlag::Soft) {
                    Value * const final = b.CreateICmpNE(signal, unterminated);
                    if (soft) {
                        soft = b.CreateAnd(soft, final);
                    } else {
                        soft = final;
                    }
                }
            }
        }
    }
    Value * signal = aborted;
    if (soft) {
        signal = b.CreateSelect(soft, aborted, unterminated);
        if (hard) {
            signal = b.CreateSelect(hard, fatal, signal);
        }
    } else {
        assert (hard == nullptr);
    }
    return signal;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getTerminationSignal
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ Constant * PipelineCompiler::getTerminationSignal(KernelBuilder & b, const TerminationSignal type) {
    return b.getSize(static_cast<unsigned>(type));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief signalAbnormalTermination
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::signalAbnormalTermination(KernelBuilder & b) {
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);
    BasicBlock * const exitBlock = b.GetInsertBlock();
    mTerminatedSignalPhi->addIncoming(aborted, exitBlock);
    mCurrentNumOfStridesAtTerminationPhi->addIncoming(mUpdatedNumOfStrides, exitBlock);
    if (mIsPartitionRoot) {
        mPotentialSegmentLengthAtTerminationPhi->addIncoming(mPotentialSegmentLength, exitBlock);
        mFinalPartialStrideFixedRateRemainderAtTerminationPhi->addIncoming(mFinalPartialStrideFixedRateRemainderPhi, exitBlock);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isClosed
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::isClosed(KernelBuilder & b, const StreamSetPort inputPort, const bool normally) {
    return isClosed(b, getInputBufferVertex(inputPort), normally);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isClosed
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::isClosed(KernelBuilder & b, const unsigned streamSet, const bool normally) {
    const BufferNode & bn = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(bn.isConstant())) {
        return b.getFalse();
    }
    const auto e = in_edge(streamSet, mBufferGraph);
    const auto producer = source(e, mBufferGraph);
    if (LLVM_UNLIKELY(producer == PipelineInput)) {
        if (mIsNestedPipeline) {
            const BufferPort & bp = mBufferGraph[e];
            return mInputIsClosed[bp.Port.Number];
        } else {
            return mIsFinal;
        }
    }
    return hasKernelTerminated(b, producer, normally && kernelCanTerminateAbnormally(producer));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief kernelCanTerminateAbnormally
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::kernelCanTerminateAbnormally(const unsigned kernel) const {
    if (kernel != PipelineInput) {
        for (const Attribute & attr : getKernel(kernel)->getAttributes()) {
            switch (attr.getKind()) {
                case AttrId::CanTerminateEarly:
                case AttrId::MayFatallyTerminate:
                    return true;
                default: continue;
            }
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkIfKernelIsAlreadyTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkIfKernelIsAlreadyTerminated(KernelBuilder & b) {
    if (HasTerminationSignal.test(mKernelId)) {
        Value * const signal = readTerminationSignal(b, mKernelId);
        mKernelTerminationSignal[mKernelId] = signal;
        mInitialTerminationSignal = signal;
        mKernelIsClosed[mKernelId] = nullptr;
        mInitiallyTerminated = hasKernelTerminated(b, mKernelId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkPropagatedTerminationSignals
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkPropagatedTerminationSignals(KernelBuilder & b) {
    const auto n = in_degree(mKernelId, mTerminationPropagationGraph);
    if (n != 0) {

        Value * const val = b.getScalarField(CONSUMER_TERMINATION_COUNT_PREFIX + std::to_string(mKernelId));
        Value * const allConsumersFinished = b.CreateICmpEQ(val, b.getSize(n));
        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const np = b.CreateBasicBlock(prefix + "_noPropagatedTerminationSignal", mKernelCheckOutputSpace);

        BasicBlock * caughtPropagatedTerminationSignal = nullptr;
        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            caughtPropagatedTerminationSignal = b.CreateBasicBlock(prefix + "_caughtPropagatedTerminationSignal", mKernelTerminated);
        } else {
            caughtPropagatedTerminationSignal = mKernelTerminated;
        }

        b.CreateUnlikelyCondBr(allConsumersFinished, caughtPropagatedTerminationSignal, np);
        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            b.SetInsertPoint(caughtPropagatedTerminationSignal);
            assert (!mIsIOProcessThread);
            acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
            b.CreateBr(mKernelTerminated);
        }
        BasicBlock * const entryPoint = b.GetInsertBlock();
        mTerminatedSignalPhi->addIncoming(getTerminationSignal(b, TerminationSignal::Aborted), entryPoint);
        mCurrentNumOfStridesAtTerminationPhi->addIncoming(mCurrentNumOfStridesAtLoopEntryPhi, entryPoint);
        if (mIsPartitionRoot) {
            Constant * const ZERO = b.getSize(0);
            mPotentialSegmentLengthAtTerminationPhi->addIncoming(ZERO, entryPoint);
            mFinalPartialStrideFixedRateRemainderAtTerminationPhi->addIncoming(ZERO, entryPoint);
        }

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto inputPort = mBufferGraph[e].Port;
            Value * const itemCount = mInitiallyProcessedItemCount[inputPort]; assert (itemCount);
            mProcessedItemCountAtTerminationPhi[inputPort]->addIncoming(itemCount, entryPoint);
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            const auto streamSet = target(e, mBufferGraph);
            Value * const itemCount = mInitiallyProducedItemCount[streamSet]; assert (itemCount);
            mProducedAtTerminationPhi[port]->addIncoming(itemCount, entryPoint);
        }

        b.SetInsertPoint(np);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readTerminationSignal
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readTerminationSignal(KernelBuilder & b, const unsigned kernelId) const {
    assert (HasTerminationSignal.test(kernelId));
    const auto name = TERMINATION_PREFIX + std::to_string(kernelId);
    auto ref = b.getScalarFieldPtr(name);
    assert (ref.second == b.getSizeTy());
    return b.CreateAlignedLoad(b.getSizeTy(), ref.first, SizeTyABIAlignment, true, name);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeTerminationSignal(KernelBuilder & b, const unsigned kernelId, Value * const signal) const {
    assert (HasTerminationSignal.test(kernelId));
    Value * const ptr = b.getScalarFieldPtr(TERMINATION_PREFIX + std::to_string(kernelId)).first;
    b.CreateAlignedStore(signal, ptr, SizeTyABIAlignment, true);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readCountableItemCountsAfterAbnormalTermination
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readCountableItemCountsAfterAbnormalTermination(KernelBuilder & b) {

    auto isCountableType = [](const Value * const ptr, const Binding & binding) {
        return ptr ? isCountable(binding) : false;
    };

    const auto numOfOutputs = numOfStreamOutputs(mKernelId);
    Vec<Value *> finalProduced(numOfOutputs);
    for (unsigned i = 0; i < numOfOutputs; i++) {
        const StreamSetPort port (PortType::Output, i);
        finalProduced[i] = mProducedItemCount[port];
        if (isCountableType(mReturnedProducedItemCountPtr[port], getOutputBinding(port))) {
            finalProduced[i] = b.CreateAlignedLoad(b.getSizeTy(), mReturnedProducedItemCountPtr[port], SizeTyABIAlignment);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, makeBufferName(mKernelId, port) +
                       "_producedAfterAbnormalTermination = %" PRIu64, finalProduced[i]);
            #endif
        }
    }
    BasicBlock * const exitBlock = b.GetInsertBlock();

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto inputPort = mBufferGraph[e].Port;
        Value * const itemCount = mProcessedItemCount[inputPort]; assert (itemCount);
        mProcessedItemCountAtTerminationPhi[inputPort]->addIncoming(itemCount, exitBlock);
    }

    for (unsigned i = 0; i < numOfOutputs; i++) {
        const StreamSetPort port (PortType::Output, i);
        mProducedAtTerminationPhi[port]->addIncoming(finalProduced[i], exitBlock);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief informInputKernelsOfTermination
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::propagateTerminationSignal(KernelBuilder & b) {

    if (CheckAssertions) {

        bool hasPrincipal = false;
        Value * atLeastOneExhausted = nullptr;
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;

            Value * const closed = isClosed(b, br.Port);
            Value * fullyConsumed = nullptr;
            if (LLVM_UNLIKELY(br.isZeroExtended())) {
                fullyConsumed = closed;
            } else {
                Value * const avail = mLocallyAvailableItems[streamSet];
                Value * const processed = mProcessedItemCountAtTerminationPhi[br.Port]; assert (processed);
                fullyConsumed = b.CreateAnd(closed, b.CreateICmpULE(avail, processed));
            }

            if (LLVM_UNLIKELY(br.isPrincipal())) {
                if (atLeastOneExhausted) {
                    RecursivelyDeleteTriviallyDeadInstructions(atLeastOneExhausted);
                }
                atLeastOneExhausted = fullyConsumed;
                hasPrincipal = true;
                break;
            }
            if (atLeastOneExhausted) {
                atLeastOneExhausted = b.CreateOr(atLeastOneExhausted, fullyConsumed);
            } else {
                atLeastOneExhausted = fullyConsumed;
            }
        }

        if (atLeastOneExhausted) {
            Constant * const completed = getTerminationSignal(b, TerminationSignal::Completed);
            Value * const notTerminatedNormally = b.CreateICmpNE(mTerminatedSignalPhi, completed);
            Value * const expectedTermination = b.CreateOr(notTerminatedNormally, atLeastOneExhausted);
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Kernel %s in partition %" PRId64 " was terminated before exhausting ";
            if (hasPrincipal) {
                out << "its principal input.";
            } else {
                out << "at least one of its inputs.";
            }
            b.CreateAssert(expectedTermination, out.str(),
                mCurrentKernelName, b.getSize(mCurrentPartitionId),
                mKernelName[mCurrentPartitionRoot]);
        }

    }

    // Most kernels terminate after their data sources being exhausted but some may terminate
    // "unexpectedly" when the kernel itself sends a termination signal. If every consumer of
    // a kernel has the ability to terminate in such a fashion, we want to be able to propagate
    // those signals up to the source kernel to terminate it early.

    for (const auto e : make_iterator_range(out_edges(mCurrentPartitionId, mTerminationPropagationGraph))) {
        const auto id = target(e, mTerminationPropagationGraph);
        Value * const signal = b.getScalarFieldPtr(CONSUMER_TERMINATION_COUNT_PREFIX + std::to_string(id)).first;
        b.CreateAtomicFetchAndAdd(b.getSize(1), signal);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief verifyPostInvocationTerminationSignal
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::verifyPostInvocationTerminationSignal(KernelBuilder & b) {

    Value * const isTerminated = b.CreateIsNotNull(mTerminatedAtExitPhi);

    if (mIsPartitionRoot) {
        constexpr auto msg =
            "Partition root %s in partition %" PRId64 " should have been flagged as terminated "
            "after invocation.";
        Value * const valid = b.CreateOr(b.CreateNot(mFinalPartitionSegment), isTerminated);
        b.CreateAssert(valid, msg, mCurrentKernelName, b.getSize(mCurrentPartitionId));
    } else {
        constexpr auto msg =
            "Kernel %s in partition %" PRId64 " should have been flagged as terminated "
            "after partition root %s was terminated.";
        Value * valid = b.CreateICmpEQ(mFinalPartitionSegment, isTerminated);
        if (mKernelCanTerminateEarly) {
            valid = b.CreateOr(valid, isTerminated);
        }
        b.CreateAssert(valid, msg,
            mCurrentKernelName, b.getSize(mCurrentPartitionId),
            mKernelName[mCurrentPartitionRoot]);
    }

}

}
