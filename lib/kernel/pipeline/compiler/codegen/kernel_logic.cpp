#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setActiveKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::setActiveKernel(KernelBuilder & b, const unsigned kernelId, const bool allowThreadLocal, const bool getCommonThreadLocal) {
    assert (kernelId >= FirstKernel && kernelId <= LastKernel);
    mKernelId = kernelId;
    mKernel = getKernel(kernelId);
    mKernelSharedHandle = nullptr;

    if (LLVM_LIKELY(mKernel->isStateful())) {
        Value * handle = b.getScalarFieldPtr(makeKernelName(kernelId)).first;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            PointerType * pty = mKernel->getSharedStateType()->getPointerTo();
            handle = b.CreateAlignedLoad(pty, handle, PtrTyABIAlignment);
        }
        mKernelSharedHandle = handle;
    }
    mKernelThreadLocalHandle = nullptr;
    if (mKernel->hasThreadLocal() && allowThreadLocal) {
        mKernelThreadLocalHandle = getThreadLocalHandlePtr(b, mKernelId);
        if (getCommonThreadLocal) {
            mKernelCommonThreadLocalHandle = getThreadLocalHandlePtr(b, mKernelId, true);
        }
    }


    mCurrentKernelName = mKernelName[mKernelId];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeFullyProcessedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::computeFullyProcessedItemCounts(KernelBuilder & b, Value * const terminated) {

    Constant * const sz_MAX_INT = ConstantInt::getAllOnesValue(b.getSizeTy());

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        Value * processed = nullptr;
        if (mUpdatedProcessedDeferredPhi[port]) {
            processed = mUpdatedProcessedDeferredPhi[port];
        } else {
            processed = mUpdatedProcessedPhi[port];
        }

        const Binding & input = br.Binding;

        if (LLVM_UNLIKELY(input.hasAttribute(AttrId::BlockSize))) {
            // If the input rate has a block size attribute then --- for the purpose of determining how many
            // items have been consumed --- we consider a stream set to be fully processed when an entire
            // stride has been processed.
            Constant * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
            processed = b.CreateAnd(processed, ConstantExpr::getNeg(BLOCK_WIDTH));
        }

        Value * const avail = sz_MAX_INT; // mLocallyAvailableItems[source(e, mBufferGraph)];
        Value * const fullyProcessed = b.CreateSelect(terminated, avail, processed);

        mFullyProcessedItemCount[port] = fullyProcessed;
        if (LLVM_UNLIKELY(CheckAssertions)) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bn.Locality == BufferLocality::ThreadLocal) {
                Value * const produced = mLocallyAvailableItems[streamSet]; assert (produced);
                Value * const fullyConsumed = b.CreateICmpEQ(produced, processed);
                Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);
                Value * const fatalError = b.CreateICmpEQ(mTerminatedAtLoopExitPhi, fatal);
                Value * const valid = b.CreateOr(fullyConsumed, fatalError);
                Constant * const bindingName = b.GetString(input.getName());

                b.CreateAssert(valid,
                                "%s.%s: local available item count (%" PRId64 ") does not match "
                                "its processed item count (%" PRId64 ")",
                                mCurrentKernelName, bindingName,
                                produced, processed);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeFullyProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::computeFullyProducedItemCounts(KernelBuilder & b, Value * const terminated) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        Value * produced = nullptr;
        if (LLVM_UNLIKELY(bn.OutputItemCountId != streamSet)) {
            produced = mLocallyAvailableItems[bn.OutputItemCountId];
        } else {
            if (mUpdatedProducedDeferredPhi[port]) {
                produced = mUpdatedProducedDeferredPhi[port];
            } else {
                produced = mUpdatedProducedPhi[port];
            }

            const Binding & output = br.Binding;
            if (LLVM_UNLIKELY(output.hasAttribute(AttrId::Delayed))) {
                const auto & D = output.findAttribute(AttrId::Delayed);
                produced = b.CreateSaturatingSub(produced, b.getSize(D.amount()));
            }
            if (LLVM_UNLIKELY(output.hasAttribute(AttrId::BlockSize))) {
                Constant * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
                produced = b.CreateAnd(produced, ConstantExpr::getNeg(BLOCK_WIDTH));
            }
            produced = b.CreateSelect(terminated, mUpdatedProducedPhi[port], produced);
        }
        assert (isFromCurrentFunction(b, produced, false));
        mFullyProducedItemCount[port]->addIncoming(produced, mKernelLoopExitPhiCatch);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addLookahead
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::addLookahead(KernelBuilder & b, const BufferPort & inputPort, Value * const itemCount) const {
    if (LLVM_LIKELY(inputPort.LookAhead == 0)) {
        return itemCount;
    }
    Constant * const lookAhead = b.getSize(inputPort.LookAhead);
    return b.CreateAdd(itemCount, lookAhead);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief subtractLookahead
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::subtractLookahead(KernelBuilder & b, const BufferPort & inputPort, Value * const itemCount) {
    if (LLVM_LIKELY(inputPort.LookAhead == 0)) {
        return itemCount;
    }
    Constant * const lookAhead = b.getSize(inputPort.LookAhead);
    Value * const closed = isClosed(b, inputPort.Port);
    Value * const reducedItemCount = b.CreateSaturatingSub(itemCount, lookAhead);
    return b.CreateSelect(closed, itemCount, reducedItemCount);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadLocalHandlePtr
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getThreadLocalHandlePtr(KernelBuilder & b, const unsigned kernelIndex, const bool commonThreadLocal) const {
    const Kernel * const kernel = getKernel(kernelIndex);
    assert ("getThreadLocalHandlePtr should not have been called" && kernel->hasThreadLocal());
    const auto prefix = makeKernelName(kernelIndex);
    Value * handle = nullptr;
    if (LLVM_UNLIKELY(commonThreadLocal)) {
        assert (mCommonThreadLocalHandle);
        handle = getThreadLocalScalarFieldPtr(b, mCommonThreadLocalHandle, prefix + KERNEL_THREAD_LOCAL_SUFFIX).first;
        assert (handle);
    } else {
        handle = getScalarFieldPtr(b, prefix + KERNEL_THREAD_LOCAL_SUFFIX).first;
    }
    if (LLVM_UNLIKELY(isKernelFamilyCall(kernelIndex))) {
        StructType * const localStateTy = kernel->getThreadLocalStateType();
        if (LLVM_UNLIKELY(CheckAssertions)) {
            b.CreateAssert(handle, "null handle load");
        }
        handle = b.CreateAlignedLoad(localStateTy->getPointerTo(), handle, PtrTyABIAlignment);
    }
    assert (handle->getType()->isPointerTy());
    return handle;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasAtLeastOneNonGreedyInput
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasAtLeastOneNonGreedyInput() const {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[e];
        const Binding & binding = bp.Binding;
        if (!binding.getRate().isGreedy()) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasAtLeastOneNonGreedyInput
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasAnyGreedyInput(const unsigned kernelId) const {
    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[e];
        const Binding & binding = bp.Binding;
        if (binding.getRate().isGreedy()) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isDataParallel
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::isDataParallel(const size_t kernel) const {
    #ifdef ALLOW_INTERNALLY_SYNCHRONIZED_KERNELS_TO_BE_DATA_PARALLEL
    return mIsStatelessKernel.test(kernel) || mIsInternallySynchronized.test(kernel);
    #else
    return mIsStatelessKernel.test(kernel);
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasPrincipalInputRate
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasPrincipalInputRate() const {
    unsigned numOfPrincipalInputs = 0;
    unsigned totalNumOfFixedRateInputs = 0;
    bool nonFixedPrincipalAttribute = false;
    for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[e];
        assert (port.Port.Type == PortType::Input);
        if (LLVM_UNLIKELY(port.isPrincipal())) {
            nonFixedPrincipalAttribute |= !port.isFixed();
            ++numOfPrincipalInputs;
            assert (totalNumOfFixedRateInputs == 0);
        }
        if (port.isFixed()) {
            ++totalNumOfFixedRateInputs;
        }
    }
    if (LLVM_UNLIKELY(numOfPrincipalInputs > 1 || nonFixedPrincipalAttribute)) {
        SmallVector<char, 512> tmp;
        raw_svector_ostream msg(tmp);
        msg << "Error: " << mKernel->getName();
        if (nonFixedPrincipalAttribute) {
            msg << " has a Principal attribute assigned to a non-Fixed rate input";
        } else {
            msg << " has more than one input with a Principal attribute";
        }
        report_fatal_error(msg.str());
    }
    assert (totalNumOfFixedRateInputs >= numOfPrincipalInputs);
    return (numOfPrincipalInputs == 1 && numOfPrincipalInputs < totalNumOfFixedRateInputs);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getInputBufferVertex(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBufferVertex(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetBuffer * PipelineCompiler::getInputBuffer(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBuffer(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCompiler::getInputBinding(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBinding(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getOutputBufferVertex(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBufferVertex(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCompiler::getOutputBinding(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBinding(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetBuffer * PipelineCompiler::getOutputBuffer(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBuffer(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCompiler::getBinding(const StreamSetPort port) const {
    return PipelineCommonGraphFunctions::getBinding(mKernelId, port);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReference
 ** ------------------------------------------------------------------------------------------------------------- */
const StreamSetPort PipelineCompiler::getReference(const StreamSetPort port) const {
    return PipelineCommonGraphFunctions::getReference(mKernelId, port);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief reset
 ** ------------------------------------------------------------------------------------------------------------- */
template <typename Vec>
void reset(Vec & vec, const size_t n) {
    vec.resize(n);
    std::memset(vec.data(), 0, n * sizeof(typename Vec::value_type));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearInternalStateForCurrentKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::clearInternalStateForCurrentKernel() {

    // TODO: make it so these are only needed in debug mode for assertion checks?

    mExecuteStridesIndividually = false;
    mCurrentKernelIsStateFree = false;
    mAllowDataParallelExecution = false;
    mHasPrincipalInput = false;

    mHasMoreInput = nullptr;
    mHasZeroExtendedInput = nullptr;
    mStrideStepSize = nullptr;
    mAnyClosed = nullptr;

    mPrincipalFixedRateFactor = nullptr;
    mHasExhaustedClosedInput = nullptr;
    mStrideStepSizeAtLoopEntryPhi = nullptr;
    mKernelInsufficientInput = nullptr;
    mKernelTerminated = nullptr;
    mKernelInitiallyTerminated = nullptr;
    mKernelInitiallyTerminatedExit = nullptr;
    mInitiallyTerminated = nullptr;

    mMaximumNumOfStrides = nullptr;
    mNumOfLinearStridesPhi = nullptr;
    mNumOfLinearStrides = nullptr;
    mFixedRateFactorPhi = nullptr;
    mFinalPartialStrideFixedRateRemainderPhi = nullptr;
    mIsFinalInvocationPhi = nullptr;
    mIsFinalInvocation = nullptr;

    assert (mKernelId >= FirstKernel);
    assert (mKernelId <= LastKernel);

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    mInternalAccessibleInputItems.reset(numOfInputs);
    mInitiallyProcessedItemCount.reset(numOfInputs);
    mInitiallyProcessedDeferredItemCount.reset(numOfInputs);
    mAlreadyProcessedPhi.reset(numOfInputs);
    mAlreadyProcessedDeferredPhi.reset(numOfInputs);
    mIsInputZeroExtended.reset(numOfInputs);
    mInputVirtualBaseAddressPhi.reset(numOfInputs);
    mFirstInputStrideLength.reset(numOfInputs);
    mLinearInputItemsPhi.reset(numOfInputs);
    mReturnedProcessedItemCountPtr.reset(numOfInputs);
    mProcessedItemCountPtr.reset(numOfInputs);
    mProcessedItemCount.reset(numOfInputs);
    mProcessedDeferredItemCountPtr.reset(numOfInputs);
    mProcessedDeferredItemCount.reset(numOfInputs);
    mExhaustedInputPort.reset(numOfInputs);
    mExhaustedInputPortPhi.reset(numOfInputs);
    mCurrentProcessedItemCountPhi.reset(numOfInputs);
    mCurrentProcessedDeferredItemCountPhi.reset(numOfInputs);
    mCurrentLinearInputItems.reset(numOfInputs);
    mUpdatedProcessedPhi.reset(numOfInputs);
    mUpdatedProcessedDeferredPhi.reset(numOfInputs);
    mConsumedItemCountsAtLoopExitPhi.reset(numOfInputs);
    mFullyProcessedItemCount.reset(numOfInputs);

    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);
    mInternalWritableOutputItems.reset(numOfOutputs);
    mAlreadyProducedPhi.reset(numOfOutputs);
    mAlreadyProducedDelayedPhi.reset(numOfOutputs);
    mAlreadyProducedDeferredPhi.reset(numOfOutputs);
    mFirstOutputStrideLength.reset(numOfOutputs);
    mLinearOutputItemsPhi.reset(numOfOutputs);
    mReturnedOutputVirtualBaseAddressPtr.reset(numOfOutputs);
    mReturnedProducedItemCountPtr.reset(numOfOutputs);
    mProducedItemCountPtr.reset(numOfOutputs);
    mProducedItemCount.reset(numOfOutputs);
    mProducedDeferredItemCountPtr.reset(numOfOutputs);
    mProducedDeferredItemCount.reset(numOfOutputs);    
    mCurrentProducedItemCountPhi.reset(numOfOutputs);
    mCurrentProducedDeferredItemCountPhi.reset(numOfOutputs);
    mCurrentLinearOutputItems.reset(numOfOutputs);
    mProducedAtJumpPhi.reset(numOfOutputs);
    mProducedAtTerminationPhi.reset(numOfOutputs);
    mProducedAtTermination.reset(numOfOutputs);
    mUpdatedProducedPhi.reset(numOfOutputs);
    mUpdatedProducedDeferredPhi.reset(numOfOutputs);
    mFullyProducedItemCount.reset(numOfOutputs);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelAssertions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelAssertions(KernelBuilder & b) {
    SmallVector<char, 256> tmp;
    for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
        raw_svector_ostream out(tmp);
        out << kernel << "." << getKernel(kernel)->getName();
        mKernelName[kernel] = b.GetString(out.str());
        tmp.clear();
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernelName
 ** ------------------------------------------------------------------------------------------------------------- */
LLVM_READNONE std::string PipelineCompiler::makeKernelName(const size_t kernelIndex) const {
    std::string tmp;
    raw_string_ostream out(tmp);
    out << kernelIndex;
    #ifdef PRINT_DEBUG_MESSAGES
    out << '.' << getKernel(kernelIndex)->getName();
    #endif
    out.flush();
    return tmp;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeBufferName
 ** ------------------------------------------------------------------------------------------------------------- */
LLVM_READNONE std::string PipelineCompiler::makeBufferName(const size_t kernelIndex, const StreamSetPort port) const {
    std::string tmp;
    raw_string_ostream out(tmp);
    out << kernelIndex;
    #ifdef PRINT_DEBUG_MESSAGES
    out << '.' << getKernel(kernelIndex)->getName()
        << '.' << getBinding(kernelIndex, port).getName();
    #else
    if (port.Type == PortType::Input) {
        out << 'I';
    } else { // if (port.Type == PortType::Output) {
        out << 'O';
    }
    out.write_hex(port.Number);
    #endif
    out.flush();
    return tmp;
}

}
