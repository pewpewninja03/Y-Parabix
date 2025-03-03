#include "../pipeline_compiler.hpp"
#include <kernel/pipeline/optimizationbranch.h>

// TODO: if we have multiple copies of the same type of kernel executing sequentially, we could avoid
// generating an "execution call" for each and instead pass in different handles/item counts. This
// could improve I-Cache utilization.

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief executeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::executeKernel(KernelBuilder & b) {
    #ifndef NDEBUG
    Value * const initialSegNum = mSegNo;
    #endif
    assert (FirstKernel <= mKernelId && mKernelId <= LastKernel);
    clearInternalStateForCurrentKernel();
    checkForPartitionEntry(b);
    assert ("every kernel ought to be marked a partition root if jumping is disabled" && (!mIsIOProcessThread || mIsPartitionRoot));
    mFixedRateLCM = getLCMOfFixedRateInputs(mKernel);
    mKernelIsInternallySynchronized = mIsInternallySynchronized.test(mKernelId);
    mKernelCanTerminateEarly = mKernel->canSetTerminateSignal();
    mIsOptimizationBranch = isa<OptimizationBranch>(mKernel);
    mRecordHistogramData = recordsAnyHistogramData();
    mExecuteStridesIndividually =
        mKernel->hasAttribute(AttrId::ExecuteStridesIndividually)
            || ((mRecordHistogramData || mUsesIllustrator) && !hasAnyGreedyInput(mKernelId));
    mCurrentKernelIsStateFree = mIsStatelessKernel.test(mKernelId);
    mProducesCrossThreadedData = mKernelProducesCrossThreadedData.test(mKernelId);
    mHasPrincipalInputRate = hasPrincipalInputRate();
    #ifndef DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION
    #ifdef ALLOW_INTERNALLY_SYNCHRONIZED_KERNELS_TO_BE_DATA_PARALLEL
    mAllowDataParallelExecution = mCurrentKernelIsStateFree || mKernelIsInternallySynchronized;
    #else
    mAllowDataParallelExecution = mCurrentKernelIsStateFree;
    #endif
    #endif

    bool checkInputChannels = false;
    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[input];
        checkInputChannels |= port.canModifySegmentLength();
        mHasPrincipalInput |= port.isPrincipal();
    }

    bool checkOutputChannels = false;
    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[output];
        if (port.canModifySegmentLength()) {
            checkOutputChannels = true;
            break;
        }
    }

    mMayHaveInsufficientIO = checkInputChannels || checkOutputChannels;

    assert (mNextPartitionEntryPoint);
    assert (mCurrentPartitionId < KernelPartitionId[PipelineOutput]);
    assert (PartitionJumpTargetId[mCurrentPartitionId] > mCurrentPartitionId);

    const auto prefix = makeKernelName(mKernelId);

    // TODO: if a kernel has circular buffers and the produced/consumption rate is not synchronous
    // and the GCD of the stride step length of the producer/consumer is 1 but the stride step length
    // of the consumer is > 1, we may get a scenario in which the partition root needs to check the
    // raw produced item counts rather than the accessible ones to determine the segment length.
    // We coumMayLoopToEntryld bypass this by having a larger overflow region but doing so would cause us to memcpy
    // more data than necessary.

    /// -------------------------------------------------------------------------------------
    /// BASIC BLOCK CONSTRUCTION
    /// -------------------------------------------------------------------------------------

    mKernelLoopEntry = b.CreateBasicBlock(prefix + "_loopEntry", mNextPartitionEntryPoint);
    mKernelCheckOutputSpace = b.CreateBasicBlock(prefix + "_checkOutputSpace", mNextPartitionEntryPoint);
    mKernelLoopCall = b.CreateBasicBlock(prefix + "_executeKernel", mNextPartitionEntryPoint);
    mKernelCompletionCheck = b.CreateBasicBlock(prefix + "_normalCompletionCheck", mNextPartitionEntryPoint);
    if (mMayHaveInsufficientIO) {
        mKernelInsufficientInput = b.CreateBasicBlock(prefix + "_insufficientInput", mNextPartitionEntryPoint);
    }
    mKernelInitiallyTerminated = nullptr;
    mKernelJumpToNextUsefulPartition = nullptr;

    assert (mIsPartitionRoot || !mIsIOProcessThread);

    if (LLVM_UNLIKELY(mIsPartitionRoot || mKernelCanTerminateEarly)) {
        mKernelInitiallyTerminated = b.CreateBasicBlock(prefix + "_initiallyTerminated", mNextPartitionEntryPoint);
        if (LLVM_LIKELY(mIsPartitionRoot && !mIsIOProcessThread && !mUsesNestedSynchronizationVariable)) {
            // if we are actually jumping over any kernels, create the basicblock for the code to perform it.
            const auto jumpId = PartitionJumpTargetId[mCurrentPartitionId];
            assert (jumpId > mCurrentPartitionId);
            if (jumpId != (mCurrentPartitionId + 1) || KernelPartitionId[mKernelId + 1] == mCurrentPartitionId) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream nm(tmp);
                nm << prefix << "_jumpFromPartition_" << mCurrentPartitionId
                   << "_to_" << PartitionJumpTargetId[mCurrentPartitionId];
                mKernelJumpToNextUsefulPartition = b.CreateBasicBlock(nm.str(), mNextPartitionEntryPoint);
            }
        }
    }

    mKernelTerminated = b.CreateBasicBlock(prefix + "_terminated", mNextPartitionEntryPoint);
    mKernelLoopExit = b.CreateBasicBlock(prefix + "_loopExit", mNextPartitionEntryPoint);
    // The phi catch simplifies compilation logic by "forward declaring" the loop exit point.
    // Subsequent optimization phases will collapse it into the correct exit block.
    mKernelLoopExitPhiCatch = b.CreateBasicBlock(prefix + "_kernelExitPhiCatch", mNextPartitionEntryPoint);
    mKernelExit = b.CreateBasicBlock(prefix + "_kernelExit", mNextPartitionEntryPoint);

    /// -------------------------------------------------------------------------------------
    /// KERNEL / PARTITION ENTRY BLOCK
    /// -------------------------------------------------------------------------------------

    checkIfKernelIsAlreadyTerminated(b);
    readProcessedItemCounts(b);
    readProducedItemCounts(b);
    readAvailableItemCounts(b);
    readConsumedItemCounts(b);
    recordUnconsumedItemCounts(b);
    detemineMaximumNumberOfStrides(b);
    remapThreadLocalBufferMemory(b);
    checkForSufficientIO(b);
    mFinalPartialStrideFixedRateRemainderPhi = nullptr;
    if (mIsPartitionRoot || mKernelCanTerminateEarly) {
        b.CreateUnlikelyCondBr(mInitiallyTerminated, mKernelInitiallyTerminated, mKernelLoopEntry);
    } else {
        b.CreateBr(mKernelLoopEntry);
    }
    mKernelLoopStart = b.GetInsertBlock();

    /// -------------------------------------------------------------------------------------
    /// PHI NODE INITIALIZATION
    /// -------------------------------------------------------------------------------------

    // Set up some PHI nodes early to simplify accumulating their incoming values.
    initializeKernelLoopEntryPhis(b);
    initializeKernelCheckOutputSpacePhis(b);
    if (mMayHaveInsufficientIO) {
        initializeKernelInsufficientIOExitPhis(b);
    }
    if (mKernelJumpToNextUsefulPartition) {
        initializeJumpToNextUsefulPartitionPhis(b);
    }
    initializeKernelTerminatedPhis(b);
    initializeKernelLoopExitPhis(b);
    initializeKernelExitPhis(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL LOOP ENTRY
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelLoopEntry);
    checkPropagatedTerminationSignals(b);
    determineNumOfLinearStrides(b);
    mIsFinalInvocation = mIsFinalInvocationPhi;

    // When tracing blocking I/O, test all I/O streams but do not execute the
    // kernel if any stream is insufficient.
    if (LLVM_UNLIKELY(TraceIO && mMayHaveInsufficientIO)) {
        b.CreateUnlikelyCondBr(mBranchToLoopExit, mKernelInsufficientInput, mKernelLoopCall);
    } else {
        b.CreateBr(mKernelLoopCall);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL CALL
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelLoopCall);
    writeKernelCall(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL EXPLICIT TERMINATION CHECK
    /// -------------------------------------------------------------------------------------

    if (mKernelCanTerminateEarly) {

        Value * const aborted = b.CreateIsNotNull(mTerminatedExplicitly);
        BasicBlock * const explicitTermination =
            b.CreateBasicBlock(prefix + "_explicitTermination", mKernelCompletionCheck);
        b.CreateUnlikelyCondBr(aborted, explicitTermination, mKernelCompletionCheck);

        b.SetInsertPoint(explicitTermination);
        // If the kernel explicitly terminates, it must set its processed/produced item counts.
        // Otherwise, the pipeline will update any countable rates, even upon termination.
        readCountableItemCountsAfterAbnormalTermination(b);
        // TODO: We could have a *fixed-rate* source kernel be a partition root but will need to
        // calculate how many items are the stride "remainder" here.
        signalAbnormalTermination(b);
        b.CreateBr(mKernelTerminated);

    } else { // kernel cannot terminate early

        b.CreateBr(mKernelCompletionCheck);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL NORMAL COMPLETION CHECK
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelCompletionCheck);
    normalCompletionCheck(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL TERMINATED
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelTerminated);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + prefix + ".terminated at segment %" PRIu64, mSegNo);
    #endif
    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
        assert (!mIsIOProcessThread);
        acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
    }
    clearUnwrittenOutputData(b);
    splatMultiStepPartialSumValues(b);
    if (LLVM_UNLIKELY(mCurrentKernelIsStateFree)) {
        writeInternalProcessedAndProducedItemCounts(b, true);
    } else if (LLVM_UNLIKELY(mProducesCrossThreadedData)) {
        // When we have a cross-threaded buffer, we need to write out their produced counts
        // prior to writing the termination signal otherwise we risk a consumer assuming
        // a streamset is closed but does not know the correct item count. Rather than
        // potentially slowing down the loop exit by writing an unnecessary value to the
        // global state, we just write it here despite knowing it'll be written again later.
        writeCrossThreadedProducedItemCountAfterTermination(b);
    }
    if (HasTerminationSignal.test(mKernelId)) {
        // Synchronization numbers continually increase as pipelines acquire/release their locks
        // but when we have a cross-threaded buffer, we also store the synchronization number
        // of the terminating segment so that a cross-threaded consumer knows can determine
        // whether every producer of a buffer it's reading has completed the same terminating
        // segment and only then treating the closed state as truthful.
        if (LLVM_UNLIKELY(mProducesCrossThreadedData)) {
            b.setScalarField(CROSS_THREADED_TERMINATION_SEGMENT_NUMBER_PREFIX + std::to_string(mKernelId), mSegNo);
        }
        writeTerminationSignal(b, mKernelId, mTerminatedSignalPhi);
        propagateTerminationSignal(b);
    }
    // We do not release the pre-invocation synchronization lock in the execution phase
    // when a kernel is terminating.
    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
        releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
    }
    updatePhisAfterTermination(b);
    b.CreateBr(mKernelLoopExit);

    /// -------------------------------------------------------------------------------------
    /// KERNEL INSUFFICIENT IO EXIT
    /// -------------------------------------------------------------------------------------

    if (mMayHaveInsufficientIO) {
        writeInsufficientIOExit(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL LOOP EXIT
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelLoopExit);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + prefix + ".loopExit = %" PRIu64, mSegNo);
    #endif
    writeUpdatedItemCounts(b);
    assert (mTerminatedAtLoopExitPhi);
    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Value * const terminated = b.CreateICmpNE(mTerminatedAtLoopExitPhi, unterminated);
    computeFullyProcessedItemCounts(b, terminated);
    computeMinimumConsumedItemCounts(b);
    computeFullyProducedItemCounts(b, terminated);
    if (mIsPartitionRoot && !mIsIOProcessThread) {
        updateNextSlidingWindowSize(b, mMaximumNumOfStridesAtLoopExitPhi, mPotentialSegmentLengthAtLoopExitPhi);
    }
    replacePhiCatchWithCurrentBlock(b, mKernelLoopExitPhiCatch, mKernelExit);
    b.CreateBr(mKernelExit);

    /// -------------------------------------------------------------------------------------
    /// KERNEL INITIALLY TERMINATED EXIT
    /// -------------------------------------------------------------------------------------

    if (mIsPartitionRoot || mKernelCanTerminateEarly) {
        writeInitiallyTerminatedPartitionExit(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL PREPARE FOR PARTITION JUMP
    /// -------------------------------------------------------------------------------------

    if (mKernelJumpToNextUsefulPartition) {
        assert (!mIsIOProcessThread);
        writeJumpToNextPartition(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL EXIT
    /// -------------------------------------------------------------------------------------

    b.SetInsertPoint(mKernelExit);
    recordFinalProducedItemCounts(b);
    writeConsumedItemCounts(b);
    mKernelTerminationSignal[mKernelId] = mTerminatedAtExitPhi; assert (mTerminatedAtExitPhi);
    if (mIsPartitionRoot) {
        recordStridesPerSegment(b, mKernelId, mTotalNumOfStridesAtExitPhi);
    }
    recordProducedItemCountDeltas(b);
    // chain the progress state so that the next one carries on from this one
    assert (isFromCurrentFunction(b, mAnyProgressedAtExitPhi, false));
    mPipelineProgress = mAnyProgressedAtExitPhi;
    if (mIsPartitionRoot) {
        assert (mTotalNumOfStridesAtExitPhi);
        mNumOfPartitionStrides = mTotalNumOfStridesAtExitPhi;
        assert (isFromCurrentFunction(b, mFinalPartitionSegmentAtExitPhi, false));
        mFinalPartitionSegment = mFinalPartitionSegmentAtExitPhi;
        if (LLVM_UNLIKELY(mIsIOProcessThread)) {
            mThreadLocalScalingFactor = nullptr;
        } else {
            // NOTE: we use the partition root's max num of strides as a common scaling factor for
            // thread local buffer memory placement. Since we won't actually know how many strides
            // have been executed until after the root kernel has finished processing, we assume the
            // maximum was used.
            mThreadLocalScalingFactor =
                b.CreateCeilUDivRational(mMaximumNumOfStridesAtExitPhi, MaximumNumOfStrides[mKernelId]);
        }

    }

    if (LLVM_UNLIKELY(CheckAssertions)) {
        verifyPostInvocationTerminationSignal(b);
    }
    assert ("segment number should not have been modified prior to kernel exit" && initialSegNum == mSegNo);
    checkForPartitionExit(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief normalCompletionCheck
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::normalCompletionCheck(KernelBuilder & b) {

    ConstantInt * const i1_TRUE = b.getTrue();

    if (LLVM_LIKELY(!mAllowDataParallelExecution)) {
        mHasMoreInput = hasMoreInput(b);
    }

    assert (mHasMoreInput);

    BasicBlock * const exitBlockAfterLoopAgainTest = b.GetInsertBlock();
    if (mStrideStepSizeAtLoopEntryPhi) {
        mStrideStepSizeAtLoopEntryPhi->addIncoming(mStrideStepSize, exitBlockAfterLoopAgainTest);
    }

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto & br = mBufferGraph[e];
        const auto port = br.Port;
        assert (mProcessedItemCount[port]);
        mAlreadyProcessedPhi[port]->addIncoming(mProcessedItemCount[port], exitBlockAfterLoopAgainTest);
        if (mAlreadyProcessedDeferredPhi[port]) {
            assert (mProcessedDeferredItemCount[port]);
            mAlreadyProcessedDeferredPhi[port]->addIncoming(mProcessedDeferredItemCount[port], exitBlockAfterLoopAgainTest);
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto & br = mBufferGraph[e];
        const auto port = br.Port;
        assert (mProducedItemCount[port]);
        mAlreadyProducedPhi[port]->addIncoming(mProducedItemCount[port], exitBlockAfterLoopAgainTest);
        if (mAlreadyProducedDeferredPhi[port]) {
            assert (mProducedDeferredItemCount[port]);
            mAlreadyProducedDeferredPhi[port]->addIncoming(mProducedDeferredItemCount[port], exitBlockAfterLoopAgainTest);
        }
    }

    mAlreadyProgressedPhi->addIncoming(i1_TRUE, exitBlockAfterLoopAgainTest);
    mExecutedAtLeastOnceAtLoopEntryPhi->addIncoming(i1_TRUE, exitBlockAfterLoopAgainTest);
    mCurrentNumOfStridesAtLoopEntryPhi->addIncoming(mUpdatedNumOfStrides, exitBlockAfterLoopAgainTest);

    if (LLVM_UNLIKELY(mIsOptimizationBranch)) {
        assert (mOptimizationBranchSelectedBranch);
        mOptimizationBranchPriorScanStatePhi->addIncoming(mOptimizationBranchSelectedBranch, exitBlockAfterLoopAgainTest);
    }

    const auto prefix = makeKernelName(mKernelId);
    BasicBlock * const isFinalCheck = b.CreateBasicBlock(prefix + "_isFinalCheck", mKernelTerminated);
    b.CreateUnlikelyCondBr(mHasMoreInput, mKernelLoopEntry, isFinalCheck);

    b.SetInsertPoint(isFinalCheck);
    Value * terminationSignal = nullptr;
    if (LLVM_UNLIKELY(mKernel->hasAttribute(AttrId::MustExplicitlyTerminate))) {
        if (mIsPartitionRoot) {
            terminationSignal = getTerminationSignal(b, TerminationSignal::None);
        } else {
            const auto root = getTerminationSignalIndex(mKernelId);
            assert (KernelPartitionId[root] == mCurrentPartitionId);
            terminationSignal = mKernelTerminationSignal[root];
        }
    } else {
        terminationSignal = mIsFinalInvocationPhi; assert (terminationSignal);
        if (!mIsPartitionRoot) {
            const auto root = FirstKernelInPartition[mCurrentPartitionId];
            assert (KernelPartitionId[root] == mCurrentPartitionId);
            Value * const rootSignal = mKernelTerminationSignal[root];
            Value * const isFinal = b.CreateIsNotNull(terminationSignal);
            terminationSignal = b.CreateSelect(isFinal, terminationSignal, rootSignal);
        }
    }
    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
        assert (!mIsIOProcessThread);
        acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
    }
    BasicBlock * const exitBlock = b.GetInsertBlock();
    // update KernelTerminated phi nodes
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto inputPort = mBufferGraph[e].Port;
        const auto streamSet = source(e, mBufferGraph);
        Value * const itemCount = mLocallyAvailableItems[streamSet]; assert (itemCount);
        mProcessedItemCountAtTerminationPhi[inputPort]->addIncoming(itemCount, exitBlock);
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        assert (mProducedItemCount[port]);
        mProducedAtTerminationPhi[port]->addIncoming(mProducedItemCount[port], exitBlock);
    }
    assert (terminationSignal->getType() == mTerminatedSignalPhi->getType());
    mTerminatedSignalPhi->addIncoming(terminationSignal, exitBlock);
    mCurrentNumOfStridesAtTerminationPhi->addIncoming(mUpdatedNumOfStrides, exitBlock);
    Value * const isFinal = b.CreateIsNotNull(terminationSignal);
    if (mIsPartitionRoot) {
        assert (mUpdatedNumOfStrides);
        Value * const updatedNumOfStrides = b.CreateMulRational(mUpdatedNumOfStrides, mPartitionStrideRateScalingFactor);
        mTotalNumOfStridesAtLoopExitPhi->addIncoming(updatedNumOfStrides, exitBlock);
        mPotentialSegmentLengthAtTerminationPhi->addIncoming(mPotentialSegmentLength, exitBlock);
        mFinalPartialStrideFixedRateRemainderAtTerminationPhi->addIncoming(mFinalPartialStrideFixedRateRemainderPhi, exitBlock);
        mMaximumNumOfStridesAtLoopExitPhi->addIncoming(mMaximumNumOfStrides, exitBlock);
        mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b.getFalse(), exitBlock);
        mPotentialSegmentLengthAtLoopExitPhi->addIncoming(mPotentialSegmentLength, exitBlock);
    }
    b.CreateUnlikelyCondBr(isFinal, mKernelTerminated, mKernelLoopExit);
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        mUpdatedProcessedPhi[port]->addIncoming(mProcessedItemCount[port], exitBlock);
        if (mUpdatedProcessedDeferredPhi[port]) {
            assert (mProcessedDeferredItemCount[port]);
            mUpdatedProcessedDeferredPhi[port]->addIncoming(mProcessedDeferredItemCount[port], exitBlock);
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        mUpdatedProducedPhi[port]->addIncoming(mProducedItemCount[port], exitBlock);
        if (mUpdatedProducedDeferredPhi[port]) {
            assert (mProducedDeferredItemCount[port]);
            mUpdatedProducedDeferredPhi[port]->addIncoming(mProducedDeferredItemCount[port], exitBlock);
        }
    }
    mTerminatedAtLoopExitPhi->addIncoming(terminationSignal, exitBlock);
    mAnyProgressedAtLoopExitPhi->addIncoming(i1_TRUE, exitBlock);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelLoopEntryPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelLoopEntryPhis(KernelBuilder & b) {
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const boolTy = b.getInt1Ty();

    assert ("kernel loop start must be created before initializing loop entry phi nodes" && mKernelLoopStart);

    b.SetInsertPoint(mKernelLoopEntry);

    if (StrideStepLength[mKernelId] > 1 && mIsPartitionRoot) {
        ConstantInt * strideStep = b.getSize(StrideStepLength[mKernelId]);
        mStrideStepSizeAtLoopEntryPhi =
            b.CreatePHI(sizeTy, 2, makeKernelName(mKernelId) + "_strideStepSizePhi");
        mStrideStepSizeAtLoopEntryPhi->addIncoming(strideStep, mKernelLoopStart);
        mStrideStepSize = mStrideStepSizeAtLoopEntryPhi;
    }

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto prefix = makeBufferName(mKernelId, port);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_alreadyProcessed");
        mAlreadyProcessedPhi[port] = phi;
        mCurrentProcessedItemCountPhi[port] = phi;
        assert (mInitiallyProcessedItemCount[port]);
        phi->addIncoming(mInitiallyProcessedItemCount[port], mKernelLoopStart);
        Value * const value = mInitiallyProcessedDeferredItemCount[port];
        if (value) {
            PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_alreadyProcessedDeferred");
            assert (phi);
            phi->addIncoming(value, mKernelLoopStart);
            mAlreadyProcessedDeferredPhi[port] = phi;
            mCurrentProcessedDeferredItemCountPhi[port] = phi;
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto prefix = makeBufferName(mKernelId, port);
        const auto streamSet = target(e, mBufferGraph);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_alreadyProduced");
        mAlreadyProducedPhi[port] = phi;
        mCurrentProducedItemCountPhi[port] = phi;
        assert (mInitiallyProducedItemCount[streamSet]);
        phi->addIncoming(mInitiallyProducedItemCount[streamSet], mKernelLoopStart);
        if (mInitiallyProducedDeferredItemCount[streamSet]) {
            PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_alreadyProducedDeferred");
            mAlreadyProducedDeferredPhi[port] = phi;
            mCurrentProducedDeferredItemCountPhi[port] = phi;
            phi->addIncoming(mInitiallyProducedDeferredItemCount[streamSet], mKernelLoopStart);
        }
    }
    const auto prefix = makeKernelName(mKernelId);
    mAlreadyProgressedPhi = b.CreatePHI(boolTy, 2, prefix + "_madeProgress");
    assert (mPipelineProgress);
    mAlreadyProgressedPhi->addIncoming(mPipelineProgress, mKernelLoopStart);

    // Since we may loop and call the kernel again, we want to mark that we've progressed
    // if we execute any kernel even if we could not complete a full segment.
    mExecutedAtLeastOnceAtLoopEntryPhi = b.CreatePHI(boolTy, 2, prefix + "_executedAtLeastOnce");
    mExecutedAtLeastOnceAtLoopEntryPhi->addIncoming(b.getFalse(), mKernelLoopStart);
    mCurrentNumOfStridesAtLoopEntryPhi = b.CreatePHI(sizeTy, 2, prefix + "_currentNumOfStrides");
    mCurrentNumOfStridesAtLoopEntryPhi->addIncoming(b.getSize(0), mKernelLoopStart);

    if (mIsOptimizationBranch) {
        mOptimizationBranchPriorScanStatePhi = b.CreatePHI(boolTy, 2, prefix + "_optBrScanState");
        mOptimizationBranchPriorScanStatePhi->addIncoming(b.getFalse(), mKernelLoopStart);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelCheckOutputSpacePhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelCheckOutputSpacePhis(KernelBuilder & b) {
    b.SetInsertPoint(mKernelCheckOutputSpace);
    IntegerType * const sizeTy = b.getSizeTy();
    const auto makeExhaustedInputPhi =
        mKernelIsInternallySynchronized && StrideStepLength[mKernelId] > 1 && mIsPartitionRoot;
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto inputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_linearlyAccessible");
        mLinearInputItemsPhi[inputPort] = phi;
        mCurrentLinearInputItems[inputPort] = phi;
        Type * const bufferTy = getInputBuffer(inputPort)->getPointerType();
        mInputVirtualBaseAddressPhi[inputPort] = b.CreatePHI(bufferTy, 2, prefix + "_baseAddress");
        if (LLVM_UNLIKELY(makeExhaustedInputPhi)) {
            mExhaustedInputPortPhi[inputPort] = b.CreatePHI(b.getInt1Ty(), 2, prefix + "_isExhaustedPhi");
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto outputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_linearlyWritable");
        mLinearOutputItemsPhi[outputPort] = phi;
        mCurrentLinearOutputItems[outputPort] = phi;
    }
    const auto prefix = makeKernelName(mKernelId);
    mNumOfLinearStridesPhi = b.CreatePHI(sizeTy, 2, prefix + "_numOfLinearStridesPhi");
    if (LLVM_LIKELY(mKernel->hasFixedRateIO())) {
        mFixedRateFactorPhi = b.CreatePHI(sizeTy, 2, prefix + "_fixedRateFactorPhi");
    }
    mCurrentFixedRateFactor = mFixedRateFactorPhi;
    mIsFinalInvocationPhi = b.CreatePHI(sizeTy, 2, prefix + "_isFinalPhi");
    if (mIsPartitionRoot) {
        mFinalPartialStrideFixedRateRemainderPhi = b.CreatePHI(sizeTy, 2, prefix + "_partialPartitionStridesPhi");
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelTerminatedPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelTerminatedPhis(KernelBuilder & b) {
    b.SetInsertPoint(mKernelTerminated);
    Type * const sizeTy = b.getSizeTy();
    const auto prefix = makeKernelName(mKernelId);
    mTerminatedSignalPhi = b.CreatePHI(sizeTy, 2, prefix + "_terminatedSignal");
    mCurrentNumOfStridesAtTerminationPhi = b.CreatePHI(sizeTy, 2, prefix + "_currentNumOfStridesAtTermination");
    if (mIsPartitionRoot) {
        mPotentialSegmentLengthAtTerminationPhi =
            b.CreatePHI(sizeTy, 2, prefix + "_potentialSegmentLengthAtTermination");
        mFinalPartialStrideFixedRateRemainderAtTerminationPhi =
            b.CreatePHI(sizeTy, 2, prefix + "_partialPartitionStridesAtTerminationPhi");
    }
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto inputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_finalProcessed");
        mProcessedItemCountAtTerminationPhi[inputPort] = phi;
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto outputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);
        PHINode * const phi = b.CreatePHI(sizeTy, 2, prefix + "_finalProduced");
        mProducedAtTerminationPhi[outputPort] = phi;
        mProducedAtTermination[outputPort] = phi;
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelTerminatedPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeJumpToNextUsefulPartitionPhis(KernelBuilder & b) {
    assert (mKernelJumpToNextUsefulPartition);
    b.SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const sizeTy = b.getSizeTy();
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mProducedAtJumpPhi[port] = b.CreatePHI(sizeTy, 2, prefix + "_producedAtJumpPhi");
    }
    mMaximumNumOfStridesAtJumpPhi = b.CreatePHI(sizeTy, 2, prefix + "_maxNumOfStridesAtJumpPhi");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelInsufficientIOExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelInsufficientIOExitPhis(KernelBuilder & b) {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelLoopExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelLoopExitPhis(KernelBuilder & b) {
    b.SetInsertPoint(mKernelLoopExit);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const boolTy = b.getInt1Ty();
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mUpdatedProcessedPhi[port] = b.CreatePHI(sizeTy, 2, prefix + "_updatedProcessedAtLoopExit");
        if (mAlreadyProcessedDeferredPhi[port]) {
            mUpdatedProcessedDeferredPhi[port] = b.CreatePHI(sizeTy, 2, prefix + "_updatedProcessedDeferredAtLoopExit");
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mUpdatedProducedPhi[port] = b.CreatePHI(sizeTy, 2, prefix + "_updatedProducedAtLoopExit");
        if (mAlreadyProducedDeferredPhi[port]) {
            mUpdatedProducedDeferredPhi[port] = b.CreatePHI(sizeTy, 2, prefix + "_updatedProcessedDeferredAtLoopExit");
        }
    }
    mTerminatedAtLoopExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_terminatedAtLoopExit");
    mAnyProgressedAtLoopExitPhi = b.CreatePHI(boolTy, 2, prefix + "_anyProgressAtLoopExit");
    if (mIsPartitionRoot) {
        mTotalNumOfStridesAtLoopExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_totalNumOfStridesAtLoopExit");
        mFinalPartitionSegmentAtLoopExitPhi = b.CreatePHI(boolTy, 2, prefix + "_finalPartitionSegmentAtLoopExitPhi");
        mMaximumNumOfStridesAtLoopExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_maxNumOfStridesAtLoopExit");
        mPotentialSegmentLengthAtLoopExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_potentialSegmentLengthAtLoopExit");
    } else {
        mTotalNumOfStridesAtLoopExitPhi = nullptr;
        mFinalPartitionSegmentAtLoopExitPhi = nullptr;
        mMaximumNumOfStridesAtLoopExitPhi = nullptr;
        mPotentialSegmentLengthAtLoopExitPhi = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInsufficientIOExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInsufficientIOExit(KernelBuilder & b) {

    // A partition root will always have an insufficient I/O check since they control how many strides the
    // other kernels in the partition will execute. If a kernel has non-linear I/O, however, we need to test
    // whether we've finished executing.

    b.SetInsertPoint(mKernelInsufficientInput);

    if (LLVM_UNLIKELY(CheckAssertions && mAllowDataParallelExecution)) {
        b.CreateAssert(b.CreateNot(mExecutedAtLeastOnceAtLoopEntryPhi),
                        "%s: is a data-parallel kernel with an invalid loop again check",
                        mCurrentKernelName);
    }

    Value * currentNumOfStrides = nullptr;
    assert (mCurrentNumOfStridesAtLoopEntryPhi);
    currentNumOfStrides = b.CreateMulRational(mCurrentNumOfStridesAtLoopEntryPhi, mPartitionStrideRateScalingFactor);

    bool hasBranchToLoopExit = false;

    if (mKernelJumpToNextUsefulPartition) {
        assert (mIsPartitionRoot);
        mMaximumNumOfStridesAtJumpPhi->addIncoming(mMaximumNumOfStrides, b.GetInsertBlock());
        // TODO: check whether we need to release/acquire the pre/post locks here too
        b.CreateLikelyCondBr(mExecutedAtLeastOnceAtLoopEntryPhi, mKernelLoopExit, mKernelJumpToNextUsefulPartition);
        hasBranchToLoopExit = true;
    } else {
        // if this is not a partition root, it is not responsible for determining
        // whether the partition is out of input
        hasBranchToLoopExit = true;
        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            assert (!mIsIOProcessThread);
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        }
        b.CreateBr(mKernelLoopExit);
    }


    BasicBlock * const exitBlock = b.GetInsertBlock();

    if (hasBranchToLoopExit && mIsPartitionRoot) {
        mMaximumNumOfStridesAtLoopExitPhi->addIncoming(mMaximumNumOfStrides, exitBlock);
        mPotentialSegmentLengthAtLoopExitPhi->addIncoming(b.getSize(0), exitBlock);
    }
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto & br = mBufferGraph[e];
        const auto port = br.Port;
        mUpdatedProcessedPhi[port]->addIncoming(mAlreadyProcessedPhi[port], exitBlock);
        if (mAlreadyProcessedDeferredPhi[port]) {
            mUpdatedProcessedDeferredPhi[port]->addIncoming(mAlreadyProcessedDeferredPhi[port], exitBlock);
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto & br = mBufferGraph[e];
        const auto port = br.Port;
        mUpdatedProducedPhi[port]->addIncoming(mAlreadyProducedPhi[port], exitBlock);
        if (mAlreadyProducedDeferredPhi[port]) {
            mUpdatedProducedDeferredPhi[port]->addIncoming(mAlreadyProducedDeferredPhi[port], exitBlock);
        }
    }

    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b.getFalse(), exitBlock);
        mTotalNumOfStridesAtLoopExitPhi->addIncoming(currentNumOfStrides, exitBlock);
    }

    assert (isFromCurrentFunction(b, mAlreadyProgressedPhi, false));
    mAnyProgressedAtLoopExitPhi->addIncoming(mAlreadyProgressedPhi, exitBlock);
    assert (mInitialTerminationSignal);
    mTerminatedAtLoopExitPhi->addIncoming(mInitialTerminationSignal, exitBlock);

    if (mKernelJumpToNextUsefulPartition) {
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto & br = mBufferGraph[e];
            if (LLVM_UNLIKELY(br.isRelative())) {
                continue;
            }
            const auto port = br.Port;
            Value * produced = nullptr;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                produced = mAlreadyProducedDeferredPhi[port];
            } else {
                produced = mAlreadyProducedPhi[port];
            }
            assert (isFromCurrentFunction(b, produced, false));
            mProducedAtJumpPhi[port]->addIncoming(produced, exitBlock);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelExitPhis(KernelBuilder & b) {
    b.SetInsertPoint(mKernelExit);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const boolTy = b.getInt1Ty();

    mTerminatedAtExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_terminatedAtKernelExit");
    assert (mTerminatedAtLoopExitPhi);
    mTerminatedAtExitPhi->addIncoming(mTerminatedAtLoopExitPhi, mKernelLoopExitPhiCatch);
    mTotalNumOfStridesAtExitPhi = nullptr;
    if (mIsPartitionRoot) {
        assert (mTotalNumOfStridesAtLoopExitPhi);
        mTotalNumOfStridesAtExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_totalNumOfStridesAtExit");
        mTotalNumOfStridesAtExitPhi->addIncoming(mTotalNumOfStridesAtLoopExitPhi, mKernelLoopExitPhiCatch);
        mMaximumNumOfStridesAtExitPhi = b.CreatePHI(sizeTy, 2, prefix + "_maxNumOfStridesAtExit");
        mMaximumNumOfStridesAtExitPhi->addIncoming(mMaximumNumOfStridesAtLoopExitPhi, mKernelLoopExitPhiCatch);
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        PHINode * const fullyProduced = b.CreatePHI(sizeTy, 2, prefix + "_fullyProducedAtKernelExit");
        mFullyProducedItemCount[port] = fullyProduced;
    }

    PHINode * const progress = b.CreatePHI(boolTy, 2, prefix + "_anyProgressAtKernelExit");
    assert (isFromCurrentFunction(b, mAnyProgressedAtLoopExitPhi, false));
    progress->addIncoming(mAnyProgressedAtLoopExitPhi, mKernelLoopExitPhiCatch);
    mAnyProgressedAtExitPhi = progress;

    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtExitPhi = b.CreatePHI(boolTy, 2, prefix + "_anyProgressAtKernelExit");
        assert (isFromCurrentFunction(b, mFinalPartitionSegmentAtLoopExitPhi, false));
        mFinalPartitionSegmentAtExitPhi->addIncoming(mFinalPartitionSegmentAtLoopExitPhi, mKernelLoopExitPhiCatch);
    }




}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateKernelExitPhisAfterInitiallyTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateKernelExitPhisAfterInitiallyTerminated(KernelBuilder & b) {
    Constant * const completed = getTerminationSignal(b, TerminationSignal::Completed);
    mTerminatedAtExitPhi->addIncoming(completed, mKernelInitiallyTerminatedExit);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferPort & br = mBufferGraph[e];
        Value * produced = nullptr;
        if (LLVM_UNLIKELY(br.isDeferred())) {
            produced = mInitiallyProducedDeferredItemCount[streamSet];
        } else {
            produced = mInitiallyProducedItemCount[streamSet];
        }
        const auto port = br.Port;
        assert (isFromCurrentFunction(b, produced, false));
        if (mProducedAtJumpPhi[port]) {
            mProducedAtJumpPhi[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
        }
        mFullyProducedItemCount[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
    }

    mAnyProgressedAtExitPhi->addIncoming(mPipelineProgress, mKernelInitiallyTerminatedExit);
    if (mIsPartitionRoot) {
        Constant * const ZERO = b.getSize(0);
        mTotalNumOfStridesAtExitPhi->addIncoming(ZERO, mKernelInitiallyTerminatedExit);
        mMaximumNumOfStridesAtExitPhi->addIncoming(ZERO, mKernelInitiallyTerminatedExit);
        mFinalPartitionSegmentAtExitPhi->addIncoming(b.getTrue(), mKernelInitiallyTerminatedExit);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updatePhiCountAfterTermination
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updatePhisAfterTermination(KernelBuilder & b) {
    BasicBlock * const exitBlock = b.GetInsertBlock();
    mTerminatedAtLoopExitPhi->addIncoming(mTerminatedSignalPhi, exitBlock);
    mAnyProgressedAtLoopExitPhi->addIncoming(b.getTrue(), exitBlock);
    if (mIsPartitionRoot) {
        Value * finalNumOfStrides = mCurrentNumOfStridesAtTerminationPhi;
        if (mFinalPartialStrideFixedRateRemainderAtTerminationPhi) {
            const Rational fixedRateFactor = mFixedRateLCM * Rational{mKernel->getStride()};
            Value * fixedRateItems = b.CreateMulRational(finalNumOfStrides, fixedRateFactor);
            fixedRateItems = b.CreateAdd(fixedRateItems, mFinalPartialStrideFixedRateRemainderAtTerminationPhi);
            finalNumOfStrides = b.CreateMulRational(fixedRateItems, mPartitionStrideRateScalingFactor / fixedRateFactor);
        } else {
            finalNumOfStrides = b.CreateMulRational(finalNumOfStrides, mPartitionStrideRateScalingFactor);
        }
        mTotalNumOfStridesAtLoopExitPhi->addIncoming(finalNumOfStrides, exitBlock);
        mMaximumNumOfStridesAtLoopExitPhi->addIncoming(mMaximumNumOfStrides, exitBlock);
        mPotentialSegmentLengthAtLoopExitPhi->addIncoming(mPotentialSegmentLengthAtTerminationPhi, exitBlock);
        mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b.getTrue(), exitBlock);
    }
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto streamSet = source(e, mBufferGraph);
        Value * const totalCount = mLocallyAvailableItems[streamSet];
        mUpdatedProcessedPhi[port]->addIncoming(totalCount, exitBlock);
        if (mUpdatedProcessedDeferredPhi[port]) {
            mUpdatedProcessedDeferredPhi[port]->addIncoming(totalCount, exitBlock);
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        Value * const produced = mProducedAtTermination[port];
        mUpdatedProducedPhi[port]->addIncoming(produced, exitBlock);
        if (mUpdatedProducedDeferredPhi[port]) {
            mUpdatedProducedDeferredPhi[port]->addIncoming(produced, exitBlock);
        }
    }
}


}
