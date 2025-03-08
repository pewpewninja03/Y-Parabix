#include "../pipeline_compiler.hpp"

// Suppose T1 and T2 are two pipeline threads where all segment processing
// of kernel Ki in T1 logically happens before Ki in T2.

// Any stateless kernel (or kernel marked as internally synchronized) with
// countable input rates that is not a source, sink or side-effecting can
// be executed in parallel once we've calculated the "future" item count
// position(s). However, T2 may finish before T1 and a Kj>i in T2 could
// attempt to consume unfinished data from T1. So we must ensure that T1
// is always completely finished before T2 may execute Kj.

// For any such kernel, we require two counters. The first marks that T1
// has computed T2's initial processed item counts. The second informs T2
// when T1 has finished writing to its output streams. T2 may begin p
// rocessing once it acquires the first lock but cannot write its output
// until acquiring the second.

// If each stride of output of Ki cannot be guaranteed to be written on
// a cache-aligned boundary, regardless of input state, a temporary output
// buffer is required. After acquiring the second lock, the data
// must be copied to the shared stream set. To minimize redundant copies,
// if at the point of executing Ki,
// we require an additional lock that indicates whether some kernel "owns"
// the actual stream set.

// Even though T1 and T2 process a segment per call, a segment may require
// several iterations (due to buffer contraints, final stride processing,
// etc.) Thus to execute a stateful internally synchronized kernel, we must
// hold both buffer locks until reaching the last partial segment.

// TODO: Fix cycle counter and serialize option for nested pipelines

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief obtainCurrentSegmentNumber
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::obtainCurrentSegmentNumber(KernelBuilder & b, BasicBlock * const entryBlock) {
    if (mIsNestedPipeline) {
        assert (!mIsIOProcessThread);
        assert (mSegNo == mExternalSegNo && mSegNo);
    } else if ((mUseDynamicMultithreading || UseJumpGuidedSynchronization) && !mIsIOProcessThread) {
        Value * const segNoPtr = b.getScalarFieldPtr(NEXT_LOGICAL_SEGMENT_NUMBER).first;
        // NOTE: this must be atomic or the pipeline will deadlock when some thread
        // fetches a number before the prior one to fetch the same number updates it.
        mSegNo = b.CreateAtomicFetchAndAdd(b.getSize(1), segNoPtr);
    } else {
        Value * initialSegNo = mSegNo;
        PHINode * const segNo = b.CreatePHI(initialSegNo->getType(), 2, "segNo");
        segNo->addIncoming(initialSegNo, entryBlock);
        mSegNo = segNo;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief incrementCurrentSegNo
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::incrementCurrentSegNo(KernelBuilder & b, BasicBlock * const exitBlock) {
    if (mIsNestedPipeline) {
        return;
    }
    if ((mUseDynamicMultithreading || UseJumpGuidedSynchronization) && !mIsIOProcessThread) {
        return;
    }
    Value * segNo = mSegNo;  assert (mSegNo);
    Value * const nextSegNo = b.CreateAdd(segNo, mNumOfFixedThreads); assert (mNumOfFixedThreads);
    cast<PHINode>(segNo)->addIncoming(nextSegNo, exitBlock);
}

namespace  {

LLVM_READNONE std::string __getSyncLockNameString(const unsigned type) {
    switch (type) {
        case SYNC_LOCK_PRE_INVOCATION: return "pre";
        case SYNC_LOCK_POST_INVOCATION: return "post";
        case SYNC_LOCK_FULL: return "full";
        default: llvm_unreachable("unknown sync lock?");
    }
    return nullptr;
}

LLVM_READNONE Constant * __getSyncLockName(KernelBuilder & b, const unsigned type) {
    switch (type) {
        case SYNC_LOCK_PRE_INVOCATION: return b.GetString("pre-invocation ");
        case SYNC_LOCK_POST_INVOCATION: return b.GetString("post-invocation ");
        case SYNC_LOCK_FULL: return b.GetString("");
        default: llvm_unreachable("unknown sync lock?");
    }
    return nullptr;
}

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief acquireSynchronizationLock
 *
 * Before the segment is processed, this loads the segment number of the kernel state and ensures the previous
 * segment is complete (by checking that the acquired segment number is equal to the desired segment number).
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::acquireSynchronizationLock(KernelBuilder & b, const unsigned kernelId, const unsigned type, Value * const segNo) {
    if (isMultithreaded()) {
        // TODO: make this an function?
        const auto prefix = makeKernelName(kernelId) + ":" + __getSyncLockNameString(type);
        const auto serialize = codegen::DebugOptionIsSet(codegen::SerializeThreads);
        const unsigned waitingOnIdx = serialize ? LastKernel : kernelId;
        Value * const waitingOnPtr = getSynchronizationLockPtrForKernel(b, waitingOnIdx, type);
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + ": waiting for %ssegment number %" PRIu64 ", initially %" PRIu64,
                   __getSyncLockName(b, type), segNo, b.CreateAtomicLoadAcquire(b.getSizeTy(), waitingOnPtr));
        #endif
        assert (b.GetInsertBlock());
        BasicBlock * const nextNode = b.GetInsertBlock()->getNextNode();
        BasicBlock * const acquire = b.CreateBasicBlock(prefix + "_LSN_acquire", nextNode);
        BasicBlock * const acquired = b.CreateBasicBlock(prefix + "_LSN_acquired", nextNode);
        b.CreateBr(acquire);

        b.SetInsertPoint(acquire);
        Value * const currentSegNo = b.CreateAtomicLoadAcquire(b.getSizeTy(), waitingOnPtr);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const pendingOrReady = b.CreateICmpULE(currentSegNo, segNo);
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "%s: acquired %ssegment number is %" PRIu64 " "
                   "but was expected to be no more than %" PRIu64;
            b.CreateAssert(pendingOrReady, out.str(), mKernelName[kernelId], __getSyncLockName(b, type),
                            currentSegNo, segNo);
        }
        Value * const ready = b.CreateICmpEQ(segNo, currentSegNo);
        b.CreateLikelyCondBr(ready, acquired, acquire);

        b.SetInsertPoint(acquired);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "# " + prefix + " acquired %ssegment number %" PRIu64, __getSyncLockName(b, type), segNo);
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief acquireSynchronizationLockWithTimingInstrumentation
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::acquireSynchronizationLockWithTimingInstrumentation(KernelBuilder & b, const unsigned kernelId, const unsigned lockType, Value * const segNo) {
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        startPAPIMeasurement(b, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
    }
    #endif
    if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
        startCycleCounter(b, CycleCounter::KERNEL_SYNCHRONIZATION);
    }
    acquireSynchronizationLock(b, kernelId, lockType, segNo);
    if (LLVM_UNLIKELY(EnableCycleCounter || mUseDynamicMultithreading)) {
        updateCycleCounter(b, kernelId, CycleCounter::KERNEL_SYNCHRONIZATION);
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        accumPAPIMeasurementWithoutReset(b, kernelId, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseCurrentSegment
 *
 * After executing the kernel, the segment number must be incremented to release the kernel for the next thread.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseSynchronizationLock(KernelBuilder & b, const unsigned kernelId, const unsigned type, Value * const segNo) {
    if (isMultithreaded() || TraceProducedItemCounts || TraceUnconsumedItemCounts || TraceIO) {
        Value * const waitingOnPtr = getSynchronizationLockPtrForKernel(b, kernelId, type);
        Value * const nextSegNo = b.CreateAdd(segNo, b.getSize(1));
        if (LLVM_UNLIKELY(CheckAssertions)) {
            DataLayout DL(b.getModule());
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(13, 0, 0)
            llvm::MaybeAlign align = Align(DL.getTypeStoreSize(nextSegNo->getType()));
#endif
            Value * const updated = b.CreateAtomicCmpXchg(waitingOnPtr, segNo, nextSegNo,
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(13, 0, 0)
                                                           *align,
#endif
                                                           AtomicOrdering::Release, AtomicOrdering::Acquire);
            Value * const observed = b.CreateExtractValue(updated, { 0 });
            Value * const success = b.CreateExtractValue(updated, { 1 });
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "%s: released %ssegment number is %" PRIu64
                   " but was expected to be %" PRIu64;
            b.CreateAssert(success, out.str(), mKernelName[kernelId], __getSyncLockName(b, type), observed, segNo);
        } else {
            b.CreateAtomicStoreRelease(nextSegNo, waitingOnPtr);
        }
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeKernelName(kernelId) + ":" + __getSyncLockNameString(type);
        debugPrint(b, prefix + ": released %ssegment number %" PRIu64, __getSyncLockName(b, type), segNo);
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief waitUntilCurrentSegmentNumberIsLessThan
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::waitUntilCurrentSegmentNumberIsLessThan(KernelBuilder & b, const unsigned kernelId, Value * const windowLength) {

    // TODO: this should affect dynamic multithreading sync cost

    assert (b.GetInsertBlock());
    BasicBlock * const nextNode = b.GetInsertBlock()->getNextNode();
    const auto lockType = isDataParallel(kernelId) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
    const auto prefix = makeKernelName(kernelId) + ":" + __getSyncLockNameString(lockType);
    BasicBlock * const segmentCheckLoop = b.CreateBasicBlock(prefix + "_crossTheadWaitLoop", nextNode);
    BasicBlock * const segmentCheckExit = b.CreateBasicBlock(prefix + "_crossTheadWaitExit", nextNode);
    Value * const syncLockPtr = getSynchronizationLockPtrForKernel(b, kernelId, lockType);
    Value * terminatedPtr = nullptr;
    if (mIsIOProcessThread) {
        std::tie(terminatedPtr, std::ignore) = b.getScalarFieldPtr(COMPUTE_THREAD_TERMINATION_STATE);
    }
    b.CreateBr(segmentCheckLoop);

    b.SetInsertPoint(segmentCheckLoop);
    Value * min = b.CreateAtomicLoadAcquire(b.getSizeTy(), syncLockPtr);
    assert (!mIsIOProcessThread || windowLength);
    if (mIsIOProcessThread) {
        min = b.CreateAdd(min, windowLength);
    }
    Value * const isProgressedFarEnough = b.CreateICmpULT(mSegNo, min);
    Value * isTerminated = nullptr;
    if (mIsIOProcessThread) {
        isTerminated = b.CreateIsNotNull(b.CreateAlignedLoad(b.getSizeTy(), terminatedPtr, SizeTyABIAlignment, true));
    } else {
        isTerminated = b.CreateIsNotNull(readTerminationSignal(b, kernelId));
    }
    b.CreateLikelyCondBr(b.CreateOr(isProgressedFarEnough, isTerminated), segmentCheckExit, segmentCheckLoop);

    b.SetInsertPoint(segmentCheckExit);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getSynchronizationLockPtrForKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getSynchronizationLockPtrForKernel(KernelBuilder & b, const unsigned kernelId, const unsigned type) const {
    Value * ptr = getScalarFieldPtr(b, makeKernelName(kernelId) + LOGICAL_SEGMENT_SUFFIX[type]).first;
    ptr->setName(makeKernelName(kernelId) + __getSyncLockNameString(type));
    return ptr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief obtainNextSegmentNumber
 ** ------------------------------------------------------------------------------------------------------------- */

Value * PipelineCompiler::obtainNextSegmentNumber(KernelBuilder & b) {
    assert (UseJumpGuidedSynchronization);
    Value * ptr; Type * ty;
    std::tie(ptr, ty) = getScalarFieldPtr(b,
        NEXT_LOGICAL_SEGMENT_NUMBER + std::to_string(mKernelId));
    assert (ty == b.getSizeTy());
    const auto prefix = makeKernelName(mKernelId);
    LoadInst * const nextSegNo = b.CreateAlignedLoad(ty, ptr, SizeTyABIAlignment,  prefix + "_nextSegNo");
    b.CreateAlignedStore(b.CreateAdd(nextSegNo, b.getSize(1)), ptr, SizeTyABIAlignment);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, prefix + ": obtained %" PRIu64 "-th next segment number %" PRIu64,
               b.getSize(mKernelId), nextSegNo);
    #endif
    if (LLVM_UNLIKELY(CheckAssertions)) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "%s: nested-segment number is %" PRIu64
               " but was expected to be no more than %" PRIu64;
        Value * const success = b.CreateICmpULE(nextSegNo, mSegNo);
        b.CreateAssert(success, out.str(), mKernelName[mKernelId], nextSegNo, mSegNo);
    }
    return nextSegNo;
}

}
