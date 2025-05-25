#include "../pipeline_compiler.hpp"
#include <queue>

// Each partition root determines how much data that it (and consequently its partition) can
// process in a single segment / pipeline iteration. However, it does not necessarily need to
// transfer all of the data provided to it and at times it may be beneficial to withhold data
// to better balance thread workloads.

// The functions here are designed to dynamically managed the maximum segment length of a
// partition root to promote this. By doing so, they may have to malloc a larger thread local
// memory pool or increase the repetition length of repeating streamsets.

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addSegmentLengthSlidingWindowKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addSegmentLengthSlidingWindowKernelProperties(KernelBuilder & b, const size_t kernelId, const size_t groupId) {
    assert (FirstKernel <= kernelId && kernelId <= LastKernel);
    assert ("not root?" && (FirstKernelInPartition[KernelPartitionId[kernelId]] == kernelId));
    if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || mIsNestedPipeline) {
        assert (FirstComputePartitionId <= KernelPartitionId[kernelId] && KernelPartitionId[kernelId] <= LastComputePartitionId);
        mTarget->addInternalScalar(b.getSizeTy(), SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(kernelId), groupId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeInitialSlidingWindowSegmentLengths
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeInitialSlidingWindowSegmentLengths(KernelBuilder & b, Value * const segmentLengthScalingFactor) {
    if (LLVM_UNLIKELY(CheckAssertions)) {
        b.CreateAssert(segmentLengthScalingFactor, "segmentLengthScalingFactor cannot be zero %s", mCurrentKernelName);
    }

    for (unsigned i = FirstComputePartitionId; i <= LastComputePartitionId; ++i) {
        const auto f = FirstKernelInPartition[i];
        if (MinimumNumOfStrides[f] != MaximumNumOfStrides[f] || mIsNestedPipeline) {
            Value * const init = b.CreateMulRational(segmentLengthScalingFactor, MaximumNumOfStrides[f]);
            b.setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(f), init);
        } else {
            assert (!b.hasScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(f)));
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeFlowControl
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeFlowControl(KernelBuilder & b) {
    if (num_edges(ThreadLocalPlacement) > 0 > 0 && !mIsIOProcessThread) {
        mThreadLocalMemorySizePtr = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES).first;
    } else {
        mThreadLocalMemorySizePtr = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief detemineMaximumNumberOfStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::detemineMaximumNumberOfStrides(KernelBuilder & b) {
    // The partition root kernel determines the amount of data processed based on the partition input.
    // During normal execution, it always performs max num of strides worth of work. However, if this
    // segment is the last segment, mPartitionSegmentLength is artificially raised to a value of ONE
    // even though we likely can only execute a partial segment worth of work. The root kernel
    // calculates how many strides can be performed and sets mNumOfPartitionStrides to that value.

    // To avoid having every kernel test their I/O during normal execution, the non-root kernels in
    // the same partition refer to the mNumOfPartitionStrides to determine how their segment length.

    if (mIsPartitionRoot) {
        assert (mCurrentPartitionId == KernelPartitionId[mKernelId]);
        assert (mKernelId == FirstKernelInPartition[KernelPartitionId[mKernelId]]);
        const auto firstKernelOfNextPartition = FirstKernelInPartition[mCurrentPartitionId + 1];


        // If the min and max num of strides is equal, we almost certainly have strictly fixed
        // rate input into this partition. However if this a nested pipeline, we cannot assume
        // that the outer pipeline will feed data to this at a fixed rate.
        if (mBufferGraph[mKernelId].permitSlidingWindow()) {
            assert (!mIsIOProcessThread);

            mMaximumNumOfStrides = b.getScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId));

            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * check = b.CreateOr(b.CreateICmpNE(mMaximumNumOfStrides, b.getSize(0)), mInitiallyTerminated);
                b.CreateAssert(check, "Maximum number of strides cannot be zero %s (A)", mCurrentKernelName);
            }

            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "%s.maxNumOfStrides=%" PRIu64, mCurrentKernelName, mMaximumNumOfStrides);
            #endif

            // calculate how much memory is required by this partition relative to max num of strides
            // and determine if the current thread local buffer can fit it.

        } else {

            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * check = b.CreateOr(b.CreateICmpNE(mExpectedNumOfStridesMultiplier, b.getSize(0)), mInitiallyTerminated);
                b.CreateAssert(check, "Expected number of strides multipler cannot be zero %s (B)", mCurrentKernelName);
            }

            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, getName() + "_mExpectedNumOfStridesMultiplier = %" PRIu64, mExpectedNumOfStridesMultiplier);
            #endif

            const Rational factor{mTarget->getStride(), mKernel->getStride()}; assert (factor > 0);
            mMaximumNumOfStrides = b.CreateCeilUMulRational(mExpectedNumOfStridesMultiplier, factor);
        }



        const auto hasThreadLocal = out_degree(mCurrentPartitionId, ThreadLocalPlacement);

        Value * threadLocalPtr = nullptr;
        Type * threadLocalTy = nullptr;
        if (hasThreadLocal) {
            std::tie(threadLocalPtr, threadLocalTy) = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);

            const auto m = PartitionCount + LastStreamSet - FirstStreamSet + 1;
            assert (num_vertices(ThreadLocalPlacement) == m);
            std::vector<unsigned> toVisit(m, 0);
            #ifndef NDEBUG
            for (unsigned i = 0; i < PartitionCount; ++i) {
                assert(in_degree(i, ThreadLocalPlacement) == 0);
            }
            #endif
            for (unsigned i = PartitionCount; i < m; ++i) {
                toVisit[i] = in_degree(i, ThreadLocalPlacement);
            }

            const auto pageSize = getPageSize();
            const auto log2PageSize = floor_log2(pageSize);
            const auto useShift = is_pow2(pageSize);
            ConstantInt * PAGE_SIZE = b.getSize(useShift ? log2PageSize : pageSize);

            std::queue<unsigned> Q;
            std::vector<Value *> endPosition(m);

            assert (in_degree(mCurrentPartitionId, ThreadLocalPlacement) == 0);
            endPosition[mCurrentPartitionId] = b.getSize(0);

            assert (mMaximumNumOfStrides);

            Value * memoryForSegment = nullptr;
            for (auto u = mCurrentPartitionId;;) {
                for (auto e : make_iterator_range(out_edges(u, ThreadLocalPlacement))) {
                    const auto v = target(e, ThreadLocalPlacement);
                    assert (v >= PartitionCount);
                    assert (toVisit[v] > 0);
                    if (--toVisit[v] == 0) {
                        Rational R{ThreadLocalPlacement[e], StrideStepLength[mKernelId]};
                        Value * off = b.CreateCeilUMulRational(mMaximumNumOfStrides, R);
                        if (LLVM_LIKELY(useShift)) {
                            off = b.CreateShl(off, PAGE_SIZE);
                        } else {
                            off = b.CreateMul(off, PAGE_SIZE);
                        }
                        assert (endPosition[u]);
                        mThreadLocalStartOffset[v] = endPosition[u];
                        endPosition[v] = b.CreateAdd(endPosition[u], off);
                        if (out_degree(v, ThreadLocalPlacement) == 0) {
                            memoryForSegment = b.CreateUMax(memoryForSegment, endPosition[v]);
                        } else {
                            Q.push(v);
                        }
                    }
                }
                if (Q.empty()) {
                    break;
                }
                assert (Q.front() != u);
                u = Q.front();
                assert (toVisit[u] == 0);
                Q.pop();
                assert (Q.front() != u);
            }

            if (mBufferGraph[mKernelId].permitSlidingWindow()) {
                BasicBlock * const expandThreadLocalMemory = b.CreateBasicBlock();
                BasicBlock * const afterExpansion = b.CreateBasicBlock();
                Value * currentMem = b.CreateAlignedLoad(b.getSizeTy(), mThreadLocalMemorySizePtr, SizeTyABIAlignment);
                Value * const needsExpansion = b.CreateICmpUGT(memoryForSegment, currentMem);
                b.CreateCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

                b.SetInsertPoint(expandThreadLocalMemory);

                b.CreateFree(b.CreateAlignedLoad(threadLocalTy, threadLocalPtr, PtrTyABIAlignment));
                // At minimum, we want to double the required space to minimize future reallocs
                Value * expanded = b.CreateRoundUp(memoryForSegment, currentMem);
                b.CreateAlignedStore(expanded, mThreadLocalMemorySizePtr, SizeTyABIAlignment);
                #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
                expanded = b.CreateMul(expanded, b.getSize(THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER));
                #endif
                Value * const base = b.CreatePageAlignedMalloc(expanded);
                b.CreateAlignedStore(base, threadLocalPtr, PtrTyABIAlignment);
                b.CreateBr(afterExpansion);

                b.SetInsertPoint(afterExpansion);
            }

            mThreadLocalStreamSetBaseAddress = b.CreateAlignedLoad(threadLocalTy, threadLocalPtr, PtrTyABIAlignment);
        } else {
            mThreadLocalStreamSetBaseAddress = nullptr;
//            mThreadLocalScalingFactor = nullptr;
        }

    } else {
        assert (!mIsIOProcessThread);
        const Rational ratio{StrideStepLength[mKernelId], StrideStepLength[mCurrentPartitionRoot]};
        const auto factor = ratio / mPartitionStrideRateScalingFactor;
        assert (factor.numerator() > 0);
        mMaximumNumOfStrides = b.CreateMulRational(mNumOfPartitionStrides, factor);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateNextSlidingWindowSize
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateNextSlidingWindowSize(KernelBuilder & b, Value * const maxNumOfStrides, Value * const potentialNumOfStrides) {
    assert (!mIsIOProcessThread);
    assert (mIsPartitionRoot);
    if (MinimumNumOfStrides[mKernelId] != MaximumNumOfStrides[mKernelId] || mIsNestedPipeline) {
        ConstantInt * const TWO = b.getSize(2);
        Value * const A = b.CreateMul(maxNumOfStrides, TWO);
        Value * const B = b.CreateAdd(maxNumOfStrides, potentialNumOfStrides);
        assert (StrideStepLength[mKernelId] > 0);
        ConstantInt * const stepLength = b.getSize(StrideStepLength[mKernelId] * 2U);
        Value * const C = b.CreateRoundUp(B, stepLength);
        Value * const D = b.CreateUDiv(C, TWO);
        Value * const higher = b.CreateICmpUGT(potentialNumOfStrides, maxNumOfStrides);
        Value * const nextMaxNumOfStrides = b.CreateSelect(higher, A, D);
        b.setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId), nextMaxNumOfStrides);
    }
}

} // end of namespace
