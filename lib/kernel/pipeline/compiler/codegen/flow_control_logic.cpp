#include "../pipeline_compiler.hpp"

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
    if (mBufferGraph[kernelId].controlsSlidingWindow()) {
        mTarget->addInternalScalar(b.getSizeTy(), SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(kernelId), groupId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateBufferScalingFactor
 ** ------------------------------------------------------------------------------------------------------------- */
Rational PipelineCompiler::calculateBufferScalingFactor(const unsigned kernelId) const {
    assert (kernelId == FirstKernelInPartition[KernelPartitionId[kernelId]]);
    Rational scale{0};
    if (in_degree(kernelId, mBufferGraph) == 0) {
        assert (out_degree(kernelId, mBufferGraph) > 0);
        for (auto input : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto streamSet = target(input, mBufferGraph);
            assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
            const auto & bn = mBufferGraph[streamSet];
            scale = std::max(scale, bn.RelativeIORate);
        }
        scale *= Rational{mTarget->getStride(), getKernel(kernelId)->getStride()};
        assert (scale.numerator() > 0);
    } else {
        assert (in_degree(kernelId, mBufferGraph) > 0);
        for (auto input : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const auto streamSet = source(input, mBufferGraph);
            assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
            const auto & bn = mBufferGraph[streamSet];
            scale = std::max(scale, bn.RelativeIORate);
        }
        assert (scale.numerator() > 0);
    }

    return scale;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeInitialSlidingWindowSegmentLengths
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeInitialSlidingWindowSegmentLengths(KernelBuilder & b, Value * const segmentLengthScalingFactor) {
    if (LLVM_UNLIKELY(CheckAssertions())) {
        b.CreateAssert(segmentLengthScalingFactor, "segmentLengthScalingFactor cannot be zero %s", mCurrentKernelName);
    }
    for (unsigned i = 1; i < (PartitionCount - 1); ++i) {
        const auto f = FirstKernelInPartition[i];
        if (mBufferGraph[f].controlsSlidingWindow()) {
            const auto factor = calculateBufferScalingFactor(f);
            Value * init = b.CreateMulRational(segmentLengthScalingFactor, factor);
            init = b.CreateRoundUpRational(init, StrideStepLength[f]);
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
    if (num_edges(ThreadLocalPlacement) > 0) {
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
        assert (mKernelId == FirstKernelInPartition[mCurrentPartitionId]);

        // If the min and max num of strides is equal, we almost certainly have strictly fixed
        // rate input into this partition. However if this a nested pipeline, we cannot assume
        // that the outer pipeline will feed data to this at a fixed rate.

        const auto & bn = mBufferGraph[mKernelId];
        if (bn.controlsSlidingWindow()) {
            mMaximumNumOfStrides = b.getScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId));
        } else if (bn.permitSlidingWindow()) {
            mMaximumNumOfStrides = nullptr;
        } else {
            const auto factor = calculateBufferScalingFactor(mKernelId);
            mMaximumNumOfStrides = b.CreateCeilUMulRational(mExpectedNumOfStridesMultiplier, factor);
            mMaximumNumOfStrides = b.CreateRoundUpRational(mMaximumNumOfStrides, StrideStepLength[mKernelId]);
        }
        allocateThreadLocalMemoryForMaximumNumOfStrides(b);
    } else {
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
    assert (mIsPartitionRoot);
    const auto & bn = mBufferGraph[mKernelId];
    if (bn.controlsSlidingWindow()) {
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
