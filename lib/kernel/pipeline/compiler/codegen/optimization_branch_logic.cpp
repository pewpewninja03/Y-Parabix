#include "../pipeline_compiler.hpp"
#include <kernel/pipeline/optimizationbranch.h>

namespace kernel {

// The condition stream marker defines regions in which we cannot take the optimization branch and we
// are always safe to take the "normal" (non-optimization) branch with the only observable difference
// being a (single-threaded) performance penalty.

// When we have multiple threads simultaneously executing this kernel, the (K+1)-th thread can only
// begin processing data after the K-th thread starts executing its *final* subsegment (assuming that
// both the final subsegment of the K-th thread and the first subsegment of the (K+1) thread branches
// along the same path.

// Consequently, we do not want to aggressively alternate between optimization and normal path branches
// if only a small amount of work can be done as this would delay other threads from executing. To
// accommodate this, we have a minimum span length critera that defines how many strides of data
// must be permitted by the optimization branch before executing it.

#define MINIMUM_SPAN_LENGTH_OF_OPTIMIZATION_BRANCH 1

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isEitherOptimizationBranchKernelInternallySynchronized
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::isEitherOptimizationBranchKernelInternallySynchronized() const {
    const OptimizationBranch * const optBr = cast<OptimizationBranch>(mKernel);
    const auto a = optBr->getNonZeroKernel()->hasAttribute(AttrId::InternallySynchronized);
    const auto b = optBr->getAllZeroKernel()->hasAttribute(AttrId::InternallySynchronized);
    if (LLVM_UNLIKELY(a != b)) {
        report_fatal_error("PipelineComiler does not currently support OptimizationBranch"
                           " with differing internally synchronized values");
    }
    return a;
}

inline bool isConstantOne(const Value * const value) {
    return (isa<ConstantInt>(value) && cast<ConstantInt>(value)->isOne());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkOptimizationBranchSpanLength
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::checkOptimizationBranchSpanLength(KernelBuilder & b, Value * const numOfLinearStrides) {
#if 0
    const OptimizationBranch * const optBr = cast<OptimizationBranch>(mKernel);
    Relationship * const cond = optBr->getCondition();
    if (LLVM_UNLIKELY(!isa<StreamSet>(cond))) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }

    RelationshipType condInput;
    for (const auto e : make_iterator_range(in_edges(mKernelId, mStreamGraph))) {
        const auto binding = source(e, mStreamGraph);
        assert (mStreamGraph[binding].Type == RelationshipNode::IsBinding);
        const auto f = first_in_edge(binding, mStreamGraph);
        assert (mStreamGraph[f].Reason != ReasonType::Reference);
        const auto streamSet = source(f, mStreamGraph);
        const auto & rn = mStreamGraph[streamSet];
        assert (rn.Type == RelationshipNode::IsStreamSet);
        if (cast<StreamSet>(rn.Relationship) == cond) {
            condInput = mStreamGraph[e];
            assert (condInput.Type == PortType::Input);
            break;
        }
    }

    const auto streamSetIdx = getInputBufferVertex(condInput);

    const BufferNode & bn = mBufferGraph[streamSetIdx];
    StreamSetBuffer * const buffer = bn.OutputBuffer;

    if (LLVM_UNLIKELY(!isConstantOne(buffer->getStreamSetCount(b)))) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }


    const Binding & condBinding = getInputBinding(condInput);
    const ProcessingRate & condRate = condBinding.getRate();

    if (LLVM_UNLIKELY(!condRate.isFixed())) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }

    const auto bw = b.getBitBlockWidth();

    const auto strideRate = condRate.getRate()
        * (mKernel->getStride() * cast<StreamSet>(cond)->getFieldWidth())
        / Rational{bw};

    assert (strideRate.denominator() == 1);

    const auto blocksPerStride = strideRate.numerator();

    Constant * const BLOCKS_PER_STRIDE = b.getSize(blocksPerStride);

    IntegerType * const sizeTy = b.getSizeTy();
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const sz_MIN_LENGTH = b.getSize(MINIMUM_SPAN_LENGTH_OF_OPTIMIZATION_BRANCH);
    Constant * const BIT_BLOCK_WIDTH = b.getSize(bw);

    VectorType * const bitBlockTy = b.getBitBlockType();

    Value * const baseAddress = buffer->getBaseAddress(b);

    const auto prefix = makeKernelName(mKernelId);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const scanLengthEndOfRegularSpan = b.CreateBasicBlock(prefix + "_scanLengthEndOfRegularSpan", mKernelLoopCall);
    BasicBlock * const scanLengthFindEndOfOptSpan = b.CreateBasicBlock(prefix + "_scanLengthFindEndOfOptSpan", mKernelLoopCall);
    BasicBlock * const scanLengthCheck = b.CreateBasicBlock(prefix + "_scanLengthCheck", mKernelLoopCall);
    BasicBlock * const scanLengthExit = b.CreateBasicBlock(prefix + "_scanLengthExit", mKernelLoopCall);

    Value * const totalExecutedNumOfStrides =
        b.CreateExactUDiv(mCurrentProcessedItemCountPhi[condInput], BIT_BLOCK_WIDTH);
    Value * const limit = b.CreateAdd(totalExecutedNumOfStrides, numOfLinearStrides);

    // Prior scan state is always initially 0
    b.CreateUnlikelyCondBr(mOptimizationBranchPriorScanStatePhi, scanLengthFindEndOfOptSpan, scanLengthEndOfRegularSpan);

    // Assume that we just left a regular branch. For us to get back to this loop, we must have
    // located an optimization span of sufficient length. Keep scanning ahead and determine
    // the first
    b.SetInsertPoint(scanLengthFindEndOfOptSpan);
    PHINode * const optNumOfStridesPhi = b.CreatePHI(sizeTy, 3, prefix + "_optNumOfStridesPhi");
    // on the final partial stride, we'd have 0-3 strides here.
    optNumOfStridesPhi->addIncoming(totalExecutedNumOfStrides, entry);
    Value * const optIdx = b.CreateMul(optNumOfStridesPhi, BLOCKS_PER_STRIDE);
    Value * optAddr = buffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, optIdx);
    optAddr = b.CreatePointerCast(optAddr, bitBlockTy->getPointerTo());
    Value * optCondVal = b.CreateLoad(bitBlockTy, optAddr);
    for (unsigned i = 1; i < blocksPerStride; ++i) {
        Value * const val = b.CreateLoad(bitBlockTy, b.CreateGEP(bitBlockTy, optAddr, b.getInt32(i)));
        optCondVal = b.CreateOr(optCondVal, val);
    }
    Value * const foundNonOpt = b.bitblock_any(optCondVal);
    Value * const optNextNumOfStrides = b.CreateAdd(optNumOfStridesPhi, sz_ONE);
    Value * const optAtLimit = b.CreateICmpUGE(optNextNumOfStrides, limit);
    Value * const optDone = b.CreateOr(foundNonOpt, optAtLimit);
    optNumOfStridesPhi->addIncoming(optNextNumOfStrides, scanLengthFindEndOfOptSpan);
    b.CreateCondBr(optDone, scanLengthExit, scanLengthFindEndOfOptSpan);

    // Assume that we're starting in a regular branch span and locate the end of it, which can
    // happen when we locate an optimization span of sufficient length or the end of the input.
    b.SetInsertPoint(scanLengthEndOfRegularSpan);
    PHINode * const regNumOfStridesPhi = b.CreatePHI(sizeTy, 2, prefix + "_regNumOfStridesPhi");
    regNumOfStridesPhi->addIncoming(totalExecutedNumOfStrides, entry);
    PHINode * const regOptimizedSpanLengthPhi = b.CreatePHI(sizeTy, 2, prefix + "_regSpanLengthPhi");
    regOptimizedSpanLengthPhi->addIncoming(sz_ZERO, entry);

    Value * const regIdx = b.CreateMul(regNumOfStridesPhi, BLOCKS_PER_STRIDE);
    Value * regAddr = buffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, regIdx);
    regAddr = b.CreatePointerCast(regAddr, bitBlockTy->getPointerTo());
    Value * regCondVal = b.CreateLoad(bitBlockTy, regAddr);
    for (unsigned i = 1; i < blocksPerStride; ++i) {
        Value * const val = b.CreateLoad(bitBlockTy, b.CreateGEP(bitBlockTy, regAddr, b.getInt32(i)));
        regCondVal = b.CreateOr(regCondVal, val);
    }
    regCondVal = b.bitblock_any(regCondVal);

    Value * const regIncOptimizedSpanLength = b.CreateAdd(regOptimizedSpanLengthPhi, sz_ONE);
    Value * const regNextOptimizedSpanLength = b.CreateSelect(regCondVal, sz_ZERO, regIncOptimizedSpanLength);
    Value * const regSpanLargeEnough = b.CreateICmpEQ(regNextOptimizedSpanLength, sz_MIN_LENGTH);
    Value * const regNextNumOfStrides = b.CreateAdd(regNumOfStridesPhi, sz_ONE);
    Value * const regAtLimit = b.CreateICmpUGE(regNextNumOfStrides, limit);
    Value * const regDone = b.CreateOr(regSpanLargeEnough, regAtLimit);

    regOptimizedSpanLengthPhi->addIncoming(regNextOptimizedSpanLength, scanLengthEndOfRegularSpan);
    regNumOfStridesPhi->addIncoming(regNextNumOfStrides, scanLengthEndOfRegularSpan);
    b.CreateCondBr(regDone, scanLengthCheck, scanLengthEndOfRegularSpan);

    b.SetInsertPoint(scanLengthCheck);
    Value * const anyRegularStrides = b.CreateICmpNE(regNextNumOfStrides, regNextOptimizedSpanLength);
    Value * const finishedScanning = b.CreateOr(anyRegularStrides, regAtLimit);
    Value * const regNumOfStrides = b.CreateSelect(regSpanLargeEnough, regNextNumOfStrides, limit);
    optNumOfStridesPhi->addIncoming(regNextNumOfStrides, scanLengthCheck);
    b.CreateCondBr(finishedScanning, scanLengthExit, scanLengthFindEndOfOptSpan);

    b.SetInsertPoint(scanLengthExit);
    PHINode * const finalNumOfStridesPhi = b.CreatePHI(sizeTy, 3, prefix + "_scanLengthNumOfStridesPhi");
    finalNumOfStridesPhi->addIncoming(optNextNumOfStrides, scanLengthFindEndOfOptSpan);
    finalNumOfStridesPhi->addIncoming(regNumOfStrides, scanLengthCheck);
    #ifdef OPTIMIZATION_BRANCH_ALWAYS_TAKES_REGULAR_BRANCH
    mOptimizationBranchSelectedBranch = b.getTrue();
    #else
    PHINode * const chosenBranchPhi = b.CreatePHI(b.getInt1Ty(), 2);
    chosenBranchPhi->addIncoming(b.getFalse(), scanLengthFindEndOfOptSpan);
    chosenBranchPhi->addIncoming(anyRegularStrides, scanLengthCheck);
    mOptimizationBranchSelectedBranch = chosenBranchPhi;
    #endif

    Value * const finalNumOfStrides = b.CreateSub(finalNumOfStridesPhi, totalExecutedNumOfStrides);
    Value * const isFinal = b.CreateICmpEQ(numOfLinearStrides, sz_ZERO);
    Value * const selectedNumOfStrides = b.CreateSelect(isFinal, sz_ZERO, finalNumOfStrides);

    if (LLVM_UNLIKELY(CheckAssertions())) {
        Value * const valid = b.CreateICmpULE(selectedNumOfStrides, numOfLinearStrides);
        b.CreateAssert(valid, "%s: optimization branch span length (%" PRIu64 ") "
                               "exceeds maximum num of strides (%" PRIu64 ")",
                        b.GetString(prefix), selectedNumOfStrides, numOfLinearStrides);
        Value * const valid2 = b.CreateOr(b.CreateICmpNE(selectedNumOfStrides, sz_ZERO), isFinal);
        b.CreateAssert(valid2, "%s: optimization branch span length "
                                "cannot be zero unless parsing final stride)",
                        b.GetString(prefix));
    }

    return selectedNumOfStrides;
#endif
    return nullptr;
}

}
