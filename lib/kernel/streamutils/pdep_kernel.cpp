/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/deletion.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/core/idisa_target.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/bitwise/bixlogic.h>
#include <pablo/pablo_kernel.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/intrusive/detail/math.hpp>
#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>

using boost::intrusive::detail::floor_log2;
using namespace llvm;
using namespace pablo;
using namespace kernel;

static cl::opt<bool> ElemSpread("ElemSpread", cl::desc("Use ElemSpreadKernel in place of byte spread by mask"), cl::init(true), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> SeparatedMergeByMask("separated-merge-by-mask", cl::desc("implement merge-by-mask by combining two spread-by-mask steps"), cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> UnalignedLoads("UnalignedLoads", cl::desc("Use unaligned loads in ElemSpread"), cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> RecursiveSpreadMaskCalculation("RecursiveSpreadMaskCalculation", cl::desc("Use recursive multi-kernel approach to insertion spread mask calculation (legacy)"), cl::init(false), cl::cat(codegen::CodeGenOptions));

namespace kernel {

class ElemSpreadKernel final  : public MultiBlockKernel {
public:
    ElemSpreadKernel(LLVMTypeSystemInterface & ts,
                       StreamSet * mask,
                       StreamSet * source,
                       StreamSet * spread);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const unsigned mElemWidth;
};

void SpreadByMask(PipelineBuilder & P,
                  StreamSet * mask, StreamSet * toSpread, StreamSet * outputs,
                  unsigned streamOffset,
                  bool zeroExtend,
                  StreamExpandOptimization opt,
                  unsigned expansionFieldWidth,
                  ProcessingRateProbabilityDistribution itemsPerOutputUnit) {
    if (toSpread->getFieldWidth() < 8) {
        unsigned streamCount = outputs->getNumElements();
        StreamSet * const expanded = P.CreateStreamSet(streamCount);
        Scalar * base = P.CreateConstant(P.getSize(streamOffset));
        P.CreateKernelCall<StreamExpandKernel>(mask, toSpread, expanded, base, zeroExtend, opt, expansionFieldWidth, itemsPerOutputUnit);
        P.CreateKernelCall<FieldDepositKernel>(mask, expanded, outputs, expansionFieldWidth);
    } else {
        Scalar * offset = nullptr;
        bool useElemSpread = ElemSpread;
        if (streamOffset || toSpread->getNumElements() != outputs->getNumElements()) {
            offset = P.CreateConstant(P.getSize(streamOffset));
            useElemSpread = false;
        }
        if (useElemSpread) {
            P.CreateKernelCall<ElemSpreadKernel>(mask, toSpread, outputs);
        } else {
            P.CreateKernelCall<ByteSpreadByMaskKernel>(toSpread, mask, outputs, offset);
        }
    }
}

//  TODO:  Consider adding toFill parameter to SpreadByMask so that filter expansion
//  can be implemented by a single SpreadByMask call.
void ExpandFilter(PipelineBuilder & P,
                  StreamSet * spreadMask, StreamSet * filter, StreamSet * expanded) {
    StreamSet * initialSpread = P.CreateStreamSet(1, 1);
    SpreadByMask(P, spreadMask, filter, initialSpread);
    StreamSet * filler = P.CreateStreamSet(1, 1);
    Invert(P, spreadMask, filler);
    OrCombine(P, initialSpread, filler, expanded);
}

class ElemMergeKernel final  : public MultiBlockKernel {
public:
    ElemMergeKernel(LLVMTypeSystemInterface & ts,
                       StreamSet * mask,
                       StreamSet * source1,
                       StreamSet * source2,
                       StreamSet * merged);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const unsigned mElemWidth;
};

void MergeByMask(PipelineBuilder & P,
                 StreamSet * mask, StreamSet * a, StreamSet * b, StreamSet * merged) {
    unsigned elems = merged->getNumElements();
    unsigned fw = merged->getFieldWidth();
    if ((a->getNumElements() != elems) || (b->getNumElements() != elems)) {
        llvm::report_fatal_error("MergeByMask called with incompatible element counts");
    }
    if ((a->getFieldWidth() != fw) || (b->getFieldWidth() != fw)) {
        llvm::report_fatal_error("MergeByMask called with incompatible field widths");
    }
    if (SeparatedMergeByMask) {
        StreamSet * expandedA = P.CreateStreamSet(elems);
        SpreadByMask(P, mask, a, expandedA);
        StreamSet * inverted = P.CreateStreamSet(1);
        Invert(P, mask, inverted);
        StreamSet * expandedB = P.CreateStreamSet(elems);
        SpreadByMask(P, inverted, b, expandedB);
        OrCombine(P, expandedA, expandedB, merged);
    } else if (ElemSpread && (elems == 1) && (fw >= 8)) {
        P.CreateKernelCall<ElemMergeKernel>(mask, a, b, merged);
    } else {
        unsigned streamOffset = 0;
        Scalar * base = P.CreateConstant(P.getSize(streamOffset));
        int expansionFieldWidth=64;
        P.CreateKernelCall<StreamMergeKernel>(mask,a,b,merged,base,expansionFieldWidth);
    }
}

const unsigned StreamExpandStrideSize = 4;

StreamExpandKernel::StreamExpandKernel(LLVMTypeSystemInterface & ts,
                                       StreamSet * mask,
                                       StreamSet * source,
                                       StreamSet * expanded,
                                       Scalar * base,
                                       bool zeroExtend,
                                       const StreamExpandOptimization opt,
                                       const unsigned FieldWidth, ProcessingRateProbabilityDistribution itemsPerOutputUnitProbability)
: MultiBlockKernel(ts, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "streamExpand"  << StreamExpandStrideSize << ':' << FieldWidth;
                        if (opt == StreamExpandOptimization::NullCheck)  {
                            nm << 'N';
                        }
                        nm << '_' << source->getNumElements() << ':' << expanded->getNumElements();
                        if (zeroExtend) nm << "z";
                        nm.flush();
                        return tmp;
                    }(),
{Bind("marker", mask, Principal())},
{Binding{"output", expanded}},
// input scalar
{Binding{"base", base}},
{}, {})
, mFieldWidth(FieldWidth)
, mSelectedStreamCount(expanded->getNumElements())
, mOptimization(opt) {
    setStride(StreamExpandStrideSize * ts.getBitBlockWidth());
    if (zeroExtend) {
        mInputStreamSets.emplace_back(Bind("source", source, PopcountOf("marker"), itemsPerOutputUnitProbability, ZeroExtended(), EmptyReadOverflow()));
    } else {
        mInputStreamSets.emplace_back(Bind("source", source, PopcountOf("marker"), itemsPerOutputUnitProbability, EmptyReadOverflow()));
    }
}

void StreamExpandKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    Type * fieldWidthTy = b.getIntNTy(mFieldWidth);
    Type * sizeTy = b.getSizeTy();
    const unsigned numFields = b.getBitBlockWidth() / mFieldWidth;

    Constant * const ZERO = b.getSize(0);
    Constant * BLOCK_WIDTH = ConstantInt::get(sizeTy, b.getBitBlockWidth());
    Constant * FIELD_WIDTH = ConstantInt::get(sizeTy, mFieldWidth);
    Constant * fwSplat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth));
    Constant * fw_sub1Splat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth - 1));

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * expandLoop = b.CreateBasicBlock("expandLoop");
    BasicBlock * expansionDone = b.CreateBasicBlock("expansionDone");
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        ConstantInt * const mult = b.getSize(getStride() / b.getBitBlockWidth());
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }

    Value * processedSourceItems = b.getProcessedItemCount("source");
    Value * initialSourceOffset = b.CreateURem(processedSourceItems, BLOCK_WIDTH);
    Value * const streamBase = b.getScalarField("base");

    SmallVector<Value *, 16> pendingData(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        pendingData[i] = b.loadInputStreamBlock("source", streamOffset, ZERO);
    }

    b.CreateBr(expandLoop);
    // Main Loop
    b.SetInsertPoint(expandLoop);
    const unsigned incomingCount = (mOptimization == StreamExpandOptimization::NullCheck) ? 3 : 2;
    PHINode * blockNoPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    SmallVector<PHINode *, 16> pendingDataPhi(mSelectedStreamCount);
    blockNoPhi->addIncoming(ZERO, entry);
    pendingOffsetPhi->addIncoming(initialSourceOffset, entry);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi[i]->addIncoming(pendingData[i], entry);
    }

    Value * deposit_mask = b.loadInputStreamBlock("marker", ZERO, blockNoPhi);
    Value * nextBlk = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    if (mOptimization == StreamExpandOptimization::NullCheck) {
        BasicBlock * expandLoopContinue = b.CreateBasicBlock("expandLoopContinue");
        BasicBlock * nullMarkers = b.CreateBasicBlock("nullMarkers");
        b.CreateCondBr(b.bitblock_any(deposit_mask), expandLoopContinue, nullMarkers);

        b.SetInsertPoint(nullMarkers);
        Constant * zeroes = b.allZeroes();
        for (unsigned i = 0; i < mSelectedStreamCount; i++) {
            b.storeOutputStreamBlock("output", b.getInt32(i), blockNoPhi, zeroes);
        }
        blockNoPhi->addIncoming(nextBlk, nullMarkers);
        pendingOffsetPhi->addIncoming(pendingOffsetPhi, nullMarkers);
        for (unsigned i = 0; i < mSelectedStreamCount; i++) {
            pendingDataPhi[i]->addIncoming(pendingDataPhi[i], nullMarkers);
        }
        b.CreateCondBr(moreToDo, expandLoop, expansionDone);

        b.SetInsertPoint(expandLoopContinue);
    }
    // Calculate the field values and offsets we need for assembling a
    // a full block of source bits.  Assembly will use the following operations.
    // A = b.simd_srlv(fw, b.mvmd_dsll(fw, source, pending, field_offset_lo), bit_offset);
    // B = b.simd_sllv(fw, b.mvmd_dsll(fw, source, pending, field_offset_hi), shift_fwd);
    // all_source_bits = simd_or(A, B);
    Value * pendingOffset = b.CreateURem(pendingOffsetPhi, BLOCK_WIDTH);

    // Value * pendingItems = b.CreateURem(b.CreateSub(bwConst, pendingOffset), bwConst);
    Value * pendingItems = b.CreateSub(BLOCK_WIDTH, pendingOffset);

    Value * field_offset_lo = b.CreateCeilUDiv(pendingItems, FIELD_WIDTH);
    Value * bit_offset = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset, FIELD_WIDTH));
    // Carefully avoid a shift by the full fieldwith (which gives a poison value).
    // field_offset_lo + 1 unless the bit_offset is 0, in which case it is just field_offset_lo.
    Value * field_offset_hi =  b.CreateUDiv(pendingItems, FIELD_WIDTH);
    // fw - bit_offset, unless bit_offset is 0, in which case, the shift_fwd is 0.
    Value * shift_fwd = b.CreateURem(b.CreateSub(fwSplat, bit_offset), fwSplat);

    // Once all source bits are assembled, they need to be distributed to the
    // output fields in accord with the popcounts of the deposit mask fields.
    // The bits for each output field will typically come from (at most) two
    // source fields, with offsets.  Calculate the field numbers and offsets.

    Value * fieldPopCounts = b.simd_popcount(mFieldWidth, deposit_mask);
    // For each field determine the (partial) sum popcount of all fields prior to
    // the current field.

    Value * partialSum = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts);
    assert (partialSum->getType() == fwSplat->getType());
    Value * const blockPopCount = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum, numFields - 1), sizeTy);
    partialSum = b.mvmd_slli(mFieldWidth, partialSum, 1);
    assert (partialSum->getType() == fwSplat->getType());
    Value * const source_field_lo = b.CreateUDiv(partialSum, fwSplat);
    Value * const source_field_hi = b.CreateUDiv(b.CreateAdd(partialSum, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo = b.CreateAnd(partialSum, fw_sub1Splat);  // parallel URem
    Value * const source_shift_hi = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo), fw_sub1Splat);
    // The source stream may not be positioned at a block boundary.  Partial data
    // has been saved in the kernel state, determine the next full block number
    // for loading source streams.
    Value * const newPendingOffset = b.CreateAdd(pendingOffsetPhi, blockPopCount);
    Value * const srcBlockNo = b.CreateUDiv(newPendingOffset, BLOCK_WIDTH);

    // Now load and process source streams.
    SmallVector<Value *, 16> sourceData(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        sourceData[i] = b.loadInputStreamBlock("source", streamOffset, srcBlockNo);
        Value * A = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData[i], pendingDataPhi[i], field_offset_lo), bit_offset);
        Value * B = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData[i], pendingDataPhi[i], field_offset_hi), shift_fwd);
        Value * full_source_block = b.CreateOr(A, B, "toExpand");
        Value * C = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block, source_field_lo), source_shift_lo);
        Value * D = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block, source_field_hi), source_shift_hi);
        Value * output = b.bitCast(b.CreateOr(C, D, "expanded"));
        b.storeOutputStreamBlock("output", b.getInt32(i), blockNoPhi, output);
    }
    //
    // Update loop control Phis for the next iteration.
    //
    blockNoPhi->addIncoming(nextBlk, b.GetInsertBlock());
    pendingOffsetPhi->addIncoming(newPendingOffset, b.GetInsertBlock());
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi[i]->addIncoming(sourceData[i], b.GetInsertBlock());
    }
    //
    // Now continue the loop if there are more blocks to process.
    b.CreateCondBr(moreToDo, expandLoop, expansionDone);

    b.SetInsertPoint(expansionDone);
}

ElemMergeKernel::ElemMergeKernel(LLVMTypeSystemInterface & ts,
                                       StreamSet * mask,
                                       StreamSet * source1,
                                       StreamSet * source2,
                                       StreamSet * merged)
: MultiBlockKernel(ts, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "elemMerge";
                        nm << '_' << source1->getFieldWidth();
                        nm.flush();
                        return tmp;
                    }(),
{Binding("mask", mask, FixedRate(1), Principal()),
 Binding("source1", source1, PopcountOf("mask")),
 Binding("source2", source2, PopcountOfNot("mask"))},
{Binding{"merged", merged}},
{}, {}, {}), mElemWidth(source1->getFieldWidth()) {
    //setStride(ts.getBitBlockWidth()/mElemWidth);
}

void ElemMergeKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    const unsigned maskWidth = b.getBitBlockWidth()/mElemWidth;
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const maskTy = b.getIntNTy(maskWidth);
    Type * const elemVecTy = b.fwVectorType(mElemWidth);

    Constant * const ZERO = b.getSize(0);
    Constant * const ELEMS_PER_PACK = ConstantInt::get(sizeTy, maskWidth);
    Constant * const UREM_MASK = ConstantInt::get(sizeTy, maskWidth - 1);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const elemMergeLoop = b.CreateBasicBlock("elemMergeLoop");
    BasicBlock * const elemMergeDone = b.CreateBasicBlock("elemMergeDone");

    Value * numOfIterations = numOfStrides;
    if (getStride() != maskWidth) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        numOfIterations = b.CreateMul(numOfStrides, b.getSize(getStride() / maskWidth));
    }
    Value * const processedMaskItems = b.getProcessedItemCount("mask");
    Value * const rawMaskPtr = b.getRawInputPointer("mask", processedMaskItems);
    Value * const maskBasePtr = b.CreatePointerCast(rawMaskPtr, maskTy->getPointerTo());

    std::vector<std::string> source = {"source1", "source2"};
    Value * initialSourceOffset[2];
    Value * sourcePackPtr[2];
    Value * initialPendingData[2];
    for (unsigned i = 0; i < 2; i++) {
        Value * const processedSourceItems = b.getProcessedItemCount(source[i]);
        initialSourceOffset[i] = b.CreateURem(processedSourceItems, ELEMS_PER_PACK);
        Value * const sourceItemBase = b.CreateSub(processedSourceItems, initialSourceOffset[i]);
        Value * const sourceBasePtr = b.getRawInputPointer(source[i], sourceItemBase);
        sourcePackPtr[i] = b.CreatePointerCast(sourceBasePtr, elemVecTy->getPointerTo());
        initialPendingData[i] = b.CreateLoad(elemVecTy, sourcePackPtr[i]);
    }

    Value * const producedItems = b.getProducedItemCount("merged");
    Value * const rawOutputPtr = b.getRawOutputPointer("merged", producedItems);
    Value * const outputPackPtr = b.CreatePointerCast(rawOutputPtr, elemVecTy->getPointerTo());

    b.CreateBr(elemMergeLoop);

    b.SetInsertPoint(elemMergeLoop);
    PHINode * const maskNoPhi = b.CreatePHI(sizeTy, 2);
    maskNoPhi->addIncoming(ZERO, entry);
    PHINode * const outputOffsetPhi = b.CreatePHI(sizeTy, 2);
    outputOffsetPhi->addIncoming(ZERO, entry);
    PHINode * sourceOffsetPhi[2];
    PHINode * pendingDataPhi[2];
    for (unsigned i = 0; i < 2; i++) {
        sourceOffsetPhi[i] = b.CreatePHI(sizeTy, 2);
        pendingDataPhi[i] = b.CreatePHI(elemVecTy, 2);
        sourceOffsetPhi[i]->addIncoming(initialSourceOffset[i], entry);
        pendingDataPhi[i]->addIncoming(initialPendingData[i], entry);
    }

    Value * mask[2];
    mask[0] = b.CreateLoad(maskTy, b.CreateGEP(maskTy, maskBasePtr, maskNoPhi));
    mask[1] = b.CreateNot(mask[0]);

    Value * spread[2];
    for (unsigned i = 0; i < 2; i++) {
        Value * const maskPopCount = b.CreateZExtOrTrunc(b.CreatePopcount(mask[i]), sizeTy);
        Value * const pendingPackNo = b.CreateUDiv(sourceOffsetPhi[i], ELEMS_PER_PACK);
        // Elements in the pending pack may already have been processed.
        Value * const pendingElemsDone = b.CreateAnd(sourceOffsetPhi[i], UREM_MASK);
        // If the offset within the pack is nonzero, we have pending data elements to keep.
        Value * pendingElemsToKeep = b.CreateAnd(b.CreateSub(ELEMS_PER_PACK, pendingElemsDone), UREM_MASK);
        Value * const updatedSourceOffset = b.CreateAdd(sourceOffsetPhi[i], maskPopCount);
        Value * const nextPackNo = b.CreateUDiv(updatedSourceOffset, ELEMS_PER_PACK);
        Value * const nextPackElems = b.CreateAnd(updatedSourceOffset, UREM_MASK);
        // We can only safely load a pack if our mask tells us that there are new elements to load.
        Value * const packToLoad = b.CreateSelect(b.CreateIsNull(nextPackElems), pendingPackNo, nextPackNo);
        Value * const packPtr = b.CreateGEP(elemVecTy, sourcePackPtr[i], packToLoad);
        Value * const newPack = b.CreateLoad(elemVecTy, packPtr);
        Value * const shiftedData = b.mvmd_dsll(mElemWidth, newPack, pendingDataPhi[i], pendingElemsToKeep);
        spread[i] = b.mvmd_expand(mElemWidth, shiftedData, mask[i]);
        sourceOffsetPhi[i]->addIncoming(updatedSourceOffset, elemMergeLoop);
        pendingDataPhi[i]->addIncoming(newPack, elemMergeLoop);
    }

    Value * const merged = b.CreateOr(spread[0], spread[1]);
    Value * const outputPtr = b.CreateGEP(elemVecTy, outputPackPtr, maskNoPhi);
    b.CreateStore(merged, outputPtr);

    Value * const nextStride = b.CreateAdd(maskNoPhi, b.getSize(1));
    Value * const moreToDo = b.CreateICmpNE(nextStride, numOfIterations);

    maskNoPhi->addIncoming(nextStride, elemMergeLoop);
    outputOffsetPhi->addIncoming(b.CreateAdd(outputOffsetPhi, ELEMS_PER_PACK), elemMergeLoop);
    b.CreateCondBr(moreToDo, elemMergeLoop, elemMergeDone);

    b.SetInsertPoint(elemMergeDone);
}

ElemSpreadKernel::ElemSpreadKernel(LLVMTypeSystemInterface & ts,
                                       StreamSet * mask,
                                       StreamSet * source,
                                       StreamSet * spread)
: MultiBlockKernel(ts, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "elemSpread";
                        nm << '_' << source->getFieldWidth();
                        if (UnalignedLoads) nm << "u";
                        nm.flush();
                        return tmp;
                    }(),
{Binding("mask", mask, FixedRate(1), Principal()),
 Binding("source", source, PopcountOf("mask"))},
{Binding{"spread", spread}},
{}, {}, {}), mElemWidth(source->getFieldWidth()) {
}

void ElemSpreadKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    const unsigned maskWidth = b.getBitBlockWidth()/mElemWidth;
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const maskTy = b.getIntNTy(maskWidth);
    IntegerType * const elemTy = b.getIntNTy(mElemWidth);
    IntegerType * const metaMaskTy = elemTy;
    Type * const elemVecTy = b.fwVectorType(mElemWidth);

    Constant * const ZERO = b.getSize(0);
    Constant * const ELEMS_PER_PACK = ConstantInt::get(sizeTy, maskWidth);
    Constant * const UREM_MASK = ConstantInt::get(sizeTy, maskWidth - 1);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const blockAtATimeLoop = b.CreateBasicBlock("blockAtATimeLoop");
    BasicBlock * const scanPackLoop = b.CreateBasicBlock("scanPackLoop");
    BasicBlock * const packsDone = b.CreateBasicBlock("packsDone");
    BasicBlock * const elemSpreadDone = b.CreateBasicBlock("elemSpreadDone");

    // Set up the source input pointer and initial offset data.
    Value * const processedSourceItems = b.getProcessedItemCount("source");
    Value * initialSourceOffset = nullptr;
    Value * sourcePtr = nullptr;
    Value * initialPendingData = nullptr;
    if (UnalignedLoads) {
        initialSourceOffset = processedSourceItems;
        Value * const sourceBasePtr = b.getRawInputPointer("source", initialSourceOffset);
        sourcePtr = b.CreatePointerCast(sourceBasePtr, elemTy->getPointerTo());
    } else {
        initialSourceOffset = b.CreateURem(processedSourceItems, ELEMS_PER_PACK);
        Value * const sourceItemBase = b.CreateSub(processedSourceItems, initialSourceOffset);
        Value * const sourceBasePtr = b.getRawInputPointer("source", sourceItemBase);
        sourcePtr = b.CreatePointerCast(sourceBasePtr, elemVecTy->getPointerTo());
        initialPendingData = b.CreateLoad(elemVecTy, sourcePtr);
    }

    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        ConstantInt * const mult = b.getSize(getStride() / b.getBitBlockWidth());
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }

    b.CreateBr(blockAtATimeLoop);

    b.SetInsertPoint(blockAtATimeLoop);
    PHINode * blockNoPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockNoPhi->addIncoming(ZERO, entry);
    PHINode * const blockSourceOffsetPhi = b.CreatePHI(sizeTy, 2);
    PHINode * blockPendingDataPhi = nullptr;
    if (UnalignedLoads) {
        blockSourceOffsetPhi->addIncoming(ZERO, entry);
    } else {
        blockSourceOffsetPhi->addIncoming(initialSourceOffset, entry);
        blockPendingDataPhi = b.CreatePHI(elemVecTy, 2);
        blockPendingDataPhi->addIncoming(initialPendingData, entry);
    }

    Value * const nextBlock = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * const moreBlocksToDo = b.CreateICmpNE(nextBlock, numOfBlocks);

    Value * const maskVector = b.loadInputStreamBlock("mask", ZERO, blockNoPhi);
    Value * const metaMask = b.CreateZExtOrTrunc(b.hsimd_signmask(maskWidth, b.simd_any(maskWidth, maskVector)), metaMaskTy);

    // I/O pointers for the current block
    Value * const rawMaskPtr = b.getInputStreamBlockPtr("mask", ZERO, blockNoPhi);
    Value * const maskBasePtr = b.CreatePointerCast(rawMaskPtr, maskTy->getPointerTo());
    Value * const rawOutputPtr = b.getOutputStreamBlockPtr("spread", ZERO, blockNoPhi);
    Value * const outputPackPtr = b.CreatePointerCast(rawOutputPtr, elemVecTy->getPointerTo());

    // Store zeroes for the entire block, in case we skip packs with
    // an empty mask.
    Value * const zeroElemVec = b.fwCast(mElemWidth, b.allZeroes());
    for (unsigned i = 0; i < mElemWidth; i++) {
        b.storeOutputStreamPack("spread", ZERO, b.getSize(i), blockNoPhi, zeroElemVec);
    }

    b.CreateCondBr(b.CreateIsNull(metaMask), packsDone, scanPackLoop);

    b.SetInsertPoint(scanPackLoop);
    PHINode * const metaMaskPhi = b.CreatePHI(metaMaskTy, 2);
    metaMaskPhi->addIncoming(metaMask, blockAtATimeLoop);
    PHINode * const sourceOffsetPhi = b.CreatePHI(sizeTy, 2);
    sourceOffsetPhi->addIncoming(blockSourceOffsetPhi, blockAtATimeLoop);
    PHINode * pendingDataPhi = nullptr;
    if (!UnalignedLoads) {
        pendingDataPhi = b.CreatePHI(elemVecTy, 2);
        pendingDataPhi->addIncoming(blockPendingDataPhi, blockAtATimeLoop);
    }

    Value * const nextNonEmptyMaskNo = b.CreateZExtOrTrunc(b.CreateCountForwardZeroes(metaMaskPhi), sizeTy);
    Value * const mask = b.CreateLoad(maskTy, b.CreateGEP(maskTy, maskBasePtr, nextNonEmptyMaskNo));
    Value * const outputPtr = b.CreateGEP(elemVecTy, outputPackPtr, nextNonEmptyMaskNo);

    Value * const maskPopCount = b.CreateZExtOrTrunc(b.CreatePopcount(mask), sizeTy);
    Value * const updatedSourceOffset = b.CreateAdd(sourceOffsetPhi, maskPopCount);

    Value * newPack = nullptr;
    Value * spreadableData = nullptr;
    if (UnalignedLoads) {
        Value * const sourceItemPtr = b.CreateGEP(elemTy, sourcePtr, sourceOffsetPhi);
        Value * const packPtr = b.CreatePointerCast(sourceItemPtr, elemVecTy->getPointerTo());
        spreadableData = b.CreateAlignedLoad(elemVecTy, packPtr, 1);
    } else {
        Value * const pendingPackNo = b.CreateUDiv(sourceOffsetPhi, ELEMS_PER_PACK);
        // Elements in the pending pack may already have been processed.
        Value * const pendingElemsDone = b.CreateAnd(sourceOffsetPhi, UREM_MASK);
        // If the offset within the pack is nonzero, we have pending data elements to keep.
        Value * pendingElemsToKeep = b.CreateAnd(b.CreateSub(ELEMS_PER_PACK, pendingElemsDone), UREM_MASK);
        Value * const nextPackNo = b.CreateUDiv(updatedSourceOffset, ELEMS_PER_PACK);
        Value * const nextPackElems = b.CreateAnd(updatedSourceOffset, UREM_MASK);
        // We can only safely load a pack if our mask tells us that there are new elements to load.
        Value * const packToLoad = b.CreateSelect(b.CreateIsNull(nextPackElems), pendingPackNo, nextPackNo);
        Value * const packPtr = b.CreateGEP(elemVecTy, sourcePtr, packToLoad);
        newPack = b.CreateLoad(elemVecTy, packPtr);
        spreadableData = b.mvmd_dsll(mElemWidth, newPack, pendingDataPhi, pendingElemsToKeep);
    }
    
    Value * const spread = b.mvmd_expand(mElemWidth, spreadableData, mask);
    //Value * maskNo =  b.CreateAdd(b.CreateMul(blockNoPhi, b.getSize(8)), nextNonEmptyMaskNo);

    b.CreateStore(spread, outputPtr);

    Value * const nextMetaMask = b.CreateResetLowestBit(metaMaskPhi, "nextMetaMask");
    metaMaskPhi->addIncoming(nextMetaMask, scanPackLoop);
    sourceOffsetPhi->addIncoming(updatedSourceOffset, scanPackLoop);
    if (!UnalignedLoads) {
        pendingDataPhi->addIncoming(newPack, scanPackLoop);
    }
    b.CreateCondBr(b.CreateIsNull(nextMetaMask), packsDone, scanPackLoop);

    b.SetInsertPoint(packsDone);
    PHINode * const loopEndSourceOffsetPhi = b.CreatePHI(sizeTy, 2);
    loopEndSourceOffsetPhi->addIncoming(blockSourceOffsetPhi, blockAtATimeLoop);
    loopEndSourceOffsetPhi->addIncoming(updatedSourceOffset, scanPackLoop);
    PHINode * loopEndPendingDataPhi = nullptr;
    if (!UnalignedLoads) {
        loopEndPendingDataPhi = b.CreatePHI(elemVecTy, 2);
        loopEndPendingDataPhi->addIncoming(blockPendingDataPhi, blockAtATimeLoop);
        loopEndPendingDataPhi->addIncoming(newPack, scanPackLoop);
    }

    BasicBlock * const elemSpreadFinal = b.GetInsertBlock();
    blockNoPhi->addIncoming(nextBlock, elemSpreadFinal);
    blockSourceOffsetPhi->addIncoming(loopEndSourceOffsetPhi, elemSpreadFinal);
    if (!UnalignedLoads) {
        blockPendingDataPhi->addIncoming(loopEndPendingDataPhi, elemSpreadFinal);
    }
    b.CreateCondBr(moreBlocksToDo, blockAtATimeLoop, elemSpreadDone);

    b.SetInsertPoint(elemSpreadDone);
}

/*************************StreamMergeKernel*********************************/
constexpr unsigned StreamMergeStrideSize = 1;

StreamMergeKernel::StreamMergeKernel(LLVMTypeSystemInterface & ts,
                                       StreamSet * mask,
                                       StreamSet * source1,
                                       StreamSet * source2,
                                       StreamSet * merged,
                                       Scalar * base,
                                       const unsigned FieldWidth)
: MultiBlockKernel(ts, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "streamMerge"  << StreamMergeStrideSize << ':' << FieldWidth;
                        nm << '_' << source1->getFieldWidth() << ':' << merged->getNumElements();
                        nm.flush();
                        return tmp;
                    }(),
{Binding("marker", mask, FixedRate(1), Principal()),Binding("source1", source1, PopcountOf("marker")), Binding("source2", source2, PopcountOfNot("marker"))},
{Binding{"mergedStreamSet", merged}},
// input scalar
{Binding{"base", base}},
{}, {})
, mFieldWidth(FieldWidth)
, mSelectedStreamCount(merged->getNumElements()){
    setStride(StreamMergeStrideSize * ts.getBitBlockWidth());
}

void StreamMergeKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    Type * fieldWidthTy = b.getIntNTy(mFieldWidth);
    Type * sizeTy = b.getSizeTy();
    const unsigned numFields = b.getBitBlockWidth() / mFieldWidth;

    Constant * const ZERO = b.getSize(0);
    Constant * BLOCK_WIDTH = ConstantInt::get(sizeTy, b.getBitBlockWidth());
    Constant * FIELD_WIDTH = ConstantInt::get(sizeTy, mFieldWidth);
    Constant * fwSplat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth));
    Constant * fw_sub1Splat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth - 1));

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * mergeLoop = b.CreateBasicBlock("mergeLoop");
    BasicBlock * mergenceDone = b.CreateBasicBlock("mergenceDone");
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        ConstantInt * const mult = b.getSize(getStride() / b.getBitBlockWidth());
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }
    Value * processedSourceItems1 = b.getProcessedItemCount("source1");
    Value * processedSourceItems2 = b.getProcessedItemCount("source2");
    Value * initialSourceOffset1 = b.CreateURem(processedSourceItems1, BLOCK_WIDTH);
    Value * initialSourceOffset2 = b.CreateURem(processedSourceItems2, BLOCK_WIDTH);
    Value * const streamBase = b.getScalarField("base");

    SmallVector<Value *, 16> pendingData1(mSelectedStreamCount);
    SmallVector<Value *, 16> pendingData2(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        pendingData1[i] = b.loadInputStreamBlock("source1", streamOffset, ZERO);
        pendingData2[i] = b.loadInputStreamBlock("source2", streamOffset, ZERO);
    }

    b.CreateBr(mergeLoop);
    // Main Loop
    b.SetInsertPoint(mergeLoop);
    const unsigned incomingCount = 2;
    PHINode * blockNoPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi1 = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi2 = b.CreatePHI(b.getSizeTy(), incomingCount);
    SmallVector<PHINode *, 16> pendingDataPhi1(mSelectedStreamCount);
    SmallVector<PHINode *, 16> pendingDataPhi2(mSelectedStreamCount);
    blockNoPhi->addIncoming(ZERO, entry);
    pendingOffsetPhi1->addIncoming(initialSourceOffset1, entry);
    pendingOffsetPhi2->addIncoming(initialSourceOffset2, entry);

    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi1[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi1[i]->addIncoming(pendingData1[i], entry);
        pendingDataPhi2[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi2[i]->addIncoming(pendingData2[i], entry);
    }
    Value * mask1 = b.loadInputStreamBlock("marker", ZERO, blockNoPhi);
    Value * mask2 = b.CreateNot(mask1);
    Value * nextBlk = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

    Value * pendingOffset1 = b.CreateURem(pendingOffsetPhi1, BLOCK_WIDTH);
    Value * pendingOffset2 = b.CreateURem(pendingOffsetPhi2, BLOCK_WIDTH);

    Value * pendingItems1 = b.CreateSub(BLOCK_WIDTH, pendingOffset1);
    Value * pendingItems2 = b.CreateSub(BLOCK_WIDTH, pendingOffset2);

    Value * field_offset_lo1 = b.CreateCeilUDiv(pendingItems1, FIELD_WIDTH);
    Value * bit_offset1 = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset1, FIELD_WIDTH));

    Value * field_offset_lo2 = b.CreateCeilUDiv(pendingItems2, FIELD_WIDTH);
    Value * bit_offset2 = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset2, FIELD_WIDTH));

    Value * field_offset_hi1 =  b.CreateUDiv(pendingItems1, FIELD_WIDTH);
    Value * field_offset_hi2 =  b.CreateUDiv(pendingItems2, FIELD_WIDTH);

    Value * shift_fwd1 = b.CreateURem(b.CreateSub(fwSplat, bit_offset1), fwSplat);
    Value * shift_fwd2 = b.CreateURem(b.CreateSub(fwSplat, bit_offset2), fwSplat);

    Value * fieldPopCounts1 = b.simd_popcount(mFieldWidth, mask1);
    Value * fieldPopCounts2 = b.simd_popcount(mFieldWidth, mask2);


    Value * partialSum1 = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts1);
    Value * partialSum2 = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts2);
    Value * const blockPopCount1 = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum1, numFields - 1), sizeTy);
    Value * const blockPopCount2 = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum2, numFields - 1), sizeTy);
    partialSum1 = b.mvmd_slli(mFieldWidth, partialSum1, 1);
    partialSum2 = b.mvmd_slli(mFieldWidth, partialSum2, 1);

    Value * const source_field_lo1 = b.CreateUDiv(partialSum1, fwSplat);
    Value * const source_field_hi1 = b.CreateUDiv(b.CreateAdd(partialSum1, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo1 = b.CreateAnd(partialSum1, fw_sub1Splat);  // parallel URem
    Value * const source_shift_hi1 = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo1), fw_sub1Splat);

    Value * const source_field_lo2 = b.CreateUDiv(partialSum2, fwSplat);
    Value * const source_field_hi2 = b.CreateUDiv(b.CreateAdd(partialSum2, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo2 = b.CreateAnd(partialSum2, fw_sub1Splat);
    Value * const source_shift_hi2 = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo2), fw_sub1Splat);

    Value * const newPendingOffset1 = b.CreateAdd(pendingOffsetPhi1, blockPopCount1);
    Value * const srcBlockNo1 = b.CreateUDiv(newPendingOffset1, BLOCK_WIDTH);

    Value * const newPendingOffset2 = b.CreateAdd(pendingOffsetPhi2, blockPopCount2);
    Value * const srcBlockNo2 = b.CreateUDiv(newPendingOffset2, BLOCK_WIDTH);

    // Now load and process source streams.
    SmallVector<Value *, 16> sourceData1(mSelectedStreamCount);
    SmallVector<Value *, 16> sourceData2(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        sourceData1[i] = b.loadInputStreamBlock("source1", streamOffset, srcBlockNo1);
        sourceData2[i] = b.loadInputStreamBlock("source2", streamOffset, srcBlockNo2);
        Value * A1 = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData1[i], pendingDataPhi1[i], field_offset_lo1), bit_offset1);
        Value * B1 = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData1[i], pendingDataPhi1[i], field_offset_hi1), shift_fwd1);
        Value * full_source_block1 = b.CreateOr(A1, B1, "toExpand1");
        Value * C1 = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block1, source_field_lo1), source_shift_lo1);
        Value * D1 = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block1, source_field_hi1), source_shift_hi1);
        Value * expanded1 = b.bitCast(b.CreateOr(C1, D1, "expanded1"));

        Value * A2 = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData2[i], pendingDataPhi2[i], field_offset_lo2), bit_offset2);
        Value * B2 = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData2[i], pendingDataPhi2[i], field_offset_hi2), shift_fwd2);
        Value * full_source_block2 = b.CreateOr(A2, B2, "toExpand1");
        Value * C2 = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block2, source_field_lo2), source_shift_lo2);
        Value * D2 = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block2, source_field_hi2), source_shift_hi2);
        Value * expanded2 = b.bitCast(b.CreateOr(C2, D2, "expanded2"));

        Value * toMerge1 = b.simd_pdep(mFieldWidth, expanded1, mask1);
        Value * toMerge2 = b.simd_pdep(mFieldWidth, expanded2, mask2);
        Value * merged=b.CreateOr(toMerge1,toMerge2);//combine 2 inputs using bitwise OR
        b.storeOutputStreamBlock("mergedStreamSet", b.getInt32(i), blockNoPhi, merged);
    }
    //
    // Update loop control Phis for the next iteration.
    //
    blockNoPhi->addIncoming(nextBlk, b.GetInsertBlock());
    pendingOffsetPhi1->addIncoming(newPendingOffset1, b.GetInsertBlock());
    pendingOffsetPhi2->addIncoming(newPendingOffset2, b.GetInsertBlock());
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi1[i]->addIncoming(sourceData1[i], b.GetInsertBlock());
        pendingDataPhi2[i]->addIncoming(sourceData2[i], b.GetInsertBlock());
    }
    // Now continue the loop if there are more blocks to process.
    b.CreateCondBr(moreToDo, mergeLoop, mergenceDone);

    b.SetInsertPoint(mergenceDone);
}
/*******************************************************************************************/

FieldDepositKernel::FieldDepositKernel(LLVMTypeSystemInterface & ts
                                       , StreamSet * mask, StreamSet * input, StreamSet * output
                                       , const unsigned fieldWidth)
: MultiBlockKernel(ts, "FieldDeposit" + std::to_string(fieldWidth) + "_" + std::to_string(input->getNumElements()),
{Bind("depositMask", mask)
, Bind("inputStreamSet", input)},
{Bind("outputStreamSet", output)},
{}, {}, {})
, mFieldWidth(fieldWidth)
, mStreamCount(input->getNumElements()) {

}

void PDEPFieldDepositLogic(KernelBuilder & b, llvm::Value * const numOfStrides, unsigned fieldWidth, unsigned streamCount, unsigned stride);

void FieldDepositKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    if (b.hasFeature(IDISA::Feature::AVX_BMI2) && ((mFieldWidth == 32) || (mFieldWidth == 64))) {
        PDEPFieldDepositLogic(b, numOfStrides, mFieldWidth, mStreamCount, getStride());
    } else {
        BasicBlock * entry = b.GetInsertBlock();
        BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
        BasicBlock * done = b.CreateBasicBlock("done");
        Constant * const ZERO = b.getSize(0);
        Value * numOfBlocks = numOfStrides;
        if (getStride() != b.getBitBlockWidth()) {
            numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
        }
        b.CreateBr(processBlock);

        b.SetInsertPoint(processBlock);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        Value * depositMask = b.loadInputStreamBlock("depositMask", ZERO, blockOffsetPhi);
        for (unsigned j = 0; j < mStreamCount; ++j) {
            Value * input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j), blockOffsetPhi);
            Value * output = b.simd_pdep(mFieldWidth, input, depositMask);
            b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), blockOffsetPhi, output);
        }
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, processBlock);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
        b.CreateCondBr(moreToDo, processBlock, done);

        b.SetInsertPoint(done);
    }
}

//#define PREFER_FIELD_STORES_OVER_INSERT_ELEMENT

void PDEPFieldDepositLogic(KernelBuilder & b, llvm::Value * const numOfStrides, unsigned fieldWidth, unsigned streamCount, unsigned stride) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
    Type * fieldTy = b.getIntNTy(fieldWidth);
    Type * fieldPtrTy = PointerType::get(fieldTy, 0);
#elif PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
    Type * fieldTy = b.getIntNTy(fieldWidth);
    Type * fieldPtrTy = PointerType::get(fieldTy, 0);
#endif
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * done = b.CreateBasicBlock("done");
    Constant * const ZERO = b.getSize(0);
    const unsigned fieldsPerBlock = b.getBitBlockWidth()/fieldWidth;
    Value * numOfBlocks = numOfStrides;
    if (stride != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(stride/b.getBitBlockWidth())));
    }
    b.CreateBr(processBlock);
    b.SetInsertPoint(processBlock);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    SmallVector<Value *, 16> mask(fieldsPerBlock);
    //  When operating on fields individually, we can use vector load/store with
    //  extract/insert element operations, or we can use individual field load
    //  and stores.   Individual field operations require fewer total operations,
    //  but more memory instructions.   It may be that vector load/extract is better,
    //  while field store is better.   Vector insert then store creates long dependence
    //  chains.
    //
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
    Value * depositMaskPtr = b.getInputStreamBlockPtr("depositMask", ZERO, blockOffsetPhi);
    depositMaskPtr = b.CreatePointerCast(depositMaskPtr, fieldPtrTy);
    for (unsigned i = 0; i < fieldsPerBlock; i++) {
        mask[i] = b.CreateLoad(fieldTy, b.CreateGEP(fiedlTy, depositMaskPtr, b.getInt32(i)));
    }
#else

    Value * depositMask = b.fwCast(fieldWidth, b.loadInputStreamBlock("depositMask", ZERO, blockOffsetPhi));
    for (unsigned i = 0; i < fieldsPerBlock; i++) {
        mask[i] = b.CreateExtractElement(depositMask, b.getInt32(i));
    }
#endif
    for (unsigned j = 0; j < streamCount; ++j) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
        Value * inputPtr = b.getInputStreamBlockPtr("inputStreamSet", b.getInt32(j), blockOffsetPhi);
        inputPtr = b.CreatePointerCast(inputPtr, fieldPtrTy);
#else
        Value * const input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j), blockOffsetPhi);
        Value * inputStrm = b.fwCast(fieldWidth, input);
#endif
#ifdef PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
        Value * outputPtr = b.getOutputStreamBlockPtr("outputStreamSet", b.getInt32(j), blockOffsetPhi);
        outputPtr = b.CreatePointerCast(outputPtr, fieldPtrTy);
#else
        // Value * outputStrm = b.fwCast(mPDEPWidth, b.allZeroes());
        Value * outputStrm = UndefValue::get(b.fwVectorType(fieldWidth));
#endif
        for (unsigned i = 0; i < fieldsPerBlock; i++) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
            Value * field = b.CreateLoad(fieldTy, b.CreateGEP(fieldTy, inputPtr, b.getInt32(i)));
#else
            Value * field = b.CreateExtractElement(inputStrm, b.getInt32(i));
#endif
            Value * compressed = b.CreatePdeposit(field, mask[i]);
#ifdef PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
            b.CreateStore(compressed, b.CreateGEP(fieldTy, outputPtr, b.getInt32(i)));
        }
#else
        outputStrm = b.CreateInsertElement(outputStrm, compressed, b.getInt32(i));
    }
    b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), blockOffsetPhi, outputStrm);
#endif
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, processBlock, done);
    b.SetInsertPoint(done);
}

PDEPFieldDepositKernel::PDEPFieldDepositKernel(LLVMTypeSystemInterface & ts
                                               , StreamSet * mask, StreamSet * input, StreamSet * output
                                               , const unsigned fieldWidth)
: MultiBlockKernel(ts, "PDEPFieldDeposit" + std::to_string(fieldWidth) + "_" + std::to_string(input->getNumElements()) ,
                   {Binding{"depositMask", mask},
                    Binding{"inputStreamSet", input}},
                   {Binding{"outputStreamSet", output}},
                   {}, {}, {})
, mPDEPWidth(fieldWidth)
, mStreamCount(input->getNumElements()) {
    if ((fieldWidth != 32) && (fieldWidth != 64))
        llvm::report_fatal_error("Unsupported PDEP width for PDEPFieldDepositKernel");
}

void PDEPFieldDepositKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    PDEPFieldDepositLogic(b, numOfStrides, mPDEPWidth, mStreamCount, getStride());
}


PDEPkernel::PDEPkernel(LLVMTypeSystemInterface & ts, const unsigned swizzleFactor, std::string name)
: MultiBlockKernel(ts, std::move(name),
                   // input stream sets
{Binding{ts.getStreamSetTy(), "marker", FixedRate(), Principal()},
    Binding{ts.getStreamSetTy(swizzleFactor), "source", PopcountOf("marker"), BlockSize(ts.getBitBlockWidth() / swizzleFactor) }},
                   // output stream set
{Binding{ts.getStreamSetTy(swizzleFactor), "output", FixedRate(), BlockSize(ts.getBitBlockWidth() / swizzleFactor)}},
{}, {}, {})
, mSwizzleFactor(swizzleFactor) {
}

void PDEPkernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * const finishedStrides = b.CreateBasicBlock("finishedStrides");
    const auto pdepWidth = b.getBitBlockWidth() / mSwizzleFactor;
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    ConstantInt * const PDEP_WIDTH = b.getSize(pdepWidth);

    Constant * const ZERO = b.getSize(0);
    Value * const sourceItemCount = b.getProcessedItemCount("source");

    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
    }
    Value * const initialSourceOffset = b.CreateURem(sourceItemCount, BLOCK_WIDTH);
    b.CreateBr(processBlock);

    b.SetInsertPoint(processBlock);
    PHINode * const strideIndex = b.CreatePHI(b.getSizeTy(), 2);
    strideIndex->addIncoming(ZERO, entry);
    PHINode * const bufferPhi = b.CreatePHI(b.getBitBlockType(), 2);
    bufferPhi->addIncoming(Constant::getNullValue(b.getBitBlockType()), entry);
    PHINode * const sourceOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    sourceOffsetPhi->addIncoming(initialSourceOffset, entry);
    PHINode * const bufferSizePhi = b.CreatePHI(b.getSizeTy(), 2);
    bufferSizePhi->addIncoming(ZERO, entry);

    // Extract the values we will use in the main processing loop
    Value * const markerStream = b.getInputStreamBlockPtr("marker", ZERO, strideIndex);
    Type * const vecType = b.fwVectorType(pdepWidth);
    Value * const selectors = b.CreateBlockAlignedLoad(vecType, markerStream);
    Value * const numOfSelectors = b.simd_popcount(pdepWidth, selectors);

    // For each element of the marker block
    Value * bufferSize = bufferSizePhi;
    Value * sourceOffset = sourceOffsetPhi;
    Value * buffer = bufferPhi;
    for (unsigned i = 0; i < mSwizzleFactor; i++) {

        // How many bits will we deposit?
        Value * const required = b.CreateExtractElement(numOfSelectors, b.getSize(i));

        // Aggressively enqueue any additional bits
        BasicBlock * const entry = b.GetInsertBlock();
        BasicBlock * const enqueueBits = b.CreateBasicBlock();
        b.CreateBr(enqueueBits);

        b.SetInsertPoint(enqueueBits);
        PHINode * const updatedBufferSize = b.CreatePHI(bufferSize->getType(), 2);
        updatedBufferSize->addIncoming(bufferSize, entry);
        PHINode * const updatedSourceOffset = b.CreatePHI(sourceOffset->getType(), 2);
        updatedSourceOffset->addIncoming(sourceOffset, entry);
        PHINode * const updatedBuffer = b.CreatePHI(buffer->getType(), 2);
        updatedBuffer->addIncoming(buffer, entry);

        // Calculate the block and swizzle index of the current swizzle row
        Value * const blockOffset = b.CreateUDiv(updatedSourceOffset, BLOCK_WIDTH);
        Value * const swizzleIndex = b.CreateUDiv(b.CreateURem(updatedSourceOffset, BLOCK_WIDTH), PDEP_WIDTH);
        Value * const swizzle = b.CreateBlockAlignedLoad(b.getBitBlockType(), b.getInputStreamBlockPtr("source", swizzleIndex, blockOffset));
        Value * const swizzleOffset = b.CreateURem(updatedSourceOffset, PDEP_WIDTH);

        // Shift the swizzle to the right to clear off any used bits ...
        Value * const swizzleShift = b.simd_fill(pdepWidth, swizzleOffset);
        Value * const unreadBits = b.CreateLShr(swizzle, swizzleShift);

        // ... then to the left to align the bits with the buffer and combine them.
        Value * const bufferShift = b.simd_fill(pdepWidth, updatedBufferSize);
        Value * const pendingBits = b.CreateShl(unreadBits, bufferShift);

        buffer = b.CreateOr(updatedBuffer, pendingBits);
        updatedBuffer->addIncoming(buffer, enqueueBits);

        // Update the buffer size with the number of bits we have actually enqueued
        Value * const maxBufferSize = b.CreateAdd(b.CreateSub(PDEP_WIDTH, swizzleOffset), updatedBufferSize);
        bufferSize = b.CreateUMin(maxBufferSize, PDEP_WIDTH);
        updatedBufferSize->addIncoming(bufferSize, enqueueBits);

        // ... and increment the source offset by the number we actually inserted
        Value * const inserted = b.CreateSub(bufferSize, updatedBufferSize);
        sourceOffset = b.CreateAdd(updatedSourceOffset, inserted);
        updatedSourceOffset->addIncoming(sourceOffset, enqueueBits);

        // INVESTIGATE: we can branch at most once here. I'm not sure whether the potential
        // branch misprediction is better or worse than always filling from two swizzles to
        // ensure that we have enough bits to deposit.
        BasicBlock * const depositBits = b.CreateBasicBlock();
        b.CreateUnlikelyCondBr(b.CreateICmpULT(bufferSize, required), enqueueBits, depositBits);

        b.SetInsertPoint(depositBits);
        // Apply PDEP to each element of the combined swizzle using the current PDEP mask
        Value * const mask = b.CreateExtractElement(selectors, i);
        Value* result = b.simd_pdep(pdepWidth, buffer, b.simd_fill(pdepWidth, mask));
        // Store the result
        Value * const outputStreamPtr = b.getOutputStreamBlockPtr("output", b.getSize(i), strideIndex);
        b.CreateBlockAlignedStore(result, outputStreamPtr);
        // Shift away any used bits from the buffer and decrement our buffer size by the number we used
        Value * const usedShift = b.simd_fill(pdepWidth, required);
        buffer = b.CreateLShr(buffer, usedShift);
        bufferSize = b.CreateSub(bufferSize, required);
    }

    BasicBlock * const finishedBlock = b.GetInsertBlock();
    sourceOffsetPhi->addIncoming(sourceOffset, finishedBlock);
    bufferSizePhi->addIncoming(bufferSize, finishedBlock);
    bufferPhi->addIncoming(buffer, finishedBlock);
    Value * const nextStrideIndex = b.CreateAdd(strideIndex, b.getSize(1));
    strideIndex->addIncoming(nextStrideIndex, finishedBlock);
    b.CreateLikelyCondBr(b.CreateICmpNE(nextStrideIndex, numOfBlocks), processBlock, finishedStrides);

    b.SetInsertPoint(finishedStrides);
}

std::string InsertString(StreamSet * mask, InsertPosition p) {
    std::string s = std::to_string(mask->getNumElements()) + "x1_";
    return s + (p == InsertPosition::Before ? "Before" : "After");
}

class UnitInsertionExtractionMasks final : public BlockOrientedKernel {
public:
    UnitInsertionExtractionMasks(LLVMTypeSystemInterface & ts,
                                 StreamSet * insertion_mask, StreamSet * stream01, StreamSet * valid01, InsertPosition p = InsertPosition::Before)
    : BlockOrientedKernel(ts, "unitInsertionExtractionMasks" + InsertString(insertion_mask, p),
        {Binding{"insertion_mask", insertion_mask}},
        {Binding{"stream01", stream01, FixedRate(2)}, Binding{"valid01", valid01, FixedRate(2)}},
        {}, {},
        {InternalScalar{ScalarType::NonPersistent, ts.getBitBlockType(), "EOFmask"}}),
    mInsertPos(p) {}
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
    void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * const remainingBytes) override;
private:
    const InsertPosition mInsertPos;
};

void UnitInsertionExtractionMasks::generateDoBlockMethod(KernelBuilder & b) {
    Value * fileExtentMask = b.CreateNot(b.getScalarField("EOFmask"));
    Value * insertion_mask = b.loadInputStreamBlock("insertion_mask", b.getSize(0), b.getSize(0));
    const auto n = b.getInputStreamSet("insertion_mask")->getNumElements();
    for (unsigned i = 1; i < n; i++) {
        insertion_mask = b.CreateOr(insertion_mask, b.loadInputStreamBlock("insertion_mask", b.getSize(i), b.getSize(0)));
    }
    Constant * mask01 = nullptr;
    Value * extract_mask_lo = nullptr;
    Value * extract_mask_hi = nullptr;
    if (mInsertPos == InsertPosition::Before) {
        mask01 = b.simd_himask(2);
        extract_mask_lo = b.esimd_mergel(1, insertion_mask, fileExtentMask);
        extract_mask_hi = b.esimd_mergeh(1, insertion_mask, fileExtentMask);
    } else {
        mask01 = b.simd_lomask(2);
        extract_mask_lo = b.esimd_mergel(1, fileExtentMask, insertion_mask);
        extract_mask_hi = b.esimd_mergeh(1, fileExtentMask, insertion_mask);
    }
    b.storeOutputStreamBlock("stream01", b.getSize(0), b.getSize(0), mask01);
    b.storeOutputStreamBlock("stream01", b.getSize(0), b.getSize(1), mask01);
    b.storeOutputStreamBlock("valid01", b.getSize(0), b.getSize(0), extract_mask_lo);
    b.storeOutputStreamBlock("valid01", b.getSize(0), b.getSize(1), extract_mask_hi);
}

void UnitInsertionExtractionMasks::generateFinalBlockMethod(KernelBuilder & b, Value * const remainingBytes) {
    // Standard Pablo convention for final block processing: set a bit marking
    // the position just past EOF, as well as a mask marking all positions past EOF.
    b.setScalarField("EOFmask", b.bitblock_mask_from(remainingBytes));
    RepeatDoBlockLogic(b);
}

#define USE_FILTER_BY_MASK_KERNEL
void UnitInsertionSpreadMask(PipelineBuilder & P,
                             StreamSet * insertion_mask,
                             StreamSet * spread_mask,
                             InsertPosition p) {
    if (RecursiveSpreadMaskCalculation) {
        auto stream01 = P.CreateStreamSet(1);
        auto valid01 = P.CreateStreamSet(1);
        P.CreateKernelCall<UnitInsertionExtractionMasks>(insertion_mask, stream01, valid01, p);
#ifndef USE_FILTER_BY_MASK_KERNEL
        FilterByMask(P, valid01, stream01, spread_mask, spreadCountDensity);
#else
        P.CreateKernelCall<FilterByMaskKernel>
            (Select(valid01, {0}),
             SelectOperationList{Select(stream01, {0})},
             spread_mask, 64);
#endif
    } else {
        P.CreateKernelCall<UnitInsertionSpreadMaskKernel>(insertion_mask, spread_mask, p);
    }
}

UnitInsertionSpreadMaskKernel::UnitInsertionSpreadMaskKernel(LLVMTypeSystemInterface & ts,
                                 StreamSet * insertion_mask, StreamSet * spread_mask, InsertPosition p)
    : BlockOrientedKernel(ts, "UnitInsertionSpreadMaskKernel" + InsertString(insertion_mask, p),
        {Binding{"insertion_mask", insertion_mask}},
        {Binding{"spread_mask", spread_mask, BoundedRate(1, 2), EmptyWriteOverflow()}},
        {}, {},
        {InternalScalar{ScalarType::NonPersistent, ts.getBitBlockType(), "EOFmask"}}),
    mInsertPos(p) {}

void UnitInsertionSpreadMaskKernel::generateDoBlockMethod(KernelBuilder & b) {
    const unsigned packs_per_block = b.getBitBlockWidth()/pack_width;
    Type * packTy = b.getIntNTy(pack_width);
    Type * packPtrTy = packTy->getPointerTo();
    Type * sizeTy = b.getSizeTy();
    Constant * pONE = b.getIntN(pack_width, 1);
    Constant * PACK_BITS = b.getSize(pack_width);
    Value * fileExtentMask = b.CreateNot(b.getScalarField("EOFmask"));
    Value * insertion_mask = b.loadInputStreamBlock("insertion_mask", b.getSize(0), b.getSize(0));
    const auto n = b.getInputStreamSet("insertion_mask")->getNumElements();
    for (unsigned i = 1; i < n; i++) {
        insertion_mask = b.CreateOr(insertion_mask, b.loadInputStreamBlock("insertion_mask", b.getSize(i), b.getSize(0)));
    }
    Value * produced = b.getProducedItemCount("spread_mask");
    Value * offset = b.CreateURem(produced, PACK_BITS);
    Value * produced_base = b.CreateSub(produced, offset);
    Value * spread_mask_pack_ptr = b.getRawOutputPointer("spread_mask", produced_base);
    spread_mask_pack_ptr = b.CreatePointerCast(spread_mask_pack_ptr, packPtrTy);
    Value * initial_pending = b.CreateLoad(packTy, spread_mask_pack_ptr);
    Value * pending_bit_mask = b.CreateSub(b.CreateShl(pONE, b.CreateZExtOrTrunc(offset, packTy)), pONE);
    Value * pending = b.CreateAnd(initial_pending, pending_bit_mask);

    Constant * mask01 = nullptr;
    Value * extract_mask[2];
    if (mInsertPos == InsertPosition::Before) {
        mask01 = b.simd_himask(2);
        extract_mask[0] = b.esimd_mergel(1, insertion_mask, fileExtentMask);
        extract_mask[1] = b.esimd_mergeh(1, insertion_mask, fileExtentMask);
    } else {
        mask01 = b.simd_lomask(2);
        extract_mask[0] = b.esimd_mergel(1, fileExtentMask, insertion_mask);
        extract_mask[1] = b.esimd_mergeh(1, fileExtentMask, insertion_mask);
    }
    for (unsigned blk = 0; blk < 2; blk++) {
        extract_mask[blk] = b.fwCast(pack_width, extract_mask[blk]);
        Value * extracted = b.simd_pext(pack_width, mask01, extract_mask[blk]);
        for (unsigned i = 0; i < packs_per_block; i++) {
            Value * pack_mask = b.CreateExtractElement(extract_mask[blk], i);
            Value * newbits = b.CreatePopcount(pack_mask);
            Value * spread_pack = b.CreateExtractElement(extracted, i);
            pending = b.CreateOr(pending, b.CreateShl(spread_pack, offset));
            // We may have filled the pack; write it out in case we move on.
            b.CreateStore(pending, spread_mask_pack_ptr);
            Value * Rshift = b.CreateURem(b.CreateSub(PACK_BITS, offset), PACK_BITS);
            Value * pack_overflow = b.CreateLShr(spread_pack, Rshift);
            offset = b.CreateAdd(offset, newbits);
            produced = b.CreateAdd(produced, newbits);
            Value * pack_filled = b.CreateICmpUGE(offset, PACK_BITS);
            pending = b.CreateSelect(pack_filled, pack_overflow, pending);
            spread_mask_pack_ptr = b.CreateGEP(packTy, spread_mask_pack_ptr, b.CreateZExt(pack_filled, sizeTy));
            offset = b.CreateURem(produced, PACK_BITS);
        }
    }
    b.CreateStore(pending, spread_mask_pack_ptr);
    b.setProducedItemCount("spread_mask", produced);
}

void UnitInsertionSpreadMaskKernel::generateFinalBlockMethod(KernelBuilder & b, Value * const remainingBytes) {
    // Standard Pablo convention for final block processing: set a bit marking
    // the position just past EOF, as well as a mask marking all positions past EOF.
    b.setScalarField("EOFmask", b.bitblock_mask_from(remainingBytes));
    RepeatDoBlockLogic(b);
}

class SpreadMaskStep final : public pablo::PabloKernel {
public:
    SpreadMaskStep(LLVMTypeSystemInterface & ts,
                   StreamSet * bixnum, StreamSet * result, InsertPosition p = InsertPosition::Before) :
    pablo::PabloKernel(ts, "spreadMaskStep_" + InsertString(bixnum, p),
                {Bind("bixnum", bixnum, LookAhead(1))},
                {Bind("result", result)}), mInsertPos(p) {}
protected:
    void generatePabloMethod() override;
private:
    const InsertPosition mInsertPos;
};

void SpreadMaskStep::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum insert_counts = getInputStreamSet("bixnum");
    pablo::BixNum has_insert(1);
    has_insert[0] = bnc.UGT(insert_counts, 0);
    // Since we have inserted one position, subtract one from the insert count.
    pablo::BixNum remaining_to_insert = bnc.SubModular(insert_counts, has_insert);
    // Split the original insert counts into 2
    unsigned remaining_bits = insert_counts.size() - 1;
    pablo::BixNum divided_counts = bnc.HighBits(insert_counts, remaining_bits);
    pablo::BixNum divided_remaining = bnc.HighBits(remaining_to_insert, remaining_bits);
    for (unsigned i = 0; i < remaining_bits; i++) {
        std::string vname = "divided_counts[" + std::to_string(i) + "]";
        if (mInsertPos == InsertPosition::Before) {
            divided_counts[i] = pb.createOr(divided_remaining[i], pb.createLookahead(divided_counts[i], 1), vname);
        } else {
            divided_counts[i] = pb.createOr(divided_counts[i], pb.createAdvance(divided_remaining[i], 1), vname);
        }
    }
    pablo::Var * result = getOutputStreamVar("result");
    for (unsigned i = 0; i < remaining_bits; i++) {
        pb.createAssign(pb.createExtract(result, i), divided_counts[i]);
    }
}

void InsertionSpreadMask(PipelineBuilder & P,
                         StreamSet * bixNumInsertCount,
                         StreamSet * spread_mask,
                         InsertPosition pos) {
    unsigned steps = bixNumInsertCount->getNumElements();
    if (steps == 1) {
        UnitInsertionSpreadMask(P, bixNumInsertCount, spread_mask, pos);
    } else {
        if (RecursiveSpreadMaskCalculation) {
            /* Create a spread mask that adds one spread position for any position
             at which there is at least one item to insert.  */
            StreamSet * spread1_mask = P.CreateStreamSet(1);
            UnitInsertionSpreadMask(P, bixNumInsertCount, spread1_mask, pos);
            /* Spread out the counts so that there are two positions for each nonzero entry. */
            StreamSet * spread_counts = P.CreateStreamSet(steps);
            SpreadByMask(P, spread1_mask, bixNumInsertCount, spread_counts, false, 0);
            /* Divide the count at each original position equally into the
             two positions that were created by the unit spread process. */
            StreamSet * reduced_counts = P.CreateStreamSet(steps - 1);
            P.CreateKernelCall<SpreadMaskStep>(spread_counts, reduced_counts, pos);
            StreamSet * submask = P.CreateStreamSet(1);
            InsertionSpreadMask(P, reduced_counts, submask, pos);
            SpreadByMask(P, submask, spread1_mask, spread_mask, false, 0);
        } else {
            P.CreateKernelCall<InsertionSpreadMaskKernel>(bixNumInsertCount, spread_mask, pos);
        }
    }
}

InsertionSpreadMaskKernel::InsertionSpreadMaskKernel(LLVMTypeSystemInterface & ts,
                                                     StreamSet * bixNumInsertCount,
                                                     StreamSet * spread_mask,
                                                     InsertPosition p)
    : TwoLevelScanKernel(ts, "InsertionSpreadMaskKernel" + InsertString(bixNumInsertCount, p),
                         ScanWordWidth,
                         "bixNumInsertCount",
                         {LoopVar("bn_processed", ts.getSizeTy()),
                          LoopVar("sm_produced", ts.getSizeTy()),
                          LoopVar("sm_pending", ts.getIntNTy(ScanWordWidth))}),
      mBixBits(bixNumInsertCount->getNumElements()),
      mExpansionWidth(1U << mBixBits), mInsertPos(p) {
        mMasks.resize(mBixBits);
        mInputStreamSets.push_back(Binding{"bixNumInsertCount", bixNumInsertCount});
        mOutputStreamSets.push_back(Binding{"spread_mask", spread_mask, BoundedRate(1, mExpansionWidth), EmptyWriteOverflow()});
    }

void InsertionSpreadMaskKernel::initialize(KernelBuilder & b) {
    IntegerType * const scanWordTy = b.getIntNTy(ScanWordWidth);
    Constant * const SCANWORD_WIDTH = b.getSize(ScanWordWidth);
    Value * initialProcessed = b.getProcessedItemCount("bixNumInsertCount");
    Value * initialProduced = b.getProducedItemCount("spread_mask");
    Value * sm_offset = b.CreateURem(initialProduced, SCANWORD_WIDTH);
    Value * sm_base = b.CreateSub(initialProduced, sm_offset);
    Value * sm_base_ptr = b.getRawOutputPointer("spread_mask", sm_base);
    sm_base_ptr = b.CreatePointerCast(sm_base_ptr, scanWordTy->getPointerTo());
    Value * sm_initial = b.CreateLoad(scanWordTy, sm_base_ptr);
    Constant * sw_ONE = b.getIntN(ScanWordWidth, 1);
    Value * offset_mask = b.CreateSub(b.CreateShl(sw_ONE, b.CreateZExtOrTrunc(sm_offset, scanWordTy)), sw_ONE);
    Value * initialPending = b.CreateAnd(sm_initial, offset_mask);
    mLoopVarInitialValues.resize(3);
    mLoopVarInitialValues[bn_processed] = initialProcessed;
    mLoopVarInitialValues[sm_produced] = initialProduced;
    mLoopVarInitialValues[sm_pending] = initialPending;
}

void InsertionSpreadMaskKernel::wordPrologueLogic(KernelBuilder & b,
                                                  Value * absWordPos,
                                                  std::vector<Value *> indexWord,
                                                  std::vector<Value *> & loopVars) {
    IntegerType * sizeTy = b.getSizeTy();
    IntegerType * scanWordTy = b.getIntNTy(ScanWordWidth);
    Constant * ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    Constant * const SCANWORD_BITS = b.getSize(ScanWordWidth);
    Constant * const sw_ONES = ConstantInt::getAllOnesValue(scanWordTy);
    BasicBlock * indexWordsReady = b.GetInsertBlock();
    BasicBlock * fillLoop = b.CreateBasicBlock("fillLoop");
    BasicBlock * fillComplete = b.CreateBasicBlock("fillComplete");

    for (unsigned i = 0; i < mMasks.size(); i++) {
        mMasks[i] = indexWord[i];
    }
    Value * inputAdvance = b.CreateSub(absWordPos, loopVars[bn_processed]);
    Value * sm_word = b.CreateUDiv(loopVars[sm_produced], SCANWORD_BITS);
    Value * sm_offset = b.CreateURem(loopVars[sm_produced], SCANWORD_BITS);
    Value * sm_base = b.CreateSub(loopVars[sm_produced], sm_offset);
    Value * sm_word_ptr = b.getRawOutputPointer("spread_mask", sm_base);
    sm_word_ptr = b.CreatePointerCast(sm_word_ptr, scanWordTy->getPointerTo());

    Value * offset_mask = b.CreateSub(b.CreateShl(ONE, sm_offset), ONE);
    offset_mask = b.CreateZExtOrTrunc(offset_mask, scanWordTy);
    // We may fill the current pending scanword.
    Value * pending_filled = b.CreateOr(loopVars[sm_pending], b.CreateNot(offset_mask));
    // Calculate the final produced values upon updating of the
    // output mask with inputAdvance 1 bits.
    Value * updated_produced = b.CreateAdd(loopVars[sm_produced], inputAdvance);
    Value * final_word = b.CreateUDiv(updated_produced, SCANWORD_BITS);
    Value * final_offset = b.CreateURem(updated_produced, SCANWORD_BITS);
    Value * final_mask = b.CreateSub(b.CreateShl(ONE, final_offset), ONE);
    final_mask = b.CreateZExtOrTrunc(final_mask, scanWordTy);
    //
    Value * wordsToFill = b.CreateSub(final_word, sm_word);
    Value * doesNotFill = b.CreateICmpEQ(sm_word, final_word);
    pending_filled = b.CreateSelect(doesNotFill, b.CreateAnd(final_mask, pending_filled), pending_filled);
    b.CreateStore(pending_filled, sm_word_ptr);
    Value * final_pending = b.CreateSelect(doesNotFill, pending_filled, final_mask);
    
    b.CreateCondBr(doesNotFill, fillComplete, fillLoop);

    b.SetInsertPoint(fillLoop);
    PHINode * fillCounter = b.CreatePHI(sizeTy, 2);
    fillCounter->addIncoming(ZERO, indexWordsReady);
    Value * nextScanWord = b.CreateAdd(fillCounter, ONE);
    Value * atFinal = b.CreateICmpEQ(nextScanWord, wordsToFill);
    Value * toStore = b.CreateSelect(atFinal, final_pending, sw_ONES);
    b.CreateStore(toStore, b.CreateGEP(scanWordTy, sm_word_ptr, nextScanWord));
    fillCounter->addIncoming(nextScanWord, fillLoop);
    b.CreateCondBr(atFinal, fillComplete, fillLoop);

    b.SetInsertPoint(fillComplete);
    loopVars[bn_processed] = absWordPos;
    loopVars[sm_produced] = updated_produced;
    loopVars[sm_pending] = final_pending;
}

void InsertionSpreadMaskKernel::generateProcessingLogic(KernelBuilder & b,
                                                        Value * absItemPos,
                                                        std::vector<Value *> & loopVars) {
    IntegerType * sizeTy = b.getSizeTy();
    IntegerType * scanWordTy = b.getIntNTy(ScanWordWidth);
    Constant * ONE = b.getSize(1);
    Constant * const SCANWORD_BITS = b.getSize(ScanWordWidth);
    Constant * sw_ZERO = b.getIntN(ScanWordWidth, 0);
    Constant * sw_ONE = b.getIntN(ScanWordWidth, 1);
    Value * sm_offset = b.CreateURem(loopVars[sm_produced], SCANWORD_BITS);
    Value * sm_base = b.CreateSub(loopVars[sm_produced], sm_offset);
    Value * sm_word_ptr = b.getRawOutputPointer("spread_mask", sm_base);
    sm_word_ptr = b.CreatePointerCast(sm_word_ptr, scanWordTy->getPointerTo());
    // Determine how far we have advanced the input, and generate this many
    // 1 bits to the output spread mask.
    Value * advanceAmt = b.CreateSub(absItemPos, loopVars[bn_processed]);
    Value * advanceAmt_mask = b.CreateSub(b.CreateShl(ONE, advanceAmt), ONE);
    Value * advanced_mask = b.CreateZExtOrTrunc(b.CreateShl(advanceAmt_mask, sm_offset), scanWordTy);
    Value * pending_updated = b.CreateOr(loopVars[sm_pending], advanced_mask);
    // We may have filled the word; write it out in case we move on.
    b.CreateStore(pending_updated, sm_word_ptr);

    Value * producedAfterAdvance = b.CreateAdd(loopVars[sm_produced], advanceAmt);
    Value * sm_word_advance = b.CreateICmpUGE(b.CreateSub(producedAfterAdvance, sm_base), SCANWORD_BITS);
    Value * new_offset = b.CreateURem(producedAfterAdvance, SCANWORD_BITS);
    Value * new_mask = b.CreateSub(b.CreateShl(ONE, new_offset), ONE);
    Value * pendingAfterAdvance = b.CreateSelect(sm_word_advance, new_mask, pending_updated);
    sm_word_ptr = b.CreateSelect(sm_word_advance, b.CreateGEP(scanWordTy, sm_word_ptr, ONE), sm_word_ptr);

    Value * posInScanWord = b.CreateURem(absItemPos, SCANWORD_BITS);
    posInScanWord = b.CreateZExtOrTrunc(posInScanWord, scanWordTy);
    Value * insertAmt = b.getSize(0);
    for (unsigned i = 0; i < mBixBits; i++) {
        Value * theBit = b.CreateAnd(b.CreateLShr(mMasks[i], posInScanWord), sw_ONE);
        theBit = b.CreateZExtOrTrunc(theBit, sizeTy);
        insertAmt = b.CreateAdd(insertAmt, b.CreateShl(theBit, b.getSize(i)));
    }
    //  Depending on InsertPosition (Before or After), generate insertAmt zeroes
    //  plus a single 1 bit to the output spreadmask.
    Value * totalInsert = b.CreateAdd(insertAmt, ONE);
    Value * producedAfterInsert = b.CreateAdd(producedAfterAdvance, totalInsert);
    Value * pendingAfterInsert = nullptr;

    if (mInsertPos == InsertPosition::After) {
        // Generate a 1 bit followed by insertAmt zeroes.
        Value * sw_offset = b.CreateZExtOrTrunc(new_offset, scanWordTy);
        Value * newPending = b.CreateOr(pendingAfterAdvance, b.CreateShl(sw_ONE, sw_offset));
        // We may have filled the word; write it out in case we move on.
        b.CreateStore(newPending, sm_word_ptr);
        Value * updatedOffset = b.CreateAdd(totalInsert, new_offset);
        Value * pack_filled = b.CreateICmpUGE(updatedOffset, SCANWORD_BITS);
        pendingAfterInsert = b.CreateSelect(pack_filled, sw_ZERO, newPending);
    } else {
        // Generate insertAmt zeroes followed by a 1 bit.
        Value * bitOffset = b.CreateAdd(new_offset, insertAmt);
        Value * overflow = b.CreateICmpUGE(bitOffset, SCANWORD_BITS);
        bitOffset = b.CreateURem(bitOffset, SCANWORD_BITS);
        Value * placedBit = b.CreateShl(sw_ONE, b.CreateZExtOrTrunc(bitOffset, scanWordTy));
        Value * currentPending = b.CreateSelect(overflow, pendingAfterAdvance, b.CreateOr(pendingAfterAdvance, placedBit));
        // We may have filled the word; write it out in case we move on.
        b.CreateStore(currentPending, sm_word_ptr);
        Value * updatedOffset = b.CreateAdd(totalInsert, new_offset);
        Value * pack_filled = b.CreateICmpUGE(updatedOffset, SCANWORD_BITS);
        pendingAfterInsert = b.CreateSelect(pack_filled, placedBit, currentPending);
        Value * zero_pending = b.CreateXor(pack_filled, overflow);
        pendingAfterInsert = b.CreateSelect(zero_pending, sw_ZERO, pendingAfterInsert);
    }
    loopVars[bn_processed] = b.CreateAdd(absItemPos, ONE);
    loopVars[sm_produced] = producedAfterInsert;
    loopVars[sm_pending] = pendingAfterInsert;
}

void InsertionSpreadMaskKernel::finalize(KernelBuilder & b, std::vector<Value *> & loopVarFinalValues) {
    BasicBlock * finalizeSpread = b.GetInsertBlock();
    BasicBlock * finalFillLoop = b.CreateBasicBlock("finalFillLoop");
    BasicBlock * finalFillComplete = b.CreateBasicBlock("finalFillComplete");
    IntegerType * sizeTy = b.getSizeTy();
    IntegerType * scanWordTy = b.getIntNTy(ScanWordWidth);
    Constant * ZERO = b.getSize(0);
    Constant * ONE = b.getSize(1);
    Constant * const SCANWORD_BITS = b.getSize(ScanWordWidth);
    Constant * const sw_ONES = ConstantInt::getAllOnesValue(scanWordTy);

    Value * avail = b.getAvailableItemCount("bixNumInsertCount");
    Value * stillToProcess = b.CreateSub(avail, loopVarFinalValues[bn_processed]);
    // Need to append stillToProcess 1s to the spreadmask output.
    Value * sm_word = b.CreateUDiv(loopVarFinalValues[sm_produced], SCANWORD_BITS);
    Value * sm_offset = b.CreateURem(loopVarFinalValues[sm_produced], SCANWORD_BITS);
    Value * sm_base = b.CreateSub(loopVarFinalValues[sm_produced], sm_offset);
    Value * sm_word_ptr = b.getRawOutputPointer("spread_mask", sm_base);
    sm_word_ptr = b.CreatePointerCast(sm_word_ptr, scanWordTy->getPointerTo());

    Value * offset_mask = b.CreateSub(b.CreateShl(ONE, sm_offset), ONE);
    offset_mask = b.CreateZExtOrTrunc(offset_mask, scanWordTy);
    // We may fill the current pending scanword.
    Value * pending_filled = b.CreateOr(loopVarFinalValues[sm_pending], b.CreateNot(offset_mask));
    // Calculate the final produced values upon updating of the
    // output mask with inputAdvance 1 bits.
    Value * updated_produced = b.CreateAdd(loopVarFinalValues[sm_produced], stillToProcess);
    Value * final_word = b.CreateUDiv(updated_produced, SCANWORD_BITS);
    Value * final_offset = b.CreateURem(updated_produced, SCANWORD_BITS);
    Value * final_mask = b.CreateSub(b.CreateShl(ONE, final_offset), ONE);
    final_mask = b.CreateZExtOrTrunc(final_mask, scanWordTy);
    //
    Value * wordsToFill = b.CreateSub(final_word, sm_word);
    Value * doesNotFill = b.CreateICmpEQ(sm_word, final_word);
    pending_filled = b.CreateSelect(doesNotFill, b.CreateAnd(final_mask, pending_filled), pending_filled);
    b.CreateStore(pending_filled, sm_word_ptr);
    Value * final_pending = b.CreateSelect(doesNotFill, pending_filled, final_mask);
    
    b.CreateCondBr(doesNotFill, finalFillComplete, finalFillLoop);

    b.SetInsertPoint(finalFillLoop);
    PHINode * fillCounter = b.CreatePHI(sizeTy, 2);
    fillCounter->addIncoming(ZERO, finalizeSpread);
    Value * nextScanWord = b.CreateAdd(fillCounter, ONE);
    Value * atFinal = b.CreateICmpEQ(nextScanWord, wordsToFill);
    Value * toStore = b.CreateSelect(atFinal, final_pending, sw_ONES);
    b.CreateStore(toStore, b.CreateGEP(scanWordTy, sm_word_ptr, nextScanWord));
    fillCounter->addIncoming(nextScanWord, finalFillLoop);
    b.CreateCondBr(atFinal, finalFillComplete, finalFillLoop);

    b.SetInsertPoint(finalFillComplete);
    b.setProducedItemCount("spread_mask", updated_produced);
}

ByteCombine::ByteCombine(LLVMTypeSystemInterface & ts,
                     StreamSet * const byteStream1,
                     StreamSet * const byteStream2,
                     StreamSet * const outputBytes)
: MultiBlockKernel(ts, "ByteCombine"
, {Binding{"byteStream1", byteStream1}, Binding{"byteStream2", byteStream2}}
, {Binding{"outputBytes", outputBytes}}, {}, {}, {}) {
}

void ByteCombine::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * combineLoop = b.CreateBasicBlock("combineLoop");
    BasicBlock * combineDone = b.CreateBasicBlock("combineDone");
    Constant * const sz_ZERO = b.getSize(0);
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
    }
    b.CreateBr(combineLoop);

    b.SetInsertPoint(combineLoop);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(sz_ZERO, entry);
    for (unsigned i = 0; i < 8; i++) {
        Value * bytepack1 = b.loadInputStreamPack("byteStream1", sz_ZERO, b.getInt32(i), blockOffsetPhi);
        Value * bytepack2 = b.loadInputStreamPack("byteStream2", sz_ZERO, b.getInt32(i), blockOffsetPhi);
        Value * combined = b.CreateOr(bytepack1, bytepack2);
        b.storeOutputStreamPack("outputBytes", sz_ZERO, b.getInt32(i), blockOffsetPhi, combined);
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, combineLoop);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, combineLoop, combineDone);

    b.SetInsertPoint(combineDone);
}

ByteReplaceByMask::ByteReplaceByMask(LLVMTypeSystemInterface & b, StreamSet * mask, StreamSet * ToFill, StreamSet * Filler, StreamSet * output)
: MultiBlockKernel(b, "ByteReplaceByMask" + ToFill->shapeString(),
{Binding{"mask", mask}, Binding{"ToFill", ToFill}, Binding{"Filler", Filler, PopcountOf("mask")}},
{Binding{"output", output, FixedRate(), InOut("ToFill")}}, {}, {}, {}) {
    assert(mask->getNumElements() == 1);
    assert(mask->getFieldWidth() == 1);
    assert(ToFill->getFieldWidth() == ToFill->getFieldWidth());
    assert(ToFill->getFieldWidth() == output->getFieldWidth());
    assert(ToFill->getNumElements() == 1);
    assert(Filler->getNumElements() == 1);
    assert(output->getNumElements() == 1);
}

void ByteReplaceByMask::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packLoopShortCut = b.CreateBasicBlock("packLoopShortCut");
    BasicBlock * packLoopMain = b.CreateBasicBlock("packLoopMain");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const sz_ZERO = b.getSize(0);
    Value * initPos = b.getProcessedItemCount("Filler");
    const auto fieldWidth = getInputStreamSet(1)->getFieldWidth();
    FixedVectorType * dataVecTy = b.fwVectorType(fieldWidth);
    FixedVectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / fieldWidth), fieldWidth);

    b.CreateBr(packLoop);

    b.SetInsertPoint(packLoop);
    PHINode * toReadPosPhi = b.CreatePHI(b.getSizeTy(), 3);
    toReadPosPhi->addIncoming(initPos, entry);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 3);
    blockOffsetPhi->addIncoming(sz_ZERO, entry);
    //
    Value * toReadPos = toReadPosPhi;
    // Preparing next iteration.
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);
    // Load spread vector
    Value * spreadVec = b.loadInputStreamBlock("mask", sz_ZERO, blockOffsetPhi);
    spreadVec = b.CreateBitCast(spreadVec, popVecTy);
    Value * nullSpread = b.CreateNot(b.bitblock_any(spreadVec));
    b.CreateCondBr(nullSpread, packLoopShortCut, packLoopMain);

    b.SetInsertPoint(packLoopShortCut);
    toReadPosPhi->addIncoming(toReadPos, packLoopShortCut);
    blockOffsetPhi->addIncoming(nextBlk, packLoopShortCut);
    b.CreateCondBr(moreToDo, packLoop, packFinalize);

    b.SetInsertPoint(packLoopMain);
    // Output tracking
    for (unsigned i = 0; i < fieldWidth; ++i) {
        Value * spreadElem = b.CreateExtractElement(spreadVec, b.getInt32(i));
        Value * elementPopCount = b.CreatePopcount(spreadElem);

        // Get a pointer to the next unprocessed item
        Value * toReadPtr = b.getRawInputPointer("Filler", toReadPos);
        toReadPtr = b.CreatePointerCast(toReadPtr, dataVecTy->getPointerTo());
        Value * data = b.CreateAlignedLoad(dataVecTy, toReadPtr, 1);

        // Expand the loaded data
        Value * const expanded = b.CreateBitCast(b.mvmd_expand(fieldWidth, data, spreadElem), b.getBitBlockType());
        // Merge the expanded data in the i-th pack of the current stride
        Value * toFill = b.loadInputStreamPack("ToFill", sz_ZERO, b.getSize(i), blockOffsetPhi);
        b.storeOutputStreamPack("output", sz_ZERO, b.getSize(i), blockOffsetPhi, b.CreateOr(expanded, toFill));
        // Update the read position for the next pack
        toReadPos = b.CreateAdd(toReadPos, b.CreateZExt(elementPopCount, b.getSizeTy()));
    }
    // Finalize loop
    toReadPosPhi->addIncoming(toReadPos, packLoopMain);
    blockOffsetPhi->addIncoming(nextBlk, packLoopMain);
    b.CreateCondBr(moreToDo, packLoop, packFinalize);

    b.SetInsertPoint(packFinalize);
}

ByteSpreadByMaskKernel::ByteSpreadByMaskKernel(LLVMTypeSystemInterface & b, StreamSet * const byteStream, StreamSet * const spread, StreamSet * const output, Scalar * streamOffset)
: MultiBlockKernel(b, [&]() {
   std::string tmp;
   raw_string_ostream nm(tmp);
   nm << "ByteSpreadByMask" << output->getNumElements() << 'x' << byteStream->getFieldWidth();
   if (streamOffset) {
       nm << 'S';
   }
   nm.flush();
   return tmp;
}(),
{Binding{"spread", spread}, Binding{"byteStream", byteStream, PopcountOf("spread"), EmptyReadOverflow()}},
{Binding{"output", output}}, {}, {}, {}) {
    if (streamOffset) {
        mInputScalars.emplace_back("offset", streamOffset);
    }
    const auto fieldWidth = byteStream->getFieldWidth();
    if (LLVM_UNLIKELY(fieldWidth < 8)) {
        report_fatal_error(Twine{getName(), ": does not support field widths under 8-bits"});
    }
    if (LLVM_UNLIKELY(output->getFieldWidth() != fieldWidth)) {
        report_fatal_error(Twine{getName(), ": input field width does not match output field width"});
    }
    if (output->getNumElements() > byteStream->getNumElements()) {
        report_fatal_error(Twine{getName(), ": number of output streams exceeds the input streamset size"});
    }
}

void ByteSpreadByMaskKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    const auto fieldWidth = getInputStreamSet(1)->getFieldWidth();
    const auto numElements = getOutputStreamSet(0)->getNumElements();
    const auto numInputElements = getInputStreamSet(1)->getNumElements();

    Value * initPos = b.getProcessedItemCount("byteStream");

    SmallVector<Value *, 32> pending(numElements);

    ConstantInt * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
    const auto fieldsPerBlock = (b.getBitBlockWidth() / fieldWidth);
    ConstantInt * const BLOCK_WIDTH_MASK = b.getSize(b.getBitBlockWidth() - 1);
    ConstantInt * const FIELD_WIDTH_MASK = b.getSize(fieldWidth - 1);
    ConstantInt * const FIELDS_PER_BLOCK = b.getSize(fieldsPerBlock);
    ConstantInt * const FIELDS_PER_BLOCK_MASK = b.getSize(fieldsPerBlock - 1);

    // TODO: can we safely trust this stream if we read past the data even if we do not actually produce
    // anything from it?

    ConstantInt * const LOG_2_FIELDS_PER_BLOCK = b.getSize(floor_log2(fieldsPerBlock));

    FixedVectorType * dataVecTy = b.fwVectorType(fieldWidth); // FixedVectorType::get(b.getIntNTy(fieldWidth), b.getBitBlockWidth() / fieldWidth);
    FixedVectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / fieldWidth), fieldWidth);

    if (numElements == 1 && numInputElements == 1) {

        BasicBlock * packLoopShortCut = b.CreateBasicBlock("packLoopShortCut");
        BasicBlock * packLoopMain = b.CreateBasicBlock("packLoopMain");
        b.CreateBr(packLoop);

        b.SetInsertPoint(packLoop);
        PHINode * toReadPosPhi = b.CreatePHI(b.getSizeTy(), 3);
        toReadPosPhi->addIncoming(initPos, entry);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 3);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);
        //
        Value * toReadPos = toReadPosPhi;
        // Preparing next iteration.
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);
        // Load spread vector
        Value * spreadVec = b.loadInputStreamBlock("spread", sz_ZERO, blockOffsetPhi);
        spreadVec = b.CreateBitCast(spreadVec, popVecTy);
        Value * nullSpread = b.CreateNot(b.bitblock_any(spreadVec));
        b.CreateCondBr(nullSpread, packLoopShortCut, packLoopMain);

        b.SetInsertPoint(packLoopShortCut);
        for (unsigned i = 0; i < fieldWidth; ++i) {
            b.storeOutputStreamPack("output", sz_ZERO, b.getSize(i), blockOffsetPhi, b.allZeroes());
        }
        toReadPosPhi->addIncoming(toReadPos, packLoopShortCut);
        blockOffsetPhi->addIncoming(nextBlk, packLoopShortCut);
        b.CreateCondBr(moreToDo, packLoop, packFinalize);

        b.SetInsertPoint(packLoopMain);
        // Output tracking
        for (unsigned i = 0; i < fieldWidth; ++i) {
            Value * spreadElem = b.CreateExtractElement(spreadVec, b.getInt32(i));
            Value * elementPopCount = b.CreatePopcount(spreadElem);

            // Get a pointer to the next unprocessed item
            Value * toReadPtr = b.getRawInputPointer("byteStream", toReadPos);
            toReadPtr = b.CreatePointerCast(toReadPtr, dataVecTy->getPointerTo());
            Value * data = b.CreateAlignedLoad(dataVecTy, toReadPtr, 1);

            // Expand the loaded data
            Value * const expanded = b.CreateBitCast(b.mvmd_expand(fieldWidth, data, spreadElem), b.getBitBlockType());
            // Store the expanded data in the i-th pack of the current stride
            b.storeOutputStreamPack("output", sz_ZERO, b.getSize(i), blockOffsetPhi, expanded);
            // Update the write position for the next pack
            toReadPos = b.CreateAdd(toReadPos, b.CreateZExt(elementPopCount, b.getSizeTy()));
        }
        // Finalize loop
        toReadPosPhi->addIncoming(toReadPos, packLoopMain);
        blockOffsetPhi->addIncoming(nextBlk, packLoopMain);
        b.CreateCondBr(moreToDo, packLoop, packFinalize);

        b.SetInsertPoint(packFinalize);
    } else { // need to access these in a block-by-block manner
        Value * baseStreamIndex = nullptr;
        if (b.hasScalarField("offset")) {
            baseStreamIndex = b.getScalarField("offset");
            if (DebugOptionIsSet(codegen::EnableAsserts)) {
                Value * const maxElem = b.CreateAdd(baseStreamIndex, b.getSize(numElements));
                Value * valid = b.CreateICmpULE(maxElem, b.getSize(numInputElements));
                b.CreateAssert(valid, "%s: stream index plus output streamset size exceeds input streamset size",
                               b.GetString(getName()), baseStreamIndex, b.getSize(numElements), b.getSize(numInputElements));
            }
        }

        Value * initialPosition = b.CreateAnd(initPos, BLOCK_WIDTH_MASK);
        Value * initialPackIndex = b.CreateLShr(initialPosition, LOG_2_FIELDS_PER_BLOCK);
        Value * initialShift = b.CreateAnd(initPos, FIELDS_PER_BLOCK_MASK);

        for (unsigned i = 0; i < numElements; ++i) {
            Value * streamIndex = b.getSize(i);
            pending[i] = b.CreateBitCast(b.loadInputStreamPack("byteStream", streamIndex, initialPackIndex, sz_ZERO), dataVecTy);
            pending[i] = b.mvmd_srl(fieldWidth, pending[i], initialShift);
            assert (pending[i]->getType() == dataVecTy);
        }

        SmallVector<Value *, 32> spreadElem(fieldWidth);
        SmallVector<Value *, 32> popCount(fieldWidth + 2);
        SmallVector<Value *, 32> consumedCountAtField(fieldWidth + 1);
        SmallVector<Value *, 32> packIndex(fieldWidth);
        SmallVector<Value *, 32> blockIndex(fieldWidth);
        SmallVector<PHINode *, 32> pendingPhi(numElements);

        SmallVector<Value *, 32> numberOfUnconsumedDataUnits(fieldWidth);
        SmallVector<Value *, 32> someItemsAreUndeposited(fieldWidth);

        SmallVector<Value *, 32> unitsRemainingOfPriorLoadedBlock(fieldWidth);
        SmallVector<Value *, 32> preservePendingData(fieldWidth);

        Constant * ZERO_VEC = ConstantVector::getNullValue(dataVecTy);

        b.CreateBr(packLoop);

        b.SetInsertPoint(packLoop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);
        PHINode * shiftPhi = b.CreatePHI(b.getSizeTy(), 2);
        shiftPhi->addIncoming(initialShift, entry);
        PHINode * consumedPhi = b.CreatePHI(b.getSizeTy(), 2);
        consumedPhi->addIncoming(initialPosition, entry);
        for (unsigned i = 0; i < numElements; ++i) {
            PHINode * pPhi = b.CreatePHI(dataVecTy, 2);
            pPhi->addIncoming(pending[i], entry);
            pendingPhi[i] = pPhi;
            pending[i] = pPhi;
        }

        // Load spread vector
        Value * spreadVec = b.loadInputStreamBlock("spread", sz_ZERO, blockOffsetPhi);
        FixedVectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / fieldWidth), fieldWidth);
        spreadVec = b.CreateBitCast(spreadVec, popVecTy);

        // Output tracking
        popCount[0] = shiftPhi;
        consumedCountAtField[0] = consumedPhi;

        for (unsigned i = 0; i < numElements; ++i) {

            Value * streamIndex = b.getSize(i);

            pending[i] = pendingPhi[i];

            for (unsigned j = 0; j < fieldWidth; ++j) {

                if (i == 0) {
                    spreadElem[j] = b.CreateExtractElement(spreadVec, b.getInt32(j));
                    popCount[j + 1] = b.CreateZExt(b.CreatePopcount(spreadElem[j]), b.getSizeTy());
                    consumedCountAtField[j + 1] = b.CreateAdd(consumedCountAtField[j], popCount[j + 1]);
                    Value * const pos = b.CreateAdd(consumedCountAtField[j], FIELDS_PER_BLOCK_MASK);
                    packIndex[j] = b.CreateAnd(b.CreateLShr(pos, LOG_2_FIELDS_PER_BLOCK), FIELD_WIDTH_MASK);
                    blockIndex[j] = b.CreateLShr(pos, LOG_2_BLOCK_WIDTH);
                    numberOfUnconsumedDataUnits[j] = b.CreateAnd(b.CreateNeg(consumedCountAtField[j]), FIELDS_PER_BLOCK_MASK);
                    someItemsAreUndeposited[j] = b.CreateICmpNE(popCount[j + 1], FIELDS_PER_BLOCK);
                    unitsRemainingOfPriorLoadedBlock[j] = b.CreateSub(popCount[j + 1], numberOfUnconsumedDataUnits[j]);
                    Value * const consumedAllPriorData = b.CreateICmpUGT(popCount[j + 1], numberOfUnconsumedDataUnits[j]);
                    Value * const canReuseLoadedData = b.CreateICmpNE(numberOfUnconsumedDataUnits[j], sz_ZERO);
                    preservePendingData[j] = b.CreateAnd(consumedAllPriorData, canReuseLoadedData);
                }

                Value * const newData = b.CreateBitCast(b.loadInputStreamPack("byteStream", streamIndex, packIndex[j], blockIndex[j]), dataVecTy);

                Value * result = b.mvmd_sll(fieldWidth, newData, numberOfUnconsumedDataUnits[j]);

                assert (result->getType() == dataVecTy);
                assert (pending[i]->getType() == dataVecTy);

                result = b.CreateOr(result, pending[i]);

                // Expand the loaded data
                Value * const expanded = b.CreateBitCast(b.mvmd_expand(fieldWidth, result, spreadElem[j]), b.getBitBlockType());
                // Store the expanded data in the i-th pack of the current stride
                b.storeOutputStreamPack("output", streamIndex, b.getSize(j), blockOffsetPhi, expanded);
                Value * priorPending = b.mvmd_srl(fieldWidth, result, popCount[j + 1]);
                assert (priorPending->getType() == dataVecTy);
                priorPending = b.CreateSelect(someItemsAreUndeposited[j], priorPending, ZERO_VEC);
                Value * newPending = b.mvmd_srl(fieldWidth, newData, unitsRemainingOfPriorLoadedBlock[j]);
                assert (newPending->getType() == dataVecTy);
                newPending = b.CreateSelect(preservePendingData[j], newPending, ZERO_VEC);
                pending[i] = b.CreateOr(newPending, priorPending);
            }

            assert (pending[i]->getType() == dataVecTy);
            pendingPhi[i]->addIncoming(pending[i], packLoop);
        }

        // Finalize loop
        consumedPhi->addIncoming(consumedCountAtField[fieldWidth], packLoop);
        shiftPhi->addIncoming(popCount[fieldWidth], packLoop);

        Value * nextBlk = b.CreateAdd(blockOffsetPhi, sz_ONE);
        blockOffsetPhi->addIncoming(nextBlk, packLoop);

        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);
        b.CreateCondBr(moreToDo, packLoop, packFinalize);

        b.SetInsertPoint(packFinalize);
    }


}

}
