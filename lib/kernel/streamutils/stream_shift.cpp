/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/stream_shift.h>
#include <pablo/builder.hpp>
#include <kernel/core/kernel_builder.h>

using namespace llvm;
using namespace pablo;

namespace kernel {

ShiftForward::ShiftForward(LLVMTypeSystemInterface & ts, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount)
: PabloKernel(ts, "ShftFwd" + std::to_string(outputs->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"inputs", inputs}}, {Binding{"outputs", outputs}}),
mShiftAmount(shiftAmount)
{   assert(outputs->getNumElements() == inputs->getNumElements());
}

void ShiftForward::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");

    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createAdvance(sourceStreams[i], mShiftAmount));
    }
}

ShiftBack::ShiftBack(LLVMTypeSystemInterface & ts, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount)
: PabloKernel(ts, "ShftBack" + std::to_string(outputs->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"inputs", inputs, FixedRate(1), LookAhead(shiftAmount)}}, {Binding{"outputs", outputs}}),
mShiftAmount(shiftAmount)
{   assert(outputs->getNumElements() == inputs->getNumElements());
}

void ShiftBack::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");
    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createLookahead(sourceStreams[i], mShiftAmount, "shiftback_" + std::to_string(i)));
    }
}

IndexedAdvance::IndexedAdvance(LLVMTypeSystemInterface & ts, StreamSet * index, StreamSet * inputs, StreamSet * shifted, unsigned shiftAmount)
: PabloKernel(ts, "IndexedAdvance" + std::to_string(shifted->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"index", index}, Binding{"inputs", inputs}}, {Binding{"shifted", shifted}}),
mShiftAmount(shiftAmount)
{   assert(shifted->getNumElements() == inputs->getNumElements());
}

void IndexedAdvance::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");
    PabloAST * index = getInputStreamSet("index")[0];
    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createIndexedAdvance(sourceStreams[i], index, mShiftAmount));
    }
}

IndexedShiftBack::IndexedShiftBack(LLVMTypeSystemInterface & ts, StreamSet * index, StreamSet * markers, StreamSet * shifted)
    : MultiBlockKernel(ts, "IndexedShiftBack" + markers->shapeString(),
// inputs
{Binding{"indexStream", index}, Binding{"markerStream", markers}},
// outputs
{Binding{"shiftResults", shifted, FixedRate(1), Deferred()}},
{}, {}, {}) { 
        //setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS)
    }

const unsigned BITS_PER_BYTE = 8;
const unsigned SIZE_T_BITS = sizeof(size_t) * BITS_PER_BYTE;

struct ScanWordParameters {
    unsigned width;
    unsigned indexWidth;
    Type * const Ty;
    Type * const pointerTy;
    Constant * const WIDTH;
    Constant * const ix_MAXBIT;
    Constant * const WORDS_PER_BLOCK;
    Constant * const WORDS_PER_STRIDE;

    ScanWordParameters(KernelBuilder & b, unsigned stride) :
#ifdef PREFER_NARROW_SCANWIDTH
    width(std::max(BITS_PER_BYTE, stride/SIZE_T_BITS)),
#else
    width(std::min(SIZE_T_BITS, stride/BITS_PER_BYTE)),
#endif
    indexWidth(stride/width),
    Ty(b.getIntNTy(width)),
    pointerTy(Ty->getPointerTo()),
    WIDTH(b.getSize(width)),
    ix_MAXBIT(b.getSize(indexWidth - 1)),
    WORDS_PER_BLOCK(b.getSize(b.getBitBlockWidth()/width)),
    WORDS_PER_STRIDE(b.getSize(indexWidth))
    {   //  The stride must be a power of 2 and a multiple of the BitBlock width.
        assert((((stride & (stride - 1)) == 0) && (stride >= b.getBitBlockWidth()) && (stride <= SIZE_T_BITS * SIZE_T_BITS)));
    }
};

void IndexedShiftBack::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    ScanWordParameters sw(b, mStride);

    //llvm::errs() << "mStride = " << mStride << "\n";
    //llvm::errs() << "sw.width = " << sw.width << "\n";
    //llvm::errs() << "sw.indexWidth = " << sw.indexWidth << "\n";
    Constant * const sz_STRIDE = b.getSize(mStride);
    assert ((mStride % b.getBitBlockWidth()) == 0);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_BLOCKWIDTH = b.getSize(b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const nonemptyStride = b.CreateBasicBlock("nonemptyStride");
    BasicBlock * const outputCatchupLoop = b.CreateBasicBlock("outputCatchupLoop");
    BasicBlock * const strideLoop = b.CreateBasicBlock("strideLoop");
    BasicBlock * const strideFinalize = b.CreateBasicBlock("strideFinalize");
    BasicBlock * const emptyStride = b.CreateBasicBlock("emptyStride");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");

    Value * const initialInputPos = b.getProcessedItemCount("markerStream");
    Value * const initialOutputPos = b.getProducedItemCount("shiftResults");
    Value * const initialInputBlock = b.CreateUDiv(initialInputPos, sz_BLOCKWIDTH);
    Value * const initialOutputBlock = b.CreateUDiv(initialOutputPos, sz_BLOCKWIDTH);
    Value * const producedBlockOffset = b.CreateSub(initialInputBlock, initialOutputBlock);
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const confirmedOutputPosition = b.CreatePHI(sizeTy, 2);
    confirmedOutputPosition->addIncoming(initialOutputPos, entryBlock);
    Value * stridePos = b.CreateAdd(initialInputPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);

    b.CreateBr(stridePrecomputation);
    // Precompute scanword masks.  For each scanword of the given stream:
    // index stream: the indexMask bit is 1 iff the scanword has at least one bit set.
    // marker stream:  lowMarkerMask is 1 iff there is a marker one bit at the lowest
    //                 index position in the scanword
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const indexMaskAccum = b.CreatePHI(sizeTy, 2);
    indexMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const lowMarkerMaskAccum = b.CreatePHI(sizeTy, 2);
    lowMarkerMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);

    Value * inputBlockOffset = b.CreateAdd(strideBlockOffset, blockNo);
    Value * indexBitBlock = b.loadInputStreamBlock("indexStream", sz_ZERO, inputBlockOffset);
    Value * markerBitBlock = b.loadInputStreamBlock("markerStream", sz_ZERO, inputBlockOffset);
    //b.CallPrintRegister("indexBitBlock", indexBitBlock);
    //b.CallPrintRegister("markerBitBlock", markerBitBlock);
    Value * const anyIndex = b.simd_any(sw.width, indexBitBlock);
    Value * const lowIndex = b.simd_and(indexBitBlock, b.simd_sub(sw.width, b.allZeroes(), indexBitBlock));
    Value * const lowMarker = b.simd_any(sw.width, b.simd_and(lowIndex, markerBitBlock));
    Value * indexWord = b.CreateZExt(b.hsimd_signmask(sw.width, anyIndex), sizeTy);
    Value * lowMarkerWord = b.CreateZExt(b.hsimd_signmask(sw.width, lowMarker), sizeTy);
    Value * scanWordPos = b.CreateMul(blockNo, sw.WORDS_PER_BLOCK);
    Value * indexMask = b.CreateOr(indexMaskAccum, b.CreateShl(indexWord, scanWordPos), "indexMask");
    Value * lowMarkerMask = b.CreateOr(lowMarkerMaskAccum, b.CreateShl(lowMarkerWord, scanWordPos), "lowMarkerMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);

    indexMaskAccum->addIncoming(indexMask, stridePrecomputation);
    lowMarkerMaskAccum->addIncoming(lowMarkerMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    //  If the index mask is zero, we don't have any bits to select/write
    //  and the output doesn't advance.   Move on to the next stride.
    b.CreateUnlikelyCondBr(b.CreateIsNull(indexMask), emptyStride, nonemptyStride);

    b.SetInsertPoint(nonemptyStride);
    //  Find the low markers of nonempty scan words.
    //b.CallPrintInt("indexMask", indexMask);
    //b.CallPrintInt("lowMarkerMask", lowMarkerMask);
    Value * extractedLowMarkers = b.CreatePextract(lowMarkerMask, indexMask);
    //  The low markers are shifted back to be deposited with the prior
    //  nonempty scanword.
    Value * shiftBackLowMarkers = b.CreateLShr(extractedLowMarkers, sz_ONE);
    Value * highMarkerMask = b.CreatePdeposit(shiftBackLowMarkers, indexMask);
    //b.CallPrintInt("shiftBackLowMarkers", shiftBackLowMarkers);
    //
    //  If there is a marker at the lowest index position in this stride,
    //  extract it to deposit at the recorded previous output position, if any.
    Value * lowIndexPositionMarker = b.CreateTrunc(extractedLowMarkers, b.getInt1Ty());
    Value * outputOffset = b.CreateAnd(confirmedOutputPosition, b.getSize(b.getBitBlockWidth() - 1));
    Value * bitPosition = b.bitblock_set_bit(outputOffset);
    Value * bitToUpdate = b.CreateSelect(lowIndexPositionMarker, bitPosition, b.allZeroes());
    Value * priorOutputBlock = b.CreateUDiv(confirmedOutputPosition, sz_BLOCKWIDTH);
    Value * outputBlockOffset = b.CreateSub(priorOutputBlock, initialOutputBlock);
    Value * priorWrittenPtr = b.getOutputStreamBlockPtr("shiftResults", sz_ZERO, outputBlockOffset);
    Value * priorWritten = b.CreateBlockAlignedLoad(blockTy, priorWrittenPtr);
    Value * updated = b.simd_or(bitToUpdate, priorWritten);
    Value * inputBlock = b.CreateAdd(initialInputBlock, strideBlockOffset);
    Value * catchupBlocks = b.CreateSub(inputBlock, priorOutputBlock);

    b.CreateCondBr(b.CreateICmpULE(priorOutputBlock, inputBlock), outputCatchupLoop, strideLoop);

    b.SetInsertPoint(outputCatchupLoop);
    PHINode * const catchUpBlockNo = b.CreatePHI(sizeTy, 2);
    catchUpBlockNo->addIncoming(sz_ZERO, nonemptyStride);
    PHINode * const blockToWrite = b.CreatePHI(blockTy, 2);
    blockToWrite->addIncoming(updated, nonemptyStride);

    Value * outputBlockNo = b.CreateAdd(outputBlockOffset, catchUpBlockNo);
    b.storeOutputStreamBlock("shiftResults", sz_ZERO, outputBlockNo, blockToWrite);
    Value * nextCatchupBlock = b.CreateAdd(catchUpBlockNo, sz_ONE);

    catchUpBlockNo->addIncoming(nextCatchupBlock, outputCatchupLoop);
    blockToWrite->addIncoming(b.allZeroes(), outputCatchupLoop);
    b.CreateCondBr(b.CreateICmpNE(outputBlockNo, catchupBlocks), outputCatchupLoop, strideLoop);

    b.SetInsertPoint(strideLoop);
    PHINode * const blkNo = b.CreatePHI(sizeTy, 3);
    blkNo->addIncoming(sz_ZERO, nonemptyStride);
    blkNo->addIncoming(sz_ZERO, outputCatchupLoop);
    PHINode * const highMarkerPhi = b.CreatePHI(sizeTy, 3);
    highMarkerPhi->addIncoming(highMarkerMask, nonemptyStride);
    highMarkerPhi->addIncoming(highMarkerMask, outputCatchupLoop);

    Value * inputBlockNo = b.CreateAdd(strideBlockOffset, blkNo);
    Value * const indexBlock = b.loadInputStreamBlock("indexStream", sz_ZERO, inputBlockNo);
    Value * const indexBitCount = b.simd_popcount(sw.width, indexBitBlock);
    Value * markerBlock = b.loadInputStreamBlock("markerStream", sz_ZERO, inputBlockNo);
    //  Pack the existing markers of each scanword corresponding to index positions.
    Value * packedMarkers = b.simd_pext(sw.width, markerBlock, indexBlock);
    //b.CallPrintRegister("packedMarkers", packedMarkers);
    //  Identify the scanwords in this block that will receive new high markers.
    Value * newHighMarkerGroup = b.CreateTrunc(highMarkerPhi, b.getIntNTy(b.getBitBlockWidth()/sw.width));
    // Align the new high markers in their scanwords.
    Value * spreadHighMarkers = b.esimd_bitspread(sw.width, newHighMarkerGroup);
    //b.CallPrintRegister("spreadHighMarkers", spreadHighMarkers);
    // Move them into position
    Value * packedHighIndexPosition = b.simd_sub(sw.width, indexBitCount, b.simd_fill(sw.width, b.CreateZExtOrTrunc(sz_ONE, sw.Ty)));
    Value * newMarkersInPackedPosition = b.simd_sllv(sw.width, spreadHighMarkers, packedHighIndexPosition);
    //b.CallPrintRegister("newMarkersInPackedPosition", newMarkersInPackedPosition);
    Value * newPackedMarkers = b.simd_or(b.simd_srli(sw.width, packedMarkers, 1), newMarkersInPackedPosition);
    //b.CallPrintRegister("newPackedMarkers", newPackedMarkers);
    Value * shiftedMarkers = b.simd_pdep(sw.width, newPackedMarkers, indexBlock);
    //b.CallPrintRegister("shiftedMarkers", shiftedMarkers);
    Value * outputBlock = b.CreateAdd(inputBlockNo, producedBlockOffset);
    b.storeOutputStreamBlock("shiftResults", sz_ZERO, outputBlock, shiftedMarkers);
    
    // Update loop control variables.
    Value * const nextBlkNo = b.CreateAdd(blkNo, sz_ONE);
    blkNo->addIncoming(nextBlkNo, strideLoop);
    Value * nextHighMarkerPhi = b.CreateLShr(highMarkerPhi, sw.WORDS_PER_BLOCK);
    highMarkerPhi->addIncoming(nextHighMarkerPhi, strideLoop);
    b.CreateCondBr(b.CreateICmpNE(nextBlkNo, sz_BLOCKS_PER_STRIDE), strideLoop, strideFinalize);

    b.SetInsertPoint(strideFinalize);
    //  Determining the producedItemCount == the position prior to the last index bit
    Value * indexWordBasePtr = b.getInputStreamBlockPtr("indexStream", sz_ZERO, strideBlockOffset);
    indexWordBasePtr = b.CreateBitCast(indexWordBasePtr, sw.pointerTy);
    //
    // Make sure that we are counting zeroes confined to the index width.
    Value * emptyWordsAtEnd = b.CreateCountReverseZeroes(b.CreateTrunc(indexMask, b.getIntNTy(sw.indexWidth)));
    emptyWordsAtEnd = b.CreateZExtOrTrunc(emptyWordsAtEnd, sizeTy);
    Value * finalNonEmptyWordIndex = b.CreateSub(b.getSize(b.getBitBlockWidth()/sw.width - 1), emptyWordsAtEnd);
    Value * lastIndexWord = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, indexWordBasePtr, finalNonEmptyWordIndex));
    Value * emptyPosnsAtEnd = b.CreateZExtOrTrunc(b.CreateCountReverseZeroes(lastIndexWord), sizeTy);
    Value * posInWord = b.CreateSub(b.getSize(sw.width - 1), emptyPosnsAtEnd);
    Value * posInStride = b.CreateAdd(b.CreateMul(finalNonEmptyWordIndex, sw.WIDTH), posInWord);
    Value * finalIndexPosition = b.CreateAdd(stridePos, posInStride);
    
    strideNo->addIncoming(nextStrideNo, strideFinalize);
    confirmedOutputPosition->addIncoming(finalIndexPosition, strideFinalize);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);
    
    b.SetInsertPoint(emptyStride);
    strideNo->addIncoming(nextStrideNo, emptyStride);
    confirmedOutputPosition->addIncoming(confirmedOutputPosition, emptyStride);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    PHINode * const producedItemPosition = b.CreatePHI(sizeTy, 2);
    producedItemPosition->addIncoming(finalIndexPosition, strideFinalize);
    producedItemPosition->addIncoming(confirmedOutputPosition, emptyStride);
    b.setProducedItemCount("shiftResults", producedItemPosition);
}

}
