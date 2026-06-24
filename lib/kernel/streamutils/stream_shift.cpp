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

const unsigned BITS_PER_BYTE = 8;
const unsigned SIZE_T_BITS = sizeof(size_t) * BITS_PER_BYTE;

IndexedShiftBack::IndexedShiftBack(LLVMTypeSystemInterface & ts, StreamSet * index, StreamSet * markers, StreamSet * shifted)
    : MultiBlockKernel(ts, "IndexedShiftBack" + markers->shapeString(),
// inputs
{Binding{"indexStream", index}, Binding{"markerStream", markers}},
// outputs
{Binding{"shiftResults", shifted, FixedRate(1), Deferred()}},
{}, {}, {}), mNumMarkerStreams(markers->getNumElements()) {
        //setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
    }

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
    Value * const avail = b.getAvailableItemCount("markerStream");
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const confirmedOutputPosition = b.CreatePHI(sizeTy, 2);
    confirmedOutputPosition->addIncoming(initialOutputPos, entryBlock);
    Value * const stridePos = b.CreateAdd(initialInputPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * const strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * const nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    //b.CallPrintInt("stridePos", stridePos);

    b.CreateBr(stridePrecomputation);
    // Precompute scanword masks.  For each scanword of the given stream:
    // index stream: the indexMask bit is 1 iff the scanword has at least one bit set.
    // marker stream:  lowMarkerMask is 1 iff there is a marker one bit at the lowest
    //                 index position in the scanword
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const indexMaskAccum = b.CreatePHI(sizeTy, 2);
    indexMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    std::vector<PHINode *> lowMarkerMaskAccum(mNumMarkerStreams);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        lowMarkerMaskAccum[i] = b.CreatePHI(sizeTy, 2);
        lowMarkerMaskAccum[i]->addIncoming(sz_ZERO, stridePrologue);
    }

    Value * const inputBlockOffset = b.CreateAdd(strideBlockOffset, blockNo);
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);
    Value * const indexBitBlock = b.loadInputStreamBlock("indexStream", sz_ZERO, inputBlockOffset);
    //b.CallPrintRegister("indexBitBlock", indexBitBlock);
    Value * const anyIndex = b.simd_any(sw.width, indexBitBlock);
    Value * const lowIndex = b.simd_and(indexBitBlock, b.simd_sub(sw.width, b.allZeroes(), indexBitBlock));
    Value * const indexWord = b.CreateZExt(b.hsimd_signmask(sw.width, anyIndex), sizeTy);
    Value * const scanWordPos = b.CreateMul(blockNo, sw.WORDS_PER_BLOCK);
    Value * const indexMask = b.CreateOr(indexMaskAccum, b.CreateShl(indexWord, scanWordPos), "indexMask");
    std::vector<Value *> lowMarkerMask(mNumMarkerStreams);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        Value * const markerBitBlock = b.loadInputStreamBlock("markerStream", b.getSize(i), inputBlockOffset);
        //b.CallPrintRegister("markerBitBlock", markerBitBlock);
        Value * const lowMarker = b.simd_any(sw.width, b.simd_and(lowIndex, markerBitBlock));
        Value * const lowMarkerWord = b.CreateZExt(b.hsimd_signmask(sw.width, lowMarker), sizeTy);
        Value * const lowMarkerShl = b.CreateShl(lowMarkerWord, scanWordPos);
        lowMarkerMask[i] = b.CreateOr(lowMarkerMaskAccum[i], lowMarkerShl, "lowMarkerMask" + std::to_string(i));
        lowMarkerMaskAccum[i]->addIncoming(lowMarkerMask[i], stridePrecomputation);
    }
    indexMaskAccum->addIncoming(indexMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    //  If the index mask is zero, we don't have any bits to select/write
    //  and the output doesn't advance.   Move on to the next stride.
    b.CreateUnlikelyCondBr(b.CreateIsNull(indexMask), emptyStride, nonemptyStride);

    b.SetInsertPoint(nonemptyStride);
    Value * const outputOffset = b.CreateAnd(confirmedOutputPosition, b.getSize(b.getBitBlockWidth() - 1));
    // Position of the prior index bit within its block.
    Value * const bitPosition = b.bitblock_set_bit(outputOffset);
    Value * const priorOutputBlock = b.CreateUDiv(confirmedOutputPosition, sz_BLOCKWIDTH);
    Value * const outputBlockOffset = b.CreateSub(priorOutputBlock, initialOutputBlock);
    //  Find the low markers of nonempty scan words.
    std::vector<Value *> highMarkerMask(mNumMarkerStreams);
    std::vector<Value *> updated(mNumMarkerStreams);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        Value * const extractedLowMarkers = b.CreatePextract(lowMarkerMask[i], indexMask);
        //  The low markers are shifted back to be deposited with the prior
        //  nonempty scanword.
        Value * const shiftBackLowMarkers = b.CreateLShr(extractedLowMarkers, sz_ONE);
        highMarkerMask[i] = b.CreatePdeposit(shiftBackLowMarkers, indexMask);
        //b.CallPrintInt("shiftBackLowMarkers", shiftBackLowMarkers);
        //b.CallPrintInt("highMarkerMask[i]", highMarkerMask[i]);
        //
        //  If there is a marker at the lowest index position in this stride,
        //  extract it to deposit at the recorded previous output position, if any.
        Value * const lowIndexPositionMarker = b.CreateTrunc(extractedLowMarkers, b.getInt1Ty());
        Value * const bitToUpdate = b.CreateSelect(lowIndexPositionMarker, bitPosition, b.allZeroes());
        Value * const priorWrittenPtr = b.getOutputStreamBlockPtr("shiftResults", b.getSize(i), outputBlockOffset);
        Value * const priorWritten = b.CreateBlockAlignedLoad(blockTy, priorWrittenPtr);
        updated[i] = b.simd_or(bitToUpdate, priorWritten);
    }
    Value * const inputBlock = b.CreateAdd(initialInputBlock, strideBlockOffset);
    Value * const catchupBlocks = b.CreateSub(inputBlock, priorOutputBlock);
    b.CreateCondBr(b.CreateIsNotNull(catchupBlocks), outputCatchupLoop, strideLoop);

    b.SetInsertPoint(outputCatchupLoop);
    PHINode * const catchUpBlockNo = b.CreatePHI(sizeTy, 2);
    catchUpBlockNo->addIncoming(sz_ZERO, nonemptyStride);
    std::vector<PHINode *> blockToWrite(mNumMarkerStreams);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        blockToWrite[i] = b.CreatePHI(blockTy, 2);
        blockToWrite[i]->addIncoming(updated[i], nonemptyStride);
    }
    Value * const outputBlockNo = b.CreateAdd(outputBlockOffset, catchUpBlockNo);
    Value * const nextCatchupBlock = b.CreateAdd(catchUpBlockNo, sz_ONE);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        b.storeOutputStreamBlock("shiftResults", b.getSize(i), outputBlockNo, blockToWrite[i]);
        blockToWrite[i]->addIncoming(b.allZeroes(), outputCatchupLoop);
    }
    catchUpBlockNo->addIncoming(nextCatchupBlock, outputCatchupLoop);
    b.CreateCondBr(b.CreateICmpNE(nextCatchupBlock, catchupBlocks), outputCatchupLoop, strideLoop);

    b.SetInsertPoint(strideLoop);
    PHINode * const blkNo = b.CreatePHI(sizeTy, 3);
    blkNo->addIncoming(sz_ZERO, nonemptyStride);
    blkNo->addIncoming(sz_ZERO, outputCatchupLoop);
    std::vector<PHINode *> highMarkerPhi(mNumMarkerStreams);
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        highMarkerPhi[i] = b.CreatePHI(sizeTy, 3);
        highMarkerPhi[i]->addIncoming(highMarkerMask[i], nonemptyStride);
        highMarkerPhi[i]->addIncoming(highMarkerMask[i], outputCatchupLoop);
    }

    Value * const inputBlockNo = b.CreateAdd(strideBlockOffset, blkNo);
    Value * const nextBlkNo = b.CreateAdd(blkNo, sz_ONE);
    Value * const outputBlock = b.CreateAdd(inputBlockNo, producedBlockOffset);
    Value * const indexBlock = b.loadInputStreamBlock("indexStream", sz_ZERO, inputBlockNo);
    Value * const indexBitCount = b.simd_popcount(sw.width, indexBitBlock);
    Value * const packedHighIndexPosition = b.simd_sub(sw.width, indexBitCount, b.simd_fill(sw.width, b.CreateZExtOrTrunc(sz_ONE, sw.Ty)));
    for (unsigned i = 0; i < mNumMarkerStreams; i++) {
        Value * const markerBlock = b.loadInputStreamBlock("markerStream", b.getSize(i), inputBlockNo);
        //b.CallPrintRegister("markerBlock", markerBlock);
        //  Pack the existing markers of each scanword corresponding to index positions.
        Value * const packedMarkers = b.simd_pext(sw.width, markerBlock, indexBlock);
        //b.CallPrintRegister("packedMarkers", packedMarkers);
        //  Identify the scanwords in this block that will receive new high markers.
        Value * const newHighMarkerGroup = b.CreateTrunc(highMarkerPhi[i], b.getIntNTy(b.getBitBlockWidth()/sw.width));
        // Align the new high markers in their scanwords.
        Value * const spreadHighMarkers = b.esimd_bitspread(sw.width, newHighMarkerGroup);
        //b.CallPrintRegister("spreadHighMarkers", spreadHighMarkers);
        // Move them into position
        Value * const newMarkersInPackedPosition = b.simd_sllv(sw.width, spreadHighMarkers, packedHighIndexPosition);
        //b.CallPrintRegister("newMarkersInPackedPosition", newMarkersInPackedPosition);
        Value * const newPackedMarkers = b.simd_or(b.simd_srli(sw.width, packedMarkers, 1), newMarkersInPackedPosition);
        //b.CallPrintRegister("newPackedMarkers", newPackedMarkers);
        Value * const shiftedMarkers = b.simd_pdep(sw.width, newPackedMarkers, indexBlock);
        //b.CallPrintRegister("shiftedMarkers", shiftedMarkers);
        b.storeOutputStreamBlock("shiftResults", b.getSize(i), outputBlock, shiftedMarkers);
        Value * const nextHighMarkerPhi = b.CreateLShr(highMarkerPhi[i], sw.WORDS_PER_BLOCK);
        highMarkerPhi[i]->addIncoming(nextHighMarkerPhi, strideLoop);
    }
    blkNo->addIncoming(nextBlkNo, strideLoop);
    b.CreateCondBr(b.CreateICmpNE(nextBlkNo, sz_BLOCKS_PER_STRIDE), strideLoop, strideFinalize);

    b.SetInsertPoint(strideFinalize);
    //  Determining the producedItemCount == the position prior to the last index bit
    Value * const indexStreamPtr = b.getInputStreamBlockPtr("indexStream", sz_ZERO, strideBlockOffset);
    Value * const indexWordBasePtr = b.CreateBitCast(indexStreamPtr, sw.pointerTy);
    //
    // Make sure that we are counting zeroes confined to the index width.
    Value * emptyWordsAtEnd = b.CreateCountReverseZeroes(b.CreateTrunc(indexMask, b.getIntNTy(sw.indexWidth)));
    emptyWordsAtEnd = b.CreateZExtOrTrunc(emptyWordsAtEnd, sizeTy);
    Value * const finalNonEmptyWordIndex = b.CreateSub(b.getSize(b.getBitBlockWidth()/sw.width - 1), emptyWordsAtEnd);
    Value * const lastIndexWord = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, indexWordBasePtr, finalNonEmptyWordIndex));
    Value * const emptyPosnsAtEnd = b.CreateZExtOrTrunc(b.CreateCountReverseZeroes(lastIndexWord), sizeTy);
    Value * const posInWord = b.CreateSub(b.getSize(sw.width - 1), emptyPosnsAtEnd);
    Value * const posInStride = b.CreateAdd(b.CreateMul(finalNonEmptyWordIndex, sw.WIDTH), posInWord);
    Value * const finalIndexPosition = b.CreateAdd(stridePos, posInStride);
    
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
    // The produced item position must be the location of the final index
    // or the previously confirmed output position until processing the final
    // stride.
    Value * finalProduced = b.CreateSelect(b.isFinal(), avail, producedItemPosition);
    //b.CallPrintInt("ISB produced items", finalProduced);
    b.setProducedItemCount("shiftResults", finalProduced);
}

}
