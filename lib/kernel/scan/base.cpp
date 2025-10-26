/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/scan/base.h>

#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Module.h>

#define IS_POW_2(i) ((i > 0) && ((i & (i - 1)) == 0))

using namespace llvm;

namespace kernel {

SingleStreamScanKernelTemplate::ScanWordContext::ScanWordContext(LLVMTypeSystemInterface & ts, unsigned strideWidth)
: width(std::max(minScanWordWidth, strideWidth / strideMaskWidth))
, wordsPerBlock(ts.getBitBlockWidth() / width)
, wordsPerStride(strideMaskWidth)
, fieldWidth(width)
, Ty(ts.getIntNTy(width))
, PointerTy(Ty->getPointerTo())
, StrideMaskTy(ts.getIntNTy(strideMaskWidth))
, WIDTH(ts.getSize(width))
, WORDS_PER_BLOCK(ts.getSize(wordsPerBlock))
, WORDS_PER_STRIDE(ts.getSize(wordsPerStride))
, NUM_BLOCKS_PER_STRIDE(ts.getSize(strideWidth / ts.getBitBlockWidth()))
{
    assert (IS_POW_2(strideWidth) && strideWidth >= ts.getBitBlockWidth() && strideWidth <= MaxStrideWidth);
}

void SingleStreamScanKernelTemplate::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    Type * const sizeTy = b.getSizeTy();
    Value * const sz_ZERO = b.getSize(0);
    Value * const sz_ONE = b.getSize(1);

    mEntryBlock = b.GetInsertBlock();
    mStrideStart = b.CreateBasicBlock("strideStart");
    mBuildStrideMask = b.CreateBasicBlock("buildStrideMask");
    mMaskReady = b.CreateBasicBlock("maskReady");
    mProcessMask = b.CreateBasicBlock("processMask");
    mProcessWord = b.CreateBasicBlock("processWord");
    mWordDone = b.CreateBasicBlock("wordDone");
    mStrideDone = b.CreateBasicBlock("strideDone");
    mExitBlock = b.CreateBasicBlock("exitBlock");
    initialize(b);
    Value * const scanStreamProcessedItemCount = b.getProcessedItemCount("scan");
    Value * const MASK_ZERO = Constant::getNullValue(mSW.StrideMaskTy);
    b.CreateBr(mStrideStart);

    b.SetInsertPoint(mStrideStart);
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2, "strideNo");
    strideNo->addIncoming(sz_ZERO, mEntryBlock);
    mStrideNo = strideNo;
    Value * const nextStrideNo = b.CreateAdd(strideNo, sz_ONE, "nextStrideNo");
    willProcessStride(b, strideNo);
    b.CreateBr(mBuildStrideMask);

    b.SetInsertPoint(mBuildStrideMask);
    PHINode * maskAccumPhi = b.CreatePHI(mSW.StrideMaskTy, 2);
    maskAccumPhi->addIncoming(MASK_ZERO, mStrideStart);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, mStrideStart);
    mBlockNo = blockNo;
    maskBuildingIterationHead(b);
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);
    blockNo->addIncoming(nextBlockNo, mBuildStrideMask);
    Value * const blockIndex = b.CreateAdd(blockNo, b.CreateMul(strideNo, mSW.NUM_BLOCKS_PER_STRIDE));
    maskBuildingIterationBody(b, blockIndex);
    Value * const block = b.loadInputStreamBlock("scan", b.getInt32(0), blockIndex);
    Value * const any = b.simd_any(mSW.fieldWidth, block);
    Value * const signMask = b.CreateZExt(b.hsimd_signmask(mSW.fieldWidth, any), mSW.StrideMaskTy);
    Value * const shiftedSignMask = b.CreateShl(signMask, b.CreateZExtOrTrunc(b.CreateMul(blockNo, mSW.WORDS_PER_BLOCK), mSW.StrideMaskTy));
    Value * const strideMask = b.CreateOr(maskAccumPhi, shiftedSignMask);
    maskAccumPhi->addIncoming(strideMask, mBuildStrideMask);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, mSW.NUM_BLOCKS_PER_STRIDE), mBuildStrideMask, mMaskReady);

    b.SetInsertPoint(mMaskReady);
    didBuildMask(b, strideMask);
    b.CreateLikelyCondBr(b.CreateICmpNE(strideMask, MASK_ZERO), mProcessMask, mStrideDone);

    b.SetInsertPoint(mProcessMask);
    PHINode * const processingMask = b.CreatePHI(mSW.StrideMaskTy, 2, "processingMask");
    mProcessingMask = processingMask;
    processingMask->addIncoming(strideMask, mMaskReady);
    Value * const wordOffset = b.CreateCountForwardZeroes(processingMask, "wordOffset", true);
    mWordOffset = wordOffset;
    Value * const strideOffset = b.CreateMul(strideNo, mSW.NUM_BLOCKS_PER_STRIDE);
    Value * const blockNumOfWord = b.CreateUDiv(wordOffset, mSW.WORDS_PER_BLOCK);
    Value * const blockOffset = b.CreateURem(wordOffset, mSW.WORDS_PER_BLOCK);
    Value * const processingBlockIndex = b.CreateAdd(strideOffset, blockNumOfWord);
    Value * const stridePtr = b.CreateBitCast(b.getInputStreamBlockPtr("scan", b.getInt32(0), processingBlockIndex), mSW.PointerTy);
    Value * const wordPtr = b.CreateGEP(mSW.Ty, stridePtr, blockOffset);
    Value * const word = b.CreateLoad(mSW.Ty, wordPtr);
    willProcessWord(b, word);
    b.CreateBr(mProcessWord);

    b.SetInsertPoint(mProcessWord);
    PHINode * const processingWord = b.CreatePHI(mSW.Ty, 2, "processingWord");
    processingWord->addIncoming(word, mProcessMask);
    mProcessingWord = processingWord;
    Value * const bitIndex_InWord = b.CreateZExt(b.CreateCountForwardZeroes(processingWord, "inWord", true), sizeTy);
    Value * const wordIndex_InStride = b.CreateMul(wordOffset, mSW.WIDTH);
    Value * const strideIndex = b.CreateAdd(scanStreamProcessedItemCount, b.CreateMul(strideNo, b.getSize(mStride)));
    Value * const absoluteWordIndex = b.CreateAdd(strideIndex, wordIndex_InStride);
    Value * const absoluteIndex = b.CreateAdd(absoluteWordIndex, bitIndex_InWord);

    generateProcessingLogic(b, absoluteIndex, processingBlockIndex, bitIndex_InWord);

    Value * const processedWord = b.CreateResetLowestBit(processingWord);
    processingWord->addIncoming(processedWord, mProcessWord);
    b.CreateCondBr(b.CreateICmpNE(processedWord, Constant::getNullValue(mSW.Ty)), mProcessWord, mWordDone);

    b.SetInsertPoint(mWordDone);
    didProcessWord(b);
    Value * const processedMask = b.CreateResetLowestBit(processingMask, "processedMask");
    processingMask->addIncoming(processedMask, mWordDone);
    b.CreateCondBr(b.CreateICmpNE(processedMask, MASK_ZERO), mProcessMask, mStrideDone);

    b.SetInsertPoint(mStrideDone);
    didProcessStride(b, strideNo);
    strideNo->addIncoming(nextStrideNo, mStrideDone);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), mStrideStart, mExitBlock);

    b.SetInsertPoint(mExitBlock);
    finalize(b);
}

const uint32_t SingleStreamScanKernelTemplate::MaxStrideWidth = 4096;

SingleStreamScanKernelTemplate::SingleStreamScanKernelTemplate(LLVMTypeSystemInterface & ts, std::string && name, StreamSet * scan)
: MultiBlockKernel(ts, name + "_sb" + std::to_string(codegen::ScanBlocks), {{"scan", scan}}, {}, {}, {}, {})
, mSW(ts, std::min(codegen::ScanBlocks * ts.getBitBlockWidth(), MaxStrideWidth))
{
    assert (scan->getNumElements() == 1 && scan->getFieldWidth() == 1);
    uint32_t strideWidth = std::min(codegen::ScanBlocks * ts.getBitBlockWidth(), MaxStrideWidth);
    if (!IS_POW_2(codegen::ScanBlocks)) {
        report_fatal_error("scan-blocks must be a power of 2");
    }
    if ((codegen::ScanBlocks * ts.getBitBlockWidth()) > MaxStrideWidth) {
        report_fatal_error(StringRef("scan-blocks exceeds maximum allowed size of ") + std::to_string(MaxStrideWidth / ts.getBitBlockWidth()));
    }
    setStride(strideWidth);
}

MultiStrideKernel::MultiStrideKernel(LLVMTypeSystemInterface & ts,
                                     std::string && kernel_name,
                                     unsigned maxStrideBlocks,
                                     std::vector<LoopVar> loopVars) :
    MultiBlockKernel(ts, kernel_name + std::to_string(maxStrideBlocks), {}, {}, {}, {}, {}),
    mMaxStrideBlocks(maxStrideBlocks), mLoopVars(loopVars) {}


void MultiStrideKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const unsigned blockWidth = b.getBitBlockWidth();
    const unsigned loopVariableCount = mLoopVars.size();
    Type * sizeTy = b.getSizeTy();
    Constant * const ZERO = b.getSize(0);
    Constant * const MAX_BLOCKS = b.getSize(mMaxStrideBlocks);

    Value * numOfBlocks = numOfStrides;
    if (getStride() != blockWidth) {
        assert ((getStride() % blockWidth) == 0);
        ConstantInt * const mult = b.getSize(getStride() / blockWidth);
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }
    BasicBlock * multiStrideLoop = b.CreateBasicBlock("multiStrideLoop");
    BasicBlock * multiStrideExit = b.CreateBasicBlock("multiStrideExit");

    // Perform initialization of global variables including the
    // initial values of any loop control variables required.
    // This is an overridden method that must be provided by
    // the concrete subclass.
    initialize(b);

    BasicBlock * loopPredecessor = b.GetInsertBlock();
    b.CreateBr(multiStrideLoop);

    b.SetInsertPoint(multiStrideLoop);
    PHINode * const priorBlocksDone = b.CreatePHI(sizeTy, 2);
    priorBlocksDone->addIncoming(ZERO, loopPredecessor);
    std::vector<PHINode *> loopVarPhi(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        loopVarPhi[i] = b.CreatePHI(mLoopVars[i].Ty, 2, mLoopVars[i].Name);
        loopVarPhi[i]->addIncoming(mLoopVarInitialValues[i], loopPredecessor);
    }

    Value * blocksRemaining = b.CreateSub(numOfBlocks, priorBlocksDone);
    Value * blocksToDo = b.CreateSelect(b.CreateICmpUGE(blocksRemaining, MAX_BLOCKS), MAX_BLOCKS, blocksRemaining);
    Value * finalBlocksDone = b.CreateAdd(priorBlocksDone, blocksToDo);

    std::vector<Value *> updatedLoopValues(loopVariableCount);
    strideLogic(b, priorBlocksDone, blocksToDo, loopVarPhi, updatedLoopValues);

    BasicBlock * multiStrideFinal = b.GetInsertBlock();
    priorBlocksDone->addIncoming(finalBlocksDone, multiStrideFinal);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        loopVarPhi[i]->addIncoming(updatedLoopValues[i], multiStrideFinal);
    }
    b.CreateCondBr(b.CreateICmpNE(finalBlocksDone, numOfBlocks), multiStrideLoop, multiStrideExit);
    b.SetInsertPoint(multiStrideExit);
    finalize(b, updatedLoopValues);
}

unsigned maxScanBlocks(LLVMTypeSystemInterface & ts, unsigned scanWordWidth) {
    unsigned size_t_bits = sizeof(size_t) * 8;
    unsigned scanWordsPerBlock = ts.getBitBlockWidth()/scanWordWidth;
    return size_t_bits / scanWordsPerBlock;
}

TwoLevelScanKernel::TwoLevelScanKernel(LLVMTypeSystemInterface & ts,
                                       std::string && kernel_name,
                                       unsigned scanWordWidth,
                                       std::string scanStreamName,
                                       std::vector<LoopVar> loopVars) :
    MultiStrideKernel(ts, kernel_name + "_" + std::to_string(scanWordWidth),
                      maxScanBlocks(ts, scanWordWidth), loopVars),
    mScanWordWidth(scanWordWidth), mScanStreamName(scanStreamName) {}

void TwoLevelScanKernel::strideLogic(KernelBuilder & b,
                                     Value * priorBlocksDone, Value * blocksToDo,
                                     std::vector<PHINode *> loopVarPhi,
                                     std::vector<Value *> & loopVarUpdates) {
    const unsigned streamCount = b.getInputStreamSet(mScanStreamName)->getNumElements();
    const unsigned loopVariableCount = mLoopVars.size();
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const scanWordTy = b.getIntNTy(mScanWordWidth);
    Constant * const ZERO = ConstantInt::getNullValue(sizeTy);
    Constant * const ONE = b.getSize(1);
    Constant * const SCANWORD_WIDTH = b.getSize(mScanWordWidth);
    Constant * const SCANWORDS_PER_BLOCK = b.getSize(b.getBitBlockWidth()/mScanWordWidth);
    Constant * BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());

    Value * baseProcessed = b.getProcessedItemCount(mScanStreamName);
    Value * priorProcessed = b.CreateAdd(b.CreateMul(priorBlocksDone, BLOCK_WIDTH), baseProcessed);

    BasicBlock * const metaMaskLoop = b.CreateBasicBlock("metaMaskLoop");
    BasicBlock * const scanWordLoop = b.CreateBasicBlock("scanWordLoop");
    BasicBlock * scanWordDone = b.CreateBasicBlock("scanWordDone");
    BasicBlock * strideLogicDone = b.CreateBasicBlock("strideLogicDone");

    std::vector<Value *> masks(streamCount);
    generateIndexComputation(b, priorBlocksDone, blocksToDo, masks);
    Value * metaMask = masks[0];
    for (unsigned i = 0; i < masks.size(); i++) {
        metaMask = b.CreateOr(metaMask, masks[i]);
    }
    BasicBlock * loopPredecessor = b.GetInsertBlock();
    b.CreateLikelyCondBr(b.CreateICmpNE(metaMask, ZERO), metaMaskLoop, strideLogicDone);

    b.SetInsertPoint(metaMaskLoop);
    PHINode * const remainingMaskPhi = b.CreatePHI(sizeTy, 2);
    remainingMaskPhi->addIncoming(metaMask, loopPredecessor);
    PHINode * const outerItemsProcessed = b.CreatePHI(sizeTy, 2);
    outerItemsProcessed->addIncoming(priorProcessed, loopPredecessor);
    std::vector<PHINode *> outerLoopPhi(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        outerLoopPhi[i] = b.CreatePHI(mLoopVars[i].Ty, 2);
        outerLoopPhi[i]->addIncoming(loopVarPhi[i], loopPredecessor);
    }

    Value * wordsToSkip = b.CreateCountForwardZeroes(remainingMaskPhi);
    Value * scanWordBlock = b.CreateAdd(priorBlocksDone, b.CreateUDiv(wordsToSkip, SCANWORDS_PER_BLOCK));
    Value * wordPosInBlock = b.CreateURem(wordsToSkip, SCANWORDS_PER_BLOCK);
    Value * wordBasePosition = b.CreateMul(wordsToSkip, SCANWORD_WIDTH);
    wordBasePosition = b.CreateAdd(priorProcessed, wordBasePosition);

    Value * scanWord = nullptr;
    std::vector<Value *> indexWord(streamCount);
    for (unsigned i = 0; i < streamCount; i++) {
        Value * base_ptr = b.getInputStreamBlockPtr(mScanStreamName, b.getSize(i), scanWordBlock);
        base_ptr = b.CreatePointerCast(base_ptr, scanWordTy->getPointerTo());
        indexWord[i] = b.CreateLoad(scanWordTy, b.CreateGEP(scanWordTy, base_ptr, wordPosInBlock));
        if (i == 0) {
            scanWord = indexWord[i];
        } else {
            scanWord = b.CreateOr(scanWord, indexWord[i]);
        }
    }
    std::vector<Value *> outerLoopVars(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        outerLoopVars[i] = outerLoopPhi[i];
    }
    wordPrologueLogic(b, wordBasePosition, indexWord, outerLoopVars);
    BasicBlock * metaMaskDone = b.GetInsertBlock();
    b.CreateBr(scanWordLoop);

    b.SetInsertPoint(scanWordLoop);
    PHINode * const remainingWordPhi = b.CreatePHI(scanWordTy, 2);
    remainingWordPhi->addIncoming(scanWord, metaMaskDone);
    PHINode * const priorItemsProcessed = b.CreatePHI(sizeTy, 2);
    priorItemsProcessed->addIncoming(outerItemsProcessed, metaMaskDone);
    std::vector<PHINode *> innerLoopPhi(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        innerLoopPhi[i] = b.CreatePHI(mLoopVars[i].Ty, 2);
        innerLoopPhi[i]->addIncoming(outerLoopVars[i], metaMaskDone);
    }

    Value * indexInWord = b.CreateZExtOrTrunc(b.CreateCountForwardZeroes(remainingWordPhi), sizeTy);
    Value * itemPos = b.CreateAdd(wordBasePosition, indexInWord);

    std::vector<Value *> innerLoopVars(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        innerLoopVars[i] = innerLoopPhi[i];
    }
    //
    // Generate the main processing logic method for this item,
    // passing in the current values of user loop variables,
    // anticipating that the user updates the loop variables
    // as required.
    generateProcessingLogic(b, itemPos, innerLoopVars);

    BasicBlock * itemDone = b.GetInsertBlock();
    Value * const scanWordNext = b.CreateResetLowestBit(remainingWordPhi);
    remainingWordPhi->addIncoming(scanWordNext, itemDone);
    Value * itemsProcessed = b.CreateAdd(itemPos, ONE);
    priorItemsProcessed->addIncoming(itemsProcessed, itemDone);

    for (unsigned i = 0; i < loopVariableCount; i++) {
        innerLoopPhi[i]->addIncoming(innerLoopVars[i], itemDone);
    }

    b.CreateCondBr(b.CreateICmpNE(scanWordNext, Constant::getNullValue(scanWordTy)), scanWordLoop, scanWordDone);
    b.SetInsertPoint(scanWordDone);

    Value * const nextMask = b.CreateResetLowestBit(remainingMaskPhi);
    remainingMaskPhi->addIncoming(nextMask, scanWordDone);
    outerItemsProcessed->addIncoming(itemPos, scanWordDone);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        outerLoopPhi[i]->addIncoming(innerLoopVars[i], scanWordDone);
    }

    b.CreateCondBr(b.CreateICmpNE(nextMask, ZERO), metaMaskLoop, strideLogicDone);

    b.SetInsertPoint(strideLogicDone);
    std::vector<PHINode *> strideDonePhi(loopVariableCount);
    for (unsigned i = 0; i < loopVariableCount; i++) {
        strideDonePhi[i] = b.CreatePHI(mLoopVars[i].Ty, 2);
        strideDonePhi[i]->addIncoming(loopVarPhi[i], loopPredecessor);
        strideDonePhi[i]->addIncoming(innerLoopVars[i], scanWordDone);
    }
    for (unsigned i = 0; i < loopVariableCount; i++) {
        loopVarUpdates[i] = strideDonePhi[i];
    }
}

void TwoLevelScanKernel::generateIndexComputation(KernelBuilder & b,
                                                  Value * blockOffset,
                                                  Value * blocksToDo,
                                                  std::vector<Value *> & masks) {
    IntegerType * sizeTy = b.getSizeTy();
    Constant * const ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    Constant * const SCANWORDS_PER_BLOCK = b.getSize(b.getBitBlockWidth()/mScanWordWidth);
    const unsigned streamCount = b.getInputStreamSet(mScanStreamName)->getNumElements();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * SIZE_BITS = b.getSize(sizeof(std::size_t) * 8);
        Value * IndexBitsRequired = b.CreateMul(blocksToDo, SCANWORDS_PER_BLOCK);
        b.CreateAssert(b.CreateICmpULE(IndexBitsRequired, SIZE_BITS), "TLSK::genIdx: index overflow");
    }
    BasicBlock * loopPredecessor = b.GetInsertBlock();
    BasicBlock * const indexComputationLoop = b.CreateBasicBlock("indexComputationLoop");
    BasicBlock * const indexComputationDone = b.CreateBasicBlock("indexComputationDone");
    b.CreateBr(indexComputationLoop);

    b.SetInsertPoint(indexComputationLoop);
    PHINode * const blockCounter = b.CreatePHI(sizeTy, 2);
    blockCounter->addIncoming(ZERO, loopPredecessor);
    std::vector<PHINode *> maskPhi(streamCount);
    for (unsigned i = 0; i < streamCount; i++) {
        maskPhi[i] = b.CreatePHI(sizeTy, 2);
        maskPhi[i]->addIncoming(ZERO, loopPredecessor);
    }

    Value * inputBlockIndex = b.CreateAdd(blockOffset, blockCounter);
    Value * const nextBlockNo = b.CreateAdd(blockCounter, ONE);
    blockCounter->addIncoming(nextBlockNo, indexComputationLoop);
    Value * wordCounter = b.CreateMul(blockCounter, SCANWORDS_PER_BLOCK);

    for (unsigned i = 0; i < streamCount; i++) {
        Value * s = b.loadInputStreamBlock(mScanStreamName, b.getSize(i), inputBlockIndex);
        Value * anyBitInField = b.simd_any(mScanWordWidth, s);
        Value * indexMask = b.CreateZExt(b.hsimd_signmask(mScanWordWidth, anyBitInField), sizeTy);
        masks[i] = b.CreateOr(maskPhi[i], b.CreateShl(indexMask, wordCounter));
        maskPhi[i]->addIncoming(masks[i], indexComputationLoop);
    }
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, blocksToDo), indexComputationLoop, indexComputationDone);
    b.SetInsertPoint(indexComputationDone);
}

} // namespace kernel
