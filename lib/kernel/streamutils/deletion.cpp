/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/deletion.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/idisa_target.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <boost/intrusive/detail/math.hpp>
#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>

using boost::intrusive::detail::floor_log2;

using namespace llvm;

static cl::opt<bool> ElemFilter("ElemFilter", cl::desc("Use ElemFilter in place of byte filter by mask"), cl::init(true), cl::cat(codegen::CodeGenOptions));

inline size_t ceil_udiv(const size_t n, const size_t m) {
    return (n + m - 1) / m;
}

namespace kernel {

class ElemFilterKernel final  : public MultiBlockKernel {
public:
    ElemFilterKernel(LLVMTypeSystemInterface & ts,
                       StreamSet * mask,
                       StreamSet * source,
                       StreamSet * filtered);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const unsigned mElemWidth;
};


void FilterByMask(PipelineBuilder & P,
                  StreamSet * mask, StreamSet * inputs, StreamSet * outputs,
                  unsigned streamOffset,
                  unsigned extractionFieldWidth) {
    if (inputs->getFieldWidth() < 8) {
        StreamSet * const compressed = P.CreateStreamSet(outputs->getNumElements());
        std::vector<uint32_t> output_indices = streamutils::Range(streamOffset, streamOffset + outputs->getNumElements());
        P.CreateKernelCall<FieldCompressKernel>(Select(mask, {0}), SelectOperationList { Select(inputs, output_indices)}, compressed, extractionFieldWidth);
        P.CreateKernelCall<StreamCompressKernel>(mask, compressed, outputs, extractionFieldWidth);
    } else {
        Scalar * offset = nullptr;
        bool useElemFilter = ElemFilter;
        if (streamOffset || inputs->getNumElements() != outputs->getNumElements()) {
            offset = P.CreateConstant(P.getSize(streamOffset));
            useElemFilter = false;
        }
        if (useElemFilter) {
            P.CreateKernelCall<ElemFilterKernel>(mask, inputs, outputs);
        } else {
            P.CreateKernelCall<ByteFilterByMaskKernel>(inputs, mask, outputs, offset);
        }
    }
}

ElemFilterKernel::ElemFilterKernel(LLVMTypeSystemInterface & ts,
                                       StreamSet * mask,
                                       StreamSet * source,
                                       StreamSet * filtered)
: MultiBlockKernel(ts, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "elemFilter";
                        nm << '_' << source->getFieldWidth();
                        nm.flush();
                        return tmp;
                    }(),
{Binding("mask", mask, FixedRate(1), Principal()),
 Binding("source", source)},
{Binding{"filtered", filtered, PopcountOf("mask"), EmptyWriteOverflow()}},
{}, {}, {}), mElemWidth(source->getFieldWidth()) {
}
void ElemFilterKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    const unsigned maskWidth = b.getBitBlockWidth()/mElemWidth;
    IntegerType * const sizeTy = b.getSizeTy();
    IntegerType * const maskTy = b.getIntNTy(maskWidth);
    IntegerType * const metaMaskTy = b.getIntNTy(mElemWidth);
    Type * const elemVecTy = b.fwVectorType(mElemWidth);

    Constant * const ZERO = b.getSize(0);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const blockAtATimeLoop = b.CreateBasicBlock("blockAtATimeLoop");
    BasicBlock * const scanPackLoop = b.CreateBasicBlock("scanPackLoop");
    BasicBlock * const packsDone = b.CreateBasicBlock("packsDone");
    BasicBlock * const elemFilterDone = b.CreateBasicBlock("elemFilterDone");

    // Set up the filtered output pointer and initial offset data.
    Value * const initialOutputPos = b.getProducedItemCount("filtered");

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
    PHINode * const blockOutputPosPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOutputPosPhi->addIncoming(initialOutputPos, entry);

    Value * const nextBlock = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * const moreBlocksToDo = b.CreateICmpNE(nextBlock, numOfBlocks);

    Value * const maskVector = b.loadInputStreamBlock("mask", ZERO, blockNoPhi);
    //b.CallPrintRegister("maskVector", maskVector);
    Value * const metaMask = b.CreateZExtOrTrunc(b.hsimd_signmask(maskWidth, b.simd_any(maskWidth, maskVector)), metaMaskTy);
    //b.CallPrintInt("metaMask", metaMask);

    // Input pointers for the current block
    Value * const rawMaskPtr = b.getInputStreamBlockPtr("mask", ZERO, blockNoPhi);
    Value * const maskBasePtr = b.CreatePointerCast(rawMaskPtr, maskTy->getPointerTo());
    Value * const rawSourcePtr = b.getInputStreamBlockPtr("source", ZERO, blockNoPhi);
    Value * const sourcePackPtr = b.CreatePointerCast(rawSourcePtr, elemVecTy->getPointerTo());

    b.CreateCondBr(b.CreateIsNull(metaMask), packsDone, scanPackLoop);

    b.SetInsertPoint(scanPackLoop);
    PHINode * const metaMaskPhi = b.CreatePHI(metaMaskTy, 2);
    metaMaskPhi->addIncoming(metaMask, blockAtATimeLoop);
    PHINode * const outputPosPhi = b.CreatePHI(b.getSizeTy(), 2);
    outputPosPhi->addIncoming(blockOutputPosPhi, blockAtATimeLoop);

    Value * const nextNonEmptyMaskNo = b.CreateZExtOrTrunc(b.CreateCountForwardZeroes(metaMaskPhi), sizeTy);
    Value * const mask = b.CreateLoad(maskTy, b.CreateGEP(maskTy, maskBasePtr, nextNonEmptyMaskNo));
    //b.CallPrintInt("mask", mask);
    Value * const maskPopCount = b.CreateZExtOrTrunc(b.CreatePopcount(mask), sizeTy);
    //b.CallPrintInt("maskPopCount", maskPopCount);

    Value * const sourcePtr = b.CreateGEP(elemVecTy, sourcePackPtr, nextNonEmptyMaskNo);
    Value * const newPack = b.CreateLoad(elemVecTy, sourcePtr);

    Value * const compressed = b.mvmd_compress(mElemWidth, newPack, mask);
    //b.CallPrintRegister("compressed", compressed);

    Value * const ptr = b.getRawOutputPointer("filtered", outputPosPhi);
    Value * const toStorePtr = b.CreatePointerCast(ptr, elemVecTy->getPointerTo());
    b.CreateAlignedStore(compressed, toStorePtr, 1);

    Value * newOutputPos = b.CreateAdd(outputPosPhi, maskPopCount);

    Value * const nextMetaMask = b.CreateResetLowestBit(metaMaskPhi, "nextMetaMask");
    metaMaskPhi->addIncoming(nextMetaMask, scanPackLoop);
    outputPosPhi->addIncoming(newOutputPos, scanPackLoop);
    b.CreateCondBr(b.CreateIsNull(nextMetaMask), packsDone, scanPackLoop);

    b.SetInsertPoint(packsDone);
    PHINode * const loopEndOutputPosPhi = b.CreatePHI(sizeTy, 2);
    loopEndOutputPosPhi->addIncoming(blockOutputPosPhi, blockAtATimeLoop);
    loopEndOutputPosPhi->addIncoming(newOutputPos, scanPackLoop);

    BasicBlock * const elemFilterFinal = b.GetInsertBlock();
    blockNoPhi->addIncoming(nextBlock, elemFilterFinal);
    blockOutputPosPhi->addIncoming(loopEndOutputPosPhi, elemFilterFinal);
    b.CreateCondBr(moreBlocksToDo, blockAtATimeLoop, elemFilterDone);

    b.SetInsertPoint(elemFilterDone);
}

inline std::vector<Value *> parallel_prefix_deletion_masks(KernelBuilder & b, const unsigned fw, Value * del_mask) {
    Value * m = b.simd_not(del_mask);
    Value * mk = b.simd_slli(fw, del_mask, 1);
    std::vector<Value *> move_masks;
    for (unsigned shift = 1; shift < fw; shift *= 2) {
        Value * mp = mk;
        for (unsigned lookright = 1; lookright < fw; lookright *= 2) {
            mp = b.simd_xor(mp, b.simd_slli(fw, mp, lookright));
        }
        Value * mv = b.simd_and(mp, m);
        m = b.simd_or(b.simd_xor(m, mv), b.simd_srli(fw, mv, shift));
        mk = b.simd_and(mk, b.simd_not(mp));
        move_masks.push_back(mv);
    }
    return move_masks;
}

inline Value * apply_parallel_prefix_deletion(KernelBuilder & b, const unsigned fw, Value * del_mask, const std::vector<Value *> & mv, Value * strm) {
    Value * s = b.simd_and(strm, b.simd_not(del_mask));
    for (unsigned i = 0; i < mv.size(); i++) {
        unsigned shift = 1 << i;
        Value * t = b.simd_and(s, mv[i]);
        s = b.simd_or(b.simd_xor(s, t), b.simd_srli(fw, t, shift));
    }
    return s;
}

// Apply deletion to a set of stream_count input streams to produce a set of output streams.
// Kernel inputs: stream_count data streams plus one del_mask stream
// Outputs: the deleted streams, plus a partial sum popcount

void DeletionKernel::generateDoBlockMethod(KernelBuilder & b) {
    Value * delMask = b.loadInputStreamBlock("delMaskSet", b.getInt32(0));
    const auto move_masks = parallel_prefix_deletion_masks(b, mDeletionFieldWidth, delMask);
    for (unsigned j = 0; j < mStreamCount; ++j) {
        Value * input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j));
        Value * output = apply_parallel_prefix_deletion(b, mDeletionFieldWidth, delMask, move_masks, input);
        b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), output);
    }
    Value * unitCount = b.simd_popcount(mDeletionFieldWidth, b.simd_not(delMask));
    b.storeOutputStreamBlock("unitCounts", b.getInt32(0), b.bitCast(unitCount));
}

void DeletionKernel::generateFinalBlockMethod(KernelBuilder & b, Value * remainingBytes) {
    IntegerType * vecTy = b.getIntNTy(b.getBitBlockWidth());
    Value * remaining = b.CreateZExt(remainingBytes, vecTy);
    Value * EOF_del = b.bitCast(b.CreateShl(Constant::getAllOnesValue(vecTy), remaining));
    Value * delMask = b.CreateOr(EOF_del, b.loadInputStreamBlock("delMaskSet", b.getInt32(0)));
    const auto move_masks = parallel_prefix_deletion_masks(b, mDeletionFieldWidth, delMask);
    for (unsigned j = 0; j < mStreamCount; ++j) {
        Value * input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j));
        Value * output = apply_parallel_prefix_deletion(b, mDeletionFieldWidth, delMask, move_masks, input);
        b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), output);
    }
    Value * const unitCount = b.simd_popcount(mDeletionFieldWidth, b.simd_not(delMask));
    b.storeOutputStreamBlock("unitCounts", b.getInt32(0), b.bitCast(unitCount));
}

DeletionKernel::DeletionKernel(LLVMTypeSystemInterface & ts, StreamSet * input, StreamSet * delMask, StreamSet * output, StreamSet * unitCounts)
: BlockOrientedKernel(ts, "del" + std::to_string(ts.getBitBlockWidth()/output->getNumElements()) + "_" + std::to_string(output->getNumElements()),
{Binding{"inputStreamSet", input},
  Binding{"delMaskSet", delMask}},
{Binding{"outputStreamSet", output},
  Binding{"unitCounts", unitCounts, FixedRate(), RoundUpTo(ts.getBitBlockWidth())}},
{}, {}, {})
, mDeletionFieldWidth(ts.getBitBlockWidth() / output->getNumElements())
, mStreamCount(output->getNumElements()) {
}

void FieldCompressKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * done = b.CreateBasicBlock("done");
    Constant * const ZERO = b.getSize(0);
    assert (getStride() == b.getBitBlockWidth());
    b.CreateBr(processBlock);

    b.SetInsertPoint(processBlock);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    std::vector<Value *> maskVec = streamutils::loadInputSelectionsBlock(b, {mMaskOp}, blockOffsetPhi);
    std::vector<Value *> input = streamutils::loadInputSelectionsBlock(b, mInputOps, blockOffsetPhi);

    if (b.hasFeature(IDISA::Feature::AVX_BMI2)) {
        Type * fieldTy = b.getIntNTy(mFW);
        const unsigned fieldsPerBlock = b.getBitBlockWidth()/mFW;
        Value * extractionMask = b.fwCast(mFW, maskVec[0]);
        std::vector<Value *> mask(fieldsPerBlock);
        for (unsigned i = 0; i < fieldsPerBlock; i++) {
            mask[i] = b.CreateExtractElement(extractionMask, b.getInt32(i));
        }
        for (unsigned j = 0; j < input.size(); ++j) {
            Value * fieldVec = b.fwCast(mFW, input[j]);
            Value * output = UndefValue::get(extractionMask->getType());
            for (unsigned i = 0; i < fieldsPerBlock; i++) {
                Value * field = b.CreateExtractElement(fieldVec, b.getInt32(i));
                Value * compressed = b.CreatePextract(field, mask[i]);
                // Pextract is returning 32-bit integers but fieldTy is 16 bit?
                compressed = b.CreateZExtOrTrunc(compressed, fieldTy);
                output = b.CreateInsertElement(output, compressed, b.getInt32(i));
            }

            Value * outputPtr = b.getOutputStreamBlockPtr("outputStreamSet", b.getInt32(j), blockOffsetPhi);
            b.CreateStore(b.CreateBitCast(output, b.getBitBlockType()), outputPtr);

        }
    } else {
        std::vector<Value *> output = b.simd_pext(mFW, input, maskVec[0]);
        for (unsigned j = 0; j < output.size(); ++j) {
            b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), blockOffsetPhi, output[j]);
        }
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);
    b.CreateLikelyCondBr(moreToDo, processBlock, done);

    b.SetInsertPoint(done);
}

FieldCompressKernel::FieldCompressKernel(LLVMTypeSystemInterface & ts,
                                         SelectOperation const & maskOp,
                                         SelectOperationList const & inputOps,
                                         StreamSet * outputStreamSet,
                                         unsigned fieldWidth)
: MultiBlockKernel(ts, "fieldCompress" + std::to_string(fieldWidth) + "_" +
                   streamutils::genSignature(maskOp) +
                   ":" + streamutils::genSignature(inputOps),
{},
{Binding{"outputStreamSet", outputStreamSet}},
{}, {}, {})
, mFW(fieldWidth) {
    mMaskOp.operation = maskOp.operation;
    mMaskOp.bindings.push_back(std::make_pair("extractionMask", maskOp.bindings[0].second));
    // assert (streamutil::resultStreamCount(maskOp) == 1);
    mInputStreamSets.push_back({"extractionMask", maskOp.bindings[0].first, FixedRate(), Principal()});
    // assert (streamutil::resultStreamCount(inputOps) == outputStreamSet->getNumElements());
    std::unordered_map<StreamSet *, std::string> inputBindings;
    std::tie(mInputOps, inputBindings) = streamutils::mapOperationsToStreamNames(inputOps);
    for (auto const & kv : inputBindings) {
        mInputStreamSets.push_back({kv.second, kv.first, FixedRate(), ZeroExtended()});
    }
    // setStride(4 * b.getBitBlockWidth());
}

void PEXTFieldCompressKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    Type * fieldTy = b.getIntNTy(mPEXTWidth);
    Type * fieldPtrTy = PointerType::get(fieldTy, 0);
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * done = b.CreateBasicBlock("done");
    Constant * const ZERO = b.getSize(0);
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
    }
    const unsigned fieldsPerBlock = b.getBitBlockWidth()/mPEXTWidth;
    b.CreateBr(processBlock);

    b.SetInsertPoint(processBlock);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);
    std::vector<Value *> mask(fieldsPerBlock);
    Value * extractionMaskPtr = b.getInputStreamBlockPtr("extractionMask", ZERO, blockOffsetPhi);
    extractionMaskPtr = b.CreatePointerCast(extractionMaskPtr, fieldPtrTy);
    for (unsigned i = 0; i < fieldsPerBlock; i++) {
        mask[i] = b.CreateLoad(fieldTy, b.CreateGEP(fieldTy, extractionMaskPtr, b.getInt32(i)));
    }
    for (unsigned j = 0; j < mStreamCount; ++j) {
        Value * inputPtr = b.getInputStreamBlockPtr("inputStreamSet", b.getInt32(j), blockOffsetPhi);
        inputPtr = b.CreatePointerCast(inputPtr, fieldPtrTy);
        Value * outputPtr = b.getOutputStreamBlockPtr("outputStreamSet", b.getInt32(j), blockOffsetPhi);
        outputPtr = b.CreatePointerCast(outputPtr, fieldPtrTy);
        for (unsigned i = 0; i < fieldsPerBlock; i++) {
            Value * field = b.CreateLoad(fieldTy, b.CreateGEP(fieldTy, inputPtr, b.getInt32(i)));
            Value * compressed = b.CreatePextract(field, mask[i]);
            b.CreateStore(compressed, b.CreateGEP(fieldTy, outputPtr, b.getInt32(i)));
        }
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, processBlock, done);
    b.SetInsertPoint(done);
}

PEXTFieldCompressKernel::PEXTFieldCompressKernel(LLVMTypeSystemInterface & ts, const unsigned fieldWidth, const unsigned streamCount)
: MultiBlockKernel(ts, "PEXTfieldCompress" + std::to_string(fieldWidth) + "_" + std::to_string(streamCount),
                   {Binding{ts.getStreamSetTy(streamCount), "inputStreamSet"},
                       Binding{ts.getStreamSetTy(), "extractionMask"}},
                   {Binding{ts.getStreamSetTy(streamCount), "outputStreamSet"}},
                   {}, {}, {})
, mPEXTWidth(fieldWidth)
, mStreamCount(streamCount) {
    if ((fieldWidth != 32) && (fieldWidth != 64)) llvm::report_fatal_error("Unsupported PEXT width for PEXTFieldCompressKernel");
}

StreamCompressKernel::StreamCompressKernel(LLVMTypeSystemInterface & ts
                                           , StreamSet * extractionMask
                                           , StreamSet * source
                                           , StreamSet * compressedOutput
                                           , const unsigned FieldWidth)
: MultiBlockKernel(ts, "streamCompress" + std::to_string(FieldWidth) + "_" + std::to_string(source->getNumElements()),
{Bind("extractionMask", extractionMask, Principal()),
 Bind("sourceStreamSet", source, ZeroExtended())},
{Bind("compressedOutput", compressedOutput, PopcountOf("extractionMask"), EmptyWriteOverflow(), MaximumDistribution())},
{}, {}, {})
, mFW(FieldWidth)
, mStreamCount(source->getNumElements()) {
    for (unsigned i = 0; i < mStreamCount; i++) {
        addInternalScalar(ts.getBitBlockType(), "pendingOutputBlock_" + std::to_string(i));
    }
    setStride(4 * ts.getBitBlockWidth());
}

void StreamCompressKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    IntegerType * const fwTy = b.getIntNTy(mFW);
    IntegerType * const sizeTy = b.getSizeTy();

    const unsigned numFields = b.getBitBlockWidth() / mFW;
    const unsigned strideBlocks = getStride()/b.getBitBlockWidth();

    Constant * zeroSplat = Constant::getNullValue(b.fwVectorType(mFW));
    Constant * oneSplat = b.getSplat(numFields, ConstantInt::get(fwTy, 1));
    Constant * CFW = ConstantInt::get(fwTy, mFW);
    Constant * numFieldConst = ConstantInt::get(fwTy, numFields);
    Constant * fwMaskSplat = b.getSplat(numFields, ConstantInt::get(fwTy, mFW - 1));
    Constant * intSequence = b.getConstantVectorSequence(mFW, 0, numFields - 1, 1);

    Constant * BLOCK_WIDTH = ConstantInt::get(fwTy, b.getBitBlockWidth());
    Constant * BLOCK_MASK = ConstantInt::get(fwTy, b.getBitBlockWidth() - 1);

    BasicBlock * const segmentLoop = b.CreateBasicBlock("segmentLoop");
    BasicBlock * const segmentDone = b.CreateBasicBlock("segmentDone");
    BasicBlock * const writePartialData = b.CreateBasicBlock("writePartialData");
    BasicBlock * const segmentExit = b.CreateBasicBlock("segmentExit");

    Constant * const ZERO = ConstantInt::get(sizeTy, 0);
    Constant * const ONE = ConstantInt::get(sizeTy, 1);
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
    }

    Value * const produced = b.getProducedItemCount("compressedOutput");
    Value * const pendingItemCount = b.CreateAnd(b.CreateZExtOrTrunc(produced, fwTy), BLOCK_MASK);

    // TODO: we could make this kernel stateless but would have to use "bitblock_mask_to" on the
    // input to account for the fact the pipeline doesn't zero out output stream data when it
    // reuses it.

    SmallVector<Value *, 16> pendingData(mStreamCount);
    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingData[i] = b.getScalarField("pendingOutputBlock_" + std::to_string(i));
    }

    BasicBlock * const entry = b.GetInsertBlock();
    b.CreateBr(segmentLoop);
    // Main Loop
    b.SetInsertPoint(segmentLoop);
    PHINode * const inputOffsetPhi = b.CreatePHI(sizeTy, 2);
    PHINode * const outputOffsetPhi = b.CreatePHI(sizeTy, 2);
    PHINode * const pendingItemsPhi = b.CreatePHI(fwTy, 2);
    SmallVector<PHINode *, 16> pendingDataPhi(mStreamCount);

    inputOffsetPhi->addIncoming(ZERO, entry);
    outputOffsetPhi->addIncoming(ZERO, entry);
    pendingItemsPhi->addIncoming(pendingItemCount, entry);
    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingDataPhi[i] = b.CreatePHI(pendingData[i]->getType(), 2);
        pendingDataPhi[i]->addIncoming(pendingData[i], entry);
    }
    SmallVector<Value *, 16> inputOffset(strideBlocks+1);
    SmallVector<Value *, 16> outputOffset(strideBlocks+1);
    SmallVector<Value *, 16> pendingItems(strideBlocks+1);
    inputOffset[0] = inputOffsetPhi;
    outputOffset[0] = outputOffsetPhi;
    pendingItems[0] = pendingItemsPhi;
    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingData[i] = pendingDataPhi[i];
    }
    SmallVector<Value *, 8> offsets(strideBlocks);
    SmallVector<Value *, 8> currentFieldMask(strideBlocks);
    SmallVector<Value *, 8> pendingFieldIdx(strideBlocks);
    const unsigned fieldMovementSteps = floor_log2(numFields);
    SmallVector<SmallVector<Value *, 8>, 8> fieldsToMove(strideBlocks, SmallVector<Value *, 4>(fieldMovementSteps));
    SmallVector<SmallVector<Value *, 8>, 8> combine(strideBlocks, SmallVector<Value *, 4>(fieldMovementSteps));
    SmallVector<Value *, 8> doesFit(strideBlocks);
    SmallVector<Value *, 8> updatedFieldIdx(strideBlocks);
    SmallVector<Value *, 8> pendingSpaceFilled(strideBlocks);
    SmallVector<Value *, 8> shftBack(strideBlocks);
    for (unsigned blk = 0; blk < strideBlocks; blk++) {
        Value * const extractionMask = b.loadInputStreamBlock("extractionMask", ZERO, inputOffset[blk]);
        Value * const fieldPopCounts = b.simd_popcount(mFW, extractionMask);
        // For each field determine the (partial) sum popcount of all fields up to and
        // including the current field.
        Value * const partialSum = b.hsimd_partial_sum(mFW, fieldPopCounts);
        Value * const newPendingItems = b.mvmd_extract(mFW, partialSum, numFields - 1);
        //
        // Now determine for each source field the output offset of the first bit.
        // Note that this depends on the number of pending bits.
        //
        Value * pendingOffset = b.CreateURem(pendingItems[blk], CFW);
        Value * splatPending = b.simd_fill(mFW, b.CreateZExtOrTrunc(pendingOffset, fwTy));
        pendingFieldIdx[blk] = b.CreateUDiv(pendingItems[blk], CFW);
        offsets[blk] = b.simd_add(mFW, b.mvmd_slli(mFW, partialSum, 1), splatPending);
        offsets[blk] = b.simd_and(offsets[blk], fwMaskSplat); // parallel URem fw

        //
        // Determine the relative field number for each output field.   Note that the total
        // number of fields involved is numFields + 1.   However, the first field always
        // be immediately combined into the current pending data field, so we calculate
        // field numbers for all subsequent fields, (the fields that receive overflow bits).
        Value * pendingSum = b.simd_add(mFW, partialSum, splatPending);
        Value * initialFieldNo = b.simd_srli(mFW, pendingSum, floor_log2(mFW));
        //
        // Each input field contains bits from at most two output fields.  After
        // rotatioh, overflow bits are at the beginning of the rotated field, while
        // bits for the current output field will be shifted by the offset.
        // Create a mask for selecting current field bits.
        currentFieldMask[blk] = b.simd_sllv(mFW, b.allOnes(), offsets[blk]);

        // Now compress the data fields, eliminating duplicate field numbers.
        // However, field number operations will sometimes produce a zero, for
        // fields that are not selected.  So we add 1 to field numbers first.
        Value * fieldNo = b.simd_add(mFW, initialFieldNo, oneSplat);
        assert (fieldNo->getType() == oneSplat->getType());
        // Now move bits back and combine to produce consecutive field numbers.
        Value * firstFieldNoSplat = b.simd_fill(mFW, b.mvmd_extract(mFW, fieldNo, 0));
        assert (firstFieldNoSplat->getType() == fieldNo->getType());
        Value * finalFieldNo = b.CreateAdd(firstFieldNoSplat, intSequence);
        assert (finalFieldNo->getType() == fieldNo->getType());
        for (unsigned step = 0; step < fieldMovementSteps; step++) {
            unsigned mov = 1UL << step;
            assert (finalFieldNo->getType() == fieldNo->getType());
            Value * fieldMovement = b.CreateSub(finalFieldNo, fieldNo);
            Value * movSplat = b.simd_fill(mFW, b.getIntN(mFW, mov));
            fieldsToMove[blk][step] = b.simd_eq(mFW, movSplat, b.simd_and(movSplat, fieldMovement));
            Value * receivingFields = b.mvmd_srli(mFW, fieldsToMove[blk][step], mov);
            Value * newFieldNos = b.simd_if(1, receivingFields, b.mvmd_srli(mFW, fieldNo, mov), fieldNo);
            newFieldNos = b.CreateBitCast(newFieldNos, fieldNo->getType());
            combine[blk][step] = b.simd_and(b.simd_eq(mFW, fieldNo, newFieldNos), receivingFields);
            assert (newFieldNos->getType() == fieldNo->getType());
            fieldNo = newFieldNos;
        }
        //
        // Finally combine the pendingOutput and outputField data.
        // (a) shift forward outputField data to fill the pendingOutput values.
        // (b) shift back outputField data to clear data added to pendingOutput.
        //
        // However, we may need to increment pendingFieldIndex if we previously
        // filled the field with the extracted firstField value.  The first
        // value of the fieldNo vector will be 0 or 1.
        // It is possible that pendingFieldIndex will reach the total number
        // of fields held in register.  mvmd_sll may not handle this if it
        // translates to an LLVM shl.
        Value * increment = b.CreateZExtOrTrunc(b.mvmd_extract(mFW, initialFieldNo, 0), fwTy);
        updatedFieldIdx[blk] = b.CreateAdd(pendingFieldIdx[blk], increment);
        pendingSpaceFilled[blk] = b.CreateICmpEQ(updatedFieldIdx[blk], numFieldConst);
        assert (numFieldConst->getType() == updatedFieldIdx[blk]->getType());
        shftBack[blk] = b.CreateSub(numFieldConst, updatedFieldIdx[blk]);
        //
        // Now determine the total amount of pending items and whether
        // the pending data all fits within the pendingOutput.
        Value * nextPendingItems = b.CreateAdd(pendingItems[blk], newPendingItems);
        doesFit[blk] = b.CreateICmpULT(nextPendingItems, BLOCK_WIDTH);
        assert (nextPendingItems->getType() == BLOCK_WIDTH->getType());
        pendingItems[blk+1] = b.CreateSelect(doesFit[blk], nextPendingItems, b.CreateSub(nextPendingItems, BLOCK_WIDTH));
        inputOffset[blk+1] = b.CreateAdd(inputOffset[blk], ONE);
        Value * nextOutputOffset = b.CreateAdd(outputOffset[blk], ONE);
        outputOffset[blk+1] = b.CreateSelect(doesFit[blk], outputOffset[blk], nextOutputOffset);
        // Update pending data, based on whether all data fits within the current block.
    }
    for (unsigned blk = 0; blk < strideBlocks; blk++) {
        // Now process the input data block of each stream in the input stream set.
        //
        // First load all the stream set blocks and the pending data.
        SmallVector<Value *, 16> sourceBlock(mStreamCount);
        for (unsigned i = 0; i < mStreamCount; i++) {
            sourceBlock[i] = b.loadInputStreamBlock("sourceStreamSet", b.getInt32(i), inputOffset[blk]);
        }
        // Now separate the bits of each field into ones that go into the current field
        // and ones that go into the overflow field.   Extract the first field separately,
        // and then shift and combine subsequent fields.
        SmallVector<Value *, 16> pendingOutput(mStreamCount);
        SmallVector<Value *, 16> outputFields(mStreamCount);
        for (unsigned i = 0; i < mStreamCount; i++) {
            Value * alignedBits = b.simd_rotl(mFW, sourceBlock[i], offsets[blk]);
            Value * currentFieldBits = b.simd_and(alignedBits, currentFieldMask[blk]);
            Value * nextFieldBits = b.simd_xor(currentFieldBits, alignedBits);
            Value * firstField = b.mvmd_extract(mFW, currentFieldBits, 0);
            Value * vec1 = b.CreateInsertElement(zeroSplat, firstField, pendingFieldIdx[blk]);
            pendingOutput[i] = b.simd_or(pendingData[i], vec1);
            // shift back currentFieldBits to combine with nextFieldBits.
            outputFields[i] = b.simd_or(b.mvmd_srli(mFW, currentFieldBits, 1), nextFieldBits);
        }
        // Now compress the data fields, eliminating duplicate field numbers.
        for (unsigned step = 0; step < fieldMovementSteps; step++) {
            unsigned mov = 1 << step;
            for (unsigned i = 0; i < mStreamCount; i++) {
                Value * unmoved = b.simd_and(outputFields[i], combine[blk][step]);
                Value * data_to_move = b.simd_and(outputFields[i], fieldsToMove[blk][step]);
                Value * fields_back = b.mvmd_srli(mFW, data_to_move, mov);
                Value * cleared = b.simd_xor(outputFields[i], data_to_move);
                outputFields[i] = b.simd_or(fields_back, b.simd_or(unmoved, cleared));
            }
        }
        //
        // Finally combine the pendingOutput and outputField data.
        // (a) shift forward outputField data to fill the pendingOutput values.
        // (b) shift back outputField data to clear data added to pendingOutput.
        //
        for (unsigned i = 0; i < mStreamCount; i++) {
            Value * shiftedField = b.mvmd_sll(mFW, outputFields[i], updatedFieldIdx[blk]);
            Value * outputFwd = b.fwCast(mFW, shiftedField);
            shiftedField = b.CreateSelect(pendingSpaceFilled[blk], zeroSplat, outputFwd);
            pendingOutput[i] = b.simd_or(pendingOutput[i], shiftedField);
            outputFields[i] = b.mvmd_srl(mFW, outputFields[i], shftBack[blk]);
        }
        //
        // Write the pendingOutput data to outputStream.
        // Note: this data may be overwritten later, but we avoid branching.
        for (unsigned i = 0; i < mStreamCount; i++) {
            b.storeOutputStreamBlock("compressedOutput", b.getInt32(i), outputOffset[blk], pendingOutput[i]);
        }
        for (unsigned i = 0; i < mStreamCount; i++) {
            pendingData[i] = b.bitCast(b.CreateSelect(doesFit[blk], pendingOutput[i], outputFields[i]));
        }
    }
    //
    // Prepare Phi nodes for the next stride.trid
    //
    pendingItemsPhi->addIncoming(pendingItems[strideBlocks], segmentLoop);
    inputOffsetPhi->addIncoming(inputOffset[strideBlocks], segmentLoop);
    outputOffsetPhi->addIncoming(outputOffset[strideBlocks], segmentLoop);
    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingDataPhi[i]->addIncoming(pendingData[i], segmentLoop);
    }
    //
    // Now continue the loop if there are more blocks to process.
    Value * moreToDo = b.CreateICmpNE(inputOffset[strideBlocks], numOfBlocks);
    b.CreateCondBr(moreToDo, segmentLoop, segmentDone);

    b.SetInsertPoint(segmentDone);
    for (unsigned i = 0; i < mStreamCount; i++) {
        b.setScalarField("pendingOutputBlock_" + std::to_string(i), pendingData[i]);
    }
    // It's possible that we'll perfectly fill the last block of data on the
    // last iteration of the loop. If we arbritarily write the pending data,
    // this could end up incorrectly overwriting unprocessed data with 0s.
    Value * const hasMore = b.CreateICmpNE(pendingItems[strideBlocks], ZERO);
    b.CreateLikelyCondBr(hasMore, writePartialData, segmentExit);

    b.SetInsertPoint(writePartialData);
    for (unsigned i = 0; i < mStreamCount; i++) {
        b.storeOutputStreamBlock("compressedOutput", b.getInt32(i), outputOffset[strideBlocks], pendingData[i]);
    }
    b.CreateBr(segmentExit);

    b.SetInsertPoint(segmentExit);
}



#if 0

// alternate sparse-data two-iterator approach to streamcompress; not well tested.

void StreamCompressKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfBlocks) {
    IntegerType * const fwTy = b.getIntNTy(mFW);
    IntegerType * const sizeTy = b.getSizeTy();
    const unsigned numFields = b.getBitBlockWidth() / mFW;
    Constant * zeroSplat = Constant::getNullValue(b.fwVectorType(mFW));
    Constant * oneSplat = b.getSplat(numFields, ConstantInt::get(fwTy, 1));
    Constant * CFW = ConstantInt::get(fwTy, mFW);
    Constant * fwSplat = b.getSplat(numFields, CFW);
    Constant * numFieldConst = ConstantInt::get(fwTy, numFields);
    Constant * fwMaskSplat = b.getSplat(numFields, ConstantInt::get(fwTy, mFW - 1));
    Constant * BLOCK_WIDTH = ConstantInt::get(fwTy, b.getBitBlockWidth());
    Constant * BLOCK_MASK = ConstantInt::get(fwTy, b.getBitBlockWidth() - 1);

    ConstantInt * const sz_ZERO = ConstantInt::get(sizeTy, 0);
    ConstantInt * const sz_ONE = ConstantInt::get(sizeTy, 1);
    ConstantInt * const sz_STEP_SIZE = ConstantInt::get(sizeTy, sizeTy->getBitWidth());

    BasicBlock * const buildIteratorMask = b.CreateBasicBlock("buildIteratorMask");
    BasicBlock * const checkIteratorMask = b.CreateBasicBlock("checkIteratorMask");
    BasicBlock * const compressNextStride = b.CreateBasicBlock("compressNextStride");

    BasicBlock * const checkForAnotherStride = b.CreateBasicBlock("checkForAnotherStride");

    BasicBlock * const segmentDone = b.CreateBasicBlock("segmentDone");
    BasicBlock * const writePartialData = b.CreateBasicBlock("writePartialData");
    BasicBlock * const segmentExit = b.CreateBasicBlock("segmentExit");

    Value * const produced = b.getProducedItemCount("compressedOutput");
    Value * const pendingItemCount = b.CreateAnd(b.CreateZExtOrTrunc(produced, fwTy), BLOCK_MASK);

    // TODO: we could make this kernel stateless but would have to use "bitblock_mask_to" on the
    // input to account for the fact the pipeline doesn't zero out output stream data when it
    // reuses it.

    SmallVector<Value *, 16> pendingData(mStreamCount);
    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingData[i] = b.getScalarField("pendingOutputBlock_" + std::to_string(i));
    }

    Value * const tooFew = b.CreateICmpULT(numOfBlocks, sz_STEP_SIZE);
    Value * const initialMaxIterationStrides = b.CreateSelect(tooFew, numOfBlocks, sz_STEP_SIZE);

    BasicBlock * const entry = b.GetInsertBlock();
    b.CreateBr(buildIteratorMask);

    // Main Loop
    b.SetInsertPoint(buildIteratorMask);
    PHINode * const inputOffsetPhi = b.CreatePHI(sizeTy, 3);
    PHINode * const outerOutputOffsetPhi = b.CreatePHI(sizeTy, 3);
    PHINode * const outerPendingItemsPhi = b.CreatePHI(fwTy, 3);
    PHINode * const maxIterationStridesPhi = b.CreatePHI(sizeTy, 3);

    SmallVector<PHINode *, 16> outerPendingDataPhi(mStreamCount);

    inputOffsetPhi->addIncoming(sz_ZERO, entry);
    inputOffsetPhi->addIncoming(inputOffsetPhi, buildIteratorMask);

    outerOutputOffsetPhi->addIncoming(sz_ZERO, entry);
    outerOutputOffsetPhi->addIncoming(outerOutputOffsetPhi, buildIteratorMask);

    outerPendingItemsPhi->addIncoming(pendingItemCount, entry);
    outerPendingItemsPhi->addIncoming(outerPendingItemsPhi, buildIteratorMask);

    maxIterationStridesPhi->addIncoming(initialMaxIterationStrides, entry);
    maxIterationStridesPhi->addIncoming(maxIterationStridesPhi, buildIteratorMask);

    for (unsigned i = 0; i < mStreamCount; i++) {
        PHINode * const phi = b.CreatePHI(pendingData[i]->getType(), 3);
        phi->addIncoming(pendingData[i], entry);
        phi->addIncoming(phi, buildIteratorMask);
        outerPendingDataPhi[i] = phi;
    }

    PHINode * const maskOffsetPhi = b.CreatePHI(sizeTy, 3);
    maskOffsetPhi->addIncoming(sz_ZERO, entry);
    maskOffsetPhi->addIncoming(sz_ZERO, checkForAnotherStride);

    PHINode * const iteratorMaskPhi = b.CreatePHI(sizeTy, 3);
    iteratorMaskPhi->addIncoming(sz_ZERO, entry);
    iteratorMaskPhi->addIncoming(sz_ZERO, checkForAnotherStride);

    Value * const offset = b.CreateOr(inputOffsetPhi, maskOffsetPhi);
    Value * const pendingMask = b.loadInputStreamBlock("extractionMask", sz_ZERO, offset);

    Value * const hasAnyBit = b.CreateZExt(b.bitblock_any(pendingMask), sizeTy);
    Value * const maskBit = b.CreateShl(hasAnyBit, maskOffsetPhi);
    Value * const iteratorMask = b.CreateOr(iteratorMaskPhi, maskBit);

    Value * const nextMaskOffset = b.CreateAdd(maskOffsetPhi, sz_ONE);
    maskOffsetPhi->addIncoming(nextMaskOffset, buildIteratorMask);
    iteratorMaskPhi->addIncoming(iteratorMask, buildIteratorMask);
    Value * const finishedMask = b.CreateICmpNE(nextMaskOffset, maxIterationStridesPhi);
    b.CreateCondBr(finishedMask, buildIteratorMask, checkIteratorMask);

    b.SetInsertPoint(checkIteratorMask);
    Value * const hasAnyMarks = b.CreateICmpNE(iteratorMask, sz_ZERO);
    b.CreateLikelyCondBr(hasAnyMarks, compressNextStride, checkForAnotherStride);

    b.SetInsertPoint(compressNextStride);
    PHINode * const outputOffsetPhi = b.CreatePHI(sizeTy, 2);
    PHINode * const pendingItemsPhi = b.CreatePHI(fwTy, 2);
    PHINode * const currentIteratorMaskPhi = b.CreatePHI(sizeTy, 2);
    SmallVector<PHINode *, 16> pendingDataPhi(mStreamCount);

    outputOffsetPhi->addIncoming(outerOutputOffsetPhi, checkIteratorMask);
    pendingItemsPhi->addIncoming(pendingItemCount, checkIteratorMask);
    currentIteratorMaskPhi->addIncoming(iteratorMask, checkIteratorMask);

    for (unsigned i = 0; i < mStreamCount; i++) {
        PHINode * const phi = b.CreatePHI(outerPendingDataPhi[i]->getType(), 3);
        phi->addIncoming(outerPendingDataPhi[i], checkIteratorMask);
        pendingDataPhi[i] = phi;
    }
    Value * const pos = b.CreateCountForwardZeroes(currentIteratorMaskPhi);
    Value * const bitMark = b.CreateShl(sz_ONE, pos);
    Value * const nextIteratorMask = b.CreateXor(currentIteratorMaskPhi, bitMark);
    currentIteratorMaskPhi->addIncoming(nextIteratorMask, compressNextStride);

    Value * const inputOffset = b.CreateOr(inputOffsetPhi, pos);
    Value * const extractionMask = b.loadInputStreamBlock("extractionMask", sz_ZERO, inputOffset);
    Value * const fieldPopCounts = b.simd_popcount(mFW, extractionMask);
    // For each field determine the (partial) sum popcount of all fields up to and
    // including the current field.
    Value * const partialSum = b.hsimd_partial_sum(mFW, fieldPopCounts);
    Value * const newPendingItems = b.mvmd_extract(mFW, partialSum, numFields - 1);

    //
    // Now determine for each source field the output offset of the first bit.
    // Note that this depends on the number of pending bits.
    //
    Value * pendingOffset = b.CreateURem(pendingItemsPhi, CFW);
    Value * splatPending = b.simd_fill(mFW, b.CreateZExtOrTrunc(pendingOffset, fwTy));
    Value * pendingFieldIdx = b.CreateUDiv(pendingItemsPhi, CFW);
    Value * offsets = b.simd_add(mFW, b.mvmd_slli(mFW, partialSum, 1), splatPending);
    offsets = b.simd_and(offsets, fwMaskSplat); // parallel URem fw

    // Determine the relative field number for each output field.   Note that the total
    // number of fields involved is numFields + 1.   However, the first field always
    // be immediately combined into the current pending data field, so we calculate
    // field numbers for all subsequent fields, (the fields that receive overflow bits).
    Value * pendingSum = b.simd_add(mFW, partialSum, splatPending);
    Value * fieldNo = b.simd_srli(mFW, pendingSum, floor_log2(mFW));
    // Now process the input data block of each stream in the input stream set.
    //
    // First load all the stream set blocks and the pending data.
    SmallVector<Value *, 16> sourceBlock(mStreamCount);
    for (unsigned i = 0; i < mStreamCount; i++) {
        sourceBlock[i] = b.loadInputStreamBlock("sourceStreamSet", b.getInt32(i), inputOffset);
    }
    // Now separate the bits of each field into ones that go into the current field
    // and ones that go into the overflow field.   Extract the first field separately,
    // and then shift and combine subsequent fields.
    SmallVector<Value *, 16> pendingOutput(mStreamCount);
    SmallVector<Value *, 16> outputFields(mStreamCount);
    Value * backShift = b.simd_sub(mFW, fwSplat, offsets);
    for (unsigned i = 0; i < mStreamCount; i++) {
        Value * currentFieldBits = b.simd_sllv(mFW, sourceBlock[i], offsets);
        Value * nextFieldBits = b.simd_srlv(mFW, sourceBlock[i], backShift);
        Value * firstField = b.mvmd_extract(mFW, currentFieldBits, 0);
        Value * vec1 = b.CreateInsertElement(zeroSplat, firstField, pendingFieldIdx);
        pendingOutput[i] = b.simd_or(pendingDataPhi[i], vec1);
        // shift back currentFieldBits to combine with nextFieldBits.
        outputFields[i] = b.simd_or(b.mvmd_srli(mFW, currentFieldBits, 1), nextFieldBits);
    }
    // Now combine forward all fields with the same field number.  This may require
    // up to log2 numFields steps.
    for (unsigned j = 1; j < numFields; j*=2) {
        Value * select = b.simd_eq(mFW, fieldNo, b.mvmd_slli(mFW, fieldNo, j));
        for (unsigned i = 0; i < mStreamCount; i++) {
            Value * fields_fwd = b.mvmd_slli(mFW, outputFields[i], j);
            outputFields[i] = b.simd_or(outputFields[i], b.simd_and(select, fields_fwd));
       }
    }
    // Now compress the data fields, eliminating all but the last field from
    // each run of consecutive field having the same field number as a subsequent field.
    // But it may be that last field number is 0 which will compare equal to a 0 shifted in.
    // So we add 1 to field numbers first.
    Value * nonZeroFieldNo = b.simd_add(mFW, fieldNo, oneSplat);
    Value * eqNext = b.simd_eq(mFW, nonZeroFieldNo, b.mvmd_srli(mFW, nonZeroFieldNo, 1));
    Value * compressMask = b.hsimd_signmask(mFW, b.simd_not(eqNext));
    for (unsigned i = 0; i < mStreamCount; i++) {
        outputFields[i] = b.mvmd_compress(mFW, outputFields[i], compressMask);
    }
    //
    // Finally combine the pendingOutput and outputField data.
    // (a) shift forward outputField data to fill the pendingOutput values.
    // (b) shift back outputField data to clear data added to pendingOutput.
    //
    // However, we may need to increment pendingFieldIndex if we previously
    // filled the field with the extracted firstField value.  The first
    // value of the fieldNo vector will be 0 or 1.
    // It is possible that pendingFieldIndex will reach the total number
    // of fields held in register.  mvmd_sll may not handle this if it
    // translates to an LLVM shl.
    Value * increment = b.CreateZExtOrTrunc(b.mvmd_extract(mFW, fieldNo, 0), fwTy);
    pendingFieldIdx = b.CreateAdd(pendingFieldIdx, increment);
    Value * const pendingSpaceFilled = b.CreateICmpEQ(pendingFieldIdx, numFieldConst);
    Value * shftBack = b.CreateSub(numFieldConst, pendingFieldIdx);
    for (unsigned i = 0; i < mStreamCount; i++) {
        Value * shiftedField = b.mvmd_sll(mFW, outputFields[i], pendingFieldIdx);
        Value * outputFwd = b.fwCast(mFW, shiftedField);
        shiftedField = b.CreateSelect(pendingSpaceFilled, zeroSplat, outputFwd);
        pendingOutput[i] = b.simd_or(pendingOutput[i], shiftedField);
        outputFields[i] = b.mvmd_srl(mFW, outputFields[i], shftBack);
    }
    //
    // Write the pendingOutput data to outputStream.
    // Note: this data may be overwritten later, but we avoid branching.
    for (unsigned i = 0; i < mStreamCount; i++) {
        b.storeOutputStreamBlock("compressedOutput", b.getInt32(i), outputOffsetPhi, pendingOutput[i]);
    }
    // Now determine the total amount of pending items and whether
    // the pending data all fits within the pendingOutput.
    Value * nextPendingItems = b.CreateAdd(pendingItemsPhi, newPendingItems);
    Value * const doesFit = b.CreateICmpULT(nextPendingItems, BLOCK_WIDTH);
    // nextPendingItems = b.CreateSelect(doesFit, nextPendingItems, b.CreateSub(nextPendingItems, BLOCK_WIDTH));
    nextPendingItems = b.CreateAnd(nextPendingItems, BLOCK_MASK);
    pendingItemsPhi->addIncoming(nextPendingItems, compressNextStride);


    //
    // Prepare Phi nodes for the next iteration.
    //


    // But don't advance the output if all the data does fit into pendingOutput.
    Value * nextOutputOffset = b.CreateAdd(outputOffsetPhi, sz_ONE);
    nextOutputOffset = b.CreateSelect(doesFit, outputOffsetPhi, nextOutputOffset);
    outputOffsetPhi->addIncoming(nextOutputOffset, compressNextStride);

    for (unsigned i = 0; i < mStreamCount; i++) {
        pendingOutput[i] = b.bitCast(b.CreateSelect(doesFit, pendingOutput[i], outputFields[i]));
        pendingDataPhi[i]->addIncoming(pendingOutput[i], compressNextStride);
    }
    //
    // Now continue the loop if there are more blocks to process.

    Value * const moreToDo = b.CreateICmpNE(nextIteratorMask, sz_ZERO);
    b.CreateCondBr(moreToDo, compressNextStride, checkForAnotherStride);

    b.SetInsertPoint(checkForAnotherStride);
    PHINode * const nextOutputOffsetPhi = b.CreatePHI(sizeTy, 2);
    PHINode * const nextPendingItemsPhi = b.CreatePHI(fwTy, 2);
    SmallVector<PHINode *, 16> nextPendingDataPhi(mStreamCount);

    nextOutputOffsetPhi->addIncoming(outerOutputOffsetPhi, checkIteratorMask);
    nextOutputOffsetPhi->addIncoming(nextOutputOffset, compressNextStride);
    outerOutputOffsetPhi->addIncoming(nextOutputOffsetPhi, checkForAnotherStride);

    nextPendingItemsPhi->addIncoming(outerPendingItemsPhi, checkIteratorMask);
    nextPendingItemsPhi->addIncoming(nextPendingItems, compressNextStride);
    outerPendingItemsPhi->addIncoming(nextPendingItemsPhi, checkForAnotherStride);

    for (unsigned i = 0; i < mStreamCount; i++) {
        PHINode * const phi = b.CreatePHI(outerPendingDataPhi[i]->getType(), 3);
        phi->addIncoming(outerPendingDataPhi[i], checkIteratorMask);
        phi->addIncoming(pendingOutput[i], compressNextStride);
        nextPendingDataPhi[i] = phi;
        outerPendingDataPhi[i]->addIncoming(phi, checkForAnotherStride);
    }

    Value * const nextOuterInputOffset = b.CreateAdd(inputOffsetPhi, sz_STEP_SIZE);
    inputOffsetPhi->addIncoming(nextOuterInputOffset, checkForAnotherStride);

    Value * const notDone = b.CreateICmpULT(nextOuterInputOffset, numOfBlocks);
    Value * const remaining = b.CreateSub(numOfBlocks, nextOuterInputOffset);
    Value * const partialStepRemaining = b.CreateICmpUGT(numOfBlocks, nextOuterInputOffset);
    Value * const nextMaxIterationStrides = b.CreateSelect(partialStepRemaining, remaining, sz_STEP_SIZE);
    maxIterationStridesPhi->addIncoming(nextMaxIterationStrides, checkForAnotherStride);

    b.CreateCondBr(notDone, buildIteratorMask, segmentDone);

    b.SetInsertPoint(segmentDone);
    for (unsigned i = 0; i < mStreamCount; i++) {
        b.setScalarField("pendingOutputBlock_" + std::to_string(i), nextPendingDataPhi[i]);
    }

    // It's possible that we'll perfectly fill the last block of data on the
    // last iteration of the loop. If we arbritarily write the pending data,
    // this could end up incorrectly overwriting unprocessed data with 0s.
    Value * const hasMore = b.CreateICmpNE(nextPendingItemsPhi, sz_ZERO);
    b.CreateLikelyCondBr(hasMore, writePartialData, segmentExit);

    b.SetInsertPoint(writePartialData);
    for (unsigned i = 0; i < mStreamCount; i++) {
        b.storeOutputStreamBlock("compressedOutput", b.getInt32(i), nextOutputOffsetPhi, nextPendingDataPhi[i]);
    }
    b.CreateBr(segmentExit);

    b.SetInsertPoint(segmentExit);
}

#endif

Bindings makeSwizzledDeleteByPEXTOutputBindings(const std::vector<StreamSet *> & outputStreamSets, const unsigned PEXTWidth) {
    const auto n = outputStreamSets.size();
    Bindings outputs;
    outputs.reserve(n);
    outputs.emplace_back("outputSwizzle0", outputStreamSets[0], PopcountOf("selectors"), BlockSize(PEXTWidth)); // PopcountOfNot("delMaskSet")
    for (unsigned i = 1; i < n; ++i) {
        outputs.emplace_back("outputSwizzle" + std::to_string(i), outputStreamSets[i], RateEqualTo("outputSwizzle0"), BlockSize(PEXTWidth));
    }
    return outputs;
}

SwizzledDeleteByPEXTkernel::SwizzledDeleteByPEXTkernel(LLVMTypeSystemInterface & ts,
                                                       StreamSet * selectors, StreamSet * inputStreamSet,
                                                       const std::vector<StreamSet *> & outputStreamSets,
                                                       const unsigned PEXTWidth)

: MultiBlockKernel(ts, "PEXTdel" + std::to_string(PEXTWidth) + "_" + std::to_string(inputStreamSet->getNumElements()),
{Binding{"selectors", selectors}, Binding{"inputStreamSet", inputStreamSet}},
 makeSwizzledDeleteByPEXTOutputBindings(outputStreamSets, PEXTWidth),
{}, {}, {})
, mStreamCount(inputStreamSet->getNumElements())
, mSwizzleFactor(ts.getBitBlockWidth() / PEXTWidth)
, mSwizzleSetCount(ceil_udiv(mStreamCount, mSwizzleFactor))
, mPEXTWidth(PEXTWidth) {

    assert((mPEXTWidth > 0) && ((mPEXTWidth & (mPEXTWidth - 1)) == 0) && "mDelCountFieldWidth must be a power of 2");
    assert(mSwizzleFactor > 1 && "mDelCountFieldWidth must be less than the block width");
    assert((mPEXTWidth == 64 || mPEXTWidth == 32) && "PEXT width must be 32 or 64");
    assert (mSwizzleSetCount);
    assert (outputStreamSets.size() == mSwizzleSetCount);
    assert (outputStreamSets[0]->getNumElements() == mSwizzleFactor);

    addInternalScalar(ts.getBitBlockType(), "pendingSwizzleData0");
    for (unsigned i = 1; i < outputStreamSets.size(); ++i) {
        assert (outputStreamSets[i]->getNumElements() == mSwizzleFactor);
        addInternalScalar(ts.getBitBlockType(), "pendingSwizzleData" + std::to_string(i));
    }
}

void SwizzledDeleteByPEXTkernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    // We use delMask to apply the same PEXT delete operation to each stream in the input stream set

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const beginLoop = b.CreateBasicBlock("beginLoop");

    ConstantInt * const ZERO = b.getSize(0);
    ConstantInt * const BLOCK_WIDTH_MASK = b.getSize(b.getBitBlockWidth() - 1);
    ConstantInt * const PEXT_WIDTH = b.getSize(mPEXTWidth);
    ConstantInt * const LOG_2_PEXT_WIDTH = b.getSize(floor_log2(mPEXTWidth));
    ConstantInt * const LOG_2_SWIZZLE_FACTOR = b.getSize(floor_log2(mSwizzleFactor));
    ConstantInt * const PEXT_WIDTH_MASK = b.getSize(mPEXTWidth - 1);

    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
    }
    // All output groups have the same count.
    Value * const baseOutputProduced = b.getProducedItemCount(getOutputStreamSetBinding(0).getName());
    Value * const baseProducedOffset = b.CreateAnd(baseOutputProduced, BLOCK_WIDTH_MASK);

    // There is a separate vector of pending data for each swizzle group.
    std::vector<Value *> pendingData(mSwizzleSetCount);
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        pendingData[i] = b.getScalarField("pendingSwizzleData" + std::to_string(i));
    }
    b.CreateBr(beginLoop);

    b.SetInsertPoint(beginLoop);
    PHINode * const strideIndex = b.CreatePHI(numOfBlocks->getType(), 2);
    strideIndex->addIncoming(ZERO, entry);
    PHINode * const producedOffsetPhi = b.CreatePHI(numOfBlocks->getType(), 2);
    producedOffsetPhi->addIncoming(baseProducedOffset, entry);
    std::vector<PHINode *> pendingDataPhi(mSwizzleSetCount);
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        pendingDataPhi[i] = b.CreatePHI(pendingData[i]->getType(), 2);
        pendingDataPhi[i]->addIncoming(pendingData[i], entry);
        pendingData[i] = pendingDataPhi[i];
    }

    Value * const selectors = b.loadInputStreamBlock("selectors", strideIndex);

    const auto swizzleSets = makeSwizzleSets(b, selectors, strideIndex);

    Value * const newItemCounts = b.simd_popcount(mPEXTWidth, selectors);

    // Compress the PEXTedSwizzleSets
    // Output is written and committed to the output buffer one swizzle at a time.
    Value * producedOffset = producedOffsetPhi;

    // For each row i
    for (unsigned i = 0; i < mSwizzleFactor; i++) {

        // Generate code for each of the mSwizzleFactor fields making up a block.
        // We load the count for the field and process all swizzle groups accordingly.
        Value * const pendingOffset = b.CreateAnd(producedOffset, PEXT_WIDTH_MASK);
        Value * const newItemCount = b.CreateExtractElement(newItemCounts, i);
        Value * const pendingSpace = b.CreateSub(PEXT_WIDTH, pendingOffset);
        Value * const pendingSpaceFilled = b.CreateICmpUGE(newItemCount, pendingSpace);

        Value * const shiftVector = b.simd_fill(mPEXTWidth, pendingOffset);
        Value * const spaceVector = b.simd_fill(mPEXTWidth, pendingSpace);

        Value * const outputIndex = b.CreateLShr(producedOffset, LOG_2_PEXT_WIDTH);
        Value * const swizzleIndex = b.CreateAnd(outputIndex, mSwizzleFactor - 1);
        Value * const blockOffset = b.CreateLShr(outputIndex, LOG_2_SWIZZLE_FACTOR);

        // Data from the ith swizzle pack of each group is processed
        // according to the same newItemCount, pendingSpace, ...
        for (unsigned j = 0; j < mSwizzleSetCount; j++) {
            Value * const newItems = swizzleSets[j][i];
            // Combine as many of the new items as possible into the pending group.
            Value * const shiftedItems = b.CreateShl(newItems, shiftVector);
            Value * const combinedGroup = b.CreateOr(pendingData[j], shiftedItems);
            // To avoid an unpredictable branch, always store the combined group, whether full or not.
            b.storeOutputStreamBlock(getOutputStreamSetBinding(j).getName(), swizzleIndex, blockOffset, combinedGroup);
            // Any items in excess of the space available in the current pending group overflow for the next group.
            Value * overFlowGroup = b.CreateLShr(newItems, spaceVector);
            // If we filled the space, then the overflow group becomes the new pending group and the index is updated.
            pendingData[j] = b.CreateSelect(pendingSpaceFilled, overFlowGroup, combinedGroup);
        }
        producedOffset = b.CreateAdd(producedOffset, newItemCount);
    }

    BasicBlock * const finishedLoop = b.CreateBasicBlock("finishedLoop");
    Value * const nextStrideIndex = b.CreateAdd(strideIndex, b.getSize(1));
    BasicBlock * const loopEndBlock = b.GetInsertBlock();
    strideIndex->addIncoming(nextStrideIndex, loopEndBlock);
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        pendingDataPhi[i]->addIncoming(pendingData[i], loopEndBlock);
    }
    producedOffsetPhi->addIncoming(producedOffset, loopEndBlock);
    Value * const doneLoop = b.CreateICmpEQ(nextStrideIndex, numOfBlocks);

    b.CreateUnlikelyCondBr(doneLoop, finishedLoop, beginLoop);

    b.SetInsertPoint(finishedLoop);
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        b.setScalarField("pendingSwizzleData" + std::to_string(i), pendingData[i]);
    }
}

/*
Apply PEXT deletion to the blocks in strms and swizzle the result.

Q: Why is it advantageous to swizzle the PEXTed streams?

A: PEXT doesn't compress streams, if the input to a PEXT operation is 64 bits wide, the output is also 64 bits wide.

Example:
Input:     11101101
PEXT mask: 11110000
Output:    00001110

PEXT selects the bits we tell it to and stores them at contiguous lower-order bits. Higher-order bits are
cleared. This has implications if we're working with multiple streams.

For example, say we've applied PEXT on the following 4 streams using this deletion mask (inverse of PEXT mask): 00000011 00011111 00111111 00000111
(I think this diagram is backwards, PEXTed bits should be stored in lower-order bits, not higher.)
Stream 1:   abcdef00 ghi00000 jk000000 lmnop000
Stream 2:   qrstuv00 wxy00000 z1000000 23456000
Stream 3:   ABCDEF00 GHI00000 JK000000 LMNOP000
Stream 4:   QRSTUV00 WZY00000 Z1000000 23456000

If we wanted to compress each stream to remove the sequences of 0s, it's tricky. The first 32 bits of each stream
should be compress by 2 bits, the second 32 bits by 5, etc. If we swizzle the streams with a swizzle factor of 4 we have a much easier
time:

The swizzled output using a field width of 8 produces the following swizzles (swizzle factor = block width / pext field width = 4).

Swizzle 1:  abcdef00 qrstuv00 ABCDEF00 QRSTUV00
Swizzle 2:  ghi00000 wxy00000 GHI00000 WZY00000
Swizzle 3:  jk000000 z1000000 JK000000 Z1000000
Swizzle 4:  lmnop000 23456000 LMNOP000 23456000

Now we can compress each 32-bit segment of swizzle 1 by 2, each 32 bit segment of swizzle 2 by 4, etc. Once we've completed the
compression, we unswizzle to restore the 4 streams. The streams are now fully compressed!

Args:
    strms: the vector of blocks to apply PEXT operations to. strms[i] is the block associated with the ith input stream.
    masks: the PEXT deletion masks to apply to each block in strms (input mask is broken into PEXT width pieces, apply pieces
        sequentially to PEXT a full block.)

Returns:
    output (vector of Value*): Swizzled, PEXTed version of strms. See example above.
*/

SwizzledDeleteByPEXTkernel::SwizzleSets SwizzledDeleteByPEXTkernel::makeSwizzleSets(KernelBuilder & b, llvm::Value * const selectors, Value * const strideIndex) {

    Value * const m = b.fwCast(mPEXTWidth, selectors);

    std::vector<Value *> masks(mSwizzleFactor);
    for (unsigned i = 0; i < mSwizzleFactor; i++) {
        masks[i] = b.CreateExtractElement(m, i);

    }

    SwizzleSets swizzleSets;
    swizzleSets.reserve(mSwizzleSetCount);

    VectorType * const vecTy = b.fwVectorType(mPEXTWidth);

    UndefValue * const outputInitializer = UndefValue::get(vecTy);

    std::vector<Value *> input(mSwizzleFactor);
    // For each of the k swizzle sets required to apply PEXT to all input streams
    for (unsigned i = 0; i < mSwizzleSetCount; ++i) {

        for (unsigned j = 0; j < mSwizzleFactor; ++j) {
            const unsigned k = (i * mSwizzleFactor) + j;
            if (k < mStreamCount) {
                input[j] = b.CreateBitCast(b.loadInputStreamBlock("inputStreamSet", b.getInt32(k), strideIndex), vecTy);
            } else {
                input[j] = Constant::getNullValue(vecTy);
            }
        }

        // TODO: if a SIMD pext instruction exists, we should first swizzle the lanes
        // then splat the pext mask and apply it to each output row

        std::vector<Value *> output(mSwizzleFactor, outputInitializer);
        // For each of the input streams
        for (unsigned j = 0; j < mSwizzleFactor; j++) {
            for (unsigned k = 0; k < mSwizzleFactor; k++) {
                // Load block j,k
                Value * const field = b.CreateExtractElement(input[j], k);
                // Apply PEXT deletion
                Value * const selected = b.CreatePextract(field, masks[k]);

                // Then store it as our k,j-th output
                output[k] = b.CreateInsertElement(output[k], selected, j);
            }
        }
        swizzleSets.emplace_back(output);
    }

    return swizzleSets;
}


// Apply deletion to a set of stream_count input streams and produce a set of swizzled output streams.
// Kernel inputs: stream_count data streams plus one del_mask stream
// Outputs: swizzles containing the swizzled deleted streams, plus a partial sum popcount

void DeleteByPEXTkernel::generateDoBlockMethod(KernelBuilder & b) {
    Value * delMask = b.loadInputStreamBlock("delMaskSet", b.getInt32(0));
    generateProcessingLoop(b, delMask);
}

void DeleteByPEXTkernel::generateFinalBlockMethod(KernelBuilder & b, Value * remainingBytes) {
    IntegerType * vecTy = b.getIntNTy(b.getBitBlockWidth());
    Value * remaining = b.CreateZExt(remainingBytes, vecTy);
    Value * EOF_del = b.bitCast(b.CreateShl(Constant::getAllOnesValue(vecTy), remaining));
    Value * delMask = b.CreateOr(EOF_del, b.loadInputStreamBlock("delMaskSet", b.getInt32(0)));
    generateProcessingLoop(b, delMask);
}

void DeleteByPEXTkernel::generateProcessingLoop(KernelBuilder & b, Value * delMask) {
    std::vector<Value *> masks(mSwizzleFactor);
    Value * const m = b.fwCast(mPEXTWidth, b.simd_not(delMask));
    for (unsigned i = 0; i < mSwizzleFactor; i++) {
        masks[i] = b.CreateExtractElement(m, i);
    }

    for (unsigned i = 0; i < mStreamCount; ++i) {
        Value * input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(i));
        Value * value = b.fwCast(mPEXTWidth, input);
        Value * output = UndefValue::get(value->getType());
        for (unsigned j = 0; j < mSwizzleFactor; j++) {
            Value * field = b.CreateExtractElement(value, j);
            Value * const compressed = b.CreatePextract(field, masks[j]);
            output = b.CreateInsertElement(output, compressed, j);
        }
        b.storeOutputStreamBlock("outputStreamSet", b.getInt32(i), output);
    }
    Value * delCount = b.simd_popcount(mDelCountFieldWidth, b.simd_not(delMask));
    b.storeOutputStreamBlock("deletionCounts", b.getInt32(0), b.bitCast(delCount));
}

DeleteByPEXTkernel::DeleteByPEXTkernel(LLVMTypeSystemInterface & ts, unsigned fw, unsigned streamCount, unsigned PEXT_width)
: BlockOrientedKernel(ts, "PEXTdel" + std::to_string(fw) + "_" + std::to_string(streamCount) + "_" + std::to_string(PEXT_width),
              {Binding{ts.getStreamSetTy(streamCount), "inputStreamSet"},
                  Binding{ts.getStreamSetTy(), "delMaskSet"}},
              {}, {}, {}, {})
, mDelCountFieldWidth(fw)
, mStreamCount(streamCount)
, mSwizzleFactor(ts.getBitBlockWidth() / PEXT_width)
, mPEXTWidth(PEXT_width) {
    mOutputStreamSets.emplace_back(ts.getStreamSetTy(mStreamCount), "outputStreamSet", PopcountOfNot("delMaskSet"));
    mOutputStreamSets.emplace_back(ts.getStreamSetTy(), "deletionCounts");
}


//
// This kernel performs final stream compression for a set of N bitstreams, given
// (a) a set of bitstreams partially compressed within K-bit fields and stored
//     in K-bit swizzled form, and
// (b) a stream of deletion/extraction counts per K-bit stride.
//
// Restrictions:  At present, only K=64 is supported.
//                At present, N must be an exact multiple of BLOCK_SIZE/K.
//
// The kernel always consumes full blocks of input and emits data into the output
// buffer in swizzles of K items at a time.   Upon completion of a segment,
// up to K-1 pending output items per stream may be stored in the kernel state.
//
// Note: that both input streams and output streams are stored in swizzled form.
//

SwizzledBitstreamCompressByCount::SwizzledBitstreamCompressByCount(LLVMTypeSystemInterface & ts, unsigned bitStreamCount, unsigned fieldWidth)
: BlockOrientedKernel(ts, "swizzled_compress" + std::to_string(fieldWidth) + "_" + std::to_string(bitStreamCount),
                     {Binding{ts.getStreamSetTy(), "countsPerStride"}}, {}, {}, {}, {})
, mBitStreamCount(bitStreamCount)
, mFieldWidth(fieldWidth)
, mSwizzleFactor(ts.getBitBlockWidth() / fieldWidth)
, mSwizzleSetCount((mBitStreamCount + mSwizzleFactor - 1)/mSwizzleFactor) {
    assert((fieldWidth > 0) && ((fieldWidth & (fieldWidth - 1)) == 0) && "fieldWidth must be a power of 2");
    assert(mSwizzleFactor > 1 && "fieldWidth must be less than the block width");
    mInputStreamSets.push_back(Binding{ts.getStreamSetTy(mSwizzleFactor, 1), "inputSwizzle0"});
    mOutputStreamSets.push_back(Binding{ts.getStreamSetTy(mSwizzleFactor, 1), "outputSwizzle0", BoundedRate(0, 1)});
    addInternalScalar(ts.getBitBlockType(), "pendingSwizzleData0");
    for (unsigned i = 1; i < mSwizzleSetCount; i++) {
        mInputStreamSets.push_back(Binding{ts.getStreamSetTy(mSwizzleFactor, 1), "inputSwizzle" + std::to_string(i)});
        mOutputStreamSets.push_back(Binding{ts.getStreamSetTy(mSwizzleFactor, 1), "outputSwizzle" + std::to_string(i), RateEqualTo("outputSwizzle0")});
        addInternalScalar(ts.getBitBlockType(), "pendingSwizzleData" + std::to_string(i));
    }
    addInternalScalar(ts.getSizeTy(), "pendingOffset");
}

void SwizzledBitstreamCompressByCount::generateDoBlockMethod(KernelBuilder & b) {

    Type * fieldTy = b.getIntNTy(mFieldWidth);
    Type * blockTy = b.getBitBlockType();
    Value * countsPerStridePtr = b.getInputStreamBlockPtr("countsPerStride", b.getInt32(0));
    Value * countStreamPtr = b.CreatePointerCast(countsPerStridePtr, fieldTy->getPointerTo());

    // Output is written and committed to the output buffer one swizzle at a time.
    //
    Constant * blockOffsetMask = b.getSize(b.getBitBlockWidth() - 1);
    Constant * outputIndexShift = b.getSize(floor_log2(mFieldWidth));

    Value * outputProduced = b.getProducedItemCount("outputSwizzle0"); // All output groups have the same count.
    Value * producedOffset = b.CreateAnd(outputProduced, blockOffsetMask);
    Value * outputIndex = b.CreateLShr(producedOffset, outputIndexShift);

    // There may be pending data in the kernel state, for up to mFieldWidth-1 bits per stream.
    Value * pendingOffset = b.getScalarField("pendingOffset");
    // There is a separate vector of pending data for each swizzle group.
    std::vector<Value *> pendingData;
    std::vector<Value *> outputStreamPtr;
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        pendingData.push_back(b.getScalarField("pendingSwizzleData" + std::to_string(i)));
        outputStreamPtr.push_back(b.getOutputStreamBlockPtr("outputSwizzle" + std::to_string(i), b.getInt32(0)));
    }

    // Generate code for each of the mSwizzleFactor fields making up a block.
    // We load the count for the field and process all swizzle groups accordingly.
    for (unsigned i = 0; i < mSwizzleFactor; i++) {
        Value * newItemCount = b.CreateLoad(fieldTy, b.CreateGEP(fieldTy, countStreamPtr, b.getInt32(i)));
        Value * pendingSpace = b.CreateSub(b.getSize(mFieldWidth), pendingOffset);
        Value * pendingSpaceFilled = b.CreateICmpUGE(newItemCount, pendingSpace);

        Value * const fieldWidths = b.simd_fill(mFieldWidth, pendingOffset);

        // Data from the ith swizzle pack of each group is processed
        // according to the same newItemCount, pendingSpace, ...
        for (unsigned j = 0; j < mSwizzleSetCount; j++) {
            Value * newItems = b.loadInputStreamBlock("inputSwizzle" + std::to_string(j), b.getInt32(i));
            // Combine as many of the new items as possible into the pending group.
            Value * combinedGroup = b.CreateOr(pendingData[j], b.CreateShl(newItems, fieldWidths));
            // To avoid an unpredictable branch, always store the combined group, whether full or not.
            b.CreateBlockAlignedStore(combinedGroup, b.CreateGEP(blockTy, outputStreamPtr[j], outputIndex));
            // Any items in excess of the space available in the current pending group overflow for the next group.
            Value * overFlowGroup = b.CreateLShr(newItems, b.simd_fill(mFieldWidth, pendingSpace));
            // If we filled the space, then the overflow group becomes the new pending group and the index is updated.
            pendingData[j] = b.CreateSelect(pendingSpaceFilled, overFlowGroup, combinedGroup);
        }
        outputIndex = b.CreateSelect(pendingSpaceFilled, b.CreateAdd(outputIndex, b.getSize(1)), outputIndex);
        pendingOffset = b.CreateAnd(b.CreateAdd(newItemCount, pendingOffset), b.getSize(mFieldWidth-1));
    }
    b.setScalarField("pendingOffset", pendingOffset);
//    Value * newlyProduced = kb.CreateSub(kb.CreateShl(outputIndex, outputIndexShift), producedOffset);
//    Value * produced = kb.CreateAdd(outputProduced, newlyProduced);
    for (unsigned j = 0; j < mSwizzleSetCount; j++) {
        b.setScalarField("pendingSwizzleData" + std::to_string(j), pendingData[j]);
    }
//    kb.setProducedItemCount("outputSwizzle0", produced);
}

void SwizzledBitstreamCompressByCount::generateFinalBlockMethod(KernelBuilder & b, Value * /* remainingBytes */) {
    RepeatDoBlockLogic(b);
    Constant * blockOffsetMask = b.getSize(b.getBitBlockWidth() - 1);
    Constant * outputIndexShift = b.getSize(floor_log2(mFieldWidth));
    Type * blockTy = b.getBitBlockType();

    Value * outputProduced = b.getProducedItemCount("outputSwizzle0"); // All output groups have the same count.
    Value * producedOffset = b.CreateAnd(outputProduced, blockOffsetMask);
    Value * outputIndex = b.CreateLShr(producedOffset, outputIndexShift);
//    Value * pendingOffset = kb.getScalarField("pendingOffset");

    // Write the pending data.
    for (unsigned i = 0; i < mSwizzleSetCount; i++) {
        Value * pendingData = b.getScalarField("pendingSwizzleData" + std::to_string(i));
        Value * outputStreamPtr = b.getOutputStreamBlockPtr("outputSwizzle" + std::to_string(i), b.getInt32(0));
        b.CreateBlockAlignedStore(pendingData, b.CreateGEP(blockTy, outputStreamPtr, outputIndex));
    }
//    kb.setProducedItemCount("outputSwizzle0", kb.CreateAdd(pendingOffset, outputProduced));
}

const unsigned MIN_STREAMS_TO_SWIZZLE = 4;

FilterByMaskKernel::FilterByMaskKernel(LLVMTypeSystemInterface & ts,
                                         SelectOperation const & maskOp,
                                         SelectOperationList const & inputOps,
                                         StreamSet * filteredOutput,
                                         unsigned fieldWidth,
                                         ProcessingRateProbabilityDistribution insertionProbabilityDistribution)
: MultiBlockKernel(ts, "FilterByMask" + std::to_string(fieldWidth) + "_" +
                   streamutils::genSignature(maskOp) +
                   ":" + streamutils::genSignature(inputOps),
{}, {}, {}, {}, {})
, mFW(fieldWidth)
, mFieldsPerBlock(ts.getBitBlockWidth() / fieldWidth)
, mStreamCount(filteredOutput->getNumElements()) {
    mMaskOp.operation = maskOp.operation;
    mMaskOp.bindings.push_back(std::make_pair("extractionMask", maskOp.bindings[0].second));
    // assert (streamutil::resultStreamCount(maskOp) == 1);
    mInputStreamSets.push_back(Bind("extractionMask", maskOp.bindings[0].first, Principal()));
    //assert (streamutil::resultStreamCount(inputOps) == filteredOutput->getNumElements());
    std::unordered_map<StreamSet *, std::string> inputBindings;
    std::tie(mInputOps, inputBindings) = streamutils::mapOperationsToStreamNames(inputOps);
    for (auto const & kv : inputBindings) {
        mInputStreamSets.push_back(Bind(kv.second, kv.first, ZeroExtended()));
    }
    mOutputStreamSets.push_back(Bind("filteredOutput", filteredOutput, PopcountOf("extractionMask"), EmptyWriteOverflow()));
    assert (mOutputStreamSets.back().getDistribution().getTypeId() == insertionProbabilityDistribution.getTypeId());
    if (mStreamCount >= MIN_STREAMS_TO_SWIZZLE) {
        mPendingType = ts.getBitBlockType();
        mPendingSetCount = (mStreamCount + mFieldsPerBlock - 1)/mFieldsPerBlock;
    } else {
        mPendingType = ts.getIntNTy(mFW);
        mPendingSetCount = mStreamCount;
    }
    for (unsigned i = 0; i < mPendingSetCount; i++) {
        addInternalScalar(mPendingType, "pendingData" + std::to_string(i));
    }
    addInternalScalar(ts.getSizeTy(), "pendingOffset");
    setStride(16 * ts.getBitBlockWidth());
}

void FilterByMaskKernel::generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) {
    const auto Use_BMI_PEXT = kb.hasFeature(IDISA::Feature::AVX_BMI2);
    assert ((mStride % kb.getBitBlockWidth()) == 0);
    Constant * const sz_BLOCKS_PER_STRIDE = kb.getSize(mStride/kb.getBitBlockWidth());
    Constant * const sz_ZERO = kb.getSize(0);
    Constant * const sz_ONE = kb.getSize(1);
    Type * const sizeTy = kb.getSizeTy();
    Type * const fieldTy = kb.getIntNTy(mFW);
    Type * const FieldPtrTy = fieldTy->getPointerTo();
    Type * blockTy = kb.getBitBlockType();

    ConstantInt * const LOG_2_BLOCK_WIDTH = kb.getSize(floor_log2(kb.getBitBlockWidth()));
    ConstantInt * const BLOCK_WIDTH_MASK = kb.getSize(kb.getBitBlockWidth() - 1);
    ConstantInt * const FIELD_WIDTH = kb.getSize(mFW);
    ConstantInt * const LOG_2_FIELD_WIDTH = kb.getSize(floor_log2(mFW));
    ConstantInt * const FIELD_WIDTH_MASK = kb.getSize(mFW - 1);

    BasicBlock * const entryBlock = kb.GetInsertBlock();
    BasicBlock * const stridePrologue = kb.CreateBasicBlock("stridePrologue");
    BasicBlock * processBlock = kb.CreateBasicBlock("processBlock");
    BasicBlock * doExtract = kb.CreateBasicBlock("doExtract");
    BasicBlock * noExtract = kb.CreateBasicBlock("noExtract");
    BasicBlock * const strideEpilogue = kb.CreateBasicBlock("strideEpilogue");
    BasicBlock * finalizeFilter = kb.CreateBasicBlock("finalizeFilter");
    BasicBlock * writeFinal = kb.CreateBasicBlock("writeFinal");
    BasicBlock * done = kb.CreateBasicBlock("done");

    Value * const baseOutputProduced = kb.getProducedItemCount("filteredOutput");
    Value * const baseProducedOffset = kb.CreateAnd(baseOutputProduced, BLOCK_WIDTH_MASK);
    std::vector<Value *> pendingDataPtr(mPendingSetCount);
    for (unsigned i = 0; i < mPendingSetCount; i++) {
        Value * pending = kb.getScalarField("pendingData" + std::to_string(i));
        pendingDataPtr[i] = kb.CreateAllocaAtEntryPoint(pending->getType());
        kb.CreateStore(pending, pendingDataPtr[i]);
    }
    Value * produceOffsetPtr = kb.CreateAllocaAtEntryPoint(baseProducedOffset->getType());
    kb.CreateStore(baseProducedOffset, produceOffsetPtr);

    kb.CreateBr(stridePrologue);
    kb.SetInsertPoint(stridePrologue);
    PHINode * const strideNo = kb.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    Value * strideBlockOffset = kb.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = kb.CreateAdd(strideNo, sz_ONE);

    kb.CreateBr(processBlock);
    kb.SetInsertPoint(processBlock);
    // For each block iteration, we have the block offset and pending data
    // for each swizzle that are either the initial values or values carried
    // over from the previous iteration.
    PHINode * blockNo = kb.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    Value * nextBlk = kb.CreateAdd(blockNo, sz_ONE);
    Value * moreBlocksInStride = kb.CreateICmpNE(nextBlk, sz_BLOCKS_PER_STRIDE);

    Value * strideBlockIndex = kb.CreateAdd(strideBlockOffset, blockNo);

    std::vector<Value *> maskVec = streamutils::loadInputSelectionsBlock(kb, {mMaskOp}, strideBlockIndex);
    Value * extractionMask = kb.fwCast(mFW, maskVec[0]);
    kb.CreateCondBr(kb.bitblock_any(extractionMask), doExtract, noExtract);

    kb.SetInsertPoint(noExtract);
    blockNo->addIncoming(nextBlk, noExtract);
    kb.CreateCondBr(moreBlocksInStride, processBlock, strideEpilogue);

    kb.SetInsertPoint(doExtract);
    std::vector<Value *> mask(mFieldsPerBlock);
    for (unsigned i = 0; i < mFieldsPerBlock; i++) {
        mask[i] = kb.CreateExtractElement(extractionMask, kb.getInt32(i));
    }
    std::vector<Value *> input = streamutils::loadInputSelectionsBlock(kb, mInputOps, strideBlockIndex);
    for (unsigned j = 0; j < input.size(); ++j) {
        if (!Use_BMI_PEXT) {
            input[j] = kb.simd_pext(mFW, input[j], extractionMask);
        }
        input[j] = kb.fwCast(mFW, input[j]);
    }
    Value * const newItemCounts = kb.simd_popcount(mFW, extractionMask);
    //kb.CallPrintRegister("extractionMask", extractionMask);
    // For each swizzle containing mFieldsPerBlock fields.
    for (unsigned i = 0; i < mFieldsPerBlock; i++) {
        Value * producedOffset = kb.CreateLoad(sizeTy, produceOffsetPtr);
        std::vector<Value *> pendingData(mPendingSetCount);
        for (unsigned i = 0; i < mPendingSetCount; i++) {
            pendingData[i] = kb.CreateLoad(mPendingType, pendingDataPtr[i]);
        }
        Value * const pendingOffset = kb.CreateAnd(producedOffset, FIELD_WIDTH_MASK);
        Value * const newItemCount = kb.CreateExtractElement(newItemCounts, i);
        //kb.CallPrintInt("newItemCount", newItemCount);
        Value * const pendingSpace = kb.CreateSub(FIELD_WIDTH, pendingOffset);
        Value * const maskedSpace = kb.CreateAnd(pendingSpace, FIELD_WIDTH_MASK);
        Value * const pendingSpaceFilled = kb.CreateICmpUGE(newItemCount, pendingSpace);
        Value * const outputBlock = kb.CreateLShr(producedOffset, LOG_2_BLOCK_WIDTH);
        Value * const fieldIndex = kb.CreateLShr(kb.CreateAnd(producedOffset, BLOCK_WIDTH_MASK), LOG_2_FIELD_WIDTH);
        if (mStreamCount < MIN_STREAMS_TO_SWIZZLE) {
            for (unsigned j = 0; j < input.size(); ++j) {
                Value * field = kb.CreateExtractElement(input[j], kb.getInt32(i));
                Value * compressed =
                    Use_BMI_PEXT ? kb.CreatePextract(field, mask[i]) : field;
                Value * const shiftedItems = kb.CreateShl(compressed, kb.CreateZExtOrTrunc(pendingOffset, fieldTy));
                Value * const combined = kb.CreateOr(pendingData[j], shiftedItems);
                Value * outputPtr = kb.getOutputStreamBlockPtr("filteredOutput", kb.getInt32(j), outputBlock);
                outputPtr = kb.CreatePointerCast(outputPtr, FieldPtrTy);
                outputPtr = kb.CreateGEP(fieldTy, outputPtr, fieldIndex);
                kb.CreateStore(combined, outputPtr);
                Value * overFlow = kb.CreateLShr(compressed, kb.CreateZExtOrTrunc(maskedSpace, fieldTy));
                overFlow = kb.CreateSelect(kb.CreateIsNull(maskedSpace), ConstantInt::getNullValue(overFlow->getType()), overFlow);
                // If we filled the space, then the overflow group becomes the new pending group and the index is updated.
                Value * newPending = kb.CreateSelect(pendingSpaceFilled, overFlow, combined);
                kb.CreateStore(newPending, pendingDataPtr[j]);
            }
        } else {
            std::vector<Value *> swizzles(mPendingSetCount, ConstantInt::getNullValue(blockTy));
            for (unsigned j = 0; j < input.size(); ++j) {
                unsigned swizzleNo = j/mFieldsPerBlock;
                Value * field = kb.CreateExtractElement(input[j], kb.getInt32(i));
                Value * compressed = Use_BMI_PEXT ? kb.CreatePextract(field, mask[i]) : field;
                swizzles[swizzleNo] = kb.CreateInsertElement(swizzles[swizzleNo], compressed, j%mFieldsPerBlock);
                //kb.CallPrintRegister("swizzles" + std::to_string(swizzleNo), swizzles[swizzleNo]);
            }
            // Field compression into the swizzles is now complete.   Next we apply
            // stream compression to compress the fields of each swizzle and generate the
            // output.
            // Generate code for each of the mFieldsPerBlock fields making up a block.
            // We load the count for the field and process all swizzle groups accordingly.

            Value * const shiftVector = kb.simd_fill(mFW, pendingOffset);
            Value * spaceVector = kb.simd_fill(mFW, maskedSpace);
            // Data from the ith swizzle pack of each group is processed
            // according to the same newItemCount, pendingSpace, ...
            for (unsigned j = 0; j < mPendingSetCount; j++) {
                Value * const newItems = swizzles[j];
                // Combine as many of the new items as possible into the pending group.
                Value * const shiftedItems = kb.CreateShl(swizzles[j], shiftVector);
                Value * const combinedGroup = kb.CreateOr(pendingData[j], shiftedItems);
                //kb.CallPrintRegister("combinedGroup" + std::to_string(j), combinedGroup);
                // To avoid an unpredictable branch, always store the combined group, whether full or not.
                for (unsigned k = 0; k < mFieldsPerBlock; k++) {
                    unsigned strmIdx = j * mFieldsPerBlock + k;
                    Value * outputPtr = kb.getOutputStreamBlockPtr("filteredOutput", kb.getInt32(strmIdx), outputBlock);
                    outputPtr = kb.CreatePointerCast(outputPtr, FieldPtrTy);
                    outputPtr = kb.CreateGEP(fieldTy, outputPtr, fieldIndex);
                    kb.CreateStore(kb.CreateExtractElement(combinedGroup, kb.getInt32(k)), outputPtr);
                }
                // Any items in excess of the space available in the current pending group overflow for the next group.
                // However, we need to avoid a poison value arising from a shift by more than mFW-1.
                Value * overFlowGroup = kb.CreateLShr(newItems, spaceVector);
                overFlowGroup = kb.CreateSelect(kb.CreateIsNull(maskedSpace), ConstantInt::getNullValue(overFlowGroup->getType()), overFlowGroup);
                // If we filled the space, then the overflow group becomes the new pending group and the index is updated.
                Value * newPending = kb.CreateSelect(pendingSpaceFilled, overFlowGroup, combinedGroup);
                kb.CreateStore(newPending, pendingDataPtr[j]);
                //kb.CallPrintRegister("pendingData" + std::to_string(j), pendingData[j]);
            }
        }
        producedOffset = kb.CreateAdd(producedOffset, newItemCount);
        kb.CreateStore(producedOffset, produceOffsetPtr);
   }
    BasicBlock * currentBB = kb.GetInsertBlock();
    blockNo->addIncoming(nextBlk, currentBB);
    kb.CreateCondBr(moreBlocksInStride, processBlock, strideEpilogue);

    kb.SetInsertPoint(strideEpilogue);
    strideNo->addIncoming(nextStrideNo, strideEpilogue);
    kb.CreateCondBr(kb.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, finalizeFilter);

    kb.SetInsertPoint(finalizeFilter);
    for (unsigned i = 0; i < mPendingSetCount; i++) {
        kb.setScalarField("pendingData" + std::to_string(i), kb.CreateLoad(mPendingType, pendingDataPtr[i]));
    }
    // If we are in the final stride of the input stream, we must write
    // out any pending data.   However, if the producedOffset ends on
    // a field boundary, there is no data to write.   We avoid the
    // write in this case to ensure that we do not write past the end
    // of the allocated buffer.
    Value * producedOffset = kb.CreateLoad(sizeTy, produceOffsetPtr);
    Value * havePendingFieldData = kb.CreateIsNotNull(kb.CreateAnd(producedOffset, FIELD_WIDTH_MASK));
    kb.CreateCondBr(kb.CreateAnd(kb.isFinal(), havePendingFieldData), writeFinal, done);
    kb.SetInsertPoint(writeFinal);
    Value * const finalBlock = kb.CreateLShr(producedOffset, LOG_2_BLOCK_WIDTH);
    Value * const finalField = kb.CreateLShr(kb.CreateAnd(producedOffset, BLOCK_WIDTH_MASK), LOG_2_FIELD_WIDTH);
    if (mStreamCount < MIN_STREAMS_TO_SWIZZLE) {
        for (unsigned j = 0; j < mStreamCount; j++) {
            Value * outputPtr = kb.getOutputStreamBlockPtr("filteredOutput", kb.getInt32(j), finalBlock);
            outputPtr = kb.CreatePointerCast(outputPtr, FieldPtrTy);
            outputPtr = kb.CreateGEP(fieldTy, outputPtr, finalField);
            kb.CreateStore(kb.CreateLoad(mPendingType, pendingDataPtr[j]), outputPtr);
        }
    } else {
        for (unsigned j = 0; j < mPendingSetCount; j++) {
            for (unsigned k = 0; k < mFieldsPerBlock; k++) {
                unsigned strmIdx = j * mFieldsPerBlock + k;
                Value * outputPtr = kb.getOutputStreamBlockPtr("filteredOutput", kb.getInt32(strmIdx), finalBlock);
                outputPtr = kb.CreatePointerCast(outputPtr, FieldPtrTy);
                outputPtr = kb.CreateGEP(fieldTy, outputPtr, finalField);
                kb.CreateStore(kb.CreateExtractElement(kb.CreateLoad(mPendingType, pendingDataPtr[j]), kb.getInt32(k)), outputPtr);
            }
        }
    }
    kb.CreateBr(done);
    kb.SetInsertPoint(done);
}


ByteFilterByMaskKernel::ByteFilterByMaskKernel(LLVMTypeSystemInterface & b, StreamSet * const byteStream, StreamSet * const filter, StreamSet * const output, Scalar * streamOffset)
: MultiBlockKernel(b, [&]() {
    std::string tmp;
    raw_string_ostream nm(tmp);
    nm << "ByteFilterByMask" << output->getNumElements() << 'x' << byteStream->getFieldWidth();
    if (streamOffset) {
        nm << 'S';
    }
    nm.flush();
    return tmp;
}(),
{Binding{"byteStream", byteStream}, Binding{"filter", filter}},
{Binding{"output", output, PopcountOf("filter"), EmptyWriteOverflow()}}, {}, {}, {}) {
    assert (byteStream->getFieldWidth() == output->getFieldWidth());
    if (streamOffset) {
        mInputScalars.emplace_back("offset", streamOffset);
    }
}

void ByteFilterByMaskKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");

    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);

    const auto fieldWidth = getInputStreamSet(0)->getFieldWidth();

    StreamSet * const output = getOutputStreamSet(0);
    if (LLVM_UNLIKELY(output->getFieldWidth() != fieldWidth)) {
        report_fatal_error(Twine{getName(), ": input field width does not match output field width"});
    }

    const auto numElements = output->getNumElements();
    const auto numInputElements = getInputStreamSet(0)->getNumElements();
    if (numElements > getInputStreamSet(0)->getNumElements()) {
        report_fatal_error(Twine{getName(), ": number of output streams exceeds the input streamset size"});
    }

    const auto fieldsPerBlock = (b.getBitBlockWidth() / fieldWidth);

    ConstantInt * const BLOCK_WIDTH_MASK = b.getSize(b.getBitBlockWidth() - 1);

    ConstantInt * const FIELD_WIDTH_MASK = b.getSize(fieldWidth - 1);

    ConstantInt * const LOG_2_FIELD_WIDTH = b.getSize(floor_log2(fieldWidth));

    ConstantInt * const FIELDS_PER_BLOCK = b.getSize(fieldsPerBlock);

    ConstantInt * const FIELDS_PER_BLOCK_MASK = b.getSize(fieldsPerBlock - 1);

    ConstantInt * const LOG_2_FIELDS_PER_BLOCK = b.getSize(floor_log2(fieldsPerBlock));

    FixedVectorType * dataVecTy = b.fwVectorType(fieldWidth);

    Value * initToWritePos = b.getProducedItemCount("output");

    if (numElements == 1 && numInputElements == 1) {

        Value * const totalBlocks = b.CreateMul(numOfStrides, b.getSize(fieldWidth));
        Value * const baseDataPtr = b.getInputStreamPackPtr("byteStream", sz_ZERO, sz_ZERO);
        assert (fieldWidth <= b.getBitBlockWidth());
        const auto popCountSize = b.getBitBlockWidth() / fieldWidth;

        Value * const baseFilterPtr = b.getInputStreamPackPtr("filter", sz_ZERO, sz_ZERO); ;

        b.CreateBr(packLoop);

        b.SetInsertPoint(packLoop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);
        PHINode * const toWritePosPhi = b.CreatePHI(b.getSizeTy(), 2);
        toWritePosPhi->addIncoming(initToWritePos, entry);

        Value * filter = nullptr;
        if (popCountSize < 8) {
            Value * ptr = b.CreatePointerCast(baseFilterPtr, b.getInt8Ty()->getPointerTo());
            const auto packsPerByte = (8 / popCountSize);
            assert (packsPerByte > 1);
            Value * pos = b.CreateLShr(blockOffsetPhi, b.getSize(floor_log2(packsPerByte)));
            filter = b.CreateAlignedLoad(b.getInt8Ty(), b.CreateGEP(b.getInt8Ty(), ptr, pos), 1);
            filter = b.CreateZExt(filter, b.getSizeTy());
            Value * off = b.CreateAnd(blockOffsetPhi, b.getSize(packsPerByte - 1));
            filter = b.CreateLShr(filter, b.CreateShl(off, b.getSize(floor_log2(popCountSize))));
            filter = b.CreateAnd(filter, b.getSize((1UL << popCountSize) - 1UL));
        } else {
            IntegerType * const popCountTy = b.getIntNTy(popCountSize);
            Value * ptr = b.CreatePointerCast(baseFilterPtr, popCountTy->getPointerTo());
            filter = b.CreateAlignedLoad(popCountTy, b.CreateGEP(popCountTy, ptr, blockOffsetPhi), popCountSize / 8);
        }

        Value * const data = b.CreateAlignedLoad(dataVecTy, b.CreateGEP(dataVecTy, baseDataPtr, blockOffsetPhi), b.getBitBlockWidth() / 8);
        Value * const compressed = b.mvmd_compress(fieldWidth, data, filter);

        Value * const ptr = b.getRawOutputPointer("output", toWritePosPhi);

        Value * const toStorePtr = b.CreatePointerCast(ptr, compressed->getType()->getPointerTo());
        b.CreateAlignedStore(compressed, toStorePtr, 1);

        Value * const elementPopCount = b.CreatePopcount(filter);
        Value * toWritePos = b.CreateAdd(toWritePosPhi, b.CreateZExt(elementPopCount, b.getSizeTy()));

        Value * nextBlk = b.CreateAdd(blockOffsetPhi, sz_ONE);
        blockOffsetPhi->addIncoming(nextBlk, packLoop);

        toWritePosPhi->addIncoming(toWritePos, packLoop);
        Value * moreToDo = b.CreateICmpNE(nextBlk, totalBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);

        b.SetInsertPoint(packFinalize);

    } else {

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

        Value * initialPosition = b.CreateAnd(initToWritePos, BLOCK_WIDTH_MASK);
        Value * initialPackIndex = b.CreateLShr(initialPosition, LOG_2_FIELDS_PER_BLOCK);

        SmallVector<Value *, 32> pending(numElements);
        SmallVector<PHINode *, 32> pendingPhi(numElements);

        Constant * ZERO_VEC = ConstantVector::getNullValue(dataVecTy);

        for (unsigned i = 0; i < numElements; ++i) {
            Value * streamIndex = b.getSize(i);
            Value * ptr = b.getOutputStreamPackPtr("output", streamIndex, initialPackIndex, sz_ZERO);
            pending[i] = b.CreateAlignedLoad(dataVecTy, ptr, b.getBitBlockWidth() / 8);
        }

        b.CreateBr(packLoop);

        b.SetInsertPoint(packLoop);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(sz_ZERO, entry);

        PHINode * const shiftPhi = b.CreatePHI(b.getSizeTy(), 2);
        shiftPhi->addIncoming(initialPosition, entry);

        PHINode * const packIndexPhi = b.CreatePHI(b.getSizeTy(), 2);
        packIndexPhi->addIncoming(initialPackIndex, entry);

        for (unsigned i = 0; i < numElements; ++i) {
            PHINode * pPhi = b.CreatePHI(dataVecTy, 2);
            pPhi->addIncoming(pending[i], entry);
            pendingPhi[i] = pPhi;
            pending[i] = pPhi;
        }

        Value * filterVec = b.loadInputStreamBlock("filter", sz_ZERO, blockOffsetPhi);

        VectorType * popVecTy = FixedVectorType::get(b.getIntNTy(b.getBitBlockWidth() / fieldWidth), fieldWidth);

        filterVec = b.CreateBitCast(filterVec, popVecTy);

        SmallVector<Value *, 65> shiftVal(fieldWidth + 1);
        SmallVector<Value *, 65> fieldPackIndex(fieldWidth + 1);

        SmallVector<Value *, 64> filterElem(fieldWidth);
        SmallVector<Value *, 64> leftShift(fieldWidth);
        SmallVector<Value *, 64> packIndex(fieldWidth);
        SmallVector<Value *, 64> blockIndex(fieldWidth);
        SmallVector<Value *, 64> rightShift(fieldWidth);
        SmallVector<Value *, 64> keepCurrentPending(fieldWidth);


        for (unsigned i = 0; i < numElements; ++i) {

            ConstantInt * outputStreamIndex = b.getSize(i);
            Value * inputStreamIndex = outputStreamIndex;
            if (baseStreamIndex) {
                inputStreamIndex = b.CreateAdd(outputStreamIndex, baseStreamIndex);
            }

            shiftVal[0] = shiftPhi;
            fieldPackIndex[0] = packIndexPhi;

            for (unsigned j = 0; j < fieldWidth; ++j) {
                if (i == 0) {
                    filterElem[j] = b.CreateExtractElement(filterVec, b.getInt32(j));
                    Value * const elementPopCount = b.CreateZExt(b.CreatePopcount(filterElem[j]), b.getSizeTy());
                    leftShift[j] = b.CreateAnd(shiftVal[j], FIELDS_PER_BLOCK_MASK);
                    packIndex[j] = b.CreateAnd(fieldPackIndex[j], FIELD_WIDTH_MASK);
                    blockIndex[j] = b.CreateLShr(fieldPackIndex[j], LOG_2_FIELD_WIDTH);
                    rightShift[j] = b.CreateSub(FIELDS_PER_BLOCK, leftShift[j]);
                    Value * nextShiftVal = b.CreateAdd(shiftVal[j], elementPopCount);
                    Value * nextPackIndex = b.CreateLShr(nextShiftVal, LOG_2_FIELDS_PER_BLOCK);
                    keepCurrentPending[j] = b.CreateICmpEQ(fieldPackIndex[j], nextPackIndex);
                    fieldPackIndex[j + 1] = nextPackIndex;
                    shiftVal[j + 1] = nextShiftVal;
                }
                Value * const data = b.loadInputStreamPack("byteStream", inputStreamIndex, b.getSize(j), blockOffsetPhi);
                Value * const compressed = b.mvmd_compress(fieldWidth, data, filterElem[j]);
                Value * lshiftVal = b.mvmd_sll(fieldWidth, compressed, leftShift[j]);
                Value * toWriteVal = b.CreateOr(pending[i], lshiftVal);
                b.storeOutputStreamPack("output", outputStreamIndex, packIndex[j], blockIndex[j], toWriteVal);
                Value * rshiftVal = b.mvmd_srl(fieldWidth, compressed, rightShift[j]);
                rshiftVal = b.CreateSelect(b.CreateICmpNE(leftShift[j], sz_ZERO), rshiftVal, ZERO_VEC);
                pending[i] = b.CreateSelect(keepCurrentPending[j], toWriteVal, rshiftVal);
            }
        }

        for (unsigned i = 0; i < numElements; ++i) {
            pendingPhi[i]->addIncoming(pending[i], packLoop);
        }

        packIndexPhi->addIncoming(fieldPackIndex[fieldWidth], packLoop);

        shiftPhi->addIncoming(shiftVal[fieldWidth], packLoop);

        Value * nextBlk = b.CreateAdd(blockOffsetPhi, sz_ONE);
        blockOffsetPhi->addIncoming(nextBlk, packLoop);

        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfStrides);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);

        b.SetInsertPoint(packFinalize);
        Value * packIdx = b.CreateAnd(fieldPackIndex[fieldWidth], FIELD_WIDTH_MASK);
        Value * blkIdx = b.CreateLShr(fieldPackIndex[fieldWidth], LOG_2_FIELD_WIDTH);
        for (unsigned i = 0; i < numElements; ++i) {
            b.storeOutputStreamPack("output", b.getSize(i), packIdx, blkIdx, pending[i]);
        }



    }


}

}
