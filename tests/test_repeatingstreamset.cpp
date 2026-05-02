/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <boost/integer/common_factor_rt.hpp>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>

#include <llvm/IR/Instructions.h>

using namespace kernel;
using namespace llvm;
using namespace testing;
using namespace boost::integer;

static cl::opt<unsigned> optFieldWidth("field-width", cl::desc("Field width of pattern elements"), cl::init(0));

static cl::opt<unsigned> optNumElements("num-elements", cl::desc("Number of elements in pattern streamset"), cl::init(0));

static cl::opt<unsigned> optPatternLength("pattern-length", cl::desc("Length of each pattern"), cl::init(0));

static cl::opt<unsigned> optRepetitionLength("repetition-length", cl::desc("Total length of repeating data"), cl::init(0));

static cl::opt<bool> optAllowUnaligned("allow-unaligned", cl::desc("Allow unaligned access to single stream streamsets"), cl::init(true));

static cl::opt<unsigned> optUseNestedPipeline("nested", cl::desc("Depth of nested pipeline before repeating streamset comparison (0, 1 or 2)"), cl::init(0));

static cl::opt<unsigned> optNumOfTrials("trials", cl::desc("Number of tests to perform (default: 10)"), cl::init(10));

static cl::opt<bool> optUseFamilyCall("family", cl::desc("Execute nested pipeline using family kernel call"), cl::init(false));

static cl::opt<bool> optVerbose("v", cl::desc("Print verbose output"), cl::init(false));

class RepeatingSourceKernel final : public SegmentOrientedKernel {
public:
    RepeatingSourceKernel(LLVMTypeSystemInterface & driver, std::vector<std::vector<uint64_t>> pattern, StreamSet * output, Scalar * repLength, const unsigned fillSize = 1024);
protected:
    void generateDoSegmentMethod(KernelBuilder & b) override;
    StringRef getSignature() const override { return Signature; }
    bool hasSignature() const override { return true; }
private:
    LLVM_READNONE static std::string makeSignature(const std::vector<std::vector<uint64_t>> & pattern, const StreamSet * const output, const unsigned fillSize);
private:
    const std::string Signature;
    const std::vector<std::vector<uint64_t>> Pattern;
};

std::string RepeatingSourceKernel::makeSignature(const std::vector<std::vector<uint64_t>> & pattern, const StreamSet * const output, const unsigned fillSize) {
    std::string tmp;
    tmp.reserve(200);
    raw_string_ostream out(tmp);
    out << "repeating" << output->getFieldWidth() << 'C' << fillSize << ":{";
    for (const auto & vec : pattern) {
        char joiner = '{';
        for (const auto c : vec) {
            out << joiner;
            out.write_hex(c);
            joiner = ',';
        }
        out << '}';
    }
    out << "}";
    out.flush();
    return tmp;
}

RepeatingSourceKernel::RepeatingSourceKernel(LLVMTypeSystemInterface & ts, std::vector<std::vector<uint64_t>> pattern, StreamSet * output, Scalar * repLength, const unsigned fillSize)
: SegmentOrientedKernel(ts, "RS_" + getStringHash(makeSignature(pattern, output, fillSize)),
// input streams
{},
// output stream
{Binding{"output", output, BoundedRate(0, fillSize), { ManagedBuffer() }}},
// input scalar
{Binding{ts.getSizeTy(), "repLength", repLength}},
{},
// internal scalar
{})
, Signature(makeSignature(pattern, output, fillSize))
, Pattern(std::move(pattern)) {
    addAttribute(MustExplicitlyTerminate());
    setStride(1);
}

void RepeatingSourceKernel::generateDoSegmentMethod(KernelBuilder & b) {

    BasicBlock * const generateData = b.CreateBasicBlock("generateData");
    BasicBlock * const finishedDataLoop = b.CreateBasicBlock("finishedDataLoop");
    BasicBlock * const zeroExtraneousBytes = b.CreateBasicBlock("zeroExtraneousBytes");
    BasicBlock * const zeroExtraneousBlocks = b.CreateBasicBlock("zeroExtraneousBlocks");
    BasicBlock * const exit = b.CreateBasicBlock("exit");

    // build our pattern array
    const auto output = b.getOutputStreamSet("output");
    const auto fieldWidth = output->getFieldWidth();
    const auto numElements = output->getNumElements();

    const auto blockWidth = b.getBitBlockWidth();

    const auto maxVal = (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL;

    // size_t patternLength = blockWidth;
    size_t maxPatternSize = 0;
    for (unsigned i = 0; i < numElements; ++i) {
        const auto & vec = Pattern[i];
        for (auto v : vec) {
            if (LLVM_UNLIKELY(v > maxVal)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << "Value " << v << " exceeds a " << fieldWidth << "-bit value";
                report_fatal_error(StringRef(msg.str()));
            }
        }
        maxPatternSize = std::max(maxPatternSize, vec.size());
    }
    const Binding & binding = b.getOutputStreamSetBinding("output");
    const ProcessingRate & rate = binding.getRate();
    const auto fs = getStride() * rate.getUpperBound();
    assert (fs.denominator() == 1);
    const auto maxFillSize = fs.numerator();
    if (maxPatternSize > maxFillSize) {
        report_fatal_error("output rate should at least be as large as the pattern length");
    }

    if (fieldWidth > blockWidth) {
        report_fatal_error(StringRef("does not support field width sizes above ") + std::to_string(blockWidth));
    }
    if ((maxFillSize % blockWidth) != 0) {
        report_fatal_error(StringRef("output rate should be a multiple of ") + std::to_string(blockWidth)
                           + " to ensure proper streamset construction");
    }

    StreamSetBuffer * const outputBuffer = b.getOutputStreamSetBuffer("output");

    ConstantInt * const sz_ZERO = b.getSize(0);

    FixedVectorType * const vecTy = b.getBitBlockType();
    IntegerType * const intTy = cast<IntegerType>(vecTy->getScalarType());
    const auto laneWidth = intTy->getIntegerBitWidth();
    const auto numLanes = blockWidth / laneWidth;
    ArrayType * const elementTy = ArrayType::get(vecTy, fieldWidth);



    SmallVector<Constant *, 16> laneVal(numLanes);
    SmallVector<Constant *, 16> packVal(fieldWidth);
    SmallVector<GlobalVariable *, 16> streamVal(numElements);
    SmallVector<Type *, 16> streamValTy(numElements);

    Module & mod = *b.getModule();

    assert ((fieldWidth * blockWidth) % 8 == 0);
    assert ((fieldWidth * blockWidth) >= 8);

    for (unsigned p = 0; p < numElements; ++p) {
        const auto & vec = Pattern[p];
        const auto L = vec.size();
        const auto patternLength = boost::lcm<size_t>(blockWidth, L);
        const auto runLength = (patternLength / blockWidth);

        std::vector<Constant *> dataVectorArray(runLength);

        uint64_t pos = 0;
        for (unsigned r = 0; r < runLength; ++r) {
            for (uint64_t i = 0; i < fieldWidth; ++i) {
                for (uint64_t j = 0; j < numLanes; ++j) {
                    uint64_t V = 0;
                    for (uint64_t k = 0; k < laneWidth; k += fieldWidth) {
                        const auto v = vec[pos % L];
                        V |= (v << k);
                        ++pos;
                    }
                    laneVal[j] = ConstantInt::get(intTy, V, false);
                }
                packVal[i] = ConstantVector::get(laneVal);
            }
            dataVectorArray[r] = ConstantArray::get(elementTy, packVal);
            assert (b.getTypeSize(mod.getDataLayout(), dataVectorArray[r]->getType()) == (fieldWidth * blockWidth) / 8);
        }

        ArrayType * const streamTy = ArrayType::get(elementTy, runLength);
        streamValTy[p] = streamTy;
        Constant * const patternVal = ConstantArray::get(streamTy, dataVectorArray);
        GlobalVariable * const gv = new GlobalVariable(mod, streamTy, true, GlobalValue::PrivateLinkage, patternVal);
        gv->setAlignment(MaybeAlign{blockWidth /8});
        assert (streamTy->getPointerTo() == gv->getType());

        streamVal[p] = gv;
    }

    ConstantInt * const sz_BlockWidth = b.getSize(blockWidth);

    ConstantInt * const sz_strideFillSize = b.getSize(maxFillSize);

    Value * const produced = b.getProducedItemCount("output");
    Value * const consumed = b.getConsumedItemCount("output");
    Value * const fillSize = b.CreateMul(sz_strideFillSize, b.getNumOfStrides());

    b.reserveCapacity("output", fillSize);

    Value * const baseAddress = b.getBaseAddress("output");

    Value * const MAX = b.getScalarField("repLength");

    Value * const total = b.CreateUMin(MAX, b.CreateAdd(fillSize, consumed)); // produced + (fillSize - (produced - consumed))

    BasicBlock * const prepareBufferExit = b.GetInsertBlock();
    b.CreateBr(generateData);

    b.SetInsertPoint(generateData);
    PHINode * const producedPhi = b.CreatePHI(b.getSizeTy(), 3);
    producedPhi->addIncoming(produced, prepareBufferExit);

    Value * const currentIndex = b.CreateExactUDiv(producedPhi, sz_BlockWidth);

    const auto align = (blockWidth / 8);

    FixedArray<Value *, 3> offset;
    offset[0] = sz_ZERO;

    if (fieldWidth == 1) {

        offset[2] = sz_ZERO; // fieldwidth = 0

        for (unsigned i = 0; i < numElements; ++i) {
            const auto patternLength = boost::lcm<size_t>(blockWidth, Pattern[i].size());
            const auto runLength = (patternLength / blockWidth);
            offset[1] = b.CreateURem(currentIndex, b.getSize(runLength));
            Value * const src = b.CreateGEP(streamValTy[i], streamVal[i], offset);
            Value * const srcVal = b.CreateAlignedLoad(vecTy, src, blockWidth / 8);
            Value * const dst = outputBuffer->getStreamBlockPtr(b, baseAddress, b.getInt32(i), currentIndex);
            b.CreateAlignedStore(srcVal, dst, align);
        }


    } else {



        SmallVector<Constant *, 16> fieldOffset(fieldWidth);
        for (unsigned j = 0; j < fieldWidth; ++j) {
            fieldOffset[j] = b.getSize(j);
        }

        for (unsigned i = 0; i < numElements; ++i) {
            const auto patternLength = boost::lcm<size_t>(blockWidth, Pattern[i].size());
            const auto runLength = (patternLength / blockWidth);
            offset[1] = b.CreateURem(currentIndex, b.getSize(runLength));
            for (unsigned j = 0; j < fieldWidth; ++j) {
                offset[2] = fieldOffset[j];
                Value * const src = b.CreateGEP(streamValTy[i], streamVal[i], offset);
                Value * const srcVal = b.CreateAlignedLoad(vecTy, src, blockWidth / 8);
                Value * const dst = outputBuffer->getStreamPackPtr(b, baseAddress, b.getInt32(i), currentIndex, fieldOffset[j]);
                b.CreateAlignedStore(srcVal, dst, align);
            }
        }
    }


    Value * const currentProduced = b.CreateAdd(producedPhi, sz_BlockWidth);
    BasicBlock * const generateDataExit = b.GetInsertBlock();
    producedPhi->addIncoming(currentProduced, generateDataExit);
    b.CreateCondBr(b.CreateICmpULT(currentProduced, total), generateData, finishedDataLoop);

    b.SetInsertPoint(finishedDataLoop);
    b.setProducedItemCount("output", currentProduced);

    Value * const finishedGenerating = b.CreateICmpUGE(currentProduced, MAX);
    b.CreateUnlikelyCondBr(finishedGenerating, zeroExtraneousBytes, exit);

    b.SetInsertPoint(zeroExtraneousBytes);
    b.setProducedItemCount("output", MAX);
    b.setTerminationSignal();

    Value * const startIndex = b.CreateUDiv(MAX, sz_BlockWidth);

    ConstantInt * const sz_BlockMask = b.getSize(blockWidth - 1U);
    ConstantInt * const sz_FieldWidth = b.getSize(fieldWidth);

    Value * packIndex = nullptr;
    Value * maskOffset = b.CreateAnd(MAX, sz_BlockMask);
    if (fieldWidth > 1) {
        Value * const position = b.CreateMul(maskOffset, sz_FieldWidth);
        packIndex = b.CreateUDiv(position, sz_BlockWidth);
        maskOffset = b.CreateAnd(position, sz_BlockMask);
    }

    Value * const mask = b.CreateNot(b.bitblock_mask_from(maskOffset));
    for (unsigned i = 0; i < numElements; ++i) {
        Value * ptr = nullptr;
        if (fieldWidth == 1) {
            ptr = outputBuffer->getStreamBlockPtr(b, baseAddress, b.getInt32(i), startIndex);
        } else {
            ptr = outputBuffer->getStreamPackPtr(b, baseAddress, b.getInt32(i), startIndex, packIndex);
        }
        Value * const val = b.CreateBlockAlignedLoad(vecTy, ptr);
        Value * const maskedVal = b.CreateAnd(val, mask);
        b.CreateAlignedStore(maskedVal, ptr, align);
    }


    ConstantInt * const sz_ONE = b.getSize(1);

    if (fieldWidth > 1) {
        BasicBlock * const clearRemainingPacks = b.CreateBasicBlock("clearRemainingPacks", zeroExtraneousBlocks);
        BasicBlock * const clearRemainingPacksExit = b.CreateBasicBlock("clearRemainingPacksExit", zeroExtraneousBlocks);

        Constant * const vec_ZERO = ConstantVector::getNullValue(vecTy);

        Value * const firstPackIndex = b.CreateAdd(packIndex, sz_ONE);

        b.CreateCondBr(b.CreateICmpULT(firstPackIndex, sz_FieldWidth), clearRemainingPacks, clearRemainingPacksExit);

        b.SetInsertPoint(clearRemainingPacks);
        PHINode * const packIndexPhi = b.CreatePHI(b.getSizeTy(), 2);
        packIndexPhi->addIncoming(firstPackIndex, zeroExtraneousBytes);
        for (unsigned i = 0; i < numElements; ++i) {
            Value * ptr = outputBuffer->getStreamPackPtr(b, baseAddress, b.getInt32(i), startIndex, packIndexPhi);
            b.CreateAlignedStore(vec_ZERO, ptr, align);
        }
        Value * const nextPackIndex = b.CreateAdd(packIndexPhi, sz_ONE);
        packIndexPhi->addIncoming(nextPackIndex, clearRemainingPacks);
        b.CreateCondBr(b.CreateICmpULT(nextPackIndex, sz_FieldWidth), clearRemainingPacks, clearRemainingPacksExit);

        b.SetInsertPoint(clearRemainingPacksExit);
    }

    Value * const totalIndex = b.CreateUDiv(total, sz_BlockWidth);

    b.CreateCondBr(b.CreateICmpULT(startIndex, totalIndex), zeroExtraneousBlocks, exit);

    b.SetInsertPoint(zeroExtraneousBlocks);

    Value * const nextIndex = b.CreateAdd(startIndex, sz_ONE);

    Value * const endIndex = b.CreateAdd(b.CreateUDiv(total, sz_BlockWidth), sz_ONE);
    Value * const startPtr = outputBuffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, nextIndex);
    Value * const endPtr = outputBuffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, endIndex);

    DataLayout DL(b.getModule());
    Type * const intPtrTy = DL.getIntPtrType(startPtr->getType());
    Value * const startPtrInt = b.CreatePtrToInt(startPtr, intPtrTy);
    Value * const endPtrInt = b.CreatePtrToInt(endPtr, intPtrTy);
    Value * const numBytes = b.CreateSub(endPtrInt, startPtrInt);
    b.CreateMemZero(startPtr, numBytes, 1U);

    b.CreateBr(exit);


    b.SetInsertPoint(exit);

}

class StreamEq : public MultiBlockKernel {
public:
    enum class Mode { EQ, NE };

    StreamEq(LLVMTypeSystemInterface & ts, StreamSet * x, const bool unalignedLHS, StreamSet * y, const bool unalignedRHS, Scalar * outPtr);
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
private:
    inline Bindings makeInputBindings(StreamSet * lhs, const bool unalignedLHS, StreamSet * rhs, const bool unalignedRHS);
private:
    const bool UnalignedLHS;
    const bool UnalignedRHS;
};

inline Bindings StreamEq::makeInputBindings(StreamSet * lhs, const bool unalignedLHS, StreamSet * rhs, const bool unalignedRHS) {
    Bindings bindings;
    if (unalignedLHS) {
        bindings.emplace_back("lhs", lhs, FixedRate(), AllowsUnalignedAccess());
    } else {
        bindings.emplace_back("lhs", lhs);
    }
    if (unalignedRHS) {
        bindings.emplace_back("rhs", rhs, FixedRate(), AllowsUnalignedAccess());
    } else {
        bindings.emplace_back("rhs", rhs);
    }
    return bindings;
}

StreamEq::StreamEq(LLVMTypeSystemInterface & ts,
    StreamSet * lhs,
    const bool unalignedLHS,
    StreamSet * rhs,
    const bool unalignedRHS,
    Scalar * outPtr)
    : MultiBlockKernel(ts, [&]() -> std::string {
       std::string backing;
       raw_string_ostream str(backing);
       str << "StreamEq::["
           << "<i" << lhs->getFieldWidth() << ">"
           << "[" << lhs->getNumElements() << "]";
       if (unalignedLHS) {
         str << 'U';
       }
       str << ",<i" << rhs->getFieldWidth() << ">"
           << "[" << rhs->getNumElements() << "]]";
       if (unalignedRHS) {
         str << 'U';
       }
       str.flush();
       return backing;
    }(),
    {makeInputBindings(lhs, unalignedLHS, rhs, unalignedRHS)},
    {},
    {{"result_ptr", outPtr}},
    {},
    {InternalScalar(ts.getInt1Ty(), "accum")})
, UnalignedLHS(unalignedLHS)
, UnalignedRHS(unalignedRHS)
{
    assert(lhs->getFieldWidth() == rhs->getFieldWidth());
    assert(lhs->getNumElements() == rhs->getNumElements());
    addAttribute(SideEffecting());
}

void StreamEq::generateInitializeMethod(KernelBuilder & b) {
    b.setScalarField("accum", b.getInt1(true));
}

void StreamEq::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    auto istreamset = b.getInputStreamSet("lhs");
    const uint32_t FW = istreamset->getFieldWidth();
    const uint32_t COUNT = istreamset->getNumElements();

    Type * const bbTy = b.getBitBlockType();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const loopBlock = b.CreateBasicBlock("loop");
    BasicBlock * const exitBlock = b.CreateBasicBlock("exit");

    Value * const initialAccum = b.getScalarField("accum");
    Constant * const sz_ZERO = b.getSize(0);

    Value * const hasMoreItems = b.CreateICmpNE(numOfStrides, sz_ZERO);

    b.CreateLikelyCondBr(hasMoreItems, loopBlock, exitBlock);

    b.SetInsertPoint(loopBlock);
    PHINode * const strideNo = b.CreatePHI(b.getSizeTy(), 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const accumPhi = b.CreatePHI(b.getInt1Ty(), 2);
    accumPhi->addIncoming(initialAccum, entryBlock);
    Value * nextAccum = accumPhi;

    for (uint32_t i = 0; i < COUNT; ++i) {

        Constant * const I = b.getInt32(i);

        for (unsigned j = 0; j < FW; ++j) {

            Value * lhs;
            Value * rhs;
            if (FW == 1) {
                lhs = b.getInputStreamBlockPtr("lhs", I, strideNo);
                rhs = b.getInputStreamBlockPtr("rhs", I, strideNo);
            } else {
                Constant * const J = b.getInt32(j);
                lhs = b.getInputStreamPackPtr("lhs", I, J, strideNo);
                rhs = b.getInputStreamPackPtr("rhs", I, J, strideNo);
            }
            if (UnalignedLHS) {
                lhs = b.CreateAlignedLoad(bbTy, lhs, 1);
            } else {
                lhs = b.CreateBlockAlignedLoad(bbTy, lhs);
            }
            if (UnalignedRHS) {
                rhs = b.CreateAlignedLoad(bbTy, rhs, 1);
            } else {
                rhs = b.CreateBlockAlignedLoad(bbTy, rhs);
            }

            // Perform vector comparison lhs != rhs.
            // Result will be a vector of all zeros if lhs == rhs
            Value * const vComp = b.CreateICmpNE(lhs, rhs);
            Value * const vCompAsInt = b.CreateBitCast(vComp, b.getIntNTy(cast<IDISA::FixedVectorType>(vComp->getType())->getNumElements()));
            // `comp` will be `true` iff lhs == rhs (i.e., `vComp` is a vector of all zeros)
            Value * const comp = b.CreateICmpEQ(vCompAsInt, Constant::getNullValue(vCompAsInt->getType()));
            // `and` `comp` into `accum` so that `accum` will be `true` iff lhs == rhs for all blocks in the two streams
            nextAccum = b.CreateAnd(nextAccum, comp);

        }

    }

    Value * const nextStrideNo = b.CreateAdd(strideNo, b.getSize(1));
    strideNo->addIncoming(nextStrideNo, loopBlock);
    accumPhi->addIncoming(nextAccum, loopBlock);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), loopBlock, exitBlock);

    b.SetInsertPoint(exitBlock);
    PHINode * const finalAccum = b.CreatePHI(b.getInt1Ty(), 2);
    finalAccum->addIncoming(initialAccum, entryBlock);
    finalAccum->addIncoming(nextAccum, loopBlock);
    b.setScalarField("accum", finalAccum);
}

void StreamEq::generateFinalizeMethod(KernelBuilder & b) {
    // a `result` value of `true` means the assertion passed
    Value * result = b.getScalarField("accum");

    // A `ptrVal` value of `0` means that the test is currently passing and a
    // value of `1` means the test is failing. If the test is already failing,
    // then we don't need to update the test state.
    Value * resultPtr = b.getScalarField("result_ptr");
    Value * const ptrVal = b.CreateLoad(b.getInt32Ty(), resultPtr);
    Value * resultState  = b.CreateSelect(result, b.getInt32(0), b.getInt32(1));;

    Value * const newVal = b.CreateSelect(b.CreateICmpEQ(ptrVal, b.getInt32(1)), b.getInt32(1), resultState);
    b.CreateStore(newVal, resultPtr);
}

typedef void (*TestFunctionType)(uint32_t * output);

using PatternVec = std::vector<std::vector<uint64_t>>;

Kernel * createNestedRepeatingStreamSetTest(CPUDriver & driver,
                                     const PatternVec & pattern, const bool unaligned,
                                     StreamSet * const output, Scalar * const invalid) {

    auto P = CreatePipeline(driver,
                            Input<streamset_t>{"Output", output, GreedyRate(1), Deferred()}, Input<uint32_t*>("invalid", invalid),
                            InternallySynchronized(), SideEffecting());

    RepeatingStreamSet * RepeatingStream = nullptr;
    if (unaligned) {
        RepeatingStream = P.CreateUnalignedRepeatingStreamSet(output->getFieldWidth(), pattern);
    } else {
        RepeatingStream = P.CreateRepeatingStreamSet(output->getFieldWidth(), pattern);
    }

    P.CreateKernelCall<StreamEq>(RepeatingStream, unaligned, output, false, invalid);

    P.CreateKernelCall<StreamEq>(output, false, RepeatingStream, unaligned, invalid);

    return P.makeKernel();
}


Kernel * createMultiLevelNestingTest(CPUDriver & driver,
                                     const PatternVec & pattern, const bool unaligned, const bool familyCall,
                                     StreamSet * const output, Scalar * const invalid) {


    auto P = CreatePipeline(driver,
                            Input<streamset_t>{"Output", output, GreedyRate(1), Deferred()}, Input<uint32_t*>("invalid", invalid),
                            InternallySynchronized(), SideEffecting());


    auto N = createNestedRepeatingStreamSetTest(driver, pattern, unaligned, output, invalid);

    if (familyCall) {
        P.AddKernelFamilyCall(N);
    } else {
        P.AddKernelCall(N);
    }

    return P.makeKernel();
}


bool runRepeatingStreamSetTest(CPUDriver & driver, std::default_random_engine & rng) {

    auto P = CreatePipeline(driver, Input<uint32_t*>{"output"});

    unsigned fieldWidth = optFieldWidth;
    if (fieldWidth == 0) {
        std::uniform_int_distribution<uint64_t> fwDist(0, 5);
        fieldWidth = 1ULL << fwDist(rng);
    }

    unsigned numElements = optNumElements;
    if (numElements == 0) {
        std::uniform_int_distribution<uint64_t> numElemDist(1, 8);
        numElements = numElemDist(rng);
    }

    unsigned patternLength = optPatternLength;
    if (patternLength == 0) {
        std::uniform_int_distribution<uint64_t> patLength(1, 26);
        patternLength = patLength(rng);
    }

    size_t repetitionLength = optRepetitionLength;
    if (repetitionLength == 0) {
        const auto bw = P.getBitBlockWidth();
        const auto v = boost::lcm<unsigned>(patternLength, bw) * 3U;
        repetitionLength = std::max(v, 4567U);
    }

    unsigned useNestedTest = optUseNestedPipeline;
    if (optUseNestedPipeline.getNumOccurrences() == 0) {
        std::uniform_int_distribution<unsigned> nestedPipeDist(0, 2);
        useNestedTest = nestedPipeDist(rng);
    }

    std::array<bool, 2> useFamilyCall;

    if (optUseFamilyCall.getNumOccurrences() == 0) {
        for (unsigned i = 0; i < useNestedTest; ++i) {
            std::uniform_int_distribution<unsigned> familyCallDist(0, 1);
            useFamilyCall[i] = (familyCallDist(rng) != 0);
        }
    } else {
        for (unsigned i = 0; i < useNestedTest; ++i) {
            useFamilyCall[i] = optUseFamilyCall;
        }
    }

    const bool allowUnaligned = optAllowUnaligned && numElements == 1;

    std::uniform_int_distribution<uint64_t> dist(0ULL, (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL);

    std::vector<std::vector<uint64_t>> pattern(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
        auto & vec = pattern[i];
        vec.resize(patternLength);
        for (unsigned j = 0; j < patternLength; ++j) {
            vec[j] = dist(rng);
        }
    }

    StreamSet * const Output = P.CreateStreamSet(numElements, fieldWidth);

    Scalar *  const repLength = P.CreateConstant(P.getSize(repetitionLength));

    P.CreateKernelCall<RepeatingSourceKernel>(pattern, Output, repLength);

    Scalar * invalid = P.getInputScalar("output");

    if (useNestedTest == 2) {
        auto K = createMultiLevelNestingTest(driver, pattern, allowUnaligned, useFamilyCall[1], Output, invalid);

        if (useFamilyCall[0]) {
            P.AddKernelFamilyCall(K);
        } else {
            P.AddKernelCall(K);
        }
    } else if (useNestedTest == 1) {
        auto K = createNestedRepeatingStreamSetTest(driver, pattern, allowUnaligned, Output, invalid);
        if (useFamilyCall[0]) {
            P.AddKernelFamilyCall(K);
        } else {
            P.AddKernelCall(K);
        }
    } else {
        RepeatingStreamSet * RepeatingStream = nullptr;
        if (allowUnaligned) {
            RepeatingStream = P.CreateUnalignedRepeatingStreamSet(fieldWidth, pattern);
        } else {
            RepeatingStream = P.CreateRepeatingStreamSet(fieldWidth, pattern);
        }

        P.CreateKernelCall<StreamEq>(RepeatingStream, allowUnaligned, Output, false, invalid);

        P.CreateKernelCall<StreamEq>(Output, false, RepeatingStream, allowUnaligned, invalid);
    }

    const auto f = P.compile();

    if (optVerbose) {

        if (useNestedTest) {

            llvm::errs() << "NESTED ";
            bool called = false;
            if (useFamilyCall[0]) {
                llvm::errs() << "OUTER ";
                called = true;
            }
            if (useNestedTest > 1 && useFamilyCall[1]) {
                llvm::errs() << "INNER ";
                called = true;
            }
            if (called) {
                llvm::errs() << "FAMILY CALL ";
            }
        }

        llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength << " : ";

        char joiner = '[';

        for (unsigned i = 0; i < numElements; ++i) {
            auto & vec = pattern[i];
            llvm::errs() << joiner;
            joiner = '{';
            for (unsigned j = 0; j < patternLength; ++j) {
                llvm::errs() << joiner << vec[j];
                joiner = ',';
            }
            llvm::errs() << '}';
        }

        llvm::errs() << "] -- ";
//        if (result == 0) {
//            llvm::errs() << "success";
//        } else {
//            llvm::errs() << "failed";
//        }
//        llvm::errs() << '\n';
    }

    uint32_t result = 0;
    f(&result);

    if (result != 0 && !optVerbose) {

        if (useNestedTest) {

            llvm::errs() << "NESTED ";
            bool called = false;
            if (useFamilyCall[0]) {
                llvm::errs() << "OUTER ";
                called = true;
            }
            if (useNestedTest > 1 && useFamilyCall[1]) {
                llvm::errs() << "INNER ";
                called = true;
            }
            if (called) {
                llvm::errs() << "FAMILY CALL ";
            }
        }

        llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength << " : ";

        char joiner = '[';

        for (unsigned i = 0; i < numElements; ++i) {
            auto & vec = pattern[i];
            llvm::errs() << joiner;
            joiner = '{';
            for (unsigned j = 0; j < patternLength; ++j) {
                llvm::errs() << joiner << vec[j];
                joiner = ',';
            }
            llvm::errs() << '}';
        }

        llvm::errs() << "] -- ";
    }

    if (result != 0 || optVerbose) {

        if (result == 0) {
            llvm::errs() << "success";
        } else {
            llvm::errs() << "failed";
        }
        llvm::errs() << '\n';
    }


    return (result != 0);
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver driver("test");
    std::random_device rd;
    std::default_random_engine rng(rd());

    bool testResult = false;
    const unsigned maxRounds = optNumOfTrials;
    for (unsigned rounds = 0; rounds < maxRounds; ++rounds) {
        testResult |= runRepeatingStreamSetTest(driver, rng);
    }
    return testResult ? -1 : 0;
}
