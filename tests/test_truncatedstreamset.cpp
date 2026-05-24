/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>

#include <llvm/IR/Instructions.h>

using namespace kernel;
using namespace llvm;

static cl::opt<unsigned> optFieldWidth("field-width", cl::desc("Field width of pattern elements"), cl::init(0));

static cl::opt<unsigned> optNumElements("num-elements", cl::desc("Number of elements in pattern streamset"), cl::init(0));

static cl::opt<unsigned> optPatternLength("pattern-length", cl::desc("Length of each pattern"), cl::init(0));

static cl::opt<unsigned> optRepetitionLength("repetition-length", cl::desc("Total length of repeating data"), cl::init(0));

static cl::opt<unsigned> optCopyLength("copy-length", cl::desc("Copy length for repeating data"), cl::init(0));

static cl::opt<unsigned> optPassLength("pass-length", cl::desc("Truncated pass-through length for repeating data"), cl::init(0));

static cl::opt<bool> optVerbose("v", cl::desc("Verbose output"), cl::init(false));

class CopyKernel final : public SegmentOrientedKernel {
public:
    CopyKernel(LLVMTypeSystemInterface & ts, StreamSet * input, StreamSet * output, Scalar * upTo);
protected:
    void generateDoSegmentMethod(KernelBuilder & b) override;
};

CopyKernel::CopyKernel(LLVMTypeSystemInterface & ts, StreamSet * input, StreamSet * output, Scalar * upTo)
: SegmentOrientedKernel(ts, [&]() {
    std::string backing;
    raw_string_ostream str(backing);
    str << "copykernel"
        << "<i" << input->getFieldWidth() << ">"
        << "[" << input->getNumElements() << "]";
    str.flush();
    return backing;
}(),
// input streams
{Binding{"input", input}},
// output stream
{Binding{"output", output}},
// input scalar
{Binding{"upTo", upTo}},
{},
// internal scalar
{}) {
    addAttribute(CanTerminateEarly());
}

void CopyKernel::generateDoSegmentMethod(KernelBuilder & b) {

    BasicBlock * const copyPartial = b.CreateBasicBlock("copyPartial");
    BasicBlock * const segmentExit = b.CreateBasicBlock("segmentExit");

    const auto is = getInputStreamSet(0);
    const auto ne = is->getNumElements();
    const auto fw = is->getFieldWidth();
    const auto bw = b.getBitBlockWidth();

    Value * const upTo = b.getScalarField("upTo");
    Value * const processed = b.getProcessedItemCount("input");
    Value * const avail = b.getAvailableItemCount("input");

    ConstantInt * const sz_ZERO = b.getSize(0);

    Value * const inputPtr = b.getInputStreamBlockPtr("input", sz_ZERO, sz_ZERO);
    Value * const outputPtr = b.getOutputStreamBlockPtr("output", sz_ZERO, sz_ZERO);

    Value * const needsPartialCopy = b.CreateICmpUGE(avail, upTo);

    Value * const remaining = b.CreateSub(upTo, processed);
    Constant * const BLOCK_WIDTH = b.getSize(bw);
    Value * const blockIndex = b.CreateUDiv(remaining, BLOCK_WIDTH);
    Value * const toCopy = b.CreateSelect(needsPartialCopy, blockIndex, b.getNumOfStrides());
    Value * const endInputPtr = b.getInputStreamBlockPtr("input", sz_ZERO, toCopy);

    DataLayout dl(b.getModule());
    Type * const intPtrTy = dl.getIntPtrType(b.getContext());
    Value * const inputPtrInt = b.CreatePtrToInt(inputPtr, intPtrTy);
    Value * const endInputPtrInt = b.CreatePtrToInt(endInputPtr, intPtrTy);
    Value * const numBytes = b.CreateSub(endInputPtrInt, inputPtrInt);
    b.CreateMemCpy(outputPtr, inputPtr, numBytes, bw / 8);
    b.CreateLikelyCondBr(b.CreateICmpUGE(avail, upTo), copyPartial, segmentExit);

    b.SetInsertPoint(copyPartial);
    Value * items = b.CreateURem(remaining, BLOCK_WIDTH);
    if (fw > 1) {
        items = b.CreateMul(items, b.getSize(fw));
    }
    for (unsigned i = 0; i < ne; ++i) {
        Value * inputPtr = nullptr;
        Value * outputPtr = nullptr;
        Constant * const sz_I = b.getSize(i);
        Value * current = items;
        for (unsigned j = 0; j < fw; ++j) {
            if (fw == 1) {
                inputPtr = b.getInputStreamBlockPtr("input", sz_I, blockIndex);
                outputPtr = b.getOutputStreamBlockPtr("output", sz_I, blockIndex);
            } else {
                Constant * const sz_J = b.getSize(j);
                inputPtr = b.getInputStreamPackPtr("input", sz_I, sz_J, blockIndex);
                outputPtr = b.getOutputStreamPackPtr("output", sz_I, sz_J, blockIndex);
            }
            Value * const maskPos = b.CreateUMin(current, BLOCK_WIDTH);
            Value * const mask = b.CreateNot(b.bitblock_mask_from(maskPos));
            Value * val = b.CreateAnd(b.CreateAlignedLoad(b.getBitBlockType(), inputPtr, bw / 8), mask);
            b.CreateStore(val, outputPtr);
            current = b.CreateUnsignedSaturatingSub(current, BLOCK_WIDTH);
        }
    }
    b.setProducedItemCount("output", upTo);
    b.setTerminationSignal();
    b.CreateBr(segmentExit);

    b.SetInsertPoint(segmentExit);
}

class PassThroughKernel final : public SegmentOrientedKernel {
public:
    PassThroughKernel(LLVMTypeSystemInterface & ts, TruncatedStreamSet * output, Scalar * upTo);
protected:
    void generateDoSegmentMethod(KernelBuilder & b) override;
};

PassThroughKernel::PassThroughKernel(LLVMTypeSystemInterface & ts, TruncatedStreamSet * output, Scalar * upTo)
: SegmentOrientedKernel(ts, "passThroughKernel",
// input streams
{},
// output stream
{Binding{"output", output}},
// input scalar
{Binding{"upTo", upTo}},
{},
// internal scalar
{}) {
    addAttribute(CanTerminateEarly());
}

void PassThroughKernel::generateDoSegmentMethod(KernelBuilder & b) {

    BasicBlock * const termKernel = b.CreateBasicBlock("termKernel");
    BasicBlock * const segmentExit = b.CreateBasicBlock("segmentExit");
    Value * const upTo = b.getScalarField("upTo");

    Value * const max = b.getWritableOutputItems("output"); assert (max);
    Value * const avail = b.CreateAdd(b.getProducedItemCount("output"), max);
    b.CreateLikelyCondBr(b.CreateICmpULT(avail, upTo), segmentExit, termKernel);

    b.SetInsertPoint(termKernel);
    b.setProducedItemCount("output", upTo);
    b.setTerminationSignal();
    b.CreateBr(segmentExit);

    b.SetInsertPoint(segmentExit);
}

class StreamEq : public MultiBlockKernel {
public:
    enum class Mode { EQ, NE };

    StreamEq(LLVMTypeSystemInterface & ts, StreamSet * x, StreamSet * y, Scalar * outPtr);
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    void generateFinalizeMethod(KernelBuilder & b) override;

};

StreamEq::StreamEq(LLVMTypeSystemInterface & ts,
    StreamSet * lhs,
    StreamSet * rhs,
    Scalar * outPtr)
    : MultiBlockKernel(ts, [&]() -> std::string {
       std::string backing;
       raw_string_ostream str(backing);
       str << "StreamEq::["
           << "<i" << lhs->getFieldWidth() << ">"
           << "[" << lhs->getNumElements() << "]";
       str << ",<i" << rhs->getFieldWidth() << ">"
           << "[" << rhs->getNumElements() << "]]";
       str.flush();
       return backing;
    }(),
    {{"lhs", lhs}, {"rhs", rhs}},
    {},
    {{"result_ptr", outPtr}},
    {},
    {InternalScalar(ts.getInt1Ty(), "accum")})
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
            lhs = b.CreateBlockAlignedLoad(b.getBitBlockType(), lhs);
            rhs = b.CreateBlockAlignedLoad(b.getBitBlockType(), rhs);

            // Perform vector comparison lhs != rhs.
            // Result will be a vector of all zeros if lhs == rhs
            Value * const vComp = b.CreateICmpNE(lhs, rhs);
            Value * const vCompAsInt = b.CreateBitCast(vComp, b.getIntNTy(cast<IDISA::FixedVectorType>(vComp->getType())->getNumElements()));
            // `comp` will be `true` iff lhs == rhs (i.e., `vComp` is a vector of all zeros)
            Value * const comp = b.CreateICmpEQ(vCompAsInt, Constant::getNullValue(vCompAsInt->getType()));
        //    b.CallPrintInt("comp", comp);
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

typedef void (*TestFunctionType)(uint64_t copyCount, uint64_t passCount, uint32_t * output);

bool runRepeatingStreamSetTest(CPUDriver & driver,
                               uint64_t numElements,
                               uint64_t fieldWidth,
                               uint64_t patternLength,
                               uint64_t copyCountVal,
                               uint64_t passCountVal,
                               std::default_random_engine & rng) {

    auto P = CreatePipeline(driver, Input<uint64_t>{"copyCount"}, Input<uint64_t>{"passCount"}, Input<uint32_t*>{"output"});

    const auto maxVal = (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL;

    std::uniform_int_distribution<uint64_t> dist(0ULL, maxVal);

    std::vector<std::vector<uint64_t>> pattern(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
        auto & vec = pattern[i];
        vec.resize(patternLength);
        for (unsigned j = 0; j < patternLength; ++j) {
            vec[j] = dist(rng);
        }
    }

    Scalar * const copyCountScalar = P.getInputScalar("copyCount");

    RepeatingStreamSet * const RepeatingStream = P.CreateRepeatingStreamSet(fieldWidth, pattern);

    StreamSet * const Output = P.CreateStreamSet(numElements, fieldWidth);

    P.CreateKernelCall<CopyKernel>(RepeatingStream, Output, copyCountScalar);

    TruncatedStreamSet * const Trunc1 = P.CreateTruncatedStreamSet(RepeatingStream);

    TruncatedStreamSet * const Trunc2 = P.CreateTruncatedStreamSet(Output);

    P.CreateKernelCall<PassThroughKernel>(Trunc1, copyCountScalar);

    Scalar * const passCountScalar = P.getInputScalar("passCount");

    P.CreateKernelCall<PassThroughKernel>(Trunc2, passCountScalar);

    Scalar * output = P.getInputScalar("output");

    P.CreateKernelCall<StreamEq>(Trunc1, Trunc2, output);

    const auto f = P.compile();

    uint32_t result = 0;

    const bool verbose = optVerbose;

    if (verbose) {
        llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength <<
                        " copyCount = " << copyCountVal <<
                        " passCount = " << passCountVal <<
                        " -- ";
    }

    f(copyCountVal, passCountVal, &result);

    if (result != 0 || verbose) {
        if (!verbose) {
            llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength <<
                            " copyCount = " << copyCountVal <<
                            " passCount = " << passCountVal <<
                            " -- ";
        }
        if (result == 0) {
            llvm::errs() << "success";
        } else {
            llvm::errs() << "failed";
        }
        llvm::errs() << '\n';
    }

    return (result != 0);
}

bool runRandomRepeatingStreamSetTest(CPUDriver & driver, std::default_random_engine & rng) {

    size_t numElements = optNumElements;
    if (numElements == 0) {
        std::uniform_int_distribution<uint64_t> numElemDist(1, 8);
        numElements = numElemDist(rng);
    }

    size_t fieldWidth = optFieldWidth;
    if (fieldWidth == 0) {
        std::uniform_int_distribution<uint64_t> fwDist(0, 5);
        fieldWidth = 1ULL << fwDist(rng);
    }

    size_t patternLength = optPatternLength;
    if (patternLength == 0) {
        std::uniform_int_distribution<uint64_t> patLength(1, 100);
        patternLength = patLength(rng);
    }

    size_t copyCountVal = optCopyLength;
    if (copyCountVal == 0) {
        std::uniform_int_distribution<uint64_t> countDist(1, 22000);
        copyCountVal = countDist(rng);
    }

    size_t passCountVal = optPassLength;
    if (passCountVal == 0) {
        std::uniform_int_distribution<uint64_t> countDist(1, 22000);
        passCountVal = countDist(rng);
    }

    return runRepeatingStreamSetTest(driver, numElements, fieldWidth, patternLength, copyCountVal, passCountVal, rng);
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver driver("test");
    std::random_device rd;
    std::default_random_engine rng(rd());

    bool testResult = false;
    for (unsigned rounds = 0; rounds < 10; ++rounds) {
        testResult |= runRandomRepeatingStreamSetTest(driver, rng);
    }
    return testResult ? -1 : 0;
}
