#include "popcount_kernel.h"

#include <kernel/core/kernel_builder.h>
#include <boost/intrusive/detail/math.hpp>

// This option is mostly for testing lookbehind of an input streamset but
// creates a cross-thread memory dependency that is otherwise unnecessary.

// #define USE_LOOKBEHIND_FOR_LAST_VALUE // must match pipeline/compiler/config.h

using namespace llvm;

namespace kernel {

const std::string INPUT = "input";

const std::string OUTPUT_STREAM = "output";
const std::string POSITIVE_STREAM = "positive";
const std::string NEGATIVE_STREAM = "negative";

const std::string CURRENT_COUNT = "currentCount";
const std::string POSITIVE_COUNT = "positiveCount";
const std::string NEGATIVE_COUNT = "negativeCount";

using Rational = ProcessingRate::Rational;

bool isNotConstantOne(Value * const value) {
    return !isa<Constant>(value) || !cast<Constant>(value)->isOneValue();
}

using boost::intrusive::detail::floor_log2;

// #define PRINT_POP_COUNTS_TO_STDERR

// #define PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY

// #define PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMultiBlockLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void PopCountKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {


    Constant * const ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    #ifdef USE_LOOKBEHIND_FOR_LAST_VALUE
    Constant * const NEG_ONE = ConstantExpr::getNeg(ONE);
    #endif
    IntegerType * const sizeTy = b.getSizeTy();

    const Binding & input = b.getInputStreamSetBinding(INPUT);
    const ProcessingRate & rate = input.getRate();
    const Rational & rv = rate.getRate();
    assert (rv.denominator() == 1);
    const auto inputWidth = rv.numerator();
    const auto blockWidth = b.getBitBlockWidth();
    const auto sizeWidth = sizeTy->getBitWidth();

    if (isNotConstantOne(b.getInputStreamSetCount(INPUT))) {
        report_fatal_error("PopCount input stream must be a single stream");
    }

    Value * position = nullptr;
    if (LLVM_LIKELY(mType != PopCountType::BOTH)) {
        position = b.getProducedItemCount(OUTPUT_STREAM);
    } else {
        position = b.getProducedItemCount(POSITIVE_STREAM);
    }

    #if defined(PRINT_POP_COUNTS_TO_STDERR) || defined(PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY)
    ConstantInt * const STDERR = b.getInt32(STDERR_FILENO);

    Function * pthreadSelfFn = b.getModule()->getFunction("pthread_self");
    if (pthreadSelfFn == nullptr) {
        IntegerType * const pThreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);
        FunctionType * funTy = FunctionType::get(pThreadTy, false);
        pthreadSelfFn = b.LinkFunction("pthread_self", funTy, (void*)&pthread_self);
    }

    Value * const _pThreadNum = b.CreateCall(pthreadSelfFn);

    auto debugPrint = [&](const StringRef label, auto &&... params) {
        std::string tmp;
        raw_string_ostream out(tmp);
        #ifdef PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM
        out << "%016" PRIx64;
        #endif
        out << "  " << label << "\n";
        #ifdef PRINT_DEBUG_MESSAGES_INCLUDE_THREAD_NUM
        b.CreateDprintfCall(STDERR, out.str(), _pThreadNum, params...);
        #else
        b.CreateDprintfCall(STDERR, out.str(), params...);
        #endif
    };

    debugPrint("initial position = %" PRIu64, position);
    #endif

    // TODO: load the initial counts from a lookbehind
    Value * positiveArray = nullptr;
    Value * initialPositiveCount = nullptr;
    Value * negativeArray = nullptr;
    Value * initialNegativeCount = nullptr;

    if (LLVM_LIKELY(mType == PopCountType::POSITIVE || mType == PopCountType::NEGATIVE)) {
        Value * const array = b.getRawOutputPointer(OUTPUT_STREAM, position);
        #ifdef USE_LOOKBEHIND_FOR_LAST_VALUE
        Value * const count = b.CreateLoad(b.CreateInBoundsGEP(sizeTy, array, NEG_ONE));
        #else
        Value * const count = b.getScalarField("count");
        #endif
        if (LLVM_LIKELY(mType == PopCountType::POSITIVE)) {
            positiveArray = array;
            initialPositiveCount = count;
            #if defined(PRINT_POP_COUNTS_TO_STDERR) || defined(PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY)
            debugPrint("initial count(pos) = %" PRIu64, count);
            #endif
        } else { // if (mType == PopCountType::NEGATIVE) {
            negativeArray = array;
            initialNegativeCount = count;
            #if defined(PRINT_POP_COUNTS_TO_STDERR) || defined(PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY)
            debugPrint("initial count(neg) = %" PRIu64, count);
            #endif
        }
    } else { // if (mType == PopCountType::BOTH) {
        positiveArray = b.getRawOutputPointer(POSITIVE_STREAM, position);
        negativeArray = b.getRawOutputPointer(NEGATIVE_STREAM, position);
        #ifdef USE_LOOKBEHIND_FOR_LAST_VALUE
        initialPositiveCount = b.CreateLoad(b.CreateInBoundsGEP(sizeTy, positiveArray, NEG_ONE));
        initialNegativeCount = b.CreateLoad(b.CreateInBoundsGEP(sizeTy, negativeArray, NEG_ONE));
        #else
        initialPositiveCount = b.getScalarField("posCount");
        initialNegativeCount = b.getScalarField("negCount");
        #endif
        #if defined(PRINT_POP_COUNTS_TO_STDERR) || defined(PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY)
        debugPrint("initial count(pos) = %" PRIu64, initialPositiveCount);
        debugPrint("initial count(neg) = %" PRIu64, initialNegativeCount);
        #endif
    }

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const popCountLoop = b.CreateBasicBlock("Loop");
    BasicBlock * const popCountExit = b.CreateBasicBlock("Exit");
    b.CreateBr(popCountLoop);

    b.SetInsertPoint(popCountLoop);
    PHINode * const index = b.CreatePHI(sizeTy, 2);
    index->addIncoming(ZERO, entry);
    PHINode * positiveSum = nullptr;
    if (positiveArray) {
        positiveSum = b.CreatePHI(sizeTy, 2);
        positiveSum->addIncoming(initialPositiveCount, entry);
    }
    PHINode * negativeSum = nullptr;
    if (negativeArray) {
        negativeSum = b.CreatePHI(sizeTy, 2);
        negativeSum->addIncoming(initialNegativeCount, entry);
    }

    Value * sum = nullptr;
    if (LLVM_UNLIKELY(inputWidth < blockWidth)) {
        report_fatal_error("popcount input width < block width is not supported yet.");
    } else {

        const auto step = inputWidth / blockWidth;
        assert (inputWidth % blockWidth == 0);
        Constant * const STEP = b.getSize(step);
        Value * const baseIndex = b.CreateMul(index, STEP);

        Value * partialSumVector = nullptr;
        // half adder will require the same number of pop count steps when n < 4
        if (step < 4) {
            for (unsigned i = 0; i < step; ++i) {
                Constant * const I = b.getSize(i);
                Value * const idx = b.CreateAdd(baseIndex, I);
                Value * value = b.loadInputStreamBlock(INPUT, ZERO, idx);
                if (LLVM_UNLIKELY(positiveSum == nullptr)) { // only negative count
                    value = b.CreateNot(value);
                }
                Value * const count = b.simd_popcount(sizeWidth, value);
                if (i == 0) {
                    partialSumVector = count;
                } else {
                    partialSumVector = b.CreateAdd(partialSumVector, count);
                }
            }
        } else {
            const auto m = floor_log2(step + 1) + 1;
            SmallVector<Value *, 64> adders(m);
            // load the first block
            Value * value = b.loadInputStreamBlock(INPUT, ZERO, baseIndex);
            if (LLVM_UNLIKELY(positiveSum == nullptr)) { // only negative count
                value = b.CreateNot(value);
            }
            adders[0] = value;

            // load and half-add the subsequent blocks
            for (unsigned i = 1; i < step; ++i) {
                Constant * const I = b.getSize(i);
                Value * const idx = b.CreateAdd(baseIndex, I);
                Value * value = b.loadInputStreamBlock(INPUT, ZERO, idx);
                if (LLVM_UNLIKELY(positiveSum == nullptr)) { // only negative count
                    value = b.CreateNot(value);
                }
                const auto k = floor_log2(i);
                for (unsigned j = 0; j <= k; ++j) {
                    Value * const sum_in = adders[j]; assert (sum_in);
                    Value * const sum_out = b.simd_xor(sum_in, value);
                    Value * const carry_out = b.simd_and(sum_in, value);
                    adders[j] = sum_out;
                    value = carry_out;
                }
                const auto l = floor_log2(i + 1);
                if (k < l) {
                    adders[l] = value;
                }
            }
            // sum the half adders
            for (unsigned i = 0; i < m; ++i) {
                Value * const count = b.simd_popcount(sizeWidth, adders[i]);
                if (i == 0) {
                    partialSumVector = count;
                } else {
                    partialSumVector = b.CreateAdd(partialSumVector, b.CreateShl(count, i));
                }
            }
        }

        const auto field = blockWidth / sizeWidth;
        assert (blockWidth % sizeWidth == 0);
        Value * const partialSum = b.hsimd_partial_sum(sizeWidth, partialSumVector);
        sum = b.mvmd_extract(sizeWidth, partialSum, field - 1);
    }

    Value * positivePartialSum = nullptr;
    if (positiveArray) {
        positivePartialSum = b.CreateAdd(positiveSum, sum);
        positiveSum->addIncoming(positivePartialSum, popCountLoop);
        Value * const ptr = b.CreateInBoundsGEP(sizeTy, positiveArray, index);
        b.CreateStore(positivePartialSum, ptr);
        #ifdef PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY
        debugPrint("> pos[%" PRIu64 "] = %" PRIu64, b.CreateAdd(position, index), positivePartialSum);
        #elif defined(PRINT_POP_COUNTS_TO_STDERR)
        debugPrint("> pos[%" PRIu64 "] = %" PRIu64 " (0x%" PRIx64 ")", b.CreateAdd(position, index), positivePartialSum, ptr);
        #endif
    }

    Value * negativePartialSum = nullptr;
    if (negativeArray) {
        Value * negSum = sum;
        if (positiveArray) {
            Constant * const INPUT_WIDTH = b.getSize(inputWidth);
            negSum = b.CreateSub(INPUT_WIDTH, sum);
        }
        negativePartialSum = b.CreateAdd(negativeSum, negSum);
        negativeSum->addIncoming(negativePartialSum, popCountLoop);
        Value * const ptr = b.CreateInBoundsGEP(sizeTy, negativeArray, index);
        b.CreateStore(negativePartialSum, ptr);
        #ifdef PRINT_POP_COUNTS_TO_STDERR_NO_ADDRESS_DISPLAY
        debugPrint("> neg[%" PRIu64 "] = %" PRIu64, b.CreateAdd(position, index), negativePartialSum);
        #elif defined(PRINT_POP_COUNTS_TO_STDERR)
        debugPrint("> neg[%" PRIu64 "] = %" PRIu64 " (0x%" PRIx64 ")", b.CreateAdd(position, index), negativePartialSum, ptr);
        #endif
    }

    BasicBlock * const popCountLoopEnd = b.GetInsertBlock();
    Value * const nextIndex = b.CreateAdd(index, ONE);
    index->addIncoming(nextIndex, popCountLoopEnd);
    Value * const done = b.CreateICmpNE(nextIndex, numOfStrides);
    b.CreateCondBr(done, popCountLoop, popCountExit);

    b.SetInsertPoint(popCountExit);
    #ifndef USE_LOOKBEHIND_FOR_LAST_VALUE
    if (LLVM_LIKELY(mType == PopCountType::POSITIVE || mType == PopCountType::NEGATIVE)) {
        Value * count = positivePartialSum;
        if (LLVM_UNLIKELY(mType == PopCountType::NEGATIVE)) {
            count = negativePartialSum;
        }
        assert (count);
        b.setScalarField("count", count);
    } else {
        b.setScalarField("posCount", positivePartialSum);
        b.setScalarField("negCount", negativePartialSum);
    }
    #endif

}

#ifdef USE_LOOKBEHIND_FOR_LAST_VALUE
#define LOOK_BEHIND_ATTR , LookBehind(1)
#else
#define LOOK_BEHIND_ATTR
#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
PopCountKernel::PopCountKernel(LLVMTypeSystemInterface & ts, const PopCountType type, const unsigned stepFactor, StreamSet * input, StreamSet * const output)
: MultiBlockKernel(ts, TypeId::PopCountKernel, "PopCount" + std::string{type == PopCountType::POSITIVE ? "P" : "N"} + std::to_string(stepFactor)
// input streams
,{Binding{INPUT, input, FixedRate(stepFactor) }}
// output stream
,{Binding{OUTPUT_STREAM, output, FixedRate() LOOK_BEHIND_ATTR }}
// unnused I/O scalars
,{} ,{},
// internal scalar
{})
, mType(type) {
    // a block of input becomes a single integer of output
    setStride(1);
    assert (type != PopCountType::BOTH);
    #ifndef USE_LOOKBEHIND_FOR_LAST_VALUE
    addInternalScalar(ts.getSizeTy(), "count");
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
PopCountKernel::PopCountKernel(LLVMTypeSystemInterface & ts, const PopCountType type, const unsigned stepFactor, StreamSet * input, StreamSet * const positive, StreamSet * const negative)
: MultiBlockKernel(ts, TypeId::PopCountKernel, ".PopCountB" + std::to_string(stepFactor)
// input streams
,{Binding{INPUT, input, FixedRate(stepFactor) }}
// output stream
,{Binding{POSITIVE_STREAM, positive, FixedRate() LOOK_BEHIND_ATTR }
 ,Binding{NEGATIVE_STREAM, negative, FixedRate() LOOK_BEHIND_ATTR }}
// unnused I/O scalars
,{} ,{},
// internal scalar
{})
, mType(type) {
    // a block of input becomes a single integer of output
    setStride(1);
    assert (type == PopCountType::BOTH);
    #ifndef USE_LOOKBEHIND_FOR_LAST_VALUE
    addInternalScalar(ts.getSizeTy(), "posCount");
    addInternalScalar(ts.getSizeTy(), "negCount");
    #endif
}

}
