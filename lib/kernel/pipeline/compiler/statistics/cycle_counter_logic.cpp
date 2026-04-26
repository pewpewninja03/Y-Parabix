#include "../pipeline_compiler.hpp"
#include <boost/format.hpp>
#include <llvm/Support/Format.h>

// TODO: Print Total CPU Cycles

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelCycleCountProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addCycleCounterProperties(KernelBuilder & b, const unsigned kernelId, const bool jumpTarget, const bool isRoot, const unsigned groupId) {

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        // TODO: make these thread local to prevent false sharing and enable
        // analysis of thread distributions?

        auto & C = b.getContext();

        Type * const int64Ty = b.getInt64Ty();
        Type * const emptyTy = StructType::get(C);
        Type * const jumpPropertyIntTy = jumpTarget ? int64Ty : emptyTy;
        Type * const otherPropertyTy = (kernelId == PipelineOutput) ? emptyTy : int64Ty;
        Type * const numInvokePropertyTy = isRoot ? otherPropertyTy : emptyTy;
        Type * copyPropertyTy = emptyTy;
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            const auto & bn = mBufferGraph[streamSet];
            if (isa<ManagedDynamicBuffer>(bn.Buffer)) {
                copyPropertyTy = otherPropertyTy;
                break;
            }
        }

        FixedArray<Type *, NUM_OF_KERNEL_CYCLE_COUNTERS> fields;
        fields[KERNEL_SYNCHRONIZATION] = otherPropertyTy;
        fields[PARTITION_JUMP_SYNCHRONIZATION] = jumpPropertyIntTy;
        fields[BUFFER_EXPANSION] = otherPropertyTy;
        fields[BUFFER_COPY] = copyPropertyTy;
        fields[KERNEL_EXECUTION] = otherPropertyTy;
        fields[TOTAL_TIME] = otherPropertyTy;
        fields[SQ_SUM_TOTAL_TIME] = otherPropertyTy;
        fields[NUM_OF_INVOCATIONS] = numInvokePropertyTy;

        StructType * const cycleCounterTy = StructType::get(C, fields);
        const auto name = makeKernelName(kernelId) + STATISTICS_CYCLE_COUNT_SUFFIX;
        mTarget->addInternalScalar(cycleCounterTy, name, groupId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool isSynchronizationCounter(const CycleCounter type) {
    return type == CycleCounter::KERNEL_SYNCHRONIZATION || type == CycleCounter::PARTITION_JUMP_SYNCHRONIZATION;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startCycleCounter(KernelBuilder & b, const CycleCounter type) {
    assert (EnableCycleCounter || isSynchronizationCounter(type));
    Value * const counter = b.CreateReadCycleCounter();
    mCycleCounters[(unsigned)type] = counter;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startCycleCounter(KernelBuilder & b, const std::initializer_list<CycleCounter> types) {
    Value * counter = b.CreateReadCycleCounter();
    assert (types.size() > 0);
    for (auto type : types) {
        assert (EnableCycleCounter || isSynchronizationCounter(type));
        mCycleCounters[(unsigned)type] = counter;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateCycleCounter(KernelBuilder & b, const unsigned kernelId, const CycleCounter type) {
    assert (FirstKernel <= kernelId && kernelId <= PipelineOutput);
    assert (EnableCycleCounter || isSynchronizationCounter(type));

    Value * const end = b.CreateReadCycleCounter();
    Value * const start = mCycleCounters[(unsigned)type]; assert (start);
    Value * const duration = b.CreateSub(end, start);

    IntegerType * sizeTy = b.getInt64Ty();
    if (mUseDynamicMultithreading && isSynchronizationCounter(type)) {
        Value * const cur = b.CreateAlignedLoad(sizeTy, mAccumulatedSynchronizationTimePtr, SizeTyABIAlignment);
        Value * const accum = b.CreateAdd(cur, duration);
        b.CreateAlignedStore(accum, mAccumulatedSynchronizationTimePtr, SizeTyABIAlignment);
    }

    if (EnableCycleCounter) {
        const auto prefix = makeKernelName(kernelId);
        Value * ptr; Type * ty;
        std::tie(ptr, ty) = b.getScalarFieldPtr(prefix  + STATISTICS_CYCLE_COUNT_SUFFIX);
        FixedArray<Value *, 2> index;
        index[0] = b.getInt32(0);
        assert (ty->getStructElementType(type)->isIntegerTy());
        index[1] = b.getInt32(type);
        Value * const sumCounterPtr = b.CreateGEP(ty, ptr, index);
        Value * const sumRunningCount = b.CreateAlignedLoad(sizeTy, sumCounterPtr, SizeTyABIAlignment);
        Value * const sumUpdatedCount = b.CreateAdd(sumRunningCount, duration);
        b.CreateAlignedStore(sumUpdatedCount, sumCounterPtr, SizeTyABIAlignment);

        if (type == CycleCounter::TOTAL_TIME) {
            index[1] = b.getInt32(SQ_SUM_TOTAL_TIME);
            assert (ty->getStructElementType(SQ_SUM_TOTAL_TIME)->isIntegerTy());
            Value * const sqSumCounterPtr = b.CreateGEP(ty, ptr, index);
            Value * const sqSumRunningCount = b.CreateAlignedLoad(sizeTy, sqSumCounterPtr, SizeTyABIAlignment);
            Value * sqDuration = b.CreateZExt(duration, sqSumRunningCount->getType());
            sqDuration = b.CreateMul(sqDuration, sqDuration);
            Value * const sqSumUpdatedCount = b.CreateAdd(sqSumRunningCount, sqDuration);
            b.CreateAlignedStore(sqSumUpdatedCount, sqSumCounterPtr, SizeTyABIAlignment);
            if (mIsPartitionRoot) {
                index[1] = b.getInt32(NUM_OF_INVOCATIONS);
                assert (ty->getStructElementType(NUM_OF_INVOCATIONS)->isIntegerTy());
                Value * const invokePtr = b.CreateGEP(ty, ptr, index);
                Value * const invoked = b.CreateAlignedLoad(sizeTy, invokePtr, SizeTyABIAlignment);
                Value * const invoked2 = b.CreateAdd(invoked, b.getSize(1));
                b.CreateAlignedStore(invoked2, invokePtr, SizeTyABIAlignment);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateCycleCounter(KernelBuilder & b, const unsigned kernelId, Value * const cond, const CycleCounter ifTrue, const CycleCounter ifFalse) {
    assert (EnableCycleCounter || mUseDynamicMultithreading);
    assert (cond && cond->getType() == b.getInt1Ty());
    Value * const end = b.CreateReadCycleCounter();
    Value * const start = mCycleCounters[(unsigned)ifTrue]; assert (start);
    assert (mCycleCounters[(unsigned)ifFalse] == start);
    Value * const duration = b.CreateSub(end, start);
    const auto prefix = makeKernelName(kernelId);
    Value * ptr; Type * ty;
    std::tie(ptr, ty) = b.getScalarFieldPtr(prefix  + STATISTICS_CYCLE_COUNT_SUFFIX);
    assert (ty->isStructTy());
    FixedArray<Value *, 2> index;
    index[0] = b.getInt32(0);
    index[1] = b.getInt32(ifTrue);
    Value * const sumCounterPtrA = b.CreateGEP(ty, ptr, index);
    index[1] = b.getInt32(ifFalse);
    Value * const sumCounterPtrB = b.CreateGEP(ty, ptr, index);
    Value * const sumCounterPtr = b.CreateSelect(cond, sumCounterPtrA, sumCounterPtrB);
    Value * const sumRunningCount = b.CreateAlignedLoad(b.getSizeTy(), sumCounterPtr, SizeTyABIAlignment);
    Value * const sumUpdatedCount = b.CreateAdd(sumRunningCount, duration);
    b.CreateAlignedStore(sumUpdatedCount, sumCounterPtr, SizeTyABIAlignment);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTotalCycleCounterTime(KernelBuilder & b) const {
    assert (EnableCycleCounter);
    Value * const end = b.CreateReadCycleCounter();
    Value * const start = mCycleCounters[(unsigned)FULL_PIPELINE_TIME];
    Value * const duration = b.CreateSub(end, start);
    // total is thread local but gets summed at the end; no need to worry about
    // multiple threads updating it.
    Value * const ptr = getScalarFieldPtr(b, STATISTICS_CYCLE_COUNT_TOTAL).first;
    Value * const updated = b.CreateAdd(b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment), duration);
    b.CreateAlignedStore(updated, ptr, SizeTyABIAlignment);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief selectPrincipleCycleCountBinding
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetPort PipelineCompiler::selectPrincipleCycleCountBinding(const unsigned kernel) const {
    const auto numOfInputs = in_degree(kernel, mBufferGraph);
    assert (degree(kernel, mBufferGraph));
    if (numOfInputs == 0) {
        return StreamSetPort{PortType::Output, 0};
    } else {
        for (unsigned i = 0; i < numOfInputs; ++i) {
            const auto inputPort = StreamSetPort{PortType::Input, i};
            const Binding & input = getInputBinding(kernel, inputPort);
            if (LLVM_UNLIKELY(input.hasAttribute(AttrId::Principal))) {
                return inputPort;
            }
        }
        return StreamSetPort{PortType::Input, 0};
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_pipeline_cycle_counter_report
 ** ------------------------------------------------------------------------------------------------------------- */
namespace {
extern "C"
BOOST_NOINLINE
void __print_pipeline_cycle_counter_report(const uint64_t numOfKernels,
                                           const char ** kernelNames,
                                           const uint64_t * const values,
                                           const uint64_t totalCycles,
                                           const uint64_t baseItemCount) {

                         // totals contains the final event counts for the program;
                         // values has numOfKernels * numOfEvents * numOfMeasurements
                         // event counts in that order.

    unsigned maxNameLength = 4;
    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto len = std::strlen(kernelNames[i]);
        maxNameLength = std::max<unsigned>(maxNameLength, len);
    }

    const auto maxKernelIdLength = ((unsigned)std::ceil(std::log10(numOfKernels))) + 1U;

    // TODO: cycle counter reporting oddly on colours=always?

    uint64_t maxItemCount = 0;
    uint64_t maxCycleCount = 0;

    long double baseCyclesPerItem = 0L;
    if (baseItemCount) {
        baseCyclesPerItem = ((long double)(totalCycles) / (long double)(baseItemCount));
    }
    auto maxCyclesPerItem = baseCyclesPerItem;

    #ifndef NDEBUG
    const auto REQ_INTEGERS = numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);
    #endif

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto k = i * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);
        assert ((k + TOTAL_TIME) < numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1));
        const uint64_t itemCount = values[k];
        maxItemCount = std::max(maxItemCount, itemCount);
        const auto cycleCount = values[k + TOTAL_TIME + 1];
        if (itemCount > 0) {
            const auto cyclesPerItem = ((long double)(cycleCount) / (long double)(itemCount));
            maxCyclesPerItem = std::max(cyclesPerItem, maxCyclesPerItem);
        }
        assert (cycleCount <= totalCycles);
        maxCycleCount = std::max(maxCycleCount, cycleCount);
    }

    maxCyclesPerItem += std::numeric_limits<long double>::epsilon();

    const auto maxItemCountLength = std::max<unsigned>(std::ceil(std::log10(maxItemCount)), 5);
    const auto maxCycleCountLength = std::max<unsigned>(std::ceil(std::log10(maxCycleCount)), 6);
    const auto maxCyclesPerItemLength = std::max<unsigned>(std::ceil(std::log10(maxCyclesPerItem)) + 2, 4);

    auto & out = errs();

    out << "CYCLE COUNTER:\n\n";

    out << right_justify("#", maxKernelIdLength); // kernel #
    out << " NAME"; // kernel Name
    assert (maxNameLength >= 4);
    assert (maxItemCountLength >= 5);
    out.indent((maxNameLength - 4) + (maxItemCountLength - 5) + 1);
    out << "ITEMS"; // items processed
    assert (maxCycleCountLength >= 6);
    out.indent((maxCycleCountLength - 6) + 1);
    out << "CYCLES"; // CPU cycles
    assert (maxCyclesPerItemLength >= 4);
    out.indent((maxCyclesPerItemLength - 4));
    out << " RATE" // cycles per item,
           "  SYNC" // kernel synchronization %,
           "  PART" // partition synchronization %,
           "  EXPD" // buffer expansion %,
           "  COPY" // look ahead + copy back + look behind %,
           "  PIPE" // pipeline overhead %,
           "  EXEC" // execution %,
           "     %\n"; // % of Total CPU Cycles.


    unsigned k = 0;

    const long double fTotal = totalCycles;

    boost::format percfmt("%5.1f");
    boost::format fixedfmt("%.0f");
    boost::format ratefmt("%.1f");
    boost::format covfmt(" +- %4.1f\n");

    std::array<uint64_t, KERNEL_EXECUTION + 3> subtotals;
    std::fill(subtotals.begin(), subtotals.end(), 0);

    for (unsigned i = 0; i < (numOfKernels - 1); ++i) {

        assert ((k + TOTAL_TIME + 1) < REQ_INTEGERS);
        const uint64_t intItemCount = values[k++];
        const uint64_t intSubTotal = values[k + TOTAL_TIME];

        out << right_justify(std::to_string(i + 1), maxKernelIdLength)
            << ' '
            << left_justify(StringRef{kernelNames[i], std::strlen(kernelNames[i])}, maxNameLength)
            << ' '
            << right_justify(std::to_string(intItemCount), maxItemCountLength)
            << ' '
            << right_justify(std::to_string(intSubTotal), maxCycleCountLength)
            << ' ';

        const long double fSubTotal = intSubTotal;

        if (intItemCount > 0) {
            const auto cyclesPerItem = (fSubTotal / (long double)(intItemCount));
            out << right_justify((ratefmt % cyclesPerItem).str(), maxCyclesPerItemLength) << ' ';
        } else {
            out << right_justify("--", maxCyclesPerItemLength + 1);
        }

        uint64_t knownOverheads = 0;

        for (unsigned j = KERNEL_SYNCHRONIZATION; j < KERNEL_EXECUTION; ++j) {
            assert (k < REQ_INTEGERS);
            const auto v = values[k++];
            subtotals[j] += v;
            knownOverheads += v;
            const double cycPerc = (((long double)v) * 100.0L) / fSubTotal;
            out << (percfmt % cycPerc).str() << ' ';
        }
        assert (k < REQ_INTEGERS);
        const auto intExecTime = values[k++];
        knownOverheads += intExecTime;

     //   assert (knownOverheads <= intSubTotal);

        size_t intOverhead = (knownOverheads > intSubTotal) ? 0UL : (intSubTotal - knownOverheads);

        const double overheadPerc = (((long double)intOverhead) * 100.0L) / fSubTotal;
        out << (percfmt % overheadPerc).str() << ' ';

        const double execTimePerc = (((long double)intExecTime) * 100.0L) / fSubTotal;
        out << (percfmt % execTimePerc).str() << ' ';
        assert (values[k] == intSubTotal);

        subtotals[KERNEL_EXECUTION] += intOverhead;
        subtotals[KERNEL_EXECUTION + 1] += intExecTime;
        subtotals[KERNEL_EXECUTION + 2] += intSubTotal;

        ++k;

        //   assert (subtotals[KERNEL_EXECUTION + 2] <= totalCycles);

       const auto perc = (fSubTotal * 100.0L) / fTotal;
       out << (percfmt % perc).str();
       assert (k < REQ_INTEGERS);
       const long double fSqrTotalCycles = values[k++];
       assert (k < REQ_INTEGERS);
       const uint64_t n = values[k++];
       const long double N = n;
       const auto x = std::max((fSqrTotalCycles * N) - (fSubTotal * fSubTotal), 0.0L);
       const auto avgStddev = (std::sqrt(x) * 100.0L) / (N * fTotal);
       out << (covfmt % avgStddev).str();

    }

    assert ((k + TOTAL_TIME + 1) < REQ_INTEGERS);
    const uint64_t intItemCount = values[k++];
    const uint64_t intSubTotal = values[k + TOTAL_TIME];

    out << right_justify(std::to_string(numOfKernels), maxKernelIdLength)
        << ' '
        << left_justify(StringRef{kernelNames[numOfKernels - 1], std::strlen(kernelNames[numOfKernels - 1])}, maxNameLength)
        << ' '
        << right_justify(std::to_string(intItemCount), maxItemCountLength)
        << ' '
        << right_justify(std::to_string(intSubTotal), maxCycleCountLength)
        << ' ';

    const long double fSubTotal = intSubTotal;

    if (intItemCount > 0) {
        const auto cyclesPerItem = (fSubTotal / (long double)(intItemCount));
        out << right_justify((ratefmt % cyclesPerItem).str(), maxCyclesPerItemLength) << ' ';
    } else {
        out << right_justify("--", maxCyclesPerItemLength + 1);
    }

    out.indent(6);

    const uint64_t intJumpCost = values[k + CycleCounter::PARTITION_JUMP_SYNCHRONIZATION];
    const double cycPerc = (((long double)intJumpCost) * 100.0L) / fSubTotal;
    out << (percfmt % cycPerc).str() << ' ';

    out << "\n";
    out.indent(maxKernelIdLength + maxNameLength + maxItemCountLength + maxCycleCountLength + maxCyclesPerItemLength - 2);
    out << "TOTAL:";

    for (unsigned j = KERNEL_SYNCHRONIZATION; j < (KERNEL_EXECUTION + 3); ++j) {
        const auto v = subtotals[j];
        const double cycPerc = (((long double)(v * 100)) / fTotal);
        out << ' ' << (percfmt % cycPerc).str();
    }
    out << '\n';

}
} // end of anonymous namespace

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalCycleCounter(KernelBuilder & b) {
#warning not right for phases yet
#if 1
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        ConstantInt * const ZERO = b.getInt32(0);

        auto toGlobal = [&](ArrayRef<Constant *> array, Type * const type, size_t size) {
            ArrayType * const arTy = ArrayType::get(type, size);
            Constant * const ar = ConstantArray::get(arTy, array);
            Module & mod = *b.getModule();
            GlobalVariable * const gv = new GlobalVariable(mod, arTy, true, GlobalValue::PrivateLinkage, ar);
            FixedArray<Value *, 2> tmp;
            tmp[0] = ZERO;
            tmp[1] = ZERO;
            return b.CreateInBoundsGEP(arTy, gv, tmp);
        };

        std::vector<Constant *> kernelNames;
        kernelNames.reserve(PipelineOutput - FirstKernel + 1);
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const Kernel * const kernel = getKernel(i);
            kernelNames.push_back(b.GetString(kernel->getName()));
        }
        kernelNames.push_back(b.GetString(mTarget->getName()));

        assert (LastKernel < PipelineOutput);

        const auto numOfKernels = PipelineOutput - FirstKernel + 1U;

        PointerType * const int8PtrTy = b.getInt8PtrTy();

        Value * const arrayOfKernelNames = toGlobal(kernelNames, int8PtrTy, numOfKernels);

        const auto REQ_INTEGERS = numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);

        Constant * const requiredSpace =  b.getSize(sizeof(uint64_t) * REQ_INTEGERS);

        IntegerType * int64Ty = b.getInt64Ty();

        PointerType * const int64PtrTy = int64Ty->getPointerTo();

        Value * const values = b.CreatePointerCast(b.CreateAlignedMalloc(requiredSpace, sizeof(uint64_t)), int64PtrTy);

        auto currentPartitionId = -1U;

        Constant * const INT64_ZERO = b.getInt64(0);

        unsigned k = 0;

        Value * currentN = nullptr;

        Value * baseItemCount = nullptr;

        FixedArray<Value *, 2> index;
        index[0] = b.getInt32(0);

        for (unsigned i = 0; i < numOfKernels - 1; ++i) {
            assert (FirstKernel + i <= LastKernel);
            const auto prefix = makeKernelName(FirstKernel + i);

            const auto partitionId = KernelPartitionId[FirstKernel + i];
            const bool isRoot = (partitionId != currentPartitionId);
            currentPartitionId = partitionId;

            const auto binding = selectPrincipleCycleCountBinding(FirstKernel + i);
            Value * const items = b.getScalarField(makeBufferName(FirstKernel + i, binding) + ITEM_COUNT_SUFFIX);
            if (in_degree(FirstKernel + i, mBufferGraph) == 0) {
                if (baseItemCount == nullptr) {
                    baseItemCount = items;
                } else {
                    Value * const eq = b.CreateICmpEQ(baseItemCount, items);
                    baseItemCount = b.CreateSelect(eq, baseItemCount, INT64_ZERO);
                }
            }

            assert (k < REQ_INTEGERS);
            Value * const itemsPtr = b.CreateGEP(b.getInt64Ty(), values, b.getInt32(k++));
            b.CreateAlignedStore(items, itemsPtr, Int64TyABIAlignment);
            Value * cycleCountPtr; Type * cycleCountTy;

            std::tie(cycleCountPtr, cycleCountTy) = b.getScalarFieldPtr(prefix + STATISTICS_CYCLE_COUNT_SUFFIX);

            assert (cycleCountTy->getStructNumElements() == NUM_OF_KERNEL_CYCLE_COUNTERS);

            for (unsigned j = 0; j < NUM_OF_INVOCATIONS; ++j) {
                Value * sumCycles = INT64_ZERO;
                if (cycleCountTy->getStructElementType(j)->isIntegerTy()) {
                    index[1] = b.getInt32(j);
                    sumCycles = b.CreateAlignedLoad(int64Ty, b.CreateGEP(cycleCountTy, cycleCountPtr, index), Int64TyABIAlignment);
                }
                assert (k < REQ_INTEGERS);
                b.CreateAlignedStore(sumCycles, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment);
            }

            if (isRoot) {
                assert (cycleCountTy->getStructElementType(NUM_OF_INVOCATIONS)->isIntegerTy());
                index[1] = b.getInt32(NUM_OF_INVOCATIONS);
                currentN = b.CreateAlignedLoad(b.getInt64Ty(), b.CreateGEP(cycleCountTy, cycleCountPtr, index), Int64TyABIAlignment);
            } else {
                assert (cycleCountTy->getStructElementType(NUM_OF_INVOCATIONS)->isEmptyTy());
            }
            assert (currentN);
            assert (k < REQ_INTEGERS);
            b.CreateAlignedStore(currentN, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment);
            assert ((k % (NUM_OF_KERNEL_CYCLE_COUNTERS + 1)) == 0);
        }

        Value * cycleCountPtr; Type * cycleCountTy;
        const auto prefix = makeKernelName(PipelineOutput);
        std::tie(cycleCountPtr, cycleCountTy) = b.getScalarFieldPtr(prefix + STATISTICS_CYCLE_COUNT_SUFFIX);

        Value * const reportedItems = (baseItemCount == nullptr) ? INT64_ZERO : baseItemCount;
        b.CreateAlignedStore(reportedItems, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment);

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // KERNEL_SYNCHRONIZATION

        assert (cycleCountTy->getStructElementType(PARTITION_JUMP_SYNCHRONIZATION)->isIntegerTy());
        index[1] = b.getInt32(PARTITION_JUMP_SYNCHRONIZATION);
        Value * sumCycles = b.CreateAlignedLoad(int64Ty, b.CreateGEP(cycleCountTy, cycleCountPtr, index), Int64TyABIAlignment);

        b.CreateAlignedStore(sumCycles, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // PARTITION_JUMP_SYNCHRONIZATION

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // BUFFER_EXPANSION

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // BUFFER_COPY

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // KERNEL_EXECUTION

        Value * const total = b.getScalarField(STATISTICS_CYCLE_COUNT_TOTAL);

        b.CreateAlignedStore(total, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // TOTAL_TIME

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // SQ_SUM_TOTAL_TIME

        b.CreateAlignedStore(INT64_ZERO, b.CreateGEP(int64Ty, values, b.getInt32(k++)), Int64TyABIAlignment); // NUM_OF_INVOCATIONS

        assert (k == REQ_INTEGERS);

        FixedArray<Value *, 5> args;
        args[0] = ConstantInt::get(int64Ty, numOfKernels);
        args[1] = arrayOfKernelNames;
        args[2] = values;
        args[3] = total;
        args[4] = reportedItems;

        Function * const reportPrinter = b.getModule()->getFunction("__print_pipeline_cycle_counter_report");
        assert (reportPrinter);
        b.CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

        b.CreateFree(values);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeStridesPerSegment(KernelBuilder & b) const {

    if (LLVM_UNLIKELY(StatisticsOptionIsSet(codegen::TraceStridesPerSegment))) {

        const auto prefix = makeKernelName(mKernelId);

        Value * traceData; Type * traceDataTy;

        std::tie(traceData, traceDataTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);
        Type * const traceLogTy =  ArrayType::get(b.getSizeTy(), 2);

        Constant * const SZ_ZERO = b.getSize(0);
        Constant * const SZ_DEFAULT_CAPACITY = b.getSize(4096 / (sizeof(size_t) * 2));
        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);
        Constant * const THREE = b.getInt32(3);
        Constant * const MAX_INT = ConstantInt::getAllOnesValue(b.getSizeTy());

        Value * const traceDataArray = b.CreatePageAlignedMalloc(traceLogTy, SZ_DEFAULT_CAPACITY);

        // fill in the struct
        b.CreateAlignedStore(MAX_INT, b.CreateGEP(traceDataTy, traceData, {ZERO, ZERO}), SizeTyABIAlignment); // "last" num of strides
        b.CreateAlignedStore(traceDataArray, b.CreateGEP(traceDataTy, traceData, {ZERO, ONE}), PtrTyABIAlignment); // trace log
        b.CreateAlignedStore(SZ_ZERO, b.CreateGEP(traceDataTy, traceData, {ZERO, TWO}), SizeTyABIAlignment); // trace length
        b.CreateAlignedStore(SZ_DEFAULT_CAPACITY, b.CreateGEP(traceDataTy, traceData, {ZERO, THREE}), SizeTyABIAlignment); // trace capacity
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordStridesPerSegment(KernelBuilder & b, const unsigned kernelId, Value * const totalStrides) const {

    if (LLVM_UNLIKELY(StatisticsOptionIsSet(codegen::TraceStridesPerSegment))) {
        // NOTE: this records only the change to attempt to reduce the memory usage of this log.
        assert (KernelPartitionId[kernelId - 1] != KernelPartitionId[kernelId]);
        const auto prefix = makeKernelName(kernelId);

        Value * toUpdate; Type * traceTy;

        std::tie(toUpdate, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);

        Module * const m = b.getModule();
        Function * updateSegmentsPerStrideTrace = m->getFunction("$trace_strides_per_segment");
        if (LLVM_UNLIKELY(updateSegmentsPerStrideTrace == nullptr)) {
            auto ip = b.saveIP();
            LLVMContext & C = b.getContext();
            Type * const sizeTy = b.getSizeTy();
            Type * const tracePtrTy = traceTy->getPointerTo();
            FunctionType * fty = FunctionType::get(b.getVoidTy(), { tracePtrTy, sizeTy, sizeTy }, false);
            updateSegmentsPerStrideTrace = Function::Create(fty, Function::PrivateLinkage, "$trace_strides_per_segment", m);

            BasicBlock * const entry = BasicBlock::Create(C, "entry", updateSegmentsPerStrideTrace);
            BasicBlock * const update = BasicBlock::Create(C, "update", updateSegmentsPerStrideTrace);
            BasicBlock * const expand = BasicBlock::Create(C, "expand", updateSegmentsPerStrideTrace);
            BasicBlock * const write = BasicBlock::Create(C, "write", updateSegmentsPerStrideTrace);
            BasicBlock * const exit = BasicBlock::Create(C, "exit", updateSegmentsPerStrideTrace);

            b.SetInsertPoint(entry);

            auto arg = updateSegmentsPerStrideTrace->arg_begin();
            arg->setName("traceData");
            Value * traceData = &*arg++;
            arg->setName("segNo");
            Value * const segNo = &*arg++;
            arg->setName("numOfStrides");
            Value * const numOfStrides = &*arg++;
            assert (arg == updateSegmentsPerStrideTrace->arg_end());

//            Type * const recordStructTy = ArrayType::get(sizeTy, 2);

//            FixedArray<Type *, 4> traceStruct;
//            traceStruct[0] = sizeTy; // last num of strides (to avoid unnecessary loads of the trace
//                                     // log and simplify the logic for first stride)
//            traceStruct[1] = recordStructTy->getPointerTo(); // pointer to trace log
//            traceStruct[2] = sizeTy; // trace length
//            traceStruct[3] = sizeTy; // trace capacity (for realloc)

            Constant * const ZERO = b.getInt32(0);
            Constant * const ONE = b.getInt32(1);
            Constant * const TWO = b.getInt32(2);
            Constant * const THREE = b.getInt32(3);

            Value * const lastNumOfStridesField = b.CreateGEP(traceTy, traceData, {ZERO, ZERO});
            Value * const lastNumOfStrides = b.CreateAlignedLoad(sizeTy, lastNumOfStridesField, SizeTyABIAlignment);
            Value * const changed = b.CreateICmpNE(lastNumOfStrides, numOfStrides);
            b.CreateCondBr(changed, update, exit);

            b.SetInsertPoint(update);
            Value * const traceLogField = b.CreateGEP(traceTy, traceData, {ZERO, ONE});
            assert (traceLogField->getType()->isPointerTy());
            Type * const recordStructTy = ArrayType::get(sizeTy, 2);
            Type * const recordStructPtrTy = recordStructTy->getPointerTo();
            Value * const traceLog = b.CreateAlignedLoad(recordStructPtrTy, traceLogField, PtrTyABIAlignment);
            Value * const traceLengthField = b.CreateGEP(traceTy, traceData, {ZERO, TWO});
            Value * const traceLength = b.CreateAlignedLoad(sizeTy, traceLengthField, SizeTyABIAlignment);
            Value * const traceCapacityField = b.CreateGEP(traceTy, traceData, {ZERO, THREE});
            assert (traceCapacityField->getType()->isPointerTy());
            Value * const traceCapacity = b.CreateAlignedLoad(sizeTy, traceCapacityField, SizeTyABIAlignment);
            Value * const hasSpace = b.CreateICmpNE(traceLength, traceCapacity);
            b.CreateLikelyCondBr(hasSpace, write, expand);

            b.SetInsertPoint(expand);
            Value * const nextTraceCapacity = b.CreateShl(traceCapacity, 1);
            assert (traceLog->getType()->isPointerTy());
            Value * const expandedtraceLog = b.CreateRealloc(recordStructTy, traceLog, nextTraceCapacity);
            assert (expandedtraceLog->getType() == traceLog->getType());
            b.CreateAlignedStore(expandedtraceLog, traceLogField, PtrTyABIAlignment);
            b.CreateAlignedStore(nextTraceCapacity, traceCapacityField, SizeTyABIAlignment);
            b.CreateBr(write);

            b.SetInsertPoint(write);
            PHINode * const traceLogPhi = b.CreatePHI(recordStructPtrTy, 2);
            traceLogPhi->addIncoming(traceLog, update);
            traceLogPhi->addIncoming(expandedtraceLog, expand);
            b.CreateAlignedStore(segNo, b.CreateGEP(recordStructTy, traceLogPhi, {traceLength , ZERO}), SizeTyABIAlignment);
            b.CreateAlignedStore(numOfStrides, b.CreateGEP(recordStructTy, traceLogPhi, {traceLength , ONE}), SizeTyABIAlignment);
            b.CreateAlignedStore(numOfStrides, lastNumOfStridesField, SizeTyABIAlignment);
            Value * const newTraceLength = b.CreateAdd(traceLength, b.getSize(1));
            b.CreateStore(newTraceLength, traceLengthField);

            b.CreateBr(exit);

            b.SetInsertPoint(exit);
            b.CreateRetVoid();

            b.restoreIP(ip);
        }


        FixedArray<Value *, 3> args;
        args[0] = toUpdate;
        args[1] = mSegNo;
        args[2] = totalStrides;
        b.CreateCall(updateSegmentsPerStrideTrace, args);

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concludeStridesPerSegmentRecording
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::concludeStridesPerSegmentRecording(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(StatisticsOptionIsSet(codegen::TraceStridesPerSegment))) {
        auto currentPartitionId = KernelPartitionId[PipelineInput];
        Constant * const sz_ZERO = b.getSize(0);
        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {
            const auto partitionId = KernelPartitionId[kernelId];
            if (partitionId != currentPartitionId) {
                currentPartitionId = partitionId;
                recordStridesPerSegment(b, kernelId, sz_ZERO);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalStridesPerSegment(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(StatisticsOptionIsSet(codegen::TraceStridesPerSegment))) {

        IntegerType * const sizeTy = b.getSizeTy();

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        ConstantInt * const SZ_ZERO = b.getSize(0);
        ConstantInt * const SZ_ONE = b.getSize(1);

        // Print the title line
        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        Value * const maxSegNo = getMaxSegmentNumber(b);

        SmallVector<char, 1024> buffer;
        raw_svector_ostream format(buffer);
        format << "STRIDES PER SEGMENT:\n\n"
                  "SEG #";

        SmallVector<unsigned, 64> partitionRootIds;
        partitionRootIds.reserve(PartitionCount);
        for (unsigned kernel = FirstKernel, currentPartitionId = 0; kernel <= LastKernel; ++kernel) {
            const auto partitionId = KernelPartitionId[kernel];
            if (currentPartitionId != partitionId) {
                currentPartitionId = partitionId;
                partitionRootIds.push_back(kernel);
            }
        }
        assert (partitionRootIds.size() == (PartitionCount - 2));

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const Kernel * const kernel = getKernel(partitionRootIds[i]);
            std::string temp = kernel->getName();
            boost::replace_all(temp, "\"", "\\\"");
            format << ",\"" << partitionRootIds[i] << "." << temp << "\"";
        }
        format << "\n";


        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Generate line format string
        buffer.clear();

        format << "%" PRIu64; // seg #
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            format << ",%" PRIu64; // strides
        }
        format << "\n";

        // Print each kernel line
        SmallVector<Value *, 64> args((PartitionCount - 2) + 3);
        args[0] = STDERR;
        args[1] = b.GetString(format.str());

        // Pre load all of pointers to the the trace log out of the pipeline state.

        SmallVector<Value *, 64> traceLogArray(PartitionCount - 1);
        SmallVector<Value *, 64> traceLengthArray(PartitionCount - 1);

        Type * const recordStructTy = ArrayType::get(sizeTy, 2);
        Type * const recordStructPtrTy = recordStructTy->getPointerTo();

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const auto prefix = makeKernelName(partitionRootIds[i]);
            Value * traceData; Type * traceTy;
            std::tie(traceData, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);
            traceLogArray[i] = b.CreateAlignedLoad(recordStructPtrTy, b.CreateGEP(traceTy, traceData, {ZERO, ONE}), PtrTyABIAlignment);
            traceLengthArray[i] = b.CreateAlignedLoad(sizeTy, b.CreateGEP(traceTy, traceData, {ZERO, TWO}), SizeTyABIAlignment);
        }

        // Start printing the data lines

        BasicBlock * const loopEntry = b.GetInsertBlock();
        BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
        b.CreateBr(loopStart);

        b.SetInsertPoint(loopStart);
        PHINode * const segNo = b.CreatePHI(sizeTy, 2);
        segNo->addIncoming(SZ_ZERO, loopEntry);
        SmallVector<PHINode *, 64> currentIndex(PartitionCount - 1);
        SmallVector<PHINode *, 64> updatedIndex(PartitionCount - 1);
        SmallVector<PHINode *, 64> currentValue(PartitionCount - 1);
        SmallVector<PHINode *, 64> updatedValue(PartitionCount - 1);

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            currentIndex[i] = b.CreatePHI(sizeTy, 2);
            currentIndex[i]->addIncoming(SZ_ZERO, loopEntry);
            currentValue[i] = b.CreatePHI(sizeTy, 2);
            currentValue[i]->addIncoming(SZ_ZERO, loopEntry);
        }

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const auto prefix = makeKernelName(partitionRootIds[i]);

            BasicBlock * const entry = b.GetInsertBlock();
            BasicBlock * const check = b.CreateBasicBlock();
            BasicBlock * const update = b.CreateBasicBlock();
            BasicBlock * const next = b.CreateBasicBlock();
            Value * const notEndOfTrace = b.CreateICmpNE(currentIndex[i], traceLengthArray[i]);
            b.CreateLikelyCondBr(notEndOfTrace, check, next);

            b.SetInsertPoint(check);
            Value * const nextSegNo = b.CreateAlignedLoad(sizeTy, b.CreateGEP(recordStructTy, traceLogArray[i], { currentIndex[i], ZERO }), SizeTyABIAlignment);
            b.CreateCondBr(b.CreateICmpEQ(segNo, nextSegNo), update, next);

            b.SetInsertPoint(update);
            Value * const numOfStrides = b.CreateAlignedLoad(sizeTy, b.CreateGEP(recordStructTy, traceLogArray[i], { currentIndex[i], ONE }), SizeTyABIAlignment);
            Value * const currentIndex1 = b.CreateAdd(currentIndex[i], SZ_ONE);
            b.CreateBr(next);

            b.SetInsertPoint(next);
            PHINode * const nextValue = b.CreatePHI(sizeTy, 3);
            nextValue->addIncoming(SZ_ZERO, entry);
            nextValue->addIncoming(currentValue[i], check);
            nextValue->addIncoming(numOfStrides, update);
            updatedValue[i] = nextValue;

            PHINode * const nextIndex = b.CreatePHI(sizeTy, 3);
            nextIndex->addIncoming(currentIndex[i], entry);
            nextIndex->addIncoming(currentIndex[i], check);
            nextIndex->addIncoming(currentIndex1, update);
            updatedIndex[i] = nextIndex;
        }

        args[2] = segNo;
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            args[i + 3] = updatedValue[i];
        }

        b.CreateCall(fTy, Dprintf, args);

        BasicBlock * const loopExit = b.CreateBasicBlock("reportStridesPerSegmentExit");
        BasicBlock * const loopEnd = b.GetInsertBlock();
        segNo->addIncoming(b.CreateAdd(segNo, SZ_ONE), loopEnd);

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            currentIndex[i]->addIncoming(updatedIndex[i], loopEnd);
            currentValue[i]->addIncoming(updatedValue[i], loopEnd);
        }

        Value * const notDone = b.CreateICmpULT(segNo, maxSegNo);

        b.CreateLikelyCondBr(notDone, loopStart, loopExit);

        b.SetInsertPoint(loopExit);
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            b.CreateFree(traceLogArray[i]);
        }
    }
}

#define ITEM_COUNT_DELTA_CHUNK_LENGTH (64 * 1024)

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addProducedItemCountDeltaProperties(KernelBuilder & b, unsigned kernel) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        addItemCountDeltaProperties(b, kernel, STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordProducedItemCountDeltas(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        const auto n = out_degree(mKernelId, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        Vec<Value *> current(n);
        Vec<Value *> prior(n);
        for (unsigned i = 0; i != n; ++i) {
            const StreamSetPort port(PortType::Output, i);
            const auto streamSet = getOutputBufferVertex(port);
            current[i] = mFullyProducedItemCount[port]; assert (current[i]);
            prior[i] = mInitiallyProducedItemCount[streamSet]; assert (prior[i]);
        }
        recordItemCountDeltas(b, current, prior, STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printProducedItemCountDeltas(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        printItemCountDeltas(b, "PRODUCED ITEM COUNT DELTAS PER SEGMENT:", STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addUnconsumedItemCountProperties(KernelBuilder & b, unsigned kernel) const {
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        addItemCountDeltaProperties(b, kernel, STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordUnconsumedItemCounts(KernelBuilder & b) {
#warning fix me
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        const auto n = out_degree(mKernelId, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        Vec<Value *> current(n);
        Vec<Value *> prior(n);
        for (unsigned i = 0; i != n; ++i) {
            const StreamSetPort port(PortType::Output, i);
            const auto streamSet = getOutputBufferVertex(port);
            current[i] = mInitiallyProducedItemCount[streamSet];
            prior[i] = readConsumedItemCount(b, streamSet);
        }
        recordItemCountDeltas(b, current, prior, STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printUnconsumedItemCounts(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        printItemCountDeltas(b, "UNCONSUMED ITEM COUNTS PER SEGMENT:", STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordItemCountDeltas(KernelBuilder & b,
                                             const Vec<Value *> & current,
                                             const Vec<Value *> & prior,
                                             const StringRef suffix) const {
    const auto kernelName = makeKernelName(mKernelId);
    const auto fieldName = (kernelName + suffix).str();

    Value * trace;

    std::tie(trace, std::ignore) = b.getScalarFieldPtr(fieldName);

    Module * const m = b.getModule();

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const sizeTy = b.getSizeTy();

    constexpr auto logName = "__recordItemCountDelta";

    Function * logFunc = m->getFunction(logName);
    if (logFunc == nullptr) {

        PointerType * const sizePtrTy = sizeTy->getPointerTo();

        FixedArray<Type *, 4> params;
        params[0] = voidPtrTy;
        params[1] = sizeTy;
        params[2] = sizeTy;
        params[3] = sizePtrTy;

        FunctionType * fTy = FunctionType::get(b.getVoidTy(), params, false);
        logFunc = Function::Create(fTy, Function::InternalLinkage, logName, m);

        const auto ip = b.saveIP();

        auto & C = b.getContext();

        BasicBlock * const entry = BasicBlock::Create(C, "entry", logFunc);

        b.SetInsertPoint(entry);

        auto arg = logFunc->arg_begin();
        auto nextArg = [&]() {
            assert (arg != logFunc->arg_end());
            Value * const v = &*arg; assert (v);
            std::advance(arg, 1);
            return v;
        };

        FixedArray<Type *, 3> fields;
        fields[0] = sizeTy;
        fields[1] = voidPtrTy;
        fields[2] = ArrayType::get(sizeTy, ITEM_COUNT_DELTA_CHUNK_LENGTH); // Int array of size N * ITEM_COUNT_DELTA_CHUNK_LENGTH

        StructType * const logChunkTy = StructType::get(C, fields);
        PointerType * const logChunkPtrTy = logChunkTy->getPointerTo();
        PointerType * const logChunkPtrPtrTy = logChunkPtrTy->getPointerTo();

        Value * const logChunkPtrPtr = b.CreatePointerCast(nextArg(), logChunkPtrPtrTy);
        Value * const segNo = nextArg();
        Value * const N = nextArg();
        Value * const currentArray = nextArg();
        assert (arg == logFunc->arg_end());

        ConstantInt * const CHUNK_LENGTH =  ConstantInt::get(sizeTy, ITEM_COUNT_DELTA_CHUNK_LENGTH);

        DataLayout DL(b.getModule());
        const auto sizeTySize = b.getTypeSize(DL, sizeTy);
        const auto voidPtrTySize = b.getTypeSize(DL, voidPtrTy);
        Value * const currentLog = b.CreateAlignedLoad(logChunkPtrTy, logChunkPtrPtr, voidPtrTySize);
        BasicBlock * const checkLogOffset = b.CreateBasicBlock("checkLogOffset");
        BasicBlock * const allocateNewLogChunk = b.CreateBasicBlock("allocateNewLogChunk");
        BasicBlock * const writeLogEntry = b.CreateBasicBlock("writeLogEntry");
        BasicBlock * const exit = b.CreateBasicBlock("exit");

        // TODO: set this to alloc at startup

        Value * const segIndex = b.CreateURem(segNo, CHUNK_LENGTH);
        Value * const segOffset = b.CreateMul(segIndex, N);
        b.CreateCondBr(b.CreateIsNull(currentLog), allocateNewLogChunk, checkLogOffset);

        b.SetInsertPoint(checkLogOffset);

        ConstantInt * const i32_ZERO = b.getInt32(0);
        ConstantInt * const i32_ONE = b.getInt32(1);

        FixedArray<Value *, 2> offset;
        offset[0] = i32_ZERO;
        offset[1] = i32_ZERO;

        Value * const chunkStart = b.CreateAlignedLoad(sizeTy, b.CreateGEP(logChunkTy, currentLog, offset), SizeTyABIAlignment);
        Value * const canFit = b.CreateICmpULT(segNo, b.CreateAdd(chunkStart, CHUNK_LENGTH));
        b.CreateCondBr(canFit, writeLogEntry, allocateNewLogChunk);

        b.SetInsertPoint(allocateNewLogChunk);
        Value * const m = b.CreateAdd(b.CreateMul(N, CHUNK_LENGTH), b.getSize(1));
        Value * const sizeLength = b.CreateAdd(b.CreateMul(m, b.getSize(sizeTySize)), b.getSize(voidPtrTySize));
        Value * const newLog = b.CreatePointerCast(b.CreatePageAlignedMalloc(sizeLength), logChunkPtrTy);
        b.CreateAlignedStore(b.CreateSub(segNo, segIndex), b.CreateGEP(logChunkTy, newLog, offset), SizeTyABIAlignment);
        offset[1] = i32_ONE;
        b.CreateAlignedStore(currentLog, b.CreateGEP(logChunkTy, newLog, offset), PtrTyABIAlignment);
        b.CreateAlignedStore(newLog, logChunkPtrPtr, PtrTyABIAlignment);
        b.CreateBr(writeLogEntry);

        ConstantInt * const sz_ZERO = b.getSize(0);
        ConstantInt * const sz_ONE = b.getSize(1);

        b.SetInsertPoint(writeLogEntry);
        PHINode * const indexPhi = b.CreatePHI(sizeTy, 2);
        indexPhi->addIncoming(sz_ZERO, checkLogOffset);
        indexPhi->addIncoming(sz_ZERO, allocateNewLogChunk);
        PHINode * const logPhi = b.CreatePHI(logChunkPtrTy, 2);
        logPhi->addIncoming(currentLog, checkLogOffset);
        logPhi->addIncoming(newLog, allocateNewLogChunk);

        Value * const inputPtr = b.CreateGEP(sizeTy->getPointerTo(), currentArray, indexPhi);
        Value * const val = b.CreateAlignedLoad(sizeTy, inputPtr, SizeTyABIAlignment);

        FixedArray<Value *, 3> offset3;
        offset3[0] = i32_ZERO;
        offset3[1] = b.getInt32(2);

        offset3[2] = b.CreateAdd(segOffset, indexPhi);
        Value * const arrayPtr = b.CreateGEP(logChunkTy, logPhi, offset3);
        assert (arrayPtr->getType() == sizePtrTy);

        b.CreateAlignedStore(val, arrayPtr, SizeTyABIAlignment);
        Value * const nextIndex = b.CreateAdd(indexPhi, sz_ONE);
        indexPhi->addIncoming(nextIndex, writeLogEntry);
        logPhi->addIncoming(logPhi, writeLogEntry);
        b.CreateCondBr(b.CreateICmpULT(nextIndex, N), writeLogEntry, exit);

        b.SetInsertPoint(exit);
        b.CreateRetVoid();

        b.restoreIP(ip);
    }

    FixedArray<Value *, 4> args;
    args[0] = b.CreatePointerCast(trace, voidPtrTy);
    const auto n = out_degree(mKernelId, mBufferGraph);
    args[1] = mSegNo;
    ConstantInt * N = b.getSize(n);
    args[2] = N;
    // TODO: just make a single alloca at the top for the max output ports?
    Value * const array = b.CreateAllocaAtEntryPoint(sizeTy, N);
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & out = mBufferGraph[e];
        const auto i = out.Port.Number;
        Value * const delta = b.CreateSub(current[i], prior[i]);
        b.CreateAlignedStore(delta, b.CreateGEP(sizeTy, array, b.getInt32(i)), SizeTyABIAlignment);
    }
    args[3] = array;
    b.CreateCall(logFunc, args);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addItemCountDeltaProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addItemCountDeltaProperties(KernelBuilder & b, const unsigned kernel, const StringRef suffix) const {
    const auto n = out_degree(kernel, mBufferGraph);
    LLVMContext & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    ArrayType * const logTy = ArrayType::get(ArrayType::get(sizeTy, n), ITEM_COUNT_DELTA_CHUNK_LENGTH);
    StructType * const logChunkTy = StructType::get(C, { sizeTy, voidPtrTy, logTy } );
    PointerType * const traceTy = logChunkTy->getPointerTo();
    const auto fieldName = (makeKernelName(kernel) + suffix).str();
    const auto groupId = getCacheLineGroupId(kernel);
    mTarget->addInternalScalar(traceTy, fieldName, groupId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printItemCountDeltas
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printItemCountDeltas(KernelBuilder & b, const StringRef title, const StringRef suffix) const {

    IntegerType * const sizeTy = b.getSizeTy();

    Constant * const i32_ZERO = b.getInt32(0);
    Constant * const i32_ONE = b.getInt32(1);

    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);


    // Print the title line
    Function * Dprintf = b.GetDprintf();
    FunctionType * fTy = Dprintf->getFunctionType();

    Value * const maxSegNo = getMaxSegmentNumber(b);

    SmallVector<char, 100> buffer;
    raw_svector_ostream format(buffer);
    format << title << "\n\n"
              "SEG #";

    size_t totalStreamSets = 0;

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto n = out_degree(i, mBufferGraph);

        if (n == 0) {
            continue;
        }

        const Kernel * const kernel = getKernel(i);
        std::string kernelName = kernel->getName();
        boost::replace_all(kernelName, "\"", "\\\"");

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & out = br.Binding;
            format << "," << target(e, mBufferGraph)
                   << " " << kernelName
                   << '.' << out.getName();
        }

        totalStreamSets += n;
    }

    format << "\n";


    Constant * const STDERR = b.getInt32(STDERR_FILENO);

    FixedArray<Value *, 2> titleArgs;
    titleArgs[0] = STDERR;
    titleArgs[1] = b.GetString(format.str());
    b.CreateCall(fTy, Dprintf, titleArgs);

    // Generate line format string
    buffer.clear();

    format << "%" PRIu64; // seg #
    for (auto c = totalStreamSets; c; --c) {
        format << ",%" PRIu64; // processed item count delta
    }
    format << "\n";

    // Print each kernel line
    SmallVector<Value *, 64> args(totalStreamSets + 3);
    args[0] = STDERR;
    args[1] = b.GetString(format.str());

    SmallVector<Value *, 64> traceLogArray(LastKernel + 1);
    SmallVector<StructType *, 64> traceLogType(LastKernel + 1);

    LLVMContext & C = b.getContext();
    PointerType * const voidPtrTy = b.getVoidPtrTy();

    size_t maxOutputPorts = 0;

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const auto n = out_degree(i, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            traceLogArray[i] = nullptr;
            traceLogType[i] = nullptr;
        } else {
            const auto fieldName = (makeKernelName(i) + suffix).str();
            traceLogArray[i] = b.getScalarField(fieldName);
            ArrayType * const logTy = ArrayType::get(ArrayType::get(sizeTy, n), ITEM_COUNT_DELTA_CHUNK_LENGTH);
            traceLogType[i] = StructType::get(C, { sizeTy, voidPtrTy, logTy } );
            maxOutputPorts = std::max(maxOutputPorts, n);
        }
    }

    Constant * const CHUNK_LENGTH = b.getSize(ITEM_COUNT_DELTA_CHUNK_LENGTH);



    // Start printing the data lines
    BasicBlock * const loopEntry = b.GetInsertBlock();
    BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
    BasicBlock * const printLogEntryLoop = b.CreateBasicBlock("printLogEntryLoop");

    BasicBlock * const printLogEntryLoopExit = b.CreateBasicBlock("printLogEntryLoopExit");
    SmallVector<Value *, 64> currentChunk(LastKernel + 1);
    b.CreateBr(loopStart);

    b.SetInsertPoint(loopStart);
    PHINode * const baseSegNoPhi = b.CreatePHI(sizeTy, 2);
    baseSegNoPhi->addIncoming(sz_ZERO, loopEntry);

    FixedArray<Value *, 2> offset;
    offset[0] = i32_ZERO;

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        if (out_degree(i, mBufferGraph) == 0) {
            continue;
        }

        BasicBlock * const entry = b.GetInsertBlock();
        BasicBlock * const loop = b.CreateBasicBlock("getNextLogChunk", printLogEntryLoop);
        BasicBlock * const exit = b.CreateBasicBlock("getNextLogChunkExit", printLogEntryLoop);
        PointerType * logChunkPtrTy = traceLogType[i]->getPointerTo();
        Constant * nullPtr = ConstantPointerNull::get(logChunkPtrTy);
        b.CreateBr(loop);

        b.SetInsertPoint(loop);
        PHINode * current = b.CreatePHI(logChunkPtrTy, 2);
        current->addIncoming(traceLogArray[i], entry);
        offset[1] = i32_ZERO;
        Value * const chunkStart = b.CreateAlignedLoad(sizeTy, b.CreateGEP(traceLogType[i], current, offset), SizeTyABIAlignment);
        Value * const A = b.CreateICmpULE(chunkStart, baseSegNoPhi);
        Value * const B = b.CreateICmpULT(baseSegNoPhi, b.CreateAdd(chunkStart, CHUNK_LENGTH));
        Value * const found = b.CreateAnd(A, B);
        offset[1] = i32_ONE;
        Value * const nextChunkPtr = b.CreateAlignedLoad(logChunkPtrTy, b.CreateGEP(traceLogType[i], current, offset), PtrTyABIAlignment);
        Value * const noMore = b.CreateICmpEQ(nextChunkPtr, nullPtr);
        current->addIncoming(nextChunkPtr, loop);
        currentChunk[i] = b.CreateSelect(noMore, nullPtr, current);
        b.CreateCondBr(b.CreateOr(found, noMore), exit, loop);

        b.SetInsertPoint(exit);
    }

    BasicBlock * const printLogLoopEntry = b.GetInsertBlock();

    b.CreateBr(printLogEntryLoop);

    b.SetInsertPoint(printLogEntryLoop);
    PHINode * const indexPhi = b.CreatePHI(sizeTy, 2);
    indexPhi->addIncoming(sz_ZERO, printLogLoopEntry);

    size_t argIndex = 3;

    SmallVector<Value *, 16> dataVal(maxOutputPorts);

    offset[1] = b.getInt32(2);

    FixedArray<Value *, 3> dataValOffset;

    dataValOffset[0] = b.getSize(0);
    dataValOffset[1] = indexPhi;

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto m = out_degree(i, mBufferGraph);
        if (m == 0) {
            continue;
        }

        BasicBlock * const printLogEntryLoopHasData = b.CreateBasicBlock("printLogEntryLoopHasData", printLogEntryLoopExit);
        BasicBlock * const printLogEntryLoopNextKernel = b.CreateBasicBlock("printLogEntryLoopNextKernel", printLogEntryLoopExit);

        b.CreateLikelyCondBr(b.CreateIsNotNull(currentChunk[i]), printLogEntryLoopHasData, printLogEntryLoopNextKernel);

        BasicBlock * const entry = b.GetInsertBlock();

        b.SetInsertPoint(printLogEntryLoopHasData);
        for (size_t j = 0; j < m; ++j) {
            dataValOffset[2] = b.getSize(j);

            Value * const ptr = b.CreateGEP(traceLogType[i], currentChunk[i], offset);

            Value * const ptr2 = b.CreateGEP(traceLogType[i]->getStructElementType(2), ptr, dataValOffset);

            dataVal[j] = b.CreateAlignedLoad(sizeTy, ptr2, SizeTyABIAlignment);
        }
        b.CreateBr(printLogEntryLoopNextKernel);

        b.SetInsertPoint(printLogEntryLoopNextKernel);
        for (size_t j = 0; j < m; ++j) {
            PHINode * const argPhi = b.CreatePHI(sizeTy, 2);
            argPhi->addIncoming(sz_ZERO, entry);
            argPhi->addIncoming(dataVal[j], printLogEntryLoopHasData);
            assert (args[argIndex + j] == nullptr);
            args[argIndex + j] = argPhi;
        }
        argIndex += m;
    }



    Value * const actualSegNo = b.CreateAdd(baseSegNoPhi, indexPhi);

    args[2] = actualSegNo;

    b.CreateCall(fTy, Dprintf, args);

    Value * notEndOfChunk = b.CreateICmpNE(indexPhi, CHUNK_LENGTH);
    Value * notEndOfData = b.CreateICmpULE(actualSegNo, maxSegNo);

    indexPhi->addIncoming(b.CreateAdd(indexPhi, sz_ONE), b.GetInsertBlock());

    b.CreateCondBr( b.CreateAnd(notEndOfChunk, notEndOfData), printLogEntryLoop, printLogEntryLoopExit);

    b.SetInsertPoint(printLogEntryLoopExit);

    Value * const nextBaseSegNo = b.CreateAdd(baseSegNoPhi, CHUNK_LENGTH);
    baseSegNoPhi->addIncoming(nextBaseSegNo, printLogEntryLoopExit);
    Value * const notDone = b.CreateICmpULT(nextBaseSegNo, maxSegNo);

    BasicBlock * const loopExit = b.CreateBasicBlock("loopExit");
    b.CreateLikelyCondBr(notDone, loopStart, loopExit);

    b.SetInsertPoint(loopExit);
    // print final new line
    FixedArray<Value *, 2> finalArgs;
    finalArgs[0] = STDERR;
    finalArgs[1] = b.GetString("\n");
    b.CreateCall(fTy, Dprintf, finalArgs);

    offset[1] = i32_ONE;

    // and free any memory
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(traceLogArray[i] == nullptr)) continue;
        BasicBlock * const freeLoop = b.CreateBasicBlock("freeLoop");
        BasicBlock * const freeExit = b.CreateBasicBlock("freeExit");

        PointerType * const traceLogPtrTy = cast<PointerType>(traceLogType[i]->getPointerTo());
        Constant * nil = ConstantPointerNull::get(traceLogPtrTy);

        BasicBlock * const entry = b.GetInsertBlock();

        b.CreateLikelyCondBr(b.CreateICmpNE(traceLogArray[i], nil), freeLoop, freeExit);

        b.SetInsertPoint(freeLoop);
        PHINode * logArray = b.CreatePHI(traceLogPtrTy, 2);
        logArray->addIncoming(traceLogArray[i], entry);
        Value * const nextArray = b.CreateAlignedLoad(traceLogPtrTy, b.CreateGEP(traceLogType[i], logArray, offset), PtrTyABIAlignment);
        b.CreateFree(logArray);
        logArray->addIncoming(nextArray, freeLoop);
        b.CreateLikelyCondBr(b.CreateICmpNE(nextArray, nil), freeLoop, freeExit);

        b.SetInsertPoint(freeExit);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getMaxSegmentNumber
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getMaxSegmentNumber(KernelBuilder & b) const {
    auto getKernelSegNo = [&](const size_t kernelId) {
        const auto type = isDataParallel(kernelId) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, kernelId, type);
        return b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment);
    };
    return getKernelSegNo(FirstKernel);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkInstrumentationFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkInstrumentationFunctions(KernelBuilder & b) {
    b.LinkFunction("__print_pipeline_cycle_counter_report", __print_pipeline_cycle_counter_report);
}

}
