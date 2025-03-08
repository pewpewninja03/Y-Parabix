#include "../pipeline_compiler.hpp"
#include <boost/format.hpp>

namespace kernel {

#define MAX_ENTRY_GROUP_SIZE (1024 * 1024)

struct DMEntry {
    uint64_t SegNo;
    float SyncOverhead;
    uint32_t NumOfThreads;
} __attribute__((packed));


struct DMEntryGroup {
    DMEntry Entry[MAX_ENTRY_GROUP_SIZE];
    uint64_t Count;
    DMEntryGroup * Next;
};

static_assert(((sizeof(DMEntryGroup) % sizeof(void*)) == 0), "unexpected data alignment issue");

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addDynamicThreadingReportProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addDynamicThreadingReportProperties(KernelBuilder & b, const unsigned groupId) {
    assert (TraceDynamicMultithreading);

    auto & C = b.getContext();

    FixedArray<Type *, 3> DMEntryFields;
    DMEntryFields[0] = b.getInt64Ty();
    DMEntryFields[1] = TypeBuilder<float, false>::get(C);
    DMEntryFields[2] = b.getInt32Ty();
    StructType * const DMEntryTy = StructType::get(C, DMEntryFields, true);

    FixedArray<Type *, 3> DMEntryGroupFields;
    DMEntryGroupFields[0] = ArrayType::get(DMEntryTy, MAX_ENTRY_GROUP_SIZE);
    DMEntryGroupFields[1] = b.getInt64Ty();
    DMEntryGroupFields[2] = b.getVoidPtrTy();
    StructType * const DMEntryGroupTy = StructType::get(C, DMEntryGroupFields, true);

    mTarget->addInternalScalar(DMEntryGroupTy->getPointerTo(), STATISTICS_DYNAMIC_MULTITHREADING_STATE_CURRENT, groupId);

    mTarget->addInternalScalar(DMEntryGroupTy, STATISTICS_DYNAMIC_MULTITHREADING_STATE_DATA, groupId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initDynamicThreadingReportProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initDynamicThreadingReportProperties(KernelBuilder & b) {
    assert (TraceDynamicMultithreading);
    Value * data = b.getScalarFieldPtr(STATISTICS_DYNAMIC_MULTITHREADING_STATE_DATA).first;
    b.setScalarField(STATISTICS_DYNAMIC_MULTITHREADING_STATE_CURRENT, data);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordDynamicThreadingState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordDynamicThreadingState(KernelBuilder & b, Value * segNo, Value * currentSyncOverhead, Value * currentNumOfThreads) const {

    auto & C = b.getContext();

    IntegerType * i64Ty = b.getInt64Ty();

    Type * const floatTy = TypeBuilder<float, false>::get(C);

    auto & dl = b.getModule()->getDataLayout();
    const auto FloatTyABIAlignment = dl.getABITypeAlign(floatTy).value();
    const auto Int32TyABIAlignment = dl.getABITypeAlign(b.getInt32Ty()).value();

    FixedArray<Type *, 3> DMEntryFields;
    DMEntryFields[0] = i64Ty;
    DMEntryFields[1] = floatTy;
    DMEntryFields[2] = b.getInt32Ty();
    StructType * const DMEntryTy = StructType::get(C, DMEntryFields, true);

    FixedArray<Type *, 3> DMEntryGroupFields;
    DMEntryGroupFields[0] = ArrayType::get(DMEntryTy, MAX_ENTRY_GROUP_SIZE);
    DMEntryGroupFields[1] = b.getInt64Ty();
    DMEntryGroupFields[2] = b.getVoidPtrTy();
    StructType * const DMEntryGroupTy = StructType::get(C, DMEntryGroupFields, true);

    assert (TraceDynamicMultithreading);
    Value * dataPtr = b.getScalarFieldPtr(STATISTICS_DYNAMIC_MULTITHREADING_STATE_CURRENT).first;
    Value * data = b.CreateAlignedLoad(DMEntryGroupTy->getPointerTo(), dataPtr, PtrTyABIAlignment);
    Constant * const i32_ZERO = b.getInt32(0);
    Constant * const i32_ONE = b.getInt32(1);
    Constant * const i32_TWO = b.getInt32(2);
    FixedArray<Value *, 2> groupIndices;
    groupIndices[0] = i32_ZERO;
    groupIndices[1] = i32_ONE;
    Value * currentCountPtr = b.CreateGEP(DMEntryGroupTy, data, groupIndices);
    Value * const currentCount = b.CreateAlignedLoad(i64Ty, currentCountPtr, Int64TyABIAlignment);
    Value * const outOfSpace = b.CreateICmpEQ(currentCount, b.getSize(MAX_ENTRY_GROUP_SIZE));
    BasicBlock * const mallocNewChunk = b.CreateBasicBlock("mallocNewDynamicThreadingBlock");
    BasicBlock * const updateDynamicThreading = b.CreateBasicBlock("updateDynamicThreadingTrace");
    BasicBlock * const entry = b.GetInsertBlock();
    b.CreateCondBr(outOfSpace, mallocNewChunk, updateDynamicThreading);

    b.SetInsertPoint(mallocNewChunk);
    Constant * entryGroupSize = b.getSize(sizeof(DMEntryGroup));
    Value * newChunk = b.CreateAlignedMalloc(entryGroupSize, sizeof(void*));
    b.CreateMemZero(newChunk, entryGroupSize, sizeof(void*));
    groupIndices[1] = i32_TWO;
    Value * currentNextPtr = b.CreateGEP(DMEntryGroupTy, data, groupIndices);
    assert (newChunk->getType() == b.getVoidPtrTy());
    b.CreateAlignedStore(newChunk, currentNextPtr, PtrTyABIAlignment);
    newChunk = b.CreatePointerCast(newChunk, cast<PointerType>(data->getType()));
    b.CreateAlignedStore(newChunk, dataPtr, PtrTyABIAlignment);
    BasicBlock * const mallocExit = b.GetInsertBlock();
    b.CreateBr(updateDynamicThreading);

    b.SetInsertPoint(updateDynamicThreading);
    PHINode * const statePhi = b.CreatePHI(data->getType(), 2);
    statePhi->addIncoming(data, entry);
    statePhi->addIncoming(newChunk, mallocExit);
    PHINode * const count = b.CreatePHI(currentCount->getType(), 2);
    count->addIncoming(currentCount, entry);
    count->addIncoming(b.getSize(0), mallocExit);
    groupIndices[1] = i32_ONE;
    Value * const nextCount = b.CreateAdd(count, b.getSize(1));
    b.CreateAlignedStore(nextCount, b.CreateGEP(DMEntryGroupTy, statePhi, groupIndices), Int64TyABIAlignment);
    FixedArray<Value *, 4> entryIndices;
    entryIndices[0] = i32_ZERO;
    entryIndices[1] = i32_ZERO;
    entryIndices[2] = currentCount;
    entryIndices[3] = i32_ZERO;
    b.CreateAlignedStore(segNo, b.CreateGEP(DMEntryGroupTy, statePhi, entryIndices), Int64TyABIAlignment);
    entryIndices[3] = i32_ONE;
    b.CreateAlignedStore(currentSyncOverhead, b.CreateGEP(DMEntryGroupTy, statePhi, entryIndices), FloatTyABIAlignment);
    entryIndices[3] = i32_TWO;
    Value * const numThreads = b.CreateTrunc(currentNumOfThreads, b.getInt32Ty());
    b.CreateAlignedStore(numThreads, b.CreateGEP(DMEntryGroupTy, statePhi, entryIndices), Int32TyABIAlignment);

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_dynamic_multithreading_report
 ** ------------------------------------------------------------------------------------------------------------- */
extern "C" {

void __print_dynamic_multithreading_report(const DMEntryGroup * const root, const int32_t maxThreadCount, const size_t maxSegmentNumber) {

    auto ceil_log10 = [](const uint64_t v) {
        if (v < 10) {
            return 1U;
        }
        return (unsigned)std::ceil(std::log10(v));
    };

    const auto segmentLength = std::max(ceil_log10(maxSegmentNumber), 5U) + 1U;

    auto & out = errs();

    boost::format fixedfmt("%0.4f ");

    out << left_justify("SEG #", segmentLength)
        << "SYNC %  THREADS\n\n";

    for (auto c = root; c; c = c->Next) {
        const auto n = c->Count;
        const auto & E = c->Entry;
        for (size_t i = 0; i < n; ++i) {
            const auto & I = E[i];
            out << left_justify(std::to_string(I.SegNo), segmentLength)
                << (fixedfmt % I.SyncOverhead).str()
                << I.NumOfThreads
                << "\n";
        }
    }



    for (auto c = root->Next; c; ) {
        auto n = c->Next;
        free(c);
        c = n;
    }

}

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printDynamicThreadingReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printDynamicThreadingReport(KernelBuilder & b) const {
    assert (TraceDynamicMultithreading);
    Value * dataPtr = b.getScalarFieldPtr(STATISTICS_DYNAMIC_MULTITHREADING_STATE_DATA).first;

    Function * const printFn = b.getModule()->getFunction("__print_dynamic_multithreading_report");
    FixedArray<Value *, 3> args;
    args[0] = b.CreatePointerCast(dataPtr, b.getVoidPtrTy());
    args[1] = b.CreateTrunc(b.getScalarField(MAXIMUM_NUM_OF_THREADS), b.getInt32Ty());
    args[2] = getMaxSegmentNumber(b);

    b.CreateCall(printFn->getFunctionType(), printFn, args);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkDynamicThreadingReport
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ void PipelineCompiler::linkDynamicThreadingReport(KernelBuilder & b) {
    FixedArray<Type *, 3> params;
    params[0] = b.getVoidPtrTy();
    params[1] = b.getInt32Ty();
    params[2] = b.getSizeTy();
    FunctionType * funcTy = FunctionType::get(b.getVoidTy(), params, false);
    b.LinkFunction("__print_dynamic_multithreading_report", funcTy, (void*)__print_dynamic_multithreading_report);
}

}
