#include "../pipeline_compiler.hpp"
#include <llvm/Support/Format.h>

namespace kernel {

struct HistogramPortListEntry {
    uint64_t ItemCount;
    uint64_t Frequency;
    HistogramPortListEntry * Next;
};

struct HistogramPortData {
    uint32_t PortType;
    uint32_t PortNum;
    const char * BindingName;
    uint64_t Size;
    void * Data; // if Size = 0, this points to a HistogramPortListEntry; otherwise its an 64-bit array of length size.
};

struct HistogramKernelData {
    uint32_t Id;
    uint32_t NumOfPorts;
    const char * KernelName;
    HistogramPortData * PortData;
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_pipeline_histogram_report
 ** ------------------------------------------------------------------------------------------------------------- */

extern "C" {

void __print_pipeline_histogram_report(const HistogramKernelData * const data, const uint64_t numOfKernels, uint32_t reportType) {

    uint32_t maxKernelId = 0;
    size_t maxKernelNameLength = 11;
    size_t maxBindingNameLength = 12;
    uint32_t maxPortNum = 0;
    uint64_t maxItemCount = 0;
    uint64_t maxFrequency = 0;

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = data[i];
        maxKernelId = std::max(maxKernelId, K.Id);
        maxKernelNameLength = std::max(maxKernelNameLength, strlen(K.KernelName));
        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];
            maxPortNum = std::max(maxPortNum, pd.PortNum);
            maxBindingNameLength = std::max(maxBindingNameLength, strlen(pd.BindingName));
            if (pd.Size == 0) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data);
                while (e) {
                    maxItemCount = std::max(maxItemCount, e->ItemCount);
                    maxFrequency = std::max(maxFrequency, e->Frequency);
                    e = e->Next;
                }
            } else {
                const auto c = pd.Size;
                maxItemCount = std::max(maxItemCount, c);
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < c; ++k) {
                    maxFrequency = std::max(maxFrequency, L[k]);
                }
            }

        }
    }

    auto ceil_log10 = [](const uint64_t v) {
        if (v < 10) {
            return 1U;
        }
        return (unsigned)std::ceil(std::log10(v));
    };

    const auto maxKernelIdLength = ceil_log10(maxKernelId);
    const auto maxPortNumLength =  std::max(4U, ceil_log10(maxPortNum));
    const auto cw = (reportType == HistogramReportType::TransferredItems) ? 11U : 8U;
    const auto maxItemCountLength = std::max(cw, ceil_log10(maxItemCount));
    const auto maxFrequencyLength = std::max(9U, ceil_log10(maxFrequency));

    auto & out = errs();
    if (reportType == HistogramReportType::TransferredItems) {
        out << "TRANSFERRED ITEMS HISTOGRAM:\n\n";
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << "DEFERRED FROM TRANSFERRED ITEM COUNT DISTANCE HISTOGRAM:\n\n";
    }

    out << left_justify("#", maxKernelIdLength + 1); // kernel #
    out << left_justify("Kernel", maxKernelNameLength + 1);
    out << left_justify("Binding", maxBindingNameLength + 1);
    out << left_justify("Port", maxPortNumLength + 2);
    if (reportType == HistogramReportType::TransferredItems) {
        out << left_justify("Transferred", maxItemCountLength + 1);
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << left_justify("Deferred", maxItemCountLength + 1);
    }
    out << left_justify("Frequency", maxFrequencyLength + 1);
    out << "\n\n";

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = data[i];

        const auto id = std::to_string(K.Id);

        const StringRef kernelName = StringRef(K.KernelName, std::strlen(K.KernelName));

        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];

            const StringRef bindingName = StringRef(pd.BindingName, std::strlen(pd.BindingName));

            const auto portType = (pd.PortType == (uint32_t)PortType::Input) ? "I" : "O";
            const auto portNum = std::to_string(pd.PortNum);

            auto printLine = [&](const uint64_t itemCount, const uint64_t freq) {

                const auto itemCountString = std::to_string(itemCount);
                const auto freqString = std::to_string(freq);

                out << left_justify(id, maxKernelIdLength + 1)
                    << left_justify(kernelName, maxKernelNameLength + 1)
                    << left_justify(bindingName, maxBindingNameLength + 1)
                    << portType
                    << left_justify(portNum, maxPortNumLength + 1)
                    << right_justify(itemCountString, maxItemCountLength) << ' '
                    << right_justify(freqString, maxFrequencyLength) << '\n';
            };

            const auto arrayLength = pd.Size;
            if (arrayLength == 0) {
                const HistogramPortListEntry * node = static_cast<const HistogramPortListEntry *>(pd.Data); assert (node);
                // only the root node might have a frequency of 0
                if (node && node->Frequency == 0) {
                    node = node->Next;
                }
                while (node) {
                    assert (node->Frequency > 0);
                    printLine(node->ItemCount, node->Frequency);
                    node = node->Next;
                }
            } else {
                const uint64_t * const array = static_cast<const uint64_t *>(pd.Data);
                for (uint64_t i = 0; i < arrayLength; ++i) {
                    const auto f = array[i];
                    if (f) {
                        printLine(i, f);
                    }
                }
            }

        }
    }

    out << '\n';
}

} // end of extern "C"

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordsAnyHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::recordsAnyHistogramData() const {
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                return true;
            }
            const ProcessingRate & pr = bd.getRate();
            if (mGenerateTransferredItemCountHistogram && !pr.isFixed()) {
                return true;
            }
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                return true;
            }
            const ProcessingRate & pr = bd.getRate();
            if (mGenerateTransferredItemCountHistogram && !pr.isFixed()) {
                return true;
            }
        }
    }
    return false;
}

enum DynamicHistogram {

};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHistogramProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addHistogramProperties(KernelBuilder & b, const size_t kernelId, const size_t groupId) {

    IntegerType * const i64Ty = b.getInt64Ty();

    FixedArray<Type *, 3> fields;
    fields[0] = i64Ty;
    fields[1] = i64Ty;
    fields[2] = b.getVoidPtrTy();
    StructType * const listTy = StructType::get(b.getContext(), fields);

    const auto anyGreedy = hasAnyGreedyInput(kernelId);

    auto addProperties = [&](const BufferPort & br) {
        const Binding & bd = br.Binding;
        const ProcessingRate & pr = bd.getRate();
        if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(kernelId, br.Port);
            mTarget->addInternalScalar(listTy, prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (!mGenerateTransferredItemCountHistogram || (!anyGreedy && pr.isFixed())) {
            return;
        }
        const auto prefix = makeBufferName(kernelId, br.Port);
        Type * histTy = nullptr;
        if (LLVM_UNLIKELY(anyGreedy || pr.isUnknown())) {
            // we do not know how many items will be transferred with any deferred, greedy or unknown
            // rate; keep a linked list of entries. The first one will always refer to a 0-item entry
            // since it's simpler than trying to rearrange the initial entry.
            histTy = listTy;
        } else {
            histTy = ArrayType::get(i64Ty, ceiling(br.Maximum) + 1);
        }
        mTarget->addInternalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
    };

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        addProperties(mBufferGraph[e]);
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        addProperties(mBufferGraph[e]);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief freeHistogramProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freeHistogramProperties(KernelBuilder & b) {

    Function * freeLinkedListFunc = nullptr;

    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const auto anyGreedy = hasAnyGreedyInput(i);

        auto freeLinkedList = [&](Value * root) {

            if (freeLinkedListFunc == nullptr) {


                PointerType * const voidPtrTy = b.getVoidPtrTy();

                IntegerType * const i64Ty = b.getInt64Ty();

                FixedArray<Type *, 3> fields;
                fields[0] = i64Ty;
                fields[1] = i64Ty;
                fields[2] = voidPtrTy;
                StructType * const listTy = StructType::get(b.getContext(), fields);

                PointerType * const listPtrTy = listTy->getPointerTo();

                FunctionType * const funcTy = FunctionType::get(b.getVoidTy(), {listPtrTy}, false);

                const auto ip = b.saveIP();

                Module * const m = b.getModule();

                LLVMContext & C = m->getContext();
                freeLinkedListFunc = Function::Create(funcTy, Function::InternalLinkage, "__freeHistogramLinkedList", m);

                BasicBlock * const entry = BasicBlock::Create(C, "entry", freeLinkedListFunc);
                BasicBlock * const freeLoop = BasicBlock::Create(C, "freeLoop", freeLinkedListFunc);
                BasicBlock * const freeExit = BasicBlock::Create(C, "freeExit", freeLinkedListFunc);

                b.SetInsertPoint(entry);

                auto arg = freeLinkedListFunc->arg_begin();
                auto nextArg = [&]() {
                    assert (arg != freeLinkedListFunc->arg_end());
                    Value * const v = &*arg;
                    std::advance(arg, 1);
                    return v;
                };

                Value * const root = nextArg();
                root->setName("root");
                assert (arg == freeLinkedListFunc->arg_end());

                FixedArray<Value *, 2> offset;
                offset[0] = b.getInt32(0);
                offset[1] = b.getInt32(2);

                Value * const first = b.CreateAlignedLoad(voidPtrTy, b.CreateGEP(listTy, root, offset), PtrTyABIAlignment);
                Value * const nil = ConstantPointerNull::get(voidPtrTy);

                b.CreateCondBr(b.CreateICmpNE(first, nil), freeLoop, freeExit);

                b.SetInsertPoint(freeLoop);
                PHINode * const current = b.CreatePHI(voidPtrTy, 2);
                current->addIncoming(first, entry);
                Value * const currentList = b.CreatePointerCast(current, listPtrTy);
                Value * const next = b.CreateAlignedLoad(voidPtrTy, b.CreateGEP(listTy, currentList, offset), PtrTyABIAlignment);
                b.CreateFree(currentList);
                current->addIncoming(next, freeLoop);
                b.CreateCondBr(b.CreateICmpNE(next, nil), freeLoop, freeExit);

                b.SetInsertPoint(freeExit);
                b.CreateRetVoid();

                b.restoreIP(ip);
            }

            b.CreateCall(freeLinkedListFunc->getFunctionType(), freeLinkedListFunc, {root} );

        };

        auto freeProperties = [&](const BufferPort & br) {
            const Binding & bd = br.Binding;
            const ProcessingRate & pr = bd.getRate();

            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                const auto prefix = makeBufferName(i, br.Port);
                freeLinkedList(getScalarFieldPtr(b, prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX).first);
            }
            if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram && (anyGreedy || pr.isUnknown()))) {
                const auto prefix = makeBufferName(i, br.Port);
                freeLinkedList(getScalarFieldPtr(b, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX).first);
            }

        };

        for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
            freeProperties(mBufferGraph[e]);
        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            freeProperties(mBufferGraph[e]);
        }

    }


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateTransferredItemsForHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTransferredItemsForHistogramData(KernelBuilder & b) {

    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);

    IntegerType * const i64Ty = b.getInt64Ty();
    PointerType * const voidPtrTy = b.getVoidPtrTy();

    const auto anyGreedy = hasAnyGreedyInput(mKernelId);

    auto recordDynamicEntry = [&](Value * const initialEntry, Value * const itemCount, Type * listTy) {

        Module * const m = b.getModule();

        Function * func = m->getFunction("updateHistogramList");
        if (func == nullptr) {

          //  PointerType * const entryPtrTy = type->getPointerTo();
            IntegerType * const sizeTy = b.getSizeTy();

            PointerType * const listPtrTy = listTy->getPointerTo();

            FunctionType * funcTy = FunctionType::get(b.getVoidTy(), {listPtrTy, sizeTy}, false);

            ConstantInt * const i32_ZERO = b.getInt32(0);
            ConstantInt * const i32_ONE = b.getInt32(1);
            ConstantInt * const i32_TWO = b.getInt32(2);
            ConstantInt * const i64_ONE = b.getSize(1);

            const auto ip = b.saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, "updateHistogramList", m);

            BasicBlock * const entry = BasicBlock::Create(C, "entry", func);
            BasicBlock * scanLoop = BasicBlock::Create(C, "scanLoop", func);
            BasicBlock * checkForUpdateOrInsert = BasicBlock::Create(C, "checkInsert", func);
            BasicBlock * updateOrInsertEntry = BasicBlock::Create(C, "updateOrInsertEntry", func);
            BasicBlock * insertNewEntry = BasicBlock::Create(C, "insertNewEntry", func);
            BasicBlock * updateEntry = BasicBlock::Create(C, "updateEntry", func);

            b.SetInsertPoint(entry);

            auto arg = func->arg_begin();
            auto nextArg = [&]() {
                assert (arg != func->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            Value * const firstEntry = nextArg();
            firstEntry->setName("firstEntry");
            Value * const itemCount = nextArg();
            itemCount->setName("itemCount");
            assert (arg == func->arg_end());

            b.CreateUnlikelyCondBr(b.CreateICmpEQ(itemCount, sz_ZERO), updateEntry, scanLoop);

            b.SetInsertPoint(scanLoop);
            PHINode * const lastEntry = b.CreatePHI(listPtrTy, 2, "lastEntry");
            lastEntry->addIncoming(firstEntry, entry);
            PHINode * const lastItemCount = b.CreatePHI(b.getSizeTy(), 2, "lastPosition");
            lastItemCount->addIncoming(sz_ZERO, entry);

            FixedArray<Value *, 2> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_TWO;

            Value * const currentEntryPtr = b.CreateGEP(listTy, lastEntry, offset);
            Value * const currentEntry = b.CreateAlignedLoad(listPtrTy, currentEntryPtr, PtrTyABIAlignment);
            Value * const noMore = b.CreateICmpEQ(currentEntry, ConstantPointerNull::get(listPtrTy));
            b.CreateCondBr(noMore, insertNewEntry, checkForUpdateOrInsert);

            b.SetInsertPoint(checkForUpdateOrInsert);
            offset[1] = i32_ZERO;
            Value * const currentItemCount = b.CreateAlignedLoad(i64Ty, b.CreateGEP(listTy, currentEntry, offset), PtrTyABIAlignment);
            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * const valid = b.CreateICmpULT(lastItemCount, currentItemCount);
                b.CreateAssert(valid, "Histogram history error: last position (%" PRIu64
                                ") >= current position (%" PRIu64 ")", lastItemCount, currentItemCount);
            }
            lastEntry->addIncoming(currentEntry, checkForUpdateOrInsert);
            lastItemCount->addIncoming(currentItemCount, checkForUpdateOrInsert);
            b.CreateCondBr(b.CreateICmpULT(currentItemCount, itemCount), scanLoop, updateOrInsertEntry);

            b.SetInsertPoint(updateOrInsertEntry);
            b.CreateCondBr(b.CreateICmpEQ(currentItemCount, itemCount), updateEntry, insertNewEntry);

            b.SetInsertPoint(insertNewEntry);
            Value * const size = b.getTypeSize(listTy);
            Value * const newEntry = b.CreatePointerCast(b.CreateAlignedMalloc(size, sizeof(uint64_t)), listPtrTy);
            offset[1] = i32_ZERO;
            b.CreateAlignedStore(itemCount, b.CreateGEP(listTy, newEntry, offset), SizeTyABIAlignment);
            offset[1] = i32_ONE;
            b.CreateAlignedStore(i64_ONE, b.CreateGEP(listTy, newEntry, offset), Int64TyABIAlignment);
            offset[1] = i32_TWO;
            b.CreateAlignedStore(b.CreatePointerCast(currentEntry, voidPtrTy), b.CreateGEP(listTy, newEntry, offset), PtrTyABIAlignment);
            b.CreateAlignedStore(b.CreatePointerCast(newEntry, voidPtrTy), b.CreateGEP(listTy, lastEntry, offset), PtrTyABIAlignment);
            b.CreateRetVoid();

            b.SetInsertPoint(updateEntry);
            PHINode * const entryToUpdate = b.CreatePHI(listPtrTy, 2);
            entryToUpdate->addIncoming(currentEntry, updateOrInsertEntry);
            entryToUpdate->addIncoming(firstEntry, entry);
            offset[1] = i32_ONE;
            Value * const ptr = b.CreateGEP(listTy, entryToUpdate, offset);
            Value * const val = b.CreateAdd(b.CreateAlignedLoad(i64Ty, ptr, Int64TyABIAlignment), i64_ONE);
            b.CreateAlignedStore(val, ptr, Int64TyABIAlignment);
            b.CreateRetVoid();

            b.restoreIP(ip);
        }

        FixedArray<Value *, 2> args;
        args[0] = initialEntry;
        args[1] = itemCount;

        b.CreateCall(func->getFunctionType(), func, args);

    };

    auto recordPort = [&](const BufferPort & br) {
        const Binding & bd = br.Binding;
        const ProcessingRate & pr = bd.getRate();

        auto calculateDiff = [&](Value * const A, Value * const B, StringRef Name) -> Value * {
            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * const valid = b.CreateICmpUGE(A, B);
                b.CreateAssert(valid, "Expected %s.%s (%" PRIu64 ") to exceed %s rate (%" PRIu64 ")",
                                mCurrentKernelName, b.GetString(bd.getName()), A, b.GetString(Name), B);
            }
            return b.CreateSub(A, B);
        };

        if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(mKernelId, br.Port);
            Value * base; Type * type;
            std::tie(base, type) = b.getScalarFieldPtr(prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * diff = nullptr;
            if (br.Port.Type == PortType::Input) {
                diff = calculateDiff(mProcessedItemCount[br.Port], mCurrentProcessedDeferredItemCountPhi[br.Port], "processed deferred");
            } else {
                diff = calculateDiff(mProducedItemCount[br.Port], mCurrentProducedDeferredItemCountPhi[br.Port], "produced deferred");
            }
            recordDynamicEntry(base, diff, type);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (mGenerateTransferredItemCountHistogram && (anyGreedy || !pr.isFixed())) {
            const auto prefix = makeBufferName(mKernelId, br.Port);
            Value * base; Type * type;
            std::tie(base, type) = b.getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);

            Value * diff = nullptr;
            if (br.Port.Type == PortType::Input) {
                diff = calculateDiff(mProcessedItemCount[br.Port], mCurrentProcessedItemCountPhi[br.Port], "processed");
            } else {
                diff = calculateDiff(mProducedItemCount[br.Port], mCurrentProducedItemCountPhi[br.Port], "produced");
            }

            if (LLVM_LIKELY(isa<ArrayType>(type))) {
                if (LLVM_UNLIKELY(CheckAssertions)) {
                    Value * const maxSize = b.getSize(type->getArrayNumElements() - 1);
                    Value * const valid = b.CreateICmpULE(diff, maxSize);
                    Constant * const bindingName = b.GetString(bd.getName());
                    b.CreateAssert(valid, "%s.%s: attempting to update %" PRIu64 "-th value of histogram data "
                                           "but internal array can only support up to %" PRIu64 " elements",
                                            mCurrentKernelName, bindingName, diff, maxSize);
                }
                FixedArray<Value *, 2> args;
                args[0] = sz_ZERO;
                args[1] = diff;
                Value * const toInc = b.CreateGEP(type, base, args);
                b.CreateStore(b.CreateAdd(b.CreateAlignedLoad(i64Ty, toInc, Int64TyABIAlignment), sz_ONE), toInc);
            } else {
                recordDynamicEntry(base, diff, type);
            }

        }
    };

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printHistogramReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printHistogramReport(KernelBuilder & b, HistogramReportType type) const {

    // struct HistogramPortData
    FixedArray<Type *, 5> hpdFields;
    hpdFields[0] = b.getInt32Ty(); // PortType
    hpdFields[1] = b.getInt32Ty(); // PortNum
    hpdFields[2] = b.getInt8PtrTy(); // Binding Name
    hpdFields[3] = b.getInt64Ty(); // Size
    hpdFields[4] = b.getVoidPtrTy(); // Data
    StructType * const hpdTy = StructType::get(b.getContext(), hpdFields);

    // struct HistogramKernelData
    FixedArray<Type *, 4> hkdFields;
    hkdFields[0] = b.getInt32Ty(); // Id
    hkdFields[1] = b.getInt32Ty(); // NumOfPorts
    hkdFields[2] = b.getInt8PtrTy(); // KernelName
    hkdFields[3] = hpdTy->getPointerTo(); // PortData
    StructType * const hkdTy = StructType::get(b.getContext(), hkdFields);

    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    DataLayout dl(b.getModule());
    assert (CBuilder::getTypeSize(dl, hpdTy) == sizeof(HistogramPortData));
    assert (CBuilder::getTypeSize(dl, hkdTy) == sizeof(HistogramKernelData));
    END_SCOPED_REGION
    #endif

    ConstantInt * const i32_ZERO = b.getInt32(0);
    ConstantInt * const i32_ONE = b.getInt32(1);
    ConstantInt * const i32_TWO = b.getInt32(2);
    ConstantInt * const i32_THREE = b.getInt32(3);
    ConstantInt * const i32_FOUR = b.getInt32(4);

    PointerType * const voidPtrTy = b.getVoidPtrTy();

    unsigned numOfKernels = 0;

    if (type == HistogramReportType::TransferredItems) {

        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {

            auto addEntry = [](const BufferPort & br) {
                const Binding & bd = br.Binding;
                const ProcessingRate & pr = bd.getRate();
                return !pr.isFixed();
            };

            auto countKernelEntry = [&]() {
                for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }

                for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }
                return false;
            };

            numOfKernels += countKernelEntry() ? 1 : 0;
        }

    } else if (type == HistogramReportType::DeferredItems) {

        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {

            auto addEntry = [](const BufferPort & br) {
                const Binding & bd = br.Binding;
                return bd.hasAttribute(AttrId::Deferred);
            };

            auto countKernelEntry = [&]() {
                for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }

                for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }
                return false;
            };

            numOfKernels += countKernelEntry() ? 1 : 0;

        }

    }

    Value * const kernelData = b.CreateAlignedMalloc(hkdTy, b.getSize(numOfKernels), 0, b.getCacheAlignment());

    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {

        unsigned numOfPorts = 0;

        bool anyGreedy = false;

        if (type == HistogramReportType::TransferredItems) {

            anyGreedy = hasAnyGreedyInput(kernelId);

            auto countPorts = [&](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                if (anyGreedy || !bind.getRate().isFixed()) {
                    numOfPorts++;
                }
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

        } else if (type == HistogramReportType::DeferredItems) {

            auto countPorts = [&](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                if (bind.hasAttribute(AttrId::Deferred)) {
                    numOfPorts++;
                }
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }
        }

        if (numOfPorts == 0) {
            continue;
        }

        FixedArray<Value *, 2> offset;
        offset[0] = b.getInt32(index++);
        offset[1] = i32_ZERO;
        b.CreateAlignedStore(b.getInt32(kernelId), b.CreateGEP(hkdTy, kernelData, offset), Int32TyABIAlignment);
        offset[1] = i32_ONE;
        b.CreateAlignedStore(b.getInt32(numOfPorts), b.CreateGEP(hkdTy, kernelData, offset), Int32TyABIAlignment);
        offset[1] = i32_TWO;
        b.CreateAlignedStore(b.GetString(getKernel(kernelId)->getName()), b.CreateGEP(hkdTy, kernelData, offset), PtrTyABIAlignment);
        Value * const portData = b.CreateAlignedMalloc(hpdTy, b.getSize(numOfPorts), 0, b.getCacheAlignment());
        offset[1] = i32_THREE;
        b.CreateAlignedStore(portData, b.CreateGEP(hkdTy, kernelData, offset), PtrTyABIAlignment);

        unsigned portIndex = 0;

        auto writePortEntry = [&](const BufferPort & br) {
            const Binding & bind =  br.Binding;
            const ProcessingRate & pr = bind.getRate();

            if (type == HistogramReportType::TransferredItems) {
                if (LLVM_LIKELY(!anyGreedy && pr.isFixed())) {
                    return;
                }
            } else if (type == HistogramReportType::DeferredItems) {
                if (LLVM_LIKELY(!bind.hasAttribute(AttrId::Deferred))) {
                    return;
                }
            }

            assert (portIndex < numOfPorts);

            offset[0] = b.getInt32(portIndex++);
            offset[1] = i32_ZERO;
            b.CreateAlignedStore(b.getInt32((unsigned)br.Port.Type), b.CreateGEP(hpdTy, portData, offset), Int32TyABIAlignment);
            offset[1] = i32_ONE;
            b.CreateAlignedStore(b.getInt32(br.Port.Number), b.CreateGEP(hpdTy, portData, offset), Int32TyABIAlignment);
            offset[1] = i32_TWO;
            b.CreateAlignedStore(b.GetString(bind.getName()), b.CreateGEP(hpdTy, portData, offset), PtrTyABIAlignment);

            const auto prefix = makeBufferName(kernelId, br.Port);

            Value * data = nullptr;
            uint64_t maxSize = 0;

            if (type == HistogramReportType::TransferredItems) {
                data = b.getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX).first;
                if (!anyGreedy && !pr.isUnknown()) {
                    maxSize = ceiling(br.Maximum) + 1;
                }
            } else if (type == HistogramReportType::DeferredItems) {
                data = b.getScalarFieldPtr(prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX).first;
            }

            offset[1] = i32_THREE;
            b.CreateAlignedStore(b.getInt64(maxSize), b.CreateGEP(hpdTy, portData, offset), Int64TyABIAlignment);

            offset[1] = i32_FOUR;
            b.CreateAlignedStore(b.CreatePointerCast(data, voidPtrTy), b.CreateGEP(hpdTy, portData, offset), PtrTyABIAlignment);

        };

        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            writePortEntry(mBufferGraph[e]);
        }
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            writePortEntry(mBufferGraph[e]);
        }
        assert (portIndex == numOfPorts);
    }

    // call the report function
    FixedArray<Value *, 3> args;
    args[0] = b.CreatePointerCast(kernelData, voidPtrTy);
    args[1] = b.getInt64(numOfKernels);
    args[2] = b.getInt32((unsigned)type);

    Module * const m = b.getModule();

    Function * const reportPrinter = m->getFunction("__print_pipeline_histogram_report");
    assert (reportPrinter);
    b.CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

    // memory cleanup
    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {

        if (type == HistogramReportType::TransferredItems) {

            auto hasPortData = [](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                return !bind.getRate().isFixed();
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

        } else if (type == HistogramReportType::DeferredItems) {

            auto hasPortData = [](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                return bind.hasAttribute(AttrId::Deferred);
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }
        }
        continue;
free_port_data:
        FixedArray<Value *, 2> offset;
        offset[0] = b.getInt32(index++);
        offset[1] = i32_THREE;
        b.CreateFree(b.CreateAlignedLoad(hpdTy->getPointerTo(), b.CreateGEP(hkdTy, kernelData, offset), PtrTyABIAlignment));
    }
    b.CreateFree(kernelData);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkHistogramFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkHistogramFunctions(KernelBuilder & b) {
    FunctionType * funcTy = FunctionType::get(b.getVoidTy(), {b.getVoidPtrTy(), b.getInt64Ty(), b.getInt32Ty()}, false);
    b.LinkFunction("__print_pipeline_histogram_report", funcTy, (void*)__print_pipeline_histogram_report);
}

}
