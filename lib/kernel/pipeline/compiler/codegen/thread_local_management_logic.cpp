#include "../pipeline_compiler.hpp"
#include <queue>
#include <stack>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeThreadLocalMemory
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeThreadLocalMemory(KernelBuilder & b, Value * const segmentSize) {

    if (num_edges(ThreadLocalPlacement) == 0) {
        return;
    }

    #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
    constexpr size_t THREAD_LOCAL_ALLOC_SCALE = THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
    #else
    constexpr size_t THREAD_LOCAL_ALLOC_SCALE = 1;
    #endif

    using Vertex = ThreadLocalPlacementGraph::vertex_descriptor;

    const auto n = (LastStreamSet - FirstStreamSet) + 1;

    std::vector<Value *> precalculatedOffset(n, nullptr);

    std::function<Value *(Vertex)> calculatePlacement = [&](const Vertex v) -> Value * {
        assert (v >= PartitionCount);

        Value * offset = precalculatedOffset[v - PartitionCount];
        if (offset) {
            return offset;
        }
        const auto streamSet = FirstStreamSet + v - PartitionCount;
        assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);


        assert (in_degree(v, ThreadLocalPlacement) > 0);
        ThreadLocalPlacementGraph::in_edge_iterator begin, end;
        std::tie(begin, end) = in_edges(v, ThreadLocalPlacement);

        const auto first = source(*begin, ThreadLocalPlacement);
        Value * max = nullptr;
        if (first < PartitionCount) {
            assert (in_degree(v, ThreadLocalPlacement) == 1);
        } else {
            max = calculatePlacement(first);
            for (auto ei = begin; ++ei != end; ) {
                max = b.CreateUMax(max, calculatePlacement(source(*ei, ThreadLocalPlacement)));
            }
            assert (max);
        }

        const auto producer = parent(streamSet, mBufferGraph);
        Rational scale{THREAD_LOCAL_ALLOC_SCALE * mTarget->getStride() * MaximumNumOfStrides[producer], getKernel(producer)->getStride() * StrideStepLength[producer]};

        const auto & bn = mBufferGraph[streamSet];
        assert (bn.isThreadLocal());
        Value * maxStrides = b.CreateCeilUMulRational(segmentSize, bn.RelativeIORate * scale);
        if (LLVM_UNLIKELY(bn.NumOfOverflowStrides)) {
            maxStrides = b.CreateAdd(maxStrides, b.getSize(bn.NumOfOverflowStrides));
        }
        Value * off = b.CreateCeilUMulRational(maxStrides, ThreadLocalPlacement[*begin]);
        if (max) {
            off = b.CreateAdd(off, max);
        }
        precalculatedOffset[v - PartitionCount] = off;

        if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
            auto src = streamSet;
            do {
                src = parent(src, InOutStreamSetReplacement);
                assert (FirstStreamSet <= src && src <= LastStreamSet);
                precalculatedOffset[src - FirstStreamSet] = off;
            } while (LLVM_UNLIKELY(in_degree(src, InOutStreamSetReplacement) != 0));
        }

        return off;
    };

    const auto m = PartitionCount + (LastStreamSet - FirstStreamSet) + 1;
    assert (num_vertices(ThreadLocalPlacement) == m + 1);
    assert (in_degree(m, ThreadLocalPlacement) > 0);
    ThreadLocalPlacementGraph::in_edge_iterator ei, end;
    std::tie(ei, end) = in_edges(m, ThreadLocalPlacement);
    Value * memorySize = calculatePlacement(source(*ei, ThreadLocalPlacement));
    while (++ei != end) {
        auto ms = calculatePlacement(source(*ei, ThreadLocalPlacement));
        memorySize = b.CreateUMax(memorySize, ms);
    }
    assert (memorySize);
    const auto pageSize = getPageSize();
    assert (is_pow2(pageSize));
    memorySize = b.CreateShl(memorySize, b.getSize(floor_log2(pageSize)));

    assert (mTarget->hasThreadLocal());
    Value * const base = b.CreatePageAlignedMalloc(memorySize);
    PointerType * const int8PtrTy = b.getInt8PtrTy();
    b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY, b.CreatePointerCast(base, int8PtrTy));
    b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, memorySize);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief scaleThreadLocalMemoryToMaximumNumOfStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::allocateThreadLocalMemoryForMaximumNumOfStrides(KernelBuilder & b) {

    if (out_degree(mCurrentPartitionId, ThreadLocalPlacement) == 0) {
            mThreadLocalStreamSetBaseAddress = nullptr;
            return;
        }

        #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
        constexpr size_t THREAD_LOCAL_ALLOC_SCALE = THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
        #else
        constexpr size_t THREAD_LOCAL_ALLOC_SCALE = 1;
        #endif

        Value * threadLocalPtr = nullptr;
        Type * threadLocalTy = nullptr;

        const auto hasSlidingWindow = mBufferGraph[mKernelId].permitSlidingWindow();

        BasicBlock * rebuildOffsets = nullptr;

        if (hasSlidingWindow) {
            rebuildOffsets = b.CreateBasicBlock();
            b.CreateBr(rebuildOffsets);
            b.SetInsertPoint(rebuildOffsets);
        }

        std::tie(threadLocalPtr, threadLocalTy) = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);

        const auto m = PartitionCount + LastStreamSet - FirstStreamSet + 1;
        assert (num_vertices(ThreadLocalPlacement) == m + 1);
        std::vector<unsigned> toVisit(m, 0);
        for (unsigned i = PartitionCount; i < m; ++i) {
            toVisit[i] = in_degree(i, ThreadLocalPlacement);
        }

        const auto pageSize = getPageSize();
        assert (is_pow2(pageSize));

        ConstantInt * const LOG_2_PAGE_SIZE = b.getSize(floor_log2(pageSize));

        std::queue<unsigned> Q;

        assert (mMaximumNumOfStrides);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "mMaximumNumOfStrides%" PRIx64 " = %" PRIx64,
                   b.getSize(hasSlidingWindow), mMaximumNumOfStrides);
        #endif

        Value * const sz_ZERO = b.getSize(0);

        #ifndef NDEBUG
        flat_set<unsigned> visited;
        visited.emplace(mCurrentPartitionId);
        #endif

        Value * memoryForSegment = nullptr;
        for (auto u = mCurrentPartitionId;;) {
            for (auto e : make_iterator_range(out_edges(u, ThreadLocalPlacement))) {
                const auto v = target(e, ThreadLocalPlacement);
                assert (PartitionCount <= v && v <= m);
                assert (v != u);
                auto & T = toVisit[v];
                assert (T > 0);
                T--;
                if (T == 0) {



                    const auto streamSet = v + FirstStreamSet - PartitionCount;
                    const BufferNode & bn = mBufferGraph[streamSet];
                    assert (bn.isThreadLocal());
                    assert (!bn.isInOutRedirect());
                    assert (mThreadLocalStartOffset[streamSet] == nullptr);
                    assert (mThreadLocalEndOffset[streamSet] == nullptr);

                    Value * start = nullptr;
                    Value * end = nullptr;
                    Value * maxStrides = mMaximumNumOfStrides;
                    if (LLVM_UNLIKELY(bn.NumOfOverflowStrides)) {
                        maxStrides = b.CreateAdd(maxStrides, b.getSize(bn.NumOfOverflowStrides));
                    }
                    const auto & P = ThreadLocalPlacement[e];
                    Value * const off = b.CreateShl(b.CreateCeilUMulRational(maxStrides, P * THREAD_LOCAL_ALLOC_SCALE), LOG_2_PAGE_SIZE);
                    if (u < PartitionCount) {
                        start = sz_ZERO;
                        end = off;
                    } else {
                        for (auto in : make_iterator_range(in_edges(v, ThreadLocalPlacement))) {
                            const auto src = source(in, ThreadLocalPlacement) + FirstStreamSet - PartitionCount;
                            start = b.CreateUMax(start, mThreadLocalEndOffset[src]);
                        }
                        end = b.CreateAdd(start, off);
                    }

                    mThreadLocalStartOffset[streamSet] = start;
                    mThreadLocalEndOffset[streamSet] = end;


                    for (auto inOut = streamSet; LLVM_UNLIKELY(out_degree(inOut, InOutStreamSetReplacement)); ) {
                        inOut = child(inOut, InOutStreamSetReplacement);
                        mThreadLocalStartOffset[inOut] = start;
                        mThreadLocalEndOffset[inOut] = end;
                    }

                    #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
                    debugPrint(b, "mappedAddrRange" + std::to_string(streamSet) +
                               " (" + std::to_string(P.numerator()) + "/" + std::to_string(P.denominator()) +
                               ") = [%" PRIx64 ", %" PRIx64 ")", start, end);
                    #endif

                    if (ThreadLocalPlacement[v]) {
                        memoryForSegment = b.CreateUMax(memoryForSegment, end);
                    } else if (out_degree(v, ThreadLocalPlacement) > 0) {
                        #ifndef NDEBUG
                        assert (visited.count(v) == 0);
                        visited.emplace(v);
                        #endif
                        Q.push(v);
                    }
                }
            }
            if (Q.empty()) {
                break;
            }
            u = Q.front();
            assert (toVisit[u] == 0);
            Q.pop();
        }

        if (hasSlidingWindow) {
            BasicBlock * const expandThreadLocalMemory = b.CreateBasicBlock();
            BasicBlock * const afterExpansion = b.CreateBasicBlock();
            Value * currentMem = b.CreateAlignedLoad(b.getSizeTy(), mThreadLocalMemorySizePtr, SizeTyABIAlignment);

            #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
            debugPrint(b, "memoryForSegment(%" PRIx64 ") > currentMem (%" PRIx64 ") ?", memoryForSegment, currentMem);
            #endif

            Value * const needsExpansion = b.CreateICmpUGT(memoryForSegment, currentMem);
            b.CreateCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

            b.SetInsertPoint(expandThreadLocalMemory);

            b.CreateFree(b.CreateAlignedLoad(threadLocalTy, threadLocalPtr, PtrTyABIAlignment));
            // At minimum, we want to double the required space to minimize future reallocs
            Value * expanded = b.CreateRoundUp(memoryForSegment, currentMem);
            b.CreateAlignedStore(expanded, mThreadLocalMemorySizePtr, SizeTyABIAlignment);
            Value * const base = b.CreatePageAlignedMalloc(expanded);
            b.CreateAlignedStore(base, threadLocalPtr, PtrTyABIAlignment);

            b.CreateBr(rebuildOffsets);

            b.SetInsertPoint(afterExpansion);
        } else if (LLVM_UNLIKELY(CheckAssertions())) {
            Value * currentMem = b.CreateAlignedLoad(b.getSizeTy(), mThreadLocalMemorySizePtr, SizeTyABIAlignment);
            Value * const noExpansion = b.CreateICmpULE(memoryForSegment, currentMem);
            b.CreateAssert(noExpansion, "%s requires more thread-local memory (%" PRIu64 ") than maximum (%" PRIu64 ")?",
                           mCurrentKernelName, memoryForSegment, currentMem);
        }

        if (LLVM_UNLIKELY(CheckAssertions())) {
            for (auto e : make_iterator_range(edges(ThreadLocalConflictGraph))) {
                const auto x = source(e, ThreadLocalConflictGraph) + FirstStreamSet;
                assert (FirstStreamSet <= x && x <= LastStreamSet);
                const auto producer = parent(x, mBufferGraph);
                if (KernelPartitionId[producer] == mCurrentPartitionId) {
                    const auto y = target(e, ThreadLocalConflictGraph) + FirstStreamSet;
                    assert (FirstStreamSet <= y && y <= LastStreamSet);
                    Value * const after = b.CreateICmpULE(mThreadLocalEndOffset[x], mThreadLocalStartOffset[y]);
                    Value * const before = b.CreateICmpULE(mThreadLocalEndOffset[y], mThreadLocalStartOffset[x]);
                    b.CreateAssert(b.CreateOr(before, after),
                            "Thread-local "
                            "streamset %" PRIu64 " [%" PRIx64 ",%" PRIx64 ") overlaps "
                            "streamset %" PRIu64 " [%" PRIx64 ",%" PRIx64 ")",
                            b.getSize(x), mThreadLocalStartOffset[x], mThreadLocalEndOffset[x],
                            b.getSize(y), mThreadLocalStartOffset[y], mThreadLocalEndOffset[y]);
                }
            }
        }

        mThreadLocalStreamSetBaseAddress = b.CreateAlignedLoad(threadLocalTy, threadLocalPtr, PtrTyABIAlignment);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief assignThreadLocalBufferMemoryForPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::remapThreadLocalBufferMemory(KernelBuilder & b) {

    const auto blockWidth = b.getBitBlockWidth();

    auto & DL = b.getModule()->getDataLayout();

    const auto checkStreamSet = codegen::DebugOptionIsSet(codegen::EnableAsserts, codegen::EnableStreamSetAsserts);

    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(output, mBufferGraph);
        assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {

            assert (out_degree(mCurrentPartitionId, ThreadLocalPlacement) > 0);

            ExternalBuffer * const buffer = cast<ExternalBuffer>(bn.OutputBuffer);
            Value * const produced = mInitiallyProducedItemCount[streamSet];
            PointerType * const ptrTy = buffer->getPointerType();

            const auto typeWidth = b.getTypeSize(DL, buffer->getType());

            const auto fieldWidth = buffer->getFieldWidth();

            Value * offsetBytes = nullptr;
            if (fieldWidth == 1) {
                Rational bytesPerItem{typeWidth, blockWidth};
                offsetBytes = b.CreateMulRational(produced, bytesPerItem);
            } else {
                // NOTE: these multiplications effectively compute the floor of each equation and are not
                // equivalent to the fieldWidth=1 calculation despite appearing so.
                Rational fieldsPerBlock{fieldWidth, blockWidth};
                Value * const offsetVecOffset = b.CreateMulRational(produced, fieldsPerBlock);
                Rational bytesPerBlock{typeWidth, fieldWidth};
                offsetBytes = b.CreateMulRational(offsetVecOffset, bytesPerBlock);
            }
            Value * const startOffset = mThreadLocalStartOffset[streamSet]; assert (startOffset);
            Value * const virtualBaseOffset = b.CreateSub(startOffset, offsetBytes);
            Value * const ba = b.CreateGEP(b.getInt8Ty(), mThreadLocalStreamSetBaseAddress, virtualBaseOffset);
            buffer->setBaseAddress(b, b.CreatePointerCast(ba, ptrTy));
            if (LLVM_UNLIKELY(CheckAssertions())) {
                Value * size = b.CreateSub(mThreadLocalEndOffset[streamSet], mThreadLocalStartOffset[streamSet]);
                Rational itemsPerByte{blockWidth, typeWidth};
                Value * capacity = b.CreateMulRational(size, itemsPerByte);
                buffer->setCapacity(b, capacity);
            }
        }
    }
}


}
