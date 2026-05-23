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

    const auto bw = b.getBitBlockWidth();

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
        const BufferPort & bp = mBufferGraph[in_edge(streamSet, mBufferGraph)];
        if (LLVM_UNLIKELY(bp.RequiredOverflowSpace)) {
            const auto k = ceiling(Rational{bp.RequiredOverflowSpace, bw});
            maxStrides = b.CreateAdd(maxStrides, b.getSize(k));
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
    Value * const base = b.CreateAlignedMalloc(memorySize, pageSize);
    PointerType * const int8PtrTy = b.getInt8PtrTy();
    b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY, b.CreatePointerCast(base, int8PtrTy));
    b.setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, memorySize);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeThreadLocalMemoryPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeThreadLocalMemoryPhiNodes(KernelBuilder & b) {

    assert (mIsPartitionRoot);

    const auto oneAfterLastKernel = FirstKernelInPartition[mCurrentPartitionId + 1];

    IntegerType * const sizeTy = b.getSizeTy();

    PointerType * const int8PtrTy = b.getInt8PtrTy();
    const auto prefix = makeKernelName(mKernelId);
    mThreadLocalStreamSetBaseAddressAtEntryPhi = PHINode::Create(int8PtrTy, 2, prefix + "_threadLocalBaseAddressPhi", mKernelLoopEntry);
    mThreadLocalStreamSetBaseAddressAtEntryPhi->addIncoming(UndefValue::get(int8PtrTy), mKernelLoopStart);
    mThreadLocalStreamSetBaseAddress = mThreadLocalStreamSetBaseAddressAtEntryPhi;

    #ifndef NDEBUG
    mThreadLocalStartOffsetAtEntryPhi.reset(FirstStreamSet, LastStreamSet);
    mThreadLocalStartOffsetAtExitPhi.reset(FirstStreamSet, LastStreamSet);
    #endif

    Constant * const undefVal = UndefValue::get(sizeTy);

    for (auto kernel = mKernelId; kernel < oneAfterLastKernel; ++kernel) {
        for (auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isThreadLocal()) {
                const auto prefix = makeBufferName(kernel, mBufferGraph[output].Port);
                PHINode * const entryPhi = PHINode::Create(sizeTy, 2, prefix + "_threadLocalOffsetAtLoopEntry", mKernelLoopEntry);
                mThreadLocalStartOffsetAtEntryPhi[streamSet] = entryPhi;
                entryPhi->addIncoming(undefVal, mKernelLoopStart);
                mThreadLocalStartOffsetAtExitPhi[streamSet] = PHINode::Create(sizeTy, 2, prefix + "_threadLocalOffsetAtLoopExit", mKernelLoopExit);
            }
        }
    }




}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief scaleThreadLocalMemoryToMaximumNumOfStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::allocateThreadLocalMemoryForMaximumNumOfStrides(KernelBuilder & b, Value * maximumNumOfStrides) {

    assert (mIsPartitionRoot);

    if (out_degree(mCurrentPartitionId, ThreadLocalPlacement) == 0) {
        return;
    }

    #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
    constexpr size_t THREAD_LOCAL_ALLOC_SCALE = THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
    #else
    constexpr size_t THREAD_LOCAL_ALLOC_SCALE = 1;
    #endif


    BasicBlock * const allocateThreadLocal = b.CreateBasicBlock("allocateThreadLocal", mKernelLoopCall);
    BasicBlock * const allocateThreadLocalExit = b.CreateBasicBlock("allocateThreadLocalExit", mKernelLoopCall);

    PointerType * const int8PtrTy = b.getInt8PtrTy();

    BasicBlock * const entryBlock = b.GetInsertBlock();

    b.CreateUnlikelyCondBr(mExecutedAtLeastOnceAtLoopEntryPhi, allocateThreadLocalExit, allocateThreadLocal);

    b.SetInsertPoint(allocateThreadLocal);

    Value * threadLocalPtr = nullptr;
    Type * threadLocalTy = nullptr;
    std::tie(threadLocalPtr, threadLocalTy) = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);

    const auto m = PartitionCount + LastStreamSet - FirstStreamSet + 1;
    assert (num_vertices(ThreadLocalPlacement) == m + 1);
    std::vector<unsigned> toVisit(m + 1, 0);
    for (unsigned i = PartitionCount; i < m; ++i) {
        toVisit[i] = in_degree(i, ThreadLocalPlacement);
    }
    assert (in_degree(m, ThreadLocalPlacement) < -1U);
    toVisit[m] = -1U;

    const auto pageSize = getPageSize();
    assert (is_pow2(pageSize));

    ConstantInt * const LOG_2_PAGE_SIZE = b.getSize(floor_log2(pageSize));

    std::queue<unsigned> Q;

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "mMaximumNumOfStrides%" PRIx64 " = %" PRIx64,
               b.getSize(hasSlidingWindow), mMaximumNumOfStrides);
    #endif

    Value * const sz_ZERO = b.getSize(0);
    Value * const sz_ONE = b.getSize(1);

    #ifndef NDEBUG
    flat_set<unsigned> visited;
    visited.emplace(mCurrentPartitionId);
    #endif


    const auto bw = b.getBitBlockWidth();

    maximumNumOfStrides = b.CreateUMax(maximumNumOfStrides, b.getSize(StrideStepLength[mKernelId]));

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
                Value * maxStrides = maximumNumOfStrides; // mMaximumNumOfStrides;

                const BufferPort & bp = mBufferGraph[in_edge(streamSet, mBufferGraph)];
                if (LLVM_UNLIKELY(bp.RequiredOverflowSpace)) {
                    const auto k = ceiling(Rational{bp.RequiredOverflowSpace, bw});
                    maxStrides = b.CreateAdd(maxStrides, b.getSize(k));
                }
                const auto & P = ThreadLocalPlacement[e];
                Value * const off = b.CreateShl(b.CreateCeilUMulRational(maxStrides, P * THREAD_LOCAL_ALLOC_SCALE), LOG_2_PAGE_SIZE);
                if (u < PartitionCount) {
                    start = sz_ZERO;
                    end = off;
                } else {
                    for (auto in : make_iterator_range(in_edges(v, ThreadLocalPlacement))) {
                        const auto src = source(in, ThreadLocalPlacement) + FirstStreamSet - PartitionCount;
                        assert (mThreadLocalEndOffset[src]);
                        start = b.CreateUMax(start, mThreadLocalEndOffset[src]);
                    }
                    end = b.CreateAdd(start, off);
                }

                mThreadLocalStartOffset[streamSet] = start;
                mThreadLocalEndOffset[streamSet] = end;

                if (LLVM_UNLIKELY(mCheckStreamSets)) {
                    auto & dl = b.getModule()->getDataLayout();
                    ExternalBuffer * const buf = cast<ExternalBuffer>(bn.Buffer);
                    const auto ts = b.getTypeSize(dl, buf->getType());
                    buf->setCapacity(b, b.CreateMulRational(off, Rational{bw, ts}));
                }

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

    assert (memoryForSegment);

    mThreadLocalStreamSetBaseAddress = b.CreateAlignedLoad(int8PtrTy, threadLocalPtr, PtrTyABIAlignment);

    BasicBlock * const expandThreadLocalMemory = b.CreateBasicBlock("expandThreadLocalMemory", allocateThreadLocalExit);
    BasicBlock * const afterExpansion = b.CreateBasicBlock("afterExpansion", allocateThreadLocalExit);
    Value * currentMem = b.CreateAlignedLoad(b.getSizeTy(), mThreadLocalMemorySizePtr, SizeTyABIAlignment);
    Value * const needsExpansion = b.CreateICmpUGT(memoryForSegment, currentMem);
    b.CreateUnlikelyCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

    b.SetInsertPoint(expandThreadLocalMemory);
    b.CreateFree(mThreadLocalStreamSetBaseAddress);
    // At minimum, we want to double the required space to minimize future reallocs
    Value * const newSize = b.CreateRoundUpRational(b.CreateRoundUp(memoryForSegment, currentMem), pageSize);
    b.CreateAlignedStore(newSize, mThreadLocalMemorySizePtr, SizeTyABIAlignment);
    Value * const newThreadLocalBaseAddress = b.CreateAlignedMalloc(newSize, pageSize);
    b.CreateAlignedStore(newThreadLocalBaseAddress, threadLocalPtr, PtrTyABIAlignment);
    b.CreateBr(afterExpansion);

    b.SetInsertPoint(afterExpansion);
    PHINode * const basePtrPhi = b.CreatePHI(int8PtrTy, 2);
    basePtrPhi->addIncoming(mThreadLocalStreamSetBaseAddress, allocateThreadLocal);
    basePtrPhi->addIncoming(newThreadLocalBaseAddress, expandThreadLocalMemory);
    mThreadLocalStreamSetBaseAddress = basePtrPhi;

    if (LLVM_UNLIKELY(CheckAssertions())) {

//        Value * currentMem = b.CreateAlignedLoad(b.getSizeTy(), mThreadLocalMemorySizePtr, SizeTyABIAlignment);
//        Value * const noExpansion = b.CreateICmpULE(memoryForSegment, currentMem);
//        b.CreateAssert(noExpansion, "%s requires more thread-local memory (%" PRIu64 ") than maximum (%" PRIu64 ")?",
//                       mCurrentKernelName, memoryForSegment, currentMem);

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

    remapThreadLocalBufferMemory(b);
    BasicBlock * const remapExitBlock = b.GetInsertBlock();
    b.CreateBr(allocateThreadLocalExit);

    b.SetInsertPoint(allocateThreadLocalExit);
    PHINode * const threadLocalBaseAttrPhi = b.CreatePHI(int8PtrTy, 2);
    threadLocalBaseAttrPhi->addIncoming(mThreadLocalStreamSetBaseAddressAtEntryPhi, entryBlock);
    threadLocalBaseAttrPhi->addIncoming(mThreadLocalStreamSetBaseAddress, remapExitBlock);
    mThreadLocalStreamSetBaseAddress = threadLocalBaseAttrPhi;

    IntegerType * const sizeTy = b.getSizeTy();
    const auto oneAfterLastKernel = FirstKernelInPartition[mCurrentPartitionId + 1];
    for (auto kernel = mKernelId; kernel < oneAfterLastKernel; ++kernel) {
        for (auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isThreadLocal()) {
                PHINode * const phi = b.CreatePHI(sizeTy, 2);
                phi->addIncoming(mThreadLocalStartOffsetAtEntryPhi[streamSet], entryBlock);
                phi->addIncoming(mThreadLocalStartOffset[streamSet], remapExitBlock);
                mThreadLocalStartOffset[streamSet] = phi;
            }
        }
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief assignThreadLocalBufferMemoryForPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::remapThreadLocalBufferMemory(KernelBuilder & b) {

    const auto blockWidth = b.getBitBlockWidth();

    auto & DL = b.getModule()->getDataLayout();

    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(output, mBufferGraph);
        assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {

            assert (out_degree(mCurrentPartitionId, ThreadLocalPlacement) > 0);

            ExternalBuffer * const buffer = cast<ExternalBuffer>(bn.Buffer);
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

        }
    }
}


}
