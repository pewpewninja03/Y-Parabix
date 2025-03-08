#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getConsumerId
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getTruncatedStreamSetSourceId(const unsigned streamSet) const {
    const BufferNode & bn = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(bn.isTruncated())) {
        for (auto ref : make_iterator_range(in_edges(streamSet, mStreamGraph))) {
            const auto & v = mStreamGraph[ref];
            if (v.Reason == ReasonType::Reference) {
                const auto srcStreamSet = source(ref, mBufferGraph);
                assert (srcStreamSet >= FirstStreamSet && srcStreamSet <= LastStreamSet);
                return srcStreamSet;
            }
        }
        llvm_unreachable("failed to locate source for truncated streamset");
    } else {
        return streamSet;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addConsumerKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addConsumerKernelProperties(KernelBuilder & b, const unsigned kernelId) {
    IntegerType * const sizeTy = b.getSizeTy();

//    const auto addInternallySynchronizedInternalCounters = mIsInternallySynchronized.test(kernelId) && !mIsStatelessKernel.test(kernelId) ;

    const auto groupId = getCacheLineGroupId(kernelId);

    for (const auto e : make_iterator_range(out_edges(kernelId, mConsumerGraph))) {
        const auto streamSet = target(e, mConsumerGraph);
        // If the out-degree for this buffer is zero, then we've proven that its consumption rate
        // is identical to its production rate.
        const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
        if (LLVM_UNLIKELY(numOfIndependentConsumers != 0)) {

            assert (getTruncatedStreamSetSourceId(streamSet) == streamSet);

            // Although we can PHI out the thread's current min. consumed summary count for each
            // buffer, in any complex program, we'll inevitably have general register spill/reloads.
            // By keeping these as stack-allocated variables, the LLVM compiler will hopefully be
            // able to make better decisions whether it should PHI-out these variables.
            mTarget->addNonPersistentScalar(sizeTy, TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet));

            // If we're tracing the consumer item counts, we need to store one for each
            // (non-nested) consumer. Any nested consumers will have their own trace.
            Type * countTy = sizeTy;
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                countTy = ArrayType::get(sizeTy, numOfIndependentConsumers + 1);
            }

            const auto name = CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet);
            mTarget->addInternalScalar(countTy, name, groupId);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readConsumedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mConsumerGraph))) {
        const auto streamSet = target(e, mConsumerGraph);
        Value * consumed = readConsumedItemCount(b, streamSet);
        mInitialConsumedItemCount[streamSet] = consumed; assert (consumed);
        #ifdef PRINT_DEBUG_MESSAGES
        const ConsumerEdge & c = mConsumerGraph[e];
        const StreamSetPort port{PortType::Output, c.Port};
        const auto prefix = makeBufferName(mKernelId, port);
        debugPrint(b, prefix + "_consumed = %" PRIu64, consumed);
        #endif
        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const produced = mInitiallyProducedItemCount[streamSet];
            Value * valid = b.CreateICmpULE(consumed, produced);
            if (mInitiallyTerminated) {
                valid = b.CreateOr(valid, mInitiallyTerminated);
            }
            constexpr auto msg =
                "Consumed item count (%" PRId64 ") of %s.%s exceeds its produced item count (%" PRId64 ").";
            const ConsumerEdge & c = mConsumerGraph[e];
            const StreamSetPort port{PortType::Output, c.Port};
            Constant * const bindingName = b.GetString(getBinding(mKernelId, port).getName());
            b.CreateAssert(valid, msg,
                consumed, mCurrentKernelName, bindingName, produced);
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readExternalConsumerItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readExternalConsumerItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_LIKELY(bn.isOwned())) {
            const BufferPort & externalPort = mBufferGraph[e];
            const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            if (LLVM_UNLIKELY((numOfIndependentConsumers != 0) || (producer == PipelineInput))) {
                Value * const consumed = getConsumedOutputItems(externalPort.Port.Number); assert (consumed);
                setConsumedItemCount(b, streamSet, consumed, 0);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readConsumedItemCount(KernelBuilder & b, const size_t streamSet) {
#ifdef FORCE_PIPELINE_TO_PRESERVE_CONSUMED_DATA
    return b.getSize(0);
#else

    assert (in_degree(streamSet, mBufferGraph) > 0);

    Value * itemCount = nullptr;
    if (out_degree(streamSet, mConsumerGraph) == 0) {

        const auto & bn = mBufferGraph[streamSet];
        // A returned buffer never releases data.
        if (bn.isReturned()) {
            return b.getSize(0);
        }

        // This stream either has no consumers or we've proven that
        // its consumption rate is identical to its production rate.
        Value * produced = mInitiallyProducedItemCount[streamSet];
        assert (isFromCurrentFunction(b, produced, false));
        const auto e = in_edge(streamSet, mBufferGraph);
        const BufferPort & port = mBufferGraph[e];
        if (LLVM_UNLIKELY(produced == nullptr)) {
            const auto producer = source(e, mBufferGraph);
            const auto prefix = makeBufferName(producer, port.Port);
            if (LLVM_UNLIKELY(port.isDeferred())) {
                produced = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            } else {
                produced = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
            }
        }
        auto delayOrLookBehind = std::max(port.Delay, port.LookBehind);
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const auto d = std::max(br.Delay, br.LookBehind);
            delayOrLookBehind = std::max(delayOrLookBehind, d);
        }
        if (delayOrLookBehind) {
            produced = b.CreateSaturatingSub(produced, b.getSize(delayOrLookBehind));
        }
        itemCount = produced;
    } else {
        const auto id = getTruncatedStreamSetSourceId(streamSet);
        auto consumedRef = b.getScalarFieldPtr(CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id));
        Value * ptr = consumedRef.first;
        if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
            Constant * const ZERO = b.getInt32(0);
            ptr = b.CreateInBoundsGEP(consumedRef.second, ptr, { ZERO, ZERO } );
        }
        itemCount = b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment, true);
    }
    assert (itemCount);
    return itemCount;
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeTransitoryConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeTransitoryConsumedItemCount(KernelBuilder & b, const unsigned streamSet, Value * const produced) {
    const auto id = getTruncatedStreamSetSourceId(streamSet);
    if (out_degree(id, mConsumerGraph) != 0) {
        b.setScalarField(TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id), produced);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMinimumConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::computeMinimumConsumedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdateConsumedCount) {
            const StreamSetPort port(PortType::Input, c.Port);
            Value * processed = mFullyProcessedItemCount[port];
            assert (isFromCurrentFunction(b, processed, false));
            // To support the lookbehind attribute, we need to withhold the items from
            // our consumed count and rely on the initial buffer underflow to access any
            // items before the start of the physical buffer.
            const auto input = getInput(mKernelId, port);
            const BufferPort & br = mBufferGraph[input];
            if (LLVM_UNLIKELY(br.LookBehind != 0)) {
                ConstantInt * const amount = b.getSize(br.LookBehind);
                processed = b.CreateSaturatingSub(processed, amount);
            }
            const auto streamSet = source(e, mConsumerGraph);
            assert (streamSet >= FirstStreamSet && streamSet <= LastStreamSet);
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                const ConsumerEdge & c = mConsumerGraph[e]; assert (c.Index > 0);
                setConsumedItemCount(b, streamSet, processed, c.Index);
            }

            const auto id = getTruncatedStreamSetSourceId(streamSet);
            if (out_degree(id, mConsumerGraph) > 0) {
                Value * const transConsumedPtr = getScalarFieldPtr(b, TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id)).first;
                Value * const prior = b.CreateAlignedLoad(b.getSizeTy(), transConsumedPtr, SizeTyABIAlignment);
                const auto output = in_edge(streamSet, mBufferGraph);
                const auto producer = source(output, mBufferGraph);
                const auto prodPrefix = makeBufferName(producer, mBufferGraph[output].Port);
                Value * const minConsumed = b.CreateUMin(prior, processed, prodPrefix + "_minConsumed");
                b.CreateAlignedStore(minConsumed, transConsumedPtr, SizeTyABIAlignment);
                #ifdef PRINT_DEBUG_MESSAGES
                const auto consPrefix = makeBufferName(mKernelId, port);
                debugPrint(b, consPrefix + "_consumed = %" PRIu64 " -> " + prodPrefix + "_consumed' = %" PRIu64, prior, minConsumed);
                #endif
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeFinalConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeConsumedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        const auto streamSet = source(e, mConsumerGraph);
        // check to see if we've fully finished processing any stream
        if (c.Flags & ConsumerEdge::WriteConsumedCount) {
            const auto id = getTruncatedStreamSetSourceId(streamSet);
            Value * const consumed = b.getScalarField(TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id));
            #ifdef PRINT_DEBUG_MESSAGES
            const auto output = in_edge(streamSet, mBufferGraph);
            const BufferPort & br = mBufferGraph[output];
            const auto producer = source(output, mBufferGraph);
            const auto prefix = makeBufferName(producer, br.Port);
            debugPrint(b, " * writing " + prefix + "_consumed = %" PRIu64, consumed);
            #endif
            setConsumedItemCount(b, streamSet, consumed, 0);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::setConsumedItemCount(KernelBuilder & b, const size_t streamSet, Value * consumed, const unsigned slot) const {
    const auto pe = in_edge(streamSet, mBufferGraph);
    const auto producer = source(pe, mBufferGraph);
    const BufferPort & outputPort = mBufferGraph[pe];

    assert (isFromCurrentFunction(b, consumed, false));

    const auto id = getTruncatedStreamSetSourceId(streamSet);



    auto consumedRef = b.getScalarFieldPtr(CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id));
    Value * ptr = consumedRef.first;
    if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
        ptr = b.CreateInBoundsGEP(consumedRef.second, ptr, { b.getInt32(0), b.getInt32(slot) });
    }

    // if we skipped over a partition, we don't want to update the
    // current consumed value; rather than load the old consumed
    // value at the point of production and incur a potential cache
    // miss penalty, just load it here.
    Value * const prior = b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment);
    Value * const skipped = b.CreateIsNull(consumed);
    consumed = b.CreateSelect(skipped, prior, consumed);

    if (LLVM_UNLIKELY(CheckAssertions)) {
        const Binding & output = outputPort.Binding;
        // TODO: cross reference which slot the traced count is for?

        assert (mCurrentKernelName);

        b.CreateAssert(b.CreateICmpULE(prior, consumed),
                        "%s.%s: consumed item count is not monotonically nondecreasing "
                        "(prior %" PRIu64 " > current %" PRIu64 " updated by %s)",
                        mKernelName[producer], b.GetString(output.getName()),
                        prior, consumed, mCurrentKernelName);

    }

    b.CreateAlignedStore(consumed, ptr, SizeTyABIAlignment);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateExternalConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateExternalConsumedItemCounts(KernelBuilder & b) {
    for (const auto input : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(input, mBufferGraph);
        Value * const consumed = readConsumedItemCount(b, streamSet);
        const BufferPort & inputPort = mBufferGraph[input];
        b.CreateAlignedStore(consumed, getProcessedInputItemsPtr(inputPort.Port.Number), SizeTyABIAlignment);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief zeroAnySkippedTransitoryConsumedItemCountsUntil
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::zeroAnySkippedTransitoryConsumedItemCountsUntil(KernelBuilder & b, const unsigned targetKernelId) {

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        for (const auto e : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
            assert (streamSet == getTruncatedStreamSetSourceId(streamSet));
            const auto consumer = target(e, mConsumerGraph);
            if (consumer >= mKernelId) { // && consumer <= targetKernelId
                const auto name = TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet);
                Value * const transConsumedPtr = getScalarFieldPtr(b, name).first;
                b.CreateAlignedStore(b.getSize(0), transConsumedPtr, SizeTyABIAlignment);
                break;
            }
        }
    }
}

}
