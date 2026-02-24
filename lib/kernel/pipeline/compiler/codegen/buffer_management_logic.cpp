#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHandlesToPipelineKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addBufferHandlesToPipelineKernel(KernelBuilder & b, const unsigned kernelId, const unsigned groupId) {

    bool hasAnyInternalStreamSets = false;
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);

        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated() || bn.isInOutRedirect() || bn.hasZeroElementsOrWidth() || bn.isConstant())) {
            continue;
        }

        const BufferPort & rd = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, rd.Port);
        StreamSetBuffer * const buffer = bn.Buffer;

        bool requiresLGVBA = rd.isManaged();

        // external buffers already have a buffer handle
        if (LLVM_LIKELY(bn.isInternal() || bn.isConstant())) {
            Type * const handleType = buffer->getHandleType(b);
            // We automatically assign the buffer memory according to the buffer start position
            if (LLVM_UNLIKELY(bn.isConstant())) {
                const auto rs = cast<RepeatingStreamSet>(mStreamGraph[streamSet].Relationship);
                if (rs->isDynamic()) {
                    mTarget->addInternalScalar(handleType, prefix, groupId);
                } else {
                    mTarget->addNonPersistentScalar(handleType, prefix);
                }
            } else if (bn.isThreadLocal()) {
                hasAnyInternalStreamSets = true;
                mTarget->addNonPersistentScalar(handleType, prefix);
            } else if (LLVM_LIKELY(bn.isOwned() || bn.hasZeroElementsOrWidth())) {
                hasAnyInternalStreamSets = true;
                mTarget->addInternalScalar(handleType, prefix, groupId);
            } else {
                mTarget->addNonPersistentScalar(handleType, prefix);
                requiresLGVBA = true;
            }
        }

        if (requiresLGVBA) {
            mTarget->addInternalScalar(buffer->getPointerType(), prefix + LAST_GOOD_VIRTUAL_BASE_ADDRESS, groupId);
        }

    }

    if (LLVM_UNLIKELY(!mTarget->allocatesInternalStreamSets())) {
        const Kernel * const kernelObj = getKernel(kernelId);
        if (LLVM_UNLIKELY(hasAnyInternalStreamSets)) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but must do so to support " << kernelObj->getName() << ".";
            report_fatal_error(StringRef(msg.str()));
        }
        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but " << kernelObj->getName() << " must do so to be correctly initialized.";
            report_fatal_error(StringRef(msg.str()));
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadInternalStreamSetHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadInternalStreamSetHandles(KernelBuilder & b, const bool nonLocal) {
    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        return;
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated() || bn.isInOutRedirect())) continue;
        // external buffers already have a buffer handle
        StreamSetBuffer * const buffer = bn.Buffer;
        if (bn.isNonThreadLocal() == nonLocal) {
            if (LLVM_UNLIKELY(bn.isConstant())) {
                assert (nonLocal && !buffer->isDynamic());
                const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
                buffer->setHandle(b.getScalarFieldPtr(handleName));
                const auto & sn = mStreamGraph[streamSet];
                assert (sn.Type == RelationshipNode::IsStreamSet);
                if (cast<RepeatingStreamSet>(sn.Relationship)->isDynamic()) {
                    const auto lengthName = REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet);
                    Value * const mod = b.getScalarField(lengthName);
                    cast<RepeatingBuffer>(buffer)->setModulus(mod);
                } else {
                    assert(isa<Constant>(cast<RepeatingBuffer>(buffer)->getModulus()));
                }
            } else if (bn.isInternal()) {
                const auto pe = in_edge(streamSet, mBufferGraph);
                const auto producer = source(pe, mBufferGraph);
                const BufferPort & rd = mBufferGraph[pe];
                const auto handleName = makeBufferName(producer, rd.Port);
                buffer->setHandle(b.getScalarFieldPtr(handleName));
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReturnedBufferScaleFactor
 ** ------------------------------------------------------------------------------------------------------------- */
Rational PipelineCompiler::getReturnedBufferScaleFactor(const size_t streamSet) const {
    Rational scaleFactor{0, 1};
    auto updateScaleFactorForPort = [&](BufferGraph::edge_descriptor port) {
        const BufferPort & rd = mBufferGraph[port];
        const Binding & bindingRef = rd.Binding;
        const AttributeSet & attrs = bindingRef.getAttributes();
        if (attrs.hasAttribute(AttrId::ReturnedBuffer)) {
            const Attribute & attrRef = attrs.findAttribute(AttrId::ReturnedBuffer);
            scaleFactor = std::max(scaleFactor, attrRef.ratio());
        }
    };
    updateScaleFactorForPort(in_edge(streamSet, mBufferGraph));
    for (const auto port : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
        updateScaleFactorForPort(port);
    }
    return scaleFactor;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::allocateOwnedBuffers(KernelBuilder & b, Value * const allocScale, Value * const expectedSourceOutputSize, const bool nonLocal) {
    assert (allocScale);
    if (LLVM_UNLIKELY(CheckAssertions())) {
        Value * const valid = b.CreateIsNotNull(allocScale);
        b.CreateAssert(valid,
           "%s: expected number of strides for internally allocated buffers is 0",
           b.GetString(mTarget->getName()));
    }
    b.CreateAssert(allocScale, "allocScale cannot be 0");

    // recursively allocate any internal buffers for the nested kernels, giving them the correct
    // num of strides it should expect to perform

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernelObj = getKernel(i);

        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            if (nonLocal || kernelObj->hasThreadLocal()) {
                setActiveKernel(b, i, !nonLocal);
                assert (mKernel == kernelObj);
                SmallVector<Value *, 5> params;
                if (LLVM_LIKELY(mKernelSharedHandle)) {
                    params.push_back(mKernelSharedHandle);
                }
                Value * func = nullptr;
                FunctionType * funcTy;
                if (nonLocal) {
                    std::tie(func, funcTy) = getKernelAllocateSharedInternalStreamSetsFunction(b);
                } else {
                    std::tie(func, funcTy) = getKernelAllocateThreadLocalInternalStreamSetsFunction(b);
                    params.push_back(mKernelThreadLocalHandle);
                }

                const Rational factor {mTarget->getStride() * MaximumNumOfStrides[i], kernelObj->getStride()};
                assert (factor.numerator() > 0);
                params.push_back(b.CreateMulRational(allocScale, factor));
                if (LLVM_UNLIKELY(mTraceDynamicBuffers && (kernelObj->getKernelFlags() & Kernel::KernelFlags::HasInternallyManagedStreamSet) && nonLocal)) {
                    params.push_back(generateBufferExpansionFunctionForCurrentKernel(b, i));
                    params.push_back(b.CreatePointerCast(getHandle(), b.getVoidPtrTy()));
                }
                b.CreateCall(funcTy, func, params);
            }
        }
    }

    #ifdef PRINT_DEBUG_MESSAGES
    auto & dl = b.getModule()->getDataLayout();
    #endif

    // and allocate any output buffers

    for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {

        Value * reportCallback = nullptr;
        Value * sharedHandle = nullptr;

        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isTruncated() || bn.isInOutRedirect() || bn.hasZeroElementsOrWidth())) {
                continue;
            }
            assert (bn.isNonThreadLocal() || bn.isOwned());
            if (bn.isNonThreadLocal() == nonLocal && bn.isOwned()) {
                StreamSetBuffer * const buffer = bn.Buffer;

                if (LLVM_UNLIKELY(bn.isConstant())) {
                    generateGlobalDataForRepeatingStreamSet(b, streamSet);
                } else {
                    if (LLVM_LIKELY(bn.isInternal())) {
                        const BufferPort & bp = mBufferGraph[output];
                        const auto handleName = makeBufferName(kernel, bp.Port);
                        buffer->setHandle(b.getScalarFieldPtr(handleName));
                    } else {
                        assert (isFromCurrentFunction(b, buffer->getHandle(), false));
                    }
                    if (nonLocal) {
                        Value * maxStrides = allocScale;
    //                    if (LLVM_UNLIKELY(bn.NumOfOverflowStrides)) {
    //                        maxStrides = b.CreateAdd(maxStrides, b.getSize(1));
    //                    }
                        const auto & R = bn.RelativeIORate;
                        assert (R.numerator() > 0);
                        Value * multiplier = b.CreateCeilUMulRational(maxStrides, R);

                        if (LLVM_UNLIKELY(bn.isReturned() || bn.crossesPhaseBoundary())) {
                            Value * expectedBufferSize = expectedSourceOutputSize;
                            if (bn.isReturned()) {
                                auto scaleFactor = getReturnedBufferScaleFactor(streamSet);
                                if (scaleFactor.numerator() == 0) {
                                    scaleFactor = R;
                                }
                                assert (expectedSourceOutputSize);
                                expectedBufferSize = b.CreateMulRational(expectedSourceOutputSize, scaleFactor);
                            }
                            multiplier = b.CreateUMax(multiplier, expectedBufferSize);
                        }

                        if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
                            if (reportCallback == nullptr) {
                                reportCallback = generateBufferExpansionFunctionForCurrentKernel(b, kernel); assert (reportCallback);
                                assert (getHandle());
                                sharedHandle = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
                            }
                            const BufferPort & bp = mBufferGraph[output];
                            buffer->allocateBuffer(b, multiplier, reportCallback, sharedHandle, b.getSize(bp.Port.Number));
                        } else {
                            buffer->allocateBuffer(b, multiplier, nullptr, nullptr, nullptr);
                        }

                        #ifdef PRINT_DEBUG_MESSAGES
                        const auto pe = in_edge(streamSet, mBufferGraph);
                        const auto producer = source(pe, mBufferGraph);
                        const BufferPort & rd = mBufferGraph[pe];
                        const auto prefix = makeBufferName(producer, rd.Port);
                        Value * start = buffer->getMallocAddress(b);
                        const auto byteSize = b.getTypeSize(dl, buffer->getType());
                        Value * length = b.CreateMulRational(buffer->getInternalCapacity(b), Rational{byteSize, b.getBitBlockWidth()});
                        Constant * ts = b.getSize(byteSize);
                        Value * end = b.CreateGEP(b.getInt8Ty(), b.CreatePointerCast(start, b.getInt8PtrTy()), length);
                        debugPrint(b, prefix + ".inital malloc range = [%" PRIx64 ",%" PRIx64 ") [typeSize=%" PRIu64 "]", start, end, ts);
                        #endif

                    }
                }
            }
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseOwnedBuffers(KernelBuilder & b) {
    loadInternalStreamSetHandles(b, true);
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isDeallocatable() && !bn.isReturned()) {
            StreamSetBuffer * const buffer = bn.Buffer;
            if (buffer->isDynamic()) {
                assert (isFromCurrentFunction(b, buffer->getHandle(), false));
                buffer->releaseBuffer(b);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief freePendingDeletions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freePendingDeletions(KernelBuilder & b, const size_t streamSet, Value * const consumed) {
    const BufferNode & bn = mBufferGraph[streamSet];
    if (bn.isThreadLocal() || bn.isUnowned() || bn.isInOutRedirect() || bn.isTruncated() || bn.isConstant() || bn.hasZeroElementsOrWidth()) {
        return;
    }
    StreamSetBuffer * const buffer = bn.Buffer;
    if (LLVM_LIKELY(buffer->isDynamic())) {
        assert (getTruncatedStreamSetSourceId(streamSet) == streamSet);
        assert (out_degree(streamSet, mConsumerGraph) > 0);
        buffer->freePendingDeletions(b, consumed);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateExternalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateExternalProducedItemCounts(KernelBuilder & b) {
    for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(output, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isReturned()) {
            const auto pe = in_edge(streamSet, mBufferGraph);
            const auto producer = source(pe, mBufferGraph);
            const BufferPort & br = mBufferGraph[pe];
            const auto prefix = makeBufferName(producer, br.Port);

            const BufferPort & bp = mBufferGraph[output];
            const auto k = bp.Port.Number;

            Value * itemCount = nullptr;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                itemCount = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            } else {
                itemCount = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
            }

            assert (isFromCurrentFunction(b, itemCount, false));
            assert (isFromCurrentFunction(b, mProducedOutputItemPtr[k], false));

            assert (mProducedOutputItemPtr[k]->getType()->isPointerTy());
            b.CreateAlignedStore(itemCount, mProducedOutputItemPtr[k], SizeTyABIAlignment);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addLocalDynamicBufferStructs
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addLocalDynamicBufferStructs(KernelBuilder & b) {
    if (ManagedBufferStructCount == 0) {
        return;
    }
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    for (size_t i = 0; i < ManagedBufferStructCount; ++i) {
        mTarget->addNonPersistentScalar(voidPtrTy, MANAGED_STREAMSET_LOCAL_VIRTUAL_BASE_ADDRESS + std::to_string(i));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateExternalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateLocalDynamicBufferStructsUntil(KernelBuilder & b, const size_t targetKernelId) {
    for (auto kernelId = mKernelId; kernelId < targetKernelId; ++kernelId) {
        for (const auto output : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            StreamSetBuffer * const buffer = bn.Buffer;
            if (LLVM_LIKELY(buffer->isDynamic())) {
                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    assert (mKernelId < consumer && consumer <= PipelineOutput);
                    if (consumer >= targetKernelId) {
                        Value * avail = readConsumedItemCount(b, streamSet);
                        Value * const ba = buffer->getBaseAddress(b);
                        Value * vba = buffer->getVirtualBasePtr(b, ba, avail);
                        vba = b.CreatePointerCast(vba, b.getVoidPtrTy());
                        assert (bn.ManagedStructId < ManagedBufferStructCount);
                        b.setScalarField(MANAGED_STREAMSET_LOCAL_VIRTUAL_BASE_ADDRESS + std::to_string(bn.ManagedStructId), vba);
                        break;
                    }
                }
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeOutputStreamSetBuffersBeforeSegmentInvocation
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::initializeOutputStreamSetBuffersBeforeSegmentInvocation(KernelBuilder & b) const {
    // We could immediately free the old buffer if one is stored in thread local data, it relies on the idea
    // that if we have N threads, we will invoke this kernel every N segments. This isn't true if we allow
    // threads to immediately restart upon reaching a jump that branches to the end of the pipeline.

    if (LLVM_UNLIKELY(out_degree(mKernelId, mBufferGraph) == 0)) {
        return false;
    }

    bool outputModifiesSegmentLength = false;
    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[output];
        if (LLVM_UNLIKELY(port.canModifySegmentLength())) {
            outputModifiesSegmentLength = true;
        }
        const BufferNode & bn = mBufferGraph[target(output, mBufferGraph)];
        if (LLVM_UNLIKELY(bn.isTruncated() || bn.isInOutRedirect())) {
            continue;
        }
    }

    return outputModifiesSegmentLength;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resetInternalBufferHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::resetInternalBufferHandles() {
    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }
    for (auto i = FirstStreamSet; i <= LastStreamSet; ++i) {
        const BufferNode & bn = mBufferGraph[i];
        if (LLVM_UNLIKELY(bn.isInternal())) {
            StreamSetBuffer * const buffer = bn.Buffer;
            buffer->setHandle(nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructStreamSetBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::constructStreamSetBuffers(KernelBuilder & /* b */) {
    mStreamSetInputBuffers.clear();
    BufferGraph::out_edge_iterator begin, end;
    std::tie(begin, end) = out_edges(PipelineInput, mBufferGraph);
    for (auto e = begin; e != end; ++e) {
        const BufferPort & rd = mBufferGraph[*e];
        if (LLVM_UNLIKELY(rd.Port.Reason != ReasonType::Explicit)) {
            end = e;
            break;
        }
    }

    const auto numOfInputStreams = std::distance(begin, end);
    mStreamSetInputBuffers.resize(numOfInputStreams);
    for (auto e = begin; e != end; ++e) {
        const BufferPort & rd = mBufferGraph[*e];
        const auto i = rd.Port.Number;
        const auto streamSet = target(*e, mBufferGraph);
        assert (mBufferGraph[streamSet].isExternal());
        const auto j = streamSet - FirstStreamSet;
        StreamSetBuffer * const buffer = mInternalBuffers[j].release();
        assert (buffer == mBufferGraph[streamSet].Buffer);
        mStreamSetInputBuffers[i].reset(buffer); assert (buffer);
    }

    mStreamSetOutputBuffers.clear();
    const auto numOfOutputStreams = in_degree(PipelineOutput, mBufferGraph);
    mStreamSetOutputBuffers.resize(numOfOutputStreams);
    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        const auto i = rd.Port.Number;
        const auto streamSet = source(e, mBufferGraph);
        assert (mBufferGraph[streamSet].isExternal());
        const auto j = streamSet - FirstStreamSet;
        StreamSetBuffer * const buffer = mInternalBuffers[j].release();
        assert (buffer == mBufferGraph[streamSet].Buffer);
        mStreamSetOutputBuffers[i].reset(buffer); assert (buffer);
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readAvailableItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readAvailableItemCounts(KernelBuilder & b) {
    mKernelIsClosed.reset(FirstKernel, LastKernel);

    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[input];
        const auto streamSet = source(input, mBufferGraph);
        if (mLocallyAvailableItems[streamSet] == nullptr) {
            mLocallyAvailableItems[streamSet] = readAvailableItemCount(b, streamSet);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readAvailableItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readAvailableItemCount(KernelBuilder & b, const size_t streamSet) {
    const BufferNode & bn = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(bn.isConstant())) {
        return ConstantInt::getAllOnesValue(b.getSizeTy());
    }
    if (LLVM_UNLIKELY(bn.isTruncated())) {
        return readAvailableItemCount(b, getTruncatedStreamSetSourceId(streamSet));
    }
    Value * produced = nullptr;
    const auto f = in_edge(streamSet, mBufferGraph);
    const auto producer = source(f, mBufferGraph);
    if (LLVM_UNLIKELY(producer == PipelineInput)) {
        const BufferPort & outputPort = mBufferGraph[f];
        assert (outputPort.Port.Type == PortType::Output);
        assert (bn.isExternal());
        // the output port of the pipeline input is an input streamset of the pipeline kernel.
        produced = getAvailableInputItems(outputPort.Port.Number);
        writeTransitoryConsumedItemCount(b, streamSet, produced);
    } else if (LLVM_UNLIKELY(bn.ProducedPhaseId < mCurrentPipelinePhase)) {
        const BufferPort & br = mBufferGraph[f];
        const auto prefix = makeBufferName(producer, br.Port);
        #ifdef PHASES_RUN_TO_COMPLETION
        produced = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
        #else
        if (LLVM_UNLIKELY(br.isDeferred())) {
            produced = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
        } else {
            produced = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
        }
        #endif
    } else {
        assert (bn.ProducedPhaseId == mCurrentPipelinePhase);
        produced = mLocallyAvailableItems[streamSet]; assert (produced);
    }
    return produced;
}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readProcessedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readProcessedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto inputPort = br.Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);

        if (LLVM_UNLIKELY(br.isRelative())) {

            const auto ref = getReference(br.Port);
            assert (ref.Type == PortType::Input);
            Value * itemCount = mInitiallyProcessedItemCount[ref];
            itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
            mInitiallyProcessedItemCount[inputPort] = itemCount;
            if (br.isDeferred()) {
                Value * itemCount = mInitiallyProcessedDeferredItemCount[ref];
                itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
                mInitiallyProcessedDeferredItemCount[inputPort] = itemCount;
            }

        } else {

            const auto & suffix = (mAllowDataParallelExecution & !mKernelIsInternallySynchronized) ?
                STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

            auto prodRef = b.getScalarFieldPtr(prefix + suffix);
            mProcessedItemCountPtr[inputPort] = prodRef.first;
            Value * itemCount = b.CreateAlignedLoad(prodRef.second, prodRef.first, SizeTyABIAlignment);
            mInitiallyProcessedItemCount[inputPort] = itemCount;
            if (br.isDeferred()) {
                assert (!mAllowDataParallelExecution || mKernelIsInternallySynchronized);
                auto defRef = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                mProcessedDeferredItemCountPtr[inputPort] = defRef.first;
                itemCount = b.CreateAlignedLoad(defRef.second, defRef.first, SizeTyABIAlignment);
                mInitiallyProcessedDeferredItemCount[inputPort] = itemCount;
            }

        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readProducedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {

        const BufferPort & br = mBufferGraph[e];
        const auto outputPort = br.Port;
        const auto streamSet = target(e, mBufferGraph);
        const auto prefix = makeBufferName(mKernelId, outputPort);

        if (LLVM_UNLIKELY(br.isRelative())) {

            const auto ref = getReference(br.Port);

            if (ref.Type == PortType::Input) {
                Value * itemCount = mInitiallyProcessedItemCount[ref];
                itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
                mInitiallyProducedItemCount[streamSet] = itemCount;
                if (br.isDeferred()) {
                    Value * itemCount = mInitiallyProcessedDeferredItemCount[ref];
                    itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
                    mInitiallyProducedDeferredItemCount[streamSet] = itemCount;
                }
            } else {
                const auto refStreamSet = getOutputBufferVertex(ref);
                Value * itemCount = mInitiallyProducedItemCount[refStreamSet];
                itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
                mInitiallyProducedItemCount[streamSet] = itemCount;
                if (br.isDeferred()) {
                    Value * itemCount = mInitiallyProducedDeferredItemCount[refStreamSet];
                    itemCount = b.CreateMulRational(itemCount, br.getRate().getRate());
                    mInitiallyProducedDeferredItemCount[streamSet] = itemCount;
                }
            }

        } else {

            const auto & suffix = (mAllowDataParallelExecution & !mKernelIsInternallySynchronized) ?
                STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

            auto prodRef = b.getScalarFieldPtr(prefix + suffix);
            Value * itemCountPtr = prodRef.first;
            Value * itemCount = b.CreateAlignedLoad(prodRef.second, itemCountPtr, SizeTyABIAlignment);

            mProducedItemCountPtr[outputPort] = itemCountPtr;
            mInitiallyProducedItemCount[streamSet] = itemCount;

            if (br.isDeferred()) {
                assert (!mAllowDataParallelExecution || mKernelIsInternallySynchronized);
                auto defRef = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                Value * itemCountPtr = defRef.first;
                Value * itemCount = b.CreateAlignedLoad(defRef.second, itemCountPtr, SizeTyABIAlignment);
                mProducedDeferredItemCountPtr[outputPort] = itemCountPtr;
                mInitiallyProducedDeferredItemCount[streamSet] = itemCount;
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeUpdatedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeUpdatedItemCounts(KernelBuilder & b) {

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (LLVM_UNLIKELY(br.isRelative())) {
           continue;
        }
        const StreamSetPort inputPort = br.Port;
        Value * ptr = nullptr;
        if (mAllowDataParallelExecution & !mKernelIsInternallySynchronized) {
            const auto prefix = makeBufferName(mKernelId, inputPort);
            ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
        } else {
            ptr = mProcessedItemCountPtr[inputPort];
        }
        b.CreateAlignedStore(mUpdatedProcessedPhi[inputPort], ptr, SizeTyABIAlignment);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, inputPort);
        debugPrint(b, " @ writing " + prefix + "_processed = %" PRIu64, mUpdatedProcessedPhi[inputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mAllowDataParallelExecution || mKernelIsInternallySynchronized);
            b.CreateAlignedStore(mUpdatedProcessedDeferredPhi[inputPort], mProcessedDeferredItemCountPtr[inputPort], SizeTyABIAlignment);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, " @ writing " + prefix + "_processed(deferred) = %" PRIu64, mUpdatedProcessedDeferredPhi[inputPort]);
            #endif
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (LLVM_UNLIKELY(br.isRelative())) {
           continue;
        }
        const StreamSetPort outputPort = br.Port;
        Value * ptr = nullptr;
        if (mAllowDataParallelExecution & !mKernelIsInternallySynchronized) {
            const auto prefix = makeBufferName(mKernelId, outputPort);
            ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
        } else {
            ptr = mProducedItemCountPtr[outputPort];
        }
        b.CreateAlignedStore(mUpdatedProducedPhi[outputPort], ptr, SizeTyABIAlignment);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, outputPort);
        debugPrint(b, " @ writing " + prefix + "_produced = %" PRIu64, mUpdatedProducedPhi[outputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mAllowDataParallelExecution || mKernelIsInternallySynchronized);
            b.CreateAlignedStore(mUpdatedProducedDeferredPhi[outputPort], mProducedDeferredItemCountPtr[outputPort], SizeTyABIAlignment);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, " @ writing " + prefix + "_produced(deferred) = %" PRIu64, mUpdatedProducedDeferredPhi[outputPort]);
            #endif
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordFinalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordFinalProducedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto outputPort = br.Port;
        Value * const fullyProduced = mFullyProducedItemCount[outputPort]; assert (fullyProduced);

        #ifdef PRINT_DEBUG_MESSAGES
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        const auto prefix = makeBufferName(mKernelId, outputPort);
        out << " * -> " << prefix << "_avail = %" PRIu64;
        debugPrint(b, out.str(), fullyProduced);
        #endif

        const auto streamSet = target(e, mBufferGraph);
        mLocallyAvailableItems[streamSet] = fullyProduced;

        writeTransitoryConsumedItemCount(b, streamSet, fullyProduced);
        mInitialConsumedItemCount[streamSet] = nullptr;

        // update any external output port(s)
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isExternal())) {
            for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const BufferPort & external = mBufferGraph[f];
                Value * const ptr = getProducedOutputItemsPtr(external.Port.Number);
                b.CreateAlignedStore(fullyProduced, ptr, SizeTyABIAlignment);
            }
        }

        #ifdef PRINT_DEBUG_MESSAGES
        Value * const producedDelta = b.CreateSub(fullyProduced, mInitiallyProducedItemCount[streamSet]);
        debugPrint(b, prefix + "_producedΔ = %" PRIu64, producedDelta);
        #endif

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readReturnedOutputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readReturnedOutputVirtualBaseAddresses(KernelBuilder & b) const {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        assert (rd.Port.Type == PortType::Output);
        const StreamSetPort port(PortType::Output, rd.Port.Number);
        if (rd.isManaged()) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            assert (bn.isNonThreadLocal());
            Value * const ptr = mReturnedOutputVirtualBaseAddressPtr[port]; assert (ptr);
            StreamSetBuffer * const buffer = bn.Buffer;
            assert (isa<ExternalBuffer>(buffer));
            Value * vba = b.CreateAlignedLoad(buffer->getPointerType(), ptr, PtrTyABIAlignment);
            buffer->setBaseAddress(b, vba);
            buffer->setCapacity(b, mProducedItemCount[port]);
            const auto handleName = makeBufferName(mKernelId, port);
            #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
            debugPrint(b, "%s_updatedVirtualBaseAddress = 0x%" PRIx64, b.GetString(handleName), buffer->getBaseAddress(b));
            #endif
            b.setScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS, vba);
        } else {
            assert (mReturnedOutputVirtualBaseAddressPtr[port] == nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffers(KernelBuilder & b, const size_t kernelId) const {
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        // owned or external buffers do not have a mutable vba
        if (LLVM_LIKELY(bn.isOwned() || bn.isExternal() || bn.hasZeroElementsOrWidth())) {
            continue;
        }
        assert (bn.isNonThreadLocal());
        const BufferPort & rd = mBufferGraph[e];
        const auto handleName = makeBufferName(kernelId, rd.Port);
        Value * const vba = b.getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
        StreamSetBuffer * const buffer = bn.Buffer;
        buffer->setBaseAddress(b, vba);
//        if (CheckAssertions()) {
//            b.CreateAssert(vba, "%s.%s last good virtual base addresss cannot be null",
//                            mCurrentKernelName, b.GetString(rd.Binding.get().getName()));
//        }
        #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
        debugPrint(b, "%s_loadPriorVirtualBaseAddress = 0x%" PRIx64, b.GetString(handleName), buffer->getBaseAddress(b));
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getVirtualBaseAddress
 *
 * Returns the address of the "zeroth" item of the (logically-unbounded) stream set.
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getVirtualBaseAddress(KernelBuilder & b,
                                                const BufferPort & rateData,
                                                const BufferNode & bufferNode,
                                                Value * position,
                                                const bool prefetch,
                                                const bool write) const {


    const StreamSetBuffer * buffer = bufferNode.Buffer;
    assert ("buffer cannot be null!" && buffer);

    Value * addr = nullptr;
    if (buffer->isDynamic() && rateData.Port.Type == PortType::Input && bufferNode.ProducedPhaseId == mCurrentPipelinePhase) {
        addr = b.getScalarField(MANAGED_STREAMSET_LOCAL_VIRTUAL_BASE_ADDRESS + std::to_string(bufferNode.ManagedStructId));
        addr = b.CreatePointerCast(addr, buffer->getPointerType());
    } else {
        assert (isFromCurrentFunction(b, buffer->getHandle(), false));
        assert (position);
        Value * const baseAddress = buffer->getBaseAddress(b); assert (baseAddress);
        if (bufferNode.isUnowned() || bufferNode.hasZeroElementsOrWidth()) {
            addr = baseAddress;
        } else {
            addr = buffer->getVirtualBasePtr(b, baseAddress, position);
        }
    }
    #if 0
    if (prefetch) {
        ExternalBuffer tmp(0, b, buffer->getBaseType(), buffer->getAddressSpace());
        Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
        Value * const blockIndex = b.CreateLShr(position, LOG_2_BLOCK_WIDTH);
        Value * const prefetchAddr = tmp.getStreamBlockPtr(b, addr, b.getSize(0), blockIndex);
        prefetchAtLeastThreeCacheLinesFrom(b, prefetchAddr, write);
    }
    #endif
    return addr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief prefetchThreeCacheLinesFrom
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::prefetchAtLeastThreeCacheLinesFrom(KernelBuilder & b, Value * const addr, const bool write) const {
#if 0
    Module * const m = b.getModule();
    Function * const prefetchFunc = Intrinsic::getDeclaration(m, Intrinsic::prefetch);

    DataLayout dl(m);
    Type * const elemTy = addr->getType()->getPointerElementType();
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
    const auto typeSize = dl.getTypeAllocSize(elemTy);
    #else
    const auto typeSize = dl.getTypeAllocSize(elemTy).getFixedSize();
    #endif
    assert (typeSize > 0);

    IntegerType * const int32Ty = b.getInt32Ty();
    FixedArray<Value *, 4> args;
    args[1] = ConstantInt::get(int32Ty, write ? 1 : 0); // write flag
    args[2] = ConstantInt::get(int32Ty, 3); // locality
    args[3] = ConstantInt::get(int32Ty, 1); // cache type?

    const auto cl = b.getCacheAlignment();
    const auto toFetch = round_up_to<unsigned>(cl * 3, typeSize);
    Value * const baseAddr = b.CreatePointerCast(addr, b.getInt8PtrTy());
    for (unsigned i = 0; i < toFetch; i += cl) {
        args[0] = b.CreateGEP0(baseAddr, b.getSize(i));
        b.CreateCall(prefetchFunc->getFunctionType(), prefetchFunc, args);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getInputVirtualBaseAddresses(KernelBuilder & b, Vec<Value *> & baseAddresses) const {
    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & inputPort = mBufferGraph[input];
        PHINode * processed = nullptr;
        if (mCurrentProcessedDeferredItemCountPhi[inputPort.Port]) {
            processed = mCurrentProcessedDeferredItemCountPhi[inputPort.Port];
        } else {
            processed = mCurrentProcessedItemCountPhi[inputPort.Port];
        }
        const auto streamSet = source(input, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];

        if (LLVM_UNLIKELY(bn.isUnowned() && bn.isInternal())) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            assert (producer < mKernelId);
            const BufferPort & outputPort = mBufferGraph[output];
            const auto handleName = makeBufferName(producer, outputPort.Port);
            Value * const vba = b.getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
            bn.Buffer->setBaseAddress(b, vba);
        }
        Value * addr = getVirtualBaseAddress(b, inputPort, bn, processed, bn.isNonThreadLocal(), false);
        baseAddresses[inputPort.Port.Number] = addr;
    }

}


} // end of kernel namespace
