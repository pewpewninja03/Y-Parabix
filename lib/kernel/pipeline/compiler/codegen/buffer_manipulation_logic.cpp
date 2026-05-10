#include "../pipeline_compiler.hpp"

using namespace IDISA;

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateLocalZeroExtensionSpace
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::allocateLocalZeroExtensionSpace(KernelBuilder & b,
                                                          Vec<Value *> & inputBufferCapacity,
                                                          BasicBlock * const insertBefore) const {
    #ifndef DISABLE_ZERO_EXTEND
    const size_t blockWidth = b.getBitBlockWidth();

    Value * const numOfStrides = b.CreateUMax(mNumOfLinearStrides, b.getSize(1));

    auto & dl = b.getModule()->getDataLayout();

    IntegerType * const sizeTy = b.getSizeTy();

    Value * requiredSpace = nullptr;

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {

        const BufferPort & br = mBufferGraph[e];

        Value * zeroExtended = mIsInputZeroExtended[br.Port];
        assert (br.isZeroExtended() == (zeroExtended != nullptr));

        if (zeroExtended) {

            assert (HasZeroExtendedStream);

            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];

            const auto ts = b.getTypeSize(dl, bn.Buffer->getType());

            Rational scaleFactor{ts * br.Maximum.numerator(), blockWidth * blockWidth * br.Maximum.denominator()};

            Value * ns = numOfStrides;
            if (br.RequiredOverflowSpace) {
                const auto k = ceiling(Rational{br.RequiredOverflowSpace, blockWidth});
                ns = b.CreateAdd(ns, b.getSize(k));
            }
            zeroExtended = b.CreateZExt(zeroExtended, sizeTy);
            Value * const requiredBytes = b.CreateCeilUMulRational(b.CreateMul(ns, zeroExtended), scaleFactor);
            requiredSpace = b.CreateUMax(requiredBytes, requiredSpace);
        }
    }
    assert (requiredSpace);    
    requiredSpace = b.CreateRoundUpRational(requiredSpace, b.getPageSize());

    const auto prefix = makeKernelName(mKernelId);
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const expandZeroExtension =
        b.CreateBasicBlock(prefix + "_expandZeroExtensionBuffer", insertBefore);
    BasicBlock * const hasSufficientZeroExtendSpace =
        b.CreateBasicBlock(prefix + "_hasSufficientZeroExtendSpace", insertBefore);

    auto zeSpaceRef = b.getScalarFieldPtr(ZERO_EXTENDED_SPACE);
    assert (zeSpaceRef.second == sizeTy);
    Value * const currentSpace = b.CreateAlignedLoad(sizeTy, zeSpaceRef.first, SizeTyABIAlignment);

    auto zeBufferRef = b.getScalarFieldPtr(ZERO_EXTENDED_BUFFER);
    Value * const currentBuffer = b.CreateAlignedLoad(zeBufferRef.second, zeBufferRef.first, PtrTyABIAlignment);

    Value * const largeEnough = b.CreateICmpUGE(currentSpace, requiredSpace);
    b.CreateLikelyCondBr(largeEnough, hasSufficientZeroExtendSpace, expandZeroExtension);

    b.SetInsertPoint(expandZeroExtension);
    assert (b.getCacheAlignment() >= (b.getBitBlockWidth() / 8));
    b.CreateFree(currentBuffer);
    Value * const newBuffer = b.CreatePageAlignedMalloc(requiredSpace);
    b.CreateMemZero(newBuffer, requiredSpace, b.getCacheAlignment());
    b.CreateAlignedStore(requiredSpace, zeSpaceRef.first, SizeTyABIAlignment);
    b.CreateAlignedStore(newBuffer, zeBufferRef.first, PtrTyABIAlignment);
    b.CreateBr(hasSufficientZeroExtendSpace);

    b.SetInsertPoint(hasSufficientZeroExtendSpace);
    PHINode * const zeroBufferPhi = b.CreatePHI(b.getVoidPtrTy(), 2);
    zeroBufferPhi->addIncoming(currentBuffer, entry);
    zeroBufferPhi->addIncoming(newBuffer, expandZeroExtension);
    if (LLVM_UNLIKELY(mCheckStreamSets)) {
        PHINode * const allocedSpacePhi = b.CreatePHI(sizeTy, 2);
        allocedSpacePhi->addIncoming(currentSpace, entry);
        allocedSpacePhi->addIncoming(requiredSpace, expandZeroExtension);
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            Value * const zeroExtended = mIsInputZeroExtended[br.Port];
            if (zeroExtended) {
                const auto streamSet = source(e, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                const auto ts = b.getTypeSize(dl, bn.Buffer->getType());
                Rational scaleFactor{blockWidth * blockWidth * br.Maximum.denominator(), ts * br.Maximum.numerator()};
                Value * const ic = b.CreateMulRational(allocedSpacePhi, scaleFactor);
                auto & ze = inputBufferCapacity[br.Port.Number];
                ze = b.CreateSelect(zeroExtended, b.CreateAdd(mCurrentProcessedItemCountPhi[br.Port], ic), ze);
            }
        }
    }
    return zeroBufferPhi;
    #else
    return nullptr;
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getZeroExtendedInputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getZeroExtendedInputVirtualBaseAddresses(KernelBuilder & b,
                                                                const Vec<Value *> & baseAddresses,
                                                                Value * const zeroExtensionSpace,
                                                                Vec<Value *> & zeroExtendedVirtualBaseAddress) const {

    // TODO: if we reserve a "zero extension" block in the thread local memory, we could trade this logic for a memclear

    #ifndef DISABLE_ZERO_EXTEND
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & rt = mBufferGraph[e];
        assert (rt.Port.Type == PortType::Input);
        Value * const zeroExtended = mIsInputZeroExtended[rt.Port];
        if (zeroExtended) {
            PHINode * processed = nullptr;
            if (mCurrentProcessedDeferredItemCountPhi[rt.Port]) {
                processed = mCurrentProcessedDeferredItemCountPhi[rt.Port];
            } else {
                processed = mCurrentProcessedItemCountPhi[rt.Port];
            }
            const BufferNode & bn = mBufferGraph[source(e, mBufferGraph)];
            const Binding & binding = rt.Binding;
            const StreamSetBuffer * buffer = bn.Buffer;

            Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
            Constant * const ZERO = b.getSize(0);
            PointerType * const bufferType = buffer->getPointerType();
            Value * const blockIndex = b.CreateLShr(processed, LOG_2_BLOCK_WIDTH);

            // allocateLocalZeroExtensionSpace guarantees this will be large enough to satisfy the kernel
            ExternalBuffer tmp(0, b, binding.getType(), buffer->getAddressSpace());
            Value * zeroExtension = b.CreatePointerCast(zeroExtensionSpace, bufferType);
            Value * addr = tmp.getStreamBlockPtr(b, zeroExtension, ZERO, b.CreateNeg(blockIndex));
            addr = b.CreatePointerCast(addr, bufferType);
            const auto i = rt.Port.Number;
            assert (addr->getType() == baseAddresses[i]->getType());

            addr = b.CreateSelect(zeroExtended, addr, baseAddresses[i], "zeroExtendAddr");
            zeroExtendedVirtualBaseAddress[i] = addr;
        }
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addZeroInputStructProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addZeroInputStructProperties(KernelBuilder & b) const {
    const auto m = num_vertices(mZeroInputGraph);
    if (m > LastKernel) {
        FixedArray<Type *, 2> fields;
        fields[0] = b.getInt8PtrTy();
        fields[1] = b.getSizeTy();
        StructType * const truncTy = StructType::get(b.getContext(), fields);
        const auto n = m - LastKernel;
        ArrayType * const arTy = ArrayType::get(truncTy, n);
        #ifndef NDEBUG

        #endif
        mTarget->addThreadLocalScalar(arTy, ZERO_INPUT_BUFFER_STRUCT, getCacheLineGroupId(PipelineOutput));
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief zeroInputAfterFinalItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::zeroInputAfterFinalItemCount(KernelBuilder & b,
                                                    const Vec<Value *> & accessibleItems,
                                                    Vec<Value *> & inputBufferCapacity,
                                                    Vec<Value *> & inputBaseAddresses) {

    #ifndef DISABLE_INPUT_ZEROING
    const auto n = out_degree(mKernelId - FirstKernel, mZeroInputGraph);
    if (n == 0) {
        return;
    }
    assert (num_vertices(mZeroInputGraph) > LastKernel);

    IntegerType * const sizeTy = b.getSizeTy();

    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);

    ZeroInputGraph::out_edge_iterator ei_begin, ei_end;
    std::tie(ei_begin, ei_end) = out_edges(mKernelId - FirstKernel, mZeroInputGraph);

    Value * base;
    Type * traceArTy;

    std::tie(base, traceArTy) = b.getScalarFieldPtr(ZERO_INPUT_BUFFER_STRUCT);

    FixedArray<Value *, 2> indices;

    ConstantInt * i32_ZERO = b.getInt32(0);
    ConstantInt * i32_ONE = b.getInt32(1);

    indices[0] = i32_ZERO;

    const auto blockWidth = b.getBitBlockWidth();

    for (auto p : make_iterator_range(out_edges(mKernelId - FirstKernel, mZeroInputGraph))) {
        const auto portNum = mZeroInputGraph[p];

        const auto e = getInput(mKernelId, StreamSetPort{PortType::Input, portNum});

        const BufferPort & port = mBufferGraph[e];

        const auto streamSet = source(e, mBufferGraph);

        const BufferNode & bn = mBufferGraph[streamSet];
        const StreamSetBuffer * const buffer = bn.Buffer;

        const auto inputPort = port.Port;
        assert (inputPort.Type == PortType::Input);
        const Binding & input = port.Binding;
        const auto unaligned = input.hasAttribute(AttrId::AllowsUnalignedAccess);

        const auto z = target(p, mZeroInputGraph);
        assert (z >= LastKernel);
        assert (z < LastKernel + traceArTy->getArrayNumElements());

        indices[1] = b.getInt32(z - LastKernel);

        StructType * const truncTy = cast<StructType>(traceArTy->getArrayElementType());

        const auto prefix = makeBufferName(mKernelId, inputPort);

        const auto itemWidth = getItemWidth(buffer->getBaseType());
        Constant * const ITEM_WIDTH = b.getSize(itemWidth);

        PointerType * const bufferType = buffer->getPointerType();

        BasicBlock * const maskedInput = b.CreateBasicBlock(prefix + "_maskInput", mKernelCheckOutputSpace);
        BasicBlock * const selectedInput = b.CreateBasicBlock(prefix + "_selectInput", mKernelCheckOutputSpace);

        BasicBlock * const entryBlock = b.GetInsertBlock();

        Value * const availableItems = mLocallyAvailableItems[streamSet];
        const auto alwaysTruncate = bn.isUnowned() || bn.isTruncated() || bn.isConstant();

        if (LLVM_UNLIKELY(alwaysTruncate)) {
            b.CreateBr(maskedInput);
        } else {

            Value * const selectedItems = b.CreateAdd(mCurrentProcessedItemCountPhi[inputPort], accessibleItems[inputPort.Number]);
            const auto output = in_edge(streamSet, mBufferGraph);
            const BufferPort & out = mBufferGraph[output];

            Value * cond = nullptr;
            if (port.RequiredOverflowSpace == 0 && out.RequiredOverflowSpace == 0) {
                cond = b.CreateICmpNE(selectedItems, availableItems);
            } else {
                Value * unprocessedItems = selectedItems;
                if (port.RequiredOverflowSpace) {
                    unprocessedItems = b.CreateAdd(selectedItems, b.getSize(port.RequiredOverflowSpace));
                }
                Value * const tooMany = b.CreateICmpULT(unprocessedItems, availableItems);
                Value * totalItems = availableItems;
                if (out.RequiredOverflowSpace) {
                    totalItems = b.CreateAdd(totalItems, b.getSize(out.RequiredOverflowSpace));
                }
                Value * const tooFew = b.CreateICmpUGT(selectedItems, totalItems);
                cond = b.CreateOr(tooMany, tooFew);
            }
            b.CreateUnlikelyCondBr(cond, maskedInput, selectedInput);
        }

        b.SetInsertPoint(maskedInput);

        // if this is a deferred fixed rate stream, we cannot be sure how many
        // blocks will have to be provided to the kernel in order to mask out
        // the truncated input stream.


        // Generate a name to describe this masking function.
        SmallVector<char, 32> tmp;
        raw_svector_ostream name(tmp);

        name << "__maskInput" << itemWidth;

        if (unaligned) {
            name << "U";
        }

        Module * const m = b.getModule();

        PointerType * const int8PtrTy = b.getInt8PtrTy();

        Function * maskInput = m->getFunction(name.str());

        if (maskInput == nullptr) {

            const auto log2BlockWidth = floor_log2(blockWidth);
            Constant * const BLOCK_MASK = b.getSize(blockWidth - 1);
            Constant * const LOG_2_BLOCK_WIDTH = b.getSize(log2BlockWidth);

            Type * retTy = nullptr;
            if (LLVM_UNLIKELY(mCheckStreamSets)) {
                FixedArray<Type *, 2> fields;
                fields[0] = int8PtrTy;
                fields[1] = sizeTy;
                retTy = StructType::get(b.getContext(), fields);
            } else {
                retTy = int8PtrTy;
            }

            const auto ip = b.saveIP();

            FixedArray<Type *, 7> params;
            params[0] = int8PtrTy; // input buffer
            params[1] = sizeTy; // items per stride
            params[2] = sizeTy; // start
            params[3] = sizeTy; // end
            params[4] = sizeTy; // overflow
            params[5] = sizeTy; // numOfStreams
            params[6] = truncTy->getPointerTo(); // masked buffer storage ptr

            LLVMContext & C = m->getContext();

            FunctionType * const funcTy = FunctionType::get(retTy, params, false);
            maskInput = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
            if (LLVM_UNLIKELY(CheckAssertions())) {
                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
                maskInput->setHasUWTable();
                #else
                maskInput->setUWTableKind(UWTableKind::Default);
                #endif
            }

            BasicBlock * const entry = BasicBlock::Create(C, "entry", maskInput);
            BasicBlock * const allocateNewBuffer = BasicBlock::Create(C, "allocateNewBuffer", maskInput);
            BasicBlock * const allocateNewBufferExit = BasicBlock::Create(C, "allocateNewBufferExit", maskInput);
            BasicBlock * const hasDataToCopy = BasicBlock::Create(C, "hasDataToCopy", maskInput);
            BasicBlock * const maskedInputLoop = BasicBlock::Create(C, "maskInputLoop", maskInput);
            BasicBlock * const maskedInputExit = BasicBlock::Create(C, "maskInputExit", maskInput);

            b.SetInsertPoint(entry);

            auto arg = maskInput->arg_begin();
            auto nextArg = [&]() {
                assert (arg != maskInput->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            auto & DL = m->getDataLayout();
            Type * const intPtrTy = DL.getIntPtrType(int8PtrTy);

            Value * const inputBuffer = nextArg();
            inputBuffer->setName("inputBuffer");
            Value * const itemsPerSegment = nextArg();
            itemsPerSegment->setName("itemsPerSegment");
            Value * const start = nextArg();
            start->setName("start");
            Value * const end = nextArg();
            end->setName("end");
            Value * const overflow = nextArg();
            overflow->setName("overflow");
            Value * const numOfStreams = nextArg();
            numOfStreams->setName("numOfStreams");
            Value * const bufferStorage = nextArg();
            bufferStorage->setName("bufferStorage");
            assert (arg == maskInput->arg_end());

            Type * const singleElementStreamSetTy = ArrayType::get(FixedVectorType::get(IntegerType::get(C, itemWidth), 0U), 1U);
            ExternalBuffer tmp(0, b, singleElementStreamSetTy, 0);

            PointerType * const bufferPtrTy = tmp.getPointerType();

            Value * const inputAddress = b.CreatePointerCast(inputBuffer, bufferPtrTy);
            Value * const initial = b.CreateMul(b.CreateLShr(start, LOG_2_BLOCK_WIDTH), numOfStreams);
            Value * const initialPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, initial);
            Value * const initialPtrInt = b.CreatePtrToInt(initialPtr, intPtrTy);
            Value * const adjEnd = b.CreateRoundUp(b.CreateAdd(end, overflow), itemsPerSegment);
            Value * const last = b.CreateMul(b.CreateCeilUDivRational(adjEnd, blockWidth), numOfStreams);

            Value * const lastPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, last);
            Value * const lastPtrInt = b.CreatePtrToInt(lastPtr, intPtrTy);

            Value * const mallocBytes = b.CreateSub(lastPtrInt, initialPtrInt);

            const auto blockSize = b.getBitBlockWidth() / 8U;
            const auto alignment = unaligned ? 1U : blockSize;

            FixedArray<Value *, 2> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_ZERO;
            Value * const mallocedPtr = b.CreateGEP(truncTy, bufferStorage, offset);
            Value * const existingBuffer = b.CreateAlignedLoad(int8PtrTy, mallocedPtr, PtrTyABIAlignment);
            offset[1] = i32_ONE;
            Value * const mallocedSizePtr = b.CreateGEP(truncTy, bufferStorage, offset);

            Value * const existingSize = b.CreateAlignedLoad(sizeTy, mallocedSizePtr, SizeTyABIAlignment);
            Value * const needsRealloc = b.CreateICmpUGT(mallocBytes, existingSize);
            b.CreateCondBr(needsRealloc, allocateNewBuffer, allocateNewBufferExit);

            b.SetInsertPoint(allocateNewBuffer);
            Value * const allocedBytes = b.CreateRoundUpRational(mallocBytes, getPageSize());
            b.CreateFree(existingBuffer);
            Value * const newBuffer = b.CreateAlignedMalloc(allocedBytes, blockSize);
            b.CreateAlignedStore(newBuffer, mallocedPtr, PtrTyABIAlignment);
            b.CreateAlignedStore(allocedBytes, mallocedSizePtr, SizeTyABIAlignment);
            b.CreateBr(allocateNewBufferExit);

            b.SetInsertPoint(allocateNewBufferExit);
            PHINode * const maskedBuffer = b.CreatePHI(int8PtrTy, 2);
            maskedBuffer->addIncoming(existingBuffer, entry);
            maskedBuffer->addIncoming(newBuffer, allocateNewBuffer);
            PHINode * const maskedBufferSize = b.CreatePHI(sizeTy, 2);
            maskedBufferSize->addIncoming(existingSize, entry);
            maskedBufferSize->addIncoming(allocedBytes, allocateNewBuffer);

            Value * const mallocedAddress = b.CreatePointerCast(maskedBuffer, bufferPtrTy);
            Value * const outputVBA = tmp.getStreamBlockPtr(b, mallocedAddress, sz_ZERO, b.CreateNeg(initial));
            Value * const maskedAddress = b.CreatePointerCast(outputVBA, bufferPtrTy);
            assert (maskedAddress->getType() == inputAddress->getType());
            b.CreateCondBr(b.CreateIsNull(mallocBytes), maskedInputExit, hasDataToCopy);

            b.SetInsertPoint(hasDataToCopy);

            Value * const total = b.CreateLShr(end, LOG_2_BLOCK_WIDTH);
            Value * const fullCopyEnd = b.CreateMul(total, numOfStreams);
            Value * const fullCopyEndPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, fullCopyEnd);
            Value * const fullCopyEndPtrInt = b.CreatePtrToInt(fullCopyEndPtr, intPtrTy);
            Value * const fullBytesToCopy = b.CreateSub(fullCopyEndPtrInt, initialPtrInt);
            b.CreateMemCpy(maskedBuffer, initialPtr, fullBytesToCopy, alignment);
            Value * const remainingBytes = b.CreateSub(mallocBytes, fullBytesToCopy);
            b.CreateMemZero(b.CreateGEP(b.getInt8Ty(), maskedBuffer, fullBytesToCopy), remainingBytes, blockSize);

            Value * packIndex = nullptr;
            Value * maskOffset = b.CreateAnd(end, BLOCK_MASK);
            Value * initialMaskOffset = maskOffset;
            if (itemWidth > 1) {
                Value * const position = b.CreateMul(maskOffset, ITEM_WIDTH);
                packIndex = b.CreateLShr(position, LOG_2_BLOCK_WIDTH);
                maskOffset = b.CreateAnd(position, BLOCK_MASK);
            }

            Value * const mask = b.CreateNot(b.bitblock_mask_from(maskOffset));
            BasicBlock * const loopEntryBlock = b.GetInsertBlock();
            b.CreateCondBr(b.CreateICmpNE(initialMaskOffset, sz_ZERO), maskedInputLoop, maskedInputExit);

            b.SetInsertPoint(maskedInputLoop);
            PHINode * const streamIndex = b.CreatePHI(sizeTy, 2);
            streamIndex->addIncoming(sz_ZERO, loopEntryBlock);

            Value * inputPtr = nullptr;
            Value * outputPtr = nullptr;

            if (itemWidth == 1) {
                inputPtr = tmp.getStreamBlockPtr(b, inputAddress, streamIndex, fullCopyEnd);
                outputPtr = tmp.getStreamBlockPtr(b, maskedAddress, streamIndex, fullCopyEnd);
            } else {

                BasicBlock * const packCopyLoop = BasicBlock::Create(C, "packCopyLoop", maskInput, maskedInputExit);
                BasicBlock * const packCopyExit = BasicBlock::Create(C, "packCopyExit", maskInput, maskedInputExit);

                b.CreateCondBr(b.CreateICmpEQ(packIndex, sz_ZERO), packCopyExit, packCopyLoop);

                b.SetInsertPoint(packCopyLoop);
                PHINode * const packIndexPhi = b.CreatePHI(sizeTy, 2);
                packIndexPhi->addIncoming(sz_ZERO, maskedInputLoop);

                Value * const packCopyInputPtr = tmp.getStreamPackPtr(b, inputAddress, streamIndex, fullCopyEnd, packIndexPhi);
                Value * const packCopyOutputPtr = tmp.getStreamPackPtr(b, maskedAddress, streamIndex, fullCopyEnd, packIndexPhi);
                Value * const val = b.CreateAlignedLoad(b.getBitBlockType(), packCopyInputPtr, alignment);
                b.CreateAlignedStore(val, packCopyOutputPtr, alignment);

                Value * const nextPackIndex = b.CreateAdd(packIndexPhi, sz_ONE);
                packIndexPhi->addIncoming(nextPackIndex, packCopyLoop);
                b.CreateCondBr(b.CreateICmpEQ(nextPackIndex, packIndex), packCopyExit, packCopyLoop);

                b.SetInsertPoint(packCopyExit);
                inputPtr = tmp.getStreamPackPtr(b, inputAddress, streamIndex, fullCopyEnd, packIndex);
                outputPtr = tmp.getStreamPackPtr(b, maskedAddress, streamIndex, fullCopyEnd, packIndex);
            }
            assert (inputPtr->getType() == outputPtr->getType());
            Value * const val = b.CreateAlignedLoad(b.getBitBlockType(), inputPtr, alignment);
            Value * const maskedVal = b.CreateAnd(val, mask);
            b.CreateAlignedStore(maskedVal, outputPtr, alignment);

            Value * const nextIndex = b.CreateAdd(streamIndex, sz_ONE);
            Value * const notDone = b.CreateICmpNE(nextIndex, numOfStreams);
            streamIndex->addIncoming(nextIndex, b.GetInsertBlock());

            b.CreateCondBr(notDone, maskedInputLoop, maskedInputExit);

            b.SetInsertPoint(maskedInputExit);
            Value * const retAddress = b.CreatePointerCast(maskedAddress, int8PtrTy);

            if (LLVM_UNLIKELY(mCheckStreamSets)) {
                FixedArray<Value *, 2> fields;
                fields[0] = retAddress;
                fields[1] = mallocBytes;
                b.CreateAggregateRet(fields.data(), 2);
            } else {
                b.CreateRet(retAddress);
            }
            b.restoreIP(ip);
        }

        FixedArray<Value *, 7> args;

        args[0] = b.CreatePointerCast(inputBaseAddresses[inputPort.Number], int8PtrTy);

        const auto ic = port.Maximum * StrideStepLength[mKernelId];
        assert (ic.denominator() == 1);
        assert (ic.numerator() > 0);
        const auto minimumSize = round_up_to<size_t>(ic.numerator(), blockWidth);
        ConstantInt * const itemsPerSegment = b.getSize(minimumSize);

        args[1] = itemsPerSegment;
        Value * processedItems = nullptr;
        if (port.isDeferred()) {
            processedItems = mCurrentProcessedDeferredItemCountPhi[inputPort];
        } else {
            processedItems = mCurrentProcessedItemCountPhi[inputPort];
        }
        args[2] = processedItems;
        Value * const maxItems = b.CreateAdd(mCurrentProcessedItemCountPhi[inputPort], accessibleItems[inputPort.Number]);
        args[3] = maxItems;
        assert (port.RequiredOverflowSpace >= 0);
        args[4] = b.getSize(std::max<size_t>(port.RequiredOverflowSpace, minimumSize));
        args[5] = buffer->getStreamSetCount(b);
        args[6] = b.CreateGEP(traceArTy, base, indices);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + " truncating input to [%" PRIu64 ",%" PRIu64 ")", processedItems, maxItems);
        #endif

        Value * const maskVal = b.CreateCall(maskInput->getFunctionType(), maskInput, args);

        Value * maskedAddress = maskVal;
        Value * maskedCapacity = nullptr;
        if (LLVM_UNLIKELY(mCheckStreamSets)) {
            maskedAddress = b.CreateExtractValue(maskVal, {0});
            Value * const maskedBytes = b.CreateExtractValue(maskVal, {1});
            const auto & DL = m->getDataLayout();
            const auto ts = b.getTypeSize(DL, buffer->getType());
            maskedCapacity = b.CreateMulRational(maskedBytes, Rational{b.getBitBlockWidth(), ts});
            maskedCapacity = b.CreateAdd(processedItems, maskedCapacity);
        }
        assert (maskedAddress->getType()->isPointerTy());
        maskedAddress = b.CreatePointerCast(maskedAddress, bufferType);
        BasicBlock * const maskedInputLoopExit = b.GetInsertBlock();
        b.CreateBr(selectedInput);

        b.SetInsertPoint(selectedInput);
        if (LLVM_UNLIKELY(mCheckStreamSets)) {
            PHINode * const inputBufferCapacityPhi = b.CreatePHI(sizeTy, 2, "truncatedBufferCapacityPhi");
            if (!alwaysTruncate) {
                inputBufferCapacityPhi->addIncoming(inputBufferCapacity[inputPort.Number], entryBlock);
            }
            inputBufferCapacityPhi->addIncoming(maskedCapacity, maskedInputLoopExit);
            inputBufferCapacity[inputPort.Number] = inputBufferCapacityPhi;
        }

        PHINode * const baseAddrPhi = b.CreatePHI(bufferType, 2);
        if (!alwaysTruncate) {
            baseAddrPhi->addIncoming(inputBaseAddresses[inputPort.Number], entryBlock);
        }
        baseAddrPhi->addIncoming(maskedAddress, maskedInputLoopExit);

        inputBaseAddresses[inputPort.Number] = baseAddrPhi;
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief freeZeroedInputBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freeZeroedInputBuffers(KernelBuilder & b) {

    const auto n = num_vertices(mZeroInputGraph) - ((LastKernel - FirstKernel) + 1);
    if (n > 0) {
        Value * base;
        Type * traceArTy;
        std::tie(base, traceArTy) = b.getScalarFieldPtr(ZERO_INPUT_BUFFER_STRUCT);
        ConstantInt * const i32_ZERO = b.getInt32(0);
        FixedArray<Value *, 3> offset;
        offset[0] = i32_ZERO;
        offset[2] = i32_ZERO;
        // free any truncated input buffers
        for (size_t i = 0; i < n; ++i) {
            offset[1] = b.getInt32(i);
            Value * ptr = b.CreateAlignedLoad(b.getInt8PtrTy(), b.CreateGEP(traceArTy, base, offset), PtrTyABIAlignment);
            b.CreateFree(ptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearUnwrittenOutputData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::clearUnwrittenOutputData(KernelBuilder & b) {
    #ifndef DISABLE_OUTPUT_ZEROING
    const auto blockWidth = b.getBitBlockWidth();
    const auto log2BlockWidth = floor_log2(blockWidth);
    Constant * const LOG_2_BLOCK_WIDTH = b.getSize(log2BlockWidth);
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    Constant * const BLOCK_MASK = b.getSize(blockWidth - 1);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];

        // If this stream is either controlled by this kernel or is an external
        // stream, any clearing of data is the responsibility of the owner.
        // Simply ignore any external buffers for the purpose of zeroing out
        // unnecessary data.
        if (LLVM_UNLIKELY(bn.isUnowned() || bn.isTruncated() || bn.isConstant() || bn.hasZeroElementsOrWidth() || bn.isPopCountPartialSumStream())) {
            continue;
        }

        // TODO: don't do this for popcount outputs

        const StreamSetBuffer * const buffer = bn.Buffer;

        Value * const numOfStreams = buffer->getStreamSetCount(b);

        const BufferPort & rt = mBufferGraph[e];
        assert (rt.Port.Type == PortType::Output);
        const auto port = rt.Port;

        const auto itemWidth = getItemWidth(buffer->getBaseType());

        const auto prefix = makeBufferName(mKernelId, port);

        Value * produced = nullptr;
        if (LLVM_UNLIKELY(bn.OutputItemCountId != streamSet)) {
            produced = mLocallyAvailableItems[bn.OutputItemCountId];
        } else {
            produced = mProducedAtTermination[port];
        }

        Value * const blockIndex = b.CreateLShr(produced, LOG_2_BLOCK_WIDTH);
        Constant * const ITEM_WIDTH = b.getSize(itemWidth);
        Value * packIndex = nullptr;
        Value * maskOffset = b.CreateAnd(produced, BLOCK_MASK);
        if (itemWidth > 1) {
            Value * const position = b.CreateMul(maskOffset, ITEM_WIDTH);
            packIndex = b.CreateLShr(position, LOG_2_BLOCK_WIDTH);
            maskOffset = b.CreateAnd(position, BLOCK_MASK);
        }
        Value * const mask = b.CreateNot(b.bitblock_mask_from(maskOffset));

        Value * const baseAddress = buffer->getBaseAddress(b);

        DataLayout DL(b.getModule());
        Type * const intPtrTy = DL.getIntPtrType(baseAddress->getType());


        BasicBlock * const maskLoop = b.CreateBasicBlock(prefix + "_zeroUnwrittenLoop", mKernelLoopExit);
        BasicBlock * const maskExit = b.CreateBasicBlock(prefix + "_zeroUnwrittenExit", mKernelLoopExit);

        BasicBlock * const entry = b.GetInsertBlock();
        b.CreateCondBr(b.CreateICmpNE(maskOffset, sz_ZERO), maskLoop, maskExit);

        b.SetInsertPoint(maskLoop);
        PHINode * const streamIndexPhi = b.CreatePHI(b.getSizeTy(), 2, "streamIndex");
        streamIndexPhi->addIncoming(sz_ZERO, entry);
        Value * inputPtr = nullptr;
        if (itemWidth > 1) {
            inputPtr = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, packIndex);
        } else {
            inputPtr = buffer->getStreamBlockPtr(b, baseAddress, streamIndexPhi, blockIndex);
        }

        #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
        Value * const ptrInt = b.CreatePtrToInt(inputPtr, intPtrTy);
        debugPrint(b, prefix + "_zeroUnwritten_partialPtr = 0x%" PRIx64, ptrInt);
        #endif
        Value * const value = b.CreateBlockAlignedLoad(mask->getType(), inputPtr);
        Value * const maskedValue = b.CreateAnd(value, mask);
        b.CreateBlockAlignedStore(maskedValue, inputPtr);

        const auto isUnary = isa<ConstantInt>(numOfStreams) && cast<ConstantInt>(numOfStreams)->isOne();

        if (isUnary) {
            b.CreateBr(maskExit);
        } else {

            if (itemWidth > 1) {
                // Since packs are laid out sequentially in memory, it will hopefully be cheaper to zero them out here
                // because they may be within the same cache line.
                Value * const nextPackIndex = b.CreateAdd(packIndex, ONE);
                Value * const start = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, nextPackIndex);
                Value * const startInt = b.CreatePtrToInt(start, intPtrTy);
                Value * const end = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, ITEM_WIDTH);
                Value * const endInt = b.CreatePtrToInt(end, intPtrTy);
                Value * const remainingPackBytes = b.CreateSub(endInt, startInt);
                #ifdef PRINT_DEBUG_MESSAGES
                #ifndef PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY
                debugPrint(b, prefix + "_zeroUnwritten_clearRange = [0x%" PRIx64 ",0x%" PRIx64 ")", startInt, endInt);
                #endif
                debugPrint(b, prefix + "_zeroUnwritten_remainingBufferBytes = %" PRIu64, remainingPackBytes);
                #endif
                b.CreateMemZero(start, remainingPackBytes, blockWidth / 8);
            }

            Value * const nextStreamIndex = b.CreateAdd(streamIndexPhi, ONE);
            streamIndexPhi->addIncoming(nextStreamIndex, b.GetInsertBlock());
            Value * const notDone = b.CreateICmpNE(nextStreamIndex, numOfStreams);
            b.CreateCondBr(notDone, maskLoop, maskExit);

        }

        b.SetInsertPoint(maskExit);

        // Zero out any blocks we could potentially touch
        Rational maxVal = rt.Maximum * StrideStepLength[mKernelId];

        if (bn.isNonThreadLocal()) {
            for (auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
                for (auto input : make_iterator_range(out_edges(target(output, mBufferGraph), mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    if (KernelPartitionId[consumer] != mCurrentPartitionId) {
                        const auto & port = mBufferGraph[input];
                        const auto m = port.Maximum * StrideStepLength[consumer];
                        if (maxVal < m) {
                            maxVal = m;
                        }
                    }
                }
            }
        }

        const auto blocksPerStride = ceiling(maxVal / blockWidth);
        assert (blocksPerStride > 0);

        const auto doUnaryPack = (isUnary && itemWidth > 1);

        if (doUnaryPack || blocksPerStride > 1 || rt.RequiredOverflowSpace > 0) {

//            Value * startPtr = nullptr;
//            Value * endPtr = nullptr;

            auto getEndOffset = [&](Value * current) {
                if (blocksPerStride > 1) {
                    current = b.CreateRoundUpRational(current, blocksPerStride);
                }
                if (rt.RequiredOverflowSpace) {
                    const auto k = ceiling(Rational{rt.RequiredOverflowSpace, blockWidth});
                    current = b.CreateAdd(current, b.getSize(k));
                }
                return current;
            };

//            if (doUnaryPack) {
//                Value * const nextPackIndex = b.CreateAdd(packIndex, ONE);
//                Value * const nextBlockIndex = b.CreateAdd(blockIndex, ONE);
//                Value * const nextOffset = buffer->modByCapacity(b, nextBlockIndex);
//                startPtr = buffer->StreamSetBuffer::getStreamPackPtr(b, baseAddress, sz_ZERO, nextOffset, nextPackIndex);
//                endPtr = buffer->StreamSetBuffer::getStreamPackPtr(b, baseAddress, sz_ZERO, getEndOffset(nextOffset), ITEM_WIDTH);
//            }  else {
                Value * const nextBlockIndex = b.CreateAdd(blockIndex, ONE);
                Value * const nextOffset = buffer->modByCapacity(b, nextBlockIndex);
                Value * const startPtr = buffer->StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, nextOffset);
                Value * const endPtr = buffer->StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, getEndOffset(nextOffset));
//            }

            Value * const startPtrInt = b.CreatePtrToInt(startPtr, intPtrTy);
            Value * const endPtrInt = b.CreatePtrToInt(endPtr, intPtrTy);
            Value * const remainingBytes = b.CreateSub(endPtrInt, startPtrInt);

            #ifdef PRINT_DEBUG_MESSAGES
            #ifndef PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY
            debugPrint(b, prefix + "_zeroUnwritten_clearRange%" PRIu8 ",%" PRIu64 ",%" PRIu64 " = [0x%" PRIx64 ",0x%" PRIx64 ")",
                       b.getInt8(doUnaryPack), b.getInt64(blocksPerStride), b.getInt64(rt.RequiredOverflowSpace), startPtrInt, endPtrInt);
            #endif
            debugPrint(b, prefix + "_zeroUnwritten_remainingBufferBytes = %" PRIu64, remainingBytes);
            #endif
            b.CreateMemZero(startPtr, remainingBytes, blockWidth / 8);
        }

    }
    #endif
}

}
