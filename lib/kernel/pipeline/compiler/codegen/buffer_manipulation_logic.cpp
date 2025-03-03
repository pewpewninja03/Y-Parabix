#include "../pipeline_compiler.hpp"
#include <boost/interprocess/mapped_region.hpp>

using namespace IDISA;

inline unsigned getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateLocalZeroExtensionSpace
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::allocateLocalZeroExtensionSpace(KernelBuilder & b, BasicBlock * const insertBefore) const {
    #ifndef DISABLE_ZERO_EXTEND
    const auto strideSize = mKernel->getStride();
    const auto blockWidth = b.getBitBlockWidth();
    Value * requiredSpace = nullptr;

    Constant * const ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    Value * const numOfStrides = b.CreateUMax(mNumOfLinearStrides, ONE);

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {

        const BufferPort & br = mBufferGraph[e];
        if (br.isZeroExtended()) {

            assert (HasZeroExtendedStream);

            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            const Binding & input = br.Binding;

            const auto itemWidth = getItemWidth(input.getType());
            Constant * const strideFactor = b.getSize(itemWidth * strideSize / 8);
            Value * requiredBytes = b.CreateMul(numOfStrides, strideFactor); assert (requiredBytes);
            if (br.LookAhead) {
                const auto lh = (br.LookAhead * itemWidth);
                requiredBytes = b.CreateAdd(requiredBytes, b.getSize(lh));
            }
            if (LLVM_LIKELY(itemWidth < blockWidth)) {
                Constant * const factor = b.getSize(blockWidth / itemWidth);
                assert ((blockWidth % itemWidth) == 0);
                requiredBytes = b.CreateRoundUp(requiredBytes, factor);
            }
            requiredBytes = b.CreateMul(requiredBytes, bn.Buffer->getStreamSetCount(b));

            const auto fieldWidth = input.getFieldWidth();
            if (fieldWidth < 8) {
                requiredBytes = b.CreateUDiv(requiredBytes, b.getSize(8 / fieldWidth));
            } else if (fieldWidth > 8) {
                requiredBytes = b.CreateMul(requiredBytes, b.getSize(fieldWidth / 8));
            }
            const auto name = makeBufferName(mKernelId, br.Port);
            requiredBytes = b.CreateSelect(mIsInputZeroExtended[br.Port], requiredBytes, ZERO, "zeroExtendRequiredBytes");
            requiredSpace = b.CreateUMax(requiredSpace, requiredBytes);
        }
    }
    assert (requiredSpace);    
    const auto prefix = makeKernelName(mKernelId);
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const expandZeroExtension =
        b.CreateBasicBlock(prefix + "_expandZeroExtensionBuffer", insertBefore);
    BasicBlock * const hasSufficientZeroExtendSpace =
        b.CreateBasicBlock(prefix + "_hasSufficientZeroExtendSpace", insertBefore);

    auto zeSpaceRef = b.getScalarFieldPtr(ZERO_EXTENDED_SPACE);
    assert (zeSpaceRef.second == b.getSizeTy());
    Value * const currentSpace = b.CreateAlignedLoad(zeSpaceRef.second, zeSpaceRef.first, SizeTyABIAlignment);

    auto zeBufferRef = b.getScalarFieldPtr(ZERO_EXTENDED_BUFFER);
    Value * const currentBuffer = b.CreateAlignedLoad(zeBufferRef.second, zeBufferRef.first, PtrTyABIAlignment);

    requiredSpace = b.CreateRoundUp(requiredSpace, b.getSize(b.getCacheAlignment()));

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
    PHINode * const zeroBuffer = b.CreatePHI(b.getVoidPtrTy(), 2);
    zeroBuffer->addIncoming(currentBuffer, entry);
    zeroBuffer->addIncoming(newBuffer, expandZeroExtension);
    return zeroBuffer;
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
            const StreamSetBuffer * const buffer = bn.Buffer;

            Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
            Constant * const ZERO = b.getSize(0);
            PointerType * const bufferType = buffer->getPointerType();
            Value * const blockIndex = b.CreateLShr(processed, LOG_2_BLOCK_WIDTH);

            // allocateLocalZeroExtensionSpace guarantees this will be large enough to satisfy the kernel
            ExternalBuffer tmp(0, b, binding.getType(), true, buffer->getAddressSpace());
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
void PipelineCompiler::zeroInputAfterFinalItemCount(KernelBuilder & b, const Vec<Value *> & accessibleItems, Vec<Value *> & inputBaseAddresses) {
    #ifndef DISABLE_INPUT_ZEROING
    const auto n = out_degree(mKernelId - FirstKernel, mZeroInputGraph);
    if (n == 0) {
        return;
    }
    assert (num_vertices(mZeroInputGraph) > LastKernel);

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

    for (auto p : make_iterator_range(out_edges(mKernelId - FirstKernel, mZeroInputGraph))) {
        const auto portNum = mZeroInputGraph[p];

        const auto e = getInput(mKernelId, StreamSetPort{PortType::Input, portNum});

        const auto streamSet = source(e, mBufferGraph);

        const BufferNode & bn = mBufferGraph[streamSet];
        const StreamSetBuffer * const buffer = bn.Buffer;
        const BufferPort & port = mBufferGraph[e];
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

        Value * const selected = accessibleItems[inputPort.Number];
        Value * const totalNumOfItems = mLocallyAvailableItems[streamSet]; // getAccessibleInputItems(b, port);


        const auto alwaysTruncate = bn.isUnowned() || bn.isTruncated() || bn.isConstant();

        if (LLVM_UNLIKELY(alwaysTruncate)) {
            b.CreateBr(maskedInput);
        } else {
            Value * const tooMany = b.CreateICmpULT(selected, totalNumOfItems);
            Value * computeMask = tooMany;
            if (mIsInputZeroExtended[inputPort]) {
                computeMask = b.CreateAnd(tooMany, b.CreateNot(mIsInputZeroExtended[inputPort]));
            }
            b.CreateUnlikelyCondBr(computeMask, maskedInput, selectedInput);
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


            IntegerType * const sizeTy = b.getSizeTy();

            const auto blockWidth = b.getBitBlockWidth();
            const auto log2BlockWidth = floor_log2(blockWidth);
            Constant * const BLOCK_MASK = b.getSize(blockWidth - 1);
            Constant * const LOG_2_BLOCK_WIDTH = b.getSize(log2BlockWidth);

            const auto ip = b.saveIP();

            FixedArray<Type *, 6> params;
            params[0] = int8PtrTy; // input buffer
            params[1] = sizeTy; // items per stride
            params[2] = sizeTy; // start
            params[3] = sizeTy; // end
            params[4] = sizeTy; // numOfStreams
            params[5] = truncTy->getPointerTo(); // masked buffer storage ptr

            LLVMContext & C = m->getContext();

            FunctionType * const funcTy = FunctionType::get(int8PtrTy, params, false);
            maskInput = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
            if (LLVM_UNLIKELY(CheckAssertions)) {
                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
                maskInput->setHasUWTable();
                #else
                maskInput->setUWTableKind(UWTableKind::Default);
                #endif
            }

            BasicBlock * const entry = BasicBlock::Create(C, "entry", maskInput);
            b.SetInsertPoint(entry);

            auto arg = maskInput->arg_begin();
            auto nextArg = [&]() {
                assert (arg != maskInput->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };


            DataLayout DL(b.getModule());
            Type * const intPtrTy = DL.getIntPtrType(int8PtrTy);

            Value * const inputBuffer = nextArg();
            inputBuffer->setName("inputBuffer");
            Value * const itemsPerSegment = nextArg();
            itemsPerSegment->setName("itemsPerSegment");
            Value * const start = nextArg();
            start->setName("start");
            Value * const end = nextArg();
            end->setName("end");
            Value * const numOfStreams = nextArg();
            numOfStreams->setName("numOfStreams");
            Value * const bufferStorage = nextArg();
            bufferStorage->setName("bufferStorage");
            assert (arg == maskInput->arg_end());


            Type * const singleElementStreamSetTy = ArrayType::get(FixedVectorType::get(IntegerType::get(C, itemWidth), static_cast<unsigned>(0)), 1);
            ExternalBuffer tmp(0, b, singleElementStreamSetTy, true, 0);
            PointerType * const bufferPtrTy = tmp.getPointerType();

            Value * const inputAddress = b.CreatePointerCast(inputBuffer, bufferPtrTy);
            Value * const initial = b.CreateMul(b.CreateLShr(start, LOG_2_BLOCK_WIDTH), numOfStreams);
            Value * const initialPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, initial);
            Value * const initialPtrInt = b.CreatePtrToInt(initialPtr, intPtrTy);


            Value * const requiredItemsPerStream = b.CreateAdd(end, itemsPerSegment);
            Value * const requiredBlocksPerStream = b.CreateCeilUDivRational(requiredItemsPerStream, blockWidth);
            Value * const requiredBlocks = b.CreateMul(requiredBlocksPerStream, numOfStreams);
            Value * const requiredPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, requiredBlocks);
            Value * const requiredPtrInt = b.CreatePtrToInt(requiredPtr, intPtrTy);

            Value * const mallocBytes = b.CreateSub(requiredPtrInt, initialPtrInt);
            const auto blockSize = b.getBitBlockWidth() / 8;
            const auto alignment = unaligned ? 1 : blockSize;

            BasicBlock * const allocateNewBuffer = b.CreateBasicBlock("allocateNewBuffer");
            BasicBlock * const allocateNewBufferExit = b.CreateBasicBlock("allocateNewBufferExit");

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
            PHINode * maskedBuffer = b.CreatePHI(int8PtrTy, 2);
            maskedBuffer->addIncoming(existingBuffer, entry);
            maskedBuffer->addIncoming(newBuffer, allocateNewBuffer);

            BasicBlock * const hasDataToCopy = b.CreateBasicBlock("hasDataToCopy");
            BasicBlock * const maskedInputLoop = BasicBlock::Create(C, "maskInputLoop", maskInput);
            BasicBlock * const maskedInputExit = BasicBlock::Create(C, "maskInputExit", maskInput);

            Value * const mallocedAddress = b.CreatePointerCast(maskedBuffer, bufferPtrTy);
            Value * const outputVBA = tmp.getStreamBlockPtr(b, mallocedAddress, sz_ZERO, b.CreateNeg(initial));
            Value * const maskedAddress = b.CreatePointerCast(outputVBA, bufferPtrTy);
            assert (maskedAddress->getType() == inputAddress->getType());
            b.CreateCondBr(b.CreateIsNull(mallocBytes), maskedInputExit, hasDataToCopy);

            b.SetInsertPoint(hasDataToCopy);
            b.CreateMemZero(maskedBuffer, mallocBytes, blockSize);
            Value * const total = b.CreateLShr(end, LOG_2_BLOCK_WIDTH);
            Value * const fullCopyEnd = b.CreateMul(total, numOfStreams);
            Value * const fullCopyEndPtr = tmp.getStreamBlockPtr(b, inputAddress, sz_ZERO, fullCopyEnd);
            Value * const fullCopyEndPtrInt = b.CreatePtrToInt(fullCopyEndPtr, intPtrTy);
            Value * const fullBytesToCopy = b.CreateSub(fullCopyEndPtrInt, initialPtrInt);
            b.CreateMemCpy(mallocedAddress, initialPtr, fullBytesToCopy, alignment);

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
            PHINode * const streamIndex = b.CreatePHI(b.getSizeTy(), 2);
            streamIndex->addIncoming(sz_ZERO, loopEntryBlock);

            Value * inputPtr = tmp.getStreamBlockPtr(b, inputAddress, streamIndex, fullCopyEnd);
            Value * outputPtr = tmp.getStreamBlockPtr(b, maskedAddress, streamIndex, fullCopyEnd);

            assert (inputPtr->getType() == outputPtr->getType());
            if (itemWidth > 1) {
                Value * const partialCopyInputEndPtr = tmp.getStreamPackPtr(b, inputAddress, streamIndex, fullCopyEnd, packIndex);
                Value * const partialCopyInputEndPtrInt = b.CreatePtrToInt(partialCopyInputEndPtr, intPtrTy);
                Value * const partialCopyInputStartPtrInt = b.CreatePtrToInt(inputPtr, intPtrTy);
                Value * const bytesToCopy = b.CreateSub(partialCopyInputEndPtrInt, partialCopyInputStartPtrInt);
                b.CreateMemCpy(outputPtr, inputPtr, bytesToCopy, alignment);
                inputPtr = partialCopyInputEndPtr;
                outputPtr = tmp.getStreamPackPtr(b, maskedAddress, streamIndex, fullCopyEnd, packIndex);
            }
            assert (inputPtr->getType() == outputPtr->getType());
            Value * const val = b.CreateAlignedLoad(mask->getType(), inputPtr, alignment);
            Value * const maskedVal = b.CreateAnd(val, mask);
            b.CreateAlignedStore(maskedVal, outputPtr, alignment);

            Value * const nextIndex = b.CreateAdd(streamIndex, sz_ONE);
            Value * const notDone = b.CreateICmpNE(nextIndex, numOfStreams);
            streamIndex->addIncoming(nextIndex, maskedInputLoop);

            b.CreateCondBr(notDone, maskedInputLoop, maskedInputExit);

            b.SetInsertPoint(maskedInputExit);
            b.CreateRet(b.CreatePointerCast(maskedAddress, int8PtrTy));

            b.restoreIP(ip);
        }



        FixedArray<Value *, 6> args;

        args[0] = b.CreatePointerCast(inputBaseAddresses[inputPort.Number], int8PtrTy);

        const auto ic = port.Maximum * StrideStepLength[mKernelId];
        assert (ic.denominator() == 1);
        assert (ic.numerator() > 0);
        args[1] = b.getSize(ic.numerator());
        Value * processed = nullptr;
        if (port.isDeferred()) {
            processed = mCurrentProcessedDeferredItemCountPhi[inputPort];
        } else {
            processed = mCurrentProcessedItemCountPhi[inputPort];
        }
        args[2] = processed;
        args[3] = b.CreateAdd(mCurrentProcessedItemCountPhi[inputPort], selected);
        args[4] = buffer->getStreamSetCount(b);
        args[5] = b.CreateGEP(traceArTy, base, indices);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + " truncating item count from %" PRIu64 " to %" PRIu64,
                  totalNumOfItems, selected);
        #endif

        Value * const maskedAddress = b.CreatePointerCast(b.CreateCall(maskInput->getFunctionType(), maskInput, args), bufferType);
        BasicBlock * const maskedInputLoopExit = b.GetInsertBlock();
        b.CreateBr(selectedInput);

        b.SetInsertPoint(selectedInput);
        PHINode * const phi = b.CreatePHI(bufferType, 2);
        if (!alwaysTruncate) {
            phi->addIncoming(inputBaseAddresses[inputPort.Number], entryBlock);
        }
        phi->addIncoming(maskedAddress, maskedInputLoopExit);
        inputBaseAddresses[inputPort.Number] = phi;

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
        if (LLVM_UNLIKELY(bn.isUnowned() || bn.isTruncated() || bn.isConstant() || bn.hasZeroElementsOrWidth())) {
            continue;
        }

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
        BasicBlock * const maskLoop = b.CreateBasicBlock(prefix + "_zeroUnwrittenLoop", mKernelLoopExit);
        BasicBlock * const maskExit = b.CreateBasicBlock(prefix + "_zeroUnwrittenExit", mKernelLoopExit);


        Value * const baseAddress = buffer->getBaseAddress(b);

        DataLayout DL(b.getModule());
        Type * const intPtrTy = DL.getIntPtrType(baseAddress->getType());

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

        #ifdef PRINT_DEBUG_MESSAGES
        Value * const ptrInt = b.CreatePtrToInt(inputPtr, intPtrTy);
        debugPrint(b, prefix + "_zeroUnwritten_partialPtr = 0x%" PRIx64, ptrInt);
        #endif
        Value * const value = b.CreateBlockAlignedLoad(mask->getType(), inputPtr);
        Value * const maskedValue = b.CreateAnd(value, mask);

        Value * outputPtr = inputPtr;
        if (LLVM_UNLIKELY(bn.isTruncated())) {
            if (itemWidth > 1) {
                outputPtr = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, packIndex);
            } else {
                outputPtr = buffer->getStreamBlockPtr(b, baseAddress, streamIndexPhi, blockIndex);
            }
        }
        b.CreateBlockAlignedStore(maskedValue, outputPtr);
        if (itemWidth > 1) {
            // Since packs are laid out sequentially in memory, it will hopefully be cheaper to zero them out here
            // because they may be within the same cache line.
            Value * const nextPackIndex = b.CreateAdd(packIndex, ONE);
            Value * const start = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, nextPackIndex);
            Value * const startInt = b.CreatePtrToInt(start, intPtrTy);
            Value * const end = buffer->getStreamPackPtr(b, baseAddress, streamIndexPhi, blockIndex, ITEM_WIDTH);
            Value * const endInt = b.CreatePtrToInt(end, intPtrTy);
            Value * const remainingPackBytes = b.CreateSub(endInt, startInt);
            b.CreateMemZero(start, remainingPackBytes, blockWidth / 8);
        }
        BasicBlock * const maskLoopExit = b.GetInsertBlock();
        Value * const nextStreamIndex = b.CreateAdd(streamIndexPhi, ONE);
        streamIndexPhi->addIncoming(nextStreamIndex, maskLoopExit);
        Value * const notDone = b.CreateICmpNE(nextStreamIndex, numOfStreams);
        b.CreateCondBr(notDone, maskLoop, maskExit);

        b.SetInsertPoint(maskExit);

        // Zero out any blocks we could potentially touch
        const BufferPort & rd = mBufferGraph[e];
        auto strideLength = rd.Maximum + rd.Add;
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & rd = mBufferGraph[e];
            const auto d = std::max(rd.LookAhead, rd.Add);
            const auto r = rd.Maximum + d;
            strideLength = std::max(strideLength, r);
        }

        const auto blocksToZero = ceiling(Rational{strideLength.numerator(), blockWidth * strideLength.denominator()});


        if (blocksToZero > 1) {
            Value * const nextBlockIndex = b.CreateAdd(blockIndex, ONE);
            Value * const nextOffset = buffer->modByCapacity(b, nextBlockIndex);
            Value * const startPtr = buffer->StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, nextOffset);
            Value * const startPtrInt = b.CreatePtrToInt(startPtr, intPtrTy);
            Value * const endOffset = b.CreateRoundUp(nextOffset, b.getSize(blocksToZero));
            Value * const endPtr = buffer->StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, endOffset);
            Value * const endPtrInt = b.CreatePtrToInt(endPtr, intPtrTy);
            Value * const remainingBytes = b.CreateSub(endPtrInt, startPtrInt);
//            #ifdef PRINT_DEBUG_MESSAGES
//            debugPrint(b, prefix + "_zeroUnwritten_bufferStart = %" PRIu64, b.CreateSub(startPtrInt, epochInt));
//            debugPrint(b, prefix + "_zeroUnwritten_remainingBufferBytes = %" PRIu64, remainingBytes);
//            #endif
            b.CreateMemZero(startPtr, remainingBytes, blockWidth / 8);
        }
    }
    #endif
}

}
