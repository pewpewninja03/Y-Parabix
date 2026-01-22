#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateBufferExpansionFunctionForCurrentKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::generateBufferExpansionFunctionForCurrentKernel(KernelBuilder & b, const size_t kernelId) {

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    name << "__rmblog_" << mTarget->getName() << '.' << kernelId;

    Module * const m = b.getModule();

    assert ("unspecified module" && b.getModule());

    Type * const voidPtrTy = b.getVoidPtrTy();

    Function * const f0 = m->getFunction(name.str());
    if (f0) {
        return b.CreatePointerCast(f0, voidPtrTy);
    }

    bool noManagedBuffer = true;

    GlobalValue::LinkageTypes linkage = Function::InternalLinkage;

    for (auto output : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[output];
        if (LLVM_UNLIKELY(bp.isManaged())) {
            assert (getKernel(kernelId)->getKernelFlags() & Kernel::KernelFlags::HasInternallyManagedStreamSet);
            linkage = Function::ExternalLinkage;
            noManagedBuffer = false;
        }
        const auto streamSet = target(output, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (isa<ManagedDynamicBuffer>(bn.Buffer)) {
            noManagedBuffer = false;
        }
    }

    if (noManagedBuffer) {
        return nullptr;
    }

    Value * const initialSharedState = getHandle();
    Value * const initialThreadLocal = getThreadLocalHandle();

    auto ip = b.saveIP();

    ScalarValueMap originalScalarFieldMap(mScalarFieldMap);
    ScalarAliasMap originalScalarFieldMapScalarAliasMap(mScalarAliasMap);
    BindingMap originalBindingMap(mBindingMap);

    auto & DL = m->getDataLayout();

    IntegerType * const sizeTy = b.getSizeTy();

    const auto sizeTyWidth = b.getTypeSize(DL, sizeTy);
    const auto sizeTyAlign = DL.getABITypeAlign(sizeTy).value();

    PointerType * const i8PtrTy = b.getInt8PtrTy();

    const auto int8PtrTyAlign = DL.getABITypeAlign(i8PtrTy).value();

    StructType * handleTy = mTarget->getSharedStateType();
    PointerType * handlePtrTy = handleTy->getPointerTo();


    FixedArray<Type *, 4> paramTypes;
    paramTypes[0] = voidPtrTy; // pipeline handle
    paramTypes[1] = sizeTy; // port num
    paramTypes[2] = sizeTy; // produced
    paramTypes[3] = sizeTy; // capacity

    FunctionType * funcTy = FunctionType::get(b.getVoidTy(), paramTypes, false);

    Function * const f = Function::Create(funcTy, linkage, name.str(), m);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        f->setHasUWTable();
        #else
        f->setUWTableKind(UWTableKind::Default);
        #endif
    }

    LLVMContext & C = m->getContext();
    BasicBlock * const entry = BasicBlock::Create(C, "entry", f);
    const auto outputPorts = out_degree(kernelId, mBufferGraph);
    SmallVector<BasicBlock *, 8> outputPortHandler(outputPorts);
    for (unsigned i = 0; i < outputPorts; ++i) {
        outputPortHandler[i] = BasicBlock::Create(C, "port" + std::to_string(i), f);
    }
    BasicBlock * const exit = BasicBlock::Create(C, "exit", f);

    b.SetInsertPoint(entry);

    auto arg = f->arg_begin();
    auto nextArg = [&]() {
        assert (arg != f->arg_end());
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };

    Value * handle = nextArg();
    handle->setName("handle");
    handle = b.CreatePointerCast(handle, handlePtrTy);
    Value * outputPortNum = nextArg();
    outputPortNum->setName("outputPortNum");
    Value * produced = nextArg();
    produced->setName("produced");
    Value * newCapacity = nextArg();
    newCapacity->setName("capacity");
    assert (arg == f->arg_end());

    setHandle(handle);
    setThreadLocalHandle(nullptr);
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);

    const auto type = isDataParallel(kernelId) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
    Value * const syncLockPtr = getSynchronizationLockPtrForKernel(b, kernelId, type);
    Value * const currentSegNo = b.CreateAlignedLoad(sizeTy, syncLockPtr, sizeTyAlign);

    if (LLVM_LIKELY(b.supportsIndirectBr())) {

        SmallVector<Constant *, 8> jumpAddr(outputPorts);
        for (unsigned i = 0; i < outputPorts; ++i) {
            jumpAddr[i] = BlockAddress::get(f, outputPortHandler[i]);
        }

        ArrayType * const jumpAddrTableTy = ArrayType::get(i8PtrTy, outputPorts);
        Constant * const jumpAddrTable = ConstantArray::get(jumpAddrTableTy, jumpAddr);

        GlobalVariable * const jumpTargetArray =
            new GlobalVariable(*m, jumpAddrTableTy, true, linkage, jumpAddrTable);

        FixedArray<Value *, 2> jumpIndex;
        jumpIndex[0] = b.getSize(0);
        jumpIndex[1] = outputPortNum;

        Value * const targetPtr = b.CreateGEP(jumpAddrTableTy, jumpTargetArray, jumpIndex);
        Value * const target = b.CreateAlignedLoad(i8PtrTy, targetPtr, int8PtrTyAlign);

        IndirectBrInst * const br = b.CreateIndirectBr(target, outputPorts);
        for (unsigned i = 0; i < outputPorts; ++i) {
            br->addDestination(outputPortHandler[i]);
        }

    } else {

        SmallVector<BasicBlock *, 8> condCheckBlock(outputPorts + 1);

        condCheckBlock[0] = entry;
        for (unsigned i = 1; i < outputPorts; ++i) {
            condCheckBlock[i] = BasicBlock::Create(C, "", f, outputPortHandler[i]);
        }
        condCheckBlock[outputPorts] = b.CreateBasicBlock("", exit);

        for (unsigned i = 0; i < outputPorts; ++i) {

            b.SetInsertPoint(condCheckBlock[i]);
            Value * const matchingPort = b.CreateICmpEQ(outputPortNum, b.getSize(i));
            b.CreateCondBr(matchingPort, outputPortHandler[i], condCheckBlock[i + 1]);

        }

        b.SetInsertPoint(condCheckBlock[outputPorts]);
        b.CreateUnreachable();
    }

    ConstantInt * sz_ONE = b.getSize(1);

    ConstantInt * i32_ZERO = b.getInt32(0);
    ConstantInt * i32_ONE = b.getInt32(1);
    ConstantInt * i32_TWO = b.getInt32(2);
    ConstantInt * i32_THREE = b.getInt32(3);

    for (auto output : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[output];

        b.SetInsertPoint(outputPortHandler[bp.Port.Number]);

        const auto streamSet = target(output, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bp.isManaged() || isa<ManagedDynamicBuffer>(bn.Buffer)) {

            const auto prefix = makeBufferName(kernelId, bp.Port);
            Value * traceData; Type * traceDataTy;
            std::tie(traceData, traceDataTy) = b.getScalarFieldPtr(prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX);

            const auto numOfConsumers = std::max(out_degree(streamSet, mConsumerGraph), 1UL);
            Type * const entryTy = ArrayType::get(sizeTy, numOfConsumers + 3);

            FixedArray<Value *, 2> indices;
            indices[0] = i32_ZERO;
            indices[1] = i32_ZERO;

            Value * const traceLogArrayField = b.CreateGEP(traceDataTy, traceData, indices);
            Value * entryArray = b.CreateAlignedLoad(entryTy->getPointerTo(), traceLogArrayField, PtrTyABIAlignment);

            indices[1] = i32_ONE;

            Value * const traceLogCountField = b.CreateGEP(traceDataTy, traceData, indices);
            Value * const traceIndex = b.CreateAlignedLoad(sizeTy, traceLogCountField, SizeTyABIAlignment);
            Value * const traceCount = b.CreateAdd(traceIndex, sz_ONE);

            entryArray = b.CreateRealloc(entryTy, entryArray, traceCount);
            b.CreateAlignedStore(entryArray, traceLogArrayField, PtrTyABIAlignment);
            b.CreateAlignedStore(traceCount, traceLogCountField, SizeTyABIAlignment);

            entryArray = b.CreateRealloc(entryTy, entryArray, traceCount);
            b.CreateAlignedStore(entryArray, traceLogArrayField, PtrTyABIAlignment);
            b.CreateAlignedStore(traceCount, traceLogCountField, SizeTyABIAlignment);

            indices[0] = traceIndex;

            // segment num  0
            indices[1] = i32_ZERO;
            b.CreateAlignedStore(currentSegNo, b.CreateGEP(entryTy, entryArray, indices), SizeTyABIAlignment);
            // new capacity 1
            indices[1] = i32_ONE;
            b.CreateAlignedStore(newCapacity, b.CreateGEP(entryTy, entryArray, indices), SizeTyABIAlignment);
            // produced item count 2
            indices[1] = i32_TWO;
            b.CreateAlignedStore(produced, b.CreateGEP(entryTy, entryArray, indices), SizeTyABIAlignment);

            // consumer processed item count [3,n)
            const auto id = getTruncatedStreamSetSourceId(streamSet);
            assert (out_degree(id, mConsumerGraph) > 0);
            Value * consumerDataPtr; Type * consumerTy;
            std::tie(consumerDataPtr, consumerTy) = b.getScalarFieldPtr(CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id));

            indices[0] = i32_ZERO;
            indices[1] = i32_ONE;

            Value * const processedPtr = b.CreateGEP(consumerTy, consumerDataPtr, indices);

            indices[0] = traceIndex;
            indices[1] = i32_THREE;

            Value * const logPtr = b.CreateGEP(entryTy, entryArray, indices);
            Constant * const length = b.getSize(sizeTyWidth * numOfConsumers);
            b.CreateMemCpy(logPtr, processedPtr, length, sizeTyWidth);

            // TODO: if this was a maanged output, ideally we'd let the outer pipeline report it but that'd require us
            // to pass in the outer function pointer which might not have the same output portnum as the kernel does.
            // We could make this generate a wrapper function and pass the outer pipeline handle instead but then couldn't
            // correctly handle the cast where some but not all of the managed buffers were outputs of the pipeline.
            // Should we store the functionpointer and outer pipeline object in the kernel state of this pipeline?

            b.CreateBr(exit);


        } else { // unmanaged
            b.CreateUnreachable();
        }
    }

    b.SetInsertPoint(exit);
    b.CreateRetVoid();

    b.restoreIP(ip);

    setHandle(initialSharedState);
    setThreadLocalHandle(initialThreadLocal);

    mScalarFieldMap = originalScalarFieldMap;
    mScalarAliasMap = originalScalarFieldMapScalarAliasMap;
    mBindingMap = originalBindingMap;

    return b.CreatePointerCast(f, voidPtrTy);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalBufferExpansionHistory
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBufferExpansionHistory(KernelBuilder & b) {

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {

        // Print the title line
        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        size_t maxKernelLength = 0;
        size_t maxBindingLength = 0;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const Kernel * const kernel = getKernel(i);
            maxKernelLength = std::max(maxKernelLength, kernel->getName().size());
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
        }
        if (LLVM_UNLIKELY(in_degree(PipelineOutput, mConsumerGraph) > 0)) {
            maxKernelLength = std::max(maxKernelLength, mTarget->getName().size());
        }

        maxKernelLength += 4;
        maxBindingLength += 4;

        SmallVector<char, 160> buffer;
        raw_svector_ostream format(buffer);

        // TODO: if expanding buffers are supported again, we need another field here for streamset size

        format << "BUFFER EXPANSION HISTORY:\n\n"
                  "  # "  // kernel ID #  (only shown for first)
                  "KERNEL"; // kernel Name (only shown for first)
        format.indent(maxKernelLength - 6);
        format << "PORT";
        format.indent(5 + maxBindingLength - 4); // I/O Type (e.g., input port 3 = I3), Port Name
        format << " BUFFER " // buffer ID #
                  "        SEG # "
                  "     ITEM CAPACITY\n";

        Constant * const STDERR = b.getInt32(STDERR_FILENO);
        FixedArray<Value *, 2> constantArgs;
        constantArgs[0] = STDERR;
        constantArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, constantArgs);

        const auto totalLength = 4 + maxKernelLength + 4 + maxBindingLength + 7 + 15 + 18 + 2;

        // generate a single-line (-) bar
        buffer.clear();
        for (unsigned i = 0; i < totalLength; ++i) {
            format.write('-');
        }
        format.write('\n');
        Constant * const singleBar = b.GetString(format.str());
        constantArgs[1] = singleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);


        // generate the produced/processed title line
        buffer.clear();
        format.indent(totalLength - 19);
        format << "PRODUCED/PROCESSED\n";
        constantArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, constantArgs);

        // generate a double-line (=) bar
        buffer.clear();
        for (unsigned i = 0; i < totalLength; ++i) {
            format.write('=');
        }
        format.write('\n');
        Constant * const doubleBar = b.GetString(format.str());
        constantArgs[1] = doubleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);

        // Generate expansion line format string
        buffer.clear();
        format << "%3" PRIu32 " " // kernel #
                  "%-" << maxKernelLength << "s" // kernel name
                  "O%-3" PRIu32 " " // I/O type
                  "%-" << maxBindingLength << "s" // port name
                  "%7" PRIu32 // buffer ID #
                  "%14" PRIu64 " " // segment #
                  "%18" PRIu64 "\n"; // item capacity
        Constant * const expansionFormat = b.GetString(format.str());

        // Generate the item count history format string
        buffer.clear();
        format << "%3" PRIu32 " " // kernel #
                  "%-" << maxKernelLength << "s" // kernel name
                  "%c%-3" PRIu32 " " // I/O type
                  "%-" << maxBindingLength << "s" // port name
                  "%40" PRIu64 "\n"; // produced/processed item count
        Constant * const itemCountFormat = b.GetString(format.str());

        // Print each kernel line
        FixedArray<Value *, 9> expansionArgs;
        expansionArgs[0] = STDERR;
        expansionArgs[1] = expansionFormat;

        FixedArray<Value *, 8> itemCountArgs;
        itemCountArgs[0] = STDERR;
        itemCountArgs[1] = itemCountFormat;

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        Constant * const SZ_ZERO = b.getSize(0);
        Constant * const SZ_ONE = b.getSize(1);

        IntegerType * sizeTy = b.getSizeTy();

        for (auto i = FirstKernel; i <= LastKernel; ++i) {

            for (const auto output : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & br = mBufferGraph[output];
                const auto buffer = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[buffer];
                if (br.isManaged() || isa<ManagedDynamicBuffer>(bn.Buffer)) {

                    //  # KERNEL                      PORT                      BUFFER         SEG #      ITEM CAPACITY

                    expansionArgs[2] = b.getInt32(i);
                    expansionArgs[3] = b.GetString(getKernel(i)->getName());
                    const auto outputPort = br.Port.Number;
                    expansionArgs[4] = b.getInt32(outputPort);
                    const Binding & binding = br.Binding;
                    expansionArgs[5] = b.GetString(binding.getName());
                    expansionArgs[6] = b.getInt32(buffer);

                    const auto prefix = makeBufferName(i, br.Port);

                    Value * traceData; Type * traceTy;
                    std::tie(traceData, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX);

                    Value * const traceArrayField = b.CreateGEP(traceTy, traceData, {ZERO, ZERO});
                    const auto numOfConsumers = std::max(out_degree(buffer, mConsumerGraph), 1UL);

                    Type * const arrayTy = ArrayType::get(sizeTy, numOfConsumers + 3);
                    Value * const entryArray = b.CreateAlignedLoad(arrayTy->getPointerTo(), traceArrayField, PtrTyABIAlignment);

                    Value * const traceCountField = b.CreateGEP(traceTy, traceData, {ZERO, ONE});
                    Value * const traceCount = b.CreateAlignedLoad(sizeTy, traceCountField, SizeTyABIAlignment);

                    BasicBlock * const outputEntry = b.GetInsertBlock();
                    BasicBlock * const outputLoop = b.CreateBasicBlock(prefix + "_bufferExpansionReportLoop");
                    BasicBlock * const writeItemCount = b.CreateBasicBlock(prefix + "_bufferWriteItemCountLoop");

                    BasicBlock * const nextEntry = b.CreateBasicBlock(prefix + "_bufferWriteItemCountLoop");

                    BasicBlock * const outputExit = b.CreateBasicBlock(prefix + "_bufferExpansionReportExit");

                    b.CreateBr(outputLoop);

                    b.SetInsertPoint(outputLoop);
                    PHINode * const index = b.CreatePHI(b.getSizeTy(), 2);
                    index->addIncoming(SZ_ZERO, outputEntry);

                    Value * const isFirst = b.CreateICmpEQ(index, SZ_ZERO);
                    Value * const nextIndex = b.CreateAdd(index, SZ_ONE);
                    Value * const isLast = b.CreateICmpEQ(nextIndex, traceCount);
                    Value * const onlyEntry = b.CreateICmpEQ(traceCount, SZ_ONE);

                    Value * const openingBar = b.CreateSelect(onlyEntry, doubleBar, singleBar);

                    Value * const segmentNumField = b.CreateGEP(arrayTy, entryArray, {index, ZERO});
                    Value * const segmentNum = b.CreateAlignedLoad(sizeTy, segmentNumField, SizeTyABIAlignment);
                    expansionArgs[7] = segmentNum;

                    Value * const newBufferSizeField = b.CreateGEP(arrayTy, entryArray, {index, ONE});
                    Value * const newBufferSize = b.CreateAlignedLoad(sizeTy, newBufferSizeField, SizeTyABIAlignment);
                    expansionArgs[8] = newBufferSize;

                    b.CreateCall(fTy, Dprintf, expansionArgs);

                    constantArgs[1] = openingBar;

                    b.CreateCall(fTy, Dprintf, constantArgs);

                    b.CreateCondBr(isFirst, nextEntry, writeItemCount);

                    // Do not write processed/produced item counts for the first entry as they
                    // are guaranteed to be 0 and only add noise to the log output.
                    b.SetInsertPoint(writeItemCount);

                    // --------------------------------------------------------------------------------------------------

                    //  # KERNEL                      PORT                                           PRODUCED/PROCESSED

                    itemCountArgs[2] = expansionArgs[2];
                    itemCountArgs[3] = expansionArgs[3];
                    itemCountArgs[4] = b.getInt8('O');
                    itemCountArgs[5] = expansionArgs[4];
                    itemCountArgs[6] = expansionArgs[5];
                    Value * const producedField = b.CreateGEP(arrayTy, entryArray, {index, TWO});
                    itemCountArgs[7] = b.CreateAlignedLoad(sizeTy, producedField, SizeTyABIAlignment);

                    b.CreateCall(fTy, Dprintf, itemCountArgs);
                    itemCountArgs[4] = b.getInt8('I');
                    for (const auto e : make_iterator_range(out_edges(buffer, mConsumerGraph))) {
                        const ConsumerEdge & c = mConsumerGraph[e];
                        const auto consumer = target(e, mConsumerGraph);
                        if (LLVM_UNLIKELY(consumer == PipelineOutput)) {
                            itemCountArgs[2] = b.getInt32(PipelineInput);
                            itemCountArgs[3] = b.GetString(mTarget->getName());
                            itemCountArgs[5] = b.getInt32(0);
                            itemCountArgs[6] = b.GetString("");
                        } else {
                            itemCountArgs[2] = b.getInt32(consumer);
                            itemCountArgs[3] = b.GetString(getKernel(consumer)->getName());
                            itemCountArgs[5] = b.getInt32(c.Port);
                            const Binding & binding = getBinding(consumer, StreamSetPort{PortType::Input, c.Port});
                            itemCountArgs[6] = b.GetString(binding.getName());
                        }
                        Value * const processedField = b.CreateGEP(arrayTy, entryArray, {index, b.getInt32(c.Index + 2)});
                        itemCountArgs[7] = b.CreateAlignedLoad(sizeTy, processedField, SizeTyABIAlignment);
                        b.CreateCall(fTy, Dprintf, itemCountArgs);
                    }

                    Value * const closingBar = b.CreateSelect(isLast, doubleBar, singleBar);
                    constantArgs[1] = closingBar;
                    b.CreateCall(fTy, Dprintf, constantArgs);

                    b.CreateBr(nextEntry);

                    b.SetInsertPoint(nextEntry);
                    index->addIncoming(nextIndex, nextEntry);
                    b.CreateCondBr(isLast, outputExit, outputLoop);

                    b.SetInsertPoint(outputExit);
                    b.CreateFree(entryArray);
                }
            }
        }
        // print final new line
        constantArgs[1] = doubleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);
    }

}



}
