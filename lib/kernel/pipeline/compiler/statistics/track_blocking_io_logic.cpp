#include "../pipeline_compiler.hpp"
#include <boost/format.hpp>
#include <llvm/Support/Format.h>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addTrackBlockingIOSummaryProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addTrackBlockingIOSummaryProperties(KernelBuilder & b, const unsigned kernelId, const unsigned groupId) {

    assert (codegen::StatisticsOptionIsSet(codegen::EnableBlockingIOCounter));

    IntegerType * const int64Ty = b.getInt64Ty();

    // # of blocked I/O channel attempts in which no strides
    // were possible (i.e., blocked on first iteration)
    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const auto & bp = mBufferGraph[e];
        if (bp.Flags & (BufferPortType::TrackBlockedIOSummary)) {
            const auto prefix = makeBufferName(kernelId, bp.Port);
            errs() << "track adding " << prefix << "\n";
            mTarget->addInternalScalar(int64Ty, prefix + STATISTICS_BLOCKING_IO_SUFFIX, groupId);
        }
    }
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto & bp = mBufferGraph[e];
        if (bp.Flags & (BufferPortType::TrackBlockedIOSummary)) {
            const auto prefix = makeBufferName(kernelId, bp.Port);
            errs() << "track adding " << prefix << "\n";
            mTarget->addInternalScalar(int64Ty, prefix + STATISTICS_BLOCKING_IO_SUFFIX, groupId);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addTrackBlockingIOHistoryProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addTrackBlockingIOHistoryProperties(KernelBuilder & b, const unsigned kernelId, const unsigned groupId) {

    assert (codegen::StatisticsOptionIsSet(codegen::TraceBlockedIO));

    FixedArray<Type *, 3> fields;
    IntegerType * const sizeTy = b.getSizeTy();
    fields[0] = sizeTy->getPointerTo();
    fields[1] = sizeTy;
    fields[2] = sizeTy;
    StructType * const historyTy = StructType::get(b.getContext(), fields);

    // # of blocked I/O channel attempts in which no strides
    // were possible (i.e., blocked on first iteration)
    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const auto & bp = mBufferGraph[e];
        if (bp.Flags & (BufferPortType::TrackBlockedIO)) {
            const auto prefix = makeBufferName(kernelId, bp.Port);
            mTarget->addInternalScalar(historyTy, prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX, groupId);
        }
    }
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto & bp = mBufferGraph[e];
        if (bp.Flags & (BufferPortType::TrackBlockedIO)) {
            const auto prefix = makeBufferName(kernelId, bp.Port);
            mTarget->addInternalScalar(historyTy, prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX, groupId);
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordBlockingIO
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordBlockingIO(KernelBuilder & b, const BufferPort & port) const {
    assert (port.Flags & (BufferPortType::TrackBlockedIO | BufferPortType::TrackBlockedIOSummary));
    const auto prefix = makeBufferName(mKernelId, port.Port);
    if ((port.Flags & BufferPortType::TrackBlockedIOSummary) != 0) {
        Value * counterPtr; Type * ty;
        std::tie(counterPtr, ty) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
        Value * const runningCount = b.CreateAlignedLoad(ty, counterPtr, Int64TyABIAlignment);
        Value * const updatedCount = b.CreateAdd(runningCount, b.getSize(1));
        b.CreateAlignedStore(updatedCount, counterPtr, Int64TyABIAlignment);
    }
    if ((port.Flags & BufferPortType::TrackBlockedIO) != 0) {
        Value * historyPtr; Type * ty;
        std::tie(historyPtr, ty) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        IntegerType * sizeTy = b.getSizeTy();

        Value * const traceLogArrayField = b.CreateGEP(ty, historyPtr, {ZERO, ZERO});
        Value * const traceLogArray = b.CreateAlignedLoad(sizeTy->getPointerTo(), traceLogArrayField, PtrTyABIAlignment);

        Value * const traceLogCountField = b.CreateGEP(ty, historyPtr, {ZERO, ONE});
        Value * const traceLogCount = b.CreateAlignedLoad(sizeTy, traceLogCountField, SizeTyABIAlignment);

        Value * const traceLogCapacityField = b.CreateGEP(ty, historyPtr, {ZERO, TWO});
        Value * const traceLogCapacity = b.CreateAlignedLoad(sizeTy, traceLogCapacityField, SizeTyABIAlignment);

        BasicBlock * const expandHistory = b.CreateBasicBlock(prefix + "_expandHistory", mKernelLoopCall);
        BasicBlock * const recordExpansion = b.CreateBasicBlock(prefix + "_recordExpansion", mKernelLoopCall);

        Value * const hasEnough = b.CreateICmpULT(traceLogCount, traceLogCapacity);

        BasicBlock * const entryBlock = b.GetInsertBlock();
        b.CreateLikelyCondBr(hasEnough, recordExpansion, expandHistory);

        b.SetInsertPoint(expandHistory);
        Constant * const SZ_ONE = b.getSize(1);
        Constant * const SZ_MIN_SIZE = b.getSize(32);
        Value * const newCapacity = b.CreateUMax(b.CreateShl(traceLogCapacity, SZ_ONE), SZ_MIN_SIZE);
        Value * const expandedLogArray = b.CreateRealloc(b.getSizeTy(), traceLogArray, newCapacity);
        assert (expandedLogArray->getType() == traceLogArray->getType());
        b.CreateAlignedStore(expandedLogArray, traceLogArrayField, PtrTyABIAlignment);
        b.CreateAlignedStore(newCapacity, traceLogCapacityField, SizeTyABIAlignment);
        BasicBlock * const branchExitBlock = b.GetInsertBlock();
        b.CreateBr(recordExpansion);

        b.SetInsertPoint(recordExpansion);
        PHINode * const logArray = b.CreatePHI(traceLogArray->getType(), 2);
        logArray->addIncoming(traceLogArray, entryBlock);
        logArray->addIncoming(expandedLogArray, branchExitBlock);
        b.CreateAlignedStore(b.CreateAdd(traceLogCount, b.getSize(1)), traceLogCountField, SizeTyABIAlignment);
        b.CreateAlignedStore(mSegNo, b.CreateGEP(sizeTy, logArray, traceLogCount), SizeTyABIAlignment);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalBlockingIOStatistics
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBlockingIOStatistics(KernelBuilder & b) {
    if (LLVM_UNLIKELY(codegen::StatisticsOptionIsSet(codegen::EnableBlockingIOCounter))) {

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
        maxKernelLength += 4;
        maxBindingLength += 4;

        SmallVector<char, 100> buffer;
        raw_svector_ostream line(buffer);

        line << "BLOCKING I/O STATISTICS:\n\n"
                "  # "  // kernel ID #  (only shown for first)
                 "KERNEL"; // kernel Name (only shown for first)
        line.indent(maxKernelLength - 6);
        line << "PORT";
        line.indent(5 + maxBindingLength - 4); // I/O Type (e.g., input port 3 = I3), Port Name
        line << " BUFFER " // buffer ID #
                "   BLOCKED "
                 "     %%\n"; // % of blocking attempts

        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(line.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Print each kernel line

        // generate first line format string
        buffer.clear();
        line << "%3" PRIu32 " " // kernel #
                "%-" << maxKernelLength << "s" // kernel name
                "%c%-3" PRIu32 " " // I/O type
                "%-" << maxBindingLength << "s" // port name
                "%7" PRIu32 // buffer ID #
                "%11" PRIu64 " " // # of blocked attempts
                "%6.2f\n"; // % of blocking attempts
        Value * const firstLine = b.GetString(line.str());

        // generate remaining lines format string
        buffer.clear();
        line.indent(maxKernelLength + 4); // spaces for kernel #, kernel name
        line << "%c%-3" PRIu32 " " // I/O type
                "%-" << maxBindingLength << "s" // port name
                "%7" PRIu32 // buffer ID #
                "%11" PRIu64 " " // # of blocked attempts
                "%6.2f\n"; // % of blocking attempts
        Value * const remainingLines = b.GetString(line.str());

        SmallVector<Value *, 8> args;

        Type * const doubleTy = b.getDoubleTy();
        Constant * const fOneHundred = ConstantFP::get(doubleTy, 100.0);

        Constant * const I = b.getInt8('I');
        Constant * const O = b.getInt8('O');

        for (auto i = FirstKernel; i <= LastKernel; ++i) {

            const auto prefix = makeKernelName(i);
            // total # of segments processed by kernel
            const auto & lockType = LOGICAL_SEGMENT_SUFFIX[isDataParallel(i) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL];
            Value * const totalNumberOfSegments = b.getScalarField(prefix + lockType);
            Value * const fTotalNumberOfSegments = b.CreateUIToFP(totalNumberOfSegments, doubleTy);

            const Kernel * const kernel = getKernel(i);
            Constant * kernelName = b.GetString(kernel->getName());

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {

                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIOSummary) {
                    args.push_back(STDERR);
                    if (kernelName) {
                        args.push_back(firstLine);
                        args.push_back(b.getInt32(i));
                        args.push_back(kernelName);
                        kernelName = nullptr;
                    } else {
                        args.push_back(remainingLines);
                    }
                    args.push_back(I);
                    const auto inputPort = binding.Port.Number;
                    args.push_back(b.getInt32(inputPort));
                    const Binding & ref = binding.Binding;

                    args.push_back(b.GetString(ref.getName()));

                    args.push_back(b.getInt32(source(e, mBufferGraph)));

                    const auto prefix = makeBufferName(i, binding.Port);
                    Value * const blockedCount = b.getScalarField(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
                    args.push_back(blockedCount);

                    Value * const fBlockedCount = b.CreateUIToFP(blockedCount, doubleTy);
                    Value * const fBlockedCount100 = b.CreateFMul(fBlockedCount, fOneHundred);
                    args.push_back(b.CreateFDiv(fBlockedCount100, fTotalNumberOfSegments));

                    b.CreateCall(fTy, Dprintf, args);

                    args.clear();
                }
            }

            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {

                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIOSummary) {
                    args.push_back(STDERR);
                    if (kernelName) {
                        args.push_back(firstLine);
                        args.push_back(b.getInt32(i));
                        args.push_back(kernelName);
                        kernelName = nullptr;
                    } else {
                        args.push_back(remainingLines);
                    }
                    args.push_back(O);
                    const auto outputPort = binding.Port.Number;
                    args.push_back(b.getInt32(outputPort));
                    const Binding & ref = binding.Binding;
                    args.push_back(b.GetString(ref.getName()));

                    args.push_back(b.getInt32(target(e, mBufferGraph)));

                    const auto prefix = makeBufferName(i, binding.Port);
                    Value * const blockedCount = b.getScalarField(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
                    args.push_back(blockedCount);

                    Value * const fBlockedCount = b.CreateUIToFP(blockedCount, doubleTy);
                    Value * const fBlockedCount100 = b.CreateFMul(fBlockedCount, fOneHundred);
                    args.push_back(b.CreateFDiv(fBlockedCount100, fTotalNumberOfSegments));

                    b.CreateCall(fTy, Dprintf, args);

                    args.clear();
                }
            }

        }
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBlockedIOPerSegment(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(codegen::StatisticsOptionIsSet(codegen::TraceBlockedIO))) {

        IntegerType * const sizeTy = b.getSizeTy();
        PointerType * const sizePtrTy = sizeTy->getPointerTo();

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);

        ConstantInt * const sz_ZERO = b.getSize(0);
        ConstantInt * const sz_ONE = b.getSize(1);


        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        Value * const maxSegNo = getMaxSegmentNumber(b);

        // Print the first title line
        SmallVector<char, 100> buffer;
        raw_svector_ostream format(buffer);
        format << "BLOCKED I/O PER SEGMENT:\n\n";
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            unsigned count = 0;
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    ++count;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    ++count;
                }
            }
            if (count) {
                const Kernel * const kernel = getKernel(i);
                std::string temp = kernel->getName();
                boost::replace_all(temp, "\"", "\\\"");
                format << ",\"" << i << " " << temp << "\"";
                for (unsigned j = 1; j < count; ++j) {
                    format.write(',');
                }
            }
        }
        format << "\n";

        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Print the second title line
        buffer.clear();
        format << "SEG #";
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    goto has_ports;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    goto has_ports;
                }
            }
            continue;
has_ports:
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    format << ",I" << binding.Port.Number << ':' << source(e, mBufferGraph);
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    format << ",O" << binding.Port.Number << ':' << source(e, mBufferGraph);
                }
            }
        }
        format << '\n';
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Generate line format string
        buffer.clear();
        format << "%" PRIu64; // seg #
        unsigned fieldCount = 0;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            unsigned count = 0;
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    ++count;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                if (binding.Flags & BufferPortType::TrackBlockedIO) {
                    ++count;
                }
            }
            fieldCount += count;
            for (unsigned j = 0; j < count; ++j) {
                format << ",%" PRIu64; // strides
            }
        }
        format << "\n";

        // Print each kernel line
//        SmallVector<Value *, 64> lastLineArgs(fieldCount + 3);
//        lastLineArgs[0] = STDERR;
//        lastLineArgs[1] = b.GetString(format.str());

//        SmallVector<Value *, 64> currentLineArgs(fieldCount + 3);
//        currentLineArgs[0] = STDERR;
//        currentLineArgs[1] = b.GetString(format.str());

        // Pre load all of pointers to the the trace log out of the pipeline state.

        SmallVector<Value *, 64> traceLogArray(fieldCount);
        SmallVector<Value *, 64> traceLengthArray(fieldCount);

        for (unsigned i = FirstKernel, j = 0; i <= LastKernel; ++i) {

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.Flags & BufferPortType::TrackBlockedIO) {
                    const auto prefix = makeBufferName(i, bp.Port);
                    Value * historyPtr; Type * historyTy;
                    std::tie(historyPtr, historyTy) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);
                    assert (j < fieldCount);
                    traceLogArray[j] = b.CreateAlignedLoad(sizePtrTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ZERO}), PtrTyABIAlignment);
                    traceLengthArray[j] = b.CreateAlignedLoad(sizeTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ONE}), SizeTyABIAlignment);
                    j++;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.Flags & BufferPortType::TrackBlockedIO) {
                    const auto prefix = makeBufferName(i, bp.Port);
                    Value * historyPtr; Type * historyTy;
                    std::tie(historyPtr, historyTy) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);
                    assert (j < fieldCount);
                    traceLogArray[j] = b.CreateAlignedLoad(sizePtrTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ZERO}), PtrTyABIAlignment);
                    traceLengthArray[j] = b.CreateAlignedLoad(sizeTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ONE}), SizeTyABIAlignment);
                    j++;
                }
            }
        }

        // Start printing the data lines

        // To only print out when the line changes, we store the values for the last line
        // we printed as well as its line number. If the last line was not printed,
        // we print it and the new line out; otherwise we only print out the new line.

        SmallVector<PHINode *, 64> currentIndex(fieldCount);
        SmallVector<Value *, 64> updatedIndex(fieldCount);
        SmallVector<PHINode *, 64> lastPrintedValue(fieldCount);

        BasicBlock * const loopEntry = b.GetInsertBlock();
        BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
        BasicBlock * const doWriteLine = b.CreateBasicBlock("writeLine");

        BasicBlock * const printLastLine = b.CreateBasicBlock("printLastLine");
        BasicBlock * const printCurrentLine = b.CreateBasicBlock("printCurrentLine");

        BasicBlock * const checkNext = b.CreateBasicBlock("checkNext");
        b.CreateBr(loopStart);

        b.SetInsertPoint(loopStart);
        PHINode * const segNo = b.CreatePHI(sizeTy, 2);
        segNo->addIncoming(sz_ZERO, loopEntry);
        PHINode * const lastPrintedSegNo = b.CreatePHI(sizeTy, 2);
        lastPrintedSegNo->addIncoming(sz_ZERO, loopEntry);

        for (unsigned i = 0; i < fieldCount; ++i) {
            PHINode * const index = b.CreatePHI(sizeTy, 2);
            index->addIncoming(sz_ZERO, loopEntry);
            currentIndex[i] = index;
            PHINode * const val = b.CreatePHI(sizeTy, 2);
            val->addIncoming(sz_ONE, loopEntry);
            lastPrintedValue[i] = val;
        }

        SmallVector<Value *, 64> currentVal(fieldCount + 3);

        Value * writeLine = b.CreateICmpEQ(segNo, maxSegNo);
        for (unsigned i = 0; i < fieldCount; ++i) {

            BasicBlock * const entry = b.GetInsertBlock();
            BasicBlock * const check = b.CreateBasicBlock();
            BasicBlock * const update = b.CreateBasicBlock();
            BasicBlock * const next = b.CreateBasicBlock();
            Value * const notEndOfTrace = b.CreateICmpNE(currentIndex[i], traceLengthArray[i]);
            b.CreateLikelyCondBr(notEndOfTrace, check, next);

            b.SetInsertPoint(check);
            Value * const nextSegNo = b.CreateAlignedLoad(sizeTy, b.CreateGEP(sizeTy, traceLogArray[i], currentIndex[i]), SizeTyABIAlignment);
            b.CreateCondBr(b.CreateICmpEQ(segNo, nextSegNo), update, next);

            b.SetInsertPoint(update);
            Value * const currentIndex1 = b.CreateAdd(currentIndex[i], sz_ONE);
            b.CreateBr(next);

            b.SetInsertPoint(next);
            PHINode * const nextIndex = b.CreatePHI(sizeTy, 3);
            nextIndex->addIncoming(currentIndex[i], entry);
            nextIndex->addIncoming(currentIndex[i], check);
            nextIndex->addIncoming(currentIndex1, update);
            updatedIndex[i] = nextIndex;

            PHINode * const val = b.CreatePHI(sizeTy, 3);
            val->addIncoming(sz_ZERO, entry);
            val->addIncoming(sz_ZERO, check);
            val->addIncoming(sz_ONE, update);

            currentVal[i] = val;

            lastPrintedValue[i]->addIncoming(val, checkNext);

            Value * const changed = b.CreateICmpNE(val, lastPrintedValue[i]);
            writeLine = b.CreateOr(writeLine, changed);
        }

        Value * const nextSegNo = b.CreateAdd(segNo, sz_ONE);

        BasicBlock * const blockedValueListExit = b.GetInsertBlock();
        b.CreateCondBr(writeLine, doWriteLine, checkNext);

        b.SetInsertPoint(doWriteLine);
        Value * const writeLast = b.CreateICmpNE(lastPrintedSegNo, segNo);
        b.CreateCondBr(writeLast, printLastLine, printCurrentLine);

        b.SetInsertPoint(printLastLine);
        SmallVector<Value *, 64> args(fieldCount + 3);
        args[0] = STDERR;
        args[1] = b.GetString(format.str());
        args[2] = b.CreateSub(segNo, sz_ONE);
        for (unsigned i = 0; i < fieldCount; ++i) {
            args[i + 3] = lastPrintedValue[i];
        }
        b.CreateCall(fTy, Dprintf, args);
        b.CreateBr(printCurrentLine);

        b.SetInsertPoint(printCurrentLine);
        args[2] = segNo;
        for (unsigned i = 0; i < fieldCount; ++i) {
            args[i + 3] = currentVal[i];
        }
        b.CreateCall(fTy, Dprintf, args);
        b.CreateBr(checkNext);

        b.SetInsertPoint(checkNext);
        BasicBlock * const loopExit = b.CreateBasicBlock("reportStridesPerSegmentExit");
        BasicBlock * const loopEnd = b.GetInsertBlock();
        for (unsigned i = 0; i < fieldCount; ++i) {
            currentIndex[i]->addIncoming(updatedIndex[i], loopEnd);
        }
        PHINode * const printedSegNo = b.CreatePHI(sizeTy, 2);
        printedSegNo->addIncoming(nextSegNo, printCurrentLine);
        printedSegNo->addIncoming(lastPrintedSegNo, blockedValueListExit);
        lastPrintedSegNo->addIncoming(printedSegNo, loopEnd);

        segNo->addIncoming(nextSegNo, loopEnd);

        Value * const notDone = b.CreateICmpNE(segNo, maxSegNo);
        b.CreateLikelyCondBr(notDone, loopStart, loopExit);

        b.SetInsertPoint(loopExit);
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
        for (unsigned i = 0; i < fieldCount; ++i) {
            b.CreateFree(traceLogArray[i]);
        }
    }
}


}
