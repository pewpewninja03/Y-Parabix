#include "../pipeline_compiler.hpp"

#define WRITE_POPCOUNT_VALUES_TO_STDERR

// TODO: add in assertions to prove whether all countable rate pipeline I/O was satisfied in the single iteration
// Is it sufficient to verify symbolic rate of the pipeline matches the rate of the I/O?

// TODO: if a popcount ref stream is zero extended, the current partial sum replacement would not work correctly
// since the equivalent for it would be to repeat the final number infinitely.

namespace kernel {


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForSufficientIO
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForSufficientIO(KernelBuilder & b) {

    // TODO: rework the loop structure to move an initial I/O check here and rely on the loopback only
    // to determine whether there is sufficient I/O for another subsegment

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determineNumOfLinearStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::determineNumOfLinearStrides(KernelBuilder & b) {

    // If this kernel does not have an explicit do final segment, then we want to know whether this stride will
    // be the final stride of the kernel. (i.e., that it will be flagged as terminated after executing the kernel
    // code.) For this to occur, at least one of the inputs must be closed and we must pass all of the data from
    // that closed input stream to the kernel. It is possible for a stream to be closed but to have more data
    // left to process (either due to some data being divided across a buffer boundary or because another stream
    // has less data (relatively speaking) than the closed stream.

    if (LLVM_UNLIKELY(TraceIO && mMayHaveInsufficientIO)) {
        mBranchToLoopExit = b.getFalse();
    }

    // If this kernel is the root of a partition, we'll use the available input to compute how many strides the
    // kernels within the partition will execute. Otherwise we begin by bounding the kernel by the expected number
    // of strides w.r.t. its partition's root.

    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[input];
        if (port.canModifySegmentLength()) {
            const auto streamSet = source(input, mBufferGraph);
            checkForSufficientInputData(b, port, streamSet);
        } else { // make sure we have read/initialized the accessible item count
            getAccessibleInputItems(b, port);
        }
    }



    // if we've exhausted a closed input source, reduce the stride step size to 1 to ensure
    // we correctly creep up on the final partial stride

    /// TODO: this logic doesn't correctly support circular buffers. To allow for them, we
    /// need to calculate the number of strides we could process and if we can perform a
    /// full stride step, allow it to single strides when necessary when the buffer does
    /// not support a full step before wrapping around. Resize the overflow appropriately.


    ConstantInt * const sz_ONE = b.getSize(1);
    if (mStrideStepSize) {
        assert (mHasExhaustedClosedInput);
        mStrideStepSize = b.CreateSelect(mHasExhaustedClosedInput, sz_ONE, mStrideStepSize);
    } else {
        mStrideStepSize = sz_ONE;
    }

    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[output];
        if (port.canModifySegmentLength()) {
            const auto streamSet = target(output, mBufferGraph);
            checkForSufficientOutputSpace(b, port, streamSet);
        }
    }

    const auto isSourceKernel = in_degree(mKernelId, mBufferGraph) == 0;

    Value * numOfLinearStrides = nullptr;

    if (isSourceKernel) {
        numOfLinearStrides = mMaximumNumOfStrides;
    } else if (!mIsPartitionRoot) {
        numOfLinearStrides = b.CreateSub(mMaximumNumOfStrides, mCurrentNumOfStridesAtLoopEntryPhi);
    }

    Value * numOfLinearStridesForConstants = nullptr;

    if (LLVM_LIKELY(hasAtLeastOneNonGreedyInput())) {
        for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & port = mBufferGraph[input];
            if (port.canModifySegmentLength()) {
                Value * const strides = getNumOfAccessibleStrides(b, port, numOfLinearStrides);
                const BufferNode & bn = mBufferGraph[source(input, mBufferGraph)];
                if (LLVM_UNLIKELY(bn.isConstant())) {
                    numOfLinearStridesForConstants = b.CreateUMin(numOfLinearStridesForConstants, strides);
                } else {
                    numOfLinearStrides = b.CreateUMin(numOfLinearStrides, strides);
                }
            }
        }
    } else if (!isSourceKernel) {
        Value * const exhausted = checkIfInputIsExhausted(b, InputExhaustionReturnType::Conjunction);
        numOfLinearStrides = b.CreateZExt(b.CreateNot(exhausted), b.getSizeTy());
    }

    mHasMoreInput = nullptr;
    if (LLVM_UNLIKELY(numOfLinearStridesForConstants)) {
        if (LLVM_LIKELY(numOfLinearStrides)) {
            mHasMoreInput = b.CreateICmpULT(numOfLinearStridesForConstants, numOfLinearStrides);
            numOfLinearStrides = b.CreateSelect(mHasMoreInput, numOfLinearStridesForConstants, numOfLinearStrides);
        } else {
            numOfLinearStrides = numOfLinearStridesForConstants;
        }
    }

    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[output];
        if (LLVM_UNLIKELY(port.canModifySegmentLength())) {
            Value * const strides = getNumOfWritableStrides(b, port, numOfLinearStrides);
            numOfLinearStrides = b.CreateUMin(numOfLinearStrides, strides);
        }
    }

    Value * const potentialNumOfLinearStrides = numOfLinearStrides;

    if (mIsPartitionRoot) {
        if (isSourceKernel) {
            mPotentialSegmentLength = mMaximumNumOfStrides;
        } else {
            mPotentialSegmentLength = b.CreateAdd(mCurrentNumOfStridesAtLoopEntryPhi, numOfLinearStrides);
            Value * remainingLinearStrides = b.CreateSub(mMaximumNumOfStrides, mCurrentNumOfStridesAtLoopEntryPhi);
            numOfLinearStrides = b.CreateUMin(numOfLinearStrides, remainingLinearStrides);
        }
    } else {
        mPotentialSegmentLength = nullptr;
    }

    if (LLVM_UNLIKELY(mIsOptimizationBranch)) {
        numOfLinearStrides = checkOptimizationBranchSpanLength(b, numOfLinearStrides);
    }

    mNumOfLinearStrides = calculateTransferableItemCounts(b, numOfLinearStrides, potentialNumOfLinearStrides);

    mUpdatedNumOfStrides = b.CreateAdd(mCurrentNumOfStridesAtLoopEntryPhi, mNumOfLinearStrides);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[e];
        if (!port.canModifySegmentLength()) {
            const auto streamSet = target(e, mBufferGraph);
            ensureSufficientOutputSpace(b, port, streamSet);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateTransferableItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::calculateTransferableItemCounts(KernelBuilder & b, Value * const numOfLinearStrides, Value * const potentialNumOfLinearStrides) {

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    // --- lambda function start
    auto phiOutItemCounts = [&](const Vec<Value *> & accessibleItems,
                               const Vec<Value *> & inputBufferCapacity,
                               const Vec<Value *> & inputVirtualBaseAddress,
                               const Vec<Value *> & writableItems,

                               Value * const fixedRateFactor,
                               Value * const terminationSignal,
                               Value * const numOfLinearStrides,
                               Value * const fixedRatePartialStrideRemainder,

                               Value * const isInPenultimateSubSegment,

                               const bool isFinalStride) {
        BasicBlock * const exitBlock = b.GetInsertBlock();
        for (auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {

            const auto & bp = mBufferGraph[input];
            const auto port = bp.Port;
            assert (mLinearInputItemsPhi[port] && accessibleItems[port.Number]);
            mLinearInputItemsPhi[port]->addIncoming(accessibleItems[port.Number], exitBlock);
            if (LLVM_UNLIKELY(mCheckStreamSets)) {
                mInputBufferCapacityPhi[port]->addIncoming(inputBufferCapacity[port.Number], exitBlock);
            }
            assert (mInputVirtualBaseAddressPhi[port] && inputVirtualBaseAddress[port.Number]);
            mInputVirtualBaseAddressPhi[port]->addIncoming(inputVirtualBaseAddress[port.Number], exitBlock);
            if (mExhaustedInputPortPhi[port]) {
                Value * exhausted = nullptr;
                if (isFinalStride) {
                    exhausted = mExhaustedInputPort[port]; assert (exhausted);
                } else {
                    exhausted = b.getFalse();
                }
                mExhaustedInputPortPhi[port]->addIncoming(exhausted, exitBlock);
            }
        }
        for (unsigned i = 0; i < numOfOutputs; ++i) {
            const auto port = StreamSetPort{ PortType::Output, i };
            assert (writableItems[i]);
            mLinearOutputItemsPhi[port]->addIncoming(writableItems[i], exitBlock);
        }
        if (mFixedRateFactorPhi) { assert (fixedRateFactor);
            mFixedRateFactorPhi->addIncoming(fixedRateFactor, exitBlock);
        }
        mIsFinalInvocationPhi->addIncoming(terminationSignal, exitBlock);
        mNumOfLinearStridesPhi->addIncoming(numOfLinearStrides, exitBlock);
        if (mIsPartitionRoot) {
            mFinalPartialStrideFixedRateRemainderPhi->addIncoming(fixedRatePartialStrideRemainder, exitBlock);
        }
        mPenultimateSubSegmentPhi->addIncoming(isInPenultimateSubSegment, exitBlock);
    };

    // --- lambda function end

    ConstantInt * const sz_ZERO = b.getSize(0);

    Constant * const i1_False = b.getFalse();

    Constant * const i1_True = b.getTrue();

    Vec<Value *> accessibleItems(numOfInputs);

    Vec<Value *> inputBufferCapacity(numOfInputs, nullptr);
    Vec<Value *> inputVirtualBaseAddress(numOfInputs, nullptr);

    Vec<Value *> writableItems(numOfOutputs);

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);

    getInputVirtualBaseAddresses(b, inputVirtualBaseAddress);

    Value * nonFinalNumOfLinearStrides = b.CreateRoundDown(numOfLinearStrides, mStrideStepSize);
    Value * inPenultimateSubSegment = i1_False;

    if (LLVM_LIKELY(in_degree(mKernelId, mBufferGraph) > 0)) {

        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const enteringFinalSegment = b.CreateBasicBlock(prefix + "_finalSegment", mKernelCheckOutputSpace);

        BasicBlock * const enteringNonFinalSegment = b.CreateBasicBlock(prefix + "_nonFinalSegment", mKernelCheckOutputSpace);

        /// -------------------------------------------------------------------------------------
        /// HANDLE ZERO EXTENSION
        /// -------------------------------------------------------------------------------------

        Value * isFinalSegment = nullptr;
        if (mIsPartitionRoot) {
            isFinalSegment = mAnyClosed ? mAnyClosed : i1_False;
        } else {
            isFinalSegment = mFinalPartitionSegment;
        }

        BasicBlock * const nonZeroExtendExit = b.GetInsertBlock();

        BasicBlock * afterNonFinalZeroExtendExit = nullptr;

        if (LLVM_UNLIKELY(mCheckStreamSets)) {
            for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
                const auto streamSet = source(input, mBufferGraph);
                assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                const BufferPort & port = mBufferGraph[input];
                const BufferNode & bn = mBufferGraph[streamSet];
                Value * max = nullptr;
                if (LLVM_UNLIKELY(bn.isConstant())) {
                    Value * pos = nullptr;
                    if (LLVM_UNLIKELY(port.isDeferred())) {
                        pos = mCurrentProcessedDeferredItemCountPhi[port.Port];
                    } else {
                        pos = mCurrentProcessedItemCountPhi[port.Port];
                    }
                    max = b.CreateAdd(pos, b.getSize(getGuaranteedRepeatingStreamSetLength(b, streamSet, true)));
                } else {
                    Value * pos = nullptr;
                    if (bn.isThreadLocal()) {
                        pos = mInitiallyProcessedItemCount[port.Port];
                    } else {
                        pos = readConsumedItemCount(b, streamSet);
                    }
                    max = b.CreateAdd(pos, bn.Buffer->getInternalCapacity(b));
                }
                inputBufferCapacity[port.Port.Number] = max;
            }
        }

        Vec<Value *> zeroExtendedInputVirtualBaseAddress(inputVirtualBaseAddress);

        Vec<Value *> zeroExtendedInputBufferCapacity(inputBufferCapacity);

        if (mHasZeroExtendedInput) {
            BasicBlock * const checkFinal = b.CreateBasicBlock(prefix + "_checkFinal", enteringNonFinalSegment);
            Value * const isFinalOrZeroExtended = b.CreateOr(mHasZeroExtendedInput, isFinalSegment);
            b.CreateUnlikelyCondBr(isFinalOrZeroExtended, checkFinal, enteringNonFinalSegment);

            b.SetInsertPoint(checkFinal);
            Value * const zeroExtendSpace = allocateLocalZeroExtensionSpace(b, zeroExtendedInputBufferCapacity, enteringNonFinalSegment);
            getZeroExtendedInputVirtualBaseAddresses(b, inputVirtualBaseAddress, zeroExtendSpace, zeroExtendedInputVirtualBaseAddress);
            afterNonFinalZeroExtendExit = b.GetInsertBlock();
        }

        b.CreateUnlikelyCondBr(isFinalSegment, enteringFinalSegment, enteringNonFinalSegment);

        /// -------------------------------------------------------------------------------------
        /// KERNEL ENTERING FINAL OR ZERO-EXTENDED SEGMENT
        /// -------------------------------------------------------------------------------------

        b.SetInsertPoint(enteringFinalSegment);
        // if we have a potentially zero-extended buffer, use that; otherwise select the normal buffer
        Vec<Value *> truncatedInputVirtualBaseAddress(zeroExtendedInputVirtualBaseAddress);

        Value * const isFinal = b.CreateICmpEQ(numOfLinearStrides, sz_ZERO);

        BasicBlock * const enteringPenultimateSubSegment = b.CreateBasicBlock(prefix + "_penultimateSubSegment", mKernelCheckOutputSpace);
        BasicBlock * const enteringFinalStride = b.CreateBasicBlock(prefix + "_finalStride", mKernelCheckOutputSpace);
        b.CreateUnlikelyCondBr(isFinal, enteringFinalStride, enteringPenultimateSubSegment);

        /// -------------------------------------------------------------------------------------
        /// KERNEL ENTERING FINAL STRIDE
        /// -------------------------------------------------------------------------------------

        b.SetInsertPoint(enteringFinalStride);
        Value * fixedItemFactor = nullptr;
        Value * partialPartitionStride = nullptr;
        calculateFinalItemCounts(b, accessibleItems, writableItems, fixedItemFactor, partialPartitionStride);
        Constant * const completed = getTerminationSignal(b, TerminationSignal::Completed);

        Vec<Value *> truncatedInputBufferCapacity(zeroExtendedInputBufferCapacity);
        zeroInputAfterFinalItemCount(b, accessibleItems, truncatedInputBufferCapacity, truncatedInputVirtualBaseAddress);
        phiOutItemCounts(accessibleItems, truncatedInputBufferCapacity, truncatedInputVirtualBaseAddress, writableItems,
                         fixedItemFactor, completed, sz_ZERO, partialPartitionStride, i1_False, true);
        b.CreateBr(mKernelCheckOutputSpace);

        /// -------------------------------------------------------------------------------------
        /// KERNEL IN FINAL SEGMENT
        /// -------------------------------------------------------------------------------------

        b.SetInsertPoint(enteringPenultimateSubSegment);
        Value * penultimateNumOfStrides = numOfLinearStrides;
        if (mIsPartitionRoot) {
            inPenultimateSubSegment = mHasZeroExtendedInput ? isFinalSegment : i1_True;
            if (mHasExhaustedClosedInput) {
                inPenultimateSubSegment = b.CreateAnd(inPenultimateSubSegment, mHasExhaustedClosedInput);
                if (numOfLinearStrides != nonFinalNumOfLinearStrides) {
                    penultimateNumOfStrides = b.CreateSelect(mHasExhaustedClosedInput, numOfLinearStrides, nonFinalNumOfLinearStrides);
                }
            }
            if (mIsPartitionRoot && (penultimateNumOfStrides != potentialNumOfLinearStrides)) {
                inPenultimateSubSegment = b.CreateAnd(b.CreateICmpEQ(penultimateNumOfStrides, potentialNumOfLinearStrides), inPenultimateSubSegment);
            }
        } else {
            inPenultimateSubSegment = isFinalSegment;
        }
        BasicBlock * const penultimateSegmentExit = b.GetInsertBlock();
        b.CreateBr(enteringNonFinalSegment);

        /// -------------------------------------------------------------------------------------
        /// KERNEL ENTERING NON-FINAL SEGMENT
        /// ------------------------------------------------------------------------------------

        b.SetInsertPoint(enteringNonFinalSegment);
        if (nonFinalNumOfLinearStrides != penultimateNumOfStrides) {
            PHINode * const nonFinalNumOfLinearStridesPhi = b.CreatePHI(numOfLinearStrides->getType(), 3);
            nonFinalNumOfLinearStridesPhi->addIncoming(nonFinalNumOfLinearStrides, nonZeroExtendExit);
            if (afterNonFinalZeroExtendExit) {
                nonFinalNumOfLinearStridesPhi->addIncoming(nonFinalNumOfLinearStrides, afterNonFinalZeroExtendExit);
            }
            nonFinalNumOfLinearStridesPhi->addIncoming(penultimateNumOfStrides, penultimateSegmentExit);
            nonFinalNumOfLinearStrides = nonFinalNumOfLinearStridesPhi;
        }

        PHINode * const inPenultimateSegmentPhi = b.CreatePHI(b.getInt1Ty(), 3);
        inPenultimateSegmentPhi->addIncoming(i1_False, nonZeroExtendExit);
        if (afterNonFinalZeroExtendExit) {
            inPenultimateSegmentPhi->addIncoming(i1_False, afterNonFinalZeroExtendExit);
        }
        inPenultimateSegmentPhi->addIncoming(inPenultimateSubSegment, penultimateSegmentExit);
        inPenultimateSubSegment = inPenultimateSegmentPhi;

        for (size_t port = 0; port < numOfInputs; ++port) {
            Value * const ba = inputVirtualBaseAddress[port];
            Value * const ze = zeroExtendedInputVirtualBaseAddress[port];
            if (LLVM_UNLIKELY(ba != ze)) {
                PHINode * const phi = b.CreatePHI(ba->getType(), 3, "vbaPhi");
                phi->addIncoming(ba, nonZeroExtendExit);
                if (afterNonFinalZeroExtendExit) {
                    phi->addIncoming(ze, afterNonFinalZeroExtendExit);
                }
                phi->addIncoming(ze, penultimateSegmentExit);
                inputVirtualBaseAddress[port] = phi;
            }
            if (LLVM_UNLIKELY(mCheckStreamSets)) {
                Value * const bc = inputBufferCapacity[port];
                Value * const zc = zeroExtendedInputBufferCapacity[port];
                if (LLVM_UNLIKELY(bc != zc)) {
                    PHINode * const phi = b.CreatePHI(b.getSizeTy(), 3, "bufferCapacityPhi");
                    phi->addIncoming(bc, nonZeroExtendExit);
                    if (afterNonFinalZeroExtendExit) {
                        phi->addIncoming(zc, afterNonFinalZeroExtendExit);
                    }
                     phi->addIncoming(zc, penultimateSegmentExit);
                    inputBufferCapacity[port] = phi;
                }
            }
        }

        #if defined(PRINT_DEBUG_MESSAGES) && !defined(PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY)
        for (unsigned i = 0; i != numOfInputs; ++i) {
            const auto prefix = makeBufferName(mKernelId, StreamSetPort{PortType::Input, i});
            debugPrint(b, prefix + "_inputVirtualBaseAddress = %" PRIx64, inputVirtualBaseAddress[i]);
        }
        #endif
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL CALCULATE NON-FINAL INPUT COUNT
    /// -------------------------------------------------------------------------------------

    Value * fixedRateFactor = nullptr;
    if (mFixedRateFactorPhi) {
        const Rational stride(mKernel->getStride());
        fixedRateFactor  = b.CreateMulRational(nonFinalNumOfLinearStrides, stride * mFixedRateLCM);
    } else {
        fixedRateFactor = sz_ZERO;
    }

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        accessibleItems[br.Port.Number] = calculateNumOfLinearItems(b, br, nonFinalNumOfLinearStrides, "calculateNonFinal");
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        writableItems[br.Port.Number] = calculateNumOfLinearItems(b, br, nonFinalNumOfLinearStrides, "calculateNonFinal");
    }

    phiOutItemCounts(accessibleItems, inputBufferCapacity, inputVirtualBaseAddress, writableItems,
                     fixedRateFactor, unterminated, nonFinalNumOfLinearStrides, sz_ZERO, inPenultimateSubSegment, false);

    b.CreateBr(mKernelCheckOutputSpace);
    b.SetInsertPoint(mKernelCheckOutputSpace);

    assert (mPenultimateSubSegmentPhi);
    if (mHasMoreInput) {
        mHasMoreInput = b.CreateOr(mHasMoreInput, mPenultimateSubSegmentPhi);
    } else {
        mHasMoreInput = mPenultimateSubSegmentPhi;
    }

    return mNumOfLinearStridesPhi;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForSufficientInputData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForSufficientInputData(KernelBuilder & b, const BufferPort & port, const unsigned streamSet) {

    const auto inputPort = port.Port;
    assert (inputPort.Type == PortType::Input);

    // Only the partition root dictates how many strides this kernel will end up doing. All other kernels
    // simply have to trust that the root determined the correct number or we'd be forced to have an
    // under/overflow capable of containing an entire segment rather than a single stride.

    Value * processed = mCurrentProcessedItemCountPhi[port.Port];

    Value * const strideLength = calculateStrideLength(b, port, processed, mStrideStepSize, "hasSufficient");

    const auto prefix = makeBufferName(mKernelId, inputPort);

    Value * const accessible = getAccessibleInputItems(b, port); assert (accessible);

    Value * closed = isClosed(b, streamSet); assert (closed->getType() == b.getInt1Ty());
    Value * minimum = strideLength;

    Value * const required = addLookahead(b, port, minimum); assert (required);

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, prefix + "_requiredInput (%" PRIu64 ") = %" PRIu64, b.getSize(streamSet), required);
    debugPrint(b, prefix + "_accessible (%" PRIu64 ") = %" PRIu64, b.getSize(streamSet), accessible);
    debugPrint(b, prefix + "_closed = %" PRIu8, b.CreateSelect(closed, b.getInt8(1), b.getInt8(0)));
    #endif

    Value * hasEnough = b.CreateICmpUGE(accessible, required);

    if (LLVM_LIKELY(mIsPartitionRoot)) { // && !port.isZeroExtended()
        if (mAnyClosed) {
            mAnyClosed = b.CreateOr(mAnyClosed, closed);
        } else {
            mAnyClosed = closed;
        }
    }

    Value * const isExhausted = b.CreateAnd(closed, b.CreateNot(hasEnough));

    if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
        if (LLVM_UNLIKELY(port.getRate().isGreedy())) {
            mExhaustedInputPort[inputPort] = closed;
        } else {
            mExhaustedInputPort[inputPort] = isExhausted;
        }
        assert (mExhaustedInputPort[inputPort]);
    }

    if (mHasExhaustedClosedInput) {
        mHasExhaustedClosedInput = b.CreateOr(mHasExhaustedClosedInput, isExhausted);
    } else {
        mHasExhaustedClosedInput = isExhausted;
    }


    Value * sufficientInput = b.CreateOr(hasEnough, closed);
    BasicBlock * const hasInputData = b.CreateBasicBlock(prefix + "_hasInputData", mKernelCheckOutputSpace);

    BasicBlock * recordBlockedIO = nullptr;
    BasicBlock * insufficentIO = mKernelInsufficientInput;
    assert (mKernelInsufficientInput);
    Value * hasEnoughOrIsClosed = sufficientInput;
    Value * insufficient = mBranchToLoopExit;


    if (LLVM_UNLIKELY(port.Flags & (BufferPortType::TrackBlockedIO | BufferPortType::TrackBlockedIOSummary))) {
        assert (mBufferGraph[streamSet].isNonThreadLocal());
        recordBlockedIO = b.CreateBasicBlock(prefix + "_recordBlockedIO", hasInputData);
        insufficentIO = recordBlockedIO;
        // do not record the block if this not the first execution of the
        // kernel but ensure that the system knows at least one failed.
        hasEnoughOrIsClosed = sufficientInput;
        if (mExecutedAtLeastOnceAtLoopEntryPhi) {
            hasEnoughOrIsClosed = b.CreateOr(hasEnoughOrIsClosed, mExecutedAtLeastOnceAtLoopEntryPhi);
        }
        insufficient = b.CreateOr(mBranchToLoopExit, b.CreateNot(sufficientInput));
    }

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, prefix + "_hasInputData = %" PRIu8, hasEnoughOrIsClosed);
    #endif

    b.CreateLikelyCondBr(hasEnoughOrIsClosed, hasInputData, insufficentIO);

    // When tracing blocking I/O, test all I/O streams but do not execute
    // the kernel if any stream is insufficient.
    if (LLVM_UNLIKELY(recordBlockedIO)) {
        BasicBlock * const entryBlock = b.GetInsertBlock();

        b.SetInsertPoint(recordBlockedIO);
        recordBlockingIO(b, port);
        BasicBlock * const exitBlock = b.GetInsertBlock();
        b.CreateBr(hasInputData);

        b.SetInsertPoint(hasInputData);
        IntegerType * const boolTy = b.getInt1Ty();

        PHINode * const anyInsufficient = b.CreatePHI(boolTy, 2, "anyInsufficient");
        anyInsufficient->addIncoming(insufficient, entryBlock);
        anyInsufficient->addIncoming(b.getTrue(), exitBlock);
        mBranchToLoopExit = anyInsufficient;
    }

    b.SetInsertPoint(hasInputData);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForSufficientOutputSpace
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForSufficientOutputSpace(KernelBuilder & b, const BufferPort & outputPort, const unsigned streamSet) {


    // Any buffer that is managed by a nested kernel will be a dynamic buffer (or otherwise capable
    // of producing the output for any given input.) Just ignore them.

    const BufferNode & bn = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(bn.isUnowned() || bn.isThreadLocal() || bn.hasZeroElementsOrWidth())) {
        assert (!bn.Buffer->isDynamic());
        return;
    }

    Value * const writable = getWritableOutputItems(b, outputPort);
    Value * const required = getOutputStrideLength(b, outputPort, "getSufficient");
    const auto prefix = makeBufferName(mKernelId, outputPort.Port);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, prefix + "_checkWritable = %" PRIu64, writable);
    debugPrint(b, prefix + "_checkRequired = %" PRIu64, required);
    #endif
    Value * hasEnough = b.CreateICmpULE(required, writable, prefix + "_hasEnough");

    if (LLVM_UNLIKELY(bn.isTruncated())) {
        const auto id = getTruncatedStreamSetSourceId(streamSet);
        Value * closed = isClosed(b, id);
        hasEnough = b.CreateOr(hasEnough, closed);
    }

    BasicBlock * const target = b.CreateBasicBlock(prefix + "_hasOutputSpace", mKernelLoopCall);
    assert (mKernelInsufficientInput);
    b.CreateCondBr(hasEnough, target, mKernelInsufficientInput);

    b.SetInsertPoint(target);

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkIfInputIsExhausted
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::checkIfInputIsExhausted(KernelBuilder & b, InputExhaustionReturnType returnValType) {
    if (LLVM_UNLIKELY(in_degree(mKernelId, mBufferGraph) == 0)) {
        if (mIsNestedPipeline) {
            return b.isFinal();
        } else {
            return b.getFalse();
        }
    }
    if (mIsPartitionRoot) {
        Value * resultVal = nullptr;
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const BufferPort & br =  mBufferGraph[e];
            if (LLVM_UNLIKELY(br.isZeroExtended())) {
                continue;
            }
            Value * const closed = isClosed(b, streamSet); assert (closed);
            if (resultVal) {
                resultVal = b.CreateAnd(resultVal, closed);
            } else {
                resultVal = closed;
            }
        }
        assert (resultVal && "non-zero-extended stream is required");
        return resultVal;
    } else {
        return mFinalPartitionSegment;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getAccessibleInputItems
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getAccessibleInputItems(KernelBuilder & b, const BufferPort & port) {

    const auto inputPort = port.Port;
    assert (inputPort.Type == PortType::Input);


    Value * const alreadyComputed = mInternalAccessibleInputItems[inputPort];
    if (alreadyComputed) {
        return alreadyComputed;
    }

    const auto input = getInput(mKernelId, inputPort);
    const auto streamSet = source(input, mBufferGraph);

    const BufferNode & bn = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(bn.isConstant())) {
        Value * const v = b.getSize(getGuaranteedRepeatingStreamSetLength(b, streamSet, false));
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, inputPort);
        debugPrint(b, prefix + "_available(const) = %" PRIu64, v);
        #endif
        return v;
    }

    Value * const processed = mCurrentProcessedItemCountPhi[inputPort];
    Value * const available = mLocallyAvailableItems[streamSet]; assert (available);
    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeBufferName(mKernelId, inputPort);
    debugPrint(b, prefix + "_available = %" PRIu64, available);
    debugPrint(b, prefix + "_processed (%" PRIu64 ") = %" PRIu64, b.getSize(streamSet), processed);
    #endif

    Value * accessible = b.CreateSub(available, processed);
    if (LLVM_UNLIKELY(port.Add != 0)) {
        Value * const closed = isClosed(b, streamSet);
        Value * ifClosed = b.CreateAdd(available, b.getSize(port.Add));
        Value * const addedItems = b.CreateSelect(closed, ifClosed, b.getSize(0));
        accessible = b.CreateAdd(accessible, addedItems);
    }

    #ifndef DISABLE_ZERO_EXTEND
    if (LLVM_UNLIKELY(port.isZeroExtended())) {
        // To zero-extend an input stream, we must first exhaust all input for this stream before
        // switching to a "zeroed buffer". The size of the buffer will be determined by the final
        // number of non-zero-extended strides.

        // NOTE: the producer of this stream will zero out all data after its last produced item
        // that can be read by a single iteration of any consuming kernel.

        Value * const deferred = mCurrentProcessedDeferredItemCountPhi[inputPort];
        Value * const itemCount = port.isDeferred() ? deferred : processed;
        Constant * const MAX_INT = ConstantInt::getAllOnesValue(b.getSizeTy());
        Value * const closed = isClosed(b, streamSet);
        Value * const exhausted = b.CreateICmpUGE(itemCount, available);
        Value * const useZeroExtend = b.CreateAnd(closed, exhausted);
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + "_useZeroExtend = %" PRIu8, useZeroExtend);
        #endif
        mIsInputZeroExtended[inputPort] = useZeroExtend;
        if (LLVM_LIKELY(mHasZeroExtendedInput == nullptr)) {
            mHasZeroExtendedInput = useZeroExtend;
        } else {
            mHasZeroExtendedInput = b.CreateOr(mHasZeroExtendedInput, useZeroExtend);
        }
        accessible = b.CreateSelect(useZeroExtend, MAX_INT, accessible);
    }
    #endif

    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & inputBinding = port.Binding;
        Value * valid = b.CreateOr(b.CreateICmpULE(processed, available), isClosed(b, streamSet));
        Value * const zeroExtended = mIsInputZeroExtended[inputPort];
        if (zeroExtended) {
            valid = b.CreateOr(valid, zeroExtended);
        }
        b.CreateAssert(valid,
                        "%s.%s: processed count (%" PRIu64 ") exceeds total count (%" PRIu64 ") @ %s",
                        mCurrentKernelName,
                        b.GetString(inputBinding.getName()),
                        processed, available,
                        b.GetString("getAccessibleInputItems"));
    }
    // cache the values for later use
    mInternalAccessibleInputItems[inputPort] = accessible;
    return accessible;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ensureSufficientOutputSpace
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::ensureSufficientOutputSpace(KernelBuilder & b, const BufferPort & port, const unsigned streamSet) {

    const BufferNode & bn = mBufferGraph[streamSet];

    if (bn.isThreadLocal() || bn.isUnowned() || bn.isInOutRedirect() || bn.isTruncated() || bn.hasZeroElementsOrWidth()) {
        return;
    }

    const auto outputPort = port.Port;
    assert (outputPort.Type == PortType::Output);
    const auto prefix = makeBufferName(mKernelId, outputPort);
    const StreamSetBuffer * const buffer = bn.Buffer;

    Value * const required = mLinearOutputItemsPhi[outputPort];

    BasicBlock * const expandBuffer = b.CreateBasicBlock(prefix + "_expandBuffer", mKernelLoopCall);
    BasicBlock * const expanded = b.CreateBasicBlock(prefix + "_resumeAfterPossiblyModifyingBuffer", mKernelLoopCall);
    Value * const beforeExpansion = getWritableOutputItems(b, port);

    Value * hasEnoughSpace = b.CreateICmpULE(required, beforeExpansion);

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, prefix + "_writable (%" PRIu64 ") = %" PRIu64, b.getSize(streamSet), beforeExpansion);
    debugPrint(b, prefix + "_required (%" PRIu64 ") = %" PRIu64, b.getSize(streamSet), required);
    debugPrint(b, prefix + "_hasEnoughSpace = %" PRIu64, hasEnoughSpace);
    #endif

    BasicBlock * const noExpansionExit = b.GetInsertBlock();
    b.CreateLikelyCondBr(hasEnoughSpace, expanded, expandBuffer);

    b.SetInsertPoint(expandBuffer);
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_BUFFER_EXPANSION, PAPIKernelCounter::PAPI_BUFFER_COPY});
    }
    #endif
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        startCycleCounter(b, {CycleCounter::BUFFER_EXPANSION, CycleCounter::BUFFER_COPY});
    }

    // If this kernel is statefree, we have a potential problem here. Another thread may be actively
    // executing this kernel and writing data but if we perform a copyback or expansion, we can't copy
    // its "unwritten" data. Thus we need to wait for the other thread to finish processing before
    // we can proceed.

    // TODO: can we determine which locks will always dominate another?

    if (LLVM_LIKELY(bn.LockId == 0)) {
        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            acquireSynchronizationLockWithTimingInstrumentation(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        }
    } else {
        assert (bn.LockId > mKernelId);
        const auto lockType = isDataParallel(bn.LockId) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLockWithTimingInstrumentation(b, bn.LockId, lockType, mSegNo);
    }

    Value * const produced = mCurrentProducedItemCountPhi[outputPort]; assert (produced);
    Value * const consumed = mKernelConsumedItemCount[outputPort]; assert (consumed);

    BasicBlock * const afterCopyBackOrExpand = b.CreateBasicBlock(prefix + "_afterCopyBackOrExpand", mKernelLoopCall);

    // TODO: have a delete immediately value when not multithreaded?
    Value * mustExpand = nullptr;
    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
        Value * sharedHandle = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
        mustExpand = buffer->reserveCapacity(b, produced, consumed, required, mBufferExpansionFunction, sharedHandle, b.getSize(outputPort.Number));
    } else {
        mustExpand = buffer->reserveCapacity(b, produced, consumed, required, nullptr, nullptr, nullptr);
    }
    b.CreateBr(afterCopyBackOrExpand);

    b.SetInsertPoint(afterCopyBackOrExpand);
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        if (mustExpand) {
            updateCycleCounter(b, mKernelId, mustExpand, CycleCounter::BUFFER_EXPANSION, CycleCounter::BUFFER_COPY);
        } else {
            updateCycleCounter(b, mKernelId, CycleCounter::BUFFER_EXPANSION);
        }
    }
    #ifdef ENABLE_PAPI
    if (NumOfPAPIEvents) {
        if (mustExpand) {
            accumPAPIMeasurementWithoutReset(b, mKernelId, mustExpand, PAPI_BUFFER_EXPANSION, PAPI_BUFFER_COPY);
        } else {
            accumPAPIMeasurementWithoutReset(b, mKernelId, PAPI_BUFFER_EXPANSION);
        }
    }
    #endif

    Value * const afterExpansion = getWritableOutputItems(b, port, true);
    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & output = getOutputBinding(outputPort);
        Value * const sanityCheck = b.CreateICmpULE(consumed, produced);
        b.CreateAssert(sanityCheck,
                        "%s.%s: required items (%" PRIu64 ") exceeds post-expansion writable items (%" PRIu64 ")",
                        mCurrentKernelName,
                        b.GetString(output.getName()),
                        required, afterExpansion);
    }

    #if defined(PRINT_DEBUG_MESSAGES)
    debugPrint(b, prefix + "_writable' = %" PRIu64, afterExpansion);
    debugPrint(b, prefix + "_capacity' = %" PRIu64, buffer->getCapacity(b));
    #endif

    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & output = getOutputBinding(outputPort);
        b.CreateAssert(b.CreateICmpUGT(afterExpansion, beforeExpansion),
                       "%s.%s was not expanded correctly?", mCurrentKernelName, b.GetString(output.getName()));
    }

    BasicBlock * const expandBufferExit = b.GetInsertBlock();
    b.CreateBr(expanded);

    b.SetInsertPoint(expanded);
    PHINode * const phi = b.CreatePHI(b.getSizeTy(), 2);
    phi->addIncoming(beforeExpansion, noExpansionExit);
    phi->addIncoming(afterExpansion, expandBufferExit);
    mInternalWritableOutputItems[outputPort] = phi;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getNumOfWritableStrides
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getNumOfWritableStrides(KernelBuilder & b,
                                                  const BufferPort & port,
                                                  Value * const numOfLinearStrides) {

    const auto outputPort = port.Port;
    assert (outputPort.Type == PortType::Output);
    const auto bufferVertex = getOutputBufferVertex(outputPort);
    const BufferNode & bn = mBufferGraph[bufferVertex];
    assert (bn.isOwned());
    const Binding & output = port.Binding;
    Value * numOfStrides = nullptr;
    if (LLVM_UNLIKELY(output.getRate().isPartialSum())) {
        numOfStrides = getMaximumNumOfPartialSumStrides(b, port, numOfLinearStrides);
    } else {
        Value * const writable = getWritableOutputItems(b, port);
        Value * const strideLength = getOutputStrideLength(b, port, "getWritable");
        numOfStrides = b.CreateUDiv(writable, strideLength);
    }
    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeBufferName(mKernelId, outputPort);
    debugPrint(b, "> " + prefix + "_numOfStrides = %" PRIu64, numOfStrides);
    #endif
    return numOfStrides;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getWritableOutputItems
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getWritableOutputItems(KernelBuilder & b, const BufferPort & port, const bool force) {

    const auto outputPort = port.Port;
    assert (outputPort.Type == PortType::Output);

    Value * const alreadyComputed = mInternalWritableOutputItems[outputPort];
    if (alreadyComputed && !force) {
        return alreadyComputed;
    }

    const auto output = getOutput(mKernelId, outputPort);
    const auto streamSet = target(output, mBufferGraph);
    const BufferNode & bn = mBufferGraph[streamSet];
    assert (bn.isNonThreadLocal());
    const StreamSetBuffer * const buffer = bn.Buffer;

    Value * const produced = mCurrentProducedItemCountPhi[outputPort]; assert (produced);

    const auto src = mConsumerGraph[streamSet];

    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeBufferName(mKernelId, outputPort);
    debugPrint(b, prefix + "_produced%" PRIu64 " = %" PRIu64, b.getSize(src), produced);
    #endif

    Value * writable = nullptr;


    if (LLVM_UNLIKELY(src == 0)) {
        writable = ConstantInt::getAllOnesValue(b.getSizeTy());
   } else  if (LLVM_UNLIKELY(bn.isTruncated() || bn.isInOutRedirect())) {
        Value * const avail = mLocallyAvailableItems[src]; assert (avail);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + "_avail%" PRIu64 " = %" PRIu64, b.getSize(src), avail);
        #endif


        if (LLVM_UNLIKELY(CheckAssertions())) {
            const Binding & output = getOutputBinding(outputPort);
            Value * const sanityCheck = b.CreateICmpULE(produced, avail);
            b.CreateAssert(sanityCheck,
                            "%s.%s: produced count (%" PRIu64 ") exceeds avail count (%" PRIu64 ")",
                            mCurrentKernelName,
                            b.GetString(output.getName()),
                            produced, avail);
        }

        writable = b.CreateUnsignedSaturatingSub(avail, produced);
    } else {

        assert (mConsumerGraph[src] != 0);

        Value * const consumed = mKernelConsumedItemCount[outputPort]; assert (consumed);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + "_consumed%" PRIu64 " = %" PRIu64, b.getSize(src), consumed);
        #endif


        if (LLVM_UNLIKELY(CheckAssertions())) {
            const Binding & output = getOutputBinding(outputPort);
            Value * const sanityCheck = b.CreateICmpULE(consumed, produced);
            b.CreateAssert(sanityCheck,
                            "%s.%s: consumed count (%" PRIu64 ") exceeds produced count (%" PRIu64 ")",
                            mCurrentKernelName,
                            b.GetString(output.getName()),
                            consumed, produced);
        }

        Value * p = produced;
        if (LLVM_UNLIKELY(port.EmptyOverflow > 0)) {
            p = b.CreateAdd(produced, b.getSize(port.EmptyOverflow));
        }
        Value * c = consumed;
        if (bn.hasNonFixedRateConsumer() || !port.isFixed()) {
            c = b.CreateRoundDownRational(c, b.getBitBlockWidth());
        }
        writable = buffer->getLinearlyWritableItems(b, p, c);
    }

    // cache the values for later use
    mInternalWritableOutputItems[outputPort] = writable;
    return writable;
}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getNumOfAccessibleStrides
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getNumOfAccessibleStrides(KernelBuilder & b,
                                                    const BufferPort & port,
                                                    Value * const numOfLinearStrides) {
    const auto inputPort = port.Port;
    assert (inputPort.Type == PortType::Input);
    const Binding & input = port.Binding;
    const ProcessingRate & rate = input.getRate();
    Value * numOfStrides = nullptr;
    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeBufferName(mKernelId, inputPort);
    #endif
    Value * const ze = mIsInputZeroExtended[inputPort];
    if (LLVM_UNLIKELY(rate.isPartialSum())) {
        numOfStrides = getMaximumNumOfPartialSumStrides(b, port, numOfLinearStrides);
        // TODO: does a zero-extended popcount make sense?
    } else if (LLVM_UNLIKELY(rate.isGreedy())) {
        return nullptr;
    } else {
        Value * const accessible = getAccessibleInputItems(b, port); assert (accessible);
        Value * const strideLength = getInputStrideLength(b, port, "getAccessible"); assert (strideLength);
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "< " + prefix + "_accessible = %" PRIu64, accessible);
        debugPrint(b, "< " + prefix + "_strideLength = %" PRIu64, strideLength);
        #endif
        numOfStrides = b.CreateUDiv(subtractLookahead(b, port, accessible), strideLength);
        if (ze) {
            Value * const potential = b.CreateCeilUDiv(accessible, strideLength);
            numOfStrides = b.CreateSelect(isClosed(b, port.Port), potential, numOfStrides);
        }
    }
    if (ze) {
        numOfStrides = b.CreateSelect(ze, numOfLinearStrides, numOfStrides, "numOfZeroExtendedStrides");
    }
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "< " + prefix + "_numOfStrides = %" PRIu64, numOfStrides);
    #endif
    return numOfStrides;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateFinalItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::calculateFinalItemCounts(KernelBuilder & b,
                                                Vec<Value *> & accessibleItems,
                                                Vec<Value *> & writableItems,
                                                Value *& minFixedRateFactor,
                                                Value *& finalStrideRemainder) {

    Value * principalFixedRateFactor = nullptr;
    size_t principalPartitionId = 0;
    for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[e];
        assert (port.Port.Type == PortType::Input);
        Value * accessible = getAccessibleInputItems(b, port);
        const int k = port.Add;
        if (LLVM_UNLIKELY(k != 0)) {
            Value * selected;
            if (LLVM_LIKELY(k > 0)) {
                selected = b.CreateAdd(accessible, b.getSize(k));
            } else  {
                selected = b.CreateUnsignedSaturatingSub(accessible, b.getSize(k));
            }
            Value * closed = isClosed(b, port.Port, true);
            accessible = b.CreateSelect(closed, selected, accessible, "accessible");
        }
        if (LLVM_UNLIKELY(port.isPrincipal())) {
            const Binding & input = port.Binding;
            const ProcessingRate & rate = input.getRate();
            assert (rate.isFixed());
            const auto factor = mFixedRateLCM / rate.getRate();
            principalFixedRateFactor = b.CreateMulRational(accessible, factor);
            const auto streamSet = source(e, mBufferGraph);
            assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
            const auto producer = parent(streamSet, mBufferGraph);
            assert (PipelineInput <= producer && producer < mKernelId);
            principalPartitionId = KernelPartitionId[producer];
        }
        accessibleItems[port.Port.Number] = accessible;
    }

    for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[e];
        assert (port.Port.Type == PortType::Input);
        const auto inputPort = port.Port;
        Value * accessible = accessibleItems[inputPort.Number];
        if (LLVM_UNLIKELY(mIsInputZeroExtended[inputPort] != nullptr)) {
            // If this input stream is zero extended, the current input items will be MAX_INT.
            // However, since we're now in the final stride, so we can bound the stream to:

            assert (!port.isPrincipal());

            if (principalFixedRateFactor && port.isFixed()) {
                const Binding & input = port.Binding;
                const ProcessingRate & rate = input.getRate();
                const auto factor = rate.getRate() / mFixedRateLCM;
                accessible = b.CreateCeilUMulRational(principalFixedRateFactor, factor);
            } else {
                Value * maxItems = b.CreateAdd(mCurrentProcessedItemCountPhi[inputPort], getInputStrideLength(b, port, "calculateFinal"));
                // But since we may not necessarily be in our zero extension region, we must first
                // test whether we are:
                accessible = b.CreateSelect(mIsInputZeroExtended[inputPort], maxItems, accessible);
            }
        }
        accessibleItems[inputPort.Number] = accessible;
    }

    if (principalFixedRateFactor) {
        minFixedRateFactor = principalFixedRateFactor;
    } else {
        for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & port = mBufferGraph[e];
            assert (port.Port.Type == PortType::Input);
            if (port.isFixed()) {
                const Binding & input = port.Binding;
                const ProcessingRate & rate = input.getRate();
                Value * const fixedRateFactor =
                    b.CreateMulRational(accessibleItems[port.Port.Number], mFixedRateLCM / rate.getRate());
                minFixedRateFactor =
                    b.CreateUMin(minFixedRateFactor, fixedRateFactor);
            }
        }
    }

    assert (minFixedRateFactor || mFixedRateFactorPhi == nullptr);

    if (minFixedRateFactor) {
        // truncate any fixed rate input down to the length of the shortest stream
        for (auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & port = mBufferGraph[e];
            if (port.isFixed()) {
                const auto inputPort = port.Port;
                assert (inputPort.Type == PortType::Input);
                const Binding & input = port.Binding;
                const ProcessingRate & rate = input.getRate();
                assert (rate.isFixed());
                const auto factor = rate.getRate() / mFixedRateLCM;
                Value * calculated = b.CreateCeilUMulRational(minFixedRateFactor, factor);
                const auto k = std::max(port.Add, port.LookAhead);

                // ... but ensure that it reflects whether it was produced with an
                // Add/Truncate attributed rate.
                if (k) {

                    const auto stride = mKernel->getStride();

                    // (x + (g/h)) * (c/d) = (xh + g) * c/hd
                    Constant * const h = b.getSize(stride);
                    Value * const xh = b.CreateMul(minFixedRateFactor, h);
                    Value * y;
                    if (k > 0) {
                        y = b.CreateAdd(xh, b.getSize(k));
                    } else {
                        y = b.CreateUnsignedSaturatingSub(xh, b.getSize(-k));
                    }
                    const Rational r{factor.numerator(), factor.denominator() * stride}; // := factor / Rational{stride};
                    Value * const z = b.CreateCeilUMulRational(y, r);
                    calculated = b.CreateSelect(isClosed(b, inputPort, true), z, calculated);
                }

                #if 1
                if (principalFixedRateFactor && !port.isPrincipal()) {
                    const auto streamSet = source(e, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const auto producer = parent(streamSet, mBufferGraph);
                    assert (PipelineInput <= producer && producer < mKernelId);
                    const auto inputPartitionId = KernelPartitionId[producer];
                    if (inputPartitionId != principalPartitionId && inputPartitionId != mCurrentPartitionId) {
                        calculated = b.CreateUMin(calculated, accessibleItems[inputPort.Number]);
                    }
                }
                #endif
                accessibleItems[inputPort.Number] = calculated;


                #ifdef PRINT_DEBUG_MESSAGES
                const auto prefix = makeBufferName(mKernelId, inputPort);
                debugPrint(b, prefix + ".accessible' = %" PRIu64, accessibleItems[inputPort.Number]);
                #endif
            }
        }
    }

    finalStrideRemainder = nullptr;
    if (LLVM_LIKELY(mIsPartitionRoot)) {
        if (minFixedRateFactor == nullptr) {
            finalStrideRemainder = b.getSize(0);
        } else {
            finalStrideRemainder = minFixedRateFactor;
        }
        assert (finalStrideRemainder);
    }

    Constant * const sz_ONE = b.getSize(1);

    for (auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[e];
        const auto outputPort = port.Port;
        assert (outputPort.Type == PortType::Output);
        const Binding & output = port.Binding;

        Value * writable = nullptr;
        if (port.isFixed() && minFixedRateFactor) {
            const ProcessingRate & rate = output.getRate();
            const auto factor = rate.getRate() / mFixedRateLCM;
            writable = b.CreateCeilUMulRational(minFixedRateFactor, factor);
        } else {
            writable = calculateNumOfLinearItems(b, port, sz_ONE, "calculateFinal");
        }
        const auto k = port.Add;
        if (k > 0) {
            writable = b.CreateAdd(writable, b.getSize(k));
        } else if (k < 0) {
            writable = b.CreateUnsignedSaturatingSub(writable, b.getSize(-k));
        }
        // update the final item counts with any Add/RoundUp attributes
        for (const Attribute & attr : output.getAttributes()) {
            switch (attr.getKind()) {
//                case AttrId::Add:
//                    writable = b.CreateAdd(writable, b.getSize(attr.amount()));
//                    break;
//                case AttrId::Truncate:
//                    writable = b.CreateUnsignedSaturatingSub(writable, b.getSize(attr.amount()));
//                    break;
                case AttrId::RoundUpTo:
                    writable = b.CreateRoundUp(writable, b.getSize(attr.amount()));
                    break;
                default: break;
            }
        }
        writableItems[outputPort.Number] = writable;
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, outputPort);
        debugPrint(b, prefix + ".writable' = %" PRIu64, writable);
        #endif
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief revertTransitiveAddCalculation
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::revertTransitiveAddCalculation(KernelBuilder & b, const ProcessingRate & rate, Value * expectedItemCount, Value * rejectedTerminationSignal) {

    assert (mFixedRateFactorPhi);
    assert (rate.isFixed());

    const auto factor = rate.getRate() / mFixedRateLCM;
    Value * calculated = b.CreateCeilUMulRational(mFixedRateFactorPhi, factor);
    return b.CreateSelect(rejectedTerminationSignal, calculated, expectedItemCount);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputStrideLength
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getInputStrideLength(KernelBuilder & b, const BufferPort & inputPort, const StringRef location) {
    if (mFirstInputStrideLength[inputPort.Port]) {
        return mFirstInputStrideLength[inputPort.Port];
    } else {
        Value * processed = mCurrentProcessedItemCountPhi[inputPort.Port];
        Value * const strideLength = calculateStrideLength(b, inputPort, processed, nullptr, location);
        mFirstInputStrideLength[inputPort.Port] = strideLength;
        return strideLength;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputStrideLength
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getOutputStrideLength(KernelBuilder & b, const BufferPort & outputPort, const StringRef location) {

    if (mFirstOutputStrideLength[outputPort.Port]) {
        return mFirstOutputStrideLength[outputPort.Port];
    } else {
        Value * produced = mCurrentProducedItemCountPhi[outputPort.Port];
        Value * const strideLength = calculateStrideLength(b, outputPort, produced, nullptr, location);
        mFirstOutputStrideLength[outputPort.Port] = strideLength;
        return strideLength;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPopCountStepSize
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getPopCountStepSize(const StreamSetPort inputRefPort) const {
    const auto streamSet = getInputBufferVertex(mKernelId, inputRefPort);
    const auto p = edge(streamSet, mKernelId, mPartialSumStepFactorGraph);
    assert (p.second);
    return mPartialSumStepFactorGraph[p.first];
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPartialSumItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getPartialSumItemCount(KernelBuilder & b, const BufferPort & partialSumPort, Value * const previouslyTransferred, Value * offset, StringRef location) const {
    const auto port = partialSumPort.Port;
    const StreamSetPort ref = getReference(mKernelId, port);
    assert (ref.Type == PortType::Input);
    assert (previouslyTransferred);

    ConstantInt * const sz_ZERO = b.getSize(0);
    Value * position = mCurrentProcessedItemCountPhi[ref];

    if (offset) {
        if (LLVM_UNLIKELY(CheckAssertions())) {
            const Binding & binding = partialSumPort.Binding;
            b.CreateAssert(b.CreateICmpNE(offset, sz_ZERO),
                            "%s.%s: partial sum offset must be non-zero",
                            mCurrentKernelName,
                            b.GetString(binding.getName()));
        }

        ConstantInt * const sz_ONE = b.getSize(1);

        const auto step = getPopCountStepSize(ref);

        if (step > 1) {
            ConstantInt * const sz_STEP = b.getSize(step);

            if (LLVM_UNLIKELY(CheckAssertions())) {
                const Binding & binding = partialSumPort.Binding;
                b.CreateAssert(b.CreateICmpEQ(b.CreateURem(position, sz_STEP), sz_ZERO),
                                "%s.%s: partial sum reference processed count must be a multiple of %" PRIu64,
                                mCurrentKernelName,
                                b.GetString(binding.getName()),
                                sz_STEP);
            }

            offset = b.CreateMul(offset, sz_STEP);
        }

        offset = b.CreateSub(offset, sz_ONE);
        position = b.CreateAdd(position, offset);

        if (LLVM_UNLIKELY(CheckAssertions())) {

            const auto streamSet = getInputBufferVertex(mKernelId, ref);
            Value * const total = mLocallyAvailableItems[streamSet];
            const Binding & binding = partialSumPort.Binding;
            Constant * bindingName = b.GetString(binding.getName());

            b.CreateAssert(b.CreateICmpULT(position, total),
                            "%s.%s: attempting to read a partial sum reference position that "
                            "exceeds its available items (%" PRIu64 " vs. %" PRIu64 " with offset %" PRIu64 ")",
                            mCurrentKernelName,
                            bindingName,
                            position, total, offset);



            Value * const alreadyProcessed = mInitiallyProcessedItemCount[ref];

            b.CreateAssert(b.CreateICmpUGE(position, alreadyProcessed),
                            "%s.%s: partial sum reference position is less than its initially processed item count (%" PRIu64 " vs. %" PRIu64 ")",
                            mCurrentKernelName,
                            bindingName,
                            position, alreadyProcessed);

        }

    }


    const auto srcStreamSet = getInputBufferVertex(mKernelId, ref);
    const auto & srcBufferNode = mBufferGraph[srcStreamSet];
    const StreamSetBuffer * const buffer = srcBufferNode.Buffer;

    Value * currentPtr = nullptr;

    if (buffer->isDynamic()) {
        IntegerType * const sizeTy = b.getSizeTy();
        PointerType * const ptrTy = sizeTy->getPointerTo(buffer->getAddressSpace());
        assert (srcBufferNode.ManagedStructId < ManagedBufferStructCount);
        Value * const addr = b.getScalarField(MANAGED_STREAMSET_LOCAL_VIRTUAL_BASE_ADDRESS + std::to_string(srcBufferNode.ManagedStructId));
        currentPtr = b.CreateGEP(sizeTy, b.CreatePointerCast(addr, ptrTy), position);
    } else {
        currentPtr = buffer->getRawItemPointer(b, sz_ZERO, position);
    }

    Value * current = b.CreateAlignedLoad(b.getSizeTy(), currentPtr, SizeTyABIAlignment);

//    const auto producer = parent(srcStreamSet, mBufferGraph);
//    if (LLVM_UNLIKELY(HasTerminationSignal.test(producer))) {
//        current = b.CreateUMax(current, mLocallyAvailableItems[srcStreamSet]);
//    }

    #if defined(PRINT_DEBUG_MESSAGES) && defined(WRITE_POPCOUNT_VALUES_TO_STDERR)
    #ifdef PRINT_DEBUG_MESSAGES_NO_ADDRESS_DISPLAY
    debugPrint(b, "%s.%s:  < pos[%" PRIu64 "] = %" PRIu64 "\n",
               mCurrentKernelName,
               b.GetString(partialSumPort.Binding.get().getName()),
               position, current);
    #else
    debugPrint(b, "%s.%s:  < pos[%" PRIu64 "] = %" PRIu64 " (0x%" PRIx64 ")\n",
               mCurrentKernelName,
               b.GetString(partialSumPort.Binding.get().getName()),
               position, current, currentPtr);
    #endif
    #endif

    if (LLVM_UNLIKELY(TraceIO && mMayHaveInsufficientIO)) {
        current = b.CreateSelect(mBranchToLoopExit, previouslyTransferred, current);
    }
    if (LLVM_UNLIKELY(CheckAssertions())) {

        const Binding & binding = partialSumPort.Binding;
        b.CreateAssert(b.CreateICmpULE(previouslyTransferred, current),
                        "%s.%s: partial sum is not non-decreasing at %" PRIu64
                        " (prior %" PRIu64 " > current %" PRIu64 ") @ %s",
                        mCurrentKernelName,
                        b.GetString(binding.getName()),
                        position, previouslyTransferred, current,
                        b.GetString(location));


    }
    return b.CreateSub(current, previouslyTransferred);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getMaximumNumOfPartialSumStrides
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getMaximumNumOfPartialSumStrides(KernelBuilder & b,
                                                           const BufferPort & partialSumPort,
                                                           Value * const numOfLinearStrides) {

    IntegerType * const sizeTy = b.getSizeTy();
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const MAX_INT = ConstantInt::getAllOnesValue(sizeTy);

    Value * initialItemCount = nullptr;
    Value * sourceItemCount = nullptr;
    Value * minimumItemCount = MAX_INT;

    const auto port = partialSumPort.Port;

    if (port.Type == PortType::Input) {
        initialItemCount = mCurrentProcessedItemCountPhi[port];
        Value * const accessible = getAccessibleInputItems(b, partialSumPort);
        sourceItemCount = b.CreateAdd(initialItemCount, accessible);
        minimumItemCount = getInputStrideLength(b, partialSumPort, "maxPartialSum");
        sourceItemCount = subtractLookahead(b, partialSumPort, sourceItemCount);
    } else { // if (port.Type == PortType::Output) {
        initialItemCount = mCurrentProducedItemCountPhi[port];
        Value * const writable = getWritableOutputItems(b, partialSumPort);
        sourceItemCount = b.CreateAdd(initialItemCount, writable);
        minimumItemCount = getOutputStrideLength(b, partialSumPort, "maxPartialSum");
    }

    const auto ref = getReference(port);
    assert (ref.Type == PortType::Input);
    const auto prefix = makeBufferName(mKernelId, ref) + "_readPartialSum";
    const auto refBufferVertex = getInputBufferVertex(ref);
    const auto & bn = mBufferGraph[refBufferVertex];
    const StreamSetBuffer * const popCountBuffer = bn.Buffer;

    Value * inputBaseAddr = nullptr;

    if (popCountBuffer->isDynamic()) {
        PointerType * const ptrTy = sizeTy->getPointerTo(popCountBuffer->getAddressSpace());
        assert (bn.ManagedStructId < ManagedBufferStructCount);
        inputBaseAddr = b.getScalarField(MANAGED_STREAMSET_LOCAL_VIRTUAL_BASE_ADDRESS + std::to_string(bn.ManagedStructId));
        inputBaseAddr = b.CreatePointerCast(inputBaseAddr, ptrTy);
    }

    BasicBlock * const popCountLoop =
        b.CreateBasicBlock(prefix + "Loop", mKernelCheckOutputSpace);
    BasicBlock * const popCountLoopExit =
        b.CreateBasicBlock(prefix + "LoopExit", mKernelCheckOutputSpace);

    BasicBlock * const popCountEntry = b.GetInsertBlock();

    assert (numOfLinearStrides);
    Value * cond = b.CreateICmpNE(numOfLinearStrides, sz_ZERO);
    b.CreateLikelyCondBr(cond, popCountLoop, popCountLoopExit);

    // TODO: replace this with a parallel icmp check and bitscan? binary search with initial
    // check on the rightmost entry?

    b.SetInsertPoint(popCountLoop);
    PHINode * const numOfStrides = b.CreatePHI(sizeTy, 2);
    numOfStrides->addIncoming(numOfLinearStrides, popCountEntry);
    PHINode * const nextRequiredItems = b.CreatePHI(sizeTy, 2);
    nextRequiredItems->addIncoming(MAX_INT, popCountEntry);


    Value * const strideIndex = b.CreateSub(numOfStrides, sz_ONE);

    Value * offset = strideIndex;

    // get the popcount kernel's input rate so we can calculate the
    // step factor for this kernel's usage of pop count partial sum
    // stream.
    const auto step = getPopCountStepSize(ref);
    if (LLVM_UNLIKELY(step > 1)) {
        offset = b.CreateSub(b.CreateMul(numOfStrides, b.getSize(step)), sz_ONE);
    }

    Value * const pos = b.CreateAdd(mCurrentProcessedItemCountPhi[ref], offset);

    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & binding = partialSumPort.Binding;
        Constant * bindingName = b.GetString(binding.getName());
        b.CreateAssert(b.CreateICmpUGE(pos, mInitiallyProcessedItemCount[ref]),
                        "%s.%s: partial sum reference offset (%" PRIu64 ") is less than the initial processed count (%" PRIu64 ")",
                        mCurrentKernelName, bindingName, pos, mInitiallyProcessedItemCount[ref]);
    }



    Value * ptr = nullptr;
    if (inputBaseAddr) {
        ptr = b.CreateGEP(sizeTy, inputBaseAddr, pos);
    } else {
        ptr = popCountBuffer->getRawItemPointer(b, sz_ZERO, pos);
    }
    Value * const requiredItems = b.CreateAlignedLoad(b.getSizeTy(), ptr, SizeTyABIAlignment);
    Value * const notEnough = b.CreateICmpUGT(requiredItems, sourceItemCount);

    Value * const notDone = b.CreateICmpNE(strideIndex, sz_ZERO);
    Value * const repeat = b.CreateAnd(notDone, notEnough);

    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & input = getInputBinding(ref);
        Value * const inputName = b.GetString(input.getName());
        b.CreateAssert(b.CreateICmpULE(requiredItems, nextRequiredItems),
                        "%s.%s: partial sum is not non-decreasing at %" PRIu64
                        " (prior %" PRIu64 " > current %" PRIu64 ") @ getMaximumNumOfPartialSumStrides",
                        mCurrentKernelName, inputName,
                        pos, requiredItems, nextRequiredItems);
    }

    nextRequiredItems->addIncoming(requiredItems, popCountLoop);
    numOfStrides->addIncoming(strideIndex, popCountLoop);
    b.CreateCondBr(repeat, popCountLoop, popCountLoopExit);

    b.SetInsertPoint(popCountLoopExit);
    PHINode * const numOfStridesPhi = b.CreatePHI(sizeTy, 2);
    numOfStridesPhi->addIncoming(sz_ZERO, popCountEntry);
    numOfStridesPhi->addIncoming(numOfStrides, popCountLoop);
    PHINode * const requiredItemsPhi = b.CreatePHI(sizeTy, 2);
    requiredItemsPhi->addIncoming(sz_ZERO, popCountEntry);
    requiredItemsPhi->addIncoming(requiredItems, popCountLoop);
    PHINode * const nextRequiredItemsPhi = b.CreatePHI(sizeTy, 2);
    nextRequiredItemsPhi->addIncoming(minimumItemCount, popCountEntry);
    nextRequiredItemsPhi->addIncoming(nextRequiredItems, popCountLoop);

    Value * finalNumOfStrides = numOfStridesPhi;
    if (LLVM_UNLIKELY(CheckAssertions())) {
        const Binding & binding = getInputBinding(ref);
        b.CreateAssert(b.CreateICmpNE(finalNumOfStrides, MAX_INT),
                        "%s.%s: attempting to use sentinal popcount entry",
                        mCurrentKernelName,
                        b.GetString(binding.getName()));
    }
    return finalNumOfStrides;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief splatMultiStepPartialSumValues
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::splatMultiStepPartialSumValues(KernelBuilder & b) {

    // For each PopCount rate, a PopCountKernel is created that generates a sequence of partial sum values.
    // These partial sum streams can be shared amongst multiple kernels whose reference is the same streamset
    // but when the stride length of the consuming kernel(s) differs from from the stride length of the
    // PopCountKernel, a step factor must be applied to read the correct value.

    // Upon termination of a PopCountKernel, we must ensure that every step factor will read the same
    // "final" value but since the PopCountKernel is unaware of the step factor, the pipeline becomes
    // responsible for splat-ing this value to the appropriate slots.

    if (LLVM_LIKELY(out_degree(mKernelId, mPartialSumStepFactorGraph) == 0)) {
        return;
    }

    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);

    const auto bw = b.getBitBlockWidth();
    const auto fw = b.getSizeTy()->getIntegerBitWidth();
    assert ((bw % fw) == 0 && bw >= fw);
    const auto stepsPerBlock = bw / fw;

    ConstantInt * const sz_stepsPerBlock = b.getSize(stepsPerBlock);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mPartialSumStepFactorGraph))) {

        const auto streamSet = target(e, mPartialSumStepFactorGraph);
        const auto output = in_edge(streamSet, mBufferGraph);
        const BufferPort & outputPort = mBufferGraph[output];

        const BufferNode & bn = mBufferGraph[streamSet];
        StreamSetBuffer * const buffer = bn.Buffer;
        VectorType * const vecTy = b.fwVectorType(fw);
        PointerType * const vecPtrTy = vecTy->getPointerTo();

        const auto spanLength = bn.PartialSumSpanLength;

        // If we produced no items but invoked the popcount kernel, it will have left the sum
        // in the stream[produced] position. This is the only safe position to read as we may
        // have executed a final block with 0 input items.

        // TODO: is this only necessary if its a partition root?
        Value * const initial = mInitiallyProducedItemCount[streamSet];
        Value * const produced = mProducedAtTermination[outputPort.Port];
        Value * const unchanged = b.CreateICmpEQ(produced, initial);
        Value * const index = b.CreateSelect(unchanged, produced, b.CreateSub(produced, sz_ONE));
        Value * const offset = b.CreateURem(index, sz_stepsPerBlock);
        Value * const start = b.CreateSub(index, offset);
        Value * const addr = buffer->getRawItemPointer(b, sz_ZERO, start);

        Value * const vecAddr = b.CreatePointerCast(addr, vecPtrTy);
        Value * const baseValue = b.CreateBlockAlignedLoad(vecTy, vecAddr);
        Value * const total = b.CreateExtractElement(baseValue, offset);

        Value * const splat = b.simd_fill(fw, total);
        Value * const mask = b.mvmd_sll(fw, ConstantInt::getAllOnesValue(vecTy), offset);
        Value * const maskedSplat = b.CreateAnd(splat, mask);
        Value * const mergedValue = b.CreateOr(baseValue, maskedSplat);
        b.CreateBlockAlignedStore(mergedValue, vecAddr);
        for (unsigned k = 1; k <= spanLength; ++k) {
            Value * const ptr = b.CreateGEP(vecTy, vecAddr, b.getSize(k));
            b.CreateBlockAlignedStore(splat, ptr);
        }

        ConstantInt * const sz_maxStepFactor = b.getSize(spanLength * stepsPerBlock);

        Value * const final = b.CreateRoundUp(b.CreateAdd(produced, sz_stepsPerBlock), sz_maxStepFactor);

        #if defined(PRINT_DEBUG_MESSAGES) && defined(WRITE_POPCOUNT_VALUES_TO_STDERR)
        debugPrint(b, "  < splatting %" PRIu64 " across %" PRIu64 " -> %" PRIu64 "\n", total, produced, final);
        #endif

        mProducedAtTermination[outputPort.Port] = final;
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateStrideLength
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::calculateStrideLength(KernelBuilder & b, const BufferPort & port, Value * const previouslyTransferred, Value * const strideIndex, StringRef location) {
    const Binding & binding = port.Binding;
    const ProcessingRate & rate = binding.getRate();
    if (LLVM_LIKELY(rate.isFixed() || rate.isBounded())) {
        const auto baseStrideLength = rate.getUpperBound() * mKernel->getStride();
        assert (baseStrideLength.denominator() == 1);
        ConstantInt * baseStrideVal = b.getSize(baseStrideLength.numerator());
        if (strideIndex == nullptr) {
            return baseStrideVal;
        } else {
            return b.CreateMul(strideIndex, baseStrideVal);
        }
    } else if (rate.isGreedy()) {
        assert ("kernel cannot have a greedy output rate" && port.Port.Type != PortType::Output);
        return b.getSize(ceiling(rate.getLowerBound()));
    } else if (rate.isPartialSum()) {
        return getPartialSumItemCount(b, port, previouslyTransferred, strideIndex, location);
    } else if (rate.isRelative()) {
        const auto refPort = getReference(port.Port);
        const auto refInput = getInput(mKernelId, refPort);
        const BufferPort & ref = mBufferGraph[refInput];
        Value * itemCount = b.CreateUDivRational(previouslyTransferred, rate.getRate());
        Value * const baseRate = calculateStrideLength(b, ref, itemCount, strideIndex, location);
        return b.CreateMulRational(baseRate, rate.getRate());
    }
    llvm_unreachable("unexpected rate type");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateNumOfLinearItems
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::calculateNumOfLinearItems(KernelBuilder & b, const BufferPort & port, Value * const linearStrides, StringRef location) {
    assert (linearStrides);
    const Binding & binding = port.Binding;
    const ProcessingRate & rate = binding.getRate();
    if (rate.isFixed() || rate.isBounded()) {
        return b.CreateMulRational(linearStrides, rate.getUpperBound() * mKernel->getStride());
    } else if (rate.isGreedy()) {
        return getAccessibleInputItems(b, port);
    } else if (rate.isPartialSum()) {
        Value * priorItemCount = nullptr;
        if (LLVM_LIKELY(port.Port.Type == PortType::Input)) {
            priorItemCount = mCurrentProcessedItemCountPhi[port.Port];
        } else {
            priorItemCount = mCurrentProducedItemCountPhi[port.Port];
        }
        return getPartialSumItemCount(b, port, priorItemCount, linearStrides, location);
    } else if (rate.isRelative()) {
        const auto refPort = getReference(port.Port);
        const auto refInput = getInput(mKernelId, refPort);
        const BufferPort & ref = mBufferGraph[refInput];
        return calculateNumOfLinearItems(b, ref, linearStrides, location);
    }
    llvm_unreachable("unexpected rate type");
}

}
