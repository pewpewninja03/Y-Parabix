/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <pablo/carry_manager.h>

#include <pablo/carry_data.h>
#include <pablo/codegenstate.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Transforms/Utils/Local.h>
#include <pablo/branch.h>
#include <pablo/pablo_intrinsic.h>
#include <pablo/pe_advance.h>
#include <pablo/pe_scanthru.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_var.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <array>

using namespace llvm;
using namespace IDISA;
using KernelBuilder = kernel::KernelBuilder;

namespace pablo {

inline static unsigned ceil_log2(const unsigned v) {
    assert ("log2(0) is undefined!" && v != 0);
    return (sizeof(unsigned) * CHAR_BIT) - __builtin_clz(v - 1U);
}

inline static unsigned floor_log2(const unsigned v) {
    assert ("log2(0) is undefined!" && v != 0);
    return ((sizeof(unsigned) * CHAR_BIT) - 1U) - __builtin_clz(v);
}

inline static unsigned nearest_pow2(const unsigned v) {
    assert (v > 0 && v < (UINT32_MAX / 2));
    return (v < 2) ? 1 : (1U << ceil_log2(v));
}

inline static unsigned udiv(const unsigned x, const unsigned y) {
    assert (is_power_2(y));
    const unsigned z = x >> floor_log2(y);
    assert (z == (x / y));
    return z;
}

inline static unsigned ceil_udiv(const unsigned x, const unsigned y) {
    return (((x - 1) | (y - 1)) + 1) / y;
}

static Value * castToSummaryType(KernelBuilder & b, Value * carryOut, Type * summaryTy) {
    if (!(carryOut->getType()->isIntegerTy() || carryOut->getType() == b.getBitBlockType())) {
        assert (false);
    }
    Type * carryOutTy = carryOut->getType();
    if (carryOutTy == summaryTy) {
        return carryOut;
    } else if (summaryTy == b.getBitBlockType()) {
        return b.CreateBitCast(b.CreateZExt(carryOut, b.getIntNTy(b.getBitBlockWidth())), b.getBitBlockType());
    } else if (carryOutTy->isIntegerTy() && summaryTy->isIntegerTy()) {
        return b.CreateZExt(carryOut, summaryTy);
    } else {
        return b.CreateBitCast(carryOut, summaryTy);
    }
}

using TypeId = PabloAST::ClassTypeId;

inline static bool isNonAdvanceCarryGeneratingStatement(const Statement * const stmt) {
    if (IntrinsicCall const * call = dyn_cast<IntrinsicCall>(stmt)) {
        return call->isCarryProducing() && !call->isAdvanceType();
    } else {
        return isa<CarryProducingStatement>(stmt) && !isa<Advance>(stmt) && !isa<IndexedAdvance>(stmt);
    }
}

enum NonCarryCollapsingMode {
    NestedCapacity = 0,
    LastIncomingCarryLoopIteration = 1,
    NestedCarryState = 2
};

#define LONG_ADVANCE_BREAKPOINT 64

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeCarryData
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::initializeCarryData(kernel::KernelBuilder & b, PabloKernel * const kernel) {

    // Each scope constructs its own CarryData struct, which will be added to the final "carries" struct
    // that is added to the Kernel. The scope index will indicate which struct to access.

    // A CarryData struct either contains an array of CarryPackBlocks or an integer indicating the capacity of
    // the variable length CarryData struct and pointer. A variable length CarryData struct is required whenever
    // the streams accessed by a loop could vary between iterations. When resizing a CarryData struct for a
    // particular loop, the current loop struct and all nested structs need to be resized. This accommodates
    // the fact every pablo While loop must be executed at least once.

    // A nested loop may also contain a variable length CarryData struct

    // To determine whether we require a variable length CarryData struct, we test the escaped variables of
    // each loop branch to see whether they are used as the index parameter of a nested Extract statement.
    // Any scope that requires variable length CarryData, requires that all nested branches have a unique
    // set of carries for that iteration.

    assert (mKernel == nullptr);
    mKernel = kernel;

    mCarryScopes = 0;

    PabloBlock * const entryScope = kernel->getEntryScope();
    mCarryMetadata.resize(getScopeCount(entryScope));

    mCarryFrameType = analyse(b, entryScope);


    if (LLVM_UNLIKELY(mCarryFrameType->isEmptyTy())) {
        mCarryFrameType = nullptr;
    } else {
        assert (b.getTypeSize(b.getModule()->getDataLayout(), mCarryFrameType) > 0);
        kernel->addInternalScalar(mCarryFrameType, "carries");
    }

    if (mHasLoop) {
        kernel->addInternalScalar(b.getInt32Ty(), "selector");
    }
    if (mHasLongAdvance) {
        kernel->addInternalScalar(b.getInt32Ty(), "CarryBlockIndex");
    }
    for (unsigned i = 0; i < mIndexedLongAdvanceTotal; i++) {
        kernel->addInternalScalar(b.getSizeTy(), "IndexedAdvancePosition" + std::to_string(i));
    }
}

bool isDynamicallyAllocatedType(const Type * const ty) {
    if (isa<StructType>(ty) && ty->getStructNumElements() == 3) {
        return (ty->getStructElementType(NestedCarryState)->isPointerTy() && ty->getStructElementType(LastIncomingCarryLoopIteration)->isIntegerTy() && ty->getStructElementType(NestedCapacity)->isIntegerTy());
    }
    return false;
}

bool containsDynamicallyAllocatedType(const Type * const ty) {
    if (isa<StructType>(ty)) {
        for (unsigned i = 0; i < ty->getStructNumElements(); ++i) {
            if (isDynamicallyAllocatedType(ty->getStructElementType(i))) {
                return true;
            }
        }
    }
    return false;
}

void freeDynamicallyAllocatedMemory(KernelBuilder & b, StructType * frameTy, Value * const frame) {
#if 0
    FixedArray<Value *, 3> indices;
    indices[0] = b.getInt32(0);
    ConstantInt * const intNestedCarryState = b.getInt32(NestedCarryState);
    ConstantInt * const intNestedCapacity = b.getInt32(NestedCapacity);
    for (unsigned i = 0; i < frameTy->getStructNumElements(); ++i) {
        Type * nestedTy = frameTy->getStructElementType(i);
        if (isDynamicallyAllocatedType(nestedTy)) {
            indices[1] = b.getInt32(i);
            indices[2] = intNestedCarryState;

            Value * const innerFrame = b.CreateLoad(b.CreateGEP(frameTy, frame, indices));
            if (containsDynamicallyAllocatedType(innerFrame->getType())) {
                indices[2] = intNestedCapacity;
                Value *  const count = b.CreateLoad(b.CreateGEP(frameTy, frame, indices));
                BasicBlock * const entry = b.GetInsertBlock();
                BasicBlock * const cond = b.CreateBasicBlock("freeCarryDataCond");
                BasicBlock * const body = b.CreateBasicBlock("freeCarryDataLoop");
                BasicBlock * const exit = b.CreateBasicBlock("freeCarryDataExit");
                b.CreateBr(cond);

                b.SetInsertPoint(cond);
                PHINode * const index = b.CreatePHI(count->getType(), 2);
                index->addIncoming(ConstantInt::getNullValue(count->getType()), entry);
                b.CreateCondBr(b.CreateICmpNE(index, count), body, exit);

                b.SetInsertPoint(body);
                freeDynamicallyAllocatedMemory(b, frameTy->getStructElementType(i),  b.CreateGEP(frameTy, innerFrame, index));
                index->addIncoming(b.CreateAdd(index, ConstantInt::get(count->getType(), 1)), body);
                b.CreateBr(cond);

                b.SetInsertPoint(exit);
            }
            b.CreateFree(innerFrame);
        }
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseCarryData
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::releaseCarryData(kernel::KernelBuilder & b) {
    if (mHasNonCarryCollapsingLoops) {
        freeDynamicallyAllocatedMemory(b, mCarryFrameType, b.getScalarFieldPtr("carries").first);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearCarryState
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::clearCarryData(kernel::KernelBuilder & idb) {



}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeCodeGen
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::initializeCodeGen(kernel::KernelBuilder & b) {

    assert(!mCarryMetadata.empty());
    mCarryInfo = &mCarryMetadata[0];
    assert (!mCarryInfo->hasSummary());

    if (LLVM_UNLIKELY(mCarryFrameType == nullptr)) {
        return;
    }
    mCurrentFrame = b.getScalarFieldPtr("carries").first;
    mCurrentFrameIndex = 0;
    mCarryScopes = 0;
    mCurrentFrameType = mCarryFrameType;
    mCarryScopeIndex.push_back(0);

    assert (mCarryFrameStack.empty());
    assert (mCarrySummaryStack.empty());

    Type * const carryTy = b.getBitBlockType();
    mCarrySummaryStack.push_back(Constant::getNullValue(carryTy));
    if (mHasLoop) {
        mLoopSelector = b.getScalarField("selector");
        mNextLoopSelector = b.CreateXor(mLoopSelector, ConstantInt::get(mLoopSelector->getType(), 1));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief finalizeCodeGen
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::finalizeCodeGen(kernel::KernelBuilder & b) {
    if (mHasLoop) {
        b.setScalarField("selector", mNextLoopSelector);
    }
    if (mHasLongAdvance) {
        Value * idx = b.getScalarField("CarryBlockIndex");
        idx = b.CreateAdd(idx, b.getInt32(1));
        b.setScalarField("CarryBlockIndex", idx);
    }
    assert (mNestedLoopCarryInMaskPhi == nullptr);
    assert (mCarryFrameStack.empty());
    assert ("base summary value was deleted!" && (mCarryFrameType == nullptr || mCarrySummaryStack.size() >= 1));
    assert ("not all summaries were used!" && (mCarryFrameType == nullptr|| mCarrySummaryStack.size() == 1));
    assert ("base summary value was overwritten with non-zero value!" && (mCarryFrameType == nullptr || (isa<Constant>(mCarrySummaryStack[0]) && cast<Constant>(mCarrySummaryStack[0])->isNullValue())));
    mCarrySummaryStack.clear();
    assert (mCarryFrameType == nullptr || mCarryScopeIndex.size() == 1);
    mCarryScopeIndex.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterLoopScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::enterLoopScope(kernel::KernelBuilder & b) {
    assert (mHasLoop);
    ++mLoopDepth;
    enterScope(b);
    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(mCurrentFrameIndex);
        indices[1] = b.getInt32(LastIncomingCarryLoopIteration);
        Value * const lastIncomingCarryIterationPtr = b.CreateGEP(mCurrentFrameType, mCurrentFrame, indices);
        Value * const lastIncomingCarryIteration = b.CreateLoad(b.getSizeTy(), lastIncomingCarryIterationPtr);
        assert (mCurrentFrameType->getStructNumElements() == 3);
        mNonCarryCollapsingModeStack.emplace_back(cast<StructType>(mCurrentFrameType), mCurrentFrame, mCurrentFrameIndex, lastIncomingCarryIteration);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSummaryTest
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::generateEntrySummaryTest(kernel::KernelBuilder & b, Value * condition) {
    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {
        if (LLVM_LIKELY(condition->getType() == b.getBitBlockType())) {
            condition = b.bitblock_any(condition);
        }
        const NonCarryCollapsingFrame & frame = mNonCarryCollapsingModeStack.back();
        condition = b.CreateOr(condition, b.CreateIsNotNull(frame.LastIncomingCarryIteration));
    } else if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        assert ("summary condition cannot be null!" && condition);
        assert ("summary test was not generated" && mNextSummaryTest);
        assert ("summary condition and test must have the same context!" && &condition->getContext() == &mNextSummaryTest->getContext());
        if (mNextSummaryTest->getType() == condition->getType()) {
            condition = b.CreateOr(condition, mNextSummaryTest);
        } else if (mNextSummaryTest->getType()->canLosslesslyBitCastTo(condition->getType())) {
            condition = b.CreateOr(condition, b.CreateBitCast(mNextSummaryTest, condition->getType()));
        } else {
            mNextSummaryTest = castToSummaryType(b, mNextSummaryTest, condition->getType());
            condition = b.CreateOr(condition, mNextSummaryTest);
        }
        mNextSummaryTest = nullptr;
    }
    if (LLVM_LIKELY(condition->getType() == b.getBitBlockType())) {
        condition = b.bitblock_any(condition);
    }
    assert ("summary test was not consumed" && (mNextSummaryTest == nullptr));
    return condition;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterLoopBody
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::enterLoopBody(kernel::KernelBuilder & b, BasicBlock * const entryBlock) {
    assert (mLoopDepth > 0);
    Type * const carryTy = getSummaryTypeFromCurrentFrame(b);
    if (mLoopDepth == 1) {
        Constant * const ONES = Constant::getAllOnesValue(carryTy);
        mNestedLoopCarryInMaskPhi = b.CreatePHI(carryTy, 2, "loopCarryInMask");
        mNestedLoopCarryInMaskPhi->addIncoming(ONES, entryBlock);
    }

    if (mCarryInfo->hasSummary()) {
        Constant * const ZEROES = Constant::getNullValue(carryTy);
        PHINode * const phiCarryOutSummary = b.CreatePHI(carryTy, 2, "whileCarryOutSummary");
        phiCarryOutSummary->addIncoming(ZEROES, entryBlock);

        // Replace the incoming carry summary with the phi node and add the phi node to the stack  so that we can
        // properly OR it into the outgoing summary value.
        // NOTE: this may change the base summary value; when exiting to the base scope, replace this summary with
        // a null value to prevent subsequent nested scopes from inheriting the summary of this scope.

        // (1) Carry-ins: (a) incoming carry data first iterations, (b) zero thereafter
        // (2) Carry-out accumulators: (a) zero first iteration, (b) |= carry-out of each iteration
        mCarrySummaryStack.push_back(phiCarryOutSummary); // original carry out summary phi
        if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {
            mCarrySummaryStack.push_back(ZEROES); // accumulated carry out summary value
        } else {
            mCarrySummaryStack.push_back(phiCarryOutSummary); // accumulated carry out summary value
        }
    }

    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {

        assert (mCarryInfo->getNestedCarryStateType());

        DataLayout DL(b.getModule());

        IntegerType * const sizeTy = b.getSizeTy();

        ConstantInt * const ZERO = b.getSize(0);

        NonCarryCollapsingFrame & frame = mNonCarryCollapsingModeStack.back();

        // Check whether we need to resize the carry state
        PHINode * const lastNonZeroCarryOutIterationPhi = b.CreatePHI(sizeTy, 2);
        lastNonZeroCarryOutIterationPhi->addIncoming(ZERO, entryBlock);
        frame.LastNonZeroIteration = lastNonZeroCarryOutIterationPhi;

        PHINode * const indexPhi = b.CreatePHI(sizeTy, 2);
        indexPhi->addIncoming(ZERO, entryBlock);
        frame.LoopIterationPhi = indexPhi;

        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(frame.NestedFrameIndex);
        indices[1] = b.getInt32(NestedCapacity);
        assert (frame.OuterFrameType->getStructNumElements() == 3);
        assert (frame.OuterFrameType->getStructElementType(NestedCapacity) == sizeTy);
        Value * const capacityPtr = b.CreateGEP(frame.OuterFrameType, frame.OuterFrame, indices);
        Value * const capacity = b.CreateLoad(sizeTy, capacityPtr);

        BasicBlock * const entry = b.GetInsertBlock();
        BasicBlock * const resizeCarryState = b.CreateBasicBlock("ResizeCarryState");
        BasicBlock * const reallocExisting = b.CreateBasicBlock("ReallocExisting");
        BasicBlock * const createNew = b.CreateBasicBlock("CreateNew");
        BasicBlock * const resumeKernel = b.CreateBasicBlock("ResumeKernel");

        indices[1] = b.getInt32(NestedCarryState);
        Value * const carryStateArrayPtr = b.CreateGEP(frame.OuterFrameType, frame.OuterFrame, indices);
        StructType * nestedCarryTy = mCarryInfo->getNestedCarryStateType(); assert (nestedCarryTy);
        Type * nestedCarryPtrTy = frame.OuterFrameType->getStructElementType(NestedCarryState);

        assert (nestedCarryPtrTy->isPointerTy());
        Value * const carryStateArray = b.CreateLoad(nestedCarryPtrTy, carryStateArrayPtr);
        assert (carryStateArray->getType()->isPointerTy());

        b.CreateLikelyCondBr(b.CreateICmpNE(indexPhi, capacity), resumeKernel, resizeCarryState);

        // RESIZE CARRY BLOCK
        b.SetInsertPoint(resizeCarryState);
        const auto blockSize = b.getBitBlockWidth() / 8;
        Constant * const carryStateTySize = b.getTypeSize(nestedCarryTy);
        b.CreateLikelyCondBr(b.CreateICmpNE(capacity, ZERO), reallocExisting, createNew);

        // REALLOCATE EXISTING
        b.SetInsertPoint(reallocExisting);
        Value * const newCapacity = b.CreateShl(capacity, 1);
        b.CreateStore(newCapacity, capacityPtr);
        Value * const capacitySize = b.CreateMul(capacity, carryStateTySize);
        Value * const newCapacitySize = b.CreateShl(capacitySize, 1); // x 2
        Value * newCarryStateArray = b.CreatePageAlignedMalloc(newCapacitySize);
        b.CreateMemCpy(newCarryStateArray, carryStateArray, capacitySize, b.getCacheAlignment());
        b.CreateFree(carryStateArray);
        Value * const startNewArrayPtr = b.CreateGEP(b.getInt8Ty(), b.CreatePointerCast(newCarryStateArray, b.getInt8PtrTy()), capacitySize);
        b.CreateMemZero(startNewArrayPtr, capacitySize, blockSize);
        newCarryStateArray = b.CreatePointerCast(newCarryStateArray, nestedCarryPtrTy);
        b.CreateStore(newCarryStateArray, carryStateArrayPtr);
        b.CreateBr(resumeKernel);

        // CREATE NEW
        b.SetInsertPoint(createNew);
        Constant * const initialCarryStateCapacity = b.getSize(8); // 2^3
        b.CreateStore(initialCarryStateCapacity, capacityPtr);

        Constant * const initialCapacitySize = ConstantExpr::getMul(initialCarryStateCapacity, carryStateTySize);
        Value * initialArray = b.CreatePageAlignedMalloc(initialCapacitySize);
        b.CreateMemZero(initialArray, initialCapacitySize, blockSize);
        initialArray = b.CreatePointerCast(initialArray, nestedCarryPtrTy);
        b.CreateStore(initialArray, carryStateArrayPtr);
        b.CreateBr(resumeKernel);

        // RESUME KERNEL
        b.SetInsertPoint(resumeKernel);
        PHINode * const updatedCarryStateArrayPhi = b.CreatePHI(nestedCarryPtrTy, 3, "carryStatePtr");
        updatedCarryStateArrayPhi->addIncoming(carryStateArray, entry);
        updatedCarryStateArrayPhi->addIncoming(initialArray, createNew);
        updatedCarryStateArrayPhi->addIncoming(newCarryStateArray, reallocExisting);

        // NOTE: the 3 here is only to pass the assertion later. It refers to the number of elements in the carry data struct.
        assert (mCurrentFrameType->getStructNumElements() == 3);
        mCarryFrameStack.emplace_back(mCurrentFrame, mCurrentFrameType, 3);

        mCurrentFrame = b.CreateGEP(nestedCarryTy, updatedCarryStateArrayPhi, indexPhi);
        assert (mCurrentFrameType->getStructElementType(NestedCarryState)->isPointerTy());
        assert (nestedCarryTy->getPointerTo() == mCurrentFrameType->getStructElementType(NestedCarryState));
        mCurrentFrameType = nestedCarryTy;
        mCurrentFrameIndex = 0;

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief leaveLoopBody
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::leaveLoopBody(kernel::KernelBuilder & b) {
    assert (mLoopDepth > 0);
    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {

        assert (mCarryInfo->hasSummary());

        Value * const carryOutAccumulator = mCarrySummaryStack.back();

        const CarryFrame & outer = mCarryFrameStack.back();
        mCurrentFrame = outer.Frame;
        mCurrentFrameType = cast<StructType>(outer.Type);
        mCurrentFrameIndex = 3;
        assert (outer.Index == 3);
        mCarryFrameStack.pop_back();

        BasicBlock * const exitBlock = b.GetInsertBlock();

        NonCarryCollapsingFrame & nestingFrame = mNonCarryCollapsingModeStack.back();

        PHINode * const loopIndexPhi = nestingFrame.LoopIterationPhi;
        Value * const nextLoopIndex = b.CreateAdd(loopIndexPhi, b.getSize(1));
        loopIndexPhi->addIncoming(nextLoopIndex, exitBlock);

        PHINode * const lastNonZeroCarryOutIterationPhi = cast<PHINode>(nestingFrame.LastNonZeroIteration);
        Value * const anyCarryOut = b.bitblock_any(carryOutAccumulator);
        Value * const nonZeroCarryOutIteration = b.CreateSelect(anyCarryOut, nextLoopIndex, lastNonZeroCarryOutIterationPhi, "nonZeroCarryOut");
        lastNonZeroCarryOutIterationPhi->addIncoming(nonZeroCarryOutIteration, exitBlock);

        Value * const lastIncomingCarryIteration = nestingFrame.LastIncomingCarryIteration;
        assert (lastIncomingCarryIteration->getType()->isIntegerTy());
        mNextSummaryTest = b.CreateICmpULT(loopIndexPhi, lastIncomingCarryIteration);

        nestingFrame.LastNonZeroIteration = nonZeroCarryOutIteration;
    }

    if (mLoopDepth == 1) {
        Constant * const ZEROES = Constant::getNullValue(mNestedLoopCarryInMaskPhi->getType());
        mNestedLoopCarryInMaskPhi->addIncoming(ZEROES, b.GetInsertBlock());
    }

    if (mCarryInfo->hasSummary()) {
        const auto n = mCarrySummaryStack.size(); assert (n > 2);

        // (1) Carry-ins: (a) incoming carry data first iterations, (b) zero thereafter
        // (2) Carry-out accumulators: (a) zero first iteration, (b) |= carry-out of each iteration

        Value * const carryOut = mCarrySummaryStack[n - 1];
        PHINode * const phiCarryOut = cast<PHINode>(mCarrySummaryStack[n - 2]);
        phiCarryOut->addIncoming(carryOut, b.GetInsertBlock());
        mCarrySummaryStack[n - 2] = carryOut; // replace summary out phi with the final value
        mCarrySummaryStack.pop_back(); // discard updated carry out value
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSummaryTest
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::generateExitSummaryTest(kernel::KernelBuilder & b, Value * condition) {
    if (LLVM_LIKELY(condition->getType() == b.getBitBlockType())) {
        condition = b.bitblock_any(condition);
    }
    if (LLVM_UNLIKELY(mCarryInfo && mCarryInfo->nonCarryCollapsingMode())) {
        assert ("summary condition cannot be null!" && condition);
        assert ("summary test was not generated" && mNextSummaryTest);
        assert ("summary condition and test must have the same context!" && &condition->getContext() == &mNextSummaryTest->getContext());
        condition = b.CreateOr(condition, mNextSummaryTest);
        mNextSummaryTest = nullptr;
    }
    assert ("summary test was not consumed" && (mNextSummaryTest == nullptr));
    return condition;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief leaveLoopScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::leaveLoopScope(kernel::KernelBuilder & b, BasicBlock * const entryBlock, BasicBlock * const exitBlock) {
    PHINode * nonZeroCarryOutPhi = nullptr;
    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {
        nonZeroCarryOutPhi = b.CreatePHI(b.getSizeTy(), 2);
        nonZeroCarryOutPhi->addIncoming(b.getSize(0), entryBlock);
        const NonCarryCollapsingFrame & nestingFrame = mNonCarryCollapsingModeStack.back();
        Value * const nonZeroCarryOutIteration = nestingFrame.LastNonZeroIteration;
        assert (nonZeroCarryOutIteration->getType()->isIntegerTy());
        nonZeroCarryOutPhi->addIncoming(nonZeroCarryOutIteration, exitBlock);
    }
    phiCurrentCarryOutSummary(b, entryBlock, exitBlock);
    if (LLVM_LIKELY(!mCarryInfo->nonCarryCollapsingMode())) {
        writeCurrentCarryOutSummary(b);
    }
    combineCarryOutSummary(b, 2);
    if (LLVM_UNLIKELY(mCarryInfo->nonCarryCollapsingMode())) {
        const NonCarryCollapsingFrame & nestingFrame = mNonCarryCollapsingModeStack.back();
        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(nestingFrame.NestedFrameIndex);
        indices[1] = b.getInt32(LastIncomingCarryLoopIteration);
        Value * const lastIncomingCarryIterationPtr = b.CreateGEP(nestingFrame.OuterFrameType, nestingFrame.OuterFrame, indices);
        assert (lastIncomingCarryIterationPtr->getType()->isPointerTy());
        b.CreateStore(nonZeroCarryOutPhi, lastIncomingCarryIterationPtr);
        mNonCarryCollapsingModeStack.pop_back();
    }
    leaveScope();
    assert (mLoopDepth > 0);
    assert (mCarrySummaryStack.size() > 0);
    --mLoopDepth;
    if (mLoopDepth == 0) {
        mNestedLoopCarryInMaskPhi = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterIfScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::enterIfScope(kernel::KernelBuilder & b) {
    ++mIfDepth;
    enterScope(b);
    // We zero-initialized the nested summary value and later OR in the current summary into the escaping summary
    // so that upon processing the subsequent block iteration, we branch into this If scope iff a carry out was
    // generated by a statement within this If scope and not by a dominating statement in the outer scope.
    if (mCarryInfo->hasSummary()) {
        assert (mNextSummaryTest);
        Type * const summaryTy = getSummaryTypeFromCurrentFrame(b);
        mCarrySummaryStack.push_back(Constant::getNullValue(summaryTy)); // new carry out summary accumulator
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterIfBody
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::enterIfBody(kernel::KernelBuilder & /* b */, BasicBlock * const /* entryBlock */) {
    assert (mIfDepth > 0);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief leaveIfBody
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::leaveIfBody(kernel::KernelBuilder & b, BasicBlock * const exitBlock) {
    assert (mIfDepth > 0);
    assert (exitBlock);
    writeCurrentCarryOutSummary(b);
    combineCarryOutSummary(b, 1);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief leaveIfScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::leaveIfScope(kernel::KernelBuilder & b, BasicBlock * const entryBlock, BasicBlock * const exitBlock) {
    assert (mIfDepth > 0);
    --mIfDepth;
    phiOuterCarryOutSummary(b, entryBlock, exitBlock);
    leaveScope();
}

/** ------------------------------------------------------------------------------------------------------------ *
 * @brief enterScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::enterScope(kernel::KernelBuilder & b) {
    if (LLVM_UNLIKELY(mCarryFrameType == nullptr)) {
        return;
    }
    // Store the state of the current frame and update the scope state
    mCarryFrameStack.emplace_back(mCurrentFrame, mCurrentFrameType, mCurrentFrameIndex + 1);
    mCarryScopeIndex.push_back(++mCarryScopes);
    mCarryInfo = &mCarryMetadata[mCarryScopes];
    // Check whether we're still within our struct bounds; if this fails, either the Pablo program changed during
    // compilation or a memory corruption has occured.
    assert (mCurrentFrameIndex < mCurrentFrameType->getStructNumElements());
    mCurrentFrame = b.CreateGEP(mCurrentFrameType, mCurrentFrame, {b.getInt32(0), b.getInt32(mCurrentFrameIndex)});
    mCurrentFrameType = cast<StructType>(mCurrentFrameType->getStructElementType(mCurrentFrameIndex));
    // Verify we're pointing to a carry frame struct
    mCurrentFrameIndex = 0;
    mNextSummaryTest = nullptr;
    if (LLVM_LIKELY(mCarryInfo->hasSummary() && !mCarryInfo->nonCarryCollapsingMode())) {
        mNextSummaryTest = readCarryInSummary(b);
        if (mCarryInfo->hasExplicitSummary()) {
            mCurrentFrameIndex = 1;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief leaveScope
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::leaveScope() {
    if (LLVM_UNLIKELY(mCarryFrameType == nullptr)) {
        return;
    }
    // Sanity test: are there remaining carry frames?
    assert (!mCarryFrameStack.empty());
    const CarryFrame & outer = mCarryFrameStack.back();
    mCurrentFrame = outer.Frame;
    mCurrentFrameType = cast<StructType>(outer.Type);
    mCurrentFrameIndex = outer.Index;

    mCarryFrameStack.pop_back();
    mCarryScopeIndex.pop_back();
    assert (!mCarryScopeIndex.empty());
    if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        mCarrySummaryStack.pop_back();
    }
    mCarryInfo = &mCarryMetadata[mCarryScopeIndex.back()];
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief combineCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::combineCarryOutSummary(kernel::KernelBuilder & b, const unsigned offset) {
    if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        const auto n = mCarrySummaryStack.size(); assert (n > 0);
        // combine the outer summary with the nested summary so that when
        // we leave the scope, we'll properly phi out the value of the new
        // outer summary
        if (n > 2) {
            Value * const nested = mCarrySummaryStack[n - 1];
            Value * const outer = mCarrySummaryStack[n - 2];
            assert (nested->getType() == outer->getType());
            mCarrySummaryStack[n - offset] =  b.CreateOr(outer, nested);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeCurrentCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::writeCurrentCarryOutSummary(kernel::KernelBuilder & b) {
    if (LLVM_LIKELY(mCarryInfo->hasExplicitSummary())) {
        const auto n = mCarrySummaryStack.size(); assert (n > 0);
        writeCarryOutSummary(b, mCarrySummaryStack[n - 1]);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiCurrentCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
inline void CarryManager::phiCurrentCarryOutSummary(kernel::KernelBuilder & b, BasicBlock * const entryBlock, BasicBlock * const exitBlock) {
    if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        const auto n = mCarrySummaryStack.size(); assert (n > 0);
        Value * const nested = mCarrySummaryStack[n - 1];
        Type * const nestedTy = nested->getType();
        Constant * const ZEROES = Constant::getNullValue(nestedTy);
        PHINode * const phi = b.CreatePHI(nestedTy, 2, "summary");
        phi->addIncoming(ZEROES, entryBlock);
        phi->addIncoming(nested, exitBlock);
        mCarrySummaryStack[n - 1] = phi;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOuterCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
inline void CarryManager::phiOuterCarryOutSummary(kernel::KernelBuilder & b, BasicBlock * const entryBlock, BasicBlock * const exitBlock) {
    if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        const auto n = mCarrySummaryStack.size(); assert (n > 0);
        if (n > 2) {
            // When leaving a nested If scope with a summary value, phi out the summary to ensure the
            // appropriate summary is stored in the outer scope.
            Value * nested = mCarrySummaryStack[n - 1];
            Value * outer = mCarrySummaryStack[n - 2];
            if (nested->getType() != outer->getType()) {
                if (outer->getType() == b.getBitBlockType()) {
                    nested = b.bitCast(b.CreateZExt(nested, b.getIntNTy(b.getBitBlockWidth())));
                } else {
                    nested = b.CreateZExt(nested, outer->getType());
                }
            }
            PHINode * const phi = b.CreatePHI(nested->getType(), 2, "summary");
            phi->addIncoming(outer, entryBlock);
            phi->addIncoming(nested, exitBlock);
            mCarrySummaryStack[n - 2] = phi;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::addCarryInCarryOut(kernel::KernelBuilder & b, const Statement * const operation, Value * const e1, Value * const e2) {
    assert (operation && (isNonAdvanceCarryGeneratingStatement(operation)));
    Value * const carryIn = getNextCarryIn(b);
    Value * carryOut, * result;
    std::tie(carryOut, result) = b.bitblock_add_with_carry(e1, e2, carryIn);
    assert (carryIn->getType() == carryOut->getType());
    setNextCarryOut(b, carryOut);
    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief subBorrowInBorrowOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::subBorrowInBorrowOut(kernel::KernelBuilder & b, const Statement * operation, Value * const e1, Value * const e2) {
    assert (operation);
    Value * const borrowIn = getNextCarryIn(b);
    Value * borrowOut, * result;
    std::tie(borrowOut, result) = b.bitblock_subtract_with_borrow(e1, e2, borrowIn);
    assert (borrowIn->getType() == borrowOut->getType());
    setNextCarryOut(b, borrowOut);
    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief advanceCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::advanceCarryInCarryOut(kernel::KernelBuilder & b, const Advance * const advance, Value * const value) {
    const auto shiftAmount = advance->getAmount();
    if (LLVM_LIKELY(shiftAmount < LONG_ADVANCE_BREAKPOINT)) {
        Value * const carryIn = getNextCarryIn(b);
        Value * carryOut, * result;
        std::tie(carryOut, result) = b.bitblock_advance(value, carryIn, shiftAmount);
        setNextCarryOut(b, carryOut);
        return result;
    } else {
        return longAdvanceCarryInCarryOut(b, value, shiftAmount);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief indexedAdvanceCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::indexedAdvanceCarryInCarryOut(kernel::KernelBuilder & b, const IndexedAdvance * const advance, Value * const strm, Value * const index_strm) {
    const auto shiftAmount = advance->getAmount();
    if (LLVM_LIKELY(shiftAmount < LONG_ADVANCE_BREAKPOINT)) {
        return shortIndexedAdvanceCarryInCarryOut(b, shiftAmount, strm, index_strm);
    } else if (shiftAmount <= b.getBitBlockWidth()) {
        FixedArray<Value *, 3> indices;
        Constant * const ZERO = b.getInt32(0);
        indices[0] = ZERO;
        indices[1] = b.getInt32(mCurrentFrameIndex);
        indices[2] = ZERO;
        Value * carryPtr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
        Type * carryTy = mCurrentFrameType->getStructElementType(mCurrentFrameIndex)->getArrayElementType();
        Value * carryIn = b.CreateLoad(carryTy, carryPtr);

        mCurrentFrameIndex++;

        Value * carryOut, * result;
        std::tie(carryOut, result) = b.bitblock_indexed_advance(strm, index_strm, carryIn, shiftAmount);
        b.CreateStore(carryOut, carryPtr);
        if (mCarryInfo->hasExplicitSummary()) {
            addToCarryOutSummary(b, strm);
        }
        return result;
    } else {
        const auto summaryFrame = mCurrentFrameIndex;
        if (mIfDepth > 0 || mLoopDepth > 0) {
            // Skip over summary frame to perform the long indexed advance.
            mCurrentFrameIndex++;
        }
        Type * iBitBlock = b.getIntNTy(b.getBitBlockWidth());
        Constant * blockWidth = b.getSize(b.getBitBlockWidth());
        Constant * blockWidth_1 = b.getSize(b.getBitBlockWidth() - 1);
        Value * carryPosition = b.getScalarField("IndexedAdvancePosition" + std::to_string(mIndexedLongAdvanceIndex));
        Value * carryBlockEndPos = b.CreateAdd(carryPosition, blockWidth_1);
        unsigned carry_blocks = nearest_pow2(1+ceil_udiv(shiftAmount, b.getBitBlockWidth()));
        Constant * carryQueueBlocks = b.getSize(carry_blocks);
        Value * carryBlock = b.CreateTrunc(b.CreateURem(b.CreateUDiv(carryPosition, blockWidth), carryQueueBlocks), b.getInt32Ty());
        Value * carryEndBlock = b.CreateTrunc(b.CreateURem(b.CreateUDiv(carryBlockEndPos, blockWidth), carryQueueBlocks), b.getInt32Ty());
        Constant * const ZERO = b.getInt32(0);
        Constant * const CFI = b.getInt32(mCurrentFrameIndex);
        Value * lo_GEP = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, {ZERO, CFI, carryBlock});
        Value * hi_GEP = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, {ZERO, CFI, carryEndBlock});
        Value * c_lo = b.CreateBitCast(b.CreateLoad(b.getBitBlockType(), lo_GEP), iBitBlock);
        Value * c_hi = b.CreateBitCast(b.CreateLoad(b.getBitBlockType(), hi_GEP), iBitBlock);
        Value * lo_shift = b.CreateZExt(b.CreateURem(carryPosition, blockWidth), iBitBlock);
        Value * hi_shift = b.CreateZExt(b.CreateSub(blockWidth_1, b.CreateURem(carryBlockEndPos, blockWidth)), iBitBlock);
        Value * carryIn = b.CreateOr(b.CreateLShr(c_lo, lo_shift), b.CreateShl(c_hi, hi_shift));
        Value * carryOut, * result;
        std::tie(carryOut, result) = b.bitblock_indexed_advance(strm, index_strm, carryIn, shiftAmount);
        carryOut = b.CreateBitCast(carryOut, iBitBlock);
        Value * adv = b.mvmd_extract(sizeof(size_t) * 8, b.simd_popcount(b.getBitBlockWidth(), index_strm), 0);
        b.setScalarField("IndexedAdvancePosition" + std::to_string(mIndexedLongAdvanceIndex), b.CreateAdd(carryPosition, adv));
        Value * carryOutPosition = b.CreateAdd(carryPosition, b.getSize(shiftAmount));
        Value * carryOutEndPos = b.CreateAdd(carryOutPosition, blockWidth_1);
        carryBlock = b.CreateTrunc(b.CreateURem(b.CreateUDiv(carryOutPosition, blockWidth), carryQueueBlocks), b.getInt32Ty());
        carryEndBlock = b.CreateTrunc(b.CreateURem(b.CreateUDiv(carryOutEndPos, blockWidth), carryQueueBlocks), b.getInt32Ty());
        lo_GEP = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, {ZERO, CFI, carryBlock});
        hi_GEP = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, {ZERO, CFI, carryEndBlock});
        lo_shift = b.CreateZExt(b.CreateURem(carryOutPosition, blockWidth), iBitBlock);
        hi_shift = b.CreateZExt(b.CreateSub(blockWidth_1, b.CreateURem(carryOutEndPos, blockWidth)), iBitBlock);
        c_lo = b.CreateOr(b.CreateBitCast(b.CreateLoad(b.getBitBlockType(), lo_GEP), iBitBlock), b.CreateShl(carryOut, lo_shift));
        c_hi = b.CreateLShr(carryOut, hi_shift);
        b.CreateStore(b.CreateBitCast(c_lo, b.getBitBlockType()), lo_GEP);
        b.CreateStore(b.CreateBitCast(c_hi, b.getBitBlockType()), hi_GEP);
        mIndexedLongAdvanceIndex++;
        mCurrentFrameIndex++;
        // Now handle the summary.
        if (mIfDepth > 0 || mLoopDepth > 0) {
            const auto summaryBlocks = ceil_udiv(shiftAmount, b.getBitBlockWidth());
            const auto summarySize = ceil_udiv(summaryBlocks, b.getBitBlockWidth());
            Constant * const SF = b.getInt32(summaryFrame);
            for (unsigned i = 0; i < summarySize; i++) {
                // All ones summary for now.
                b.CreateStore(b.allOnes(), b.CreateGEP(mCurrentFrameType,  mCurrentFrame, {ZERO, SF, b.getInt32(i)}));
            }
        }
        return result;
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief shortIndexedAdvanceCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::shortIndexedAdvanceCarryInCarryOut(kernel::KernelBuilder & b, const unsigned shiftAmount, Value * const strm, Value * const index_strm) {
    Value * const carryIn = getNextCarryIn(b);
    Value * carryOut, * result;
    std::tie(carryOut, result) = b.bitblock_indexed_advance(strm, index_strm, carryIn, shiftAmount);
    setNextCarryOut(b, carryOut);
    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief longAdvanceCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
inline Value * CarryManager::longAdvanceCarryInCarryOut(kernel::KernelBuilder & b, Value * const value, const unsigned shiftAmount) {

    assert (mHasLongAdvance);
    assert (shiftAmount >= LONG_ADVANCE_BREAKPOINT);
    assert (value);

    const auto blockWidth = b.getBitBlockWidth();
    Type * const streamTy = b.getIntNTy(blockWidth);
    Constant * const ZERO = b.getInt32(0);
    FixedArray<Value *, 3> indices;

    indices[0] = ZERO;

    if (mCarryInfo->hasSummary()) {
        if (shiftAmount > blockWidth) {

            // TODO: once CEILING(shiftAmount / 256) > 2, consider using a half-adder/subtractor strategy?

            Value * carry = b.CreateZExt(b.bitblock_any(value), streamTy);
            const auto summaryBlocks = ceil_udiv(shiftAmount, blockWidth);
            const auto summarySize = ceil_udiv(summaryBlocks, blockWidth);
            FixedVectorType * const bitBlockTy = b.getBitBlockType();
            IntegerType * const laneTy = cast<IntegerType>(bitBlockTy->getElementType());
            const auto laneWidth = laneTy->getIntegerBitWidth();

            assert (summarySize > 0);
            assert (is_power_2(laneWidth));

            indices[1] = b.getInt32(mCurrentFrameIndex);

            Type * const elemTy = mCurrentFrameType->getStructElementType(mCurrentFrameIndex)->getArrayElementType();

            for (unsigned i = 1;;++i) {

                assert (i <= summarySize);

                indices[2] = b.getInt32(i - 1);

                Value * const ptr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
                Value * const prior = b.CreateBitCast(b.CreateLoad(elemTy, ptr), streamTy);

                Value * advanced = nullptr;
                if (LLVM_LIKELY(summaryBlocks < laneWidth)) {
                    advanced = b.CreateOr(b.CreateShl(prior, 1), carry);
                    carry = b.CreateLShr(prior, summaryBlocks - 1);
                } else {
                    std::tie(carry, advanced) = b.bitblock_advance(b.bitCast(prior), carry, 1);
                }
                Value * stream = b.CreateBitCast(advanced, bitBlockTy);
                if (LLVM_LIKELY(i == summarySize)) {
                    const auto n = bitBlockTy->getNumElements();
                    SmallVector<Constant *, 16> mask(n);
                    const auto m = udiv(summaryBlocks, laneWidth);
                    if (m) {
                        std::fill_n(mask.data(), m, ConstantInt::getAllOnesValue(laneTy));
                    }
                    mask[m] = ConstantInt::get(laneTy, (1UL << (summaryBlocks & (laneWidth - 1))) - 1UL);
                    if (n > m) {
                        std::fill_n(mask.data() + m + 1, n - m, UndefValue::get(laneTy));
                    }
                    stream = b.CreateAnd(stream, ConstantVector::get(mask));
                    addToCarryOutSummary(b, stream);
                    b.CreateStore(stream, ptr);
                    break;
                }
                addToCarryOutSummary(b, stream);
                b.CreateStore(stream, ptr);
            }

            mCurrentFrameIndex++;

        } else {
            addToCarryOutSummary(b, value);
        }
    }

    indices[1] = b.getInt32(mCurrentFrameIndex++);

    // special case using a single buffer entry and the carry_out value.
    if (LLVM_LIKELY((shiftAmount < blockWidth) && (mLoopDepth == 0))) {

        indices[2] = ZERO;

        Value * const buffer = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
        Value * const carryIn = b.CreateLoad(b.getBitBlockType(), buffer);
        b.CreateStore(value, buffer);
        /* Very special case - no combine */
        if (LLVM_UNLIKELY(shiftAmount == blockWidth)) {
            return b.CreateBitCast(carryIn, b.getBitBlockType());
        }
        Value * const block0_shr = b.CreateLShr(b.CreateBitCast(carryIn, streamTy), blockWidth - shiftAmount);
        Value * const block1_shl = b.CreateShl(b.CreateBitCast(value, streamTy), shiftAmount);
        return b.CreateBitCast(b.CreateOr(block1_shl, block0_shr), b.getBitBlockType());
    } else { //
        const unsigned blockShift = shiftAmount & (blockWidth - 1);
        const unsigned summaryBlocks = ceil_udiv(shiftAmount, blockWidth);

        // Create a mask to implement circular buffer indexing
        Value * const indexMask = b.getInt32(nearest_pow2(summaryBlocks) - 1);
        Value * const blockIndex = b.getScalarField("CarryBlockIndex");
        assert (blockIndex->getType() == indexMask->getType());
        Value * const carryIndex0 = b.CreateSub(blockIndex, b.getInt32(summaryBlocks));
        indices[2] = b.CreateAnd(carryIndex0, indexMask);
        Value * const carryInPtr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
        Value * const carryIn = b.CreateLoad(b.getBitBlockType(), carryInPtr);
        indices[2] = b.CreateAnd(blockIndex, indexMask);
        Value * const carryOutPtr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
        assert (carryIn->getType() == b.getBitBlockType());

        // If the long advance is an exact multiple of BitBlockWidth, we simply return the oldest
        // block in the long advance carry data area.
        if (LLVM_UNLIKELY(blockShift == 0)) {
            b.CreateStore(value, carryOutPtr);
            return carryIn;
        } else { // Otherwise we need to combine data from the two oldest blocks.
            Value * const carryIndex1 = b.CreateSub(blockIndex, b.getInt32(summaryBlocks - 1));
            indices[2] = b.CreateAnd(carryIndex1, indexMask);

            Value * const carryInPtr2 = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
            Value * const carryIn2 = b.CreateLoad(b.getBitBlockType(), carryInPtr2);
            b.CreateStore(value, carryOutPtr);

            Value * const b0 = b.CreateLShr(b.CreateBitCast(carryIn, streamTy), blockWidth - blockShift);
            Value * const b1 = b.CreateShl(b.CreateBitCast(carryIn2, streamTy), blockShift);
            return b.CreateBitCast(b.CreateOr(b1, b0), b.getBitBlockType());
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getNextCarryIn
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::getNextCarryIn(kernel::KernelBuilder & b) {
    assert (mCurrentFrameIndex < mCurrentFrameType->getStructNumElements());
    Constant * const ZERO = b.getInt32(0);
    FixedArray<Value *, 3> indices;
    indices[0] = ZERO;
    indices[1] = b.getInt32(mCurrentFrameIndex);
    indices[2] = mLoopDepth == 0 ? ZERO : mLoopSelector;
    Type * carryTy = mCurrentFrameType->getStructElementType(mCurrentFrameIndex)->getArrayElementType();
    mCarryPackPtr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
    Value * const carryIn = b.CreateLoad(carryTy, mCarryPackPtr);
    if (mLoopDepth > 0) {
        b.CreateStore(Constant::getNullValue(carryIn->getType()), mCarryPackPtr);
    }
    return carryIn;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setNextCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::setNextCarryOut(kernel::KernelBuilder & b, Value * carryOut) {
    assert (mCurrentFrameIndex < mCurrentFrameType->getStructNumElements());
//    assert("carry out type does not match carry location type" && carryOut->getType() == mCarryPackPtr->getType()->getPointerElementType());
    if (mCarryInfo->hasSummary()) {
        addToCarryOutSummary(b, carryOut);
    }
    if (mLoopDepth != 0) {
        FixedArray<Value *, 3> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(mCurrentFrameIndex);
        indices[2] =  mNextLoopSelector;
        Type * carryTy = mCurrentFrameType->getStructElementType(mCurrentFrameIndex)->getArrayElementType();
        mCarryPackPtr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
  //      assert("carry out type does not match carry location type" && carryOut->getType() == mCarryPackPtr->getType()->getPointerElementType());
        if (LLVM_LIKELY(!mCarryInfo->nonCarryCollapsingMode())) {
            Value * accum = b.CreateLoad(carryTy, mCarryPackPtr);
            carryOut = b.CreateOr(carryOut, accum);
        }
    }
    ++mCurrentFrameIndex;
    b.CreateStore(carryOut, mCarryPackPtr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readCarryInSummary
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CarryManager::readCarryInSummary(kernel::KernelBuilder & b) const {
    assert (mCarryInfo->hasSummary());
    unsigned count = 2;
    if (LLVM_UNLIKELY(mCarryInfo->hasBorrowedSummary())) {
        Type * frameTy = mCurrentFrameType;
        count = 1;
        while (frameTy->isStructTy()) {
            ++count;
            assert (frameTy->getStructNumElements() > 0);
            frameTy = frameTy->getStructElementType(0);
        }
    }
    SmallVector<Value *, 16> indices(count + 1);
    Constant * const ZERO = b.getInt32(0);
    std::fill(indices.begin(), indices.end(), ZERO);
    if (mLoopDepth != 0) {
        indices[count] = mLoopSelector;
    }
    Value * const ptr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
    Value * summary = b.CreateLoad(b.getBitBlockType(), ptr);
    if (mNestedLoopCarryInMaskPhi) {
        summary = b.CreateAnd(summary, mNestedLoopCarryInMaskPhi);
    }
    return summary;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::writeCarryOutSummary(kernel::KernelBuilder & b, Value * const summary) const {
    assert (mCarryInfo->hasExplicitSummary());
    Constant * const ZERO = b.getInt32(0);
    FixedArray<Value *, 3> indices;
    indices[0] = ZERO;
    indices[1] = ZERO;
    indices[2] = mLoopDepth == 0 ? ZERO : mNextLoopSelector;
    Value * ptr = b.CreateGEP(mCurrentFrameType,  mCurrentFrame, indices);
//    assert ("summary type does not match defined type in frame" && ptr->getType()->getPointerElementType() == summary->getType());
    b.CreateStore(summary, ptr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addToCarryOutSummary
 ** ------------------------------------------------------------------------------------------------------------- */
void CarryManager::addToCarryOutSummary(kernel::KernelBuilder & b, Value * const value) {
    assert (mCarryInfo->hasSummary());
    // No need to add to summary if using an implicit summary
    if (mCarryInfo->hasSummary()) {
        assert ("cannot add null summary value!" && value);
        assert ("summary stack is empty!" && !mCarrySummaryStack.empty());
        Value * & summary = mCarrySummaryStack.back();
        Value * const carryOut = castToSummaryType(b, value, summary->getType());
        summary = b.CreateOr(summary, carryOut);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enumerate
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned CarryManager::getScopeCount(const PabloBlock * const scope, unsigned index) {
    for (const Statement * stmt : *scope) {
        if (LLVM_UNLIKELY(isa<Branch>(stmt))) {
            index = getScopeCount(cast<Branch>(stmt)->getBody(), index);
        }
    }
    return index + 1;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasNonEmptyCarryStruct
 ** ------------------------------------------------------------------------------------------------------------- */
bool CarryManager::hasNonEmptyCarryStruct(const Type * const frameTy) {
    if (frameTy->isStructTy()) {
        for (unsigned i = 0; i < frameTy->getStructNumElements(); ++i) {
            if (hasNonEmptyCarryStruct(frameTy->getStructElementType(i))) {
                return true;
            }
        }
        return false;
    }
    return true;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isEmptyCarryStruct
 ** ------------------------------------------------------------------------------------------------------------- */
bool CarryManager::isEmptyCarryStruct(const std::vector<Type *> & frameTys) {
    for (const Type * const t : frameTys) {
        if (LLVM_LIKELY(hasNonEmptyCarryStruct(t))) {
            return false;
        }
    }
    return true;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief analyse
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * CarryManager::analyse(kernel::KernelBuilder & b, const PabloBlock * const scope) {
    return analyse(b, scope, 0, 0, false);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief analyse
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * CarryManager::analyse(kernel::KernelBuilder & b, const PabloBlock * const scope,
                                   const unsigned ifDepth, const unsigned loopDepth,
                                   const bool inNonCarryCollapsingLoop) {
    assert ("scope cannot be null!" && scope);
    assert ("entry scope (and only the entry scope) must be in scope 0"
            && (mCarryScopes == 0 ? (scope == mKernel->getEntryScope()) : (scope != mKernel->getEntryScope())));
    assert (mCarryScopes < mCarryMetadata.size());
    Type * const carryTy = b.getBitBlockType();
    Type * const blockTy = b.getBitBlockType();

    const auto carryScopeIndex = mCarryScopes++;
    Type * const carryPackType = ArrayType::get(carryTy, ((loopDepth == 0) ? 1 : 2));
    std::vector<Type *> state;
    for (const Statement * stmt : *scope) {
        if (LLVM_UNLIKELY(isa<Advance>(stmt) || isa<IndexedAdvance>(stmt))) {
            int64_t amount;
            if (LLVM_LIKELY(isa<Advance>(stmt))) {
                amount = cast<Advance>(stmt)->getAmount();
            } else {
                amount = cast<IndexedAdvance>(stmt)->getAmount();
            }
            Type * type = carryPackType;
            if (LLVM_UNLIKELY(amount >= LONG_ADVANCE_BREAKPOINT)) {
                const auto blockWidth = b.getBitBlockWidth();
                const auto blocks = ceil_udiv(amount, blockWidth);
                unsigned required = blocks;
                if (loopDepth > 0) {
                    required++;
                }
                if (LLVM_UNLIKELY(isa<IndexedAdvance>(stmt))) {
                    required++;
                    mIndexedLongAdvanceTotal++;
                }
                type = ArrayType::get(blockTy, nearest_pow2(required));
                if (LLVM_UNLIKELY(blocks != 1 && (ifDepth > 0 || loopDepth > 0))) {
                    const auto summarySize = ceil_udiv(blocks, blockWidth);
                    // 1 bit will mark the presense of any bit in each block.
                    state.push_back(ArrayType::get(blockTy, summarySize));
                }
                mHasLongAdvance = true;
            }
            state.push_back(type);
        } else if (LLVM_UNLIKELY(isNonAdvanceCarryGeneratingStatement(stmt))) {
            state.push_back(carryPackType);
        } else if (LLVM_UNLIKELY(isa<If>(stmt))) {
            state.push_back(analyse(b, cast<If>(stmt)->getBody(), ifDepth + 1, loopDepth, false));
        } else if (LLVM_UNLIKELY(isa<While>(stmt))) {
            mHasLoop = true;
            const PabloBlock * const nestedScope = cast<While>(stmt)->getBody();
            const auto carryCollapsingMode = cast<While>(stmt)->isRegular();
            state.push_back(analyse(b, nestedScope, ifDepth, loopDepth + 1, !carryCollapsingMode));
        }
    }
    // Build the carry state struct and add the summary pack if needed.
    CarryData & cd = mCarryMetadata[carryScopeIndex];
    StructType * carryState = nullptr;
    CarryData::SummaryKind summaryKind = CarryData::NoSummary;

    // if we have at least one non-empty carry state, check if we need a summary
    if (LLVM_UNLIKELY(isEmptyCarryStruct(state) || inNonCarryCollapsingLoop)) {
        carryState = StructType::get(b.getContext(), state);
    } else {
        // do we have a summary or a sequence of nested empty structs?
        if (LLVM_LIKELY(ifDepth > 0 || loopDepth > 0)) {
            if (LLVM_LIKELY(state.size() > 1)) {
                summaryKind = CarryData::ExplicitSummary;
                state.insert(state.begin(), carryPackType);
            } else {
                summaryKind = CarryData::ImplicitSummary;
                if (hasNonEmptyCarryStruct(state[0])) {
                    summaryKind = CarryData::BorrowedSummary;
                }
            }
        }
        carryState = StructType::get(b.getContext(), state);
    }

    // If we're in a loop and cannot use collapsing carry mode, convert the carry state struct into a capacity,
    // carry state pointer, and summary pointer struct.
    if (LLVM_UNLIKELY(inNonCarryCollapsingLoop && state.size() > 0)) {
        mHasNonCarryCollapsingLoops = true;
        cd.setNestedCarryStateType(carryState);
        summaryKind = (CarryData::SummaryKind)(CarryData::ExplicitSummary | CarryData::NonCarryCollapsingMode);
        FixedArray<Type *, 3> fields;
        fields[NestedCapacity] = b.getSizeTy();
        fields[LastIncomingCarryLoopIteration] = b.getSizeTy();
        fields[NestedCarryState] = carryState->getPointerTo();
        carryState = StructType::get(b.getContext(), fields);
        assert (isDynamicallyAllocatedType(carryState));
    }
    cd.setSummaryKind(summaryKind);
    assert (carryState);
    return carryState;
}

Type * CarryManager::getSummaryTypeFromCurrentFrame(kernel::KernelBuilder & b) const {
    return b.getBitBlockType();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
CarryManager::CarryManager() noexcept
: mKernel(nullptr)
, mCurrentFrame(nullptr)
, mCurrentFrameIndex(0)
, mCarryInfo(nullptr)
, mNextSummaryTest(nullptr)
, mIfDepth(0)
, mHasLongAdvance(false)
, mIndexedLongAdvanceTotal(0)
, mIndexedLongAdvanceIndex(0)
, mHasNonCarryCollapsingLoops(false)
, mHasLoop(false)
, mLoopDepth(0)
, mNestedLoopCarryInMaskPhi(nullptr)
, mLoopSelector(nullptr)
, mNextLoopSelector(nullptr)
, mCarryPackPtr(nullptr)
, mCarryScopes(0)
, mCarrySummaryStack() {

}

}
