/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <pablo/pablo_compiler.h>
#include <pablo/pablo_kernel.h>
#include <pablo/codegenstate.h>
#include <pablo/boolean.h>
#include <pablo/arithmetic.h>
#include <pablo/branch.h>
#include <pablo/pablo_intrinsic.h>
#include <pablo/pe_advance.h>
#include <pablo/pe_debugprint.h>
#include <pablo/pe_lookahead.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_scanthru.h>
#include <pablo/pe_infile.h>
#include <pablo/pe_count.h>
#include <pablo/pe_everynth.h>
#include <pablo/pe_integer.h>
#include <pablo/pe_string.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_repeat.h>
#include <pablo/pe_pack.h>
#include <pablo/pe_var.h>
#include <pablo/pe_illustrator.h>
#include <pablo/ps_assign.h>
#include <pablo/ps_terminate.h>
#include <pablo/carry_manager.h>
#include <pablo/compressed_carry_manager.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/STLExtras.h> // for make_unique
#include <boost/container/flat_set.hpp>
#include <kernel/core/streamset.h>
#include <pablo/printer_pablos.h>
#include <tuple>

using namespace llvm;
using namespace kernel;

namespace pablo {

using TypeId = PabloAST::ClassTypeId;
using IllustratorTypeId = Illustrate::IllustratorTypeId;

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(12, 0, 0)
using FixedVectorType = llvm::VectorType;
#endif

using Vars = boost::container::flat_set<const Var *>;

template <typename T>
using Vec = SmallVector<T, 64>;

inline static unsigned getAlignment(const Type * const type) {
    return type->getPrimitiveSizeInBits() / 8;
}

inline static unsigned getAlignment(const Value * const expr) {
    return getAlignment(expr->getType());
}

void PabloCompiler::initializeKernelData(KernelBuilder & b) {
    mBranchCount = 0;
    examineBlock(b, mKernel->getEntryScope());
    mCarryManager->initializeCarryData(b, mKernel);
    if (CompileOptionIsSet(PabloCompilationFlags::EnableProfiling)) {
        const auto count = (mBranchCount * 2) + 1;
        mKernel->addInternalScalar(ArrayType::get(mKernel->getSizeTy(), count), "profile");
        mBasicBlock.reserve(count);
    }
}

void PabloCompiler::releaseKernelData(KernelBuilder & b) {
    mCarryManager->releaseCarryData(b);
}

void PabloCompiler::clearCarryData(KernelBuilder & b) {
    mCarryManager->clearCarryData(b);
}

void PabloCompiler::initializeIllustrator(KernelBuilder & b) {
    SmallVector<size_t, 8> loopIds;
    loopIds.push_back(1);
    size_t loopId = 1;
    if (identifyIllustratedValues(b, mKernel->getEntryScope(), loopIds, loopId)) {
        mContainsIllustratedValue.push_back(nullptr); // add a fake entry for the entry scope
    }
    assert (loopIds.size() == 1 && loopIds[0] == 1);
}

bool PabloCompiler::identifyIllustratedValues(KernelBuilder & b, const PabloBlock * const block, SmallVector<size_t, 8> & loopIds, size_t & currentLoopId) {
    bool containsAnyIllustratedValue = false;
    for (const Statement * statement : *block) {
        if (const Illustrate * il = dyn_cast<Illustrate>(statement)) {
            Constant * kernelName = b.GetString(getName());
            Constant * streamName = b.GetString(il->getName());


            Type * ty = il->getExpr()->getType();
            size_t rowCount = 1;
            if (LLVM_LIKELY(isa<ArrayType>(ty))) {
                rowCount = ty->getArrayNumElements();
                ty = ty->getArrayElementType();
            }
            if (LLVM_LIKELY(isa<VectorType>(ty))) {
                ty = cast<VectorType>(ty)->getElementType();
            }

            const size_t fieldWidth = ty->getPrimitiveSizeInBits();
            registerIllustrator(b, b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT),
                                kernelName, streamName, getHandle(),
                                rowCount, 1, fieldWidth, MemoryOrdering::RowMajor,
                                il->getIllustratorType(), il->getReplacementCharacter(0), il->getReplacementCharacter(1),
                                loopIds);

            containsAnyIllustratedValue = true;

        } else if (isa<Branch>(statement)) {
            const auto origLoopId = currentLoopId;
            loopIds.push_back(++currentLoopId);
            const bool any = identifyIllustratedValues(b, cast<Branch>(statement)->getBody(), loopIds, currentLoopId);
            if (LLVM_UNLIKELY(any)) {
                if (isa<While>(statement)) {
                    mContainsIllustratedValue.push_back(cast<While>(statement));
                }
            } else {
                currentLoopId = origLoopId;
            }
            loopIds.pop_back();
            containsAnyIllustratedValue |= any;
        }
    }
    return containsAnyIllustratedValue;
}

void PabloCompiler::compile(KernelBuilder & b) {
    assert (mCarryManager);
    mCarryManager->initializeCodeGen(b);
    assert (mKernel);
    PabloBlock * const entryBlock = mKernel->getEntryScope(); assert (entryBlock);
    mMarker.emplace(entryBlock->createZeroes(), b.allZeroes());
    mMarker.emplace(entryBlock->createOnes(), b.allOnes());
    mBranchCount = 0;
    addBranchCounter(b);
    mEntryBlock = b.GetInsertBlock();
    if (LLVM_UNLIKELY(codegen::EnableIllustrator && !mContainsIllustratedValue.empty())) {
        Value * ptr; Type * ty;
        std::tie(ptr, ty) = b.getScalarFieldPtr(KERNEL_ILLUSTRATOR_STRIDE_NUM);
        mIllustratorStrideNum = b.CreateLoad(ty, ptr);
        Value * val = b.CreateAdd(mIllustratorStrideNum, b.getSize(1));
        b.CreateStore(val, ptr);
        Function * enterKernel = b.getModule()->getFunction(KERNEL_ILLUSTRATOR_ENTER_KERNEL);
        FixedArray<Value *, 2> args;
        args[0] = b.CreatePointerCast(b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT), b.getVoidPtrTy());
        args[1] = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
        b.CreateCall(enterKernel, args);
    }
    compileBlock(b, entryBlock);
    mCarryManager->finalizeCodeGen(b);
    if (LLVM_UNLIKELY(codegen::EnableIllustrator && !mContainsIllustratedValue.empty())) {
        Function * exitKernel = b.getModule()->getFunction(KERNEL_ILLUSTRATOR_EXIT_KERNEL);
        FixedArray<Value *, 2> args;
        args[0] = b.CreatePointerCast(b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT), b.getVoidPtrTy());
        args[1] = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
        b.CreateCall(exitKernel, args);
    }
}

const Var * PabloCompiler::findInputParam(const Statement * const stmt, const Var * const param) const {
    const Var * expr = param;
    while (isa<Extract>(expr)) {
        expr = cast<Extract>(expr)->getArray();
    }
    const Var * const var = cast<Var>(expr);
    if (LLVM_LIKELY(var->isKernelParameter())) {
        return param;
    } else { // has an input been assigned to a Var?
        for (const PabloAST * const user : var->users()) {
            if (const Assign * const assign = dyn_cast<Assign>(user)) {
                if (assign->getVariable() == var) {
                    const PabloAST * const ref = assign->getValue();
                    if (LLVM_LIKELY(isa<Var>(ref) && dominates(assign, stmt))) {
                        const Var * expr = cast<Var>(ref);
                        while (isa<Extract>(expr)) {
                            expr = cast<Extract>(expr)->getArray();
                        }
                        for (unsigned i = 0; i < mKernel->getNumOfInputs(); ++i) {
                            if (expr == mKernel->getInput(i)) {
                                // found a matching assign/ref to an input var
                                return cast<Var>(ref);
                            }
                        }
                    }
                }
            }
        }
    }
    return nullptr;
}

void PabloCompiler::examineBlock(KernelBuilder & b, const PabloBlock * const block) {
    for (const Statement * stmt : *block) {
        if (LLVM_UNLIKELY(isa<Lookahead>(stmt))) {
            const Lookahead * const la = cast<Lookahead>(stmt);
            auto e = la->getExpression();
            if (LLVM_UNLIKELY(!isa<Var>(e))) {
                report_fatal_error("Lookahead " + stmt->getName() + " can only be performed on an input streamset");
            }
            const PabloAST * array = findInputParam(stmt, cast<Var>(e));
            while (array && isa<Extract>(array)) {
                array = cast<Extract>(array)->getArray();
            }
            bool notFound = true;
            for (unsigned i = 0; i < mKernel->getNumOfInputs(); ++i) {
                const Var * const input = mKernel->getInput(i);
                if (array == input) {
                    const auto & binding = mKernel->getInputStreamSetBinding(i);
                    size_t maxLookAhead = 0;
                    for (const auto & attr : binding.getAttributes()) {
                        if (attr.getKind() == kernel::Attribute::KindId::LookAhead) {
                            maxLookAhead = std::max(maxLookAhead, attr.amount());
                        }
                    }
                    if (LLVM_UNLIKELY(maxLookAhead < la->getAmount())) {
                        std::string tmp;
                        raw_string_ostream out(tmp);
                        array->print(out);
                        out << " must have a lookahead attribute of at least " << la->getAmount();
                        report_fatal_error(StringRef(out.str()));
                    }
                    notFound = false;
                    break;
                }
            }
            if (LLVM_UNLIKELY(notFound)) {
                report_fatal_error("Lookahead " + stmt->getName() + " can only be performed on an input streamset");
            }
        } else if (LLVM_UNLIKELY(isa<Branch>(stmt))) {
            ++mBranchCount;
            examineBlock(b, cast<Branch>(stmt)->getBody());
        } else if (LLVM_UNLIKELY(isa<Count>(stmt))) {
            mKernel->addInternalScalar(stmt->getType(), stmt->getName().str());
        } else if (LLVM_UNLIKELY(isa<EveryNth>(stmt))) {
            const auto fieldWidth = b.getSizeTy()->getBitWidth();
            mKernel->addInternalScalar(b.getIntNTy(fieldWidth), stmt->getName().str());
        }
    }
}

void PabloCompiler::addBranchCounter(KernelBuilder & b) {
    if (CompileOptionIsSet(PabloCompilationFlags::EnableProfiling)) {
        Value * ptr; Type * ty;
        std::tie(ptr, ty) = b.getScalarFieldPtr("profile");
        assert (mBasicBlock.size() < ty->getArrayNumElements());
        ptr = b.CreateGEP(ty, ptr, {b.getInt32(0), b.getInt32(mBasicBlock.size())});
        const auto alignment = getAlignment(ty);
        Value * value = b.CreateAlignedLoad(ty, ptr, alignment, false, "branchCounter");
        value = b.CreateAdd(value, ConstantInt::get(cast<IntegerType>(value->getType()), 1));
        b.CreateAlignedStore(value, ptr, alignment);
        mBasicBlock.push_back(b.GetInsertBlock());
    }
}

inline void PabloCompiler::compileBlock(KernelBuilder & b, const PabloBlock * const block) {
    for (const Statement * statement : *block) {
        compileStatement(b, statement);
    }
}

Vars getRootVars(const Branch * const branch) {
    Vars vars;
    for (Var * var : branch->getEscaped()) {
        while (isa<Extract>(var)) {
            var = cast<Extract>(var)->getArray();
        }
        vars.insert(var);
    }
    return vars;
}

void PabloCompiler::compileIf(KernelBuilder & b, const If * const ifStatement) {

    //
    //  The If-ElseZero stmt:
    //  if <predicate:expr> then <body:stmt>* elsezero <defined:var>* endif
    //  If the value of the predicate is nonzero, then determine the values of variables
    //  <var>* by executing the given statements.  Otherwise, the value of the
    //  variables are all zero.  Requirements: (a) no variable that is defined within
    //  the body of the if may be accessed outside unless it is explicitly
    //  listed in the variable list, (b) every variable in the defined list receives
    //  a value within the body, and (c) the logical consequence of executing
    //  the statements in the event that the predicate is zero is that the
    //  values of all defined variables indeed work out to be 0.
    //
    //  Simple Implementation with Phi nodes:  a phi node in the if exit block
    //  is inserted for each variable in the defined variable list.  It receives
    //  a zero value from the ifentry block and the defined value from the if
    //  body.
    //

    using IncomingVec = Vec<std::pair<const Var *, Value *>>;

    BasicBlock * const ifEntryBlock = b.GetInsertBlock();
    ++mBranchCount;
    BasicBlock * const ifBodyBlock = b.CreateBasicBlock("if.body_" + std::to_string(mBranchCount));
    BasicBlock * const ifEndBlock = b.CreateBasicBlock("if.end_" + std::to_string(mBranchCount));

    IncomingVec incoming;

    const auto escaped = getRootVars(ifStatement);

    for (const Var * var : escaped) {
        if (LLVM_UNLIKELY(var->isKernelParameter())) {
            compileExpression(b, var, false);
        } else {
            auto f = mMarker.find(var);
            if (LLVM_UNLIKELY(f == mMarker.end())) {
                std::string tmp;
                raw_string_ostream out(tmp);
                var->print(out);
                out << " is uninitialized prior to entering ";
                ifStatement->print(out);
                report_fatal_error(StringRef(out.str()));
            }
            incoming.emplace_back(var, f->second);
        }
    }

    mCarryManager->enterIfScope(b);

    Value * condition = compileExpression(b, ifStatement->getCondition());
    Value * const cond = mCarryManager->generateEntrySummaryTest(b, condition);
    b.CreateCondBr(cond, ifBodyBlock, ifEndBlock);

    // Entry processing is complete, now handle the body of the if.
    b.SetInsertPoint(ifBodyBlock);

    mCarryManager->enterIfBody(b, ifEntryBlock);

    addBranchCounter(b);

    compileBlock(b, ifStatement->getBody());

    mCarryManager->leaveIfBody(b, b.GetInsertBlock());

    BasicBlock * ifExitBlock = b.GetInsertBlock();

    b.CreateBr(ifEndBlock);

    ifEndBlock->moveAfter(ifExitBlock);

    //End Block
    b.SetInsertPoint(ifEndBlock);

    mCarryManager->leaveIfScope(b, ifEntryBlock, ifExitBlock);

    for (const auto & i : incoming) {
        const Var * var; Value * incoming;
        std::tie(var, incoming) = i;

        auto f = mMarker.find(var);
        if (LLVM_UNLIKELY(f == mMarker.end())) {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << "PHINode creation error: ";
            var->print(out);
            out << " was not assigned an outgoing value.";
            report_fatal_error(StringRef(out.str()));
        }

        Value * const outgoing = f->second;
        if (LLVM_UNLIKELY(incoming == outgoing)) {
            continue;
        }

        if (LLVM_UNLIKELY(incoming->getType() != outgoing->getType())) {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << "PHINode creation error: incoming type of ";
            var->print(out);
            out << " (";
            incoming->getType()->print(out);
            out << ") differs from the outgoing type (";
            outgoing->getType()->print(out);
            out << ") within ";
            ifStatement->print(out);
            report_fatal_error(StringRef(out.str()));
        }
        SmallVector<char, 64> tmp;
        raw_svector_ostream name(tmp);
        PabloPrinter::print(var, name);
        PHINode * phi = b.CreatePHI(incoming->getType(), 2, name.str());
        phi->addIncoming(incoming, ifEntryBlock);
        phi->addIncoming(outgoing, ifExitBlock);
        f->second = phi;
    }

    addBranchCounter(b);
}

void PabloCompiler::compileWhile(KernelBuilder & b, const While * const whileStatement) {

    using IncomingVec = Vec<std::tuple<const Var *, Value *, Value *>>;

    BasicBlock * const whileEntryBlock = b.GetInsertBlock();

    const auto escaped = getRootVars(whileStatement);

#ifdef ENABLE_BOUNDED_WHILE
    PHINode * bound_phi = nullptr;  // Needed for bounded while loops.
#endif
    // On entry to the while structure, proceed to execute the first iteration
    // of the loop body unconditionally. The while condition is tested at the end of
    // the loop.

    for (const Var * var : escaped) {
        if (LLVM_UNLIKELY(var->isKernelParameter())) {
            mMarker.insert({var, compileExpression(b, var, false)});
        }
    }

    Value * illustratorObj = nullptr;

    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        const auto f = std::find(mContainsIllustratedValue.begin(), mContainsIllustratedValue.end(), whileStatement);
        if (LLVM_UNLIKELY(f != mContainsIllustratedValue.end())) {
            illustratorObj = b.CreatePointerCast(b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT), b.getVoidPtrTy());
            Function * fIllustratorEnterLoop = b.getModule()->getFunction(KERNEL_ILLUSTRATOR_ENTER_LOOP);
            FixedArray<Value *, 2> args;
            args[0] = illustratorObj;
            args[1] = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
            b.CreateCall(fIllustratorEnterLoop, args);
        }
    }

    mCarryManager->enterLoopScope(b);

    BasicBlock * whileBodyBlock = b.CreateBasicBlock("while.body_" + std::to_string(mBranchCount));
    BasicBlock * whileEndBlock = b.CreateBasicBlock("while.end_" + std::to_string(mBranchCount));
    ++mBranchCount;

    const PabloAST * const cond = whileStatement->getCondition();
    Value * outerCondition = compileExpression(b, cond);
    Value * const outerCond = mCarryManager->generateEntrySummaryTest(b, outerCondition);
    b.CreateCondBr(outerCond, whileBodyBlock, whileEndBlock);

    b.SetInsertPoint(whileBodyBlock);

    //
    // There are 3 sets of Phi nodes for the while loop.
    // (1) Carry-ins: (a) incoming carry data first iterations, (b) zero thereafter
    // (2) Carry-out accumulators: (a) zero first iteration, (b) |= carry-out of each iteration
    // (3) Next nodes: (a) values set up before loop, (b) modified values calculated in loop.
#ifdef ENABLE_BOUNDED_WHILE
    // (4) The loop bound, if any.
#endif

    IncomingVec variants;

    // for any Next nodes in the loop body, initialize to (a) pre-loop value.
    for (const auto var : escaped) {
        if (LLVM_UNLIKELY(!var->isKernelParameter())) {
            auto f = mMarker.find(var);
            if (LLVM_UNLIKELY(f == mMarker.end())) {
                std::string tmp;
                raw_string_ostream out(tmp);
                out << "PHINode creation error: ";
                var->print(out);
                out << " is uninitialized prior to entering ";
                whileStatement->print(out);
                report_fatal_error(StringRef(out.str()));
            }
            Value * entryValue = f->second;
            SmallVector<char, 64> tmp;
            raw_svector_ostream name(tmp);
            PabloPrinter::print(var, name);
            PHINode * const phi = b.CreatePHI(entryValue->getType(), 2, name.str());
            phi->addIncoming(entryValue, whileEntryBlock);
            f->second = phi;
            assert(mMarker[var] == phi);
            variants.emplace_back(var, phi, entryValue);
        }
    }
#ifdef ENABLE_BOUNDED_WHILE
    if (whileStatement->getBound()) {
        bound_phi = b.CreatePHI(b.getSizeTy(), 2, "while_bound");
        bound_phi->addIncoming(b.getSize(whileStatement->getBound()), whileEntryBlock);
    }
#endif

//    const auto f = mMarker.find(cond);
//    if (LLVM_LIKELY(f != mMarker.end())) {
//        mMarker.erase(f);
//    }

    mCarryManager->enterLoopBody(b, whileEntryBlock);
    addBranchCounter(b);
    if (LLVM_UNLIKELY(illustratorObj)) {
        Function * fIllustratorIterateLoop = b.getModule()->getFunction(KERNEL_ILLUSTRATOR_ITERATE_LOOP);
        assert (fIllustratorIterateLoop);
        FixedArray<Value *, 2> args;
        args[0] = illustratorObj;
        args[1] = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
        b.CreateCall(fIllustratorIterateLoop, args);
    }
    compileBlock(b, whileStatement->getBody());
    mCarryManager->leaveLoopBody(b);
    // After the whileBody has been compiled, we may be in a different basic block.
    BasicBlock * const whileExitBlock = b.GetInsertBlock();

#ifdef ENABLE_BOUNDED_WHILE
    if (whileStatement->getBound()) {
        Value * new_bound = b.CreateSub(bound_phi, b.getSize(1));
        bound_phi->addIncoming(new_bound, whileExitBlock);
        condition = b.CreateAnd(condition, b.CreateICmpUGT(new_bound, ConstantInt::getNullValue(b.getSizeTy())));
    }
#endif

    // and for any variant nodes in the loop body
    for (auto & variant : variants) {
        const Var * var; Value * incomingPhi;
        std::tie(var, incomingPhi, std::ignore) = variant;
        const auto f = mMarker.find(var);
        if (LLVM_UNLIKELY(f == mMarker.end())) {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << "PHINode creation error: ";
            var->print(out);
            out << " is no longer assigned a value.";
            report_fatal_error(StringRef(out.str()));
        }

        Value * const outgoingValue = f->second;

        if (LLVM_UNLIKELY(incomingPhi->getType() != outgoingValue->getType())) {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << "PHINode creation error: incoming type of ";
            var->print(out);
            out << " (";
            incomingPhi->getType()->print(out);
            out << ") differs from the outgoing type (";
            outgoingValue->getType()->print(out);
            out << ") within ";
            whileStatement->print(out);
            report_fatal_error(StringRef(out.str()));
        }

        // update the final outgoing value for the loop
        std::get<1>(variant) = outgoingValue;
        cast<PHINode>(incomingPhi)->addIncoming(outgoingValue, whileExitBlock);
    }

    // Terminate the while loop body with a conditional branch back.
    Value * const innerCondition = compileExpression(b, cond);
    Value * const exitCond = mCarryManager->generateExitSummaryTest(b, innerCondition);
    b.CreateCondBr(exitCond, whileBodyBlock, whileEndBlock);

    whileEndBlock->moveAfter(whileExitBlock);

    b.SetInsertPoint(whileEndBlock);
    // phi out any escaped variant
    for (const auto & variant : variants) {
        const Var * var; Value * outgoingValue; Value * incomingValue;
        std::tie(var, outgoingValue, incomingValue) = variant;
        PHINode * const escapedValue = b.CreatePHI(outgoingValue->getType(), 2);
        escapedValue->addIncoming(incomingValue, whileEntryBlock);
        escapedValue->addIncoming(outgoingValue, whileExitBlock);
        auto f = mMarker.find(var);
        assert (f != mMarker.end());
        f->second = escapedValue;
    }
    mCarryManager->leaveLoopScope(b, whileEntryBlock, whileExitBlock);
    if (LLVM_UNLIKELY(illustratorObj)) {
        Module * m = b.getModule();
        Function * fIllustratorExitLoop = m->getFunction(KERNEL_ILLUSTRATOR_EXIT_LOOP);
        assert (fIllustratorExitLoop);
        FixedArray<Value *, 2> args;
        args[0] = illustratorObj;
        args[1] = b.CreatePointerCast(getHandle(), b.getVoidPtrTy());
        b.CreateCall(fIllustratorExitLoop, args);
    }

    addBranchCounter(b);
}

void PabloCompiler::compileStatement(KernelBuilder & b, const Statement * const stmt) {
    if (LLVM_UNLIKELY(isa<If>(stmt))) {
        compileIf(b, cast<If>(stmt));
    } else if (LLVM_UNLIKELY(isa<While>(stmt))) {
        compileWhile(b, cast<While>(stmt));
    } else {
        const PabloAST * expr = stmt;
        Value * value = nullptr;
        if (isa<And>(stmt)) {
            Value * const op0 = compileExpression(b, stmt->getOperand(0));
            Value * const op1 = compileExpression(b, stmt->getOperand(1));
            value = b.simd_and(op0, op1, stmt->getName());
        } else if (isa<Or>(stmt)) {
            Value * const op0 = compileExpression(b, stmt->getOperand(0));
            Value * const op1 = compileExpression(b, stmt->getOperand(1));
            value = b.simd_or(op0, op1, stmt->getName());
        } else if (isa<Xor>(stmt)) {
            Value * const op0 = compileExpression(b, stmt->getOperand(0));
            Value * const op1 = compileExpression(b, stmt->getOperand(1));
            value = b.simd_xor(op0, op1, stmt->getName());
        } else if (const Sel * sel = dyn_cast<Sel>(stmt)) {
            Value* ifMask = compileExpression(b, sel->getCondition());
            Value* ifTrue = b.simd_and(ifMask, compileExpression(b, sel->getTrueExpr()));
            Value* ifFalse = b.simd_and(b.simd_not(ifMask), compileExpression(b, sel->getFalseExpr()));
            value = b.simd_or(ifTrue, ifFalse, stmt->getName());
        } else if (isa<Not>(stmt)) {
            Value * const op = compileExpression(b, stmt->getOperand(0));
            assert (op->getType()->isVectorTy());
            assert (cast<FixedVectorType>(op->getType())->getElementCount().getFixedValue() > 0);
            value = b.simd_not(op, stmt->getName());
        } else if (isa<Advance>(stmt)) {
            const Advance * const adv = cast<Advance>(stmt);
            // If our expr is an Extract op on a mutable Var then we need to pass the index value to the carry
            // manager so that it properly selects the correct carry bit.
            Value * expr = compileExpression(b, adv->getExpression());
            value = mCarryManager->advanceCarryInCarryOut(b, adv, expr);
        } else if (isa<IndexedAdvance>(stmt)) {
            const IndexedAdvance * const adv = cast<IndexedAdvance>(stmt);
            Value * strm = compileExpression(b, adv->getExpression());
            Value * index_strm = compileExpression(b, adv->getIndex());
            // If our expr is an Extract op on a mutable Var then we need to pass the index value to the carry
            // manager so that it properly selects the correct carry bit.
            value = mCarryManager->indexedAdvanceCarryInCarryOut(b, adv, strm, index_strm);
        } else if (const MatchStar * mstar = dyn_cast<MatchStar>(stmt)) {
            Value * const marker = compileExpression(b, mstar->getMarker());
            Value * const cc = compileExpression(b, mstar->getCharClass());
            Value * const marker_and_cc = b.simd_and(marker, cc);
            Value * const sum = mCarryManager->addCarryInCarryOut(b, mstar, marker_and_cc, cc);
            value = b.simd_or(b.simd_xor(sum, cc), marker, mstar->getName());
        } else if (const ScanThru * sthru = dyn_cast<ScanThru>(stmt)) {
            Value * const from = compileExpression(b, sthru->getScanFrom());
            Value * const thru = compileExpression(b, sthru->getScanThru());
            Value * const sum = mCarryManager->addCarryInCarryOut(b, sthru, from, thru);
            value = b.simd_and(sum, b.simd_not(thru), sthru->getName());
        } else if (const ScanTo * sthru = dyn_cast<ScanTo>(stmt)) {
            Value * const marker_expr = compileExpression(b, sthru->getScanFrom());
            Value * const to = b.simd_or(compileExpression(b, sthru->getScanTo()), b.getScalarField("EOFbit"));
            Value * const sum = mCarryManager->addCarryInCarryOut(b, sthru, marker_expr, b.simd_not(to));
            value = b.simd_and(sum, to, sthru->getName());
        } else if (const AdvanceThenScanThru * sthru = dyn_cast<AdvanceThenScanThru>(stmt)) {
            Value * const from = compileExpression(b, sthru->getScanFrom());
            Value * const thru = compileExpression(b, sthru->getScanThru());
            Value * const sum = mCarryManager->addCarryInCarryOut(b, sthru, from, b.simd_or(from, thru));
            value = b.simd_and(sum, b.simd_not(thru), sthru->getName());
        } else if (const AdvanceThenScanTo * sthru = dyn_cast<AdvanceThenScanTo>(stmt)) {
            Value * const from = compileExpression(b, sthru->getScanFrom());
            Value * const to = b.simd_or(compileExpression(b, sthru->getScanTo()), b.getScalarField("EOFbit"));
            Value * const sum = mCarryManager->addCarryInCarryOut(b, sthru, from, b.simd_or(from, b.simd_not(to)));
            value = b.simd_and(sum, to, sthru->getName());
        } else if (const TerminateAt * s = dyn_cast<TerminateAt>(stmt)) {
            Value * signal_strm = compileExpression(b, s->getExpr());
            BasicBlock * signalCallBack = b.CreateBasicBlock("signalCallBack");
            BasicBlock * postSignal = b.CreateBasicBlock("postSignal");
            b.CreateCondBr(b.bitblock_any(signal_strm), signalCallBack, postSignal);
            b.SetInsertPoint(signalCallBack);
            // Perhaps check for handler address and skip call back if none???
            Value * handler = b.getScalarField("handler_address");
            Function * const dispatcher = b.getModule()->getFunction("signal_dispatcher"); assert (dispatcher);
            FunctionType * fTy = dispatcher->getFunctionType();
            b.CreateCall(fTy, dispatcher, {handler, ConstantInt::get(b.getInt32Ty(), s->getSignalCode())});
            //Value * rel_position = b.createCountForwardZeroes(signal_strm);
            //Value * position = b.CreateAdd(b.getProcessedItemCount(), rel_position);
            b.setFatalTerminationSignal();
            b.CreateBr(postSignal);
            b.SetInsertPoint(postSignal);
            value = signal_strm;
        } else if (LLVM_UNLIKELY(isa<Assign>(stmt))) {
            expr = cast<Assign>(stmt)->getVariable();
            value = compileExpression(b, cast<Assign>(stmt)->getValue());
            if (cast<Var>(expr)->isKernelParameter()) {
                Value * const ptr = compileExpression(b, expr, false);
                Type * const type = value->getType();
                auto & dl = b.getModule()->getDataLayout();
                const auto align = dl.getABITypeAlign(type).value();
                b.CreateAlignedStore(value, ptr, align);
                value = ptr;
            }
        } else if (const InFile * e = dyn_cast<InFile>(stmt)) {
            Value * EOFmask = b.getScalarField("EOFmask");
            value = b.simd_and(compileExpression(b, e->getExpr()), b.simd_not(EOFmask), stmt->getName());
        } else if (const AtEOF * e = dyn_cast<AtEOF>(stmt)) {
            Value * EOFbit = b.getScalarField("EOFbit");
            value = b.simd_and(compileExpression(b, e->getExpr()), EOFbit);
        } else if (const Count * c = dyn_cast<Count>(stmt)) {
            Value * EOFbit = b.getScalarField("EOFbit");
            Value * EOFmask = b.getScalarField("EOFmask");
            Value * const to_count = b.simd_and(b.simd_or(b.simd_not(EOFmask), EOFbit), compileExpression(b, c->getExpr()));
            Value * ptr; Type * ty;
            std::tie(ptr, ty) = b.getScalarFieldPtr(stmt->getName().str());
            const auto alignment = getAlignment(ty);
            Value * const countSoFar = b.CreateAlignedLoad(ty, ptr, alignment, c->getName() + "_accumulator");
            const auto fieldWidth = b.getSizeTy()->getBitWidth();
            Value * bitBlockCount = b.simd_popcount(b.getBitBlockWidth(), to_count);
            value = b.CreateAdd(b.mvmd_extract(fieldWidth, bitBlockCount, 0), countSoFar, "countSoFar");
            b.CreateAlignedStore(value, ptr, alignment);
        } else if (const EveryNth * e = dyn_cast<EveryNth>(stmt)) {
            Value * EOFbit = b.getScalarField("EOFbit");
            Value * EOFmask = b.getScalarField("EOFmask");
            Value * const to_count = b.simd_and(b.simd_or(b.simd_not(EOFmask), EOFbit), compileExpression(b, e->getExpr()));
            Value * ptr; Type * ty;
            std::tie(ptr, ty) = b.getScalarFieldPtr(stmt->getName().str());
            const auto alignment = getAlignment(ty);
            Value * const pending = b.CreateAlignedLoad(ty, ptr, alignment, e->getName() + "_accumulator");
            const auto fieldWidth = b.getSizeTy()->getBitWidth();
            const auto blockWidth = b.getBitBlockWidth();
            const auto hiBlock = (blockWidth / fieldWidth) - 1;
            const uint64_t n = e->getN()->value();
            size_t mask = 0x0;
            for (unsigned i = 0; i < sizeof(size_t) * 8; i += n) { mask = (mask << n) | 0x1; }
            Value * const vmask = b.getIntN(fieldWidth, mask);
            Value * const vn = b.getIntN(fieldWidth, n);
            Value * const fieldCounts = b.simd_popcount(fieldWidth, to_count);
            Value * const sumCounts = b.hsimd_partial_sum(fieldWidth, fieldCounts);
            Value * const splatPending = b.simd_fill(fieldWidth, pending);
            Value * const sumCountPend = b.simd_add(fieldWidth, sumCounts, splatPending);
            Value * const splatN = b.simd_fill(fieldWidth, vn);
            Value * const finalSumCounts = b.mvmd_dslli(fieldWidth, sumCountPend, splatPending, 1);
            Value * const shift = b.CreateURem(b.CreateSub(splatN, b.CreateURem(finalSumCounts, splatN)), splatN);
            Value * const splatMask = b.simd_fill(fieldWidth, vmask);
            Value * const finalNthMask = b.simd_sllv(fieldWidth, splatMask, shift);
            value = b.simd_pdep(fieldWidth, finalNthMask, to_count);
            Value * const pendingOut = b.CreateURem(b.mvmd_extract(fieldWidth, sumCountPend, hiBlock), vn);
            b.CreateAlignedStore(pendingOut, ptr, alignment);
        } else if (const Lookahead * l = dyn_cast<Lookahead>(stmt)) {
            const Var * stream = findInputParam(l, cast<Var>(l->getExpression()));
            Value * index = nullptr;
            if (LLVM_UNLIKELY(isa<Extract>(stream))) {
                index = compileExpression(b, cast<Extract>(stream)->getIndex(), true);
                stream = cast<Extract>(stream)->getArray();
            } else {
                index = b.getInt32(0);
            }
            const auto bit_shift = (l->getAmount() % b.getBitBlockWidth());
            const auto block_shift = (l->getAmount() / b.getBitBlockWidth());
            Value * ptr = b.getInputStreamBlockPtr(stream->getName(), index, b.getSize(block_shift));
            // Value * base = b.CreatePointerCast(b.getBaseAddress(cast<Var>(stream)->getName()), ptr->getType());
            Value * lookAhead = b.CreateBlockAlignedLoad(b.getBitBlockType(), ptr);
            if (LLVM_UNLIKELY(bit_shift == 0)) {  // Simple case with no intra-block shifting.
                value = lookAhead;
            } else { // Need to form shift result from two adjacent blocks.
                Value * ptr1 = b.getInputStreamBlockPtr(stream->getName(), index, b.getSize(block_shift + 1));
                Value * lookAhead1 = b.CreateBlockAlignedLoad(b.getBitBlockType(), ptr1);
                value = b.mvmd_dslli(1, lookAhead1, lookAhead, b.getBitBlockWidth() - bit_shift);
                value = b.CreateBitCast(value, b.getBitBlockType());
            }
        } else if (const Repeat * const s = dyn_cast<Repeat>(stmt)) {
            value = compileExpression(b, s->getValue());
            Type * const ty = s->getType();
            if (LLVM_LIKELY(ty->isVectorTy())) {
                const auto repeatWidth = value->getType()->getIntegerBitWidth();
                value = b.bitCast(b.simd_fill(repeatWidth, value));
            } else {
                value = b.CreateZExtOrTrunc(value, ty);
            }
        } else if (const PackH * const p = dyn_cast<PackH>(stmt)) {
            const auto sourceWidth = cast<FixedVectorType>(p->getValue()->getType())->getElementType()->getIntegerBitWidth();
            const auto packWidth = p->getFieldWidth()->value();
            assert (sourceWidth == packWidth);
            Value * const base = compileExpression(b, p->getValue(), false);
            const auto result_packs = sourceWidth/2;
            Type * bt = b.getBitBlockType();
            if (LLVM_LIKELY(result_packs > 1)) {
                value = b.CreateAllocaAtEntryPoint(ArrayType::get(bt, result_packs));
            }
            Constant * const ZERO = b.getInt32(0);

            for (unsigned i = 0; i < result_packs; ++i) {
                Value * A = b.CreateLoad(bt, b.CreateGEP(bt, base, {ZERO, b.getInt32(i * 2)}));
                Value * B = b.CreateLoad(bt, b.CreateGEP(bt, base, {ZERO, b.getInt32(i * 2 + 1)}));
                Value * P = b.bitCast(b.hsimd_packh(packWidth, A, B));
                if (LLVM_UNLIKELY(result_packs == 1)) {
                    value = P;
                    break;
                }
                b.CreateStore(P, b.CreateGEP(bt, value, {ZERO, b.getInt32(i)}));
            }
        } else if (const PackL * const p = dyn_cast<PackL>(stmt)) {
            const auto sourceWidth = cast<FixedVectorType>(p->getValue()->getType())->getElementType()->getIntegerBitWidth();
            const auto packWidth = p->getFieldWidth()->value();
            assert (sourceWidth == packWidth);
            Value * const base = compileExpression(b, p->getValue(), false);
            const auto result_packs = sourceWidth/2;
            Type * bt = b.getBitBlockType();
            if (LLVM_LIKELY(result_packs > 1)) {
                value = b.CreateAllocaAtEntryPoint(ArrayType::get(bt, result_packs));
            }
            Constant * const ZERO = b.getInt32(0);
            for (unsigned i = 0; i < result_packs; ++i) {
                Value * A = b.CreateLoad(bt, b.CreateGEP(bt, base, {ZERO, b.getInt32(i * 2)}));
                Value * B = b.CreateLoad(bt, b.CreateGEP(bt, base, {ZERO, b.getInt32(i * 2 + 1)}));
                Value * P = b.bitCast(b.hsimd_packl(packWidth, A, B));
                if (LLVM_UNLIKELY(result_packs == 1)) {
                    value = P;
                    break;
                }
                b.CreateStore(P, b.CreateGEP(b.getBitBlockType(), value, {ZERO, b.getInt32(i)}));
            }
        } else if (const DebugPrint * const d = dyn_cast<DebugPrint>(stmt)) {
          SmallVector<char, 64> tmp;
          raw_svector_ostream name(tmp);
          value = compileExpression(b, stmt->getOperand(0));
          stmt->print(name);
          if (value->getType()->isVectorTy()) {
              b.CallPrintRegister(name.str(), value);
          } else if (value->getType()->isIntegerTy()) {
              b.CallPrintInt(name.str(), value);
          }
        } else if (isa<Ternary>(stmt)) {
            unsigned char const mask = cast<Integer>(stmt->getOperand(0))->value();
            Value * const op0 = compileExpression(b, stmt->getOperand(1));
            Value * const op1 = compileExpression(b, stmt->getOperand(2));
            Value * const op2 = compileExpression(b, stmt->getOperand(3));
            value = b.simd_ternary(mask, op0, op1, op2);
            assert (value->getType() == b.getBitBlockType());
        } else if (const Illustrate * const il = dyn_cast<Illustrate>(stmt)) {
            // Should we use the name as the streamName? what if this is in a loop?
            // TODO: need to fix pablo printer still
            if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
                Value * const op = compileExpression(b, stmt->getOperand(0));
                Constant * const blockWidth = b.getSize(b.getBitBlockWidth());
                Value * const from = b.CreateMul(mIllustratorStrideNum, blockWidth);
                Value * const length = b.CreateSub(blockWidth, b.getScalarField("EOFUnnecessaryData"));
                Value * const to = b.CreateAdd(from, length);

                Constant * kernelName = b.GetString(getName());
                Constant * streamName = b.GetString(il->getName());

                Value * addr = b.CreateAllocaAtEntryPoint(op->getType());
                b.CreateStore(op, addr);

                captureStreamData(b, kernelName, streamName, getHandle(),
                                  mIllustratorStrideNum, op->getType(), MemoryOrdering::RowMajor, addr, from, to);
            }
            return;
        } else if (const IntrinsicCall * const call = dyn_cast<IntrinsicCall>(stmt)) {
            const auto n = call->getNumOperands();
            SmallVector<Value *, 2> argv;
            for (unsigned i = 0; i < n; ++i) {
                PabloAST * const arg = call->getOperand(i);
                argv.push_back(compileExpression(b, arg));
            }

            auto assertArgCount = [&](const size_t expectedArgCount) {
                if (LLVM_UNLIKELY(expectedArgCount != n)) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream out(tmp);
                    out << "PabloCompiler: intrinsic id "
                        << (uint32_t) call->getIntrinsic() << ": "
                        << "invlaid number of arguments, "
                        << "expected " << expectedArgCount << ","
                        << "got " << argv.size();
                    report_fatal_error(StringRef(out.str()));
                }
            };

            switch (call->getIntrinsic()) {
                case Intrinsic::SpanUpTo:
                    assertArgCount(2);
                    value = mCarryManager->subBorrowInBorrowOut(b, stmt, argv[1], argv[0]);
                    break;
                case Intrinsic::InclusiveSpan:
                    assertArgCount(2);
                    value = b.simd_or(argv[1], mCarryManager->subBorrowInBorrowOut(b, stmt, argv[1], argv[0]));
                    break;
                case Intrinsic::ExclusiveSpan:
                    assertArgCount(2);
                    value = b.simd_and(b.simd_not(argv[0]), mCarryManager->subBorrowInBorrowOut(b, stmt, argv[1], argv[0]));
                    break;
                case Intrinsic::PrintRegister:
                    assertArgCount(1);
                    {
                        SmallVector<char, 256> tmp;
                        raw_svector_ostream out(tmp);
                        PabloPrinter::print(cast<PabloAST>(stmt), out);
                        b.CallPrintRegister(out.str(), argv[0]);
                    }
                    value = argv[0];
                    break;
                default:
                    {
                        SmallVector<char, 256> tmp;
                        raw_svector_ostream out(tmp);
                        out << "PabloCompiler: intrinsic id "
                            << (uint32_t) call->getIntrinsic()
                            << " was not recognized by the compiler";
                        report_fatal_error(StringRef(out.str()));
                    }
                    break;
            }

        } else {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "PabloCompiler: ";
            stmt->print(out);
            out << " was not recognized by the compiler";
            report_fatal_error(StringRef(out.str()));
        }
        assert (expr);
        assert (value);
        mMarker[expr] = value;
        if (DebugOptionIsSet(DumpTrace)) {
            dumpValueToConsole(b, expr, value);
        }
    }
}

unsigned getIntegerBitWidth(const Type * ty) {
    if (ty->isArrayTy()) {
        assert (ty->getArrayNumElements() == 1);
        ty = ty->getArrayElementType();
    }
    if (ty->isVectorTy()) {
        assert (cast<FixedVectorType>(ty)->getNumElements() == 0);
        ty = cast<FixedVectorType>(ty)->getElementType();
    }
    return ty->getIntegerBitWidth();
}

Value * PabloCompiler::compileExpression(KernelBuilder & b, const PabloAST * const expr, const bool ensureLoaded) {

    const auto f = mMarker.find(expr);
    Value * value = nullptr;
    if (LLVM_LIKELY(f != mMarker.end())) {
        value = f->second;
    } else {
        if (isa<Integer>(expr)) {
            value = ConstantInt::get(cast<Integer>(expr)->getType(), cast<Integer>(expr)->value());
        } else if (isa<Zeroes>(expr)) {
            value = b.allZeroes();
        } else if (LLVM_UNLIKELY(isa<Ones>(expr))) {
            value = b.allOnes();
        } else if (isa<Extract>(expr)) {
            const Extract * const extract = cast<Extract>(expr);
            const Var * const var = extract->getArray();
            Value * const index = compileExpression(b, extract->getIndex());
            value = getPointerToVar(b, var, index);
        } else if (LLVM_UNLIKELY(isa<Var>(expr))) {
            const Var * const var = cast<Var>(expr);
            if (LLVM_LIKELY(var->isKernelParameter())) {
                const auto ip = b.saveIP();
                Instruction * const inst = mEntryBlock->getFirstNonPHI();
                if (inst) {
                    b.SetInsertPoint(mEntryBlock, inst->getIterator());
                }
                if (var->isScalar()) {
                    value = b.getScalarFieldPtr(var->getName()).first;
                } else if (var->isReadOnly()) {
                    value = b.getInputStreamBlockPtr(var->getName(), b.getInt32(0));
                } else if (var->isReadNone()) {
                    value = b.getOutputStreamBlockPtr(var->getName(), b.getInt32(0));
                }
                if (inst) {
                    b.restoreIP(ip);
                }
            } else { // use before def error
                SmallVector<char, 128> tmp;
                raw_svector_ostream out(tmp);
                out << "PabloCompiler: ";
                expr->print(out);
                out << " is not a scalar value or was used before definition";
                report_fatal_error(StringRef(out.str()));
            }
        } else if (LLVM_UNLIKELY(isa<Operator>(expr))) {
            const Operator * const op = cast<Operator>(expr);
            const PabloAST * lh = op->getLH();
            const PabloAST * rh = op->getRH();
            if (isa<Var>(lh) || isa<Var>(rh)) {
                if (getIntegerBitWidth(lh->getType()) != getIntegerBitWidth(rh->getType())) {
                    llvm::report_fatal_error("Integer types must be identical!");
                }
                const unsigned intWidth = std::min(getIntegerBitWidth(lh->getType()), getIntegerBitWidth(rh->getType()));
                const unsigned maskWidth = b.getBitBlockWidth() / intWidth;
                IntegerType * const maskTy = b.getIntNTy(maskWidth);
                FixedVectorType * const vTy = b.fwVectorType(intWidth);

                Value * baseLhv = nullptr;
                Value * lhvStreamIndex = nullptr;
                if (isa<Extract>(lh)) {
                    lhvStreamIndex = compileExpression(b, cast<Extract>(lh)->getIndex());
                    lh = cast<Extract>(lh)->getArray();
                } else if (isa<Var>(lh)) {
                    lhvStreamIndex = b.getInt32(0);

                } else {
                    baseLhv = b.CreateBitCast(compileExpression(b, lh), vTy);
                }

                Value * baseRhv = nullptr;
                Value * rhvStreamIndex = nullptr;
                if (isa<Extract>(rh)) {
                    rhvStreamIndex = compileExpression(b, cast<Extract>(rh)->getIndex());
                    rh = cast<Extract>(rh)->getArray();
                } else if (isa<Var>(rh)) {
                    rhvStreamIndex = b.getInt32(0);
                } else {
                    baseRhv = b.CreateBitCast(compileExpression(b, rh), vTy);
                }

                const TypeId typeId = op->getClassTypeId();

                if (LLVM_UNLIKELY(typeId == TypeId::Add || typeId == TypeId::Subtract)) {

                    value = b.CreateAllocaAtEntryPoint(vTy, b.getInt32(intWidth));

                    for (unsigned i = 0; i < intWidth; ++i) {
                        llvm::Constant * const index = b.getInt32(i);
                        Value * lhv = nullptr;
                        if (baseLhv) {
                            lhv = baseLhv;
                        } else {
                            lhv = getPointerToVar(b, cast<Var>(lh), lhvStreamIndex, index);
                            lhv = b.CreateBlockAlignedLoad(vTy, lhv);
                        }


                        Value * rhv = nullptr;
                        if (baseRhv) {
                            rhv = baseRhv;
                        } else {
                            rhv = getPointerToVar(b, cast<Var>(rh), rhvStreamIndex, index);
                            rhv = b.CreateBlockAlignedLoad(vTy, rhv);
                        }


                        Value * result = nullptr;
                        if (typeId == TypeId::Add) {
                            result = b.CreateAdd(lhv, rhv);
                        } else { // if (typeId == TypeId::Subtract) {
                            result = b.CreateSub(lhv, rhv);
                        }
                        b.CreateAlignedStore(result, b.CreateGEP(vTy, value, {b.getInt32(0), b.getInt32(i)}), getAlignment(result));
                    }

                } else {



                    value = UndefValue::get(b.fwVectorType(maskWidth));

                    for (unsigned i = 0; i < intWidth; ++i) {
                        llvm::Constant * const index = b.getInt32(i);
                        Value * lhv = nullptr;
                        if (baseLhv) {
                            lhv = baseLhv;
                        } else {
                            lhv = getPointerToVar(b, cast<Var>(lh), lhvStreamIndex, index);
                            lhv = b.CreateBlockAlignedLoad(vTy, lhv);
                        }


                        Value * rhv = nullptr;
                        if (baseRhv) {
                            rhv = baseRhv;
                        } else {
                            rhv = getPointerToVar(b, cast<Var>(rh), rhvStreamIndex, index);
                            rhv = b.CreateBlockAlignedLoad(vTy, rhv);
                        }

                        Value * comp = nullptr;
                        switch (typeId) {
                            case TypeId::GreaterThanEquals:
                            case TypeId::LessThan:
                                comp = b.simd_ult(intWidth, lhv, rhv);
                                break;
                            case TypeId::Equals:
                            case TypeId::NotEquals:
                                comp = b.simd_eq(intWidth, lhv, rhv);
                                break;
                            case TypeId::LessThanEquals:
                            case TypeId::GreaterThan:
                                comp = b.simd_ugt(intWidth, lhv, rhv);
                                break;
                            default: llvm_unreachable("invalid vector operator id");
                        }
                        Value * const mask = b.CreateZExtOrTrunc(b.hsimd_signmask(intWidth, comp), maskTy);
                        value = b.mvmd_insert(maskWidth, value, mask, i);
                    }
                    value = b.CreateBitCast(value, b.getBitBlockType());
                    switch (typeId) {
                        case TypeId::GreaterThanEquals:
                        case TypeId::LessThanEquals:
                        case TypeId::NotEquals:
                            value = b.CreateNot(value);
                        default: break;
                    }
                }

            } else {
                Value * const lhv = compileExpression(b, lh);
                Value * const rhv = compileExpression(b, rh);
                switch (op->getClassTypeId()) {
                    case TypeId::Add:
                        value = b.CreateAdd(lhv, rhv); break;
                    case TypeId::Subtract:
                        value = b.CreateSub(lhv, rhv); break;
                    case TypeId::LessThan:
                        value = b.CreateICmpULT(lhv, rhv); break;
                    case TypeId::LessThanEquals:
                        value = b.CreateICmpULE(lhv, rhv); break;
                    case TypeId::Equals:
                        value = b.CreateICmpEQ(lhv, rhv); break;
                    case TypeId::GreaterThanEquals:
                        value = b.CreateICmpUGE(lhv, rhv); break;
                    case TypeId::GreaterThan:
                        value = b.CreateICmpUGT(lhv, rhv); break;
                    case TypeId::NotEquals:
                        value = b.CreateICmpNE(lhv, rhv); break;
                    default: llvm_unreachable("invalid scalar operator id");
                }
            }
        } else { // use before def error
            std::string tmp;
            raw_string_ostream out(tmp);
            out << "PabloCompiler: ";
            expr->print(out);
            out << " was used before definition";
            report_fatal_error(StringRef(out.str()));
        }
        assert (value);
        // mMarker.insert({expr, value});
    }
    if (LLVM_UNLIKELY(value->getType()->isPointerTy() && ensureLoaded)) {

        size_t align = 0;
        Type * type = expr->getType();
        if (type->isIntegerTy()) {
            align = cast<IntegerType>(type)->getBitWidth() / 8;
        } else {
            type = b.getBitBlockType();
            align = b.getBitBlockWidth() / 8;
        }

        assert (type->getPointerTo() == value->getType());

        value = b.CreateAlignedLoad(type, value, align);
    }
    return value;
}

Value * PabloCompiler::getPointerToVar(KernelBuilder & b, const Var * var, Value * index1, Value * index2)  {

    assert (var && index1);
    if (LLVM_LIKELY(var->isKernelParameter())) {
        Value * ptr = nullptr;
        const auto ip = b.saveIP();
        Instruction * const inst = mEntryBlock->getFirstNonPHI();
        if (inst) {
            b.SetInsertPoint(mEntryBlock, inst->getIterator());
        }
        if (LLVM_UNLIKELY(var->isScalar())) {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << mKernel->getName();
            out << ": cannot index scalar value ";
            var->print(out);
            report_fatal_error(StringRef(out.str()));
        } else if (var->isReadOnly()) {
            if (index2) {
                ptr = b.getInputStreamPackPtr(var->getName(), index1, index2);
            } else {
                ptr = b.getInputStreamBlockPtr(var->getName(), index1);
            }
        } else if (var->isReadNone()) {
            if (index2) {
                ptr = b.getOutputStreamPackPtr(var->getName(), index1, index2);
            } else {
                ptr = b.getOutputStreamBlockPtr(var->getName(), index1);
            }
        } else {
            std::string tmp;
            raw_string_ostream out(tmp);
            out << mKernel->getName();
            out << ": stream ";
            var->print(out);
            out << " cannot be read from or written to";
            report_fatal_error(StringRef(out.str()));
        }
        if (inst) {
            b.restoreIP(ip);
        }
        return ptr;
    } else {
        Value * const ptr = compileExpression(b, var, false);
        SmallVector<Value *, 3> offsets;
        offsets.push_back(ConstantInt::getNullValue(index1->getType()));
        offsets.push_back(index1);
        if (index2) {
            offsets.push_back(index2);
        }
        Type * type = var->getType();
        if (type->isVectorTy()) {
            type = kernel::StreamSetBuffer::resolveType(b, type);
        }
        return b.CreateGEP(type, ptr, offsets);
    }
}

inline std::unique_ptr<CarryManager> makeCarryManager() {
    switch (CarryMode) {
        case PabloCarryMode::BitBlock:
            return std::make_unique<CarryManager>();
        case PabloCarryMode::Compressed:
            return std::make_unique<CompressedCarryManager>();
    }
    llvm_unreachable("Unknown CarryManager type!");
}

PabloCompiler::PabloCompiler(PabloKernel * const kernel)
: BlockKernelCompiler(kernel)
, mKernel(kernel)
, mCarryManager(makeCarryManager())
, mBranchCount(0) {
    assert ("PabloKernel cannot be null!" && kernel);
}

void PabloCompiler::dumpValueToConsole(KernelBuilder & b, const PabloAST * expr, llvm::Value * value) {
    SmallVector<char, 256> tmp;
    raw_svector_ostream name(tmp);
    expr->print(name);
    if (value->getType()->isVectorTy()) {
        b.CallPrintRegister(name.str(), value);
    } else if (value->getType()->isIntegerTy()) {
        b.CallPrintInt(name.str(), value);
    }
}

}
