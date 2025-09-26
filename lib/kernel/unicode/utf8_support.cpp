/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/utf8_support.h>
#include <pablo/builder.hpp>
#include <pablo/pe_var.h>           // for Var
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>

using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;

void UTF8_index::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;
    bool useDirectCC = getInput(0)->getType()->getArrayNumElements() == 1;
    if (useDirectCC) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(pb.createExtract(getInput(0), pb.getInteger(0)));
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getInputStreamSet("source"));
    }

    Zeroes * const ZEROES = pb.createZeroes();
    PabloAST * const u8pfx = ccc->compileCC(makeByte(0xC0, 0xFF), pb);


    Var * const nonFinal = pb.createVar("nonFinal", u8pfx);
    Var * const u8invalid = pb.createVar("u8invalid", ZEROES);
    Var * const valid_pfx = pb.createVar("valid_pfx", u8pfx);

    auto it = pb.createScope();
    pb.createIf(u8pfx, it);
    PabloAST * const u8pfx2 = ccc->compileCC(makeByte(0xC2, 0xDF), it);
    PabloAST * const u8pfx3 = ccc->compileCC(makeByte(0xE0, 0xEF), it);
    PabloAST * const u8pfx4 = ccc->compileCC(makeByte(0xF0, 0xF4), it);

    //
    // Two-byte sequences
    Var * const anyscope = it.createVar("anyscope", ZEROES);
    auto it2 = it.createScope();
    it.createIf(u8pfx2, it2);
    it2.createAssign(anyscope, it2.createAdvance(u8pfx2, 1));


    //
    // Three-byte sequences
    Var * const EF_invalid = it.createVar("EF_invalid", ZEROES);
    auto it3 = it.createScope();
    it.createIf(u8pfx3, it3);
    PabloAST * const u8scope32 = it3.createAdvance(u8pfx3, 1);
    it3.createAssign(nonFinal, it3.createOr(nonFinal, u8scope32));
    PabloAST * const u8scope33 = it3.createAdvance(u8pfx3, 2);
    PabloAST * const u8scope3X = it3.createOr(u8scope32, u8scope33);
    it3.createAssign(anyscope, it3.createOr(anyscope, u8scope3X));

    PabloAST * const advE0 = it3.createAdvance(ccc->compileCC(makeByte(0xE0), it3), 1, "advEO");
    PabloAST * const range80_9F = ccc->compileCC(makeByte(0x80, 0x9F), it3);
    PabloAST * const E0_invalid = it3.createAnd(advE0, range80_9F, "E0_invalid");

    PabloAST * const advED = it3.createAdvance(ccc->compileCC(makeByte(0xED), it3), 1, "advED");
    PabloAST * const rangeA0_BF = ccc->compileCC(makeByte(0xA0, 0xBF), it3);
    PabloAST * const ED_invalid = it3.createAnd(advED, rangeA0_BF, "ED_invalid");

    PabloAST * const EX_invalid = it3.createOr(E0_invalid, ED_invalid);
    it3.createAssign(EF_invalid, EX_invalid);

    //
    // Four-byte sequences
    auto it4 = it.createScope();
    it.createIf(u8pfx4, it4);
    PabloAST * const u8scope42 = it4.createAdvance(u8pfx4, 1, "u8scope42");
    PabloAST * const u8scope43 = it4.createAdvance(u8scope42, 1, "u8scope43");
    PabloAST * const u8scope44 = it4.createAdvance(u8scope43, 1, "u8scope44");
    PabloAST * const u8scope4nonfinal = it4.createOr(u8scope42, u8scope43);
    it4.createAssign(nonFinal, it4.createOr(nonFinal, u8scope4nonfinal));
    PabloAST * const u8scope4X = it4.createOr(u8scope4nonfinal, u8scope44);
    it4.createAssign(anyscope, it4.createOr(anyscope, u8scope4X));
    PabloAST * const F0_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF0), it4), 1), ccc->compileCC(makeByte(0x80, 0x8F), it4));
    PabloAST * const F4_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF4), it4), 1), ccc->compileCC(makeByte(0x90, 0xBF), it4));
    PabloAST * const FX_invalid = it4.createOr(F0_invalid, F4_invalid);
    it4.createAssign(EF_invalid, it4.createOr(EF_invalid, FX_invalid));

    //
    // Invalid cases
    PabloAST * const legalpfx = it.createOr(it.createOr(u8pfx2, u8pfx3), u8pfx4);
    //  Any scope that does not have a suffix byte, and any suffix byte that is not in
    //  a scope is a mismatch, i.e., invalid UTF-8.
    PabloAST * const u8suffix = ccc->compileCC("u8suffix", makeByte(0x80, 0xBF), it);
    PabloAST * const mismatch = it.createXor(anyscope, u8suffix);
    //
    PabloAST * const pfx_invalid = it.createXor(valid_pfx, legalpfx);
    it.createAssign(u8invalid, it.createOr(pfx_invalid, it.createOr(mismatch, EF_invalid)));
    PabloAST * const u8valid = it.createNot(u8invalid, "u8valid");
    //
    it.createAssign(nonFinal, it.createAnd(nonFinal, u8valid));

    Var * const u8index = getOutputStreamVar("u8index");
    PabloAST * u8final = pb.createInFile(pb.createNot(nonFinal));
    if (getNumOfStreamInputs() > 1) {
        u8final = pb.createOr(u8final, getInputStreamSet("u8_LB")[0]);
    }
    pb.createAssign(pb.createExtract(u8index, pb.getInteger(0)), u8final);
}

UTF8_index::UTF8_index(LLVMTypeSystemInterface & ts, StreamSet * Source, StreamSet * u8index, StreamSet * u8_LB)
: PabloKernel(ts, [&]() -> std::string {
    std::stringstream s;
    s << "UTF8_index_";
    s << Source->getNumElements() << "x" << Source->getFieldWidth();
    if (u8_LB) {
        s << "_LB";
    }
    return s.str();}(),
{}, {Binding{"u8index", u8index}}) {
    mInputStreamSets.push_back(Binding{"source", Source});
    if (u8_LB) {
        mInputStreamSets.push_back(Binding{"u8_LB", u8_LB, FixedRate(), Principal()});
    }
}

U8Spans::U8Spans(LLVMTypeSystemInterface & ts, StreamSet * marks, StreamSet * u8index, StreamSet * spans, pablo::BitMovementMode m)
: PabloKernel(ts, "U8Spans_" + marks->shapeString() + pablo::BitMovementMode_string(m), {}, {Binding{"spans", spans}}),
    mBitMovement(m) {
        if (m == pablo::BitMovementMode::LookAhead) {
            mInputStreamSets.push_back(Binding{"marks", marks, FixedRate(1), LookAhead(3)});
            mInputStreamSets.push_back(Binding{"u8index", u8index, FixedRate(1), LookAhead(3)});
        } else {
            mInputStreamSets.push_back(Binding{"marks", marks});
            mInputStreamSets.push_back(Binding{"u8index", u8index});
        }

    }

void U8Spans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> marks = getInputStreamSet("marks");
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    std::vector<PabloAST *> spans(marks.size());
    if (mBitMovement == BitMovementMode::LookAhead) {
        PabloAST * back1InSpan = pb.createNot(u8index);
        PabloAST * ix_or_next = pb.createOr(u8index, pb.createLookahead(u8index, 1));
        PabloAST * back2InSpan = pb.createNot(ix_or_next);
        PabloAST * ix_or_next2 = pb.createOr(ix_or_next, pb.createLookahead(u8index, 2));
        PabloAST * back3InSpan = pb.createNot(ix_or_next2);
        for (unsigned i = 0; i < marks.size(); i++) {
            pb.createIntrinsicCall(pablo::Intrinsic::PrintRegister, marks[i]);
            PabloAST * back1 = pb.createAnd(pb.createLookahead(marks[i], 1), back1InSpan);
            PabloAST * back2 = pb.createAnd(pb.createLookahead(marks[i], 2), back2InSpan);
            PabloAST * back3 = pb.createAnd(pb.createLookahead(marks[i], 3), back3InSpan);
            spans[i] = pb.createOr(marks[i], pb.createOr3(back1, back2, back3));
        }
    } else {
        for (unsigned i = 0; i < marks.size(); i++) {
            pb.createIntrinsicCall(pablo::Intrinsic::PrintRegister, marks[i]);
            spans[i] = pb.createMatchStar(marks[i], pb.createNot(u8index));
        }
    }
    writeOutputStreamSet("spans", spans);
}
