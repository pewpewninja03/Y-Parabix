/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <json/json_support.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_kernel.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pablo_kernel.h>
#include <re/adt/re_re.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <kernel/re/regexp_kernel.h>
#include <re/adt/adt.h>
#include <re/adt/re_re.h>
#include <re/parse/parser.h>
#include <re/unicode/regex_passes.h>
#include <grep/grep_engine.h>
#include <grep/grep_kernel.h>
#include <kernel/streamutils/pdep_kernel.h>

//
// Certain characters within JSON strings must be escaped, including
// double quotes, backslashes and ASCII controls 0x00-0X1F.
//
// For such characters, two kernels are defined:
//
// (a)  JSON_Escape_Sequence_Expansion which computes a bixnum for the number of
//      positions to insert to accomodate each escape sequence.
// (b)  JSON_Escape_Sequence_Translation which performs the transformation of
//      raw characters requiring escaping to their escaped form.
//
// The escape sequences for backslash and several common control characters
// generate two character escape sequences (insertion of one position required).
//
// \ -> \\, 0x08 (BS) -> \b, 0x09 (TAB) -> \t, 0x0A(LF) -> \n, 0x0A(FF) -> \f, 0x0D(CR) -> \r
//
// Other controls generate 6-character escape sequences of the form \u00xy
// where xy are the two hexadecimal digits of the control sequence value.
//
using namespace kernel;
using namespace pablo;

namespace json {

class JSON_Escape_Sequence_Expansion : public PabloKernel {
public:
    JSON_Escape_Sequence_Expansion(LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * dataMask, StreamSet * insertBixNum)
        : PabloKernel(ts, "JSON_Escape_Sequence_Expansion" + Basis->shapeString(),
                      {Binding{"basis", Basis},
                       Binding{"dataMask", dataMask}},
                      {Binding{"insertBixNum", insertBixNum}}) {}
protected:
    void generatePabloMethod() override;
};

void JSON_Escape_Sequence_Expansion::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    cc::Parabix_CC_Compiler_Builder ccc(basis);
    BixNumCompiler bnc(pb);    
    re::CC * unit_controls = re::makeCC(re::makeCC(0x8, 0xA), re::makeCC(0xC, 0XD));
    re::CC * expand_by_1 = re::makeCC(re::makeCC('\\'), unit_controls);
    re::CC * expand_by_5 = subtractCC(re::makeCC(0x00, 0x1F), unit_controls);
    PabloAST * Expand1 = ccc.compileCC(expand_by_1, pb);
    PabloAST * Expand5 = ccc.compileCC(expand_by_5, pb);
    BixNum Expansion = bnc.AddModular(bnc.Create(Expand5, 5), bnc.Create(Expand1, 1));
    writeOutputStreamSet("insertBixNum", Expansion);
}

//
// Translating escape sequences, assuming that the necessary
// space for expanded escape sequences have been created by inserting
// zeroes after the escaped character.

class JSON_Escape_Sequence_Translation : public PabloKernel {
public:
    JSON_Escape_Sequence_Translation(LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * spreadMask, StreamSet * translatedBasis)
        : PabloKernel(ts, "JSON_Escape_Sequence_Translation" + Basis->shapeString(),
                      {Binding{"basis", Basis},
                       Binding{"spreadMask", spreadMask}},
                      {Binding{"translatedBasis", translatedBasis, FixedRate(), InOut("basis")}}) {
            mUseInOut = !codegen::DebugOptionIsSet(codegen::DisableInOutAttributes);
        }
protected:
    void generatePabloMethod() override;
    bool mUseInOut;
};

void JSON_Escape_Sequence_Translation::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    cc::Parabix_CC_Compiler_Builder ccc(basis);
    BixNumCompiler bnc(pb);
    PabloAST * spreadMask = getInputStreamSet("spreadMask")[0];
    auto ZEROES = pb.createZeroes();
    std::vector<Var *> basisVar(7);
    if (!mUseInOut) {
        for (unsigned i = 0; i < 7; i++) {
            basisVar[i] = pb.createVar("outVar" + std::to_string(i), ZEROES);
        }
    }
    re::CC * unit_controls = re::makeCC(re::makeCC(0x8, 0xA), re::makeCC(0xC, 0XD));
    re::CC * expand_by_1 = re::makeCC(re::makeCC('\\'), unit_controls);
    re::CC * expand_by_5 = subtractCC(re::makeCC(0x00, 0x1F), unit_controls);
    PabloAST * Expand1 = ccc.compileCC(expand_by_1, pb);
    PabloAST * Expand5 = ccc.compileCC(expand_by_5, pb);
    // Ensure that we don't treat inserted nulls as characters to escape.
    Expand5 = pb.createAnd(Expand5, spreadMask);
    PabloAST * AnyEscape = pb.createOr(Expand1, Expand5);
    auto nested = pb.createScope();
    pb.createIf(AnyEscape, nested);
    // All bit modifications for escape translation involve 7 bit ASCII values only.
    // Create Vars to receive the produced values.
    if (mUseInOut) {
        for (unsigned i = 0; i < 7; i++) {
            basisVar[i] = nested.createVar("outVar" + std::to_string(i), ZEROES);
        }
    }
    std::vector<PabloAST *> outputBasis(7);
    // For all escape sequences, backslash (0x5C) is the first character of the escape sequence.
    // Bits 2, 3, 4 and 6 are set for any escape, bits 0, 1, and 5 are cleared.
    PabloAST * preserve = nested.createNot(AnyEscape);
    outputBasis[6] = nested.createOr(basis[6], AnyEscape);
    outputBasis[5] = nested.createAnd(basis[5], preserve);
    outputBasis[4] = nested.createOr(basis[4], AnyEscape);
    outputBasis[3] = nested.createOr(basis[3], AnyEscape);
    outputBasis[2] = nested.createOr(basis[2], AnyEscape);
    outputBasis[1] = nested.createAnd(basis[1], preserve);
    outputBasis[0] = nested.createAnd(basis[0], preserve);
    // To escape a backslash, we create a second backslash (0x5C).
    // We only need to or in bits 2, 3, 4 and 6.
    PabloAST * bslash = ccc.compileCC(re::makeCC('\\'), nested);
    PabloAST * bslash_next = nested.createAdvance(bslash, 1);
    outputBasis[6] = nested.createOr(outputBasis[6], bslash_next);
    outputBasis[4] = nested.createOr(outputBasis[4], bslash_next);
    outputBasis[3] = nested.createOr(outputBasis[3], bslash_next);
    outputBasis[2] = nested.createOr(outputBasis[2], bslash_next);
    //
    // Translation of unit controls
    // 0x08 -> 0x62 (b), 0x09 -> 0x74 (t), 0x0A-> 0x6E , 0x0C -> 0x66, 0x0D -> 0x72
    // Note that the high hex digit is either 6 or 7. It is 7 if
    // the low bit of the character to be escaped is 1 (0x09, 0x0D)
    PabloAST * controls_next = nested.createAdvance(nested.createXor(Expand1, bslash), 1);
    outputBasis[6] = nested.createOr(outputBasis[6], controls_next);
    outputBasis[5] = nested.createOr(outputBasis[5], controls_next);
    outputBasis[4] = nested.createOr(outputBasis[4], nested.createAnd(controls_next, nested.createAdvance(basis[0], 1)));
    //
    // bit 3 is set only for LF (0xA) -> \n
    PabloAST * LF_next = nested.createAdvance(ccc.compileCC(re::makeCC(0x0A), nested), 1);
    outputBasis[3] = nested.createOr(outputBasis[3], nested.createAnd(controls_next, LF_next));
    // bit 2 is set for 0x09 -> 0x74 (t), 0x0A-> 0x6E , 0x0C -> 0x66
    PabloAST * TAB_next = nested.createAdvance(ccc.compileCC(re::makeCC(0x09), nested), 1);
    PabloAST * FF_next = nested.createAdvance(ccc.compileCC(re::makeCC(0x0C), nested), 1);
    PabloAST * LF_TAB_FF_next = nested.createOr3(LF_next, TAB_next, FF_next);
    outputBasis[2] = nested.createOr(outputBasis[2], nested.createAnd(controls_next, LF_TAB_FF_next));
    // bit 1 is set for all except tab.
    outputBasis[1] = nested.createOr(outputBasis[1], nested.createAnd(controls_next, nested.createNot(TAB_next)));
    //
    // For hexadecimal escapes (expected to be rare), we will create a further nested block.
    // But first Assign the computed output basis streams, so we have the final values,
    // in the event the hexadecimal block is skipped.
    for (unsigned i = 0; i < 7; i++) {
        nested.createAssign(basisVar[i], outputBasis[i]);
    }
    auto hex_scope = nested.createScope();
    nested.createIf(Expand5, hex_scope);
    re::CC * hexA_F = re::makeCC(re::makeCC(0x0A, 0x0F), re::makeCC(0xC, 0XD));
    // The first character after the open escape (\) is the
    // letter u (0x75).
    PabloAST * u_pos = hex_scope.createAdvance(Expand5, 1);
    outputBasis[6] = hex_scope.createOr(outputBasis[6], u_pos);
    outputBasis[5] = hex_scope.createOr(outputBasis[5], u_pos);
    outputBasis[4] = hex_scope.createOr(outputBasis[4], u_pos);
    outputBasis[2] = hex_scope.createOr(outputBasis[2], u_pos);
    outputBasis[0] = hex_scope.createOr(outputBasis[0], u_pos);
    //
    PabloAST * hex_pos1 = hex_scope.createAdvance(u_pos, 1);
    PabloAST * hex_pos2 = hex_scope.createAdvance(hex_pos1, 1);
    PabloAST * hex_pos3 = hex_scope.createAdvance(hex_pos2, 1);
    PabloAST * hex_pos4 = hex_scope.createAdvance(hex_pos3, 1);
    PabloAST * finalHexAF = hex_scope.createAdvance(ccc.compileCC(hexA_F, hex_scope), 5);
    finalHexAF = hex_scope.createAnd(finalHexAF, hex_pos4);
    //
    // The hexadecimal sequence to be produced is in the pattern 00[01][0-9A-F].
    // The numerals [0-9] are all in the ASCII range 0x30-0x39.
    // Set the high 4 bits of each digit accordingly.
    PabloAST * scope3X = hex_scope.createOr3(hex_pos1, hex_pos2, hex_pos3);
    scope3X = hex_scope.createOr(scope3X, hex_scope.createXor(hex_pos4, finalHexAF));
    outputBasis[5] = hex_scope.createOr(outputBasis[5], scope3X);
    outputBasis[4] = hex_scope.createOr(outputBasis[4], scope3X);
    //
    // For hexadecimal A-F values, the ASCII range is 0x41 to 0x46.
    // The high 4 bits are 0100.
    outputBasis[6] = hex_scope.createOr(outputBasis[6], hex_scope.createAnd(hex_pos4, finalHexAF));
    //
    //  Now set the low 4 bits. 
    //  For the first two hex digits, the low 4 bits remain as 0.
    //  For the third digit, the low bit is 0 or 1, depending
    //  on bit 4 of the character to be escaped.
    PabloAST * bit4moved = hex_scope.createAdvance(basis[4], 4);
    outputBasis[0] = hex_scope.createOr(outputBasis[0], hex_scope.createAnd(bit4moved, hex_pos3));
    //
    // The final hex digit is based on the 4 low bits of the
    // escaped character, moved forward 5 positions.
    std::vector<PabloAST *> low4(4);
    for (unsigned i = 0; i < 4; i++) {
        low4[i] = hex_scope.createAdvance(basis[i], 5);
    }
    // For hex A-F values, the low 4 bits are in the range 10-15, so we subract 9.
    BixNumCompiler bnc2(hex_scope);
    BixNum AF_low4 = bnc2.SubModular(low4, 9);
    for (unsigned i = 0; i < 4; i++) {
        PabloAST * bit = hex_scope.createSel(finalHexAF, AF_low4[i], low4[i]);
        outputBasis[i] = hex_scope.createOr(outputBasis[i], hex_scope.createAnd(bit, hex_pos4));
    }
    // Now assign the computed values so that they are available in the outer scope.
    for (unsigned i = 0; i < 7; i++) {
        hex_scope.createAssign(basisVar[i], outputBasis[i]);
    }
    //
    // Finally write out the values.
    Var * outVar = getOutputStreamVar("translatedBasis");
    for (unsigned i = 0; i < 7; i++) {
        if (mUseInOut) {
            nested.createAssign(nested.createExtract(outVar, nested.getInteger(i)), basisVar[i]);
        } else {
            pb.createAssign(pb.createExtract(outVar, pb.getInteger(i)), basisVar[i]);
        }
    }
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

void EscapeStringSpecials(PipelineBuilder & P, StreamSet * BasisBits, StreamSet * stringMask, StreamSet * EscapedBasis) {
    StreamSet * JSON_Escape_Insert_Bixnum = P.CreateStreamSet(3);
    P.CreateKernelCall<JSON_Escape_Sequence_Expansion>(BasisBits, stringMask, JSON_Escape_Insert_Bixnum);
    SHOW_BIXNUM(JSON_Escape_Insert_Bixnum);

    StreamSet * const EscapeSpreadMask = P.CreateStreamSet(1);
    InsertionSpreadMask(P, JSON_Escape_Insert_Bixnum, EscapeSpreadMask, kernel::InsertPosition::After);
    SHOW_STREAM(EscapeSpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(BasisBits->getNumElements());
    SpreadByMask(P, EscapeSpreadMask, BasisBits, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    P.CreateKernelCall<JSON_Escape_Sequence_Translation>(ExpandedBasis, EscapeSpreadMask, EscapedBasis);
    SHOW_BIXNUM(EscapedBasis);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)

const std::vector<std::string> AtomicREs = {"null", "true", "false", "-?(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[Ee][-+][0-9]+)?"};

void JSON_Value_Matching(PipelineBuilder & P, JSON_Atomic atom_bitset, StreamSet * BasisBits, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * matches) {
    RE_CompilerContext ctxt;
    ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
    ctxt.setMatchRegions(fieldStarts, fieldFollows);
    std::string matchRegex = "";
    unsigned i = 0;
    for (JSON_Atomic k = NullLiteral; k <= NumericLiteral; k = static_cast<JSON_Atomic>(k << 1)) {
        if ((k & atom_bitset) == k) {
            if (matchRegex != "") {
                matchRegex += "|";
            }
            matchRegex += AtomicREs[i];
        }
        i++;
    }
    re::RE * matchRE = toUTF8(re::RE_Parser::parse("^(?:" + matchRegex + ")$"));
    RE_PipelineBuilder RE_PB(P, ctxt);
    StreamSet * const atomicMatches = P.CreateStreamSet(1);
    RE_PB.matchSearchPipeline(matchRE, atomicMatches);
    P.CreateKernelCall<MatchedLinesKernel>(atomicMatches, fieldFollows, matches);
    SHOW_STREAM(matches);
}

}
