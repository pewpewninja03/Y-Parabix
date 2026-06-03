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
// \ -> \\, " -> \", 0x08 (BS) -> \b, 0x09 (TAB) -> \t, 0x0A(LF) -> \n, 0x0A(FF) -> \f, 0x0D(CR) -> \r
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
    PabloAST * dataMask = getInputStreamSet("dataMask")[0];
    cc::Parabix_CC_Compiler_Builder ccc(basis);
    BixNumCompiler bnc(pb);
    re::CC * quote_backslash = re::makeCC(re::makeCC('"'), re::makeCC('\\'));
    re::CC * unit_controls = re::makeCC(re::makeCC(0x8, 0xA), re::makeCC(0xC, 0XD));
    re::CC * expand_by_1 = re::makeCC(quote_backslash, unit_controls);
    re::CC * expand_by_5 = subtractCC(re::makeCC(0x00, 0x1F), unit_controls);
    PabloAST * Expand1 = pb.createAnd(ccc.compileCC(expand_by_1, pb), dataMask);
    PabloAST * Expand5 = pb.createAnd(ccc.compileCC(expand_by_5, pb), dataMask);
    BixNum Expansion = bnc.AddModular(bnc.Create(Expand5, 5), bnc.Create(Expand1, 1));
    writeOutputStreamSet("insertBixNum", Expansion);
}

StreamSet * EscapeSequenceExpansionBixNum(PipelineBuilder & P, StreamSet * Basis, StreamSet * stringMask) {
    StreamSet * insertBixNum = P.CreateStreamSet(3);  // Max insert is 5, requiring a 3-bit bixnum.
    P.CreateKernelCall<JSON_Escape_Sequence_Expansion>(Basis, stringMask, insertBixNum);
    return insertBixNum;
}

//
// Translating escape sequences, assuming that the necessary
// space for expanded escape sequences have been created by inserting
// zeroes before the escaped character.

class JSON_Escape_Sequence_Translation : public PabloKernel {
public:
    JSON_Escape_Sequence_Translation(LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * spreadMask, StreamSet * translatedBasis)
        : PabloKernel(ts, "JSON_Escape_Sequence_Translation" + Basis->shapeString(),
                      {Binding{"basis", Basis, FixedRate(), LookAhead(1)},
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
    PabloAST * insertMask = pb.createNot(getInputStreamSet("spreadMask")[0]);
    auto ZEROES = pb.createZeroes();
    std::vector<Var *> basisVar(7);
    if (!mUseInOut) {
        for (unsigned i = 0; i < 7; i++) {
            basisVar[i] = pb.createVar("outVar" + std::to_string(i), ZEROES);
        }
    }
    auto nested = pb.createScope();
    pb.createIf(insertMask, nested);
    BixNumCompiler bnc(nested);
    PabloAST * insert1 = nested.createAnd(insertMask, nested.createNot(nested.createAdvance(insertMask, 1)));
    //
    // For all escape sequences, backslash (0x5C) is the first character of the escape sequence.
    // Bits 2, 3, 4 and 6 are set for any escape, bits 0, 1, and 5 are cleared.
    PabloAST * preserve = nested.createNot(insert1);
    std::vector<PabloAST *> outputBasis(7);
    outputBasis[6] = nested.createOr(basis[6], insert1);
    outputBasis[5] = nested.createAnd(basis[5], preserve);
    outputBasis[4] = nested.createOr(basis[4], insert1);
    outputBasis[3] = nested.createOr(basis[3], insert1);
    outputBasis[2] = nested.createOr(basis[2], insert1);
    outputBasis[1] = nested.createAnd(basis[1], preserve);
    outputBasis[0] = nested.createAnd(basis[0], preserve);
    //
    //  Further processing depends on the first escaped position,
    //  that is, is the next basis character to be escaped, or
    //  are there more inserted characters marking a \u00xy sequence.
    //
    PabloAST * escaped_position_1 = nested.createAdvance(insert1, 1);
    PabloAST * u00xy_start = nested.createAnd(escaped_position_1, insertMask);
    PabloAST * unit_escape = nested.createXor(escaped_position_1, u00xy_start);
    //
    // Unit escapes may be escaped quotes or backslashes, but these
    // are already encoded in the basis data, so we only need consider
    // translation of unit controls.
    // 0x08 -> 0x62 (b), 0x09 -> 0x74 (t), 0x0A-> 0x6E (n), 0x0C -> 0x66 (f), 0x0D -> 0x72 (r)
    // We only need consider the low 4 bits of the basis.
    PabloAST * unit_controls = pb.createAnd(unit_escape, bnc.ULT(basis, 0x10));
    // The high 4 bits of the controls are all zero, and must translate to 6 or 7.
    // It is 7 if the low bit of the character to be escaped is 1 (0x09, 0x0D)
    outputBasis[6] = nested.createOr(outputBasis[6], unit_controls);
    outputBasis[5] = nested.createOr(outputBasis[5], unit_controls);
    outputBasis[4] = nested.createOr(outputBasis[4], nested.createAnd(unit_controls, basis[0]));
    //
    // For the low 4 positions, we determine which bits have to be flipped.
    BixNum low4 = bnc.Truncate(basis, 4);
    // bit 3 is flipped for all but LF (0xA) -> 0x6E (\n)
    PabloAST * at_LF = nested.createAnd(unit_controls, bnc.EQ(low4, 0x0A));
    outputBasis[3] = nested.createXor(outputBasis[3], nested.createXor(unit_controls, at_LF));
    // bit 2 is flipped for 0x09 -> 0x74, 0x0A-> 0x6E, 0x0D -> 0x72
    PabloAST * at_TAB = nested.createAnd(unit_controls, bnc.EQ(low4, 0x09));
    PabloAST * at_CR = nested.createAnd(unit_controls, bnc.EQ(low4, 0x0D));
    outputBasis[2] = nested.createXor(outputBasis[2], nested.createOr3(at_TAB, at_LF, at_CR));
    // bit 1 is flipped for 0x08 -> 0x62, 0x0C -> 0x66, 0x0D -> 0x72
    PabloAST * at_BS = nested.createAnd(unit_controls, bnc.EQ(low4, 0x08));
    PabloAST * at_FF = nested.createAnd(unit_controls, bnc.EQ(low4, 0x0C));
    outputBasis[1] = nested.createXor(outputBasis[1], nested.createOr3(at_BS, at_FF, at_CR));
    // bit 0 is zeroed for all unit controls
    outputBasis[0] = nested.createAnd(outputBasis[0], nested.createNot(unit_controls));
    //
    // For hexadecimal escapes (expected to be rare), we will create a further nested block.
    // But first Assign the computed output basis streams, so we have the final values,
    // in the event the hexadecimal block is skipped.
    for (unsigned i = 0; i < 7; i++) {
        nested.createAssign(basisVar[i], outputBasis[i]);
    }
    auto hex_scope = nested.createScope();
    nested.createIf(u00xy_start, hex_scope);
    BixNumCompiler bnc2(hex_scope);
    // The first character after the open escape (\) is the
    // letter u (0x75).
    outputBasis[6] = hex_scope.createOr(outputBasis[6], u00xy_start);
    outputBasis[5] = hex_scope.createOr(outputBasis[5], u00xy_start);
    outputBasis[4] = hex_scope.createOr(outputBasis[4], u00xy_start);
    outputBasis[2] = hex_scope.createOr(outputBasis[2], u00xy_start);
    outputBasis[0] = hex_scope.createOr(outputBasis[0], u00xy_start);
    //
    PabloAST * hex_pos1 = hex_scope.createAdvance(u00xy_start, 1);
    PabloAST * hex_pos2 = hex_scope.createAdvance(hex_pos1, 1);
    PabloAST * hex_pos3 = hex_scope.createAdvance(hex_pos2, 1);
    PabloAST * hex_pos4 = hex_scope.createAdvance(hex_pos3, 1);
    // We need to encode a hexadecimal digit A to F at the final
    // position if the low 4 digits of the control character is
    // 10 or greater.
    PabloAST * finalHexAF = hex_scope.createAnd(bnc2.UGE(low4, 10), hex_pos4);
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
    PabloAST * bit4moved = hex_scope.createLookahead(basis[4], 1);
    outputBasis[0] = hex_scope.createOr(outputBasis[0], hex_scope.createAnd(bit4moved, hex_pos3));
    //
    // The final hex digit is based on the 4 low bits of the escaped control.
    // For hex A-F values, the low 4 bits are in the range 10-15, so we subract 9.
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

void EscapeStringTranslation(PipelineBuilder & P, StreamSet * BasisBits, StreamSet * EscapeSpreadMask, StreamSet * EscapedBasis) {
    P.CreateKernelCall<JSON_Escape_Sequence_Translation>(BasisBits, EscapeSpreadMask, EscapedBasis);
    SHOW_BIXNUM(EscapedBasis);
}

const std::vector<std::string> JSON_Kind_REs = {"null", "true", "false", "-?(?:0|[1-9][0-9]*)(?:[.][0-9]+)?(?:[Ee][-+][0-9]+)?", "\".*\""};

void JSON_Value_Matching(PipelineBuilder & P, JSON_ValueKind val_bitset, StreamSet * BasisBits, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * matches) {
    RE_CompilerContext ctxt;
    ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
    ctxt.setMatchRegions(fieldStarts, fieldFollows);
    std::string matchRegex = "";
    unsigned i = 0;
    for (JSON_ValueKind k = NullLiteral; k <= QuotedString; k = static_cast<JSON_ValueKind>(k << 1)) {
        if ((k & val_bitset) == k) {
            if (matchRegex != "") {
                matchRegex += "|";
            }
            matchRegex += JSON_Kind_REs[i];
        }
        i++;
    }
    re::RE * matchRE = toUTF8(re::RE_Parser::parse("^(?:" + matchRegex + ")$"));
    RE_PipelineBuilder RE_PB(P, ctxt);
    StreamSet * const val_matches = P.CreateStreamSet(1);
    RE_PB.matchSearchPipeline(matchRE, val_matches);
    P.CreateKernelCall<MatchedLinesKernel>(val_matches, fieldFollows, matches);
    SHOW_STREAM(matches);
}

void JSON_Value_Quoted(PipelineBuilder & P, StreamSet * BasisBits, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * quoted) {
    RE_CompilerContext ctxt;
    ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
    ctxt.setMatchRegions(fieldStarts, fieldFollows);
    re::RE * quotedRE = toUTF8(re::RE_Parser::parse("^\".*\"$"));
    RE_PipelineBuilder RE_PB(P, ctxt);
    StreamSet * const quotedValues = P.CreateStreamSet(1);
    RE_PB.matchSearchPipeline(quotedRE, quotedValues);
    P.CreateKernelCall<MatchedLinesKernel>(quotedValues, fieldFollows, quoted);
    SHOW_STREAM(quoted);
}

}
