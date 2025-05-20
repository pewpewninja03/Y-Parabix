/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <fcntl.h>
#include <string>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/codegenstate.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/bixnum/bixnum.h>
#include <grep/grep_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/core/unicode_set.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decomposition Options", "Decomposition Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));
static cl::opt<bool> U21("U21", cl::desc("perform character translation via 21-bit Unicode"),  cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

std::vector<re::CC *> Hangul_Composables() {
    UCD::UnicodeSet Hangul_L(Hangul::LBase, Hangul::LBase + Hangul::LCount - 1);
    UCD::UnicodeSet Hangul_V(Hangul::VBase, Hangul::VBase + Hangul::VCount - 1);
    UCD::UnicodeSet Hangul_T(Hangul::TBase, Hangul::TBase + Hangul::TCount - 1);
    UCD::UnicodeSet Hangul_LV;
    for (unsigned i = 0; i < Hangul::LCount * Hangul::VCount; i++) {
        Hangul_LV.insert(Hangul::SBase + i * Hangul::TCount);
    }
    return {re::makeCC(Hangul_L, &cc::Unicode),
            re::makeCC(Hangul_V, &cc::Unicode),
            re::makeCC(Hangul_LV, &cc::Unicode),
            re::makeCC(Hangul_T, &cc::Unicode)};
}

class Hangul_Composition : public pablo::PabloKernel {
public:
    Hangul_Composition(LLVMTypeSystemInterface & ts,
                          StreamSet * Basis, StreamSet * L_V_T_Composables, 
                          StreamSet * Output_Basis, StreamSet * SelectionMask);
protected:
    void generatePabloMethod() override;
};

Hangul_Composition::Hangul_Composition (LLVMTypeSystemInterface & ts,
                                              StreamSet * Basis, StreamSet * L_V_T_Composables,
                                              StreamSet * Output_Basis, StreamSet * SelectionMask)
: PabloKernel(ts, "Hangul_Composition" + Basis->shapeString(),
// inputs
    {Binding{"Basis", Basis, FixedRate(1), LookAhead(5)}, Binding{"L_V_T_Composables", L_V_T_Composables, FixedRate(1), LookAhead(5)}},
// output
    {Binding{"Output_Basis", Output_Basis}, Binding{"SelectionMask", SelectionMask}}) {
}

void Hangul_Composition::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> L_V_T_Composables = getInputStreamSet("L_V_T_Composables");
    PabloAST * Hangul_L = L_V_T_Composables[0];
    PabloAST * Hangul_V = L_V_T_Composables[1];
    PabloAST * Hangul_LV = L_V_T_Composables[2];
    PabloAST * Hangul_T = L_V_T_Composables[3];
    //
    //  Set up variables to receive the output basis bit streams.
    std::vector<Var *> outputVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        outputVar[i] = pb.createVar("outputVar" + std::to_string(i), Basis[i]);
    }
    //  Selection mask is all ones to select every position by default.
    Var * maskVar = pb.createVar("maskVar", pb.createOnes());
    //
    // All calculations depend on having an initial L, if there are none,
    // we can skip.
    auto nested = pb.createScope();
    
    pb.createIf(pb.createOr(Hangul_L, Hangul_LV), nested);
    BixNumCompiler bnc(nested);
    //
    // First determine whether we have LV and/or LVT sequences.
    // LV sequences are identified either as precomposed LV characters
    // or combinations of an L followed by a V.
    // For UTF-8 each L, V, T are 3 code units in length.
    // For Unicode or UTF-16, only a single code unit is needed.
    unsigned codeUnitsPerChar = Basis.size() > 8 ? 1 : 3;
    PabloAST * V_ahead = nested.createLookahead(Hangul_V, codeUnitsPerChar);
    PabloAST * L_prefix = nested.createAnd(Hangul_L, V_ahead);
    PabloAST * V_suffix = nested.createAdvance(L_prefix, codeUnitsPerChar);
    PabloAST * T_ahead = nested.createLookahead(Hangul_T, codeUnitsPerChar);
    PabloAST * LV_sequence = nested.createOr(Hangul_LV, V_suffix);
    PabloAST * LVT_sequence = nested.createAnd(LV_sequence, T_ahead);
    PabloAST * T_suffix = nested.createAdvance(LVT_sequence, codeUnitsPerChar);
    //
    // Given LV/LVT combinations, character replacement will be performed at
    // the position of the V or LV character.   Leading L characters and
    // trailing T characters are marked for deletion.
    //
    PabloAST * Deletable = nested.createOr(L_prefix, T_suffix);
    // For UTF-8 the two suffix bytes will also be deleted.
    if (Basis.size() == 8) {
        Deletable = nested.createOr3(Deletable,
                                     nested.createAdvance(Deletable, 1),
                                     nested.createAdvance(Deletable, 2));
    }
    //  Zero out any positions in the mask that are to be deleted.
    nested.createAssign(maskVar, nested.createXor(maskVar, Deletable));
    //
    //  Now calculate replacement precomposed characters.   There will
    //  be five significant bits for each L, V and T character to produce
    //  the corresponding LIndex, VIndex and TIndex values.
    const unsigned index_bits = 5;
    unsigned indexMask = (1 << index_bits) - 1;  // mask to select low bits of index only
    //
    //  The LIndex of any L character is determined by taking the low 5 bits
    //  after subtracting LBase (0x1100) from the L codepoint value.   But
    //  since the low bits of LBase are 0, no actual subtraction is needed.
    //  However, we do need to move the bits forward to the V_suffix position.
    //  For UTF-8, the LIndex is actually at the final byte position
    //  so advance of a single position moves the index value correctly.
    //
    BixNum LIndex(index_bits);
    for (unsigned i = 0; i < index_bits; i++) {
        LIndex[i] = nested.createAnd(nested.createAdvance(Basis[i], 1), V_suffix);
    }
    //
    //  The VIndex of any V character is determined by taking the low 5 bits
    //  after subtracting VBase (0x1161) from the V codepoint value.   We use
    //  5-bit subtraction by masking off the higher bits first.
    BixNum VIndex(index_bits);
    //  The VIndex is required at the V_suffix position, but
    //  for UTF-8 must be shifted back 2 from the 3rd byte position.
    const unsigned shiftBack = codeUnitsPerChar - 1;
    if (Basis.size() == 8) {
        for (unsigned i = 0; i < index_bits; i++) {
            VIndex[i] = nested.createLookahead(Basis[i], shiftBack);
        }
    } else {
        VIndex = bnc.Truncate(Basis, 5);
    }
    VIndex = bnc.Select(V_suffix, bnc.SubModular(VIndex, Hangul::VBase & indexMask));
    //
    //  We now combine the Lindex and Vindex values to compute the SIndex of
    //  the LV combination and then the full codepoint value by adding in SBase.
    BixNum LV_combined = bnc.AddFull(bnc.MulFull(LIndex, Hangul::VCount), VIndex);
    BixNum SIndex = bnc.MulFull(LV_combined, Hangul::TCount);
    BixNum Computed_LV = bnc.Select(V_suffix, bnc.AddFull(SIndex, Hangul::SBase));
    //
    // Now we merge this with any data for precomposed LV characters.
    BixNum Merged_LV(Computed_LV.size());
    if (Basis.size() == 8) {
        for (unsigned i = 0; i < Computed_LV.size(); i++) {
            if (i < 6) {
                Merged_LV[i] = nested.createSel(Hangul_LV, nested.createLookahead(Basis[i], 2), Computed_LV[i]);
            } else if (i < 12) {
                Merged_LV[i] = nested.createSel(Hangul_LV, nested.createLookahead(Basis[i-6], 1), Computed_LV[i]);
            } else {
                Merged_LV[i] = nested.createSel(Hangul_LV, Basis[i-12], Computed_LV[i]);
            }
        }
    } else {
        Merged_LV = bnc.Select(Hangul_LV, Basis, Computed_LV);
    }
    //
    //  The TIndex value defaults to zero, or is determined from shifting bits
    //  back from the T_suffix position.
    BixNum TIndex(index_bits);
    for (unsigned i = 0; i < index_bits; i++) {
        TIndex[i] = nested.createLookahead(Basis[i], codeUnitsPerChar + shiftBack);
    }
    TIndex = bnc.Select(LVT_sequence, bnc.SubModular(TIndex, Hangul::TBase & indexMask));
    //
    //  Finally compute the composed characters
    BixNum Composed = bnc.AddFull(Merged_LV, TIndex);
    if (Basis.size() == 8) {
        PabloAST * LV_suffix1 = nested.createAdvance(LV_sequence, 1);
        PabloAST * LV_suffix2 = nested.createAdvance(LV_sequence, 2);
        //
        //  Since V, LV and LVT characters all have the same 3-byte UTF-8
        //  structure, only the low 6 bits of each UTF-8 byte change.
        for (unsigned i = 0; i < 6; i++) {
            PabloAST * updated = Basis[i];
            if (i < 4) {
                // At prefix (LV_sequence) positions, set 4 data bits, keep high 4 bits unchanged.
                updated = nested.createSel(LV_sequence, Composed[i+12], updated);
            }
            // Suffix bytes - set 6 data bits for each, keep top two bits unchanged.
            updated = nested.createSel(LV_suffix1, nested.createAdvance(Composed[i+6], 1), updated);
            updated = nested.createSel(LV_suffix2, nested.createAdvance(Composed[i], 2), updated);
            nested.createAssign(outputVar[i], updated);
        }
    } else {
        Composed = bnc.ZeroExtend(Composed, Basis.size());
        for (unsigned i = 0; i < Basis.size(); i++) {
            nested.createAssign(outputVar[i], nested.createSel(LV_sequence, Composed[i], Basis[i]));
        }
    }
    writeOutputStreamSet("Output_Basis", outputVar);
    writeOutputStreamSet("SelectionMask", std::vector<Var *>{maskVar});
}

class ApplyTransform : public pablo::PabloKernel {
public:
    ApplyTransform(LLVMTypeSystemInterface & ts,
                   StreamSet * Basis, StreamSet * Xfrms, StreamSet * Output);
protected:
    void generatePabloMethod() override;
};

ApplyTransform::ApplyTransform (LLVMTypeSystemInterface & ts,
                                StreamSet * Basis, StreamSet * Xfrms, StreamSet * Output)
: PabloKernel(ts, "xfrm_" + std::to_string(Basis->getNumElements()) + "x1_" + std::to_string(Xfrms->getNumElements()),
// inputs
{Binding{"basis", Basis},
 Binding{"xfrms", Xfrms}
},
// output
{Binding{"output_basis", Output}}) {
}

void ApplyTransform::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> xfrms = getInputStreamSet("xfrms");
    std::vector<PabloAST *> transformed(basis.size());
    for (unsigned i = 0; i < basis.size(); i++) {
        if (i < xfrms.size()) {
            transformed[i] = pb.createXor(xfrms[i], basis[i]);
        } else {
            transformed[i] = basis[i];
        }
    }
    writeOutputStreamSet("output_basis", transformed);
}


class DelPriorToSelectMask : public PabloKernel {
public:
    DelPriorToSelectMask
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * DeletePrior,
                                       StreamSet * SelectMask);
protected:
    void generatePabloMethod() override;
};

DelPriorToSelectMask::DelPriorToSelectMask
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * DeletePrior,
                                   StreamSet * SelectMask)
: PabloKernel(ts, "DelPriorToSelectMask",
{Binding{"Basis", Basis}, Binding{"DeletePrior", DeletePrior, FixedRate(), LookAhead(4)}},
{Binding{"SelectMask", SelectMask}}) {}

void DelPriorToSelectMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    PabloAST * DelPrior = getInputStreamSet("DeletePrior")[0];
    BixNumCompiler bnc(pb);
    PabloAST * pfx4 = bnc.UGE(Basis, 0xF0);
    PabloAST * pfx3or4 = bnc.UGE(Basis, 0xE0);
    PabloAST * pfx = bnc.UGE(Basis, 0xC2);
    PabloAST * pfx3 = pb.createXor(pfx3or4, pfx4);
    PabloAST * pfx2 = pb.createXor(pfx, pfx3or4);
    PabloAST * delpfx4 = pb.createAnd(pfx4, pb.createLookahead(DelPrior, 4));
    PabloAST * thirdlast = pb.createOr(pfx3, pb.createAdvance(pfx4, 1));
    PabloAST * del3 = pb.createAnd(thirdlast, pb.createLookahead(DelPrior, 3));
    PabloAST * secondlast = pb.createOr(pfx2, pb.createAdvance(thirdlast, 1));
    PabloAST * del2 = pb.createAnd(secondlast, pb.createLookahead(DelPrior, 2));
    PabloAST * del1 = pb.createLookahead(DelPrior, 1);
    PabloAST * del = pb.createOr(pb.createOr(delpfx4, del3), pb.createOr(del2, del1));
    std::vector<PabloAST *> select_streamset = {pb.createInFile(pb.createNot(del))};
    writeOutputStreamSet("SelectMask", select_streamset);
}
using namespace UCD;

const static UnicodeSet::run_t __uset_0_runs[] = {{Empty, 77}, {Mixed, 1}, {Empty, 34738}};
const static UnicodeSet::bitquad_t __uset_0_quads[] = {0x40000000};
const static UnicodeSet uset_0{__uset_0_runs, 3, __uset_0_quads, 1};
const UnicodeSet & E0_9be_uset = uset_0;
const static UnicodeSet::run_t __uset_1_runs[] = {{Empty, 78}, {Mixed, 1}, {Empty, 34737}};
const static UnicodeSet::bitquad_t __uset_1_quads[] = {0x00000080};
const static UnicodeSet uset_1{__uset_1_runs, 3, __uset_1_quads, 1};
const UnicodeSet & E0_9c7_uset = uset_1;
const static UnicodeSet::run_t __uset_2_runs[] = {{Empty, 78}, {Mixed, 1}, {Empty, 34737}};
const static UnicodeSet::bitquad_t __uset_2_quads[] = {0x00800000};
const static UnicodeSet uset_2{__uset_2_runs, 3, __uset_2_quads, 1};
const UnicodeSet & E0_9d7_uset = uset_2;
const static UnicodeSet::run_t __uset_3_runs[] = {{Empty, 89}, {Mixed, 1}, {Empty, 34726}};
const static UnicodeSet::bitquad_t __uset_3_quads[] = {0x40000000};
const static UnicodeSet uset_3{__uset_3_runs, 3, __uset_3_quads, 1};
const UnicodeSet & E0_b3e_uset = uset_3;
const static UnicodeSet::run_t __uset_4_runs[] = {{Empty, 90}, {Mixed, 1}, {Empty, 34725}};
const static UnicodeSet::bitquad_t __uset_4_quads[] = {0x00000080};
const static UnicodeSet uset_4{__uset_4_runs, 3, __uset_4_quads, 1};
const UnicodeSet & E0_b47_uset = uset_4;
const static UnicodeSet::run_t __uset_5_runs[] = {{Empty, 90}, {Mixed, 1}, {Empty, 34725}};
const static UnicodeSet::bitquad_t __uset_5_quads[] = {0x00400000};
const static UnicodeSet uset_5{__uset_5_runs, 3, __uset_5_quads, 1};
const UnicodeSet & E0_b56_uset = uset_5;
const static UnicodeSet::run_t __uset_6_runs[] = {{Empty, 90}, {Mixed, 1}, {Empty, 34725}};
const static UnicodeSet::bitquad_t __uset_6_quads[] = {0x00800000};
const static UnicodeSet uset_6{__uset_6_runs, 3, __uset_6_quads, 1};
const UnicodeSet & E0_b57_uset = uset_6;
const static UnicodeSet::run_t __uset_7_runs[] = {{Empty, 92}, {Mixed, 1}, {Empty, 34723}};
const static UnicodeSet::bitquad_t __uset_7_quads[] = {0x00040000};
const static UnicodeSet uset_7{__uset_7_runs, 3, __uset_7_quads, 1};
const UnicodeSet & E0_b92_uset = uset_7;
const static UnicodeSet::run_t __uset_8_runs[] = {{Empty, 93}, {Mixed, 1}, {Empty, 34722}};
const static UnicodeSet::bitquad_t __uset_8_quads[] = {0x40000000};
const static UnicodeSet uset_8{__uset_8_runs, 3, __uset_8_quads, 1};
const UnicodeSet & E0_bbe_uset = uset_8;
const static UnicodeSet::run_t __uset_9_runs[] = {{Empty, 94}, {Mixed, 1}, {Empty, 34721}};
const static UnicodeSet::bitquad_t __uset_9_quads[] = {0x00000040};
const static UnicodeSet uset_9{__uset_9_runs, 3, __uset_9_quads, 1};
const UnicodeSet & E0_bc6_uset = uset_9;
const static UnicodeSet::run_t __uset_10_runs[] = {{Empty, 94}, {Mixed, 1}, {Empty, 34721}};
const static UnicodeSet::bitquad_t __uset_10_quads[] = {0x00000080};
const static UnicodeSet uset_10{__uset_10_runs, 3, __uset_10_quads, 1};
const UnicodeSet & E0_bc7_uset = uset_10;
const static UnicodeSet::run_t __uset_11_runs[] = {{Empty, 94}, {Mixed, 1}, {Empty, 34721}};
const static UnicodeSet::bitquad_t __uset_11_quads[] = {0x00800000};
const static UnicodeSet uset_11{__uset_11_runs, 3, __uset_11_quads, 1};
const UnicodeSet & E0_bd7_uset = uset_11;
const static UnicodeSet::run_t __uset_12_runs[] = {{Empty, 101}, {Mixed, 1}, {Empty, 34714}};
const static UnicodeSet::bitquad_t __uset_12_quads[] = {0x80000000};
const static UnicodeSet uset_12{__uset_12_runs, 3, __uset_12_quads, 1};
const UnicodeSet & E0_cbf_uset = uset_12;
const static UnicodeSet::run_t __uset_13_runs[] = {{Empty, 102}, {Mixed, 1}, {Empty, 34713}};
const static UnicodeSet::bitquad_t __uset_13_quads[] = {0x00000004};
const static UnicodeSet uset_13{__uset_13_runs, 3, __uset_13_quads, 1};
const UnicodeSet & E0_cc2_uset = uset_13;
const static UnicodeSet::run_t __uset_14_runs[] = {{Empty, 102}, {Mixed, 1}, {Empty, 34713}};
const static UnicodeSet::bitquad_t __uset_14_quads[] = {0x00000040};
const static UnicodeSet uset_14{__uset_14_runs, 3, __uset_14_quads, 1};
const UnicodeSet & E0_cc6_uset = uset_14;
const static UnicodeSet::run_t __uset_15_runs[] = {{Empty, 102}, {Mixed, 1}, {Empty, 34713}};
const static UnicodeSet::bitquad_t __uset_15_quads[] = {0x00000400};
const static UnicodeSet uset_15{__uset_15_runs, 3, __uset_15_quads, 1};
const UnicodeSet & E0_cca_uset = uset_15;
const static UnicodeSet::run_t __uset_16_runs[] = {{Empty, 102}, {Mixed, 1}, {Empty, 34713}};
const static UnicodeSet::bitquad_t __uset_16_quads[] = {0x00200000};
const static UnicodeSet uset_16{__uset_16_runs, 3, __uset_16_quads, 1};
const UnicodeSet & E0_cd5_uset = uset_16;
const static UnicodeSet::run_t __uset_17_runs[] = {{Empty, 102}, {Mixed, 1}, {Empty, 34713}};
const static UnicodeSet::bitquad_t __uset_17_quads[] = {0x00400000};
const static UnicodeSet uset_17{__uset_17_runs, 3, __uset_17_quads, 1};
const UnicodeSet & E0_cd6_uset = uset_17;
const static UnicodeSet::run_t __uset_18_runs[] = {{Empty, 105}, {Mixed, 1}, {Empty, 34710}};
const static UnicodeSet::bitquad_t __uset_18_quads[] = {0x40000000};
const static UnicodeSet uset_18{__uset_18_runs, 3, __uset_18_quads, 1};
const UnicodeSet & E0_d3e_uset = uset_18;
const static UnicodeSet::run_t __uset_19_runs[] = {{Empty, 106}, {Mixed, 1}, {Empty, 34709}};
const static UnicodeSet::bitquad_t __uset_19_quads[] = {0x00000040};
const static UnicodeSet uset_19{__uset_19_runs, 3, __uset_19_quads, 1};
const UnicodeSet & E0_d46_uset = uset_19;
const static UnicodeSet::run_t __uset_20_runs[] = {{Empty, 106}, {Mixed, 1}, {Empty, 34709}};
const static UnicodeSet::bitquad_t __uset_20_quads[] = {0x00000080};
const static UnicodeSet uset_20{__uset_20_runs, 3, __uset_20_quads, 1};
const UnicodeSet & E0_d47_uset = uset_20;
const static UnicodeSet::run_t __uset_21_runs[] = {{Empty, 106}, {Mixed, 1}, {Empty, 34709}};
const static UnicodeSet::bitquad_t __uset_21_quads[] = {0x00800000};
const static UnicodeSet uset_21{__uset_21_runs, 3, __uset_21_quads, 1};
const UnicodeSet & E0_d57_uset = uset_21;
const static UnicodeSet::run_t __uset_22_runs[] = {{Empty, 110}, {Mixed, 1}, {Empty, 34705}};
const static UnicodeSet::bitquad_t __uset_22_quads[] = {0x00008000};
const static UnicodeSet uset_22{__uset_22_runs, 3, __uset_22_quads, 1};
const UnicodeSet & E0_dcf_uset = uset_22;
const static UnicodeSet::run_t __uset_23_runs[] = {{Empty, 110}, {Mixed, 1}, {Empty, 34705}};
const static UnicodeSet::bitquad_t __uset_23_quads[] = {0x02000000};
const static UnicodeSet uset_23{__uset_23_runs, 3, __uset_23_quads, 1};
const UnicodeSet & E0_dd9_uset = uset_23;
const static UnicodeSet::run_t __uset_24_runs[] = {{Empty, 110}, {Mixed, 1}, {Empty, 34705}};
const static UnicodeSet::bitquad_t __uset_24_quads[] = {0x80000000};
const static UnicodeSet uset_24{__uset_24_runs, 3, __uset_24_quads, 1};
const UnicodeSet & E0_ddf_uset = uset_24;
const static UnicodeSet::run_t __uset_25_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_25_quads[] = {0x00000001};
const static UnicodeSet uset_25{__uset_25_runs, 3, __uset_25_quads, 1};
const UnicodeSet & E0_f40_uset = uset_25;
const static UnicodeSet::run_t __uset_26_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_26_quads[] = {0x00000004};
const static UnicodeSet uset_26{__uset_26_runs, 3, __uset_26_quads, 1};
const UnicodeSet & E0_f42_uset = uset_26;
const static UnicodeSet::run_t __uset_27_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_27_quads[] = {0x00001000};
const static UnicodeSet uset_27{__uset_27_runs, 3, __uset_27_quads, 1};
const UnicodeSet & E0_f4c_uset = uset_27;
const static UnicodeSet::run_t __uset_28_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_28_quads[] = {0x00020000};
const static UnicodeSet uset_28{__uset_28_runs, 3, __uset_28_quads, 1};
const UnicodeSet & E0_f51_uset = uset_28;
const static UnicodeSet::run_t __uset_29_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_29_quads[] = {0x00400000};
const static UnicodeSet uset_29{__uset_29_runs, 3, __uset_29_quads, 1};
const UnicodeSet & E0_f56_uset = uset_29;
const static UnicodeSet::run_t __uset_30_runs[] = {{Empty, 122}, {Mixed, 1}, {Empty, 34693}};
const static UnicodeSet::bitquad_t __uset_30_quads[] = {0x08000000};
const static UnicodeSet uset_30{__uset_30_runs, 3, __uset_30_quads, 1};
const UnicodeSet & E0_f5b_uset = uset_30;
const static UnicodeSet::run_t __uset_31_runs[] = {{Empty, 124}, {Mixed, 1}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_31_quads[] = {0x00010000};
const static UnicodeSet uset_31{__uset_31_runs, 3, __uset_31_quads, 1};
const UnicodeSet & E0_f90_uset = uset_31;
const static UnicodeSet::run_t __uset_32_runs[] = {{Empty, 124}, {Mixed, 1}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_32_quads[] = {0x00040000};
const static UnicodeSet uset_32{__uset_32_runs, 3, __uset_32_quads, 1};
const UnicodeSet & E0_f92_uset = uset_32;
const static UnicodeSet::run_t __uset_33_runs[] = {{Empty, 124}, {Mixed, 1}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_33_quads[] = {0x10000000};
const static UnicodeSet uset_33{__uset_33_runs, 3, __uset_33_quads, 1};
const UnicodeSet & E0_f9c_uset = uset_33;
const static UnicodeSet::run_t __uset_34_runs[] = {{Empty, 125}, {Mixed, 1}, {Empty, 34690}};
const static UnicodeSet::bitquad_t __uset_34_quads[] = {0x00000002};
const static UnicodeSet uset_34{__uset_34_runs, 3, __uset_34_quads, 1};
const UnicodeSet & E0_fa1_uset = uset_34;
const static UnicodeSet::run_t __uset_35_runs[] = {{Empty, 125}, {Mixed, 1}, {Empty, 34690}};
const static UnicodeSet::bitquad_t __uset_35_quads[] = {0x00000040};
const static UnicodeSet uset_35{__uset_35_runs, 3, __uset_35_quads, 1};
const UnicodeSet & E0_fa6_uset = uset_35;
const static UnicodeSet::run_t __uset_36_runs[] = {{Empty, 125}, {Mixed, 1}, {Empty, 34690}};
const static UnicodeSet::bitquad_t __uset_36_quads[] = {0x00000800};
const static UnicodeSet uset_36{__uset_36_runs, 3, __uset_36_quads, 1};
const UnicodeSet & E0_fab_uset = uset_36;
const static UnicodeSet::run_t __uset_37_runs[] = {{Empty, 125}, {Mixed, 1}, {Empty, 34690}};
const static UnicodeSet::bitquad_t __uset_37_quads[] = {0x00200000};
const static UnicodeSet uset_37{__uset_37_runs, 3, __uset_37_quads, 1};
const UnicodeSet & E0_fb5_uset = uset_37;
const static UnicodeSet::run_t __uset_38_runs[] = {{Empty, 125}, {Mixed, 1}, {Empty, 34690}};
const static UnicodeSet::bitquad_t __uset_38_quads[] = {0x00800000};
const static UnicodeSet uset_38{__uset_38_runs, 3, __uset_38_quads, 1};
const UnicodeSet & E0_fb7_uset = uset_38;
const static UnicodeSet::run_t __uset_39_runs[] = {{Empty, 129}, {Mixed, 1}, {Empty, 34686}};
const static UnicodeSet::bitquad_t __uset_39_quads[] = {0x00000020};
const static UnicodeSet uset_39{__uset_39_runs, 3, __uset_39_quads, 1};
const UnicodeSet & E1_1025_uset = uset_39;
const static UnicodeSet::run_t __uset_40_runs[] = {{Empty, 129}, {Mixed, 1}, {Empty, 34686}};
const static UnicodeSet::bitquad_t __uset_40_quads[] = {0x00004000};
const static UnicodeSet uset_40{__uset_40_runs, 3, __uset_40_quads, 1};
const UnicodeSet & E1_102e_uset = uset_40;
const static UnicodeSet::run_t __uset_41_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_41_quads[] = {0x00000020};
const static UnicodeSet uset_41{__uset_41_runs, 3, __uset_41_quads, 1};
const UnicodeSet & E1_1b05_uset = uset_41;
const static UnicodeSet::run_t __uset_42_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_42_quads[] = {0x00000080};
const static UnicodeSet uset_42{__uset_42_runs, 3, __uset_42_quads, 1};
const UnicodeSet & E1_1b07_uset = uset_42;
const static UnicodeSet::run_t __uset_43_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_43_quads[] = {0x00000200};
const static UnicodeSet uset_43{__uset_43_runs, 3, __uset_43_quads, 1};
const UnicodeSet & E1_1b09_uset = uset_43;
const static UnicodeSet::run_t __uset_44_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_44_quads[] = {0x00000800};
const static UnicodeSet uset_44{__uset_44_runs, 3, __uset_44_quads, 1};
const UnicodeSet & E1_1b0b_uset = uset_44;
const static UnicodeSet::run_t __uset_45_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_45_quads[] = {0x00002000};
const static UnicodeSet uset_45{__uset_45_runs, 3, __uset_45_quads, 1};
const UnicodeSet & E1_1b0d_uset = uset_45;
const static UnicodeSet::run_t __uset_46_runs[] = {{Empty, 216}, {Mixed, 1}, {Empty, 34599}};
const static UnicodeSet::bitquad_t __uset_46_quads[] = {0x00020000};
const static UnicodeSet uset_46{__uset_46_runs, 3, __uset_46_quads, 1};
const UnicodeSet & E1_1b11_uset = uset_46;
const static UnicodeSet::run_t __uset_47_runs[] = {{Empty, 217}, {Mixed, 1}, {Empty, 34598}};
const static UnicodeSet::bitquad_t __uset_47_quads[] = {0x00200000};
const static UnicodeSet uset_47{__uset_47_runs, 3, __uset_47_quads, 1};
const UnicodeSet & E1_1b35_uset = uset_47;
const static UnicodeSet::run_t __uset_48_runs[] = {{Empty, 217}, {Mixed, 1}, {Empty, 34598}};
const static UnicodeSet::bitquad_t __uset_48_quads[] = {0x04000000};
const static UnicodeSet uset_48{__uset_48_runs, 3, __uset_48_quads, 1};
const UnicodeSet & E1_1b3a_uset = uset_48;
const static UnicodeSet::run_t __uset_49_runs[] = {{Empty, 217}, {Mixed, 1}, {Empty, 34598}};
const static UnicodeSet::bitquad_t __uset_49_quads[] = {0x10000000};
const static UnicodeSet uset_49{__uset_49_runs, 3, __uset_49_quads, 1};
const UnicodeSet & E1_1b3c_uset = uset_49;
const static UnicodeSet::run_t __uset_50_runs[] = {{Empty, 217}, {Mixed, 1}, {Empty, 34598}};
const static UnicodeSet::bitquad_t __uset_50_quads[] = {0x40000000};
const static UnicodeSet uset_50{__uset_50_runs, 3, __uset_50_quads, 1};
const UnicodeSet & E1_1b3e_uset = uset_50;
const static UnicodeSet::run_t __uset_51_runs[] = {{Empty, 217}, {Mixed, 1}, {Empty, 34598}};
const static UnicodeSet::bitquad_t __uset_51_quads[] = {0x80000000};
const static UnicodeSet uset_51{__uset_51_runs, 3, __uset_51_quads, 1};
const UnicodeSet & E1_1b3f_uset = uset_51;
const static UnicodeSet::run_t __uset_52_runs[] = {{Empty, 218}, {Mixed, 1}, {Empty, 34597}};
const static UnicodeSet::bitquad_t __uset_52_quads[] = {0x00000004};
const static UnicodeSet uset_52{__uset_52_runs, 3, __uset_52_quads, 1};
const UnicodeSet & E1_1b42_uset = uset_52;
const static UnicodeSet::run_t __uset_53_runs[] = {{Empty, 2185}, {Mixed, 1}, {Empty, 32630}};
const static UnicodeSet::bitquad_t __uset_53_quads[] = {0x00000080};
const static UnicodeSet uset_53{__uset_53_runs, 3, __uset_53_quads, 1};
const UnicodeSet & F0_11127_uset = uset_53;
const static UnicodeSet::run_t __uset_54_runs[] = {{Empty, 2185}, {Mixed, 1}, {Empty, 32630}};
const static UnicodeSet::bitquad_t __uset_54_quads[] = {0x00020000};
const static UnicodeSet uset_54{__uset_54_runs, 3, __uset_54_quads, 1};
const UnicodeSet & F0_11131_uset = uset_54;
const static UnicodeSet::run_t __uset_55_runs[] = {{Empty, 2185}, {Mixed, 1}, {Empty, 32630}};
const static UnicodeSet::bitquad_t __uset_55_quads[] = {0x00040000};
const static UnicodeSet uset_55{__uset_55_runs, 3, __uset_55_quads, 1};
const UnicodeSet & F0_11132_uset = uset_55;
const static UnicodeSet::run_t __uset_56_runs[] = {{Empty, 2201}, {Mixed, 1}, {Empty, 32614}};
const static UnicodeSet::bitquad_t __uset_56_quads[] = {0x40000000};
const static UnicodeSet uset_56{__uset_56_runs, 3, __uset_56_quads, 1};
const UnicodeSet & F0_1133e_uset = uset_56;
const static UnicodeSet::run_t __uset_57_runs[] = {{Empty, 2202}, {Mixed, 1}, {Empty, 32613}};
const static UnicodeSet::bitquad_t __uset_57_quads[] = {0x00000080};
const static UnicodeSet uset_57{__uset_57_runs, 3, __uset_57_quads, 1};
const UnicodeSet & F0_11347_uset = uset_57;
const static UnicodeSet::run_t __uset_58_runs[] = {{Empty, 2202}, {Mixed, 1}, {Empty, 32613}};
const static UnicodeSet::bitquad_t __uset_58_quads[] = {0x00800000};
const static UnicodeSet uset_58{__uset_58_runs, 3, __uset_58_quads, 1};
const UnicodeSet & F0_11357_uset = uset_58;
const static UnicodeSet::run_t __uset_59_runs[] = {{Empty, 2204}, {Mixed, 1}, {Empty, 32611}};
const static UnicodeSet::bitquad_t __uset_59_quads[] = {0x00000004};
const static UnicodeSet uset_59{__uset_59_runs, 3, __uset_59_quads, 1};
const UnicodeSet & F0_11382_uset = uset_59;
const static UnicodeSet::run_t __uset_60_runs[] = {{Empty, 2204}, {Mixed, 1}, {Empty, 32611}};
const static UnicodeSet::bitquad_t __uset_60_quads[] = {0x00000010};
const static UnicodeSet uset_60{__uset_60_runs, 3, __uset_60_quads, 1};
const UnicodeSet & F0_11384_uset = uset_60;
const static UnicodeSet::run_t __uset_61_runs[] = {{Empty, 2204}, {Mixed, 1}, {Empty, 32611}};
const static UnicodeSet::bitquad_t __uset_61_quads[] = {0x00000800};
const static UnicodeSet uset_61{__uset_61_runs, 3, __uset_61_quads, 1};
const UnicodeSet & F0_1138b_uset = uset_61;
const static UnicodeSet::run_t __uset_62_runs[] = {{Empty, 2204}, {Mixed, 1}, {Empty, 32611}};
const static UnicodeSet::bitquad_t __uset_62_quads[] = {0x00010000};
const static UnicodeSet uset_62{__uset_62_runs, 3, __uset_62_quads, 1};
const UnicodeSet & F0_11390_uset = uset_62;
const static UnicodeSet::run_t __uset_63_runs[] = {{Empty, 2205}, {Mixed, 1}, {Empty, 32610}};
const static UnicodeSet::bitquad_t __uset_63_quads[] = {0x01000000};
const static UnicodeSet uset_63{__uset_63_runs, 3, __uset_63_quads, 1};
const UnicodeSet & F0_113b8_uset = uset_63;
const static UnicodeSet::run_t __uset_64_runs[] = {{Empty, 2205}, {Mixed, 1}, {Empty, 32610}};
const static UnicodeSet::bitquad_t __uset_64_quads[] = {0x08000000};
const static UnicodeSet uset_64{__uset_64_runs, 3, __uset_64_quads, 1};
const UnicodeSet & F0_113bb_uset = uset_64;
const static UnicodeSet::run_t __uset_65_runs[] = {{Empty, 2206}, {Mixed, 1}, {Empty, 32609}};
const static UnicodeSet::bitquad_t __uset_65_quads[] = {0x00000004};
const static UnicodeSet uset_65{__uset_65_runs, 3, __uset_65_quads, 1};
const UnicodeSet & F0_113c2_uset = uset_65;
const static UnicodeSet::run_t __uset_66_runs[] = {{Empty, 2206}, {Mixed, 1}, {Empty, 32609}};
const static UnicodeSet::bitquad_t __uset_66_quads[] = {0x00000200};
const static UnicodeSet uset_66{__uset_66_runs, 3, __uset_66_quads, 1};
const UnicodeSet & F0_113c9_uset = uset_66;
const static UnicodeSet::run_t __uset_67_runs[] = {{Empty, 2213}, {Mixed, 1}, {Empty, 32602}};
const static UnicodeSet::bitquad_t __uset_67_quads[] = {0x00010000};
const static UnicodeSet uset_67{__uset_67_runs, 3, __uset_67_quads, 1};
const UnicodeSet & F0_114b0_uset = uset_67;
const static UnicodeSet::run_t __uset_68_runs[] = {{Empty, 2213}, {Mixed, 1}, {Empty, 32602}};
const static UnicodeSet::bitquad_t __uset_68_quads[] = {0x02000000};
const static UnicodeSet uset_68{__uset_68_runs, 3, __uset_68_quads, 1};
const UnicodeSet & F0_114b9_uset = uset_68;
const static UnicodeSet::run_t __uset_69_runs[] = {{Empty, 2213}, {Mixed, 1}, {Empty, 32602}};
const static UnicodeSet::bitquad_t __uset_69_quads[] = {0x04000000};
const static UnicodeSet uset_69{__uset_69_runs, 3, __uset_69_quads, 1};
const UnicodeSet & F0_114ba_uset = uset_69;
const static UnicodeSet::run_t __uset_70_runs[] = {{Empty, 2213}, {Mixed, 1}, {Empty, 32602}};
const static UnicodeSet::bitquad_t __uset_70_quads[] = {0x20000000};
const static UnicodeSet uset_70{__uset_70_runs, 3, __uset_70_quads, 1};
const UnicodeSet & F0_114bd_uset = uset_70;
const static UnicodeSet::run_t __uset_71_runs[] = {{Empty, 2221}, {Mixed, 1}, {Empty, 32594}};
const static UnicodeSet::bitquad_t __uset_71_quads[] = {0x00008000};
const static UnicodeSet uset_71{__uset_71_runs, 3, __uset_71_quads, 1};
const UnicodeSet & F0_115af_uset = uset_71;
const static UnicodeSet::run_t __uset_72_runs[] = {{Empty, 2221}, {Mixed, 1}, {Empty, 32594}};
const static UnicodeSet::bitquad_t __uset_72_quads[] = {0x01000000};
const static UnicodeSet uset_72{__uset_72_runs, 3, __uset_72_quads, 1};
const UnicodeSet & F0_115b8_uset = uset_72;
const static UnicodeSet::run_t __uset_73_runs[] = {{Empty, 2221}, {Mixed, 1}, {Empty, 32594}};
const static UnicodeSet::bitquad_t __uset_73_quads[] = {0x02000000};
const static UnicodeSet uset_73{__uset_73_runs, 3, __uset_73_quads, 1};
const UnicodeSet & F0_115b9_uset = uset_73;
const static UnicodeSet::run_t __uset_74_runs[] = {{Empty, 2249}, {Mixed, 1}, {Empty, 32566}};
const static UnicodeSet::bitquad_t __uset_74_quads[] = {0x00010000};
const static UnicodeSet uset_74{__uset_74_runs, 3, __uset_74_quads, 1};
const UnicodeSet & F0_11930_uset = uset_74;
const static UnicodeSet::run_t __uset_75_runs[] = {{Empty, 2249}, {Mixed, 1}, {Empty, 32566}};
const static UnicodeSet::bitquad_t __uset_75_quads[] = {0x00200000};
const static UnicodeSet uset_75{__uset_75_runs, 3, __uset_75_quads, 1};
const UnicodeSet & F0_11935_uset = uset_75;
const static UnicodeSet::run_t __uset_76_runs[] = {{Empty, 2824}, {Mixed, 1}, {Empty, 31991}};
const static UnicodeSet::bitquad_t __uset_76_quads[] = {0x40000000};
const static UnicodeSet uset_76{__uset_76_runs, 3, __uset_76_quads, 1};
const UnicodeSet & F0_1611e_uset = uset_76;
const static UnicodeSet::run_t __uset_77_runs[] = {{Empty, 2824}, {Mixed, 1}, {Empty, 31991}};
const static UnicodeSet::bitquad_t __uset_77_quads[] = {0x80000000};
const static UnicodeSet uset_77{__uset_77_runs, 3, __uset_77_quads, 1};
const UnicodeSet & F0_1611f_uset = uset_77;
const static UnicodeSet::run_t __uset_78_runs[] = {{Empty, 2825}, {Mixed, 1}, {Empty, 31990}};
const static UnicodeSet::bitquad_t __uset_78_quads[] = {0x00000001};
const static UnicodeSet uset_78{__uset_78_runs, 3, __uset_78_quads, 1};
const UnicodeSet & F0_16120_uset = uset_78;
const static UnicodeSet::run_t __uset_79_runs[] = {{Empty, 2825}, {Mixed, 1}, {Empty, 31990}};
const static UnicodeSet::bitquad_t __uset_79_quads[] = {0x00000002};
const static UnicodeSet uset_79{__uset_79_runs, 3, __uset_79_quads, 1};
const UnicodeSet & F0_16121_uset = uset_79;
const static UnicodeSet::run_t __uset_80_runs[] = {{Empty, 2825}, {Mixed, 1}, {Empty, 31990}};
const static UnicodeSet::bitquad_t __uset_80_quads[] = {0x00000004};
const static UnicodeSet uset_80{__uset_80_runs, 3, __uset_80_quads, 1};
const UnicodeSet & F0_16122_uset = uset_80;
const static UnicodeSet::run_t __uset_81_runs[] = {{Empty, 2825}, {Mixed, 1}, {Empty, 31990}};
const static UnicodeSet::bitquad_t __uset_81_quads[] = {0x00000200};
const static UnicodeSet uset_81{__uset_81_runs, 3, __uset_81_quads, 1};
const UnicodeSet & F0_16129_uset = uset_81;
const static UnicodeSet::run_t __uset_82_runs[] = {{Empty, 2923}, {Mixed, 1}, {Empty, 31892}};
const static UnicodeSet::bitquad_t __uset_82_quads[] = {0x00000008};
const static UnicodeSet uset_82{__uset_82_runs, 3, __uset_82_quads, 1};
const UnicodeSet & F0_16d63_uset = uset_82;
const static UnicodeSet::run_t __uset_83_runs[] = {{Empty, 2923}, {Mixed, 1}, {Empty, 31892}};
const static UnicodeSet::bitquad_t __uset_83_quads[] = {0x00000080};
const static UnicodeSet uset_83{__uset_83_runs, 3, __uset_83_quads, 1};
const UnicodeSet & F0_16d67_uset = uset_83;
const static UnicodeSet::run_t __uset_84_runs[] = {{Empty, 2923}, {Mixed, 1}, {Empty, 31892}};
const static UnicodeSet::bitquad_t __uset_84_quads[] = {0x00000200};
const static UnicodeSet uset_84{__uset_84_runs, 3, __uset_84_quads, 1};
const UnicodeSet & F0_16d69_uset = uset_84;
const static UnicodeSet::run_t __uset_90_runs[] = {{Empty, 1}, {Mixed, 3}, {Empty, 34812}};
const static UnicodeSet::bitquad_t __uset_90_quads[] = {0x70000000, 0x07fdfbf6, 0x07fdfbf6};
const static UnicodeSet uset_90{__uset_90_runs, 3, __uset_90_quads, 3};
const UnicodeSet & ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a_uset = uset_90;
const static UnicodeSet::run_t __uset_91_runs[] = {
{Empty, 2}, {Mixed, 2}, {Empty, 1}, {Mixed, 4}, {Empty, 1}, {Mixed, 2},
{Empty, 1}, {Mixed, 1}, {Empty, 3}, {Mixed, 1}, {Empty, 10}, {Mixed, 3},
{Empty, 15}, {Mixed, 2}, {Empty, 1955}, {Mixed, 1}, {Empty, 2},
{Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_91_quads[] = {
0x00000408, 0x00000408, 0x00000100, 0x00308484, 0x00308484, 0x000c000c,
0x0c003000, 0x80000f03, 0x00018003, 0x00000300, 0x82a20000, 0x82a25222,
0x00004e22, 0x5f7f0000, 0x000407db, 0x00001000, 0x00e38000};
const static UnicodeSet uset_91{__uset_91_runs, 19, __uset_91_quads, 17};
const UnicodeSet & ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7_uset = uset_91;
const static UnicodeSet::run_t __uset_92_runs[] = {{Empty, 26}, {Mixed, 1}, {Empty, 34789}};
const static UnicodeSet::bitquad_t __uset_92_quads[] = {0x00000010};
const static UnicodeSet uset_92{__uset_92_runs, 3, __uset_92_quads, 1};
const UnicodeSet & CD_344_uset = uset_92;
const static UnicodeSet::run_t __uset_93_runs[] = {
{Empty, 26}, {Mixed, 1}, {Empty, 96}, {Mixed, 2}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_93_quads[] = {0x00000010, 0x00280000, 0x00000002};
const static UnicodeSet uset_93{__uset_93_runs, 5, __uset_93_quads, 3};
const UnicodeSet & CD_344_f73_5_81_uset = uset_93;
const static UnicodeSet::run_t __uset_94_runs[] = {{Empty, 123}, {Mixed, 1}, {Empty, 34692}};
const static UnicodeSet::bitquad_t __uset_94_quads[] = {0x00080000};
const static UnicodeSet uset_94{__uset_94_runs, 3, __uset_94_quads, 1};
const UnicodeSet & E0_f73_uset = uset_94;
const static UnicodeSet::run_t __uset_95_runs[] = {{Empty, 123}, {Mixed, 2}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_95_quads[] = {0x00280000, 0x00000002};
const static UnicodeSet uset_95{__uset_95_runs, 3, __uset_95_quads, 2};
const UnicodeSet & E0_f73_5_81_uset = uset_95;
const static UnicodeSet::run_t __uset_96_runs[] = {{Empty, 123}, {Mixed, 1}, {Empty, 34692}};
const static UnicodeSet::bitquad_t __uset_96_quads[] = {0x00200000};
const static UnicodeSet uset_96{__uset_96_runs, 3, __uset_96_quads, 1};
const UnicodeSet & E0_f75_uset = uset_96;
const static UnicodeSet::run_t __uset_97_runs[] = {{Empty, 124}, {Mixed, 1}, {Empty, 34691}};
const static UnicodeSet::bitquad_t __uset_97_quads[] = {0x00000002};
const static UnicodeSet uset_97{__uset_97_runs, 3, __uset_97_quads, 1};
const UnicodeSet & E0_f81_uset = uset_97;
const static UnicodeSet::run_t __uset_300_runs[] = {{Empty, 26}, {Mixed, 2}, {Empty, 34788}};
const static UnicodeSet::bitquad_t __uset_300_quads[] = {0x0000000b, 0x00100000};
const static UnicodeSet uset_300{__uset_300_runs, 3, __uset_300_quads, 2};
const UnicodeSet & CD_340_1_3_74_uset = uset_300;
const static UnicodeSet::run_t __uset_301_runs[] = {
{Empty, 26}, {Mixed, 1}, {Empty, 1}, {Mixed, 1}, {Empty, 34787}};
const static UnicodeSet::bitquad_t __uset_301_quads[] = {0x00000008, 0x00000080};
const static UnicodeSet uset_301{__uset_301_runs, 5, __uset_301_quads, 2};
const UnicodeSet & CD_343_87_uset = uset_301;
const static UnicodeSet::run_t __uset_302_runs[] = {{Empty, 27}, {Mixed, 1}, {Empty, 34788}};
const static UnicodeSet::bitquad_t __uset_302_quads[] = {0x00100000};
const static UnicodeSet uset_302{__uset_302_runs, 3, __uset_302_quads, 1};
const UnicodeSet & CD_374_uset = uset_302;
const static UnicodeSet::run_t __uset_303_runs[] = {{Empty, 27}, {Mixed, 2}, {Empty, 34787}};
const static UnicodeSet::bitquad_t __uset_303_quads[] = {0x00100000, 0x00000080};
const static UnicodeSet uset_303{__uset_303_runs, 3, __uset_303_quads, 2};
const UnicodeSet & CD_374_87_uset = uset_303;
const static UnicodeSet::run_t __uset_304_runs[] = {{Empty, 27}, {Mixed, 1}, {Empty, 34788}};
const static UnicodeSet::bitquad_t __uset_304_quads[] = {0x40100000};
const static UnicodeSet uset_304{__uset_304_runs, 3, __uset_304_quads, 1};
const UnicodeSet & CD_374_e_uset = uset_304;
const static UnicodeSet::run_t __uset_305_runs[] = {{Empty, 27}, {Mixed, 1}, {Empty, 34788}};
const static UnicodeSet::bitquad_t __uset_305_quads[] = {0x40000000};
const static UnicodeSet uset_305{__uset_305_runs, 3, __uset_305_quads, 1};
const UnicodeSet & CD_37e_uset = uset_305;
const static UnicodeSet::run_t __uset_306_runs[] = {{Empty, 28}, {Mixed, 1}, {Empty, 34787}};
const static UnicodeSet::bitquad_t __uset_306_quads[] = {0x00000080};
const static UnicodeSet uset_306{__uset_306_runs, 3, __uset_306_quads, 1};
const UnicodeSet & CE_387_uset = uset_306;
const static UnicodeSet::run_t __uset_307_runs[] = {{Empty, 251}, {Mixed, 1}, {Empty, 34564}};
const static UnicodeSet::bitquad_t __uset_307_quads[] = {0x00020000};
const static UnicodeSet uset_307{__uset_307_runs, 3, __uset_307_quads, 1};
const UnicodeSet & E1_1f71_uset = uset_307;
const static UnicodeSet::run_t __uset_308_runs[] = {{Empty, 251}, {Mixed, 1}, {Empty, 34564}};
const static UnicodeSet::bitquad_t __uset_308_quads[] = {0x2aaa0000};
const static UnicodeSet uset_308{__uset_308_runs, 3, __uset_308_quads, 1};
const UnicodeSet & E1_1f71_3_5_7_9_b_d_uset = uset_308;
const static UnicodeSet::run_t __uset_309_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 3}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_309_quads[] = {
0x2aaa0000, 0x08000000, 0x08000000, 0x0a000008};
const static UnicodeSet uset_309{__uset_309_runs, 5, __uset_309_quads, 4};
const UnicodeSet & E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b_uset = uset_309;
const static UnicodeSet::run_t __uset_3010_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 3}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3010_quads[] = {
0x2aaa0000, 0x48000000, 0x08080a00, 0x2a004808};
const static UnicodeSet uset_3010{__uset_3010_runs, 5, __uset_3010_quads, 4};
const UnicodeSet & E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_uset = uset_3010;
const static UnicodeSet::run_t __uset_3011_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 1}, {Empty, 1},
{Mixed, 1}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3011_quads[] = {0x00aa0000, 0x08000000, 0x2000c000};
const static UnicodeSet uset_3011{__uset_3011_runs, 7, __uset_3011_quads, 3};
const UnicodeSet & E1_1f71_3_5_7_bb_ee_f_fd_uset = uset_3011;
const static UnicodeSet::run_t __uset_3012_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 2}, {Mixed, 2}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3012_quads[] = {0x00aa0000, 0x08080a00, 0x2a004808};
const static UnicodeSet uset_3012{__uset_3012_runs, 5, __uset_3012_quads, 3};
const UnicodeSet & E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d_uset = uset_3012;
const static UnicodeSet::run_t __uset_3013_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 1}, {Empty, 1},
{Mixed, 1}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3013_quads[] = {0x0a0a0000, 0x48000000, 0x0a008800};
const static UnicodeSet uset_3013{__uset_3013_runs, 7, __uset_3013_quads, 3};
const UnicodeSet & E1_1f71_3_9_b_bb_e_eb_f_f9_b_uset = uset_3013;
const static UnicodeSet::run_t __uset_3014_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 3}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3014_quads[] = {
0x22220000, 0x48000000, 0x08080200, 0x2200c808};
const static UnicodeSet uset_3014{__uset_3014_runs, 5, __uset_3014_quads, 4};
const UnicodeSet & E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d_uset = uset_3014;
const static UnicodeSet::run_t __uset_3015_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 3}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3015_quads[] = {
0x28280000, 0x40000000, 0x00080800, 0x0000c008};
const static UnicodeSet uset_3015{__uset_3015_runs, 5, __uset_3015_quads, 4};
const UnicodeSet & E1_1f73_5_b_d_be_cb_d3_e3_e_f_uset = uset_3015;
const static UnicodeSet::run_t __uset_3016_runs[] = {
{Empty, 251}, {Mixed, 1}, {Empty, 1}, {Mixed, 1}, {Empty, 1},
{Mixed, 1}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3016_quads[] = {0x2a000000, 0x08000000, 0x0a004800};
const static UnicodeSet uset_3016{__uset_3016_runs, 7, __uset_3016_quads, 3};
const UnicodeSet & E1_1f79_b_d_bb_eb_e_f9_b_uset = uset_3016;
const static UnicodeSet::run_t __uset_3017_runs[] = {{Empty, 255}, {Mixed, 1}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3017_quads[] = {0x00008000};
const static UnicodeSet uset_3017{__uset_3017_runs, 3, __uset_3017_quads, 1};
const UnicodeSet & E1_1fef_uset = uset_3017;
const static UnicodeSet::run_t __uset_3018_runs[] = {{Empty, 255}, {Mixed, 1}, {Empty, 34560}};
const static UnicodeSet::bitquad_t __uset_3018_quads[] = {0x20000000};
const static UnicodeSet uset_3018{__uset_3018_runs, 3, __uset_3018_quads, 1};
const UnicodeSet & E1_1ffd_uset = uset_3018;
const static UnicodeSet::run_t __uset_3019_runs[] = {
{Empty, 256}, {Mixed, 1}, {Empty, 8}, {Mixed, 1}, {Empty, 15},
{Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3019_quads[] = {0x00000003, 0x00000840, 0x00000400};
const static UnicodeSet uset_3019{__uset_3019_runs, 7, __uset_3019_quads, 3};
const UnicodeSet & E2_2000_1_2126_b_232a_uset = uset_3019;
const static UnicodeSet::run_t __uset_3020_runs[] = {{Empty, 265}, {Mixed, 1}, {Empty, 34550}};
const static UnicodeSet::bitquad_t __uset_3020_quads[] = {0x00000040};
const static UnicodeSet uset_3020{__uset_3020_runs, 3, __uset_3020_quads, 1};
const UnicodeSet & E2_2126_uset = uset_3020;
const static UnicodeSet::run_t __uset_3021_runs[] = {
{Empty, 265}, {Mixed, 1}, {Empty, 15}, {Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3021_quads[] = {0x00000040, 0x00000600};
const static UnicodeSet uset_3021{__uset_3021_runs, 5, __uset_3021_quads, 2};
const UnicodeSet & E2_2126_2329_a_uset = uset_3021;
const static UnicodeSet::run_t __uset_3022_runs[] = {
{Empty, 265}, {Mixed, 1}, {Empty, 15}, {Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3022_quads[] = {0x00000440, 0x00000600};
const static UnicodeSet uset_3022{__uset_3022_runs, 5, __uset_3022_quads, 2};
const UnicodeSet & E2_2126_a_2329_a_uset = uset_3022;
const static UnicodeSet::run_t __uset_3023_runs[] = {{Empty, 265}, {Mixed, 1}, {Empty, 34550}};
const static UnicodeSet::bitquad_t __uset_3023_quads[] = {0x00000840};
const static UnicodeSet uset_3023{__uset_3023_runs, 3, __uset_3023_quads, 1};
const UnicodeSet & E2_2126_b_uset = uset_3023;
const static UnicodeSet::run_t __uset_3024_runs[] = {{Empty, 265}, {Mixed, 1}, {Empty, 34550}};
const static UnicodeSet::bitquad_t __uset_3024_quads[] = {0x00000400};
const static UnicodeSet uset_3024{__uset_3024_runs, 3, __uset_3024_quads, 1};
const UnicodeSet & E2_212a_uset = uset_3024;
const static UnicodeSet::run_t __uset_3025_runs[] = {
{Empty, 265}, {Mixed, 1}, {Empty, 15}, {Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3025_quads[] = {0x00000c00, 0x00000600};
const static UnicodeSet uset_3025{__uset_3025_runs, 5, __uset_3025_quads, 2};
const UnicodeSet & E2_212a_b_2329_a_uset = uset_3025;
const static UnicodeSet::run_t __uset_3026_runs[] = {{Empty, 265}, {Mixed, 1}, {Empty, 34550}};
const static UnicodeSet::bitquad_t __uset_3026_quads[] = {0x00000800};
const static UnicodeSet uset_3026{__uset_3026_runs, 3, __uset_3026_quads, 1};
const UnicodeSet & E2_212b_uset = uset_3026;
const static UnicodeSet::run_t __uset_3027_runs[] = {
{Empty, 265}, {Mixed, 1}, {Empty, 15}, {Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3027_quads[] = {0x00000800, 0x00000600};
const static UnicodeSet uset_3027{__uset_3027_runs, 5, __uset_3027_quads, 2};
const UnicodeSet & E2_212b_2329_a_uset = uset_3027;
const static UnicodeSet::run_t __uset_3028_runs[] = {{Empty, 281}, {Mixed, 1}, {Empty, 34534}};
const static UnicodeSet::bitquad_t __uset_3028_quads[] = {0x00000600};
const static UnicodeSet uset_3028{__uset_3028_runs, 3, __uset_3028_quads, 1};
const UnicodeSet & E2_2329_a_uset = uset_3028;
const static UnicodeSet::run_t __uset_3029_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3029_quads[] = {
0x9f0f1fed, 0xc7f0f1db, 0xe73e0f2f, 0xf392d8e8, 0x7f8067c7, 0x3dbf0f0f,
0xf1bb00dc, 0xf04f2f80, 0x10213f93, 0x1ffffc65, 0xf8000000, 0xffff37ff,
0x0000003f, 0xfffc0000, 0x03fbffff};
const static UnicodeSet uset_3029{__uset_3029_runs, 3, __uset_3029_quads, 15};
const UnicodeSet & EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9_uset = uset_3029;
const static UnicodeSet::run_t __uset_3030_runs[] = {
{Empty, 1992}, {Mixed, 12}, {Empty, 1}, {Mixed, 2}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3030_quads[] = {
0x870f058d, 0xc7f010d9, 0x270a0323, 0x7012c000, 0x0f8023c0, 0x019f0003,
0x1089005c, 0x504f0780, 0x10000780, 0x0000fc65, 0xf8000000, 0x000037ff,
0xfffc0000, 0x03e7ffff};
const static UnicodeSet uset_3030{__uset_3030_runs, 5, __uset_3030_quads, 14};
const UnicodeSet & EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9_uset = uset_3030;
const static UnicodeSet::run_t __uset_3031_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3031_quads[] = {
0x3cefae8d, 0xa1c062ed, 0x97fa1ba5, 0xac8d2cf0, 0x35baf9f4, 0x86782576,
0x9eff86fd, 0x67e945c6, 0x14e138a0, 0x2a4da064, 0x8e0e8c42, 0xea8409a8,
0x6468bcfa, 0x113235f9, 0x015da157};
const static UnicodeSet uset_3031{__uset_3031_runs, 3, __uset_3031_quads, 15};
const UnicodeSet & EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8_uset = uset_3031;
const static UnicodeSet::run_t __uset_3032_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3032_quads[] = {
0x16c4765f, 0x56cb9d7e, 0x54c9d008, 0x04a7a4eb, 0x8ae87a9f, 0x1a0d479d,
0x7db498f0, 0xcb1ab815, 0x16a53b0c, 0x5b3e4c44, 0x092c1ed8, 0x252b24ef,
0x13b3acdd, 0x6c79efc2, 0x02fe7817};
const static UnicodeSet uset_3032{__uset_3032_runs, 3, __uset_3032_quads, 15};
const UnicodeSet & EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9_uset = uset_3032;
const static UnicodeSet::run_t __uset_3033_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3033_quads[] = {
0x8923c9bf, 0x25e9270f, 0x4283eb19, 0x08c667e4, 0x43566b5d, 0x652ccb4c,
0xd1e8bf76, 0x223fb47b, 0x70842aea, 0x19c1a001, 0x878004ee, 0x0f090370,
0x279fc03c, 0x0023c7fe, 0x030047c6};
const static UnicodeSet uset_3033{__uset_3033_runs, 3, __uset_3033_quads, 15};
const UnicodeSet & EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9_uset = uset_3033;
const static UnicodeSet::run_t __uset_3034_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3034_quads[] = {
0xa937203f, 0x40f326fc, 0x7d087122, 0x34bfc3a1, 0x0387b818, 0xc6211133,
0x06c40f24, 0x7071118f, 0x000405e4, 0xe0010005, 0xf80000ff, 0x0009387f,
0x07ffffc0, 0xfffc0000, 0x00fb8007};
const static UnicodeSet uset_3034{__uset_3034_runs, 3, __uset_3034_quads, 15};
const UnicodeSet & EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7_uset = uset_3034;
const static UnicodeSet::run_t __uset_3035_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3035_quads[] = {
0xc95d403f, 0x087f99a1, 0x7a4f1ff7, 0xbc9cd191, 0xf6934f85, 0xaf175347,
0xf41d58ee, 0xae8eb59d, 0x5f6034b0, 0xc0653000, 0x14cff1a7, 0xcf3e126f,
0x34fd2fb2, 0x91fba60d, 0x03e46717};
const static UnicodeSet uset_3035{__uset_3035_runs, 3, __uset_3035_quads, 15};
const UnicodeSet & EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9_uset = uset_3035;
const static UnicodeSet::run_t __uset_3036_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3036_quads[] = {
0xf893c1f6, 0xc36eab25, 0xff47524b, 0xb0bba8ea, 0x39057f94, 0x952ea7be,
0x398c0bf2, 0xf59a1c43, 0x2f213528, 0xfe467c60, 0x988ffd37, 0xff360cb3,
0x1881cfc0, 0x01c5c80e, 0x0018181a};
const static UnicodeSet uset_3036{__uset_3036_runs, 3, __uset_3036_quads, 15};
const UnicodeSet & EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4_uset = uset_3036;
const static UnicodeSet::run_t __uset_3037_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3037_quads[] = {
0x7fc364fa, 0xf830e806, 0x61a178af, 0x75fac250, 0xde36bfe0, 0xc88aec36,
0x4798dd3a, 0xa2148363, 0x66a0243f, 0xaac42c60, 0x518c8090, 0x8df825b6,
0x4906aa92, 0x70de7a10, 0x031c343a};
const static UnicodeSet uset_3037{__uset_3037_runs, 3, __uset_3037_quads, 15};
const UnicodeSet & EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9_uset = uset_3037;
const static UnicodeSet::run_t __uset_3038_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3038_quads[] = {
0xaa1827b2, 0xeab9cac3, 0x9c2a3627, 0x5ff4123d, 0xafbdd512, 0x9669bad8,
0xfbd6ab0c, 0x25e2f841, 0x0cc008ec, 0xdf9fe865, 0x1b7c9869, 0x487d2edb,
0xb680d84c, 0x58d4d1ba, 0x03c0777c};
const static UnicodeSet uset_3038{__uset_3038_runs, 3, __uset_3038_quads, 15};
const UnicodeSet & EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9_uset = uset_3038;
const static UnicodeSet::run_t __uset_3039_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3039_quads[] = {
0x41795932, 0x6a190a69, 0x5a0132df, 0x2828e15a, 0xa330abac, 0xc437e518,
0x2ef7f8ad, 0xb0ecda6a, 0x0fc4338b, 0x48271c20, 0x687fe1c2, 0x33a32ab2,
0x08e78250, 0x9367f9c4, 0x02163590};
const static UnicodeSet uset_3039{__uset_3039_runs, 3, __uset_3039_quads, 15};
const UnicodeSet & EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9_uset = uset_3039;
const static UnicodeSet::run_t __uset_3040_runs[] = {
{Empty, 1992}, {Mixed, 12}, {Full, 1}, {Mixed, 2}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3040_quads[] = {
0x78f0fa72, 0x380fef26, 0xd8f5fcdc, 0x8fed3fff, 0xf07fdc3f, 0xfe60fffc,
0xef76ffa3, 0xafb0f87f, 0x6fe5387f, 0xffff0000, 0x07ffffff, 0xffff1800,
0x0003ffff, 0x00ff8000};
const static UnicodeSet uset_3040{__uset_3040_runs, 5, __uset_3040_quads, 14};
const UnicodeSet & EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7_uset = uset_3040;
const static UnicodeSet::run_t __uset_3041_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3041_quads[] = {
0x97076062, 0xc877b3eb, 0xe3818a71, 0x6604798b, 0x67c470f6, 0xcc399005,
0xa0de434c, 0x3ebce293, 0x654400c2, 0xa2ec4001, 0x748017a5, 0xd7783037,
0xf4621c33, 0x19ccaef4, 0x003214ec};
const static UnicodeSet uset_3041{__uset_3041_runs, 3, __uset_3041_quads, 15};
const UnicodeSet & EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5_uset = uset_3041;
const static UnicodeSet::run_t __uset_3042_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3042_quads[] = {
0xd3cb6c42, 0xc17dda68, 0x0b3e0260, 0xd3786a0f, 0xfe9b8c49, 0x4b8f7290,
0x7a1e33ef, 0xcc53ee33, 0x10e43245, 0xe1fec061, 0xf800070f, 0x0ff63b80,
0xf81fffc0, 0x003c07ff, 0x00fb87f8};
const static UnicodeSet uset_3042{__uset_3042_runs, 3, __uset_3042_quads, 15};
const UnicodeSet & EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7_uset = uset_3042;
const static UnicodeSet::run_t __uset_3043_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3043_quads[] = {
0x0cceadde, 0x6c8bf96c, 0xadbc2786, 0xeebf8972, 0x1d8d70a3, 0xa01b50d0,
0xe6bff54a, 0xd1042853, 0x68a02c5e, 0xa45bec44, 0x10ec081d, 0x2dea0352,
0x5cbfd593, 0x906ec63f, 0x019bb96e};
const static UnicodeSet uset_3043{__uset_3043_runs, 3, __uset_3043_quads, 15};
const UnicodeSet & EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8_uset = uset_3043;
const static UnicodeSet::run_t __uset_3044_runs[] = {{Empty, 1992}, {Mixed, 15}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3044_quads[] = {
0x8677e4e8, 0xdb663b49, 0x7443372b, 0xb44550f7, 0xf36af37f, 0x70a960bf,
0x91198f8f, 0x0e73d07f, 0x76250063, 0x5c4f0845, 0x48c5db2b, 0xf40d1b9e,
0x73914649, 0xe7560e21, 0x032df365};
const static UnicodeSet uset_3044{__uset_3044_runs, 3, __uset_3044_quads, 15};
const UnicodeSet & EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9_uset = uset_3044;
const static UnicodeSet::run_t __uset_3045_runs[] = {
{Empty, 2003}, {Mixed, 1}, {Empty, 2}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3045_quads[] = {0x00001000, 0x00e38000};
const static UnicodeSet uset_3045{__uset_3045_runs, 5, __uset_3045_quads, 2};
const UnicodeSet & EF_fa6c_cf___d1_5___7_uset = uset_3045;
const static UnicodeSet::run_t __uset_3046_runs[] = {
{Empty, 2003}, {Mixed, 1}, {Empty, 2}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3046_quads[] = {0x00001000, 0x00400000};
const static UnicodeSet uset_3046{__uset_3046_runs, 5, __uset_3046_quads, 2};
const UnicodeSet & EF_fa6c_d6_uset = uset_3046;
const static UnicodeSet::run_t __uset_3047_runs[] = {{Empty, 2006}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3047_quads[] = {0x00008000};
const static UnicodeSet uset_3047{__uset_3047_runs, 3, __uset_3047_quads, 1};
const UnicodeSet & EF_facf_uset = uset_3047;
const static UnicodeSet::run_t __uset_3048_runs[] = {{Empty, 2006}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3048_quads[] = {0x00e38000};
const static UnicodeSet uset_3048{__uset_3048_runs, 3, __uset_3048_quads, 1};
const UnicodeSet & EF_facf___d1_5___7_uset = uset_3048;
const static UnicodeSet::run_t __uset_3049_runs[] = {{Empty, 2006}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3049_quads[] = {0x00210000};
const static UnicodeSet uset_3049{__uset_3049_runs, 3, __uset_3049_quads, 1};
const UnicodeSet & EF_fad0_5_uset = uset_3049;
const static UnicodeSet::run_t __uset_3050_runs[] = {{Empty, 2006}, {Mixed, 1}, {Empty, 32809}};
const static UnicodeSet::bitquad_t __uset_3050_quads[] = {0x00200000};
const static UnicodeSet uset_3050{__uset_3050_runs, 3, __uset_3050_quads, 1};
const UnicodeSet & EF_fad5_uset = uset_3050;
const static UnicodeSet::run_t __uset_3051_runs[] = {{Empty, 6080}, {Mixed, 1}, {Empty, 28735}};
const static UnicodeSet::bitquad_t __uset_3051_quads[] = {0x00000001};
const static UnicodeSet uset_3051{__uset_3051_runs, 3, __uset_3051_quads, 1};
const UnicodeSet & F0_2f800_uset = uset_3051;
const static UnicodeSet::run_t __uset_3052_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3052_quads[] = {
0x2d1b257b, 0x58c07e72, 0xcfca8313, 0x165a7228, 0x8eb73dff, 0x11dc8bfd,
0x84421ade, 0xa59661f7, 0xaf08c79b, 0xa68ec2bc, 0xf1860217, 0x0d39ec6d,
0x7dfc41e4, 0xc31a4c4f, 0x9999bf3d, 0xe8c8d355, 0x11a2d2bf};
const static UnicodeSet uset_3052{__uset_3052_runs, 3, __uset_3052_quads, 17};
const UnicodeSet & F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c_uset = uset_3052;
const static UnicodeSet::run_t __uset_3053_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3053_quads[] = {
0x696c8463, 0x0ddfae41, 0xde36fe80, 0x3a06a0af, 0x4a81bbf9, 0xf01830c6,
0x35e188be, 0xc0f493b8, 0x3abb3459, 0x2d4c4809, 0x62d118e8, 0xb5f4fe0a,
0xd5a39b31, 0xc4d497ae, 0x525bd184, 0x368091e2, 0x2aab8e91};
const static UnicodeSet uset_3053{__uset_3053_runs, 3, __uset_3053_quads, 17};
const UnicodeSet & F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d_uset = uset_3053;
const static UnicodeSet::run_t __uset_3054_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3054_quads[] = {
0x4771b343, 0xcb50e91d, 0xd337ea8e, 0xffe84d28, 0x2e78bc51, 0xb900941e,
0x6453d070, 0x658560e2, 0xbc8664d3, 0xa816b1e6, 0xefe732ff, 0x740d0040,
0x5d41a767, 0x77e44759, 0x073b27c4, 0xcd6cbfdd, 0x260d168a};
const static UnicodeSet uset_3054{__uset_3054_runs, 3, __uset_3054_quads, 17};
const UnicodeSet & F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d_uset = uset_3054;
const static UnicodeSet::run_t __uset_3055_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3055_quads[] = {
0x4f472cd1, 0xe86ffe4a, 0xebef4892, 0x9dfc7d24, 0x90b1d4a9, 0xd7a21b4c,
0xf99af305, 0x0c67f00d, 0xafa76313, 0x92823c12, 0x099d0811, 0xf57bb8e4,
0x1b12c513, 0x32c084ac, 0x9b758910, 0xd3b765d6, 0x3fbd0152};
const static UnicodeSet uset_3055{__uset_3055_runs, 3, __uset_3055_quads, 17};
const UnicodeSet & F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d_uset = uset_3055;
const static UnicodeSet::run_t __uset_3056_runs[] = {
{Empty, 6080}, {Mixed, 1}, {Full, 3}, {Mixed, 1}, {Full, 1}, {Mixed, 1},
{Full, 4}, {Mixed, 6}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3056_quads[] = {
0xefffffff, 0xfff77fff, 0xffbfffff, 0x5bffffff, 0x008447d3, 0x8007e8f0,
0x23087d24, 0x13148000, 0x00426108};
const static UnicodeSet uset_3056{__uset_3056_runs, 9, __uset_3056_quads, 9};
const UnicodeSet & F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6_uset = uset_3056;
const static UnicodeSet::run_t __uset_3057_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3057_quads[] = {
0x824c3077, 0xf8000010, 0x022cffff, 0xfff80183, 0xf63178f7, 0x80000010,
0x9f30bb78, 0xfd045009, 0x0847ffff, 0x5dfff448, 0x003425e1, 0x5b41df60,
0x80006190, 0xfffdf7f0, 0xddfc6525, 0x2c3b5023, 0x3fe2f802};
const static UnicodeSet uset_3057{__uset_3057_runs, 3, __uset_3057_quads, 17};
const UnicodeSet & F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d_uset = uset_3057;
const static UnicodeSet::run_t __uset_3058_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3058_quads[] = {
0x42002077, 0x07dfffef, 0x002df800, 0x0057efdb, 0xf6bff8e0, 0x7ffc0010,
0xffb3f087, 0x02fab881, 0xa007bb00, 0x5c083f7f, 0xee3405e1, 0x03ffa89f,
0x7fffbfc3, 0x7ff9280f, 0xc103e121, 0x17a82fa0, 0x0047f8f7};
const static UnicodeSet uset_3058{__uset_3058_runs, 3, __uset_3058_quads, 17};
const UnicodeSet & F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6_uset = uset_3058;
const static UnicodeSet::run_t __uset_3059_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3059_quads[] = {
0xefbbdff7, 0xfeefffff, 0xfdffffff, 0xd7fdeffc, 0xfe797dff, 0xbeffffef,
0xdffffbff, 0xf47eeff7, 0x57fcdfbf, 0xc55fff37, 0x8fcbdfe1, 0xa753f7dd,
0xff7ff87e, 0xfffcd78f, 0xdcf7e7df, 0xd73ddfdc, 0x1fe2fdfd};
const static UnicodeSet uset_3059{__uset_3059_runs, 3, __uset_3059_quads, 17};
const UnicodeSet & F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c_uset = uset_3059;
const static UnicodeSet::run_t __uset_3060_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3060_quads[] = {
0x5250e4ef, 0xad2510d3, 0xa21bc932, 0x3eff0ab2, 0x46e4b870, 0xf892ea8b,
0xdd6243c1, 0xdb7b1807, 0x1582d784, 0xa0a1e0c8, 0xc3488c70, 0xbf93d8b0,
0x55956171, 0x024f694b, 0xaf5fefdd, 0x6ac173f0, 0x3c1e7a93};
const static UnicodeSet uset_3060{__uset_3060_runs, 3, __uset_3060_quads, 17};
const UnicodeSet & F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d_uset = uset_3060;
const static UnicodeSet::run_t __uset_3061_runs[] = {
{Empty, 6080}, {Mixed, 2}, {Empty, 2}, {Mixed, 13}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3061_quads[] = {
0x0244207f, 0x01100000, 0x010e8200, 0xfffffff9, 0x9ff0bb7b, 0xf47aaf77,
0xa804dbbf, 0x9aa000c8, 0x01021a00, 0xfcbe2091, 0xfffbf9be, 0xfffdf7ff,
0xfef7e3df, 0x1306a023, 0x205f6108};
const static UnicodeSet uset_3061{__uset_3061_runs, 5, __uset_3061_quads, 15};
const UnicodeSet & F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d_uset = uset_3061;
const static UnicodeSet::run_t __uset_3062_runs[] = {
{Empty, 6080}, {Mixed, 2}, {Full, 1}, {Mixed, 2}, {Empty, 1},
{Mixed, 9}, {Full, 1}, {Mixed, 1}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3062_quads[] = {
0x7ff7efff, 0xffdfffef, 0xffbffe7f, 0xf63978f7, 0x00cc0000, 0x01000000,
0xa8000000, 0xb8a000c8, 0x71363a1e, 0xa41228b3, 0xff7ff87e, 0xfffcd78f,
0xfef7e3df, 0x1fe2ffff};
const static UnicodeSet uset_3062{__uset_3062_runs, 9, __uset_3062_quads, 14};
const UnicodeSet & F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c_uset = uset_3062;
const static UnicodeSet::run_t __uset_3063_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3063_quads[] = {
0xb253cd14, 0xc7eff1ee, 0x71f13be5, 0x046a0e19, 0xef414597, 0x6cd387f0,
0x077dc426, 0x99f1daf5, 0x3385fa1e, 0x6105c104, 0x6154b6f8, 0x5586acb3,
0xc5404003, 0xac7edb6f, 0x8b5aad02, 0xf8b1d0f4, 0x01c3f9c4};
const static UnicodeSet uset_3063{__uset_3063_runs, 3, __uset_3063_quads, 17};
const UnicodeSet & F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8_uset = uset_3063;
const static UnicodeSet::run_t __uset_3064_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3064_quads[] = {
0x1c1ef124, 0xd70201e3, 0x30abf3d7, 0x3df16016, 0x93bd3fc0, 0x5cc711ee,
0x77c8c198, 0x1bb1fbfe, 0x6e3b538f, 0xce208fcc, 0x2112913a, 0xd604169b,
0x1899ae71, 0x2aeac27f, 0xe91eb23c, 0x3773adad, 0x11a56922};
const static UnicodeSet uset_3064{__uset_3064_runs, 3, __uset_3064_quads, 17};
const UnicodeSet & F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c_uset = uset_3064;
const static UnicodeSet::run_t __uset_3065_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3065_quads[] = {
0xfdffff88, 0xfeefffff, 0x022dffff, 0x00400183, 0x018e8200, 0xbefffff9,
0xbffcbf7b, 0x00001009, 0xffb80000, 0xa757cb7f, 0x01363fff, 0xfcbe2091,
0xff7fdfbf, 0xfffedf8f, 0x00006525, 0x2ce95fff, 0x201d0002};
const static UnicodeSet uset_3065{__uset_3065_runs, 3, __uset_3065_quads, 17};
const UnicodeSet & F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d_uset = uset_3065;
const static UnicodeSet::run_t __uset_3066_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3066_quads[] = {
0x3df7c078, 0x07cffe10, 0xf02c2ffc, 0xf857fe24, 0xf7368217, 0x7b83ffc6,
0xff9d4b03, 0x817a87c1, 0x27b83b00, 0xa4a7f4bf, 0xee37fa07, 0xac2d83f1,
0x817fd854, 0x05fa0350, 0x623deffa, 0xfb88178c, 0x1e4487fa};
const static UnicodeSet uset_3066{__uset_3066_runs, 3, __uset_3066_quads, 17};
const UnicodeSet & F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c_uset = uset_3066;
const static UnicodeSet::run_t __uset_3067_runs[] = {
{Empty, 6080}, {Mixed, 10}, {Empty, 1}, {Mixed, 6}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3067_quads[] = {
0x904c3008, 0x01100010, 0x02000000, 0x28421183, 0x08ce8708, 0x41000016,
0x60434484, 0x0b855088, 0x08432440, 0x42083400, 0xa4000000, 0xff7bbe2d,
0x7ff8170f, 0xfef786db, 0xeceb7fff, 0x3fbd9ef7};
const static UnicodeSet uset_3067{__uset_3067_runs, 5, __uset_3067_quads, 16};
const UnicodeSet & F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d_uset = uset_3067;
const static UnicodeSet::run_t __uset_3068_runs[] = {{Empty, 6080}, {Mixed, 17}, {Empty, 28719}};
const static UnicodeSet::bitquad_t __uset_3068_quads[] = {
0xbfa81e60, 0x16c001e0, 0x0f2e07e3, 0x2e45f1a7, 0x063e05e9, 0x8b9c07c6,
0x5f1167dc, 0xc50ec16a, 0xc4467820, 0x5a863e30, 0xe0f39407, 0x4b204bee,
0x96805839, 0x3a7cf4b1, 0x83c4113a, 0xcbe214f0, 0x01b371e6};
const static UnicodeSet uset_3068{__uset_3068_runs, 3, __uset_3068_quads, 17};
const UnicodeSet & F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8_uset = uset_3068;
//
class U8_InsertionBixNum : public PabloKernel {
public:
    U8_InsertionBixNum
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * InsertionBixNum);
protected:
    void generatePabloMethod() override;
};

U8_InsertionBixNum::U8_InsertionBixNum
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * InsertionBixNum)
: PabloKernel(ts, "U8_InsertionBixNum",
              {Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"InsertionBixNum", InsertionBixNum}}) {}

void U8_InsertionBixNum::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> insertions(2, All0);
    UTF::UTF_Compiler pb_compiler(getInputStreamVar("Basis"), pb, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> pb_vars(2);
    std::vector<UnicodeSet> pb_usets(2);
    Var * ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7 = pb.createVar("ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7", All0);
    pb_vars[0] = ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7;
    pb_usets[0] = ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7_uset;
    Var * ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a = pb.createVar("ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a", All0);
    pb_vars[1] = ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a;
    pb_usets[1] = ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a_uset;

    pb_compiler.compile(pb_vars, pb_usets);
    UTF::UTF_Compiler pb_adv_compiler(getInputStreamVar("Basis"), pb, pablo::BitMovementMode::Advance);
    std::vector<Var *> pb_adv_vars(2);
    std::vector<UnicodeSet> pb_adv_usets(2);
    Var * E0_f73_5_81 = pb.createVar("E0_f73_5_81", All0);
    pb_adv_vars[0] = E0_f73_5_81;
    pb_adv_usets[0] = E0_f73_5_81_uset;
    Var * CD_344_f73_5_81 = pb.createVar("CD_344_f73_5_81", All0);
    pb_adv_vars[1] = CD_344_f73_5_81;
    pb_adv_usets[1] = CD_344_f73_5_81_uset;

    pb_adv_compiler.compile(pb_adv_vars, pb_adv_usets);
    insertions[0] = ASC_43_a_63_a_a8_c2_7_a_f_d4_5_e2_7_a_f_f4_5_102_3_12_3_4c_d_5a_b_60_1_8___b_7f_a0_1_f_b0_228_9_391_5_7_9_f_a1_5_9_c_e_b1_5_7_9_f_c1_5_9___b_e_5d0___6_8___c_e_e0_1_3_4_6___a_f2_fa6c_cf___d1_5___7;
    insertions[1] = ASC_3c___e_41_2_4___9_b___50_2___a_61_2_4___9_b___70_2___a;
    insertions[0] = pb.createOr(insertions[0], E0_f73_5_81);
    insertions[1] = pb.createOr(insertions[1], CD_344_f73_5_81);
    writeOutputStreamSet("InsertionBixNum", insertions);
}
class SingletonCanonicalization : public PabloKernel {
public:
    SingletonCanonicalization
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * SelectMask, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

SingletonCanonicalization::SingletonCanonicalization
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * SelectMask, StreamSet * XfrmBasis)
: PabloKernel(ts, "SingletonCanonicalization" + Basis->shapeString(),
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"SelectMask", SelectMask}, Binding{"XfrmBasis", XfrmBasis}}) {}

void SingletonCanonicalization::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    // DeleteVar will be inverted to produce SelectMask
    Var * DeleteVar = pb.createVar("DeleteVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }

    auto b_cc_cf = pb.createScope();
    PabloAST * pfx_cc_cf_test = pb.createAnd(bnc.UGE(Basis, 0xcc), bnc.ULE(Basis, 0xcf));
    pb.createIf(pfx_cc_cf_test, b_cc_cf);
    std::vector<PabloAST *> xfrm_cc_cf(8, All0);
    PabloAST * del_cc_cf = All0;

    UTF::UTF_Compiler b_cc_cf_compiler(getInputStreamVar("Basis"), b_cc_cf, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_cc_cf_vars(7);
    std::vector<UnicodeSet> b_cc_cf_usets(7);
    Var * CD_37e = b_cc_cf.createVar("CD_37e", All0);
    b_cc_cf_vars[0] = CD_37e;
    b_cc_cf_usets[0] = CD_37e_uset;
    Var * CD_340_1_3_74 = b_cc_cf.createVar("CD_340_1_3_74", All0);
    b_cc_cf_vars[1] = CD_340_1_3_74;
    b_cc_cf_usets[1] = CD_340_1_3_74_uset;
    Var * CD_374 = b_cc_cf.createVar("CD_374", All0);
    b_cc_cf_vars[2] = CD_374;
    b_cc_cf_usets[2] = CD_374_uset;
    Var * CD_374_87 = b_cc_cf.createVar("CD_374_87", All0);
    b_cc_cf_vars[3] = CD_374_87;
    b_cc_cf_usets[3] = CD_374_87_uset;
    Var * CE_387 = b_cc_cf.createVar("CE_387", All0);
    b_cc_cf_vars[4] = CE_387;
    b_cc_cf_usets[4] = CE_387_uset;
    Var * CD_343_87 = b_cc_cf.createVar("CD_343_87", All0);
    b_cc_cf_vars[5] = CD_343_87;
    b_cc_cf_usets[5] = CD_343_87_uset;
    Var * CD_374_e = b_cc_cf.createVar("CD_374_e", All0);
    b_cc_cf_vars[6] = CD_374_e;
    b_cc_cf_usets[6] = CD_374_e_uset;

    b_cc_cf_compiler.compile(b_cc_cf_vars, b_cc_cf_usets);
    xfrm_cc_cf[0] = b_cc_cf.createOr(xfrm_cc_cf[0], CD_340_1_3_74);
    xfrm_cc_cf[1] = b_cc_cf.createOr(xfrm_cc_cf[1], CD_374);
    xfrm_cc_cf[2] = b_cc_cf.createOr(xfrm_cc_cf[2], CD_374_87);
    xfrm_cc_cf[3] = b_cc_cf.createOr(xfrm_cc_cf[3], CE_387);
    PabloAST * CD_343_87_adv1 = b_cc_cf.createAdvance(CD_343_87, 1);
    xfrm_cc_cf[4] = b_cc_cf.createOr(xfrm_cc_cf[4], CD_343_87_adv1);
    PabloAST * CD_374_e_adv1 = b_cc_cf.createAdvance(CD_374_e, 1);
    xfrm_cc_cf[0] = b_cc_cf.createOr(xfrm_cc_cf[0], CD_374_e_adv1);
    xfrm_cc_cf[2] = b_cc_cf.createOr(xfrm_cc_cf[2], CD_374_e_adv1);
    PabloAST * CD_374_adv1 = b_cc_cf.createAdvance(CD_374, 1);
    xfrm_cc_cf[3] = b_cc_cf.createOr(xfrm_cc_cf[3], CD_374_adv1);
    PabloAST * CD_37e_adv1 = b_cc_cf.createAdvance(CD_37e, 1);
    xfrm_cc_cf[7] = b_cc_cf.createOr(xfrm_cc_cf[7], CD_37e_adv1);
    PabloAST * CE_387_adv1 = b_cc_cf.createAdvance(CE_387, 1);
    xfrm_cc_cf[5] = b_cc_cf.createOr(xfrm_cc_cf[5], CE_387_adv1);
    del_cc_cf = b_cc_cf.createOr(del_cc_cf, CD_37e);

    for (unsigned i = 0; i < 8; i++) {
        b_cc_cf.createAssign(XfrmVar[i], b_cc_cf.createOr(XfrmVar[i], xfrm_cc_cf[i]));
    }

    b_cc_cf.createAssign(DeleteVar, b_cc_cf.createOr(DeleteVar, del_cc_cf));

    auto b_e1 = pb.createScope();
    PabloAST * pfx_e1_test = bnc.EQ(Basis, 0xe1);
    pb.createIf(pfx_e1_test, b_e1);
    std::vector<PabloAST *> xfrm_e1(8, All0);
    PabloAST * del_e1 = All0;

    UTF::UTF_Compiler b_e1_compiler(getInputStreamVar("Basis"), b_e1, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_e1_vars(12);
    std::vector<UnicodeSet> b_e1_usets(12);
    Var * E1_1f71 = b_e1.createVar("E1_1f71", All0);
    b_e1_vars[0] = E1_1f71;
    b_e1_usets[0] = E1_1f71_uset;
    Var * E1_1fef = b_e1.createVar("E1_1fef", All0);
    b_e1_vars[1] = E1_1fef;
    b_e1_usets[1] = E1_1fef_uset;
    Var * E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d = b_e1.createVar("E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d", All0);
    b_e1_vars[2] = E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d;
    b_e1_usets[2] = E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d_uset;
    Var * E1_1f71_3_5_7_9_b_d = b_e1.createVar("E1_1f71_3_5_7_9_b_d", All0);
    b_e1_vars[3] = E1_1f71_3_5_7_9_b_d;
    b_e1_usets[3] = E1_1f71_3_5_7_9_b_d_uset;
    Var * E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d = b_e1.createVar("E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d", All0);
    b_e1_vars[4] = E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d;
    b_e1_usets[4] = E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_uset;
    Var * E1_1ffd = b_e1.createVar("E1_1ffd", All0);
    b_e1_vars[5] = E1_1ffd;
    b_e1_usets[5] = E1_1ffd_uset;
    Var * E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d = b_e1.createVar("E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d", All0);
    b_e1_vars[6] = E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d;
    b_e1_usets[6] = E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d_uset;
    Var * E1_1f71_3_9_b_bb_e_eb_f_f9_b = b_e1.createVar("E1_1f71_3_9_b_bb_e_eb_f_f9_b", All0);
    b_e1_vars[7] = E1_1f71_3_9_b_bb_e_eb_f_f9_b;
    b_e1_usets[7] = E1_1f71_3_9_b_bb_e_eb_f_f9_b_uset;
    Var * E1_1f71_3_5_7_bb_ee_f_fd = b_e1.createVar("E1_1f71_3_5_7_bb_ee_f_fd", All0);
    b_e1_vars[8] = E1_1f71_3_5_7_bb_ee_f_fd;
    b_e1_usets[8] = E1_1f71_3_5_7_bb_ee_f_fd_uset;
    Var * E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b = b_e1.createVar("E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b", All0);
    b_e1_vars[9] = E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b;
    b_e1_usets[9] = E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b_uset;
    Var * E1_1f73_5_b_d_be_cb_d3_e3_e_f = b_e1.createVar("E1_1f73_5_b_d_be_cb_d3_e3_e_f", All0);
    b_e1_vars[10] = E1_1f73_5_b_d_be_cb_d3_e3_e_f;
    b_e1_usets[10] = E1_1f73_5_b_d_be_cb_d3_e3_e_f_uset;
    Var * E1_1f79_b_d_bb_eb_e_f9_b = b_e1.createVar("E1_1f79_b_d_bb_eb_e_f9_b", All0);
    b_e1_vars[11] = E1_1f79_b_d_bb_eb_e_f9_b;
    b_e1_usets[11] = E1_1f79_b_d_bb_eb_e_f9_b_uset;

    b_e1_compiler.compile(b_e1_vars, b_e1_usets);
    PabloAST * E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d_adv1 = b_e1.createAdvance(E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d, 1);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], E1_1f71_3_5_7_c9_b_d3_b_e3_b_e_f9_b_d_adv1);
    PabloAST * E1_1f71_3_5_7_9_b_d_adv1 = b_e1.createAdvance(E1_1f71_3_5_7_9_b_d, 1);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], E1_1f71_3_5_7_9_b_d_adv1);
    PabloAST * E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_adv1 = b_e1.createAdvance(E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d, 1);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_adv1);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_adv1);
    xfrm_e1[6] = b_e1.createOr(xfrm_e1[6], E1_1f71_3_5_7_9_b_d_bb_e_c9_b_d3_b_e3_b_e_f9_b_d_adv1);
    PabloAST * E1_1ffd_adv1 = b_e1.createAdvance(E1_1ffd, 1);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], E1_1ffd_adv1);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], E1_1ffd_adv1);
    PabloAST * E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d_adv2 = b_e1.createAdvance(E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], E1_1f71_5_9_d_bb_e_c9_d3_b_e3_b_e_f_f9_d_adv2);
    PabloAST * E1_1f71_3_9_b_bb_e_eb_f_f9_b_adv2 = b_e1.createAdvance(E1_1f71_3_9_b_bb_e_eb_f_f9_b, 2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], E1_1f71_3_9_b_bb_e_eb_f_f9_b_adv2);
    PabloAST * E1_1f71_3_5_7_bb_ee_f_fd_adv2 = b_e1.createAdvance(E1_1f71_3_5_7_bb_ee_f_fd, 2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], E1_1f71_3_5_7_bb_ee_f_fd_adv2);
    PabloAST * E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b_adv2 = b_e1.createAdvance(E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b, 2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], E1_1f71_3_5_7_9_b_d_bb_db_e3_f9_b_adv2);
    PabloAST * E1_1f73_5_b_d_be_cb_d3_e3_e_f_adv2 = b_e1.createAdvance(E1_1f73_5_b_d_be_cb_d3_e3_e_f, 2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], E1_1f73_5_b_d_be_cb_d3_e3_e_f_adv2);
    PabloAST * E1_1f79_b_d_bb_eb_e_f9_b_adv2 = b_e1.createAdvance(E1_1f79_b_d_bb_eb_e_f9_b, 2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], E1_1f79_b_d_bb_eb_e_f9_b_adv2);
    PabloAST * E1_1fef_adv2 = b_e1.createAdvance(E1_1fef, 2);
    xfrm_e1[6] = b_e1.createOr(xfrm_e1[6], E1_1fef_adv2);
    xfrm_e1[7] = b_e1.createOr(xfrm_e1[7], E1_1fef_adv2);
    del_e1 = b_e1.createOr(del_e1, E1_1f71);
    del_e1 = b_e1.createOr(del_e1, E1_1fef);
    del_e1 = b_e1.createOr(del_e1, b_e1.createAdvance(E1_1fef, 1));

    for (unsigned i = 0; i < 8; i++) {
        b_e1.createAssign(XfrmVar[i], b_e1.createOr(XfrmVar[i], xfrm_e1[i]));
    }

    b_e1.createAssign(DeleteVar, b_e1.createOr(DeleteVar, del_e1));

    auto b_e2 = pb.createScope();
    PabloAST * pfx_e2_test = bnc.EQ(Basis, 0xe2);
    pb.createIf(pfx_e2_test, b_e2);
    std::vector<PabloAST *> xfrm_e2(8, All0);
    PabloAST * del_e2 = All0;

    UTF::UTF_Compiler b_e2_compiler(getInputStreamVar("Basis"), b_e2, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_e2_vars(10);
    std::vector<UnicodeSet> b_e2_usets(10);
    Var * E2_2126 = b_e2.createVar("E2_2126", All0);
    b_e2_vars[0] = E2_2126;
    b_e2_usets[0] = E2_2126_uset;
    Var * E2_212a = b_e2.createVar("E2_212a", All0);
    b_e2_vars[1] = E2_212a;
    b_e2_usets[1] = E2_212a_uset;
    Var * E2_2329_a = b_e2.createVar("E2_2329_a", All0);
    b_e2_vars[2] = E2_2329_a;
    b_e2_usets[2] = E2_2329_a_uset;
    Var * E2_2126_b = b_e2.createVar("E2_2126_b", All0);
    b_e2_vars[3] = E2_2126_b;
    b_e2_usets[3] = E2_2126_b_uset;
    Var * E2_2126_2329_a = b_e2.createVar("E2_2126_2329_a", All0);
    b_e2_vars[4] = E2_2126_2329_a;
    b_e2_usets[4] = E2_2126_2329_a_uset;
    Var * E2_212b = b_e2.createVar("E2_212b", All0);
    b_e2_vars[5] = E2_212b;
    b_e2_usets[5] = E2_212b_uset;
    Var * E2_212b_2329_a = b_e2.createVar("E2_212b_2329_a", All0);
    b_e2_vars[6] = E2_212b_2329_a;
    b_e2_usets[6] = E2_212b_2329_a_uset;
    Var * E2_2000_1_2126_b_232a = b_e2.createVar("E2_2000_1_2126_b_232a", All0);
    b_e2_vars[7] = E2_2000_1_2126_b_232a;
    b_e2_usets[7] = E2_2000_1_2126_b_232a_uset;
    Var * E2_2126_a_2329_a = b_e2.createVar("E2_2126_a_2329_a", All0);
    b_e2_vars[8] = E2_2126_a_2329_a;
    b_e2_usets[8] = E2_2126_a_2329_a_uset;
    Var * E2_212a_b_2329_a = b_e2.createVar("E2_212a_b_2329_a", All0);
    b_e2_vars[9] = E2_212a_b_2329_a;
    b_e2_usets[9] = E2_212a_b_2329_a_uset;

    b_e2_compiler.compile(b_e2_vars, b_e2_usets);
    xfrm_e2[0] = b_e2.createOr(xfrm_e2[0], E2_2329_a);
    PabloAST * E2_2126_b_adv1 = b_e2.createAdvance(E2_2126_b, 1);
    xfrm_e2[1] = b_e2.createOr(xfrm_e2[1], E2_2126_b_adv1);
    PabloAST * E2_2126_2329_a_adv1 = b_e2.createAdvance(E2_2126_2329_a, 1);
    xfrm_e2[3] = b_e2.createOr(xfrm_e2[3], E2_2126_2329_a_adv1);
    xfrm_e2[6] = b_e2.createOr(xfrm_e2[6], E2_2126_b_adv1);
    PabloAST * E2_212b_adv1 = b_e2.createAdvance(E2_212b, 1);
    xfrm_e2[0] = b_e2.createOr(xfrm_e2[0], E2_212b_adv1);
    PabloAST * E2_212b_2329_a_adv1 = b_e2.createAdvance(E2_212b_2329_a, 1);
    xfrm_e2[2] = b_e2.createOr(xfrm_e2[2], E2_212b_2329_a_adv1);
    PabloAST * E2_2000_1_2126_b_232a_adv2 = b_e2.createAdvance(E2_2000_1_2126_b_232a, 2);
    xfrm_e2[1] = b_e2.createOr(xfrm_e2[1], E2_2000_1_2126_b_232a_adv2);
    PabloAST * E2_2126_a_2329_a_adv2 = b_e2.createAdvance(E2_2126_a_2329_a, 2);
    xfrm_e2[0] = b_e2.createOr(xfrm_e2[0], E2_2126_a_2329_a_adv2);
    PabloAST * E2_2126_b_adv2 = b_e2.createAdvance(E2_2126_b, 2);
    xfrm_e2[2] = b_e2.createOr(xfrm_e2[2], E2_2126_b_adv2);
    xfrm_e2[3] = b_e2.createOr(xfrm_e2[3], E2_2126_b_adv2);
    PabloAST * E2_212a_b_2329_a_adv2 = b_e2.createAdvance(E2_212a_b_2329_a, 2);
    xfrm_e2[5] = b_e2.createOr(xfrm_e2[5], E2_212a_b_2329_a_adv2);
    PabloAST * E2_212a_adv2 = b_e2.createAdvance(E2_212a, 2);
    xfrm_e2[6] = b_e2.createOr(xfrm_e2[6], E2_212a_adv2);
    xfrm_e2[7] = b_e2.createOr(xfrm_e2[7], E2_212a_adv2);
    del_e2 = b_e2.createOr(del_e2, E2_2126);
    del_e2 = b_e2.createOr(del_e2, E2_212a);
    del_e2 = b_e2.createOr(del_e2, b_e2.createAdvance(E2_212a, 1));

    for (unsigned i = 0; i < 8; i++) {
        b_e2.createAssign(XfrmVar[i], b_e2.createOr(XfrmVar[i], xfrm_e2[i]));
    }

    b_e2.createAssign(DeleteVar, b_e2.createOr(DeleteVar, del_e2));

    auto b_ef = pb.createScope();
    PabloAST * pfx_ef_test = bnc.EQ(Basis, 0xef);
    pb.createIf(pfx_ef_test, b_ef);
    std::vector<PabloAST *> xfrm_ef(8, All0);
    PabloAST * del_ef = All0;

    UTF::UTF_Compiler b_ef_compiler(getInputStreamVar("Basis"), b_ef, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_ef_vars(22);
    std::vector<UnicodeSet> b_ef_usets(22);
    Var * EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7 = b_ef.createVar("EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7", All0);
    b_ef_vars[0] = EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7;
    b_ef_usets[0] = EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7_uset;
    Var * EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9 = b_ef.createVar("EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9", All0);
    b_ef_vars[1] = EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9;
    b_ef_usets[1] = EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9_uset;
    Var * EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9 = b_ef.createVar("EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9", All0);
    b_ef_vars[2] = EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9;
    b_ef_usets[2] = EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9_uset;
    Var * EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7 = b_ef.createVar("EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7", All0);
    b_ef_vars[3] = EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7;
    b_ef_usets[3] = EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7_uset;
    Var * EF_fa6c_cf___d1_5___7 = b_ef.createVar("EF_fa6c_cf___d1_5___7", All0);
    b_ef_vars[4] = EF_fa6c_cf___d1_5___7;
    b_ef_usets[4] = EF_fa6c_cf___d1_5___7_uset;
    Var * EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9 = b_ef.createVar("EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9", All0);
    b_ef_vars[5] = EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9;
    b_ef_usets[5] = EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9_uset;
    Var * EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9 = b_ef.createVar("EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9", All0);
    b_ef_vars[6] = EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9;
    b_ef_usets[6] = EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9_uset;
    Var * EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9 = b_ef.createVar("EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9", All0);
    b_ef_vars[7] = EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9;
    b_ef_usets[7] = EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9_uset;
    Var * EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8 = b_ef.createVar("EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8", All0);
    b_ef_vars[8] = EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8;
    b_ef_usets[8] = EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8_uset;
    Var * EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4 = b_ef.createVar("EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4", All0);
    b_ef_vars[9] = EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4;
    b_ef_usets[9] = EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4_uset;
    Var * EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7 = b_ef.createVar("EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7", All0);
    b_ef_vars[10] = EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7;
    b_ef_usets[10] = EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7_uset;
    Var * EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8 = b_ef.createVar("EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8", All0);
    b_ef_vars[11] = EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8;
    b_ef_usets[11] = EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8_uset;
    Var * EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9 = b_ef.createVar("EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9", All0);
    b_ef_vars[12] = EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9;
    b_ef_usets[12] = EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9_uset;
    Var * EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9 = b_ef.createVar("EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9", All0);
    b_ef_vars[13] = EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9;
    b_ef_usets[13] = EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9_uset;
    Var * EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9 = b_ef.createVar("EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9", All0);
    b_ef_vars[14] = EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9;
    b_ef_usets[14] = EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9_uset;
    Var * EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5 = b_ef.createVar("EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5", All0);
    b_ef_vars[15] = EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5;
    b_ef_usets[15] = EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5_uset;
    Var * EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9 = b_ef.createVar("EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9", All0);
    b_ef_vars[16] = EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9;
    b_ef_usets[16] = EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9_uset;
    Var * EF_fa6c_d6 = b_ef.createVar("EF_fa6c_d6", All0);
    b_ef_vars[17] = EF_fa6c_d6;
    b_ef_usets[17] = EF_fa6c_d6_uset;
    Var * EF_facf = b_ef.createVar("EF_facf", All0);
    b_ef_vars[18] = EF_facf;
    b_ef_usets[18] = EF_facf_uset;
    Var * EF_facf___d1_5___7 = b_ef.createVar("EF_facf___d1_5___7", All0);
    b_ef_vars[19] = EF_facf___d1_5___7;
    b_ef_usets[19] = EF_facf___d1_5___7_uset;
    Var * EF_fad0_5 = b_ef.createVar("EF_fad0_5", All0);
    b_ef_vars[20] = EF_fad0_5;
    b_ef_usets[20] = EF_fad0_5_uset;
    Var * EF_fad5 = b_ef.createVar("EF_fad5", All0);
    b_ef_vars[21] = EF_fad5;
    b_ef_usets[21] = EF_fad5_uset;

    b_ef_compiler.compile(b_ef_vars, b_ef_usets);
    xfrm_ef[0] = b_ef.createOr(xfrm_ef[0], EF_f900___5_d_10___2_4_5_8_b_d_f_22___7_9_a_d_30_1_4___7_e_41_5_8_c___e_53_8_a___e_60_5_7___9_e___75_7_a_c_d_83_4_b___d_f___92_7___9_a0_1_4_5_8_c_b0_5_9_a_e_f_c2_5_8___b_d2_6_7_9_a_e0___3_7_8_c_f0_4___6_c___e_fa02_5___8_a_12_20_2_30_d___47_5b___66_b___d_70_3_86___9a_b2___c2_f___d1_3___7);
    xfrm_ef[1] = b_ef.createOr(xfrm_ef[1], EF_f900_2_3_5___c_10___3_8___c_f___21_3_4_6___8_c___f_34___a_e___43_5_8___b_51___5_8___a_d___f_63_5___7_b_c_e_f_71_4_7___9_c___82_6___a_d_e_97___e_a0___3_8___b_b0___5_7_8_a___d_c2___4_6_7_d0_1_3___5_7_8_c___f_e7___b_d_f0___3_6_c___fa01_4_7___d_10_5_c_20_2_5_6_a___3c_5b___6a_c_d_70___85_b2___d1_3___9);
    xfrm_ef[2] = b_ef.createOr(xfrm_ef[2], EF_f900_2_3_7_8_a_10___3_8___a_f_20_3_4_6_7_c_34___a_e___41_5_8_9_51_3_8___a_d_6e_f_71_4_c___e_86___9_d_97___b_a0_1_b0___4_7_8_c2___4_6_d0_3_7_c_e7___a_f0___3_6_c_e_fa07___a_1c_20_2_5_6_a___f_5b___6a_c_d_b2___d2_5___9);
    xfrm_ef[3] = b_ef.createOr(xfrm_ef[3], EF_f901_4___6_9_b___f_14___7_b___e_21_2_5_8___b_d___33_b___d_42___4_6_7_a___50_2_4___7_b_c_e___6d_70_2_3_5___b_f___85_a___c_e___96_c___f_a2___f_b5_6_9___c1_5_7___f_d1_2_4___6_8___b_d___e6_b___f_f4_5_7___b_d_f___fa06_b___d_10_2_5___b_d_e_30___5a_6b_c_70___b1_cf___d7);
    xfrm_ef[4] = b_ef.createOr(xfrm_ef[4], EF_fa6c_cf___d1_5___7);
    PabloAST * EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9_adv1 = b_ef.createAdvance(EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9, 1);
    xfrm_ef[0] = b_ef.createOr(xfrm_ef[0], EF_f900___4_6_9_a_c___e_12_6_7_9_a_c_21___6_8_a___c_f___31_3_6_7_9_a_c_e_43_c_e___50_3_6_7_a_c_e_60_1_3_5___7_a_d_f___72_5_7_a_80___4_7_9_b___e_93_5___7_9_b_f_a0_2___4_7___a_e_b0_2_3_9_b_c_c4___7_b_c_f_d2_4_5_7_8_a___e_e0_2_4_b___d_f_f1_3_4_8_9_b_e_f_fa02_3_8_9_b___d_10_2_5_7_9_a_c_22_6_a_b_e_31___5_8_9_b_c_e_43_4_6_7_9___c_52_3_5_8_b_60___3_5___7_a_d_70_1_3_5_8_a_d_80_2___4_6_7_a_b_d_f___91_4_5_7___9_c_a1_6___b_d___b0_3___6_a_b_d_e_c0___2_4_b___e_d1___7_9_adv1);
    PabloAST * EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9_adv1 = b_ef.createAdvance(EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9, 1);
    xfrm_ef[2] = b_ef.createOr(xfrm_ef[2], EF_f900___5_e_10_2___4_6_8_b_e___20_5_7_8_b_c_f___36_b_40___2_4___c_50___3_6_9_b___e_60_4_7_8_c_e_f_72___4_7_a___d_f_80_2_7___b_e_90_1_4_7_9_a_c___a2_6_8_9_c_e_b0___2_4_8___b_d_f_c1___3_5___7_b_c_e_d0_2___4_a_c___e0_2___4_7_8_a_c_d_f_f1___3_7_9___b_d_f_fa04_5_7_a_c_d_15_6_8___c_e_2c_d_30_2_5_6_e___42_5_7_8_c___53_6_7_a_c_60___3_5_6_9_c_71___5_8___b_e_f_81_4_5_7___b_d_90_2___7_a_c_d_a0_2_3_9_a_d_f___b1_3___8_c_f___c2_4_8___a_d_e_d2_5___9_adv1);
    PabloAST * EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9_adv1 = b_ef.createAdvance(EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9, 1);
    xfrm_ef[4] = b_ef.createOr(xfrm_ef[4], EF_f900___5_7_8_b_e___11_5_8_b_f___23_8___a_d_30_3_5___8_a_d_40_3_4_8_9_b_d___51_7_9_e_62_5___a_d_e_71_2_6_7_b_80_2___4_6_8_9_b_d_e_91_2_4_6_8_9_e_a2_3_6_8_9_b_e_f_b2_3_5_8_a_d_e_c1_2_4___6_8___d_f_d3_5___8_c_e___e1_3___6_a_c_d_f___f5_9_d_fa01_3_5___7_9_b_d_12_7_c___e_20_d_f_30_6___8_b_c_41___3_5___7_a_57___a_f_64___6_8_9_70_3_8___b_82___5_e___94_7___a_d_a1___a_e___b1_5_c1_2_6___a_e_d8_9_adv1);
    PabloAST * EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8_adv1 = b_ef.createAdvance(EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8, 1);
    xfrm_ef[1] = b_ef.createOr(xfrm_ef[1], EF_f901___4_6___8_a_b_d_f_11___3_6_7_a_b_22_3_5_6_8_b___31_3_7_a_b_d_e_41_2_7___a_d_52___5_7_8_a_b_d_f_61_4___6_8_b_f___75_7_9___b_d___81_5_7_c___e_90_2_3_7_8_a___c_a4_6_7_c_e_b0_1_3_4_d_f_c1_3_6_8_a_c___d5_7_9_a_d___e1_4_6_b_d_f2_8_c_e_f_fa01___4_6_a_b_d_15_7_b_d_e_22_6_a_b_d___31_3_4_6_a_d_f_40_2___4_b_52_3_5___7_c_61_4_6_8_9_71_3_5___8_a_b_d_80_1_4_7_8_a_c_e___95_7_a___c_e_a0___5_9_a_e_f_b1___3_5_6_c_f_c1___3_5_6_8_b___d_f___d1_3_4_7_8_adv1);
    PabloAST * EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4_adv1 = b_ef.createAdvance(EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4, 1);
    xfrm_ef[3] = b_ef.createOr(xfrm_ef[3], EF_f901_2_4___8_e___11_4_7_b___20_2_5_8_9_b_d_f_31___3_5_6_8_9_e___41_3_6_9_c_e_50___2_6_8___f_61_3_5___7_b_d_f___71_3___5_7_c_d_f_82_4_7___e_90_2_8_b___d_a1___5_7___a_d_f_b1___3_5_8_a_c_f_c1_4___9_b_d2_3_7_8_b___d_e0_1_6_a___c_f1_3_4_7_8_a_c___f_fa03_5_8_a_c_d_10_5_8___b_d_25_6_a___e_31_2_6_9___42_4_5_8_a___53_7_b_c_f___61_4_5_7_a_b_71_2_4_5_8___f_86___b_e___90_7_b_c_a1___3_b_e___b0_2_6___8_c1_3_4_b_c_d3_4_adv1);
    PabloAST * EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7_adv1 = b_ef.createAdvance(EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7, 1);
    xfrm_ef[5] = b_ef.createOr(xfrm_ef[5], EF_f901_6_a_b_d_e_10_1_3_6___9_c_e_f_23_5_6_9_b_c_e___30_2___6_8_e_f_45_6_9_51___5_8_9_b_60___3_9_b_d_e_73___6_8_9_c_e___80_3_6_a_b_f___91_3_4_7_9___f_a4_7_9_c___e_b0___3_7___9_b_e_c0___3_5___9_c_d_d1___4_9_b___e_e0_1_4_5_9___b_d___f1_4_6_a_b_e___fa00_2_6_9_c_d_12_5___7_c_20_5_6_e_f_31___8_d___43_8___a_5b___f_67___9_b___d_71_2_4___b_86___94_b___aa_b2___5_c3___a_f___d1_3___7_adv1);
    PabloAST * EF_fa6c_cf___d1_5___7_adv1 = b_ef.createAdvance(EF_fa6c_cf___d1_5___7, 1);
    xfrm_ef[7] = b_ef.createOr(xfrm_ef[7], EF_fa6c_cf___d1_5___7_adv1);
    PabloAST * EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8_adv2 = b_ef.createAdvance(EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8, 2);
    xfrm_ef[3] = b_ef.createOr(xfrm_ef[3], EF_f900_2_3_7_9___b_d_f___13_5___7_a___d_20_2_3_5___7_9_d_e_36___8_d_f_40_2_5_7___9_b_c_51_3___a_c_f_64___7_a_b_d_70_2_3_7_a_b_d_f_82_4___8_b___f_91_3___5_7_8_a_c_d_a1_2_4___6_8_a_d_b3___6_9_a_f_c0_2___7_9_a_f___d7_9___c_f_e1_2_6___8_a_e_f0_3_5___a_d_e_fa05_7_b___d_10_5___7_a_c_22_5_6_d_f_30_2_3_6_9_b_d_41_6_a_b_f_51___3_9___b_f_63_5_7_8_b_72_7_9_b_d___f_81_3___7_a___d_f_93_5_6_a_d_e_a0_3___8_a_c_d_b1_4_5_8_c_c0___2_4_6_8_d_f_d0_2___4_6_8_adv2);
    PabloAST * EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9_adv2 = b_ef.createAdvance(EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9, 2);
    xfrm_ef[0] = b_ef.createOr(xfrm_ef[0], EF_f901_3___7_a_d_e_10_1_6___e_21_2_b_d___f_34_5_b___43_5_7_b___e_50_5_7_8_d_e_64_6_9_e_f_71_3___8_a_c___e_85___d_f_91_2_4_5_9___c_e_f_a1_2_4_5_a_b_d___f_b1_3_7_b_e_f_c1_3___5_8_a___c_e_f_d3_4_7___a_e_e0_1_5_6_8_9_f_f2_4_9_d_f___fa05_a_d_15_7_9_a_d_e_25_6_a_b_d_32_6_7_9_b_d_f_44_7_f_52_3_7_8_c_e_61_2_4_5_7_8_a_d_73___8_a_b_f_81_4_7_9_b_d_f_91_2_8_b_e_a4_9_b___e_b1___4_6_7_c___e_c1_3___5_a_c_d_d2___4_8_9_adv2);
    PabloAST * EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9_adv2 = b_ef.createAdvance(EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9, 2);
    xfrm_ef[2] = b_ef.createOr(xfrm_ef[2], EF_f901_4_5_8_b_c_e_10_3___6_8_e_20_3_5_6_9_b_30_3_4_9_b_d_e_40___4_6_7_9_c_d_50_9_b_c_e_61_3_4_6_8_d___f_73_5_b_d_82_3_5_7___9_b_d_f_94_5_8_9_d_f_a3_4_8_a_d___b2_4_5_a_e___c0_2_3_5_7_b___d2_4___7_9___b_d_e1_3_5_6_9_b_c_e_f_f2_3_5___7_c_d_f___fa01_3_7___9_c_d_12_6___b_25_a___c_30___2_5_b_e_41_6___8_d___56_b_d_e_61_4_5_7_9_b_d_70_1_5_7___9_c_d_84_6_9_f___92_5___7_b_a2_6___8_b___b2_5_6_8_9_c_f_c4_7_8_a_c_d_d1_2_4_9_adv2);
    PabloAST * EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9_adv2 = b_ef.createAdvance(EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9, 2);
    xfrm_ef[4] = b_ef.createOr(xfrm_ef[4], EF_f901_4_5_7___a_d_13_4_9_b_d_f___21_6_7_9_b_e___30_3___5_7_9_b_d___42_5_9_a_c_d_51_3_5_a___c_f_60_2___5_9_c_72_4___c_e_81_4_8_a_c_e___90_2___5_7___b_d_f_a3_4_6_7_9_b___d_f_b0_3_5_6_9_a_c_f_c2_3_8_9_b_d_f_d1_2_4_6___9_b___e0_6_b___f_f1_5___8_a_d_fa02_3_5___7_b_16_7_a_b_20_2_5_6_b_d___34_7___c_e___40_3_5_6_b_c_f_52___6_8_9_b_c_60_1_3_4_6_7_9___b_d_70_2___6_b_e_82_3_6_b_c_e_f_97_9_a_c_d_f_a1_3___5_7_8_c_e_f_b2_4_6_7_b_c_e_c2___6_8___a_c___e_d6___9_adv2);
    PabloAST * EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5_adv2 = b_ef.createAdvance(EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5, 2);
    xfrm_ef[5] = b_ef.createOr(xfrm_ef[5], EF_f901_5_6_d_e_10___2_8___a_c_f___21_3_5___9_c_d_f___32_4___6_b_e___40_4___6_9_b_f_50_7___9_d___61_3_7_8_b___e_72_9_a_d_e_81_2_4___7_c___e_92_6___a_d_e_a0_2_c_f_b0_3___5_a_b_e_f_c2_3_6_8_9_e_d1___4_6_7_d_f___e1_4_7_9_d___f_f2___5_7_9___d_fa01_6_7_12_6_8_a_d_e_20_e_32_3_5___7_9_d_f_40_2_5_7___a_c_57_a_c___e_60___2_4_5_c_d_73___6_8___a_c_e___81_4_5_a___c_91_5_6_a_c___f_a2_4___7_9___b_d_f_b2_3_6___8_b_c_c2_3_5___7_a_c_d1_4_5_adv2);
    PabloAST * EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9_adv2 = b_ef.createAdvance(EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9, 2);
    xfrm_ef[1] = b_ef.createOr(xfrm_ef[1], EF_f903_5___7_a_d___12_4___6_9_a_f_20_3_6_8_9_b___d_31_2_5_6_8_9_b_c_e___41_3_5_8___a_c_d_50_1_6_a_c___e_60___2_4___7_c_e_70_2_6_a_c_d_f___86_8_9_c___f_91_3_5_6_8_9_c___a5_7_d_e_b0_3_5_7_c___e_c0___3_7___b_f_d0_3_4_8_c_f___e6_c_e___f1_4___6_9___b_fa00_1_5_6_10_2_5_9_a_c___e_20_2_6_b_30___3_6_a___c_e_40_1_3_5_8_9_b_c_e___50_2_6_7_b_e_61___4_7___9_b_c_70_2_3_a_c___80_3_6_9_a_e_90_4_7___9_c___e_a0_5_9___b_b1_2_4_6_8___a_d___c0_2_5_6_8_9_c___d0_2_3_5_8_9_adv2);
    PabloAST * EF_fa6c_d6_adv3 = b_ef.createAdvance(EF_fa6c_d6, 3);
    xfrm_ef[1] = b_ef.createOr(xfrm_ef[1], EF_fa6c_d6_adv3);
    PabloAST * EF_facf_adv3 = b_ef.createAdvance(EF_facf, 3);
    xfrm_ef[0] = b_ef.createOr(xfrm_ef[0], EF_facf_adv3);
    PabloAST * EF_facf___d1_5___7_adv3 = b_ef.createAdvance(EF_facf___d1_5___7, 3);
    xfrm_ef[2] = b_ef.createOr(xfrm_ef[2], EF_facf___d1_5___7_adv3);
    PabloAST * EF_fad0_5_adv3 = b_ef.createAdvance(EF_fad0_5, 3);
    xfrm_ef[4] = b_ef.createOr(xfrm_ef[4], EF_fad0_5_adv3);
    PabloAST * EF_fad5_adv3 = b_ef.createAdvance(EF_fad5, 3);
    xfrm_ef[3] = b_ef.createOr(xfrm_ef[3], EF_fad5_adv3);

    for (unsigned i = 0; i < 8; i++) {
        b_ef.createAssign(XfrmVar[i], b_ef.createOr(XfrmVar[i], xfrm_ef[i]));
    }

    auto b_f0 = pb.createScope();
    PabloAST * pfx_f0_test = bnc.EQ(Basis, 0xf0);
    pb.createIf(pfx_f0_test, b_f0);
    std::vector<PabloAST *> xfrm_f0(8, All0);
    PabloAST * del_f0 = All0;

    UTF::UTF_Compiler b_f0_compiler(getInputStreamVar("Basis"), b_f0, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_f0_vars(18);
    std::vector<UnicodeSet> b_f0_usets(18);
    Var * F0_2f800 = b_f0.createVar("F0_2f800", All0);
    b_f0_vars[0] = F0_2f800;
    b_f0_usets[0] = F0_2f800_uset;
    Var * F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d = b_f0.createVar("F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d", All0);
    b_f0_vars[1] = F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d;
    b_f0_usets[1] = F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d_uset;
    Var * F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c = b_f0.createVar("F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c", All0);
    b_f0_vars[2] = F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c;
    b_f0_usets[2] = F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c_uset;
    Var * F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6 = b_f0.createVar("F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6", All0);
    b_f0_vars[3] = F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6;
    b_f0_usets[3] = F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6_uset;
    Var * F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c = b_f0.createVar("F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c", All0);
    b_f0_vars[4] = F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c;
    b_f0_usets[4] = F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c_uset;
    Var * F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d = b_f0.createVar("F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d", All0);
    b_f0_vars[5] = F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d;
    b_f0_usets[5] = F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d_uset;
    Var * F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6 = b_f0.createVar("F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6", All0);
    b_f0_vars[6] = F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6;
    b_f0_usets[6] = F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6_uset;
    Var * F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d = b_f0.createVar("F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d", All0);
    b_f0_vars[7] = F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d;
    b_f0_usets[7] = F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d_uset;
    Var * F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8 = b_f0.createVar("F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8", All0);
    b_f0_vars[8] = F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8;
    b_f0_usets[8] = F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8_uset;
    Var * F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c = b_f0.createVar("F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c", All0);
    b_f0_vars[9] = F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c;
    b_f0_usets[9] = F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c_uset;
    Var * F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d = b_f0.createVar("F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d", All0);
    b_f0_vars[10] = F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d;
    b_f0_usets[10] = F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d_uset;
    Var * F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8 = b_f0.createVar("F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8", All0);
    b_f0_vars[11] = F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8;
    b_f0_usets[11] = F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8_uset;
    Var * F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d = b_f0.createVar("F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d", All0);
    b_f0_vars[12] = F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d;
    b_f0_usets[12] = F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d_uset;
    Var * F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d = b_f0.createVar("F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d", All0);
    b_f0_vars[13] = F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d;
    b_f0_usets[13] = F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d_uset;
    Var * F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d = b_f0.createVar("F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d", All0);
    b_f0_vars[14] = F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d;
    b_f0_usets[14] = F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d_uset;
    Var * F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d = b_f0.createVar("F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d", All0);
    b_f0_vars[15] = F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d;
    b_f0_usets[15] = F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d_uset;
    Var * F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c = b_f0.createVar("F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c", All0);
    b_f0_vars[16] = F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c;
    b_f0_usets[16] = F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c_uset;
    Var * F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c = b_f0.createVar("F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c", All0);
    b_f0_vars[17] = F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c;
    b_f0_usets[17] = F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c_uset;

    b_f0_compiler.compile(b_f0_vars, b_f0_usets);
    PabloAST * F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d_adv1 = b_f0.createAdvance(F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d, 1);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], F0_2f800___6_d_12_6_9_34_8_89_f_91___3_8_a0_3___c1_3___6_8_9_b___d_f_d4___c_f___e2_4___6_8___b_d_f_f1_3___6_a_c___2f905_7___9_b_c_e_f_12_b_d_f_23_6_7_35_7_9_b_c_f_49_b_c_51_8_60_4_7_d_71___5_7_a___f_81___5_7_8_b___91_3___aa_c___b0_2___c4_6___9_d___d2_4___7_9___e1_5_d_f_f1_2_8_9_c_2fa03_8_d_e_10___4_6_d_adv1);
    PabloAST * F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c_adv1 = b_f0.createAdvance(F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c, 1);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], F0_2f800___b_d___12_4___e_20___3_5___34_6___66_9___75_7___82_4___7_b___e_90_3___5_9_a_c___f_d2_3_6_7_f8_2f91b_d_f_23_6_7_35_7_b___d_f_41___4_9_b___d_51_2_4_5_8_c___e_60_1_4_5_7_b_d_71_4_a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___9_d___d2_4___7_9___2fa0f_11_5___c_adv1);
    PabloAST * F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6_adv1 = b_f0.createAdvance(F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6, 1);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], F0_2f800___1b_d___8e_90___2_4___d5_7___2f979_b_c_e_80_1_4_6___a_e_92_7_a4___7_b_d___b2_f_c2_5_8_a___e_d3_8_9_d_ef_f2_4_8_9_c_2fa03_8_d_e_11_6_adv1);
    PabloAST * F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c_adv1 = b_f0.createAdvance(F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c, 1);
    xfrm_f0[6] = b_f0.createOr(xfrm_f0[6], F0_2f800___2_4___c_e___11_3___5_7___b_d___33_5___7_9___58_a___f_62___b_d___70_2___a_c_e___88_a___e_90_3___6_9___a3_5___b7_9___d_f___c9_b___dc_e___e2_4___b_d___f_f1___6_a_c___2f905_7___c_e_f_12___a_c_e_20___2_4_5_8___34_6_8_a_e___40_5___c_e___51_3_6___b_f_60_2___4_6___a_c___71_4_6_8___a_d_f_81___6_b___96_8___a3_7___a_c_e_f_b2___c4_6___a_d___d2_4___7_a___c_e_f_e2___4_6___c_e___f0_2___5_8___a_c_e___2fa00_2___8_a___f_11_5___c_adv1);
    PabloAST * F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d_adv1 = b_f0.createAdvance(F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d, 1);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], F0_2f803_c_d_12_3_6_c_f_24_34_8_59_60_1_7_8_c_71_6_b_d_83_8___a_f_91___3_6_7_b_a1_2_4_b8_e_c2_7_a_e_d0_1_6_d_e_e3_7_c_e_f0_2_7___9_b_2f906_a_d_10_1_6_b_2a_c_d_33_9_e_7a_d_f_80_2_3_5_9___d_f___91_3___6_8___a3_8___a_c_b3___e_c0_1_3_4_6_7_9_a_f___d2_4___7_9___ee_f0_1_3_5___7_a_b_d___2fa02_4___7_9___c_f_10_2___5_7___d_adv1);
    PabloAST * F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6_adv2 = b_f0.createAdvance(F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6, 2);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], F0_2f800___2_4___6_d_19_e_20___3_5___34_6___a_4b___50_2_3_5_60_1_3_4_6___b_d___72_4_6_85___7_b___95_7_9_a_c___f_a4_b2___e_c0___2_7_c___d1_4_5_7___e0_7_b___d_f_f1_3___7_9_2f908_9_b___d_f___12_d_f___26_8___d_33_a___c_e_40_5___8_a_52_4_5_9___b_d___64_7_b_d_f___79_80_1_6___d_f___9e_a0___3_b_d_b0_3___e_c0_5_8_d___d1_8_e_f_e5_7___b_d_f3_5_7___a_c_2fa00___2_4___7_b___12_6_adv2);
    PabloAST * F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d_adv2 = b_f0.createAdvance(F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d, 2);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], F0_2f800___2_4___6_c_d_12_3_6_9_f_24_3b___4f_52_3_5_9_60_1_7_8_73___82_4___7_b___e_90_4_5_9_a_c___f_a4_bf_c3___6_8_9_b___d_f_d4_5_8___c_f_e0_3_c_e_f2_8_a___2f912_6_b_23_6_a_c___38_a___c_e_40_5___8_a_d_52_4_5_65_6_8___c_e___70_6_8_9_b_c_e_84_7_8_d_e_9f_a4___a_c___b0_2___c0_2_5_8_a_d_e_d2___8_a___c_e___e1_5_c_e_f0_1_3___5_a_b_d_2fa01_b___f_11_5___d_adv2);
    PabloAST * F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8_adv2 = b_f0.createAdvance(F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], F0_2f802_4_8_a_b_e___11_4_6_9_c_d_f_21___3_5___8_c___33_5___a_e___40_2_5___9_b___d_50_4___8_c___e_60_3_4_9___b_71_3_5_6_a_80___2_4_7_8_a_e_90_6_8___b_d___f_a4___a_f___b1_4_6_7_a_b_d_e_c1_2_5_a_e___d0_2___6_8___a_e0_2_4___7_9_b_c_e___f0_4___8_b_c_f_2f901___4_9_b___10_2_7___9_c_d_22_8_e___30_2_8_d_e_43___7_9_a_c_d_f_52_4_6_8_d_e_60_1_4_5_7_a_b_d_f_71_2_7_8_a_c_e_80_1_e_96_8_a_e___a3_5_6_8_9_b_c_e_f_b1___6_a_b_d_f_c1_8_a_b_d_f_d1_3_4_6_8_9_b_f_e2_4___7_c_e___f0_4_5_7_b___f_2fa02_6___8_b___11_6___8_adv2);
    PabloAST * F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c_adv2 = b_f0.createAdvance(F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c, 2);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], F0_2f803___6_e___12_4___8_a___d_24_9___33_6___a_42___b_d_52_3_5_c___f_62_5_9___72_4_6_b___82_4_9_f_91_2_4_5_8___a_c___f_a1_2_6___b1_7___9_b___e_c0_1_8_9_b_e_d0_2___4_7___e0_6___a_f_f1_3___6_8_f_2f908_9_b___d_13___5_7___a_d_20___5_7_a_c___32_5_7_a_d_f___42_9_b___52_4_5_9___b_d___60_4___9_f_70_2_3_5_a_b_d_f_82_4_6_b_c_e___96_8_f_a4_6_8_9_b1_3___8_a_c1_3___b_d___d0_2___5_9_d_e_e2_3_7___a_c_f3_7___9_b___f_2fa01_3___a_f_12_6_9___c_adv2);
    PabloAST * F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d_adv2 = b_f0.createAdvance(F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d, 2);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], F0_2f803_7___18_a___33_5___7_9___50_2_3_5_9_60_1_7_8_76_89_f_91___3_7_8_a0_3___b7_9___d_f___c1_3___6_8___d_f_d2___d_f_e0_3_c_2f913___5_7___26_8_9_b_e___32_4_6_8___a_d_f___4d_51_2_4_5_8_60_4_7_d_71___5_7_a___85_7___c_e___96_8___a3_7___c_e_f_b1___c0_2_5_8_a_d_e_e0___c_e_f0_3_5___7_a_b_d_2fa01_10_2___4_d_adv2);
    PabloAST * F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8_adv2 = b_f0.createAdvance(F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8, 2);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], F0_2f805_6_9___c_13_5_7___d_f_25___8_36_7_9_a_c_40_1_5___a_51___3_5_8___b_60___2_5_7_8_c___70_2_6_9___b_d_80_3_5___8_a_91___5_9_a_a1_2_6___a_b2___4_7___9_b_f_c2___4_6___a_d_e_d0_4_8___c_e_e1_3_5_6_8_e_f_f1___3_8_a_e_f_2f905_b___e_11_2_6_a_e_f_24_5_9___d_31_2_7_9_b_c_e_40___2_a_c_f___51_4___7_d___f_61___3_5___9_b_e_75_8_9_b_e_80_3___5_b_c_e_97_9_a_c_f_a0_4_5_7_a_c___f_b2___6_9_b___d_c1_3___5_8_c_d2_6___9_f_e4___7_a_c_f1_5___9_b_e_f_2fa01_2_5___8_c___e_10_1_4_5_7_8_adv2);
    PabloAST * F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d_adv3 = b_f0.createAdvance(F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], F0_2f800___3_5___7_a_d___f_14_6_9_c_e_20_1_4_6_7_c_30_2_5_8_a_b_d_f_41_4_5_8_b_e___51_3_4_9_d_f_61_4_5_7_9_b_70___7_9___d_84___6_b___d_f_92_5___7_9_a_e_a0_1_3_7_9_b_d___f_b1_4_7_b___c0_6___9_e_d1_5_6_8_a___c_e___e2_b_c_f0_1_3___6_8_9_b_c_e_f_2f902_7___a_c_e_f_11_7_8_a_c_23_6_7_d___30_5_7_d_f_44___6_a_b_f_53_6_8_9_e_f_64_5_7_b_c_e___71_4_7___d_f_80_4___6_8_d_e_90_2_4_7_8_a_c_e_a0_1_3_6_8_b_d_e_b0___3_6_9_c0_2___4_6___b_d___d4_6_8___b_d_f_e4___9_c___e_f0_6_7_9_b_d_e_2fa00_1_4_7_9_b___e_11___4_a___d_adv3);
    PabloAST * F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d_adv3 = b_f0.createAdvance(F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d, 3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], F0_2f800_4_6_7_a_b_d_10___2_6_8___b_e_21_3_6_9___33_5_6_b_d___f_41_4_7_b_e_50___3_5___9_b_d___f_62_5_8_a___e_72___8_a___c_f_80_3_5_7_a_c_e___90_4_5_7_c_f_a2_3_6_8_9_b_c_b1_5_7___a_c_e___c0_2_8_9_c___f_d1_3_4_7_8_b___e0_2_3_c___f2_5_6_a_b_2f900_1_4_8_9_d_e_10___2_5_7___b_d_f_21_4_a___d_31_7_9_c_f_40_4_b_50_2___4_7_8_b_62_5___7_b___d_f___71_3___6_8_a_c___81_4_8_a_e_f_91_4_8_9_b_c_a2_3_5_7_a_f_b6_7_9_c_d_c4_8_b_f_d0_2_4___6_8_9_b_c_f_e1_2_4_6___8_a_d_e_f0___2_4_5_7___9_c_e_f_2fa01_4_6_8_10_2___5_7___d_adv3);
    PabloAST * F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d_adv3 = b_f0.createAdvance(F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], F0_2f800_1_5_6_a_f_12_3_5_6_8_b_d_e_20_6_9___b_d_f___34_6___8_a_b_47_9___f_51_2_4_5_9___c_e___63_5_7_d_f_71_2_9_b___d_80_3___9_b___d_f_90_7_9_b_e_a1_2_6_7_c_d_b3_4_c___f_c1___5_7_b_f_d0_5___8_a_c_d_e3___5_7___9_c_f_f2_4___7_e___2f900_3_4_6_a_c_d_10_1_3___5_7_9_b___d_20_3_b_e_32_3_6_8_a_b_d_43_5___7_b_c_50_4_6_7_9_d_e_61_3_9___f_72_4___8_a_c_d_f_80_4_5_8_9_b_c_f___91_5_7_8_a_c_e_f_a1___3_5_7___a_c_f_b2_4_6_7_a_e_f_c2_7_8_c_e___d1_3_4_6_9_c_e_e1_5___8_c_f_f7_9_a_c_d_2fa00_4_7_9___b_f___11_3_5_7_9_b_d_adv3);
    PabloAST * F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d_adv3 = b_f0.createAdvance(F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d, 3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], F0_2f800_1_6_8_9_c_d_f_10_4___6_8___a_e_20_2___4_8_b_d___f_34_6_8_9_b_e_f_41___3_7_9_b_d___52_4_5_8_9_c_e_f_63_5_8_a_b_e_73_5___80_4_6_a___d_f_93___6_9___b_d_a1___4_a_c_f_b8_b___d_f_c4___6_c_e___d1_4_6_a_d_e_e1_5___7_d_e_f0_2_7_8_a_d_e_2f900_1_4_6_7_a_d_e_11_2_7_a___d_f_21_2_5___8_c_d_f_31_2_4_b_d_f___47_9_c_d_50___2_5___b_d___f_66_70_2_3_a_c___e_80___2_5_6_8___a_d_f_90_6_8_a___c_e_a0_3_4_6_8___a_e_b2_5___a_c___e_c2_6___a_d_d0_1_3___5_8___a_e0_2___4_6___d_f_f2_3_5_6_8_a_b_e_f_2fa01_3_7_9_a_c_10_2_3_9_a_d_adv3);
    PabloAST * F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c_adv3 = b_f0.createAdvance(F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c, 3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], F0_2f800_1_3___6_8_a_d_10_1_3_4_8_a_b_d_21_4___6_9___e_36_7_b_c_e_40_1_4_8_9_f_51_3_6___b_e_f_63_5_9_c___e_71_3_4_6_9_a_c_80___8_a___d_90___2_4_5_7_9___b_f_a0_2___9_b_f_b2___4_6___8_c_c1___4_6_7_9_b_c_d1_6_a_f___e2_4___8_d_e_f1_2_4_7_8_a_d_f___2f901_3_4_7___a_e_f_13_8___b_d_f_22___5_7_9_e_f_31___3_7_9_a_d_f___42_4_9_51_2_7_8_c___60_2_3_5_6_a_b_d___70_3___5_8_a_b_82_5___8_e_92___8_a___e_a0___3_6_a_b_e_b1_3_4_8_9_e___c0_2___5_8___d_f_d0_3_4_7_8_b_c_f_e0_2_4_6_8_9_c_e_f_f3_6_7_b_d___2fa05_7_9_c_e_f_11_5_7_8_c_adv3);
    PabloAST * F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c_adv3 = b_f0.createAdvance(F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c, 3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], F0_2f802_5_8_c___f_11___4_a___c_20_1_5___8_31_8___a_c_e___42_4_6___9_c___51_3_5_7_c_d_61_2_4_d_e_70_4___8_a___d_86___d_90_2___5_7___9_c_f_a1___3_5___8_c_b0___2_6_7_a___c_e_c3_4_7_8_e_f_d3_6___a_c___e_e1___9_b___f0_4_5_7___9_b_c_2f900___3_7___9_c_e_10_1_3___5_9___b_d_e_22_3_6___b_f_35_9___b_e_f_41_3___5_8_c_f_51_4_8_d_60_1_3_4_7_9_a_c_72_9_a_c_e___80_4___6_9___b_d_f_90_3_4_7_b_c_a0___6_9_e_f_b1_3_5___7_9_b_d_c2___5_9_c_d_f_d1___4_8_b_d___e0_2_3_5_7_8_a_b_d_f___f1_4___6_8___a_c_d_2fa01_5_8_b_d_e_10_2_5_7_8_c_adv3);
    del_f0 = b_f0.createOr(del_f0, F0_2f800);

    for (unsigned i = 0; i < 8; i++) {
        b_f0.createAssign(XfrmVar[i], b_f0.createOr(XfrmVar[i], xfrm_f0[i]));
    }

    b_f0.createAssign(DeleteVar, b_f0.createOr(DeleteVar, del_f0));

    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        pb.createAssign(xfrm_out, pb.createXor(Basis[i], XfrmVar[i]));
    }
    Var * MaskOutputVar = pb.createExtract(getOutputStreamVar("SelectMask"), pb.getInteger(0));
    pb.createAssign(MaskOutputVar, pb.createInFile(pb.createNot(DeleteVar)));
}


//
class NonStarterDecomposition : public PabloKernel {
public:
    NonStarterDecomposition
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * NSD_Basis);
protected:
    void generatePabloMethod() override;
};

NonStarterDecomposition::NonStarterDecomposition
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * NSD_Basis)
: PabloKernel(ts, "NonStarterDecomposition",
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"NSD_Basis", NSD_Basis}}) {}

void NonStarterDecomposition::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<Var *> NSD_Var(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        NSD_Var[i] = pb.createVar("NSD_Basis" + std::to_string(i), Basis[i]);
    }
    UTF::UTF_Compiler pb_compiler(getInputStreamVar("Basis"), pb, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> pb_vars(4);
    std::vector<UnicodeSet> pb_usets(4);
    Var * CD_344 = pb.createVar("CD_344", All0);
    pb_vars[0] = CD_344;
    pb_usets[0] = CD_344_uset;
    Var * E0_f73 = pb.createVar("E0_f73", All0);
    pb_vars[1] = E0_f73;
    pb_usets[1] = E0_f73_uset;
    Var * E0_f75 = pb.createVar("E0_f75", All0);
    pb_vars[2] = E0_f75;
    pb_usets[2] = E0_f75_uset;
    Var * E0_f81 = pb.createVar("E0_f81", All0);
    pb_vars[3] = E0_f81;
    pb_usets[3] = E0_f81_uset;

    pb_compiler.compile(pb_vars, pb_usets);
//  Case for 344
    auto b_344 = pb.createScope();
    pb.createIf(CD_344, b_344);
    std::vector<PabloAST *> xfrm_344(8, All0);
    PabloAST * m_344_0 = CD_344;
    xfrm_344[0] = b_344.createOr(xfrm_344[0], m_344_0);
    PabloAST * m_344_1 = b_344.createAdvance(CD_344, 1);
    xfrm_344[2] = b_344.createOr(xfrm_344[2], m_344_1);
    xfrm_344[3] = b_344.createOr(xfrm_344[3], m_344_1);

    PabloAST * ins_301 = b_344.createAdvance(CD_344, 2);
    PabloAST * m_301_0 = ins_301;
    xfrm_344[2] = b_344.createOr(xfrm_344[2], m_301_0);
    xfrm_344[3] = b_344.createOr(xfrm_344[3], m_301_0);
    xfrm_344[6] = b_344.createOr(xfrm_344[6], m_301_0);
    xfrm_344[7] = b_344.createOr(xfrm_344[7], m_301_0);
    PabloAST * m_301_1 = b_344.createAdvance(ins_301, 1);
    xfrm_344[0] = b_344.createOr(xfrm_344[0], m_301_1);
    xfrm_344[7] = b_344.createOr(xfrm_344[7], m_301_1);

    for (unsigned i = 0; i < 8; i++) {
        b_344.createAssign(NSD_Var[i], b_344.createXor(NSD_Var[i], xfrm_344[i]));
    }
//  Case for f73
    auto b_f73 = pb.createScope();
    pb.createIf(E0_f73, b_f73);
    std::vector<PabloAST *> xfrm_f73(8, All0);
    PabloAST * m_f73_2 = b_f73.createAdvance(E0_f73, 2);
    xfrm_f73[1] = b_f73.createOr(xfrm_f73[1], m_f73_2);

    PabloAST * ins_f72 = b_f73.createAdvance(E0_f73, 3);
    PabloAST * m_f72_0 = ins_f72;
    xfrm_f73[5] = b_f73.createOr(xfrm_f73[5], m_f72_0);
    xfrm_f73[6] = b_f73.createOr(xfrm_f73[6], m_f72_0);
    xfrm_f73[7] = b_f73.createOr(xfrm_f73[7], m_f72_0);
    PabloAST * m_f72_1 = b_f73.createAdvance(ins_f72, 1);
    xfrm_f73[0] = b_f73.createOr(xfrm_f73[0], m_f72_1);
    xfrm_f73[2] = b_f73.createOr(xfrm_f73[2], m_f72_1);
    xfrm_f73[3] = b_f73.createOr(xfrm_f73[3], m_f72_1);
    xfrm_f73[4] = b_f73.createOr(xfrm_f73[4], m_f72_1);
    xfrm_f73[5] = b_f73.createOr(xfrm_f73[5], m_f72_1);
    xfrm_f73[7] = b_f73.createOr(xfrm_f73[7], m_f72_1);
    PabloAST * m_f72_2 = b_f73.createAdvance(ins_f72, 2);
    xfrm_f73[1] = b_f73.createOr(xfrm_f73[1], m_f72_2);
    xfrm_f73[4] = b_f73.createOr(xfrm_f73[4], m_f72_2);
    xfrm_f73[5] = b_f73.createOr(xfrm_f73[5], m_f72_2);
    xfrm_f73[7] = b_f73.createOr(xfrm_f73[7], m_f72_2);

    for (unsigned i = 0; i < 8; i++) {
        b_f73.createAssign(NSD_Var[i], b_f73.createXor(NSD_Var[i], xfrm_f73[i]));
    }
//  Case for f75
    auto b_f75 = pb.createScope();
    pb.createIf(E0_f75, b_f75);
    std::vector<PabloAST *> xfrm_f75(8, All0);
    PabloAST * m_f75_2 = b_f75.createAdvance(E0_f75, 2);
    xfrm_f75[2] = b_f75.createOr(xfrm_f75[2], m_f75_2);

    PabloAST * ins_f74 = b_f75.createAdvance(E0_f75, 3);
    PabloAST * m_f74_0 = ins_f74;
    xfrm_f75[5] = b_f75.createOr(xfrm_f75[5], m_f74_0);
    xfrm_f75[6] = b_f75.createOr(xfrm_f75[6], m_f74_0);
    xfrm_f75[7] = b_f75.createOr(xfrm_f75[7], m_f74_0);
    PabloAST * m_f74_1 = b_f75.createAdvance(ins_f74, 1);
    xfrm_f75[0] = b_f75.createOr(xfrm_f75[0], m_f74_1);
    xfrm_f75[2] = b_f75.createOr(xfrm_f75[2], m_f74_1);
    xfrm_f75[3] = b_f75.createOr(xfrm_f75[3], m_f74_1);
    xfrm_f75[4] = b_f75.createOr(xfrm_f75[4], m_f74_1);
    xfrm_f75[5] = b_f75.createOr(xfrm_f75[5], m_f74_1);
    xfrm_f75[7] = b_f75.createOr(xfrm_f75[7], m_f74_1);
    PabloAST * m_f74_2 = b_f75.createAdvance(ins_f74, 2);
    xfrm_f75[2] = b_f75.createOr(xfrm_f75[2], m_f74_2);
    xfrm_f75[4] = b_f75.createOr(xfrm_f75[4], m_f74_2);
    xfrm_f75[5] = b_f75.createOr(xfrm_f75[5], m_f74_2);
    xfrm_f75[7] = b_f75.createOr(xfrm_f75[7], m_f74_2);

    for (unsigned i = 0; i < 8; i++) {
        b_f75.createAssign(NSD_Var[i], b_f75.createXor(NSD_Var[i], xfrm_f75[i]));
    }
//  Case for f81
    auto b_f81 = pb.createScope();
    pb.createIf(E0_f81, b_f81);
    std::vector<PabloAST *> xfrm_f81(8, All0);
    PabloAST * m_f81_1 = b_f81.createAdvance(E0_f81, 1);
    xfrm_f81[0] = b_f81.createOr(xfrm_f81[0], m_f81_1);
    xfrm_f81[1] = b_f81.createOr(xfrm_f81[1], m_f81_1);
    PabloAST * m_f81_2 = b_f81.createAdvance(E0_f81, 2);
    xfrm_f81[4] = b_f81.createOr(xfrm_f81[4], m_f81_2);
    xfrm_f81[5] = b_f81.createOr(xfrm_f81[5], m_f81_2);

    PabloAST * ins_f80 = b_f81.createAdvance(E0_f81, 3);
    PabloAST * m_f80_0 = ins_f80;
    xfrm_f81[5] = b_f81.createOr(xfrm_f81[5], m_f80_0);
    xfrm_f81[6] = b_f81.createOr(xfrm_f81[6], m_f80_0);
    xfrm_f81[7] = b_f81.createOr(xfrm_f81[7], m_f80_0);
    PabloAST * m_f80_1 = b_f81.createAdvance(ins_f80, 1);
    xfrm_f81[1] = b_f81.createOr(xfrm_f81[1], m_f80_1);
    xfrm_f81[2] = b_f81.createOr(xfrm_f81[2], m_f80_1);
    xfrm_f81[3] = b_f81.createOr(xfrm_f81[3], m_f80_1);
    xfrm_f81[4] = b_f81.createOr(xfrm_f81[4], m_f80_1);
    xfrm_f81[5] = b_f81.createOr(xfrm_f81[5], m_f80_1);
    xfrm_f81[7] = b_f81.createOr(xfrm_f81[7], m_f80_1);
    PabloAST * m_f80_2 = b_f81.createAdvance(ins_f80, 2);
    xfrm_f81[7] = b_f81.createOr(xfrm_f81[7], m_f80_2);

    for (unsigned i = 0; i < 8; i++) {
        b_f81.createAssign(NSD_Var[i], b_f81.createXor(NSD_Var[i], xfrm_f81[i]));
    }

    writeOutputStreamSet("NSD_Basis", NSD_Var);
}


//
class ShortComposableTranslation : public PabloKernel {
public:
    ShortComposableTranslation
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * DeletePrior, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

ShortComposableTranslation::ShortComposableTranslation
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * DeletePrior, StreamSet * XfrmBasis)
: PabloKernel(ts, "ShortComposableTranslation",
{Binding{"Basis", Basis, FixedRate(), LookAhead(3)}},
{Binding{"DeletePrior", DeletePrior}, Binding{"XfrmBasis", XfrmBasis}}) {}

void ShortComposableTranslation::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    Var * DeletePriorVar = pb.createVar("DeletePriorVar", All0);
    std::vector<Var *> XfrmVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        XfrmVar[i] = pb.createVar("XfrmBasis" + std::to_string(i), All0);
    }

    auto b_e0 = pb.createScope();
    PabloAST * pfx_e0_test = bnc.EQ(Basis, 0xe0);
    pb.createIf(pfx_e0_test, b_e0);
    std::vector<PabloAST *> xfrm_e0(8, All0);
    PabloAST * del_prior_e0 = All0;
    UTF::UTF_Compiler b_e0_compiler(getInputStreamVar("Basis"), b_e0, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_e0_vars(39);
    std::vector<UnicodeSet> b_e0_usets(39);
    Var * E0_9c7 = b_e0.createVar("E0_9c7", All0);
    b_e0_vars[0] = E0_9c7;
    b_e0_usets[0] = E0_9c7_uset;
    Var * E0_9be = b_e0.createVar("E0_9be", All0);
    b_e0_vars[1] = E0_9be;
    b_e0_usets[1] = E0_9be_uset;
    Var * E0_9d7 = b_e0.createVar("E0_9d7", All0);
    b_e0_vars[2] = E0_9d7;
    b_e0_usets[2] = E0_9d7_uset;
    Var * E0_b47 = b_e0.createVar("E0_b47", All0);
    b_e0_vars[3] = E0_b47;
    b_e0_usets[3] = E0_b47_uset;
    Var * E0_b56 = b_e0.createVar("E0_b56", All0);
    b_e0_vars[4] = E0_b56;
    b_e0_usets[4] = E0_b56_uset;
    Var * E0_b3e = b_e0.createVar("E0_b3e", All0);
    b_e0_vars[5] = E0_b3e;
    b_e0_usets[5] = E0_b3e_uset;
    Var * E0_b57 = b_e0.createVar("E0_b57", All0);
    b_e0_vars[6] = E0_b57;
    b_e0_usets[6] = E0_b57_uset;
    Var * E0_b92 = b_e0.createVar("E0_b92", All0);
    b_e0_vars[7] = E0_b92;
    b_e0_usets[7] = E0_b92_uset;
    Var * E0_bd7 = b_e0.createVar("E0_bd7", All0);
    b_e0_vars[8] = E0_bd7;
    b_e0_usets[8] = E0_bd7_uset;
    Var * E0_bc6 = b_e0.createVar("E0_bc6", All0);
    b_e0_vars[9] = E0_bc6;
    b_e0_usets[9] = E0_bc6_uset;
    Var * E0_bbe = b_e0.createVar("E0_bbe", All0);
    b_e0_vars[10] = E0_bbe;
    b_e0_usets[10] = E0_bbe_uset;
    Var * E0_bc7 = b_e0.createVar("E0_bc7", All0);
    b_e0_vars[11] = E0_bc7;
    b_e0_usets[11] = E0_bc7_uset;
    Var * E0_cbf = b_e0.createVar("E0_cbf", All0);
    b_e0_vars[12] = E0_cbf;
    b_e0_usets[12] = E0_cbf_uset;
    Var * E0_cd5 = b_e0.createVar("E0_cd5", All0);
    b_e0_vars[13] = E0_cd5;
    b_e0_usets[13] = E0_cd5_uset;
    Var * E0_cc6 = b_e0.createVar("E0_cc6", All0);
    b_e0_vars[14] = E0_cc6;
    b_e0_usets[14] = E0_cc6_uset;
    Var * E0_cd6 = b_e0.createVar("E0_cd6", All0);
    b_e0_vars[15] = E0_cd6;
    b_e0_usets[15] = E0_cd6_uset;
    Var * E0_cc2 = b_e0.createVar("E0_cc2", All0);
    b_e0_vars[16] = E0_cc2;
    b_e0_usets[16] = E0_cc2_uset;
    Var * E0_cca = b_e0.createVar("E0_cca", All0);
    b_e0_vars[17] = E0_cca;
    b_e0_usets[17] = E0_cca_uset;
    Var * E0_d46 = b_e0.createVar("E0_d46", All0);
    b_e0_vars[18] = E0_d46;
    b_e0_usets[18] = E0_d46_uset;
    Var * E0_d3e = b_e0.createVar("E0_d3e", All0);
    b_e0_vars[19] = E0_d3e;
    b_e0_usets[19] = E0_d3e_uset;
    Var * E0_d57 = b_e0.createVar("E0_d57", All0);
    b_e0_vars[20] = E0_d57;
    b_e0_usets[20] = E0_d57_uset;
    Var * E0_d47 = b_e0.createVar("E0_d47", All0);
    b_e0_vars[21] = E0_d47;
    b_e0_usets[21] = E0_d47_uset;
    Var * E0_dd9 = b_e0.createVar("E0_dd9", All0);
    b_e0_vars[22] = E0_dd9;
    b_e0_usets[22] = E0_dd9_uset;
    Var * E0_dcf = b_e0.createVar("E0_dcf", All0);
    b_e0_vars[23] = E0_dcf;
    b_e0_usets[23] = E0_dcf_uset;
    Var * E0_ddf = b_e0.createVar("E0_ddf", All0);
    b_e0_vars[24] = E0_ddf;
    b_e0_usets[24] = E0_ddf_uset;
    Var * E0_f40 = b_e0.createVar("E0_f40", All0);
    b_e0_vars[25] = E0_f40;
    b_e0_usets[25] = E0_f40_uset;
    Var * E0_fb5 = b_e0.createVar("E0_fb5", All0);
    b_e0_vars[26] = E0_fb5;
    b_e0_usets[26] = E0_fb5_uset;
    Var * E0_f42 = b_e0.createVar("E0_f42", All0);
    b_e0_vars[27] = E0_f42;
    b_e0_usets[27] = E0_f42_uset;
    Var * E0_fb7 = b_e0.createVar("E0_fb7", All0);
    b_e0_vars[28] = E0_fb7;
    b_e0_usets[28] = E0_fb7_uset;
    Var * E0_f4c = b_e0.createVar("E0_f4c", All0);
    b_e0_vars[29] = E0_f4c;
    b_e0_usets[29] = E0_f4c_uset;
    Var * E0_f51 = b_e0.createVar("E0_f51", All0);
    b_e0_vars[30] = E0_f51;
    b_e0_usets[30] = E0_f51_uset;
    Var * E0_f56 = b_e0.createVar("E0_f56", All0);
    b_e0_vars[31] = E0_f56;
    b_e0_usets[31] = E0_f56_uset;
    Var * E0_f5b = b_e0.createVar("E0_f5b", All0);
    b_e0_vars[32] = E0_f5b;
    b_e0_usets[32] = E0_f5b_uset;
    Var * E0_f90 = b_e0.createVar("E0_f90", All0);
    b_e0_vars[33] = E0_f90;
    b_e0_usets[33] = E0_f90_uset;
    Var * E0_f92 = b_e0.createVar("E0_f92", All0);
    b_e0_vars[34] = E0_f92;
    b_e0_usets[34] = E0_f92_uset;
    Var * E0_f9c = b_e0.createVar("E0_f9c", All0);
    b_e0_vars[35] = E0_f9c;
    b_e0_usets[35] = E0_f9c_uset;
    Var * E0_fa1 = b_e0.createVar("E0_fa1", All0);
    b_e0_vars[36] = E0_fa1;
    b_e0_usets[36] = E0_fa1_uset;
    Var * E0_fa6 = b_e0.createVar("E0_fa6", All0);
    b_e0_vars[37] = E0_fa6;
    b_e0_usets[37] = E0_fa6_uset;
    Var * E0_fab = b_e0.createVar("E0_fab", All0);
    b_e0_vars[38] = E0_fab;
    b_e0_usets[38] = E0_fab_uset;

    b_e0_compiler.compile(b_e0_vars, b_e0_usets);
//  Cases for 9c7
    PabloAST * after_9c7 = b_e0.createAdvance(E0_9c7, 3);
//     9c7 + 9be => 9cb
    PabloAST * found_9c7_9be = b_e0.createAnd(after_9c7, E0_9be);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_9c7_9be);
    PabloAST * m_9be_9cb_1 = b_e0.createAdvance(found_9c7_9be, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_9be_9cb_1);
    PabloAST * m_9be_9cb_2 = b_e0.createAdvance(found_9c7_9be, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_9be_9cb_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_9be_9cb_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_9be_9cb_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_9be_9cb_2);
//     9c7 + 9d7 => 9cc
    PabloAST * found_9c7_9d7 = b_e0.createAnd(after_9c7, E0_9d7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_9c7_9d7);
    PabloAST * m_9d7_9cc_2 = b_e0.createAdvance(found_9c7_9d7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_9d7_9cc_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_9d7_9cc_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_9d7_9cc_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_9d7_9cc_2);
//  Cases for b47
    PabloAST * after_b47 = b_e0.createAdvance(E0_b47, 3);
//     b47 + b56 => b48
    PabloAST * found_b47_b56 = b_e0.createAnd(after_b47, E0_b56);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b56);
    PabloAST * m_b56_b48_2 = b_e0.createAdvance(found_b47_b56, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_b56_b48_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_b56_b48_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_b56_b48_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_b56_b48_2);
//     b47 + b3e => b4b
    PabloAST * found_b47_b3e = b_e0.createAnd(after_b47, E0_b3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b3e);
    PabloAST * m_b3e_b4b_1 = b_e0.createAdvance(found_b47_b3e, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_b3e_b4b_1);
    PabloAST * m_b3e_b4b_2 = b_e0.createAdvance(found_b47_b3e, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_b3e_b4b_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_b3e_b4b_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_b3e_b4b_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_b3e_b4b_2);
//     b47 + b57 => b4c
    PabloAST * found_b47_b57 = b_e0.createAnd(after_b47, E0_b57);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b57);
    PabloAST * m_b57_b4c_2 = b_e0.createAdvance(found_b47_b57, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_b57_b4c_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_b57_b4c_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_b57_b4c_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_b57_b4c_2);
//  Cases for b92
    PabloAST * after_b92 = b_e0.createAdvance(E0_b92, 3);
//     b92 + bd7 => b94
    PabloAST * found_b92_bd7 = b_e0.createAnd(after_b92, E0_bd7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b92_bd7);
    PabloAST * m_bd7_b94_1 = b_e0.createAdvance(found_b92_bd7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bd7_b94_1);
    PabloAST * m_bd7_b94_2 = b_e0.createAdvance(found_b92_bd7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bd7_b94_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_bd7_b94_2);
//  Cases for bc6
    PabloAST * after_bc6 = b_e0.createAdvance(E0_bc6, 3);
//     bc6 + bbe => bca
    PabloAST * found_bc6_bbe = b_e0.createAnd(after_bc6, E0_bbe);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc6_bbe);
    PabloAST * m_bbe_bca_1 = b_e0.createAdvance(found_bc6_bbe, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bbe_bca_1);
    PabloAST * m_bbe_bca_2 = b_e0.createAdvance(found_bc6_bbe, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_bbe_bca_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_bbe_bca_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_bbe_bca_2);
//     bc6 + bd7 => bcc
    PabloAST * found_bc6_bd7 = b_e0.createAnd(after_bc6, E0_bd7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc6_bd7);
    PabloAST * m_bd7_bcc_2 = b_e0.createAdvance(found_bc6_bd7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bd7_bcc_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_bd7_bcc_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_bd7_bcc_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_bd7_bcc_2);
//  Cases for bc7
    PabloAST * after_bc7 = b_e0.createAdvance(E0_bc7, 3);
//     bc7 + bbe => bcb
    PabloAST * found_bc7_bbe = b_e0.createAnd(after_bc7, E0_bbe);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc7_bbe);
    PabloAST * m_bbe_bcb_1 = b_e0.createAdvance(found_bc7_bbe, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bbe_bcb_1);
    PabloAST * m_bbe_bcb_2 = b_e0.createAdvance(found_bc7_bbe, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_bbe_bcb_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_bbe_bcb_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_bbe_bcb_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_bbe_bcb_2);
//  Cases for cbf
    PabloAST * after_cbf = b_e0.createAdvance(E0_cbf, 3);
//     cbf + cd5 => cc0
    PabloAST * found_cbf_cd5 = b_e0.createAnd(after_cbf, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cbf_cd5);
    PabloAST * m_cd5_cc0_2 = b_e0.createAdvance(found_cbf_cd5, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_cd5_cc0_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_cd5_cc0_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_cd5_cc0_2);
//  Cases for cc6
    PabloAST * after_cc6 = b_e0.createAdvance(E0_cc6, 3);
//     cc6 + cd5 => cc7
    PabloAST * found_cc6_cd5 = b_e0.createAnd(after_cc6, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cd5);
    PabloAST * m_cd5_cc7_2 = b_e0.createAdvance(found_cc6_cd5, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_cd5_cc7_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_cd5_cc7_2);
//     cc6 + cd6 => cc8
    PabloAST * found_cc6_cd6 = b_e0.createAnd(after_cc6, E0_cd6);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cd6);
    PabloAST * m_cd6_cc8_2 = b_e0.createAdvance(found_cc6_cd6, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_cd6_cc8_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_cd6_cc8_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_cd6_cc8_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_cd6_cc8_2);
//     cc6 + cc2 => cca
    PabloAST * found_cc6_cc2 = b_e0.createAnd(after_cc6, E0_cc2);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cc2);
    PabloAST * m_cc2_cca_2 = b_e0.createAdvance(found_cc6_cc2, 2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_cc2_cca_2);
//  Cases for d46
    PabloAST * after_d46 = b_e0.createAdvance(E0_d46, 3);
//     d46 + d3e => d4a
    PabloAST * found_d46_d3e = b_e0.createAnd(after_d46, E0_d3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d46_d3e);
    PabloAST * m_d3e_d4a_1 = b_e0.createAdvance(found_d46_d3e, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_d3e_d4a_1);
    PabloAST * m_d3e_d4a_2 = b_e0.createAdvance(found_d46_d3e, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_d3e_d4a_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_d3e_d4a_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_d3e_d4a_2);
//     d46 + d57 => d4c
    PabloAST * found_d46_d57 = b_e0.createAnd(after_d46, E0_d57);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d46_d57);
    PabloAST * m_d57_d4c_2 = b_e0.createAdvance(found_d46_d57, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_d57_d4c_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_d57_d4c_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_d57_d4c_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_d57_d4c_2);
//  Cases for d47
    PabloAST * after_d47 = b_e0.createAdvance(E0_d47, 3);
//     d47 + d3e => d4b
    PabloAST * found_d47_d3e = b_e0.createAnd(after_d47, E0_d3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d47_d3e);
    PabloAST * m_d3e_d4b_1 = b_e0.createAdvance(found_d47_d3e, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_d3e_d4b_1);
    PabloAST * m_d3e_d4b_2 = b_e0.createAdvance(found_d47_d3e, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_d3e_d4b_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_d3e_d4b_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_d3e_d4b_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_d3e_d4b_2);
//  Cases for dd9
    PabloAST * after_dd9 = b_e0.createAdvance(E0_dd9, 3);
//     dd9 + dcf => ddc
    PabloAST * found_dd9_dcf = b_e0.createAnd(after_dd9, E0_dcf);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_dd9_dcf);
    PabloAST * m_dcf_ddc_2 = b_e0.createAdvance(found_dd9_dcf, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_dcf_ddc_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_dcf_ddc_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_dcf_ddc_2);
//     dd9 + ddf => dde
    PabloAST * found_dd9_ddf = b_e0.createAnd(after_dd9, E0_ddf);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_dd9_ddf);
    PabloAST * m_ddf_dde_2 = b_e0.createAdvance(found_dd9_ddf, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_ddf_dde_2);
//  Cases for f40
    PabloAST * after_f40 = b_e0.createAdvance(E0_f40, 3);
//     f40 + fb5 => f69
    PabloAST * found_f40_fb5 = b_e0.createAnd(after_f40, E0_fb5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f40_fb5);
    PabloAST * m_fb5_f69_1 = b_e0.createAdvance(found_f40_fb5, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb5_f69_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb5_f69_1);
    PabloAST * m_fb5_f69_2 = b_e0.createAdvance(found_f40_fb5, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb5_f69_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb5_f69_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb5_f69_2);
//  Cases for f42
    PabloAST * after_f42 = b_e0.createAdvance(E0_f42, 3);
//     f42 + fb7 => f43
    PabloAST * found_f42_fb7 = b_e0.createAnd(after_f42, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f42_fb7);
    PabloAST * m_fb7_f43_1 = b_e0.createAdvance(found_f42_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f43_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f43_1);
    PabloAST * m_fb7_f43_2 = b_e0.createAdvance(found_f42_fb7, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb7_f43_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb7_f43_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f43_2);
//  Cases for f4c
    PabloAST * after_f4c = b_e0.createAdvance(E0_f4c, 3);
//     f4c + fb7 => f4d
    PabloAST * found_f4c_fb7 = b_e0.createAnd(after_f4c, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f4c_fb7);
    PabloAST * m_fb7_f4d_1 = b_e0.createAdvance(found_f4c_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f4d_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f4d_1);
    PabloAST * m_fb7_f4d_2 = b_e0.createAdvance(found_f4c_fb7, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f4d_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb7_f4d_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb7_f4d_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f4d_2);
//  Cases for f51
    PabloAST * after_f51 = b_e0.createAdvance(E0_f51, 3);
//     f51 + fb7 => f52
    PabloAST * found_f51_fb7 = b_e0.createAnd(after_f51, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f51_fb7);
    PabloAST * m_fb7_f52_1 = b_e0.createAdvance(found_f51_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f52_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f52_1);
    PabloAST * m_fb7_f52_2 = b_e0.createAdvance(found_f51_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f52_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb7_f52_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f52_2);
//  Cases for f56
    PabloAST * after_f56 = b_e0.createAdvance(E0_f56, 3);
//     f56 + fb7 => f57
    PabloAST * found_f56_fb7 = b_e0.createAnd(after_f56, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f56_fb7);
    PabloAST * m_fb7_f57_1 = b_e0.createAdvance(found_f56_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f57_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f57_1);
    PabloAST * m_fb7_f57_2 = b_e0.createAdvance(found_f56_fb7, 2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f57_2);
//  Cases for f5b
    PabloAST * after_f5b = b_e0.createAdvance(E0_f5b, 3);
//     f5b + fb7 => f5c
    PabloAST * found_f5b_fb7 = b_e0.createAnd(after_f5b, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f5b_fb7);
    PabloAST * m_fb7_f5c_1 = b_e0.createAdvance(found_f5b_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f5c_1);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f5c_1);
    PabloAST * m_fb7_f5c_2 = b_e0.createAdvance(found_f5b_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_f5c_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f5c_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb7_f5c_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f5c_2);
//  Cases for f90
    PabloAST * after_f90 = b_e0.createAdvance(E0_f90, 3);
//     f90 + fb5 => fb9
    PabloAST * found_f90_fb5 = b_e0.createAnd(after_f90, E0_fb5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f90_fb5);
    PabloAST * m_fb5_fb9_2 = b_e0.createAdvance(found_f90_fb5, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb5_fb9_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb5_fb9_2);
//  Cases for f92
    PabloAST * after_f92 = b_e0.createAdvance(E0_f92, 3);
//     f92 + fb7 => f93
    PabloAST * found_f92_fb7 = b_e0.createAnd(after_f92, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f92_fb7);
    PabloAST * m_fb7_f93_2 = b_e0.createAdvance(found_f92_fb7, 2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb7_f93_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f93_2);
//  Cases for f9c
    PabloAST * after_f9c = b_e0.createAdvance(E0_f9c, 3);
//     f9c + fb7 => f9d
    PabloAST * found_f9c_fb7 = b_e0.createAnd(after_f9c, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f9c_fb7);
    PabloAST * m_fb7_f9d_2 = b_e0.createAdvance(found_f9c_fb7, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_f9d_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb7_f9d_2);
    xfrm_e0[5] = b_e0.createOr(xfrm_e0[5], m_fb7_f9d_2);
//  Cases for fa1
    PabloAST * after_fa1 = b_e0.createAdvance(E0_fa1, 3);
//     fa1 + fb7 => fa2
    PabloAST * found_fa1_fb7 = b_e0.createAnd(after_fa1, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fa1_fb7);
    PabloAST * m_fb7_fa2_2 = b_e0.createAdvance(found_fa1_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_fa2_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_fb7_fa2_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb7_fa2_2);
//  Cases for fa6
    PabloAST * after_fa6 = b_e0.createAdvance(E0_fa6, 3);
//     fa6 + fb7 => fa7
    PabloAST * found_fa6_fb7 = b_e0.createAnd(after_fa6, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fa6_fb7);
    PabloAST * m_fb7_fa7_2 = b_e0.createAdvance(found_fa6_fb7, 2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb7_fa7_2);
//  Cases for fab
    PabloAST * after_fab = b_e0.createAdvance(E0_fab, 3);
//     fab + fb7 => fac
    PabloAST * found_fab_fb7 = b_e0.createAnd(after_fab, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fab_fb7);
    PabloAST * m_fb7_fac_2 = b_e0.createAdvance(found_fab_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(xfrm_e0[0], m_fb7_fac_2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_fb7_fac_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_fb7_fac_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_fb7_fac_2);
//  Cases for cca
    PabloAST * after_cca = b_e0.createAdvance(b_e0.createOr(E0_cca, found_cc6_cc2), 3);
//     cca + cd5 => ccb
    PabloAST * found_cca_cd5 = b_e0.createAnd(after_cca, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cca_cd5);
    PabloAST * m_cd5_ccb_2 = b_e0.createAdvance(found_cca_cd5, 2);
    xfrm_e0[1] = b_e0.createOr(xfrm_e0[1], m_cd5_ccb_2);
    xfrm_e0[2] = b_e0.createOr(xfrm_e0[2], m_cd5_ccb_2);
    xfrm_e0[3] = b_e0.createOr(xfrm_e0[3], m_cd5_ccb_2);
    xfrm_e0[4] = b_e0.createOr(xfrm_e0[4], m_cd5_ccb_2);

    for (unsigned i = 0; i < 8; i++) {
        b_e0.createAssign(XfrmVar[i], b_e0.createOr(XfrmVar[i], xfrm_e0[i]));
    }
    b_e0.createAssign(DeletePriorVar, b_e0.createOr(DeletePriorVar, del_prior_e0));

    auto b_e1 = pb.createScope();
    PabloAST * pfx_e1_test = bnc.EQ(Basis, 0xe1);
    pb.createIf(pfx_e1_test, b_e1);
    std::vector<PabloAST *> xfrm_e1(8, All0);
    PabloAST * del_prior_e1 = All0;
    UTF::UTF_Compiler b_e1_compiler(getInputStreamVar("Basis"), b_e1, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_e1_vars(14);
    std::vector<UnicodeSet> b_e1_usets(14);
    Var * E1_1025 = b_e1.createVar("E1_1025", All0);
    b_e1_vars[0] = E1_1025;
    b_e1_usets[0] = E1_1025_uset;
    Var * E1_102e = b_e1.createVar("E1_102e", All0);
    b_e1_vars[1] = E1_102e;
    b_e1_usets[1] = E1_102e_uset;
    Var * E1_1b05 = b_e1.createVar("E1_1b05", All0);
    b_e1_vars[2] = E1_1b05;
    b_e1_usets[2] = E1_1b05_uset;
    Var * E1_1b35 = b_e1.createVar("E1_1b35", All0);
    b_e1_vars[3] = E1_1b35;
    b_e1_usets[3] = E1_1b35_uset;
    Var * E1_1b07 = b_e1.createVar("E1_1b07", All0);
    b_e1_vars[4] = E1_1b07;
    b_e1_usets[4] = E1_1b07_uset;
    Var * E1_1b09 = b_e1.createVar("E1_1b09", All0);
    b_e1_vars[5] = E1_1b09;
    b_e1_usets[5] = E1_1b09_uset;
    Var * E1_1b0b = b_e1.createVar("E1_1b0b", All0);
    b_e1_vars[6] = E1_1b0b;
    b_e1_usets[6] = E1_1b0b_uset;
    Var * E1_1b0d = b_e1.createVar("E1_1b0d", All0);
    b_e1_vars[7] = E1_1b0d;
    b_e1_usets[7] = E1_1b0d_uset;
    Var * E1_1b11 = b_e1.createVar("E1_1b11", All0);
    b_e1_vars[8] = E1_1b11;
    b_e1_usets[8] = E1_1b11_uset;
    Var * E1_1b3a = b_e1.createVar("E1_1b3a", All0);
    b_e1_vars[9] = E1_1b3a;
    b_e1_usets[9] = E1_1b3a_uset;
    Var * E1_1b3c = b_e1.createVar("E1_1b3c", All0);
    b_e1_vars[10] = E1_1b3c;
    b_e1_usets[10] = E1_1b3c_uset;
    Var * E1_1b3e = b_e1.createVar("E1_1b3e", All0);
    b_e1_vars[11] = E1_1b3e;
    b_e1_usets[11] = E1_1b3e_uset;
    Var * E1_1b3f = b_e1.createVar("E1_1b3f", All0);
    b_e1_vars[12] = E1_1b3f;
    b_e1_usets[12] = E1_1b3f_uset;
    Var * E1_1b42 = b_e1.createVar("E1_1b42", All0);
    b_e1_vars[13] = E1_1b42;
    b_e1_usets[13] = E1_1b42_uset;

    b_e1_compiler.compile(b_e1_vars, b_e1_usets);
//  Cases for 1025
    PabloAST * after_1025 = b_e1.createAdvance(E1_1025, 3);
//     1025 + 102e => 1026
    PabloAST * found_1025_102e = b_e1.createAnd(after_1025, E1_102e);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1025_102e);
    PabloAST * m_102e_1026_2 = b_e1.createAdvance(found_1025_102e, 2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_102e_1026_2);
//  Cases for 1b05
    PabloAST * after_1b05 = b_e1.createAdvance(E1_1b05, 3);
//     1b05 + 1b35 => 1b06
    PabloAST * found_1b05_1b35 = b_e1.createAnd(after_1b05, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b05_1b35);
    PabloAST * m_1b35_1b06_2 = b_e1.createAdvance(found_1b05_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b06_2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b06_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b06_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b06_2);
//  Cases for 1b07
    PabloAST * after_1b07 = b_e1.createAdvance(E1_1b07, 3);
//     1b07 + 1b35 => 1b08
    PabloAST * found_1b07_1b35 = b_e1.createAnd(after_1b07, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b07_1b35);
    PabloAST * m_1b35_1b08_2 = b_e1.createAdvance(found_1b07_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b08_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b08_2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b08_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b08_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b08_2);
//  Cases for 1b09
    PabloAST * after_1b09 = b_e1.createAdvance(E1_1b09, 3);
//     1b09 + 1b35 => 1b0a
    PabloAST * found_1b09_1b35 = b_e1.createAnd(after_1b09, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b09_1b35);
    PabloAST * m_1b35_1b0a_2 = b_e1.createAdvance(found_1b09_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b0a_2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b0a_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b0a_2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b0a_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b0a_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b0a_2);
//  Cases for 1b0b
    PabloAST * after_1b0b = b_e1.createAdvance(E1_1b0b, 3);
//     1b0b + 1b35 => 1b0c
    PabloAST * found_1b0b_1b35 = b_e1.createAnd(after_1b0b, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b0b_1b35);
    PabloAST * m_1b35_1b0c_2 = b_e1.createAdvance(found_1b0b_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b0c_2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b0c_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b0c_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b0c_2);
//  Cases for 1b0d
    PabloAST * after_1b0d = b_e1.createAdvance(E1_1b0d, 3);
//     1b0d + 1b35 => 1b0e
    PabloAST * found_1b0d_1b35 = b_e1.createAnd(after_1b0d, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b0d_1b35);
    PabloAST * m_1b35_1b0e_2 = b_e1.createAdvance(found_1b0d_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b0e_2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b0e_2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b0e_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b0e_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b0e_2);
//  Cases for 1b11
    PabloAST * after_1b11 = b_e1.createAdvance(E1_1b11, 3);
//     1b11 + 1b35 => 1b12
    PabloAST * found_1b11_1b35 = b_e1.createAnd(after_1b11, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b11_1b35);
    PabloAST * m_1b35_1b12_2 = b_e1.createAdvance(found_1b11_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b12_2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b12_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b12_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b12_2);
//  Cases for 1b3a
    PabloAST * after_1b3a = b_e1.createAdvance(E1_1b3a, 3);
//     1b3a + 1b35 => 1b3b
    PabloAST * found_1b3a_1b35 = b_e1.createAnd(after_1b3a, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3a_1b35);
    PabloAST * m_1b35_1b3b_2 = b_e1.createAdvance(found_1b3a_1b35, 2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b3b_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b3b_2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b3b_2);
//  Cases for 1b3c
    PabloAST * after_1b3c = b_e1.createAdvance(E1_1b3c, 3);
//     1b3c + 1b35 => 1b3d
    PabloAST * found_1b3c_1b35 = b_e1.createAnd(after_1b3c, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3c_1b35);
    PabloAST * m_1b35_1b3d_2 = b_e1.createAdvance(found_1b3c_1b35, 2);
    xfrm_e1[3] = b_e1.createOr(xfrm_e1[3], m_1b35_1b3d_2);
//  Cases for 1b3e
    PabloAST * after_1b3e = b_e1.createAdvance(E1_1b3e, 3);
//     1b3e + 1b35 => 1b40
    PabloAST * found_1b3e_1b35 = b_e1.createAnd(after_1b3e, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3e_1b35);
    PabloAST * m_1b35_1b40_1 = b_e1.createAdvance(found_1b3e_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b40_1);
    PabloAST * m_1b35_1b40_2 = b_e1.createAdvance(found_1b3e_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b40_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b40_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b40_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b40_2);
//  Cases for 1b3f
    PabloAST * after_1b3f = b_e1.createAdvance(E1_1b3f, 3);
//     1b3f + 1b35 => 1b41
    PabloAST * found_1b3f_1b35 = b_e1.createAnd(after_1b3f, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3f_1b35);
    PabloAST * m_1b35_1b41_1 = b_e1.createAdvance(found_1b3f_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b41_1);
    PabloAST * m_1b35_1b41_2 = b_e1.createAdvance(found_1b3f_1b35, 2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b41_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b41_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b41_2);
//  Cases for 1b42
    PabloAST * after_1b42 = b_e1.createAdvance(E1_1b42, 3);
//     1b42 + 1b35 => 1b43
    PabloAST * found_1b42_1b35 = b_e1.createAnd(after_1b42, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b42_1b35);
    PabloAST * m_1b35_1b43_1 = b_e1.createAdvance(found_1b42_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(xfrm_e1[0], m_1b35_1b43_1);
    PabloAST * m_1b35_1b43_2 = b_e1.createAdvance(found_1b42_1b35, 2);
    xfrm_e1[1] = b_e1.createOr(xfrm_e1[1], m_1b35_1b43_2);
    xfrm_e1[2] = b_e1.createOr(xfrm_e1[2], m_1b35_1b43_2);
    xfrm_e1[4] = b_e1.createOr(xfrm_e1[4], m_1b35_1b43_2);
    xfrm_e1[5] = b_e1.createOr(xfrm_e1[5], m_1b35_1b43_2);

    for (unsigned i = 0; i < 8; i++) {
        b_e1.createAssign(XfrmVar[i], b_e1.createOr(XfrmVar[i], xfrm_e1[i]));
    }
    b_e1.createAssign(DeletePriorVar, b_e1.createOr(DeletePriorVar, del_prior_e1));

    auto b_f0 = pb.createScope();
    PabloAST * pfx_f0_test = bnc.EQ(Basis, 0xf0);
    pb.createIf(pfx_f0_test, b_f0);
    std::vector<PabloAST *> xfrm_f0(8, All0);
    PabloAST * del_prior_f0 = All0;
    UTF::UTF_Compiler b_f0_compiler(getInputStreamVar("Basis"), b_f0, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> b_f0_vars(32);
    std::vector<UnicodeSet> b_f0_usets(32);
    Var * F0_11131 = b_f0.createVar("F0_11131", All0);
    b_f0_vars[0] = F0_11131;
    b_f0_usets[0] = F0_11131_uset;
    Var * F0_11127 = b_f0.createVar("F0_11127", All0);
    b_f0_vars[1] = F0_11127;
    b_f0_usets[1] = F0_11127_uset;
    Var * F0_11132 = b_f0.createVar("F0_11132", All0);
    b_f0_vars[2] = F0_11132;
    b_f0_usets[2] = F0_11132_uset;
    Var * F0_11347 = b_f0.createVar("F0_11347", All0);
    b_f0_vars[3] = F0_11347;
    b_f0_usets[3] = F0_11347_uset;
    Var * F0_1133e = b_f0.createVar("F0_1133e", All0);
    b_f0_vars[4] = F0_1133e;
    b_f0_usets[4] = F0_1133e_uset;
    Var * F0_11357 = b_f0.createVar("F0_11357", All0);
    b_f0_vars[5] = F0_11357;
    b_f0_usets[5] = F0_11357_uset;
    Var * F0_11382 = b_f0.createVar("F0_11382", All0);
    b_f0_vars[6] = F0_11382;
    b_f0_usets[6] = F0_11382_uset;
    Var * F0_113c9 = b_f0.createVar("F0_113c9", All0);
    b_f0_vars[7] = F0_113c9;
    b_f0_usets[7] = F0_113c9_uset;
    Var * F0_11384 = b_f0.createVar("F0_11384", All0);
    b_f0_vars[8] = F0_11384;
    b_f0_usets[8] = F0_11384_uset;
    Var * F0_113bb = b_f0.createVar("F0_113bb", All0);
    b_f0_vars[9] = F0_113bb;
    b_f0_usets[9] = F0_113bb_uset;
    Var * F0_1138b = b_f0.createVar("F0_1138b", All0);
    b_f0_vars[10] = F0_1138b;
    b_f0_usets[10] = F0_1138b_uset;
    Var * F0_113c2 = b_f0.createVar("F0_113c2", All0);
    b_f0_vars[11] = F0_113c2;
    b_f0_usets[11] = F0_113c2_uset;
    Var * F0_11390 = b_f0.createVar("F0_11390", All0);
    b_f0_vars[12] = F0_11390;
    b_f0_usets[12] = F0_11390_uset;
    Var * F0_113b8 = b_f0.createVar("F0_113b8", All0);
    b_f0_vars[13] = F0_113b8;
    b_f0_usets[13] = F0_113b8_uset;
    Var * F0_114b9 = b_f0.createVar("F0_114b9", All0);
    b_f0_vars[14] = F0_114b9;
    b_f0_usets[14] = F0_114b9_uset;
    Var * F0_114ba = b_f0.createVar("F0_114ba", All0);
    b_f0_vars[15] = F0_114ba;
    b_f0_usets[15] = F0_114ba_uset;
    Var * F0_114b0 = b_f0.createVar("F0_114b0", All0);
    b_f0_vars[16] = F0_114b0;
    b_f0_usets[16] = F0_114b0_uset;
    Var * F0_114bd = b_f0.createVar("F0_114bd", All0);
    b_f0_vars[17] = F0_114bd;
    b_f0_usets[17] = F0_114bd_uset;
    Var * F0_115b8 = b_f0.createVar("F0_115b8", All0);
    b_f0_vars[18] = F0_115b8;
    b_f0_usets[18] = F0_115b8_uset;
    Var * F0_115af = b_f0.createVar("F0_115af", All0);
    b_f0_vars[19] = F0_115af;
    b_f0_usets[19] = F0_115af_uset;
    Var * F0_115b9 = b_f0.createVar("F0_115b9", All0);
    b_f0_vars[20] = F0_115b9;
    b_f0_usets[20] = F0_115b9_uset;
    Var * F0_11935 = b_f0.createVar("F0_11935", All0);
    b_f0_vars[21] = F0_11935;
    b_f0_usets[21] = F0_11935_uset;
    Var * F0_11930 = b_f0.createVar("F0_11930", All0);
    b_f0_vars[22] = F0_11930;
    b_f0_usets[22] = F0_11930_uset;
    Var * F0_1611e = b_f0.createVar("F0_1611e", All0);
    b_f0_vars[23] = F0_1611e;
    b_f0_usets[23] = F0_1611e_uset;
    Var * F0_16129 = b_f0.createVar("F0_16129", All0);
    b_f0_vars[24] = F0_16129;
    b_f0_usets[24] = F0_16129_uset;
    Var * F0_1611f = b_f0.createVar("F0_1611f", All0);
    b_f0_vars[25] = F0_1611f;
    b_f0_usets[25] = F0_1611f_uset;
    Var * F0_16120 = b_f0.createVar("F0_16120", All0);
    b_f0_vars[26] = F0_16120;
    b_f0_usets[26] = F0_16120_uset;
    Var * F0_16121 = b_f0.createVar("F0_16121", All0);
    b_f0_vars[27] = F0_16121;
    b_f0_usets[27] = F0_16121_uset;
    Var * F0_16122 = b_f0.createVar("F0_16122", All0);
    b_f0_vars[28] = F0_16122;
    b_f0_usets[28] = F0_16122_uset;
    Var * F0_16d63 = b_f0.createVar("F0_16d63", All0);
    b_f0_vars[29] = F0_16d63;
    b_f0_usets[29] = F0_16d63_uset;
    Var * F0_16d67 = b_f0.createVar("F0_16d67", All0);
    b_f0_vars[30] = F0_16d67;
    b_f0_usets[30] = F0_16d67_uset;
    Var * F0_16d69 = b_f0.createVar("F0_16d69", All0);
    b_f0_vars[31] = F0_16d69;
    b_f0_usets[31] = F0_16d69_uset;

    b_f0_compiler.compile(b_f0_vars, b_f0_usets);
//  Cases for 11131
    PabloAST * after_11131 = b_f0.createAdvance(F0_11131, 4);
//     11131 + 11127 => 1112e
    PabloAST * found_11131_11127 = b_f0.createAnd(after_11131, F0_11127);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11131_11127);
    PabloAST * m_11127_1112e_3 = b_f0.createAdvance(found_11131_11127, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_11127_1112e_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_11127_1112e_3);
//  Cases for 11132
    PabloAST * after_11132 = b_f0.createAdvance(F0_11132, 4);
//     11132 + 11127 => 1112f
    PabloAST * found_11132_11127 = b_f0.createAnd(after_11132, F0_11127);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11132_11127);
    PabloAST * m_11127_1112f_3 = b_f0.createAdvance(found_11132_11127, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_11127_1112f_3);
//  Cases for 11347
    PabloAST * after_11347 = b_f0.createAdvance(F0_11347, 4);
//     11347 + 1133e => 1134b
    PabloAST * found_11347_1133e = b_f0.createAnd(after_11347, F0_1133e);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11347_1133e);
    PabloAST * m_1133e_1134b_2 = b_f0.createAdvance(found_11347_1133e, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_1133e_1134b_2);
    PabloAST * m_1133e_1134b_3 = b_f0.createAdvance(found_11347_1133e, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_1133e_1134b_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_1133e_1134b_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1133e_1134b_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1133e_1134b_3);
//     11347 + 11357 => 1134c
    PabloAST * found_11347_11357 = b_f0.createAnd(after_11347, F0_11357);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11347_11357);
    PabloAST * m_11357_1134c_3 = b_f0.createAdvance(found_11347_11357, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_11357_1134c_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_11357_1134c_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_11357_1134c_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_11357_1134c_3);
//  Cases for 11382
    PabloAST * after_11382 = b_f0.createAdvance(F0_11382, 4);
//     11382 + 113c9 => 11383
    PabloAST * found_11382_113c9 = b_f0.createAnd(after_11382, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11382_113c9);
    PabloAST * m_113c9_11383_2 = b_f0.createAdvance(found_11382_113c9, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113c9_11383_2);
    PabloAST * m_113c9_11383_3 = b_f0.createAdvance(found_11382_113c9, 3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_113c9_11383_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_113c9_11383_3);
//  Cases for 11384
    PabloAST * after_11384 = b_f0.createAdvance(F0_11384, 4);
//     11384 + 113bb => 11385
    PabloAST * found_11384_113bb = b_f0.createAnd(after_11384, F0_113bb);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11384_113bb);
    PabloAST * m_113bb_11385_3 = b_f0.createAdvance(found_11384_113bb, 3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_113bb_11385_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_113bb_11385_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_113bb_11385_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_113bb_11385_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_113bb_11385_3);
//  Cases for 1138b
    PabloAST * after_1138b = b_f0.createAdvance(F0_1138b, 4);
//     1138b + 113c2 => 1138e
    PabloAST * found_1138b_113c2 = b_f0.createAnd(after_1138b, F0_113c2);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1138b_113c2);
    PabloAST * m_113c2_1138e_2 = b_f0.createAdvance(found_1138b_113c2, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113c2_1138e_2);
    PabloAST * m_113c2_1138e_3 = b_f0.createAdvance(found_1138b_113c2, 3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_113c2_1138e_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_113c2_1138e_3);
//  Cases for 11390
    PabloAST * after_11390 = b_f0.createAdvance(F0_11390, 4);
//     11390 + 113c9 => 11391
    PabloAST * found_11390_113c9 = b_f0.createAnd(after_11390, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11390_113c9);
    PabloAST * m_113c9_11391_2 = b_f0.createAdvance(found_11390_113c9, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113c9_11391_2);
    PabloAST * m_113c9_11391_3 = b_f0.createAdvance(found_11390_113c9, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_113c9_11391_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_113c9_11391_3);
//  Cases for 113c2
    PabloAST * after_113c2 = b_f0.createAdvance(F0_113c2, 4);
//     113c2 + 113c2 => 113c5
    PabloAST * found_113c2_113c2 = b_f0.createAnd(after_113c2, F0_113c2);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113c2);
    PabloAST * m_113c2_113c5_3 = b_f0.createAdvance(found_113c2_113c2, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113c2_113c5_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_113c2_113c5_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_113c2_113c5_3);
//     113c2 + 113b8 => 113c7
    PabloAST * found_113c2_113b8 = b_f0.createAnd(after_113c2, F0_113b8);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113b8);
    PabloAST * m_113b8_113c7_2 = b_f0.createAdvance(found_113c2_113b8, 2);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113b8_113c7_2);
    PabloAST * m_113b8_113c7_3 = b_f0.createAdvance(found_113c2_113b8, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113b8_113c7_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_113b8_113c7_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_113b8_113c7_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_113b8_113c7_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_113b8_113c7_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_113b8_113c7_3);
//     113c2 + 113c9 => 113c8
    PabloAST * found_113c2_113c9 = b_f0.createAnd(after_113c2, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113c9);
    PabloAST * m_113c9_113c8_3 = b_f0.createAdvance(found_113c2_113c9, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_113c9_113c8_3);
//  Cases for 114b9
    PabloAST * after_114b9 = b_f0.createAdvance(F0_114b9, 4);
//     114b9 + 114ba => 114bb
    PabloAST * found_114b9_114ba = b_f0.createAnd(after_114b9, F0_114ba);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114ba);
    PabloAST * m_114ba_114bb_3 = b_f0.createAdvance(found_114b9_114ba, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_114ba_114bb_3);
//     114b9 + 114b0 => 114bc
    PabloAST * found_114b9_114b0 = b_f0.createAnd(after_114b9, F0_114b0);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114b0);
    PabloAST * m_114b0_114bc_3 = b_f0.createAdvance(found_114b9_114b0, 3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_114b0_114bc_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_114b0_114bc_3);
//     114b9 + 114bd => 114be
    PabloAST * found_114b9_114bd = b_f0.createAnd(after_114b9, F0_114bd);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114bd);
    PabloAST * m_114bd_114be_3 = b_f0.createAdvance(found_114b9_114bd, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_114bd_114be_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_114bd_114be_3);
//  Cases for 115b8
    PabloAST * after_115b8 = b_f0.createAdvance(F0_115b8, 4);
//     115b8 + 115af => 115ba
    PabloAST * found_115b8_115af = b_f0.createAnd(after_115b8, F0_115af);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_115b8_115af);
    PabloAST * m_115af_115ba_3 = b_f0.createAdvance(found_115b8_115af, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_115af_115ba_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_115af_115ba_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_115af_115ba_3);
//  Cases for 115b9
    PabloAST * after_115b9 = b_f0.createAdvance(F0_115b9, 4);
//     115b9 + 115af => 115bb
    PabloAST * found_115b9_115af = b_f0.createAnd(after_115b9, F0_115af);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_115b9_115af);
    PabloAST * m_115af_115bb_3 = b_f0.createAdvance(found_115b9_115af, 3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_115af_115bb_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_115af_115bb_3);
//  Cases for 11935
    PabloAST * after_11935 = b_f0.createAdvance(F0_11935, 4);
//     11935 + 11930 => 11938
    PabloAST * found_11935_11930 = b_f0.createAnd(after_11935, F0_11930);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11935_11930);
    PabloAST * m_11930_11938_3 = b_f0.createAdvance(found_11935_11930, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_11930_11938_3);
//  Cases for 1611e
    PabloAST * after_1611e = b_f0.createAdvance(F0_1611e, 4);
//     1611e + 1611e => 16121
    PabloAST * found_1611e_1611e = b_f0.createAnd(after_1611e, F0_1611e);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_1611e);
    PabloAST * m_1611e_16121_3 = b_f0.createAdvance(found_1611e_1611e, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_1611e_16121_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_1611e_16121_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_1611e_16121_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_1611e_16121_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1611e_16121_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1611e_16121_3);
//     1611e + 16129 => 16122
    PabloAST * found_1611e_16129 = b_f0.createAnd(after_1611e, F0_16129);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_16129);
    PabloAST * m_16129_16122_3 = b_f0.createAdvance(found_1611e_16129, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_16129_16122_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_16129_16122_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_16129_16122_3);
//     1611e + 1611f => 16123
    PabloAST * found_1611e_1611f = b_f0.createAnd(after_1611e, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_1611f);
    PabloAST * m_1611f_16123_3 = b_f0.createAdvance(found_1611e_1611f, 3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_1611f_16123_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_1611f_16123_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1611f_16123_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1611f_16123_3);
//     1611e + 16120 => 16125
    PabloAST * found_1611e_16120 = b_f0.createAnd(after_1611e, F0_16120);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_16120);
    PabloAST * m_16120_16125_3 = b_f0.createAdvance(found_1611e_16120, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_16120_16125_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_16120_16125_3);
//  Cases for 16129
    PabloAST * after_16129 = b_f0.createAdvance(F0_16129, 4);
//     16129 + 1611f => 16124
    PabloAST * found_16129_1611f = b_f0.createAnd(after_16129, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16129_1611f);
    PabloAST * m_1611f_16124_3 = b_f0.createAdvance(found_16129_1611f, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_1611f_16124_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_1611f_16124_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_1611f_16124_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1611f_16124_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1611f_16124_3);
//  Cases for 16d63
    PabloAST * after_16d63 = b_f0.createAdvance(F0_16d63, 4);
//     16d63 + 16d67 => 16d69
    PabloAST * found_16d63_16d67 = b_f0.createAnd(after_16d63, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d63_16d67);
    PabloAST * m_16d67_16d69_3 = b_f0.createAdvance(found_16d63_16d67, 3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_16d67_16d69_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_16d67_16d69_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_16d67_16d69_3);
//  Cases for 16d67
    PabloAST * after_16d67 = b_f0.createAdvance(F0_16d67, 4);
//     16d67 + 16d67 => 16d68
    PabloAST * found_16d67_16d67 = b_f0.createAnd(after_16d67, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d67_16d67);
    PabloAST * m_16d67_16d68_3 = b_f0.createAdvance(found_16d67_16d67, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_16d67_16d68_3);
    xfrm_f0[1] = b_f0.createOr(xfrm_f0[1], m_16d67_16d68_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_16d67_16d68_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_16d67_16d68_3);
//  Cases for 16121
    PabloAST * after_16121 = b_f0.createAdvance(b_f0.createOr(F0_16121, found_1611e_1611e), 4);
//     16121 + 1611f => 16126
    PabloAST * found_16121_1611f = b_f0.createAnd(after_16121, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16121_1611f);
    PabloAST * m_1611f_16126_3 = b_f0.createAdvance(found_16121_1611f, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_1611f_16126_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_1611f_16126_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1611f_16126_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1611f_16126_3);
//     16121 + 16120 => 16128
    PabloAST * found_16121_16120 = b_f0.createAnd(after_16121, F0_16120);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16121_16120);
    PabloAST * m_16120_16128_3 = b_f0.createAdvance(found_16121_16120, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_16120_16128_3);
//  Cases for 16122
    PabloAST * after_16122 = b_f0.createAdvance(b_f0.createOr(F0_16122, found_1611e_16129), 4);
//     16122 + 1611f => 16127
    PabloAST * found_16122_1611f = b_f0.createAnd(after_16122, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16122_1611f);
    PabloAST * m_1611f_16127_3 = b_f0.createAdvance(found_16122_1611f, 3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_1611f_16127_3);
    xfrm_f0[4] = b_f0.createOr(xfrm_f0[4], m_1611f_16127_3);
    xfrm_f0[5] = b_f0.createOr(xfrm_f0[5], m_1611f_16127_3);
//  Cases for 16d69
    PabloAST * after_16d69 = b_f0.createAdvance(b_f0.createOr(F0_16d69, found_16d63_16d67), 4);
//     16d69 + 16d67 => 16d6a
    PabloAST * found_16d69_16d67 = b_f0.createAnd(after_16d69, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d69_16d67);
    PabloAST * m_16d67_16d6a_3 = b_f0.createAdvance(found_16d69_16d67, 3);
    xfrm_f0[0] = b_f0.createOr(xfrm_f0[0], m_16d67_16d6a_3);
    xfrm_f0[2] = b_f0.createOr(xfrm_f0[2], m_16d67_16d6a_3);
    xfrm_f0[3] = b_f0.createOr(xfrm_f0[3], m_16d67_16d6a_3);

    for (unsigned i = 0; i < 8; i++) {
        b_f0.createAssign(XfrmVar[i], b_f0.createOr(XfrmVar[i], xfrm_f0[i]));
    }
    b_f0.createAssign(DeletePriorVar, b_f0.createOr(DeletePriorVar, del_prior_f0));

    Var * DeleteOutputVar = getOutputStreamVar("DeletePrior");
    pb.createAssign(pb.createExtract(DeleteOutputVar, pb.getInteger(0)), DeletePriorVar);
    Var * XfrmOutputVar = getOutputStreamVar("XfrmBasis");
    for (unsigned i = 0; i < 8; i++) {
        Var * xfrm_out = pb.createExtract(XfrmOutputVar, pb.getInteger(i));
        pb.createAssign(xfrm_out, XfrmVar[i]);
    }
}


class InsertionsForNonStarterDecompositions : public PabloKernel {
public:
    InsertionsForNonStarterDecompositions
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * InsertionBixNum);
protected:
    void generatePabloMethod() override;
};

InsertionsForNonStarterDecompositions::InsertionsForNonStarterDecompositions
    (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                   StreamSet * InsertionBixNum)
: PabloKernel(ts, "InsertionsForNonStarterDecompositions",
{Binding{"Basis", Basis}},
{Binding{"InsertionBixNum", InsertionBixNum}}) {}

void InsertionsForNonStarterDecompositions::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * All0 = pb.createZeroes();
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> insertions(2, All0);
    UTF::UTF_Compiler pb_compiler(getInputStreamVar("Basis"), pb, pablo::BitMovementMode::LookAhead);
    std::vector<Var *> pb_vars(2);
    std::vector<UnicodeSet> pb_usets(2);
    Var * E0_f73_5_81 = pb.createVar("E0_f73_5_81", All0);
    pb_vars[0] = E0_f73_5_81;
    pb_usets[0] = E0_f73_5_81_uset;
    Var * CD_344_f73_5_81 = pb.createVar("CD_344_f73_5_81", All0);
    pb_vars[1] = CD_344_f73_5_81;
    pb_usets[1] = CD_344_f73_5_81_uset;

    pb_compiler.compile(pb_vars, pb_usets);
    insertions[0] = E0_f73_5_81;
    insertions[1] = CD_344_f73_5_81;
    writeOutputStreamSet("InsertionBixNum", insertions);
}

typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_pipeline(CPUDriver & driver) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.

    auto P = CreatePipeline(driver, Input<uint32_t>("inputFileDecriptor"));

    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");
    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    //  ReadSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    SHOW_BYTES(ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * InsertionBixNum = P.CreateStreamSet(2, 1);
    P.CreateKernelCall<U8_InsertionBixNum>(BasisBits, InsertionBixNum);
    SHOW_BIXNUM(InsertionBixNum);

    StreamSet * SpreadMask = InsertionSpreadMask(P, InsertionBixNum, kernel::InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(8, 1);
    SpreadByMask(P, SpreadMask, BasisBits, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    StreamSet * NSD_Basis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<NonStarterDecomposition>(ExpandedBasis, NSD_Basis);
    SHOW_BIXNUM(NSD_Basis);

    StreamSet * Canon_Basis = P.CreateStreamSet(8, 1);
    StreamSet * CanonSelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<SingletonCanonicalization>(NSD_Basis, CanonSelectionMask, Canon_Basis);
    SHOW_BIXNUM(Canon_Basis);
    SHOW_STREAM(CanonSelectionMask);

    StreamSet * XfrmBasis = P.CreateStreamSet(8, 1);
    StreamSet * DelPrior = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<ShortComposableTranslation>(Canon_Basis, DelPrior, XfrmBasis);
    SHOW_BIXNUM(XfrmBasis);
    SHOW_STREAM(DelPrior);

    StreamSet * XfrmedBasis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<ApplyTransform>(Canon_Basis, XfrmBasis, XfrmedBasis);
    SHOW_BIXNUM(XfrmedBasis);

    StreamSet * SelectionMask0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<DelPriorToSelectMask>(ExpandedBasis, DelPrior, SelectionMask0);
    SHOW_STREAM(SelectionMask0);

    StreamSet * FilteredBasis = P.CreateStreamSet(8, 1);
    FilterByMask(P, SelectionMask0, XfrmedBasis, FilteredBasis);
    SHOW_BIXNUM(FilteredBasis);

    StreamSet * const OutputBasis = P.CreateStreamSet(8);

    if (U21) {
        //  The u8index stream marks the final byte of each UTF-8 character sequence.
        StreamSet * u8index = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<UTF8_index>(FilteredBasis, u8index);
        SHOW_STREAM(u8index);

        //  To make Unicode calculations simpler, we will construct a stream set of
        //  all 21 Unicode bits.   As a first step we calculate these bits at
        //  the u8index positions.
        StreamSet * U21_u8indexed = P.CreateStreamSet(21, 1);
        P.CreateKernelCall<UTF8_Decoder>(FilteredBasis, U21_u8indexed);

        //  Now we construct a compressed stream set which is one-to-one with
        //  Unicode characters, by filtering out non u8index positions.
        StreamSet * U21 = P.CreateStreamSet(21, 1);
        FilterByMask(P, u8index, U21_u8indexed, U21);
        SHOW_BIXNUM(U21);

        //  Hangul composable characters are of three types, L, V, LV and T.
        //  Compute a set of 4 parallel bit streams, one for each character class.
        auto L_V_T_sets = Hangul_Composables();
        StreamSet * L_V_T =  P.CreateStreamSet(L_V_T_sets.size());
        P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, U21, L_V_T);
        SHOW_BIXNUM(L_V_T);

        StreamSet * TranslatedBasis = P.CreateStreamSet(21, 1);
        StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<Hangul_Composition>(U21, L_V_T, TranslatedBasis, SelectionMask);
        SHOW_BIXNUM(TranslatedBasis);
        SHOW_STREAM(SelectionMask);

        StreamSet * NFC_Basis = P.CreateStreamSet(21, 1);
        FilterByMask(P, SelectionMask, TranslatedBasis, NFC_Basis);
        SHOW_BIXNUM(NFC_Basis);

        // Given the 21-bit basis representation, we can now transform back
        // to a UTF-8 representation.
        U21_to_UTF8(P, NFC_Basis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);
    } else {
        auto L_V_T_sets = Hangul_Composables();
        StreamSet * L_V_T =  P.CreateStreamSet(L_V_T_sets.size());
        P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, FilteredBasis, L_V_T, pablo::BitMovementMode::LookAhead);
        SHOW_BIXNUM(L_V_T);

        StreamSet * TranslatedBasis = P.CreateStreamSet(8, 1);
        StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<Hangul_Composition>(FilteredBasis, L_V_T, TranslatedBasis, SelectionMask);
        SHOW_BIXNUM(TranslatedBasis);
        SHOW_STREAM(SelectionMask);

        FilterByMask(P, SelectionMask, TranslatedBasis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);
    }
    //  The P2SKernel transforms the basis bit streams into the corresponding
    //  byte stream, inverting the S2P process.
    StreamSet * OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);

    // The program output is generated by writing the output bytes to stdout.
    P.CreateKernelCall<StdOutKernel>(OutputBytes);

    return P.compile();
}


int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    CPUDriver driver("NFD_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    XfrmFunctionType fn;
    fn = generate_pipeline(driver);
    //
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
    } else {
        fn(fd);
        close(fd);
    }
    return 0;
}
