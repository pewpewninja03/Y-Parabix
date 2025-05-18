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
    PabloAST * mrkr_9beto_9cb = b_e0.createAdvance(found_9c7_9be, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_9beto_9cb, xfrm_e0[0]);
    mrkr_9beto_9cb = b_e0.createAdvance(mrkr_9beto_9cb, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_9beto_9cb, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_9beto_9cb, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_9beto_9cb, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_9beto_9cb, xfrm_e0[5]);
//     9c7 + 9d7 => 9cc
    PabloAST * found_9c7_9d7 = b_e0.createAnd(after_9c7, E0_9d7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_9c7_9d7);
    PabloAST * mrkr_9d7to_9cc = b_e0.createAdvance(found_9c7_9d7, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_9d7to_9cc, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_9d7to_9cc, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_9d7to_9cc, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_9d7to_9cc, xfrm_e0[4]);
//  Cases for b47
    PabloAST * after_b47 = b_e0.createAdvance(E0_b47, 3);
//     b47 + b56 => b48
    PabloAST * found_b47_b56 = b_e0.createAnd(after_b47, E0_b56);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b56);
    PabloAST * mrkr_b56to_b48 = b_e0.createAdvance(found_b47_b56, 2);
    xfrm_e0[1] = b_e0.createOr(mrkr_b56to_b48, xfrm_e0[1]);
    xfrm_e0[2] = b_e0.createOr(mrkr_b56to_b48, xfrm_e0[2]);
    xfrm_e0[3] = b_e0.createOr(mrkr_b56to_b48, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_b56to_b48, xfrm_e0[4]);
//     b47 + b3e => b4b
    PabloAST * found_b47_b3e = b_e0.createAnd(after_b47, E0_b3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b3e);
    PabloAST * mrkr_b3eto_b4b = b_e0.createAdvance(found_b47_b3e, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_b3eto_b4b, xfrm_e0[0]);
    mrkr_b3eto_b4b = b_e0.createAdvance(mrkr_b3eto_b4b, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_b3eto_b4b, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_b3eto_b4b, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_b3eto_b4b, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_b3eto_b4b, xfrm_e0[5]);
//     b47 + b57 => b4c
    PabloAST * found_b47_b57 = b_e0.createAnd(after_b47, E0_b57);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b47_b57);
    PabloAST * mrkr_b57to_b4c = b_e0.createAdvance(found_b47_b57, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_b57to_b4c, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_b57to_b4c, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_b57to_b4c, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_b57to_b4c, xfrm_e0[4]);
//  Cases for b92
    PabloAST * after_b92 = b_e0.createAdvance(E0_b92, 3);
//     b92 + bd7 => b94
    PabloAST * found_b92_bd7 = b_e0.createAnd(after_b92, E0_bd7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_b92_bd7);
    PabloAST * mrkr_bd7to_b94 = b_e0.createAdvance(found_b92_bd7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_bd7to_b94, xfrm_e0[0]);
    mrkr_bd7to_b94 = b_e0.createAdvance(mrkr_bd7to_b94, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_bd7to_b94, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_bd7to_b94, xfrm_e0[1]);
//  Cases for bc6
    PabloAST * after_bc6 = b_e0.createAdvance(E0_bc6, 3);
//     bc6 + bbe => bca
    PabloAST * found_bc6_bbe = b_e0.createAnd(after_bc6, E0_bbe);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc6_bbe);
    PabloAST * mrkr_bbeto_bca = b_e0.createAdvance(found_bc6_bbe, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_bbeto_bca, xfrm_e0[0]);
    mrkr_bbeto_bca = b_e0.createAdvance(mrkr_bbeto_bca, 1);
    xfrm_e0[2] = b_e0.createOr(mrkr_bbeto_bca, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_bbeto_bca, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_bbeto_bca, xfrm_e0[5]);
//     bc6 + bd7 => bcc
    PabloAST * found_bc6_bd7 = b_e0.createAnd(after_bc6, E0_bd7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc6_bd7);
    PabloAST * mrkr_bd7to_bcc = b_e0.createAdvance(found_bc6_bd7, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_bd7to_bcc, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_bd7to_bcc, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_bd7to_bcc, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_bd7to_bcc, xfrm_e0[4]);
//  Cases for bc7
    PabloAST * after_bc7 = b_e0.createAdvance(E0_bc7, 3);
//     bc7 + bbe => bcb
    PabloAST * found_bc7_bbe = b_e0.createAnd(after_bc7, E0_bbe);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_bc7_bbe);
    PabloAST * mrkr_bbeto_bcb = b_e0.createAdvance(found_bc7_bbe, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_bbeto_bcb, xfrm_e0[0]);
    mrkr_bbeto_bcb = b_e0.createAdvance(mrkr_bbeto_bcb, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_bbeto_bcb, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_bbeto_bcb, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_bbeto_bcb, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_bbeto_bcb, xfrm_e0[5]);
//  Cases for cbf
    PabloAST * after_cbf = b_e0.createAdvance(E0_cbf, 3);
//     cbf + cd5 => cc0
    PabloAST * found_cbf_cd5 = b_e0.createAnd(after_cbf, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cbf_cd5);
    PabloAST * mrkr_cd5to_cc0 = b_e0.createAdvance(found_cbf_cd5, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_cd5to_cc0, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_cd5to_cc0, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_cd5to_cc0, xfrm_e0[4]);
//  Cases for cc6
    PabloAST * after_cc6 = b_e0.createAdvance(E0_cc6, 3);
//     cc6 + cd5 => cc7
    PabloAST * found_cc6_cd5 = b_e0.createAnd(after_cc6, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cd5);
    PabloAST * mrkr_cd5to_cc7 = b_e0.createAdvance(found_cc6_cd5, 2);
    xfrm_e0[1] = b_e0.createOr(mrkr_cd5to_cc7, xfrm_e0[1]);
    xfrm_e0[4] = b_e0.createOr(mrkr_cd5to_cc7, xfrm_e0[4]);
//     cc6 + cd6 => cc8
    PabloAST * found_cc6_cd6 = b_e0.createAnd(after_cc6, E0_cd6);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cd6);
    PabloAST * mrkr_cd6to_cc8 = b_e0.createAdvance(found_cc6_cd6, 2);
    xfrm_e0[1] = b_e0.createOr(mrkr_cd6to_cc8, xfrm_e0[1]);
    xfrm_e0[2] = b_e0.createOr(mrkr_cd6to_cc8, xfrm_e0[2]);
    xfrm_e0[3] = b_e0.createOr(mrkr_cd6to_cc8, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_cd6to_cc8, xfrm_e0[4]);
//     cc6 + cc2 => cca
    PabloAST * found_cc6_cc2 = b_e0.createAnd(after_cc6, E0_cc2);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cc6_cc2);
    PabloAST * mrkr_cc2to_cca = b_e0.createAdvance(found_cc6_cc2, 2);
    xfrm_e0[3] = b_e0.createOr(mrkr_cc2to_cca, xfrm_e0[3]);
//  Cases for d46
    PabloAST * after_d46 = b_e0.createAdvance(E0_d46, 3);
//     d46 + d3e => d4a
    PabloAST * found_d46_d3e = b_e0.createAnd(after_d46, E0_d3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d46_d3e);
    PabloAST * mrkr_d3eto_d4a = b_e0.createAdvance(found_d46_d3e, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_d3eto_d4a, xfrm_e0[0]);
    mrkr_d3eto_d4a = b_e0.createAdvance(mrkr_d3eto_d4a, 1);
    xfrm_e0[2] = b_e0.createOr(mrkr_d3eto_d4a, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_d3eto_d4a, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_d3eto_d4a, xfrm_e0[5]);
//     d46 + d57 => d4c
    PabloAST * found_d46_d57 = b_e0.createAnd(after_d46, E0_d57);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d46_d57);
    PabloAST * mrkr_d57to_d4c = b_e0.createAdvance(found_d46_d57, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_d57to_d4c, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_d57to_d4c, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_d57to_d4c, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_d57to_d4c, xfrm_e0[4]);
//  Cases for d47
    PabloAST * after_d47 = b_e0.createAdvance(E0_d47, 3);
//     d47 + d3e => d4b
    PabloAST * found_d47_d3e = b_e0.createAnd(after_d47, E0_d3e);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_d47_d3e);
    PabloAST * mrkr_d3eto_d4b = b_e0.createAdvance(found_d47_d3e, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_d3eto_d4b, xfrm_e0[0]);
    mrkr_d3eto_d4b = b_e0.createAdvance(mrkr_d3eto_d4b, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_d3eto_d4b, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_d3eto_d4b, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_d3eto_d4b, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_d3eto_d4b, xfrm_e0[5]);
//  Cases for dd9
    PabloAST * after_dd9 = b_e0.createAdvance(E0_dd9, 3);
//     dd9 + dcf => ddc
    PabloAST * found_dd9_dcf = b_e0.createAnd(after_dd9, E0_dcf);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_dd9_dcf);
    PabloAST * mrkr_dcfto_ddc = b_e0.createAdvance(found_dd9_dcf, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_dcfto_ddc, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_dcfto_ddc, xfrm_e0[1]);
    xfrm_e0[4] = b_e0.createOr(mrkr_dcfto_ddc, xfrm_e0[4]);
//     dd9 + ddf => dde
    PabloAST * found_dd9_ddf = b_e0.createAnd(after_dd9, E0_ddf);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_dd9_ddf);
    PabloAST * mrkr_ddfto_dde = b_e0.createAdvance(found_dd9_ddf, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_ddfto_dde, xfrm_e0[0]);
//  Cases for f40
    PabloAST * after_f40 = b_e0.createAdvance(E0_f40, 3);
//     f40 + fb5 => f69
    PabloAST * found_f40_fb5 = b_e0.createAnd(after_f40, E0_fb5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f40_fb5);
    PabloAST * mrkr_fb5to_f69 = b_e0.createAdvance(found_f40_fb5, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb5to_f69, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb5to_f69, xfrm_e0[1]);
    mrkr_fb5to_f69 = b_e0.createAdvance(mrkr_fb5to_f69, 1);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb5to_f69, xfrm_e0[2]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb5to_f69, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb5to_f69, xfrm_e0[4]);
//  Cases for f42
    PabloAST * after_f42 = b_e0.createAdvance(E0_f42, 3);
//     f42 + fb7 => f43
    PabloAST * found_f42_fb7 = b_e0.createAnd(after_f42, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f42_fb7);
    PabloAST * mrkr_fb7to_f43 = b_e0.createAdvance(found_f42_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f43, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f43, xfrm_e0[1]);
    mrkr_fb7to_f43 = b_e0.createAdvance(mrkr_fb7to_f43, 1);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb7to_f43, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb7to_f43, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f43, xfrm_e0[5]);
//  Cases for f4c
    PabloAST * after_f4c = b_e0.createAdvance(E0_f4c, 3);
//     f4c + fb7 => f4d
    PabloAST * found_f4c_fb7 = b_e0.createAnd(after_f4c, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f4c_fb7);
    PabloAST * mrkr_fb7to_f4d = b_e0.createAdvance(found_f4c_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[1]);
    mrkr_fb7to_f4d = b_e0.createAdvance(mrkr_fb7to_f4d, 1);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[4]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f4d, xfrm_e0[5]);
//  Cases for f51
    PabloAST * after_f51 = b_e0.createAdvance(E0_f51, 3);
//     f51 + fb7 => f52
    PabloAST * found_f51_fb7 = b_e0.createAnd(after_f51, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f51_fb7);
    PabloAST * mrkr_fb7to_f52 = b_e0.createAdvance(found_f51_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f52, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f52, xfrm_e0[1]);
    mrkr_fb7to_f52 = b_e0.createAdvance(mrkr_fb7to_f52, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f52, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb7to_f52, xfrm_e0[2]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f52, xfrm_e0[5]);
//  Cases for f56
    PabloAST * after_f56 = b_e0.createAdvance(E0_f56, 3);
//     f56 + fb7 => f57
    PabloAST * found_f56_fb7 = b_e0.createAnd(after_f56, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f56_fb7);
    PabloAST * mrkr_fb7to_f57 = b_e0.createAdvance(found_f56_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f57, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f57, xfrm_e0[1]);
    mrkr_fb7to_f57 = b_e0.createAdvance(mrkr_fb7to_f57, 1);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f57, xfrm_e0[5]);
//  Cases for f5b
    PabloAST * after_f5b = b_e0.createAdvance(E0_f5b, 3);
//     f5b + fb7 => f5c
    PabloAST * found_f5b_fb7 = b_e0.createAnd(after_f5b, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f5b_fb7);
    PabloAST * mrkr_fb7to_f5c = b_e0.createAdvance(found_f5b_fb7, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[1]);
    mrkr_fb7to_f5c = b_e0.createAdvance(mrkr_fb7to_f5c, 1);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[3]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f5c, xfrm_e0[5]);
//  Cases for f90
    PabloAST * after_f90 = b_e0.createAdvance(E0_f90, 3);
//     f90 + fb5 => fb9
    PabloAST * found_f90_fb5 = b_e0.createAnd(after_f90, E0_fb5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f90_fb5);
    PabloAST * mrkr_fb5to_fb9 = b_e0.createAdvance(found_f90_fb5, 2);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb5to_fb9, xfrm_e0[2]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb5to_fb9, xfrm_e0[3]);
//  Cases for f92
    PabloAST * after_f92 = b_e0.createAdvance(E0_f92, 3);
//     f92 + fb7 => f93
    PabloAST * found_f92_fb7 = b_e0.createAnd(after_f92, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f92_fb7);
    PabloAST * mrkr_fb7to_f93 = b_e0.createAdvance(found_f92_fb7, 2);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb7to_f93, xfrm_e0[2]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f93, xfrm_e0[5]);
//  Cases for f9c
    PabloAST * after_f9c = b_e0.createAdvance(E0_f9c, 3);
//     f9c + fb7 => f9d
    PabloAST * found_f9c_fb7 = b_e0.createAnd(after_f9c, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_f9c_fb7);
    PabloAST * mrkr_fb7to_f9d = b_e0.createAdvance(found_f9c_fb7, 2);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_f9d, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb7to_f9d, xfrm_e0[3]);
    xfrm_e0[5] = b_e0.createOr(mrkr_fb7to_f9d, xfrm_e0[5]);
//  Cases for fa1
    PabloAST * after_fa1 = b_e0.createAdvance(E0_fa1, 3);
//     fa1 + fb7 => fa2
    PabloAST * found_fa1_fb7 = b_e0.createAnd(after_fa1, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fa1_fb7);
    PabloAST * mrkr_fb7to_fa2 = b_e0.createAdvance(found_fa1_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_fa2, xfrm_e0[0]);
    xfrm_e0[2] = b_e0.createOr(mrkr_fb7to_fa2, xfrm_e0[2]);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb7to_fa2, xfrm_e0[4]);
//  Cases for fa6
    PabloAST * after_fa6 = b_e0.createAdvance(E0_fa6, 3);
//     fa6 + fb7 => fa7
    PabloAST * found_fa6_fb7 = b_e0.createAnd(after_fa6, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fa6_fb7);
    PabloAST * mrkr_fb7to_fa7 = b_e0.createAdvance(found_fa6_fb7, 2);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb7to_fa7, xfrm_e0[4]);
//  Cases for fab
    PabloAST * after_fab = b_e0.createAdvance(E0_fab, 3);
//     fab + fb7 => fac
    PabloAST * found_fab_fb7 = b_e0.createAnd(after_fab, E0_fb7);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_fab_fb7);
    PabloAST * mrkr_fb7to_fac = b_e0.createAdvance(found_fab_fb7, 2);
    xfrm_e0[0] = b_e0.createOr(mrkr_fb7to_fac, xfrm_e0[0]);
    xfrm_e0[1] = b_e0.createOr(mrkr_fb7to_fac, xfrm_e0[1]);
    xfrm_e0[3] = b_e0.createOr(mrkr_fb7to_fac, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_fb7to_fac, xfrm_e0[4]);
//  Cases for cca
    PabloAST * after_cca = b_e0.createAdvance(b_e0.createOr(E0_cca, found_cc6_cc2), 3);
//     cca + cd5 => ccb
    PabloAST * found_cca_cd5 = b_e0.createAnd(after_cca, E0_cd5);
    del_prior_e0 = b_e0.createOr(del_prior_e0, found_cca_cd5);
    PabloAST * mrkr_cd5to_ccb = b_e0.createAdvance(found_cca_cd5, 2);
    xfrm_e0[1] = b_e0.createOr(mrkr_cd5to_ccb, xfrm_e0[1]);
    xfrm_e0[2] = b_e0.createOr(mrkr_cd5to_ccb, xfrm_e0[2]);
    xfrm_e0[3] = b_e0.createOr(mrkr_cd5to_ccb, xfrm_e0[3]);
    xfrm_e0[4] = b_e0.createOr(mrkr_cd5to_ccb, xfrm_e0[4]);

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
    PabloAST * mrkr_102eto_1026 = b_e1.createAdvance(found_1025_102e, 2);
    xfrm_e1[3] = b_e1.createOr(mrkr_102eto_1026, xfrm_e1[3]);
//  Cases for 1b05
    PabloAST * after_1b05 = b_e1.createAdvance(E1_1b05, 3);
//     1b05 + 1b35 => 1b06
    PabloAST * found_1b05_1b35 = b_e1.createAnd(after_1b05, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b05_1b35);
    PabloAST * mrkr_1b35to_1b06 = b_e1.createAdvance(found_1b05_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b06, xfrm_e1[0]);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b06, xfrm_e1[1]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b06, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b06, xfrm_e1[5]);
//  Cases for 1b07
    PabloAST * after_1b07 = b_e1.createAdvance(E1_1b07, 3);
//     1b07 + 1b35 => 1b08
    PabloAST * found_1b07_1b35 = b_e1.createAnd(after_1b07, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b07_1b35);
    PabloAST * mrkr_1b35to_1b08 = b_e1.createAdvance(found_1b07_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b08, xfrm_e1[0]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b08, xfrm_e1[2]);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b08, xfrm_e1[3]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b08, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b08, xfrm_e1[5]);
//  Cases for 1b09
    PabloAST * after_1b09 = b_e1.createAdvance(E1_1b09, 3);
//     1b09 + 1b35 => 1b0a
    PabloAST * found_1b09_1b35 = b_e1.createAnd(after_1b09, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b09_1b35);
    PabloAST * mrkr_1b35to_1b0a = b_e1.createAdvance(found_1b09_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[0]);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[1]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[2]);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[3]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b0a, xfrm_e1[5]);
//  Cases for 1b0b
    PabloAST * after_1b0b = b_e1.createAdvance(E1_1b0b, 3);
//     1b0b + 1b35 => 1b0c
    PabloAST * found_1b0b_1b35 = b_e1.createAnd(after_1b0b, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b0b_1b35);
    PabloAST * mrkr_1b35to_1b0c = b_e1.createAdvance(found_1b0b_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b0c, xfrm_e1[0]);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b0c, xfrm_e1[3]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b0c, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b0c, xfrm_e1[5]);
//  Cases for 1b0d
    PabloAST * after_1b0d = b_e1.createAdvance(E1_1b0d, 3);
//     1b0d + 1b35 => 1b0e
    PabloAST * found_1b0d_1b35 = b_e1.createAnd(after_1b0d, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b0d_1b35);
    PabloAST * mrkr_1b35to_1b0e = b_e1.createAdvance(found_1b0d_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b0e, xfrm_e1[0]);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b0e, xfrm_e1[1]);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b0e, xfrm_e1[3]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b0e, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b0e, xfrm_e1[5]);
//  Cases for 1b11
    PabloAST * after_1b11 = b_e1.createAdvance(E1_1b11, 3);
//     1b11 + 1b35 => 1b12
    PabloAST * found_1b11_1b35 = b_e1.createAnd(after_1b11, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b11_1b35);
    PabloAST * mrkr_1b35to_1b12 = b_e1.createAdvance(found_1b11_1b35, 2);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b12, xfrm_e1[0]);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b12, xfrm_e1[1]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b12, xfrm_e1[2]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b12, xfrm_e1[5]);
//  Cases for 1b3a
    PabloAST * after_1b3a = b_e1.createAdvance(E1_1b3a, 3);
//     1b3a + 1b35 => 1b3b
    PabloAST * found_1b3a_1b35 = b_e1.createAnd(after_1b3a, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3a_1b35);
    PabloAST * mrkr_1b35to_1b3b = b_e1.createAdvance(found_1b3a_1b35, 2);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b3b, xfrm_e1[1]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b3b, xfrm_e1[2]);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b3b, xfrm_e1[3]);
//  Cases for 1b3c
    PabloAST * after_1b3c = b_e1.createAdvance(E1_1b3c, 3);
//     1b3c + 1b35 => 1b3d
    PabloAST * found_1b3c_1b35 = b_e1.createAnd(after_1b3c, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3c_1b35);
    PabloAST * mrkr_1b35to_1b3d = b_e1.createAdvance(found_1b3c_1b35, 2);
    xfrm_e1[3] = b_e1.createOr(mrkr_1b35to_1b3d, xfrm_e1[3]);
//  Cases for 1b3e
    PabloAST * after_1b3e = b_e1.createAdvance(E1_1b3e, 3);
//     1b3e + 1b35 => 1b40
    PabloAST * found_1b3e_1b35 = b_e1.createAnd(after_1b3e, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3e_1b35);
    PabloAST * mrkr_1b35to_1b40 = b_e1.createAdvance(found_1b3e_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b40, xfrm_e1[0]);
    mrkr_1b35to_1b40 = b_e1.createAdvance(mrkr_1b35to_1b40, 1);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b40, xfrm_e1[0]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b40, xfrm_e1[2]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b40, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b40, xfrm_e1[5]);
//  Cases for 1b3f
    PabloAST * after_1b3f = b_e1.createAdvance(E1_1b3f, 3);
//     1b3f + 1b35 => 1b41
    PabloAST * found_1b3f_1b35 = b_e1.createAnd(after_1b3f, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b3f_1b35);
    PabloAST * mrkr_1b35to_1b41 = b_e1.createAdvance(found_1b3f_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b41, xfrm_e1[0]);
    mrkr_1b35to_1b41 = b_e1.createAdvance(mrkr_1b35to_1b41, 1);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b41, xfrm_e1[2]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b41, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b41, xfrm_e1[5]);
//  Cases for 1b42
    PabloAST * after_1b42 = b_e1.createAdvance(E1_1b42, 3);
//     1b42 + 1b35 => 1b43
    PabloAST * found_1b42_1b35 = b_e1.createAnd(after_1b42, E1_1b35);
    del_prior_e1 = b_e1.createOr(del_prior_e1, found_1b42_1b35);
    PabloAST * mrkr_1b35to_1b43 = b_e1.createAdvance(found_1b42_1b35, 1);
    xfrm_e1[0] = b_e1.createOr(mrkr_1b35to_1b43, xfrm_e1[0]);
    mrkr_1b35to_1b43 = b_e1.createAdvance(mrkr_1b35to_1b43, 1);
    xfrm_e1[1] = b_e1.createOr(mrkr_1b35to_1b43, xfrm_e1[1]);
    xfrm_e1[2] = b_e1.createOr(mrkr_1b35to_1b43, xfrm_e1[2]);
    xfrm_e1[4] = b_e1.createOr(mrkr_1b35to_1b43, xfrm_e1[4]);
    xfrm_e1[5] = b_e1.createOr(mrkr_1b35to_1b43, xfrm_e1[5]);

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
    PabloAST * mrkr_11127to_1112e = b_f0.createAdvance(found_11131_11127, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_11127to_1112e, xfrm_f0[0]);
    xfrm_f0[3] = b_f0.createOr(mrkr_11127to_1112e, xfrm_f0[3]);
//  Cases for 11132
    PabloAST * after_11132 = b_f0.createAdvance(F0_11132, 4);
//     11132 + 11127 => 1112f
    PabloAST * found_11132_11127 = b_f0.createAnd(after_11132, F0_11127);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11132_11127);
    PabloAST * mrkr_11127to_1112f = b_f0.createAdvance(found_11132_11127, 3);
    xfrm_f0[3] = b_f0.createOr(mrkr_11127to_1112f, xfrm_f0[3]);
//  Cases for 11347
    PabloAST * after_11347 = b_f0.createAdvance(F0_11347, 4);
//     11347 + 1133e => 1134b
    PabloAST * found_11347_1133e = b_f0.createAnd(after_11347, F0_1133e);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11347_1133e);
    PabloAST * mrkr_1133eto_1134b = b_f0.createAdvance(found_11347_1133e, 2);
    xfrm_f0[0] = b_f0.createOr(mrkr_1133eto_1134b, xfrm_f0[0]);
    mrkr_1133eto_1134b = b_f0.createAdvance(mrkr_1133eto_1134b, 1);
    xfrm_f0[0] = b_f0.createOr(mrkr_1133eto_1134b, xfrm_f0[0]);
    xfrm_f0[2] = b_f0.createOr(mrkr_1133eto_1134b, xfrm_f0[2]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1133eto_1134b, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1133eto_1134b, xfrm_f0[5]);
//     11347 + 11357 => 1134c
    PabloAST * found_11347_11357 = b_f0.createAnd(after_11347, F0_11357);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11347_11357);
    PabloAST * mrkr_11357to_1134c = b_f0.createAdvance(found_11347_11357, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_11357to_1134c, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_11357to_1134c, xfrm_f0[1]);
    xfrm_f0[3] = b_f0.createOr(mrkr_11357to_1134c, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_11357to_1134c, xfrm_f0[4]);
//  Cases for 11382
    PabloAST * after_11382 = b_f0.createAdvance(F0_11382, 4);
//     11382 + 113c9 => 11383
    PabloAST * found_11382_113c9 = b_f0.createAnd(after_11382, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11382_113c9);
    PabloAST * mrkr_113c9to_11383 = b_f0.createAdvance(found_11382_113c9, 2);
    xfrm_f0[0] = b_f0.createOr(mrkr_113c9to_11383, xfrm_f0[0]);
    mrkr_113c9to_11383 = b_f0.createAdvance(mrkr_113c9to_11383, 1);
    xfrm_f0[1] = b_f0.createOr(mrkr_113c9to_11383, xfrm_f0[1]);
    xfrm_f0[3] = b_f0.createOr(mrkr_113c9to_11383, xfrm_f0[3]);
//  Cases for 11384
    PabloAST * after_11384 = b_f0.createAdvance(F0_11384, 4);
//     11384 + 113bb => 11385
    PabloAST * found_11384_113bb = b_f0.createAnd(after_11384, F0_113bb);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11384_113bb);
    PabloAST * mrkr_113bbto_11385 = b_f0.createAdvance(found_11384_113bb, 3);
    xfrm_f0[1] = b_f0.createOr(mrkr_113bbto_11385, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_113bbto_11385, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_113bbto_11385, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_113bbto_11385, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_113bbto_11385, xfrm_f0[5]);
//  Cases for 1138b
    PabloAST * after_1138b = b_f0.createAdvance(F0_1138b, 4);
//     1138b + 113c2 => 1138e
    PabloAST * found_1138b_113c2 = b_f0.createAnd(after_1138b, F0_113c2);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1138b_113c2);
    PabloAST * mrkr_113c2to_1138e = b_f0.createAdvance(found_1138b_113c2, 2);
    xfrm_f0[0] = b_f0.createOr(mrkr_113c2to_1138e, xfrm_f0[0]);
    mrkr_113c2to_1138e = b_f0.createAdvance(mrkr_113c2to_1138e, 1);
    xfrm_f0[2] = b_f0.createOr(mrkr_113c2to_1138e, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_113c2to_1138e, xfrm_f0[3]);
//  Cases for 11390
    PabloAST * after_11390 = b_f0.createAdvance(F0_11390, 4);
//     11390 + 113c9 => 11391
    PabloAST * found_11390_113c9 = b_f0.createAnd(after_11390, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11390_113c9);
    PabloAST * mrkr_113c9to_11391 = b_f0.createAdvance(found_11390_113c9, 2);
    xfrm_f0[0] = b_f0.createOr(mrkr_113c9to_11391, xfrm_f0[0]);
    mrkr_113c9to_11391 = b_f0.createAdvance(mrkr_113c9to_11391, 1);
    xfrm_f0[3] = b_f0.createOr(mrkr_113c9to_11391, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_113c9to_11391, xfrm_f0[4]);
//  Cases for 113c2
    PabloAST * after_113c2 = b_f0.createAdvance(F0_113c2, 4);
//     113c2 + 113c2 => 113c5
    PabloAST * found_113c2_113c2 = b_f0.createAnd(after_113c2, F0_113c2);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113c2);
    PabloAST * mrkr_113c2to_113c5 = b_f0.createAdvance(found_113c2_113c2, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_113c2to_113c5, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_113c2to_113c5, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_113c2to_113c5, xfrm_f0[2]);
//     113c2 + 113b8 => 113c7
    PabloAST * found_113c2_113b8 = b_f0.createAnd(after_113c2, F0_113b8);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113b8);
    PabloAST * mrkr_113b8to_113c7 = b_f0.createAdvance(found_113c2_113b8, 2);
    xfrm_f0[0] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[0]);
    mrkr_113b8to_113c7 = b_f0.createAdvance(mrkr_113b8to_113c7, 1);
    xfrm_f0[0] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_113b8to_113c7, xfrm_f0[5]);
//     113c2 + 113c9 => 113c8
    PabloAST * found_113c2_113c9 = b_f0.createAnd(after_113c2, F0_113c9);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_113c2_113c9);
    PabloAST * mrkr_113c9to_113c8 = b_f0.createAdvance(found_113c2_113c9, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_113c9to_113c8, xfrm_f0[0]);
//  Cases for 114b9
    PabloAST * after_114b9 = b_f0.createAdvance(F0_114b9, 4);
//     114b9 + 114ba => 114bb
    PabloAST * found_114b9_114ba = b_f0.createAnd(after_114b9, F0_114ba);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114ba);
    PabloAST * mrkr_114bato_114bb = b_f0.createAdvance(found_114b9_114ba, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_114bato_114bb, xfrm_f0[0]);
//     114b9 + 114b0 => 114bc
    PabloAST * found_114b9_114b0 = b_f0.createAnd(after_114b9, F0_114b0);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114b0);
    PabloAST * mrkr_114b0to_114bc = b_f0.createAdvance(found_114b9_114b0, 3);
    xfrm_f0[2] = b_f0.createOr(mrkr_114b0to_114bc, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_114b0to_114bc, xfrm_f0[3]);
//     114b9 + 114bd => 114be
    PabloAST * found_114b9_114bd = b_f0.createAnd(after_114b9, F0_114bd);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_114b9_114bd);
    PabloAST * mrkr_114bdto_114be = b_f0.createAdvance(found_114b9_114bd, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_114bdto_114be, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_114bdto_114be, xfrm_f0[1]);
//  Cases for 115b8
    PabloAST * after_115b8 = b_f0.createAdvance(F0_115b8, 4);
//     115b8 + 115af => 115ba
    PabloAST * found_115b8_115af = b_f0.createAnd(after_115b8, F0_115af);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_115b8_115af);
    PabloAST * mrkr_115afto_115ba = b_f0.createAdvance(found_115b8_115af, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_115afto_115ba, xfrm_f0[0]);
    xfrm_f0[2] = b_f0.createOr(mrkr_115afto_115ba, xfrm_f0[2]);
    xfrm_f0[4] = b_f0.createOr(mrkr_115afto_115ba, xfrm_f0[4]);
//  Cases for 115b9
    PabloAST * after_115b9 = b_f0.createAdvance(F0_115b9, 4);
//     115b9 + 115af => 115bb
    PabloAST * found_115b9_115af = b_f0.createAnd(after_115b9, F0_115af);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_115b9_115af);
    PabloAST * mrkr_115afto_115bb = b_f0.createAdvance(found_115b9_115af, 3);
    xfrm_f0[2] = b_f0.createOr(mrkr_115afto_115bb, xfrm_f0[2]);
    xfrm_f0[4] = b_f0.createOr(mrkr_115afto_115bb, xfrm_f0[4]);
//  Cases for 11935
    PabloAST * after_11935 = b_f0.createAdvance(F0_11935, 4);
//     11935 + 11930 => 11938
    PabloAST * found_11935_11930 = b_f0.createAnd(after_11935, F0_11930);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_11935_11930);
    PabloAST * mrkr_11930to_11938 = b_f0.createAdvance(found_11935_11930, 3);
    xfrm_f0[3] = b_f0.createOr(mrkr_11930to_11938, xfrm_f0[3]);
//  Cases for 1611e
    PabloAST * after_1611e = b_f0.createAdvance(F0_1611e, 4);
//     1611e + 1611e => 16121
    PabloAST * found_1611e_1611e = b_f0.createAnd(after_1611e, F0_1611e);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_1611e);
    PabloAST * mrkr_1611eto_16121 = b_f0.createAdvance(found_1611e_1611e, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1611eto_16121, xfrm_f0[5]);
//     1611e + 16129 => 16122
    PabloAST * found_1611e_16129 = b_f0.createAnd(after_1611e, F0_16129);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_16129);
    PabloAST * mrkr_16129to_16122 = b_f0.createAdvance(found_1611e_16129, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_16129to_16122, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_16129to_16122, xfrm_f0[1]);
    xfrm_f0[3] = b_f0.createOr(mrkr_16129to_16122, xfrm_f0[3]);
//     1611e + 1611f => 16123
    PabloAST * found_1611e_1611f = b_f0.createAnd(after_1611e, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_1611f);
    PabloAST * mrkr_1611fto_16123 = b_f0.createAdvance(found_1611e_1611f, 3);
    xfrm_f0[2] = b_f0.createOr(mrkr_1611fto_16123, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_1611fto_16123, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1611fto_16123, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1611fto_16123, xfrm_f0[5]);
//     1611e + 16120 => 16125
    PabloAST * found_1611e_16120 = b_f0.createAnd(after_1611e, F0_16120);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_1611e_16120);
    PabloAST * mrkr_16120to_16125 = b_f0.createAdvance(found_1611e_16120, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_16120to_16125, xfrm_f0[0]);
    xfrm_f0[2] = b_f0.createOr(mrkr_16120to_16125, xfrm_f0[2]);
//  Cases for 16129
    PabloAST * after_16129 = b_f0.createAdvance(F0_16129, 4);
//     16129 + 1611f => 16124
    PabloAST * found_16129_1611f = b_f0.createAnd(after_16129, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16129_1611f);
    PabloAST * mrkr_1611fto_16124 = b_f0.createAdvance(found_16129_1611f, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_1611fto_16124, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_1611fto_16124, xfrm_f0[1]);
    xfrm_f0[3] = b_f0.createOr(mrkr_1611fto_16124, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1611fto_16124, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1611fto_16124, xfrm_f0[5]);
//  Cases for 16d63
    PabloAST * after_16d63 = b_f0.createAdvance(F0_16d63, 4);
//     16d63 + 16d67 => 16d69
    PabloAST * found_16d63_16d67 = b_f0.createAnd(after_16d63, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d63_16d67);
    PabloAST * mrkr_16d67to_16d69 = b_f0.createAdvance(found_16d63_16d67, 3);
    xfrm_f0[1] = b_f0.createOr(mrkr_16d67to_16d69, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_16d67to_16d69, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_16d67to_16d69, xfrm_f0[3]);
//  Cases for 16d67
    PabloAST * after_16d67 = b_f0.createAdvance(F0_16d67, 4);
//     16d67 + 16d67 => 16d68
    PabloAST * found_16d67_16d67 = b_f0.createAnd(after_16d67, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d67_16d67);
    PabloAST * mrkr_16d67to_16d68 = b_f0.createAdvance(found_16d67_16d67, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_16d67to_16d68, xfrm_f0[0]);
    xfrm_f0[1] = b_f0.createOr(mrkr_16d67to_16d68, xfrm_f0[1]);
    xfrm_f0[2] = b_f0.createOr(mrkr_16d67to_16d68, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_16d67to_16d68, xfrm_f0[3]);
//  Cases for 16121
    PabloAST * after_16121 = b_f0.createAdvance(b_f0.createOr(F0_16121, found_1611e_1611e), 4);
//     16121 + 1611f => 16126
    PabloAST * found_16121_1611f = b_f0.createAnd(after_16121, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16121_1611f);
    PabloAST * mrkr_1611fto_16126 = b_f0.createAdvance(found_16121_1611f, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_1611fto_16126, xfrm_f0[0]);
    xfrm_f0[3] = b_f0.createOr(mrkr_1611fto_16126, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1611fto_16126, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1611fto_16126, xfrm_f0[5]);
//     16121 + 16120 => 16128
    PabloAST * found_16121_16120 = b_f0.createAnd(after_16121, F0_16120);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16121_16120);
    PabloAST * mrkr_16120to_16128 = b_f0.createAdvance(found_16121_16120, 3);
    xfrm_f0[3] = b_f0.createOr(mrkr_16120to_16128, xfrm_f0[3]);
//  Cases for 16122
    PabloAST * after_16122 = b_f0.createAdvance(b_f0.createOr(F0_16122, found_1611e_16129), 4);
//     16122 + 1611f => 16127
    PabloAST * found_16122_1611f = b_f0.createAnd(after_16122, F0_1611f);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16122_1611f);
    PabloAST * mrkr_1611fto_16127 = b_f0.createAdvance(found_16122_1611f, 3);
    xfrm_f0[3] = b_f0.createOr(mrkr_1611fto_16127, xfrm_f0[3]);
    xfrm_f0[4] = b_f0.createOr(mrkr_1611fto_16127, xfrm_f0[4]);
    xfrm_f0[5] = b_f0.createOr(mrkr_1611fto_16127, xfrm_f0[5]);
//  Cases for 16d69
    PabloAST * after_16d69 = b_f0.createAdvance(b_f0.createOr(F0_16d69, found_16d63_16d67), 4);
//     16d69 + 16d67 => 16d6a
    PabloAST * found_16d69_16d67 = b_f0.createAnd(after_16d69, F0_16d67);
    del_prior_f0 = b_f0.createOr(del_prior_f0, found_16d69_16d67);
    PabloAST * mrkr_16d67to_16d6a = b_f0.createAdvance(found_16d69_16d67, 3);
    xfrm_f0[0] = b_f0.createOr(mrkr_16d67to_16d6a, xfrm_f0[0]);
    xfrm_f0[2] = b_f0.createOr(mrkr_16d67to_16d6a, xfrm_f0[2]);
    xfrm_f0[3] = b_f0.createOr(mrkr_16d67to_16d6a, xfrm_f0[3]);

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

    StreamSet * XfrmBasis = P.CreateStreamSet(8, 1);
    StreamSet * DelPrior = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<ShortComposableTranslation>(NSD_Basis, DelPrior, XfrmBasis);
    SHOW_BIXNUM(XfrmBasis);
    SHOW_STREAM(DelPrior);

    StreamSet * XfrmedBasis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<ApplyTransform>(NSD_Basis, XfrmBasis, XfrmedBasis);
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
