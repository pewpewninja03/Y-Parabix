/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/normalization.h>
#include <unicode/core/unicode_set.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/unicode/charclasses.h>
#include <toolchain/toolchain.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <pablo/pablo_kernel.h>

using namespace pablo;
using namespace kernel;
using namespace llvm;

Hangul_Composables::Hangul_Composables (LLVMTypeSystemInterface & ts,
                                              StreamSet * Basis, StreamSet * L_V_T_Composables,
                                              pablo::BitMovementMode m)
: PabloKernel(ts, "Hangul_Composables" + Basis->shapeString(),
// inputs
    {},
// output
    {Binding{"L_V_T_Composables", L_V_T_Composables}}), mBitMovement(m) {
    if (m == pablo::BitMovementMode::LookAhead) {
        // UTF-8 lookahead
        mInputStreamSets.push_back(Binding{"Basis", Basis, FixedRate(1), LookAhead(3)});
    } else {
        mInputStreamSets.push_back(Binding{"Basis", Basis});
    }
}

void Hangul_Composables::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb, mBitMovement);
    std::vector<Var *> Hangul_var(HC_Kind::Count);
    for (unsigned i = 0; i < HC_Kind::Count; i++) {
        Hangul_var[i] = pb.createVar("Composable_" + std::to_string(i), pb.createZeroes());
    }
    std::vector<UCD::UnicodeSet> Hangul_uset (HC_Kind::Count);
    Hangul_uset[HC_Kind::L] = UCD::UnicodeSet(Hangul::LBase, Hangul::LBase + Hangul::LCount - 1);
    Hangul_uset[HC_Kind::V] = UCD::UnicodeSet(Hangul::VBase, Hangul::VBase + Hangul::VCount - 1);
    for (unsigned i = 0; i < Hangul::LCount * Hangul::VCount; i++) {
        Hangul_uset[HC_Kind::LV].insert(Hangul::SBase + i * Hangul::TCount);
    }
    Hangul_uset[HC_Kind::T] = UCD::UnicodeSet(Hangul::TBase, Hangul::TBase + Hangul::TCount - 1);
    unicodeCompiler.compile(Hangul_var, Hangul_uset);
    writeOutputStreamSet("L_V_T_Composables", Hangul_var);
}

Hangul_Composition::Hangul_Composition (LLVMTypeSystemInterface & ts,
                                              StreamSet * Basis, StreamSet * L_V_T_Composables,
                                              StreamSet * Output_Basis)
: PabloKernel(ts, "Hangul_Composition" + Basis->shapeString(),
// inputs
    {Binding{"Basis", Basis, FixedRate(1), LookAhead(5)}, Binding{"L_V_T_Composables", L_V_T_Composables, FixedRate(1), LookAhead(5)}},
// output
    {Binding{"Output_Basis", Output_Basis}}) {
}

void Hangul_Composition::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> L_V_T_Composables = getInputStreamSet("L_V_T_Composables");
    PabloAST * Hangul_L = L_V_T_Composables[Hangul_Composables::HC_Kind::L];
    PabloAST * Hangul_V = L_V_T_Composables[Hangul_Composables::HC_Kind::V];
    PabloAST * Hangul_LV = L_V_T_Composables[Hangul_Composables::HC_Kind::LV];
    PabloAST * Hangul_T = L_V_T_Composables[Hangul_Composables::HC_Kind::T];
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
        for (unsigned i = 0; i < 8; i++) {
            PabloAST * updated = nested.createAnd(Basis[i], maskVar);
            if (i < 4) {
                // At prefix (LV_sequence) positions, set 4 data bits, keep high 4 bits unchanged.
                updated = nested.createSel(LV_sequence, Composed[i+12], updated);
            }
            if (i < 6) {
                // Suffix bytes - set 6 data bits for each, keep top two bits unchanged.
                updated = nested.createSel(LV_suffix1, nested.createAdvance(Composed[i+6], 1), updated);
                updated = nested.createSel(LV_suffix2, nested.createAdvance(Composed[i], 2), updated);
            }
            nested.createAssign(outputVar[i], updated);
        }
    } else {
        Composed = bnc.ZeroExtend(Composed, Basis.size());
        for (unsigned i = 0; i < Basis.size(); i++) {
            PabloAST * selected = nested.createAnd(Basis[i], maskVar);
            nested.createAssign(outputVar[i], nested.createSel(LV_sequence, Composed[i], selected));
        }
    }
    writeOutputStreamSet("Output_Basis", outputVar);
}

void UpdateBitXfrms(PabloBuilder & pb, std::vector<Var *> BitXfrmBasis,
                    PabloAST * marker, std::vector<PabloAST *> & sets, std::vector<BitXfrmSpec> & xfrmSpecs) {
    unsigned max_pos = 0;
    for (auto & s : xfrmSpecs) {
        if (s.position > max_pos) {
            max_pos = s.position;
        }
    }
    std::vector<std::vector<PabloAST *>> combined(max_pos + 1);
    if (marker == nullptr) {
        combined[0] = sets;
    } else {
        combined[0].resize(sets.size());
        for (unsigned i = 0; i < sets.size(); i++) {
            combined[0][i] = pb.createAnd(sets[i], marker);
        }
    }
    for (unsigned pos = 1; pos <= max_pos; pos++) {
        combined[pos].resize(sets.size());
        for (unsigned i = 0; i < sets.size(); i++) {
           combined[pos][i] = nullptr;
        }
    }
    for (auto & spec : xfrmSpecs) {
        unsigned pos = spec.position;
        unsigned idx = spec.BitXfrmIndex;
        if (combined[pos][idx] == nullptr) {
            combined[pos][idx] = pb.createAdvance(combined[0][idx], pos);
        }
        pb.createAssign(BitXfrmBasis[spec.bit], pb.createOr(BitXfrmBasis[spec.bit], combined[pos][idx]));
    }
}

//  Self-Composable Logic
//  Given Unicode characters AA and A such that AA has a canonical
//  decomposition AA ==> [A, A], the character A is called a
//  self-composable, while the character AA is called a doubleton.
//
//  Two sets of such characters are:
//  0x16121 => [0x1611e, 0x1611e]
//  0x16d68 => [0x16d67, 0x16d67]
//
//  Conversion of sequences of self-composables and doubletons to NFC form
//  include the following examples.
//
//  A AA ==> AA A
//  A A A ==> AA A
//  A AA A ==> AA AA
//  A A AA A  ==> AA AA A
//
//  For each span of As and AAs:
//  1.  For each odd_A followed immediately by an A (even_A), 
//      a.  mark the odd_A for deletion, and
//      b.  mark the even A for conversion to AA.
//  2.  For each odd_A followed by a non-empty run of AAs and then an even A:
//      a. mark the initial odd A for conversion to AA
//      b. mark the even A for conversion to AA.
//      c. mark the final AA within the run for deletion
//  3.  For each odd_A followed by a non-empty run of AAs with no final A,
//      a. mark the initial odd A for conversion to AA
//      b. mark the final AA to for conversion to A.
//      A AA AA AA X => AA AA AA A X

SCResults SelfComposableLogic(PabloBuilder & pb, std::vector<PabloAST *> Basis,
                              unsigned A_len, unsigned AA_len,
                              PabloAST * A, PabloAST * AA,
                              PabloAST * A_ahead, PabloAST * AA_ahead) {
    SCResults results;
    PabloAST * suffix = pb.createAnd(Basis[7], pb.createNot(Basis[6]));
    PabloAST * A_or_AA_follow = pb.createAdvanceThenScanThru(pb.createOr(A, AA), suffix, "selfc.A_or_AA_follow");
    PabloAST * AA_run_start = pb.createAnd(AA, pb.createNot(A_or_AA_follow), "selfc.AA_run_start");
    PabloAST * AA_run_continue = pb.createOr(suffix, AA);
    PabloAST * AA_run_follow = pb.createScanThru(AA_run_start, AA_run_continue, "selfc.AA_run_follow");
    PabloAST * A_run_start = pb.createAnd(A, pb.createOr(AA_run_follow, pb.createNot(A_or_AA_follow)), "selfc.A_run_start");
    PabloAST * A1 = pb.createEveryNth(A, pb.getInteger(2), "selfc.A1");  //  1st, 3rd, 5th, ... of all the As
    PabloAST * A2 = pb.createXor(A, A1, "selfc.A2");  //  2nd, 4th, 6th, ... of the As
    PabloAST * A1_start = pb.createAnd(A_run_start, A1, "selfc.A1_start");
    PabloAST * A2_start = pb.createAnd(A_run_start, A2, "selfc.A2_start");
    PabloAST * A_continue = A;
    PabloAST * run_continue = pb.createOr3(suffix, A, AA, "selfc.run_continue");
    PabloAST * A1_runs = pb.createMatchStar(A1_start, run_continue, "selfc.A1_runs");
    PabloAST * A2_runs = pb.createMatchStar(A2_start, run_continue, "selfc.A2_runs");
    PabloAST * A_odd = pb.createOr(pb.createAnd(A1_runs, A1), pb.createAnd(A2_runs, A2), "selfc.A_odd");
    PabloAST * A_even = pb.createOr(pb.createAnd(A1_runs, A2), pb.createAnd(A2_runs, A1), "selfc.A_even");
    //
    PabloAST * AA1 = pb.createAnd(AA, pb.createAdvance(A_odd, A_len), "selfc.AA1");
    PabloAST * A_odd_AA_run = pb.createMatchStar(AA1, AA_run_continue, "selfc.A_odd_AA_run");
    PabloAST * A_odd_AA_final = pb.createAnd3(A_odd_AA_run, AA, pb.createNot(AA_ahead), "selfc.A_odd_AA_final");
    // Rule 1a
    PabloAST * A_to_delete = pb.createAnd(A_odd, A_ahead, "selfc.A_to_delete");
    // Rules 1b, 2a, 2b, 3a 
    results.A_to_convert_to_AA = pb.createOr(A_even, pb.createAnd(A_odd, AA_ahead), "selfc.A_to_convert_to_AA");
    // Rule 2c
    PabloAST * AA_to_delete = pb.createAnd(A_odd_AA_final, A_ahead, "selfc.AA_to_delete");
    // Rule 3b
    results.AA_to_convert_to_A = pb.createAnd(A_odd_AA_final, pb.createNot(A_ahead), "selfc.AA_to_convert_to_A");
    PabloAST * to_delete = pb.createOr(A_to_delete, AA_to_delete);
    results.A_or_AA_to_delete = to_delete;
    auto min_len = std::min(A_len, AA_len);
    for (unsigned i = 2; i <= min_len; i++) {
        results.A_or_AA_to_delete = pb.createOr(results.A_or_AA_to_delete, pb.createAdvance(to_delete, i - 1));
    }
    if (A_len > min_len) {
        for (unsigned i = min_len + 1; i <= A_len; i++) {
            results.A_or_AA_to_delete = pb.createOr(results.A_or_AA_to_delete, pb.createAdvance(A_to_delete, i - 1));
        }
    } else if (AA_len > min_len) {
        for (unsigned i = min_len + 1; i <= AA_len; i++) {
            results.A_or_AA_to_delete = pb.createOr(results.A_or_AA_to_delete, pb.createAdvance(AA_to_delete, i - 1));
        }
    }
    return results;
}

class CreateU8_FilterMask : public pablo::PabloKernel {
public:
    CreateU8_FilterMask(LLVMTypeSystemInterface & ts, StreamSet * DeletionBixNum, StreamSet * DelMask);
protected:
    void generatePabloMethod() override;
private:
    unsigned mBixBits;
};

CreateU8_FilterMask::CreateU8_FilterMask (LLVMTypeSystemInterface & ts,
                                StreamSet * DeletionBixNum,
                                StreamSet * DelMask)
: PabloKernel(ts, "u8_delmask@-1_" + DeletionBixNum->shapeString(),
// inputs
{Binding{"deletion_bixnum", DeletionBixNum, FixedRate(1), LookAhead(3)}},
// output
{Binding{"selection_mask", DelMask}}),
    mBixBits(DeletionBixNum->getNumElements()) {
}

void CreateU8_FilterMask::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * deletion_bixnum = getInputStreamVar("deletion_bixnum");
    Var * del_bixnum0 = pb.createExtract(deletion_bixnum, pb.getInteger(0));
    // Deletion and insertion bixnums are calculated at the final position of
    // a UTF-8 sequence.   Deletion masks will preserve this position.
    //
    PabloAST * del_mask = pb.createLookahead(del_bixnum0, 1);
    if (mBixBits == 2) {
        Var * del_bixnum1 = pb.createExtract(deletion_bixnum, pb.getInteger(1));
        // Mark two positions for deletion.
        del_mask = pb.createOr3(del_mask, pb.createLookahead(del_bixnum1, 1), pb.createLookahead(del_bixnum1, 2));
        // If both del_bixnum0 and del_bixnum1 are 1, then 3 positions must be deleted.
        del_mask = pb.createOr(del_mask, pb.createAnd(pb.createLookahead(del_bixnum0, 3), pb.createLookahead(del_bixnum1, 3)));
    }
    PabloAST * selected = pb.createInFile(pb.createNot(del_mask));
    Var * const selection_mask = getOutputStreamVar("selection_mask");
    pb.createAssign(pb.createExtract(selection_mask, pb.getInteger(0)), selected);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

void ComputeWorkPlacement(PipelineBuilder & P,
                          std::vector<re::CC *> insertionBixNumCCs,
                          std::vector<re::CC *> deletionBixNumCCs,
                          StreamSet * U8_Basis, StreamSet * WorkSelectionMask,
                          StreamSet * WorkPlacementMask) {
    StreamSet * const U8_Insertion_BixNum = P.CreateStreamSet(insertionBixNumCCs.size());
    P.CreateKernelCall<CharClassesKernel>(insertionBixNumCCs, U8_Basis, U8_Insertion_BixNum);
    SHOW_BIXNUM(U8_Insertion_BixNum);

    StreamSet * const U8_Deletion_BixNum = P.CreateStreamSet(deletionBixNumCCs.size());
    P.CreateKernelCall<CharClassesKernel>(deletionBixNumCCs, U8_Basis, U8_Deletion_BixNum);
    SHOW_BIXNUM(U8_Deletion_BixNum);

    StreamSet * const U8_FilterMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<CreateU8_FilterMask>(U8_Deletion_BixNum, U8_FilterMask);
    SHOW_STREAM(U8_FilterMask);

    StreamSet * const U8_SpreadMask = InsertionSpreadMask(P, U8_Insertion_BixNum, kernel::InsertPosition::After);
    SHOW_STREAM(U8_SpreadMask);

    StreamSet * const U8_PostSpreadFilterMask = P.CreateStreamSet(1, 1);
    ExpandFilter(P, U8_SpreadMask, U8_FilterMask, U8_PostSpreadFilterMask);
    SHOW_STREAM(U8_PostSpreadFilterMask);

    StreamSet * const WorkExpansionMask = P.CreateStreamSet(1, 1);
    ExpandFilter(P, U8_SpreadMask, WorkSelectionMask, WorkExpansionMask);

    FilterByMask(P, U8_PostSpreadFilterMask, WorkExpansionMask, WorkPlacementMask);
    SHOW_STREAM(WorkPlacementMask);
}

