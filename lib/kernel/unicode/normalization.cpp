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

