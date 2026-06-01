/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <cstdio>
#include <vector>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/codegenstate.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/sentinel.h>
#include <kernel/streamutils/sorting.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/normalization.h>
#include <kernel/unicode/char_replacement.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/utf8_support.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/unicode/resolve_properties.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <ucd/algo/normalization.h>
#include <ucd/core/unicode_set.h>
#include <ucd/data/PropertyAliases.h>
#include <ucd/data/PropertyObjects.h>
#include <ucd/data/PropertyObjectTable.h>
#include <ucd/utf/utf_compiler.h>
#include <ucd/utf/utf_encoder.h>
#include <ucd/utf/transchar.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

const unsigned S_Index_bits = 14;
const unsigned L_Index_bits = 5;
const unsigned V_Index_bits = 5;
const unsigned T_Index_bits = 5;


NFD_BixData::NFD_BixData() {
    UCD::NFD_Engine NFD_engine(UCD::DecompositionOptions::NFD);
    UTF_Encoder u8_encoder;
    UCD::StringPropertyObject * decompMappingObj = cast<UCD::StringPropertyObject>(getPropertyObject(UCD::dm));
    auto cps = decompMappingObj->GetExplicitCps();
    for (auto cp : cps) {
        std::u32string decomp;
        NFD_engine.NFD_append1(decomp, cp);
        unsigned NFD_expand = decomp.length() - 1;
        if (NFD_expand > 0) {
            mNFD_expansion.emplace(cp, NFD_expand);
        }
        unsigned UTF8_cp_len = mU8_encoder.encoded_length(cp);
        unsigned UTF8_nfd_len = mU8_encoder.encoded_length(decomp);
        if (UTF8_nfd_len < UTF8_cp_len) {
            mUTF8_deletion.emplace(cp, UTF8_cp_len - UTF8_nfd_len);
        } else if (UTF8_nfd_len > UTF8_cp_len) {
            mUTF8_expansion.emplace(cp, UTF8_nfd_len - UTF8_cp_len);
        }
        for (unsigned i = 0; i < decomp.length(); i++) {
            mNFD_CharMap[i].emplace(cp, decomp[i]);
        }
    }
    for (UCD::codepoint_t cp = Hangul::SBase; cp <= Hangul::MaxPrecomposed; cp += Hangul::TCount) {
        mHangul_Precomposed_LV.insert(cp);
    }
    UCD::UnicodeSet Hangul_Precomposed(Hangul::SBase, Hangul::MaxPrecomposed);
    mHangul_Precomposed_LVT = Hangul_Precomposed - mHangul_Precomposed_LV;
}

std::vector<re::CC *> NFD_BixData::NFD_Insertion_BixNumCCs() {
    unicode::BitTranslationSets BixNumCCs;
    BixNumCCs.push_back(UCD::UnicodeSet());
    BixNumCCs.push_back(UCD::UnicodeSet());
    for (auto p : mNFD_expansion) {
        auto insert_amt = p.second;
        for (unsigned i = 0; i < BixNumCCs.size(); i++) {
            unsigned bit = 1u << i;
            if ((insert_amt & bit) == bit) {
                BixNumCCs[i].insert(p.first);
            }
        }
    }
    BixNumCCs[0] = BixNumCCs[0] + mHangul_Precomposed_LV;
    BixNumCCs[1] = BixNumCCs[1] + mHangul_Precomposed_LVT;
    return {re::makeCC(BixNumCCs[0], &cc::Unicode),
            re::makeCC(BixNumCCs[1], &cc::Unicode)};
}

std::vector<re::CC *> NFD_BixData::UTF8_Insertion_BixNumCCs() {
    unicode::BitTranslationSets BixNumCCs;
    BixNumCCs.push_back(UCD::UnicodeSet());
    BixNumCCs.push_back(UCD::UnicodeSet());
    BixNumCCs.push_back(UCD::UnicodeSet());
    BixNumCCs.push_back(UCD::UnicodeSet());
    for (auto p : mUTF8_expansion) {
        auto insert_amt = p.second;
        for (unsigned i = 0; i < BixNumCCs.size(); i++) {
            unsigned bit = 1u << i;
            if ((insert_amt & bit) == bit) {
                BixNumCCs[i].insert(p.first);
            }
        }
    }
    // Each LV combination requires 3 additional bytes on decompositon; set bits 0 and 1.
    BixNumCCs[0] = BixNumCCs[0] + mHangul_Precomposed_LV;
    BixNumCCs[1] = BixNumCCs[1] + mHangul_Precomposed_LV;
    // Each LVT combination requires 6 additional bytes on decompositon; set bits 1 and 2.
    BixNumCCs[1] = BixNumCCs[1] + mHangul_Precomposed_LVT;
    BixNumCCs[2] = BixNumCCs[2] + mHangul_Precomposed_LVT;
    return {re::makeCC(BixNumCCs[0], &cc::Unicode),
        re::makeCC(BixNumCCs[1], &cc::Unicode),
        re::makeCC(BixNumCCs[2], &cc::Unicode),
        re::makeCC(BixNumCCs[3], &cc::Unicode)};
}

//
// The encoded length of a decomposition is shorter than the
// original UTF8 length of the codepoint by at most 3, so
// only two bits are required.
std::vector<re::CC *> NFD_BixData::UTF8_Deletion_BixNumCCs() {
    unicode::BitTranslationSets BixNumCCs;
    BixNumCCs.push_back(UCD::UnicodeSet());
    BixNumCCs.push_back(UCD::UnicodeSet());
    for (auto p : mUTF8_deletion) {
        auto delete_amt = p.second;
        for (unsigned i = 0; i < BixNumCCs.size(); i++) {
            unsigned bit = 1u << i;
            if ((delete_amt & bit) == bit) {
                BixNumCCs[i].insert(p.first);
            }
        }
    }
    return {re::makeCC(BixNumCCs[0], &cc::Unicode),
            re::makeCC(BixNumCCs[1], &cc::Unicode)};
}

unicode::BitTranslationSets NFD_BixData::NFD_1st_BitXorCCs() {
    return unicode::ComputeBitTranslationSets(mNFD_CharMap[0]);
}

unicode::BitTranslationSets NFD_BixData::NFD_2nd_BitCCs() {
    return unicode::ComputeBitTranslationSets(mNFD_CharMap[1], unicode::XlateMode::LiteralBit);
}

unicode::BitTranslationSets NFD_BixData::NFD_3rd_BitCCs() {
    return unicode::ComputeBitTranslationSets(mNFD_CharMap[2], unicode::XlateMode::LiteralBit);
}

unicode::BitTranslationSets NFD_BixData::NFD_4th_BitCCs() {
    return unicode::ComputeBitTranslationSets(mNFD_CharMap[3], unicode::XlateMode::LiteralBit);
}

class NFD_Translation : public pablo::PabloKernel {
public:
    NFD_Translation(LLVMTypeSystemInterface & ts, NFD_BixData & BixData,
                    StreamSet * Basis, StreamSet * Output);
protected:
    void generatePabloMethod() override;
    NFD_BixData & mBixData;
};

NFD_Translation::NFD_Translation (LLVMTypeSystemInterface & ts, NFD_BixData & BixData,
                                  StreamSet * Basis, StreamSet * Output)
: PabloKernel(ts, "NFD_Translation" + std::to_string(Basis->getNumElements()) + "x1" + UTF::kernelAnnotation(),
// inputs
{Binding{"basis", Basis}},
// output
{Binding{"Output", Output}}), mBixData(BixData) {
}

void NFD_Translation::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    unicode::BitTranslationSets NFD1 = mBixData.NFD_1st_BitXorCCs();
    unicode::BitTranslationSets NFD2 = mBixData.NFD_2nd_BitCCs();
    unicode::BitTranslationSets NFD3 = mBixData.NFD_3rd_BitCCs();
    unicode::BitTranslationSets NFD4 = mBixData.NFD_4th_BitCCs();
    std::vector<Var *> NFD1_Vars(NFD1.size());
    std::vector<Var *> NFD2_Vars(NFD2.size());
    std::vector<Var *> NFD3_Vars(NFD3.size());
    std::vector<Var *> NFD4_Vars(NFD4.size());
    std::vector<Var *> all_targets(NFD1.size() + NFD2.size() + NFD3.size() + NFD4.size());
    std::vector<UCD::UnicodeSet> all_CCs(NFD1.size() + NFD2.size() + NFD3.size() + NFD4.size());

    for (unsigned i = 0; i < NFD1.size(); i++) {
        Var * v = pb.createVar("NFD1_bit" + std::to_string(i), pb.createZeroes());
        NFD1_Vars[i] = v;
        all_targets[i] = v;
        all_CCs[i] = NFD1[i];
    }
    unsigned base = NFD1.size();
    for (unsigned i = 0; i < NFD2.size(); i++) {
        Var * v = pb.createVar("NFD2_bit" + std::to_string(i), pb.createZeroes());
        NFD2_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = NFD2[i];
    }
    base = base + NFD2.size();
    for (unsigned i = 0; i < NFD3.size(); i++) {
        Var * v = pb.createVar("NFD3_bit" + std::to_string(i), pb.createZeroes());
        NFD3_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = NFD3[i];
    }
    base = base + NFD3.size();
    for (unsigned i = 0; i < NFD4.size(); i++) {
        Var * v = pb.createVar("NFD4_bit" + std::to_string(i), pb.createZeroes());
        NFD4_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = NFD4[i];
    }
    unicodeCompiler.compile(all_targets, all_CCs);
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    Var * outputVar = getOutputStreamVar("Output");
    std::vector<PabloAST *> output_basis(basis.size());
    for (unsigned i = 0; i < basis.size(); i++) {
        if (i < NFD1.size()) {
            output_basis[i] = pb.createXor(basis[i], NFD1_Vars[i]);
        } else {
            output_basis[i] = basis[i];
        }
        if (i < NFD2.size()) {
            output_basis[i] = pb.createOr(pb.createAdvance(NFD2_Vars[i], 1), output_basis[i]);
        }
        if (i < NFD3.size()) {
            output_basis[i] = pb.createOr(pb.createAdvance(NFD3_Vars[i], 2), output_basis[i]);
        }
        if (i < NFD4.size()) {
            output_basis[i] = pb.createOr(pb.createAdvance(NFD4_Vars[i], 3), output_basis[i]);
        }
        pb.createAssign(pb.createExtract(outputVar, pb.getInteger(i)), output_basis[i]);
    }
}

class LVT_Indexes : public pablo::PabloKernel {
public:
    LVT_Indexes(LLVMTypeSystemInterface & ts,
                StreamSet * U21_Basis, StreamSet * LIndex, StreamSet * VIndex, StreamSet * TIndex);
protected:
    void generatePabloMethod() override;
};

LVT_Indexes::LVT_Indexes (LLVMTypeSystemInterface & ts,
                          StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index)
: PabloKernel(ts, "Hangul::LVT_Indexes21x1",
// inputs
{Binding{"basis", Basis}},
// output
    {Binding{"L_index", L_index}, Binding{"V_index", V_index}, Binding{"T_index", T_index}}) {
}

void LVT_Indexes::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    BixNumCompiler bnc0(pb);
    // We identify the Hangul precomposed LV and LVT characters as
    // any of the SCount = 11172 characters in the range from
    // SBase = 0xAC00 to SBase + SCount - 1.   This use 21-bit
    // Bixnum comparison operations UGE and ULT.
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul::SBase), bnc0.ULT(basis, Hangul::SBase + Hangul::SCount));
    BixNum ZeroBasis(basis.size(), pb.createZeroes());

    std::vector<Var *> L_indexVar(L_Index_bits);
    for (unsigned i = 0; i < L_Index_bits; i++) {
        L_indexVar[i] = pb.createVar("L_var" + std::to_string(i), pb.createZeroes());
    }
    std::vector<Var *> V_indexVar(V_Index_bits);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_indexVar[i] = pb.createVar("V_var" + std::to_string(i), pb.createZeroes());
    }
    std::vector<Var *> T_indexVar(T_Index_bits);
    for (unsigned i = 0; i < T_Index_bits; i++) {
        T_indexVar[i] = pb.createVar("T_var" + std::to_string(i), pb.createZeroes());
    }

    auto nested = pb.createScope();
    pb.createIf(Hangul_precomposed, nested);
    BixNumCompiler bnc(nested);

    // To extract the LVT indexes, we first calculate a normalized
    // S_Index value by bixnum subtraction of SBase.
    BixNum S_Index = bnc.Select(Hangul_precomposed, bnc.SubModular(basis, Hangul::SBase), ZeroBasis);
    // As there are only S_Index_bits = 14 significant bits in
    // the result, we can optimize our calculations by truncating.
    S_Index = bnc.Truncate(S_Index, S_Index_bits);
    BixNum LV_index;
    BixNum T_index;
    // Bixnum division computes both an integer dividend and remainder.
    // When dividing by TCount = 28, we get an intermediate LV_index
    // and the final T index.
    bnc.Div(S_Index, Hangul::TCount, LV_index, T_index);
    LV_index = bnc.Truncate(LV_index, L_Index_bits + V_Index_bits);
    BixNum L_index;
    BixNum V_index;
    // A further bixnum division computes the separate L and V index values.
    bnc.Div(LV_index, Hangul::VCount, L_index, V_index);
    
    for (unsigned i = 0; i < L_Index_bits; i++) {
        nested.createAssign(L_indexVar[i], L_index[i]);
    }
    for (unsigned i = 0; i < V_Index_bits; i++) {
        nested.createAssign(V_indexVar[i], V_index[i]);
    }
    for (unsigned i = 0; i < T_Index_bits; i++) {
        nested.createAssign(T_indexVar[i], T_index[i]);
    }
    writeOutputStreamSet("L_index", L_indexVar);
    writeOutputStreamSet("V_index", V_indexVar);
    writeOutputStreamSet("T_index", T_indexVar);
}

class LVT2NFD : public pablo::PabloKernel {
public:
    LVT2NFD(LLVMTypeSystemInterface & ts,
               StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index,
               StreamSet * NFD_Basis);
protected:
    void generatePabloMethod() override;
};

LVT2NFD::LVT2NFD (LLVMTypeSystemInterface & ts,
                        StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index,
                        StreamSet * NFD_Basis)
: PabloKernel(ts, "Hangul::LVT2NFD_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis},
    Binding{"L_index", L_index},
    Binding{"V_index", V_index},
    Binding{"T_index", T_index}},
// output
{Binding{"NFD_Basis", NFD_Basis}}) {
}

void LVT2NFD::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> L_index = getInputStreamSet("L_index");
    std::vector<PabloAST *> V_index = getInputStreamSet("V_index");
    std::vector<PabloAST *> T_index = getInputStreamSet("T_index");
    BixNumCompiler bnc0(pb);
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul::SBase), bnc0.ULT(basis, Hangul::SBase + Hangul::SCount));
    // Set up Vars to receive the generated basis values.
    std::vector<Var *> basisVar(21);
    for (unsigned i = 0; i < 21; i++) {
        basisVar[i] = pb.createVar("basisVar" + std::to_string(i), basis[i]);
    }
    auto nested = pb.createScope();
    pb.createIf(Hangul_precomposed, nested);
    BixNumCompiler bnc(nested);
    BixNum LPart = bnc.ZeroExtend(L_index, 21);
    // The LPart will be encoded at the original precomposed position.
    // The bixnum calculation simply adds the L index value to the
    // base codepoint value for Hangul L characters (LBase = 0x1100).
    // Because we have zero-extended to 21 bits, this gives the full
    // Unicode codepoint value.
    LPart = bnc.AddModular(LPart, Hangul::LBase);
    // The V Part, when it exists, is one position after the opening L Part.
    // This position must have been created by insertion step.
    PabloAST * V_position = nested.createAdvance(Hangul_precomposed, 1);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_index[i] = nested.createAdvance(V_index[i], 1);
    }
    // Similar to the bixnum calculation of L characters, V and T
    // characters are computed using bixnum addition of the
    // VBase = 0x1161 and TBase = 0x11A7 values, respectively.
    BixNum VPart = bnc.ZeroExtend(V_index, 21);
    VPart = bnc.AddModular(VPart, Hangul::VBase);
    // The T Part, if it exists is two positions after the opening
    // L Part.  A T Part only exists for LVT initials (T != 0).
    PabloAST * T_position = nested.createAdvance(Hangul_precomposed, 2);
    for (unsigned i = 0; i < T_Index_bits; i++) {
        T_index[i] = nested.createAdvance(T_index[i], 2);
    }
    T_position = nested.createAnd(T_position, bnc.NEQ(T_index, 0));
    BixNum TPart = bnc.ZeroExtend(T_index, 21);
    TPart = bnc.AddModular(TPart, Hangul::TBase);
    for (unsigned i = 0; i < 21; i++) {
        PabloAST * bit = nested.createSel(Hangul_precomposed, LPart[i], basis[i]);
        bit = nested.createSel(V_position, VPart[i], bit);
        bit = nested.createSel(T_position, TPart[i], bit);
        nested.createAssign(basisVar[i], bit);
    }
    writeOutputStreamSet("NFD_Basis", basisVar);
}
//
//  Given a canonical combining class (CCC) basis stream and the u8final index stream,
//  identify starts and ends of reorderable character sequences and determine any
//  violations of canonical combining class (CCC) order within these sequences.
//  The positions of non-reorderable characters (CCC=0) as well as start of file
//  and one past end of file are consider to identify potential starts and ends.
//  Any violation within a sequences is marked at the sequence end position
//  (that is, at the next non-reorderable character or one past EOF).
//
//
class CCC_Check : public pablo::PabloKernel {
public:
    CCC_Check(LLVMTypeSystemInterface & ts, StreamSet * CCC_Basis, StreamSet * u8index, StreamSet * EOF_mark, StreamSet * CCC_SeqMarks, StreamSet * CCC_Violation);
protected:
    void generatePabloMethod() override;
private:
    unsigned mCCC_basis_size;
};

CCC_Check::CCC_Check (LLVMTypeSystemInterface & ts, StreamSet * CCC_Basis, StreamSet * u8index, StreamSet * EOF_mark, StreamSet * CCC_SeqMarks, StreamSet * CCC_Violation)
: PabloKernel(ts, "CCC_Check_u8",
// inputs
{Binding{"CCC_Basis", CCC_Basis}, Binding{"u8index", u8index}, Binding{"EOF_mark", EOF_mark, FixedRate(), LookAhead(1)}},
// output
{Binding{"CCC_SeqMarks", CCC_SeqMarks}, Binding{"CCC_Violation", CCC_Violation}}),
mCCC_basis_size(CCC_Basis->getNumElements()) {
}

void CCC_Check::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNum CCC = getInputStreamSet("CCC_Basis");
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    PabloAST * EOF_mark = getInputStreamSet("EOF_mark")[0];
    BixNumCompiler bnc0(pb);
    PabloAST * ZeroCCC = pb.createAnd(bnc0.EQ(CCC, 0), u8index);
    PabloAST * NonZeroCCC = pb.createAnd(pb.createNot(ZeroCCC), u8index);
    auto nested = pb.createScope();
    Var * first_and_last_char = pb.createVar("first_and_last_char", pb.createZeroes());
    Var * violation_mark = pb.createVar("violation_mark", pb.createZeroes());
    pb.createIf(NonZeroCCC, nested);
    //
    PabloAST * startOfFile = nested.createNot(nested.createAdvance(nested.createOnes(), 1));
    PabloAST * firstChar = nested.createScanTo(startOfFile, u8index);
    BixNumCompiler bnc(nested);
    BixNum CCC_advanced(mCCC_basis_size);
    BixNum CCC_ahead(mCCC_basis_size);
    for (unsigned i = 0; i < mCCC_basis_size; i++) {
        CCC_advanced[i] = nested.createIndexedAdvance(CCC[i], u8index, 1);
    }
    PabloAST * lastChar = nested.createLookahead(EOF_mark, 1);
    nested.createAssign(first_and_last_char, nested.createOr(firstChar, lastChar));
    PabloAST * violation = nested.createAnd(bnc.UGT(CCC_advanced, CCC), NonZeroCCC);
    PabloAST * infill = nested.createInFile(nested.createNot(u8index));
    PabloAST * seqEnd = nested.createOr(ZeroCCC, lastChar);
    PabloAST * seqFill = nested.createOr(infill, nested.createNot(seqEnd));
    PabloAST * violation_end = nested.createAnd(nested.createMatchStar(violation, seqFill), seqEnd);
    nested.createAssign(violation_mark, violation_end);
    PabloAST * seqMarks = pb.createOr(ZeroCCC, first_and_last_char);
    pb.createAssign(pb.createExtract(getOutputStreamVar("CCC_SeqMarks"), pb.getInteger(0)), seqMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("CCC_Violation"), pb.getInteger(0)), violation_mark);
}

class NFD_Narrow_Focus : public pablo::PabloKernel {
public:
    NFD_Narrow_Focus(LLVMTypeSystemInterface & ts,
                     StreamSet * u8index, StreamSet * DT_Can, StreamSet * ViolationStarts, StreamSet * CCC_SeqMarks,
                     StreamSet * mayChangeOnNFD);
protected:
    void generatePabloMethod() override;
};

NFD_Narrow_Focus::NFD_Narrow_Focus(LLVMTypeSystemInterface & ts,
                                   StreamSet * u8index, StreamSet * DT_Can, StreamSet * ViolationStarts, StreamSet * CCC_SeqMarks,
                                   StreamSet * mayChangeOnNFD)
: PabloKernel(ts, "NFD_Narrow_Focus",
// inputs
{Binding{"u8index", u8index}, Binding{"DT_Can", DT_Can}, Binding{"ViolationStarts", ViolationStarts}, Binding{"CCC_SeqMarks", CCC_SeqMarks}},
// output
{Binding{"mayChangeOnNFD", mayChangeOnNFD}}) {
}

void NFD_Narrow_Focus::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    PabloAST * DT_Can = getInputStreamSet("DT_Can")[0];
    PabloAST * ViolationStarts = getInputStreamSet("ViolationStarts")[0];
    PabloAST * CCC_SeqMarks = getInputStreamSet("CCC_SeqMarks")[0];
    PabloAST * WorkSeqStarts = pb.createOr(DT_Can, ViolationStarts);
    PabloAST * interior = pb.createMatchStar(pb.createAdvance(WorkSeqStarts, 1), pb.createNot(CCC_SeqMarks));
    PabloAST * work_items = pb.createOr(WorkSeqStarts, pb.createAnd(interior, u8index));
    pb.createAssign(pb.createExtract(getOutputStreamVar("mayChangeOnNFD"), pb.getInteger(0)), work_items);
}

//
//  NFD_Focus identifies UTF0-8 stream positions that potentially play
//  a role in either expansion or reordering during NFD decomposition.
//  These positions include all characters whose Unicode decomposition
//  type is Canonical, all characters whose combining class is nonzero
//  (and hence reorderable), as well as any nonreorderable starter following
//  either of these  (to ensure that such sequences
//  are separated).  NFD_Focus relies on inputs u8index marking the last
//  position of each UTF-8 byte sequence,  DT_Can marking the last position
//  of each character having dt=Can, and CCC_0 marking the initial position
//  of each character that is not reorderable.
//
class NFD_Focus : public pablo::PabloKernel {
public:
    NFD_Focus(LLVMTypeSystemInterface & ts, StreamSet * u8index, StreamSet * DT_Can, StreamSet * CCC_0, StreamSet * mayChangeOnNFD);
protected:
    void generatePabloMethod() override;
};

NFD_Focus::NFD_Focus(LLVMTypeSystemInterface & ts, StreamSet * u8index, StreamSet * DT_Can, StreamSet * CCC_0, StreamSet * mayChangeOnNFD)
: PabloKernel(ts, "NFD_Focus",
// inputs
{Binding{"u8index", u8index}, Binding{"DT_Can", DT_Can}, Binding{"CCC_0", CCC_0}},
// output
{Binding{"mayChangeOnNFD", mayChangeOnNFD}}) {
}

void NFD_Focus::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    PabloAST * DT_Can = getInputStreamSet("DT_Can")[0];
    PabloAST * CCC_0_final = getInputStreamSet("CCC_0")[0];
    PabloAST * CCC_Nonzero_final = pb.createAnd(u8index, pb.createNot(CCC_0_final));
    PabloAST * CanOrNZ_follow = pb.createAdvanceThenScanThru(pb.createOr(DT_Can, CCC_Nonzero_final), pb.createNot(u8index));
    PabloAST * focus = pb.createOr3(CanOrNZ_follow, DT_Can, CCC_Nonzero_final);
    pb.createAssign(pb.createExtract(getOutputStreamVar("mayChangeOnNFD"), pb.getInteger(0)), focus);
}


#define SHOW_STREAM(name) if (codegen::EnableIllustrator) mPB.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) mPB.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) mPB.captureByteData(#name, name)


StreamSet * NFD_PipelineBuilder::NFD_U21_Pipeline(StreamSet * U21_Basis) {

    // Now we have a Unicode-indexed representation of all significant
    // sequences for NFD processing.
    UCD::NFD_Engine NFD_engine(UCD::DecompositionOptions::NFD);
    // Expand to make room for decompositions.
    auto insert_ccs = NFD_engine.UnicodeInsertLengthBixNumSets();
    auto bit_translate_ccs = NFD_engine.UnicodeBitTransformSets();

    StreamSet * NFD_Basis = U21_CharToShortStringPipeline(mPB, insert_ccs, bit_translate_ccs, U21_Basis);

    //  The Hangul decomposition algorithm calculates replacements for LV and
    //  LVT characters using calculations based on three 5-bit indexes for
    //  the L, V and T characters.
    StreamSet * const LIndexBixNum = mPB.CreateStreamSet(L_Index_bits);
    StreamSet * const VIndexBixNum = mPB.CreateStreamSet(V_Index_bits);
    StreamSet * const TIndexBixNum = mPB.CreateStreamSet(T_Index_bits);
    mPB.CreateKernelCall<LVT_Indexes>(NFD_Basis, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    // Given the L, V and T indexes, the replacements for LV and LVT characters
    // can be calculated to determine the correct 21-bit representations at
    // <L, V> and <L, V, T> positions.
    StreamSet * const Hangul_NFD_Basis = mPB.CreateStreamSet(21, 1);
    mPB.CreateKernelCall<LVT2NFD>(NFD_Basis, LIndexBixNum, VIndexBixNum, TIndexBixNum, Hangul_NFD_Basis);
    SHOW_BIXNUM(Hangul_NFD_Basis);

    NFD_Basis = Hangul_NFD_Basis;

    UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
    StreamSet * const CCC_Basis = mPB.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    mPB.CreateKernelCall<UnicodePropertyBasis>(enumObj, NFD_Basis, CCC_Basis);
    SHOW_BIXNUM(CCC_Basis);

    StreamSet * const CCC_NonZero = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<bixnum::NEQ_immediate>(CCC_Basis, 0, CCC_NonZero);
    SHOW_STREAM(CCC_NonZero);

    StreamSets ToSort = {CCC_Basis, NFD_Basis};

    StreamSets SortResults = BitonicSortRuns(mPB, 8, CCC_NonZero, ToSort);
    SHOW_BIXNUM(SortResults[0]);
    SHOW_BIXNUM(SortResults[1]);

    return SortResults[1];
}

StreamSet * NFD_PipelineBuilder::NFKD_U21_Pipeline(StreamSet * U21_Basis) {

    // Now we have a Unicode-indexed representation of all significant
    // sequences for NFD processing.
    UCD::NFD_Engine NFD_engine(UCD::DecompositionOptions::NFKD);
    // Expand to make room for decompositions.
    auto insert_ccs = NFD_engine.UnicodeInsertLengthBixNumSets();
    auto bit_translate_ccs = NFD_engine.UnicodeBitTransformSets();

    StreamSet * NFD_Basis = U21_CharToShortStringPipeline(mPB, insert_ccs, bit_translate_ccs, U21_Basis);

    //  The Hangul decomposition algorithm calculates replacements for LV and
    //  LVT characters using calculations based on three 5-bit indexes for
    //  the L, V and T characters.
    StreamSet * const LIndexBixNum = mPB.CreateStreamSet(L_Index_bits);
    StreamSet * const VIndexBixNum = mPB.CreateStreamSet(V_Index_bits);
    StreamSet * const TIndexBixNum = mPB.CreateStreamSet(T_Index_bits);
    mPB.CreateKernelCall<LVT_Indexes>(NFD_Basis, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    // Given the L, V and T indexes, the replacements for LV and LVT characters
    // can be calculated to determine the correct 21-bit representations at
    // <L, V> and <L, V, T> positions.
    StreamSet * const Hangul_NFD_Basis = mPB.CreateStreamSet(21, 1);
    mPB.CreateKernelCall<LVT2NFD>(NFD_Basis, LIndexBixNum, VIndexBixNum, TIndexBixNum, Hangul_NFD_Basis);
    SHOW_BIXNUM(Hangul_NFD_Basis);

    NFD_Basis = Hangul_NFD_Basis;

    UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
    StreamSet * const CCC_Basis = mPB.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    mPB.CreateKernelCall<UnicodePropertyBasis>(enumObj, NFD_Basis, CCC_Basis);
    SHOW_BIXNUM(CCC_Basis);

    StreamSet * const CCC_NonZero = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<bixnum::NEQ_immediate>(CCC_Basis, 0, CCC_NonZero);
    SHOW_STREAM(CCC_NonZero);

    StreamSets ToSort = {CCC_Basis, NFD_Basis};

    StreamSets SortResults = BitonicSortRuns(mPB, 8, CCC_NonZero, ToSort);
    SHOW_BIXNUM(SortResults[0]);
    SHOW_BIXNUM(SortResults[1]);

    return SortResults[1];
}

void NFD_PipelineBuilder::DetermineNFD_WorkItems(StreamSet * U8_Basis, StreamSet * u8index, StreamSet * NFD_WorkItems) {
    re::PropertyExpression * dtCanProp = re::makePropertyExpression("dt", "Can");
    dtCanProp = cast<re::PropertyExpression>(UCD::linkAndResolve(dtCanProp));
    StreamSet * DT_Can = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<UnicodePropertyKernelBuilder>(dtCanProp, U8_Basis, DT_Can);
    SHOW_STREAM(DT_Can);

    re::PropertyExpression * CCC0_Prop = re::makePropertyExpression("CCC", "NR");
    CCC0_Prop = cast<re::PropertyExpression>(UCD::linkAndResolve(CCC0_Prop));
    StreamSet * const CCC_0 = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Prop, U8_Basis, CCC_0);//, BitMovementMode::LookAhead);
    SHOW_STREAM(CCC_0);

    mPB.CreateKernelCall<NFD_Focus>(u8index, DT_Can, CCC_0, NFD_WorkItems);
    SHOW_STREAM(NFD_WorkItems);
}

void NFD_PipelineBuilder::NFD_FilterStage(StreamSet * BasisBits, StreamSet * WorkSelectionMask, StreamSet * FinalWorkPlacementMask, StreamSet * WorkingBasis) {

    StreamSet * const u8index = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * NFD_WorkItems = mPB.CreateStreamSet(1, 1);
    DetermineNFD_WorkItems(BasisBits, u8index, NFD_WorkItems);

    mPB.CreateKernelCall<U8Spans>(NFD_WorkItems, u8index, WorkSelectionMask);
    SHOW_STREAM(WorkSelectionMask);

    ComputeWorkPlacement(mPB, NFD_Data.UTF8_Insertion_BixNumCCs(), NFD_Data.UTF8_Deletion_BixNumCCs(), BasisBits, WorkSelectionMask, FinalWorkPlacementMask);
    SHOW_STREAM(FinalWorkPlacementMask);

    FilterByMask(mPB, WorkSelectionMask, BasisBits, WorkingBasis);
    SHOW_BIXNUM(WorkingBasis);
}

void NFD_PipelineBuilder::NFD_U8_Pipeline(StreamSet * WorkingBasis, StreamSet * TransformedBasis) {

    StreamSet * const U21_u8indexed = mPB.CreateStreamSet(21, 1);
    mPB.CreateKernelCall<UTF8_Decoder>(WorkingBasis, U21_u8indexed);
    SHOW_BIXNUM(U21_u8indexed);

    StreamSet * const WorkingU8index = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<UTF8_index>(WorkingBasis, WorkingU8index);
    SHOW_STREAM(WorkingU8index);

    StreamSet * const U21_focus = mPB.CreateStreamSet(21, 1);
    FilterByMask(mPB, WorkingU8index, U21_u8indexed, U21_focus);
    SHOW_BIXNUM(U21_focus);

    StreamSet * NFD_U21_Results = NFD_U21_Pipeline(U21_focus);

    U21_to_UTF8(mPB, NFD_U21_Results, TransformedBasis);
    SHOW_BIXNUM(TransformedBasis);
}

void NFD_PipelineBuilder::NFKD_U8_Pipeline(StreamSet * WorkingBasis, StreamSet * TransformedBasis) {

    StreamSet * const U21_u8indexed = mPB.CreateStreamSet(21, 1);
    mPB.CreateKernelCall<UTF8_Decoder>(WorkingBasis, U21_u8indexed);
    SHOW_BIXNUM(U21_u8indexed);

    StreamSet * const WorkingU8index = mPB.CreateStreamSet(1, 1);
    mPB.CreateKernelCall<UTF8_index>(WorkingBasis, WorkingU8index);
    SHOW_STREAM(WorkingU8index);

    StreamSet * const U21_focus = mPB.CreateStreamSet(21, 1);
    FilterByMask(mPB, WorkingU8index, U21_u8indexed, U21_focus);
    SHOW_BIXNUM(U21_focus);

    StreamSet * NFD_U21_Results = NFKD_U21_Pipeline(U21_focus);

    U21_to_UTF8(mPB, NFD_U21_Results, TransformedBasis);
    SHOW_BIXNUM(TransformedBasis);
}
