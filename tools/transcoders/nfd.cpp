/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/codegenstate.h>
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <pablo/bixnum/bixnum.h>
#include <grep/grep_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/sorting.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/unicode/resolve_properties.h>
#include <string>
#include <toolchain/fileutil.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/algo/decomposition.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>
#include <unicode/utf/utf_encoder.h>
#include <unicode/utf/transchar.h>
#include <codecvt>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decompositon Options", "Decompositon Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));
static cl::opt<bool> LateU21("LateU21", cl::desc("Delay conversion to Unicode 21-bit values until after filtering"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteMerging("ByteMerging", cl::desc("Use byte stream merging of transformed and unmodified data"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteReplace("ByteReplace", cl::desc("Perform byte merging using the ByteReplaceByMask kernel"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> SeparatedPipelineStages("SeparatedPipelineStages", cl::desc("Use multiple separated pipeline stages"), cl::init(false), cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

const UCD::codepoint_t Hangul_SBase = 0xAC00;
const UCD::codepoint_t Hangul_LBase = 0x1100;
const UCD::codepoint_t Hangul_VBase = 0x1161;
const UCD::codepoint_t Hangul_TBase = 0x11A7;
const unsigned Hangul_LCount = 19;
const unsigned Hangul_VCount = 21;
const unsigned Hangul_TCount = 28;
const unsigned Hangul_NCount = 588;
const unsigned Hangul_SCount = 11172;

const unsigned S_Index_bits = 14;
const unsigned L_Index_bits = 5;
const unsigned V_Index_bits = 5;
const unsigned T_Index_bits = 5;

class NFD_BixData : public UCD::NFD_Engine {
public:
    NFD_BixData(UTF_Encoder & u8_encoder);
    std::vector<re::CC *> NFD_Insertion_BixNumCCs();
    std::vector<re::CC *> UTF8_Insertion_BixNumCCs();
    std::vector<re::CC *> UTF8_Deletion_BixNumCCs();
    unicode::BitTranslationSets NFD_1st_BitXorCCs();
    unicode::BitTranslationSets NFD_2nd_BitCCs();
    unicode::BitTranslationSets NFD_3rd_BitCCs();
    unicode::BitTranslationSets NFD_4th_BitCCs();
private:
    UTF_Encoder & mU8_encoder;
    std::unordered_map<codepoint_t, unsigned> mNFD_expansion;
    std::unordered_map<codepoint_t, unsigned> mUTF8_expansion;
    std::unordered_map<codepoint_t, unsigned> mUTF8_deletion;
    unicode::TranslationMap mNFD_CharMap[4];
    UCD::UnicodeSet mHangul_Precomposed_LV;
    UCD::UnicodeSet mHangul_Precomposed_LVT;
};

NFD_BixData::NFD_BixData(UTF_Encoder & u8_encoder) :
    UCD::NFD_Engine(UCD::DecompositionOptions::NFD), mU8_encoder(u8_encoder) {
    auto cps = decompMappingObj->GetExplicitCps();
    for (auto cp : cps) {
        std::u32string decomp;
        NFD_append1(decomp, cp);
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
    UCD::codepoint_t Max_Hangul_Precomposed = Hangul_SBase + Hangul_SCount - 1;
    for (UCD::codepoint_t cp = Hangul_SBase; cp <= Max_Hangul_Precomposed; cp += Hangul_TCount) {
        mHangul_Precomposed_LV.insert(cp);
    }
    UCD::UnicodeSet Hangul_Precomposed(Hangul_SBase, Max_Hangul_Precomposed);
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
    std::vector<re::CC *> all_CCs(NFD1.size() + NFD2.size() + NFD3.size() + NFD4.size());

    for (unsigned i = 0; i < NFD1.size(); i++) {
        Var * v = pb.createVar("NFD1_bit" + std::to_string(i), pb.createZeroes());
        NFD1_Vars[i] = v;
        all_targets[i] = v;
        all_CCs[i] = re::makeCC(NFD1[i], &cc::Unicode);
    }
    unsigned base = NFD1.size();
    for (unsigned i = 0; i < NFD2.size(); i++) {
        Var * v = pb.createVar("NFD2_bit" + std::to_string(i), pb.createZeroes());
        NFD2_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = re::makeCC(NFD2[i], &cc::Unicode);
    }
    base = base + NFD2.size();
    for (unsigned i = 0; i < NFD3.size(); i++) {
        Var * v = pb.createVar("NFD3_bit" + std::to_string(i), pb.createZeroes());
        NFD3_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = re::makeCC(NFD3[i], &cc::Unicode);
    }
    base = base + NFD3.size();
    for (unsigned i = 0; i < NFD4.size(); i++) {
        Var * v = pb.createVar("NFD4_bit" + std::to_string(i), pb.createZeroes());
        NFD4_Vars[i] = v;
        all_targets[base + i] = v;
        all_CCs[base + i] = re::makeCC(NFD4[i], &cc::Unicode);
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
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul_SBase), bnc0.ULT(basis, Hangul_SBase + Hangul_SCount));
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

    BixNum S_Index = bnc.Select(Hangul_precomposed, bnc.SubModular(basis, Hangul_SBase), ZeroBasis);
    S_Index = bnc.Truncate(S_Index, S_Index_bits);
    BixNum LV_index;
    BixNum T_index;
    bnc.Div(S_Index, Hangul_TCount, LV_index, T_index);
    LV_index = bnc.Truncate(LV_index, L_Index_bits + V_Index_bits);
    BixNum L_index;
    BixNum V_index;
    bnc.Div(LV_index, Hangul_VCount, L_index, V_index);
    
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
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul_SBase), bnc0.ULT(basis, Hangul_SBase + Hangul_SCount));
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
    LPart = bnc.AddModular(LPart, Hangul_LBase);
    // The V Part, when it exists, is one position after the opening L Part.
    PabloAST * V_position = nested.createAdvance(Hangul_precomposed, 1);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_index[i] = nested.createAdvance(V_index[i], 1);
    }
    BixNum VPart = bnc.ZeroExtend(V_index, 21);
    VPart = bnc.AddModular(VPart, Hangul_VBase);
    // The T Part, if it exists is two positions after the opening
    // L Part.  A T Part only exists for LVT initials (T != 0).
    PabloAST * T_position = nested.createAdvance(Hangul_precomposed, 2);
    for (unsigned i = 0; i < T_Index_bits; i++) {
        T_index[i] = nested.createAdvance(T_index[i], 2);
    }
    T_position = nested.createAnd(T_position, bnc.NEQ(T_index, 0));
    BixNum TPart = bnc.ZeroExtend(T_index, 21);
    TPart = bnc.AddModular(TPart, Hangul_TBase);
    for (unsigned i = 0; i < 21; i++) {
        PabloAST * bit = nested.createSel(Hangul_precomposed, LPart[i], basis[i]);
        bit = nested.createSel(V_position, VPart[i], bit);
        bit = nested.createSel(T_position, TPart[i], bit);
        nested.createAssign(basisVar[i], bit);
    }
    writeOutputStreamSet("NFD_Basis", basisVar);
}

//
//   CCC_Check computes two bitstreams:
//   (1)  Marking all positions with violations of combining class ordering,
//        marked at the first position of a misordered pair.
//   (2)  A streamset marking the beginning and end of a sequence consisting
//        of a non-combining character (CCC = 0) followed by two or more
//        combining characters (note that a sequence having a single combining
//        character can never be out of order).
//
class CCC_Check : public pablo::PabloKernel {
public:
    CCC_Check(LLVMTypeSystemInterface & ts, StreamSet * CCC_Basis, StreamSet * CCC_SeqMarks, StreamSet * CCC_Violation);
protected:
    void generatePabloMethod() override;
private:
    unsigned mCCC_basis_size;
};

CCC_Check::CCC_Check (LLVMTypeSystemInterface & ts, StreamSet * CCC_Basis, StreamSet * CCC_SeqMarks, StreamSet * CCC_Violation)
: PabloKernel(ts, "CCC_Check",
// inputs
{Binding{"CCC_Basis", CCC_Basis, FixedRate(1), LookAhead(2)}},
// output
{Binding{"CCC_SeqMarks", CCC_SeqMarks}, Binding{"CCC_Violation", CCC_Violation}}),
mCCC_basis_size(CCC_Basis->getNumElements()) {
}

void CCC_Check::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNum CCC = getInputStreamSet("CCC_Basis");
    BixNum CCC_ahead(mCCC_basis_size);
    BixNum CCC_ahead2(mCCC_basis_size);
    for (unsigned i = 0; i < mCCC_basis_size; i++) {
        CCC_ahead[i] = pb.createLookahead(CCC[i], 1);
        CCC_ahead2[i] = pb.createLookahead(CCC[i], 2);
    }
    BixNumCompiler bnc(pb);
    PabloAST * ZeroCCC = bnc.EQ(CCC, 0);
    PabloAST * ZeroCCC_ahead = bnc.EQ(CCC_ahead, 0);
    PabloAST * violation = pb.createAnd(bnc.UGT(CCC, CCC_ahead), pb.createNot(ZeroCCC_ahead));
    PabloAST * seqEnd = pb.createAnd(ZeroCCC_ahead, pb.createNot(ZeroCCC));
    PabloAST * violationInSeq = pb.createAnd(pb.createMatchStar(violation, pb.createNot(ZeroCCC)), seqEnd);
    PabloAST * seqStart = pb.createAnd3(ZeroCCC, pb.createNot(ZeroCCC_ahead), bnc.NEQ(CCC_ahead2, 0));
    pb.createAssign(pb.createExtract(getOutputStreamVar("CCC_SeqMarks"), pb.getInteger(0)), pb.createOr(seqStart, seqEnd));
    pb.createAssign(pb.createExtract(getOutputStreamVar("CCC_Violation"), pb.getInteger(0)), violationInSeq);
}

class CCC_Violation_Sequence : public pablo::PabloKernel {
public:
    CCC_Violation_Sequence(LLVMTypeSystemInterface & ts, StreamSet * ViolationStarts, StreamSet * CCC_SeqMarks, StreamSet * Violation_Seq);
protected:
    void generatePabloMethod() override;
};

CCC_Violation_Sequence::CCC_Violation_Sequence(LLVMTypeSystemInterface & ts, StreamSet * ViolationStarts, StreamSet * CCC_SeqMarks, StreamSet * Violation_Seq)
: PabloKernel(ts, "CCC_Violation_Sequence",
// inputs
{Binding{"ViolationStarts", ViolationStarts}, Binding{"CCC_SeqMarks", CCC_SeqMarks}},
// output
{Binding{"Violation_Seq", Violation_Seq}}) {
}

void CCC_Violation_Sequence::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * ViolationStarts = getInputStreamSet("ViolationStarts")[0];
    PabloAST * CCC_SeqMarks = getInputStreamSet("CCC_SeqMarks")[0];
    PabloAST * interior = pb.createMatchStar(pb.createAdvance(ViolationStarts, 1), pb.createNot(CCC_SeqMarks));
    //PabloAST * Violation_Seq = pb.createOr(ViolationStarts, interior);
    pb.createAssign(pb.createExtract(getOutputStreamVar("Violation_Seq"), pb.getInteger(0)), interior);
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

class NFD_DepositMask : public pablo::PabloKernel {
public:
    NFD_DepositMask(LLVMTypeSystemInterface & ts, StreamSet * U8_SpreadMask, StreamSet * SelectedFinalPositionMask, StreamSet * DepositMask);
protected:
    void generatePabloMethod() override;
};

NFD_DepositMask::NFD_DepositMask(LLVMTypeSystemInterface & ts, StreamSet * U8_SpreadMask, StreamSet * SelectedFinalPositionMask, StreamSet * DepositMask)
: PabloKernel(ts, "NFD_DepositMask",
// inputs
{Binding{"U8_SpreadMask", U8_SpreadMask}, Binding{"SelectedFinalPositionMask", SelectedFinalPositionMask}},
// output
{Binding{"DepositMask", DepositMask}}) {
}

void NFD_DepositMask::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * U8_SpreadMask = getInputStreamSet("U8_SpreadMask")[0];
    PabloAST * SelectedFinalPositionMask = getInputStreamSet("SelectedFinalPositionMask")[0];
    PabloAST * DepositMask = pb.createOr(SelectedFinalPositionMask, pb.createNot(U8_SpreadMask));
    pb.createAssign(pb.createExtract(getOutputStreamVar("DepositMask"), pb.getInteger(0)), DepositMask);
}

class FinalSelection : public pablo::PabloKernel {
public:
    FinalSelection(LLVMTypeSystemInterface & ts, StreamSet * SelectMask, StreamSet * Source1, StreamSet * Source2, StreamSet * Output);
protected:
    void generatePabloMethod() override;
};

FinalSelection::FinalSelection(LLVMTypeSystemInterface & ts, StreamSet * SelectMask, StreamSet * Source1, StreamSet * Source2, StreamSet * Output)
: PabloKernel(ts, "FinalSelection" + Source1->shapeString(),
// inputs
{Binding{"SelectMask", SelectMask}, Binding{"Source1", Source1}, Binding{"Source2", Source2}},
// output
{Binding{"Output", Output}}) {
}

class Invert : public PabloKernel {
public:
    Invert(LLVMTypeSystemInterface & ts, StreamSet * mask, StreamSet * inverted)
        : PabloKernel(ts, "Invert",
                      {Binding{"mask", mask}},
                      {Binding{"inverted", inverted}}) {}
protected:
    void generatePabloMethod() override;
};

void Invert::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("mask")[0];
    PabloAST * inverted = pb.createInFile(pb.createNot(mask));
    Var * outVar = getOutputStreamVar("inverted");
    pb.createAssign(pb.createExtract(outVar, pb.getInteger(0)), inverted);
}

class OrCombine : public PabloKernel {
public:
    OrCombine(LLVMTypeSystemInterface & ts, StreamSet * s1, StreamSet * s2, StreamSet * combined)
        : PabloKernel(ts, "OrCombine",
                      {Binding{"s1", s1}, Binding{"s2", s2}},
                      {Binding{"combined", combined}}) {}
protected:
    void generatePabloMethod() override;
};

void OrCombine::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * s1 = getInputStreamSet("s1")[0];
    PabloAST * s2 = getInputStreamSet("s2")[0];
    PabloAST * combined = pb.createOr(s1, s2);
    Var * outVar = getOutputStreamVar("combined");
    pb.createAssign(pb.createExtract(outVar, pb.getInteger(0)), combined);
}

void FinalSelection::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * SelectMask = getInputStreamSet("SelectMask")[0];
    std::vector<PabloAST *> Source1 = getInputStreamSet("Source1");
    std::vector<PabloAST *> Source2 = getInputStreamSet("Source2");
    std::vector<PabloAST *> Output(Source1.size());
    for (unsigned i = 0; i < Source1.size(); i++) {
        Output[i] = pb.createSel(SelectMask, Source1[i], Source2[i]);
    }
    writeOutputStreamSet("Output", Output);
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


StreamSet * NFD_U21_Pipeline(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * U21_Basis) {
    // Now we have a Unicode-indexed representation of all significant
    // sequences for NFD processing.
    // Expand to make room for decompositions.
    auto insert_ccs = NFD_Data.NFD_Insertion_BixNumCCs();
    StreamSet * const U21_Insertion_BixNum = P.CreateStreamSet(insert_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(insert_ccs, U21_Basis, U21_Insertion_BixNum);
    SHOW_BIXNUM(U21_Insertion_BixNum);

    StreamSet * const U21_SpreadMask = InsertionSpreadMask(P, U21_Insertion_BixNum, kernel::InsertPosition::After);
    SHOW_STREAM(U21_SpreadMask);

    StreamSet * const U21_ExpandedBasis = P.CreateStreamSet(21, 1);
    SpreadByMask(P, U21_SpreadMask, U21_Basis, U21_ExpandedBasis);
    SHOW_BIXNUM(U21_ExpandedBasis);

    //  The Hangul decomposition algorithm calculates replacements for LV and
    //  LVT characters using calculations based on three 5-bit indexes for
    //  the L, V and T characters.
    StreamSet * const LIndexBixNum = P.CreateStreamSet(L_Index_bits);
    StreamSet * const VIndexBixNum = P.CreateStreamSet(V_Index_bits);
    StreamSet * const TIndexBixNum = P.CreateStreamSet(T_Index_bits);
    P.CreateKernelCall<LVT_Indexes>(U21_ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    // Given the L, V and T indexes, the replacements for LV and LVT characters
    // can be calculated to determine the correct 21-bit representations at
    // <L, V> and <L, V, T> positions.
    StreamSet * const Hangul_NFD_Basis = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<LVT2NFD>(U21_ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum, Hangul_NFD_Basis);
    SHOW_BIXNUM(Hangul_NFD_Basis);

    StreamSet * const NFD_Basis = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<NFD_Translation>(NFD_Data, Hangul_NFD_Basis, NFD_Basis);
    SHOW_BIXNUM(NFD_Basis);

    UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
    StreamSet * const CCC_Basis = P.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    P.CreateKernelCall<UnicodePropertyBasis>(enumObj, NFD_Basis, CCC_Basis);
    SHOW_BIXNUM(CCC_Basis);

    StreamSet * const CCC_NonZero = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<bixnum::NEQ_immediate>(CCC_Basis, 0, CCC_NonZero);
    SHOW_STREAM(CCC_NonZero);

    StreamSets ToSort = {CCC_Basis, NFD_Basis};

    StreamSets SortResults = BitonicSortRuns(P, 8, CCC_NonZero, ToSort);
    SHOW_BIXNUM(SortResults[0]);
    SHOW_BIXNUM(SortResults[1]);

    return SortResults[1];
}

StreamSet * DetermineNFD_WorkItems(PipelineBuilder & P, StreamSet * U8_Basis, StreamSet * u8index) {
    re::RE * dtCanProp = re::makePropertyExpression("dt", "Can");
    dtCanProp = UCD::linkAndResolve(dtCanProp);
    re::Name * dtCanName = re::makeName("dt", "Can");
    dtCanName->setDefinition(dtCanProp);
    StreamSet * const DT_Can = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(dtCanName, U8_Basis, DT_Can);
    SHOW_STREAM(DT_Can);

    re::RE * CCC0_Prop = re::makePropertyExpression("CCC", "NR");
    CCC0_Prop = UCD::linkAndResolve(CCC0_Prop);
    re::Name * CCC0_Name = re::makeName("CCC", "NR");
    CCC0_Name->setDefinition(CCC0_Prop);
    StreamSet * const CCC_0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Name, U8_Basis, CCC_0);//, BitMovementMode::LookAhead);
    SHOW_STREAM(CCC_0);

    StreamSet * const NFD_WorkItems = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<NFD_Focus>(u8index, DT_Can, CCC_0, NFD_WorkItems);
    SHOW_STREAM(NFD_WorkItems);

    return NFD_WorkItems;
}

StreamSet * WorkPlacementMask(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * U8_Basis, StreamSet * SelectionMask) {
    auto u8_insert_ccs = NFD_Data.UTF8_Insertion_BixNumCCs();
    StreamSet * const U8_Insertion_BixNum = P.CreateStreamSet(u8_insert_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(u8_insert_ccs, U8_Basis, U8_Insertion_BixNum);
    SHOW_BIXNUM(U8_Insertion_BixNum);

    auto u8_deletion_ccs = NFD_Data.UTF8_Deletion_BixNumCCs();
    StreamSet * const U8_Deletion_BixNum = P.CreateStreamSet(u8_deletion_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(u8_deletion_ccs, U8_Basis, U8_Deletion_BixNum);
    SHOW_BIXNUM(U8_Deletion_BixNum);

    StreamSet * const U8_FilterMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<CreateU8_FilterMask>(U8_Deletion_BixNum, U8_FilterMask);
    SHOW_STREAM(U8_FilterMask);

    StreamSet * const U8_SpreadMask = InsertionSpreadMask(P, U8_Insertion_BixNum, kernel::InsertPosition::After);
    SHOW_STREAM(U8_SpreadMask);

    StreamSet * const ExpandedSpaceMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Invert>(U8_SpreadMask, ExpandedSpaceMask);

    StreamSet * const ExpandedFilterMask = P.CreateStreamSet(1, 1);
    SpreadByMask(P, U8_SpreadMask, U8_FilterMask, ExpandedFilterMask);
    SHOW_STREAM(ExpandedFilterMask);

    StreamSet * const U8_PostSpreadFilterMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<OrCombine>(ExpandedFilterMask, ExpandedSpaceMask, U8_PostSpreadFilterMask);
    SHOW_STREAM(U8_PostSpreadFilterMask);

    StreamSet * const WorkSpreadMask = P.CreateStreamSet(1, 1);
    SpreadByMask(P, U8_SpreadMask, SelectionMask, WorkSpreadMask);
    SHOW_STREAM(WorkSpreadMask);

    StreamSet * const WorkExpansionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<OrCombine>(WorkSpreadMask, ExpandedSpaceMask, WorkExpansionMask);
    SHOW_STREAM(WorkExpansionMask);

    StreamSet * const FinalWorkPlacementMask = P.CreateStreamSet(1, 1);
    FilterByMask(P, U8_PostSpreadFilterMask, WorkExpansionMask, FinalWorkPlacementMask);
    SHOW_STREAM(FinalWorkPlacementMask);
    return FinalWorkPlacementMask;
}
//#define STAGE1_TO_STDOUT
#ifdef STAGE1_TO_STDOUT
typedef void (*Stage1FunctionType)(char *, size_t);
#else
typedef void (*Stage1FunctionType)(kernel::StreamSetPtr &, char *, size_t);
#endif

Stage1FunctionType generate_stage1_pipeline(CPUDriver & driver) {
    
    auto P = CreatePipeline(driver,
#ifndef STAGE1_TO_STDOUT
                            Output<streamset_t>("WorkingBytes", 1, 8),
#endif
                            Input<char*>{"buffer"},
                            Input<size_t>{"length"}
                            );
    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);
    SHOW_BYTES(ByteStream);

    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    
    StreamSet * const u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);
    
    StreamSet * NFD_WorkItems = DetermineNFD_WorkItems(P, BasisBits, u8index);
    
    StreamSet * const WorkSelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<U8Spans>(NFD_WorkItems, u8index, WorkSelectionMask);
    SHOW_STREAM(WorkSelectionMask);
    
    StreamSet * const WorkingBasis = P.CreateStreamSet(8, 1);
    FilterByMask(P, WorkSelectionMask, BasisBits, WorkingBasis);
    SHOW_BIXNUM(WorkingBasis);
    
#ifdef STAGE1_TO_STDOUT
    StreamSet * const WorkingBytes = P.CreateStreamSet(1, 8);
#else
    StreamSet * const WorkingBytes = P.getOutputStreamSet("WorkingBytes");
#endif
    P.CreateKernelCall<P2SKernel>(WorkingBasis, WorkingBytes);
#ifdef STAGE1_TO_STDOUT
    P.CreateKernelCall<StdOutKernel>(WorkingBytes);
#endif
    return P.compile();
}

#define STAGE2_TO_STDOUT
#ifdef STAGE2_TO_STDOUT
typedef void (*Stage2FunctionType)(const kernel::StreamSetPtr & source_buf);
#else
typedef void (*Stage2FunctionType)(const StreamSetPtr & source_buf, StreamSetPtr & NFD_buf);
#endif


typedef void (*Stage2FunctionType)(const kernel::StreamSetPtr & source_buf);
//typedef void (*Stage2FunctionType)(const StreamSetPtr & source_buf, StreamSetPtr & NFD_buf);

Stage2FunctionType generate_stage2_pipeline(CPUDriver & driver) {
    
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkingBytes", 1, 8)
#ifndef STAGE2_TO_STDOUT
                            ,Output<streamset_t>("NFD_Bytes",  1, 8)
#endif
                            );
    
    StreamSet * WorkingData = P.getInputStreamSet("WorkingBytes");
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(WorkingData, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * const U21_u8indexed = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);

    UTF_Encoder U8_Encoder(8);
    NFD_BixData NFD_Data(U8_Encoder);
    
    StreamSet * NFD_U21_Results = NFD_U21_Pipeline(P, NFD_Data, U21_u8indexed);
    
    StreamSet * const NFD_Basis = P.CreateStreamSet(8, 1);
    U21_to_UTF8(P, NFD_U21_Results, NFD_Basis);
    SHOW_BIXNUM(NFD_Basis);
    
#ifdef STAGE2_TO_STDOUT
    StreamSet * const NFD_Bytes = P.CreateStreamSet(1, 8);
#else
    StreamSet * const NFD_Bytes = P.getOutputStreamSet("NFD_Bytes");
#endif
    P.CreateKernelCall<P2SKernel>(NFD_Basis, NFD_Bytes);
#ifdef STAGE2_TO_STDOUT
    P.CreateKernelCall<StdOutKernel>(NFD_Bytes);
#endif
    return P.compile();
}

/*
typedef void (*OutputStageFunctionType)(StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, int32_t);

OutputStageFunctionType generate_output_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("SourceMask", 1, 1),
                            Input<streamset_t>("PlacementMask", 1, 1),
                            Input<streamset_t>("PlacedData", 1, 8),
                            Input<uint32_t>("inputFileDecriptor"));
    StreamSet * SourceMask = P.getInputStreamSet("SourceMask");
    StreamSet * WorkPlacementMask = P.getInputStreamSet("PlacementMask");
    StreamSet * PlacedData = P.getInputStreamSet("PlacedData");
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");
    //
    //  Rereading the bytestream - TODO : pass in a buffer.
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    //
    //  The following Filter-Spread combination could be integrated
    //  together into a MoveByMask kernel.
    StreamSet * const NonModifiedMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Invert>(WorkPlacementMask, NonModifiedMask);
    SHOW_STREAM(NonModifiedMask);
    StreamSet * FilteredBytes = P.CreateStreamSet(1, 8);
    FilterByMask(P, SourceMask, ByteStream, FilteredBytes);
    StreamSet * PlacedSourceBytes = P.CreateStreamSet(1, 8);
    SpreadByMask(P, NonModifiedMask, FilteredBytes, PlacedSourceBytes);
    //
    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    if (ByteReplace) {
        P.CreateKernelCall<ByteReplaceByMask>(WorkPlacementMask, PlacedSourceBytes, PlacedData, OutputBytes);
    } else {
        StreamSet * const ReorderedPlaced = P.CreateStreamSet(1, 8);
        SpreadByMask(P, WorkPlacementMask, ReorderedBytes, ReorderedPlaced);
        
        P.CreateKernelCall<ByteCombine>(PlacedSourceBytes, ReorderedPlaced, OutputBytes);
    }
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
    return P.compile();
}
    
 */

typedef void (*XfrmFunctionType)(char *, size_t);

XfrmFunctionType generate_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<char*>{"buffer"}, Input<size_t>{"length"});
    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);
    SHOW_BYTES(ByteStream);

    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    
    StreamSet * const u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * NFD_WorkItems = DetermineNFD_WorkItems(P, BasisBits, u8index);

    StreamSet * const WorkSelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<U8Spans>(NFD_WorkItems, u8index, WorkSelectionMask);
    SHOW_STREAM(WorkSelectionMask);

    UTF_Encoder U8_Encoder(8);
    NFD_BixData NFD_Data(U8_Encoder);
    
    StreamSet * const FinalWorkPlacementMask = WorkPlacementMask(P, NFD_Data, BasisBits, WorkSelectionMask);

    StreamSet * const NonModifiedMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Invert>(WorkSelectionMask, NonModifiedMask);
    SHOW_STREAM(NonModifiedMask);

    StreamSet * NonModifiedBasis = nullptr;
    if (ByteMerging) {
        NonModifiedBasis = P.CreateStreamSet(1, 8);
        FilterByMask(P, NonModifiedMask, ByteStream, NonModifiedBasis);
    } else {
        NonModifiedBasis = P.CreateStreamSet(8, 1);
        FilterByMask(P, NonModifiedMask, BasisBits, NonModifiedBasis);
        SHOW_BIXNUM(NonModifiedBasis);
    }
    
    StreamSet * const U21_u8indexed = P.CreateStreamSet(21, 1);
    StreamSet * U21_focus = P.CreateStreamSet(21, 1);
    if (LateU21) {
        StreamSet * const WorkingBasis = P.CreateStreamSet(8, 1);
        FilterByMask(P, WorkSelectionMask, BasisBits, WorkingBasis);
        SHOW_BIXNUM(WorkingBasis);
        StreamSet * const WorkingU8index = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<UTF8_index>(WorkingBasis, WorkingU8index);
        SHOW_STREAM(WorkingU8index);
        P.CreateKernelCall<UTF8_Decoder>(WorkingBasis, U21_u8indexed);
        FilterByMask(P, WorkingU8index, U21_u8indexed, U21_focus);
    } else {
        P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);
        FilterByMask(P, NFD_WorkItems, U21_u8indexed, U21_focus);
    }
    SHOW_BIXNUM(U21_u8indexed);
    SHOW_BIXNUM(U21_focus);
    
    StreamSet * NFD_U21_Results = NFD_U21_Pipeline(P, NFD_Data, U21_focus);

    StreamSet * const ReorderedBasis = P.CreateStreamSet(8);
    U21_to_UTF8(P, NFD_U21_Results, ReorderedBasis);
    SHOW_BIXNUM(ReorderedBasis);

    StreamSet * const NonModifiedPlacementMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Invert>(FinalWorkPlacementMask, NonModifiedPlacementMask);

    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    if (ByteMerging) {
        StreamSet * const NonModifiedPlaced = P.CreateStreamSet(1, 8);
        SpreadByMask(P, NonModifiedPlacementMask, NonModifiedBasis, NonModifiedPlaced);

        StreamSet * const ReorderedBytes = P.CreateStreamSet(1, 8);
        P.CreateKernelCall<P2SKernel>(ReorderedBasis, ReorderedBytes);
        
        if (ByteReplace) {
            P.CreateKernelCall<ByteReplaceByMask>(FinalWorkPlacementMask, NonModifiedPlaced, ReorderedBytes, OutputBytes);
        } else {
            StreamSet * const ReorderedPlaced = P.CreateStreamSet(1, 8);
            SpreadByMask(P, FinalWorkPlacementMask, ReorderedBytes, ReorderedPlaced);

            P.CreateKernelCall<ByteCombine>(NonModifiedPlaced, ReorderedPlaced, OutputBytes);
        }
    } else {
        StreamSet * const ReorderedPlaced = P.CreateStreamSet(8);
        SpreadByMask(P, FinalWorkPlacementMask, ReorderedBasis, ReorderedPlaced);
        SHOW_BIXNUM(ReorderedPlaced);

        StreamSet * const OutputBasis = P.CreateStreamSet(8);
        MergeByMask(P, FinalWorkPlacementMask, ReorderedBasis, NonModifiedBasis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);

        P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
    }
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
    return P.compile();
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    bool useMMap = MMapPreference && canMMap(inputFile);
    AlignedFileBuffer buf;
    buf.load(inputFile, useMMap);
    size_t bytes_read = buf.getBufSize();
    if (bytes_read <= 0) return 0;
    CPUDriver driver("NFD_function");
    if (SeparatedPipelineStages) {
        Stage1FunctionType stage1 = generate_stage1_pipeline(driver);
#ifdef STAGE1_TO_STDOUT
        stage1(buf.getBuf(), buf.getBufSize());
#else
        kernel::StreamSetPtr Working;
        stage1(Working, buf.getBuf(), buf.getBufSize());
#endif
        Stage2FunctionType stage2 = generate_stage2_pipeline(driver);
#ifndef STAGE1_TO_STDOUT
#ifdef STAGE2_TO_STDOUT
        stage2(Working);
#else
        kernel::StreamSetPtr NFD_Bytes;
        stage2(Working, NFD_Bytes);
#endif
#endif
        //OutputStageFunctionType stage3 = generate_output_stage_pipeline(driver);
    } else {
        XfrmFunctionType fn = generate_pipeline(driver);
        fn(buf.getBuf(), buf.getBufSize());
    }
    buf.release();
    return 0;
}
