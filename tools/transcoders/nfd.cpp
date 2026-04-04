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
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
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
#include <kernel/bitwise/bixlogic.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/normalization.h>
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
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/algo/normalization.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>
#include <unicode/utf/utf_encoder.h>
#include <unicode/utf/transchar.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decompositon Options", "Decompositon Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));
static cl::opt<bool> NoFocus("NoFocus", cl::desc("Process the entire file without filtering to narrow the focus of work"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteMerging("ByteMerging", cl::desc("Use byte stream merging of transformed and unmodified data"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteReplace("ByteReplace", cl::desc("Perform byte merging using the ByteReplaceByMask kernel"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<int> SeparatedPipelineStages("SeparatedPipelineStages", cl::desc("Use multiple separated pipeline stages"), cl::init(0), cl::cat(NFD_Options));
static cl::opt<bool> FilterViolations("FilterViolations", cl::desc("Only include reorderable sequences in work items if they are misordered or start with a decomposable character"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> UseIndexedShiftBack("IndexedShiftBack", cl::desc("Use IndexedShiftBack in place of Filter/Spread combination"), cl::init(false), cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

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

    BixNum S_Index = bnc.Select(Hangul_precomposed, bnc.SubModular(basis, Hangul::SBase), ZeroBasis);
    S_Index = bnc.Truncate(S_Index, S_Index_bits);
    BixNum LV_index;
    BixNum T_index;
    bnc.Div(S_Index, Hangul::TCount, LV_index, T_index);
    LV_index = bnc.Truncate(LV_index, L_Index_bits + V_Index_bits);
    BixNum L_index;
    BixNum V_index;
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
    LPart = bnc.AddModular(LPart, Hangul::LBase);
    // The V Part, when it exists, is one position after the opening L Part.
    PabloAST * V_position = nested.createAdvance(Hangul_precomposed, 1);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_index[i] = nested.createAdvance(V_index[i], 1);
    }
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
class Mark_EOF : public pablo::PabloKernel {
public:
    Mark_EOF(LLVMTypeSystemInterface & ts, StreamSet * u8index, StreamSet * EOF_mark);
protected:
    void generatePabloMethod() override;
};

Mark_EOF::Mark_EOF (LLVMTypeSystemInterface & ts, StreamSet * u8index, StreamSet * EOF_mark)
: PabloKernel(ts, "Mark_EOF",
// inputs
{Binding{"u8index", u8index}},
// output
{Binding{"EOF_mark", EOF_mark}}) {
}

void Mark_EOF::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * EOFbit = pb.createAtEOF(pb.createAdvance(pb.createOnes(), 1));
    pb.createAssign(pb.createExtract(getOutputStreamVar("EOF_mark"), pb.getInteger(0)), EOFbit);
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

StreamSet * NFD_U21_Pipeline(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * U21_Basis) {
    // Now we have a Unicode-indexed representation of all significant
    // sequences for NFD processing.
    // Expand to make room for decompositions.
    auto insert_ccs = NFD_Data.NFD_Insertion_BixNumCCs();
    StreamSet * const U21_Insertion_BixNum = P.CreateStreamSet(insert_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(insert_ccs, U21_Basis, U21_Insertion_BixNum);
    SHOW_BIXNUM(U21_Insertion_BixNum);

    StreamSet * const U21_SpreadMask = P.CreateStreamSet(1, 1);
    InsertionSpreadMask(P, U21_Insertion_BixNum, U21_SpreadMask, kernel::InsertPosition::After);
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

StreamSet * DetermineNFD_WorkItems(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * U8_Basis, StreamSet * u8index) {
    re::PropertyExpression * dtCanProp = re::makePropertyExpression("dt", "Can");
    dtCanProp = cast<re::PropertyExpression>(UCD::linkAndResolve(dtCanProp));
    StreamSet * DT_Can = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(dtCanProp, U8_Basis, DT_Can);
    SHOW_STREAM(DT_Can);

    StreamSet * const NFD_WorkItems = P.CreateStreamSet(1, 1);
    if (FilterViolations) {
        UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
        StreamSet * CCC_Basis = P.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
        P.CreateKernelCall<UnicodePropertyBasis>(enumObj, U8_Basis, CCC_Basis);
        SHOW_BIXNUM(CCC_Basis);

        StreamSet * EOF_mark = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<Mark_EOF>(u8index, EOF_mark);
        SHOW_STREAM(EOF_mark);

        StreamSet * CCC_SeqMarks = P.CreateStreamSet(1, 1);
        StreamSet * CCC_Violation = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<CCC_Check>(CCC_Basis, u8index, EOF_mark, CCC_SeqMarks, CCC_Violation);
        SHOW_STREAM(CCC_SeqMarks);
        SHOW_STREAM(CCC_Violation);

        StreamSet * CCC_Violation_Start = P.CreateStreamSet(1, 1);
        if (UseIndexedShiftBack) {
            P.CreateKernelCall<IndexedShiftBack>(CCC_SeqMarks, CCC_Violation, CCC_Violation_Start);
            SHOW_STREAM(CCC_Violation_Start);
        } else {
            StreamSet * ViolationsByMarkEnd = P.CreateStreamSet(1, 1);
            FilterByMask(P, CCC_SeqMarks, CCC_Violation, ViolationsByMarkEnd);
            SHOW_STREAM(ViolationsByMarkEnd);

            StreamSet * ViolationsByMarkStart = P.CreateStreamSet(1, 1);
            P.CreateKernelCall<ShiftBack>(ViolationsByMarkEnd, ViolationsByMarkStart, 1);
            SHOW_STREAM(ViolationsByMarkStart);

            SpreadByMask(P, CCC_SeqMarks, ViolationsByMarkStart, CCC_Violation_Start);
            SHOW_STREAM(CCC_Violation_Start);
        }

        P.CreateKernelCall<NFD_Narrow_Focus>(u8index, DT_Can, CCC_Violation_Start, CCC_SeqMarks, NFD_WorkItems);
        SHOW_STREAM(NFD_WorkItems);

    } else {
        re::PropertyExpression * CCC0_Prop = re::makePropertyExpression("CCC", "NR");
        CCC0_Prop = cast<re::PropertyExpression>(UCD::linkAndResolve(CCC0_Prop));
        StreamSet * const CCC_0 = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Prop, U8_Basis, CCC_0);//, BitMovementMode::LookAhead);
        SHOW_STREAM(CCC_0);

        P.CreateKernelCall<NFD_Focus>(u8index, DT_Can, CCC_0, NFD_WorkItems);
        SHOW_STREAM(NFD_WorkItems);
    }
    return NFD_WorkItems;
}

void source_input_stage(PipelineBuilder & P, Scalar *const fileDescriptor, StreamSet * ByteStream, StreamSet * BasisBits) {

    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
}

typedef void (*SourceInputFunctionType)(StreamSetPtr &, StreamSetPtr &, uint32_t fd);

SourceInputFunctionType source_input_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Output<streamset_t>("ByteStream", 1, 8, ReturnedBuffer(1)),
                            Output<streamset_t>("BasisBits", 8, 1, ReturnedBuffer(1)),
                            Input<uint32_t>("inputFileDecriptor")
                            );
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * const ByteStream = P.getOutputStreamSet("ByteStream");
    StreamSet * const BasisBits = P.getOutputStreamSet("BasisBits");
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);
    return P.compile();
}

void NFD_FilterStage(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * BasisBits, StreamSet * WorkSelectionMask, StreamSet * FinalWorkPlacementMask, StreamSet * WorkingBasis) {

    StreamSet * const u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * NFD_WorkItems = DetermineNFD_WorkItems(P, NFD_Data, BasisBits, u8index);

    P.CreateKernelCall<U8Spans>(NFD_WorkItems, u8index, WorkSelectionMask);
    SHOW_STREAM(WorkSelectionMask);

    ComputeWorkPlacement(P, NFD_Data.UTF8_Insertion_BixNumCCs(), NFD_Data.UTF8_Deletion_BixNumCCs(), BasisBits, WorkSelectionMask, FinalWorkPlacementMask);
    SHOW_STREAM(FinalWorkPlacementMask);

    FilterByMask(P, WorkSelectionMask, BasisBits, WorkingBasis);
    SHOW_BIXNUM(WorkingBasis);
}

typedef void (*NFD_FilterFunctionType)(const StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, StreamSetPtr &);

NFD_FilterFunctionType NFD_filter_pipeline(CPUDriver & driver, NFD_BixData & NFD_Data) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("BasisBits", 8, 1),
                            Output<streamset_t>("WorkSelectionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("FinalWorkPlacementMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("WorkingBasis", 8, 1, ReturnedBuffer(1))
                            );
    StreamSet * const BasisBits = P.getInputStreamSet("BasisBits");
    StreamSet * const WorkSelectionMask = P.getOutputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getOutputStreamSet("FinalWorkPlacementMask");
    StreamSet * const WorkingBasis = P.getOutputStreamSet("WorkingBasis");

    NFD_FilterStage(P, NFD_Data, BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);
    return P.compile();
}

void NFD_Transform_Stage(PipelineBuilder & P, NFD_BixData & NFD_Data, StreamSet * WorkingBasis, StreamSet * TransformedBasis) {

    StreamSet * const U21_u8indexed = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<UTF8_Decoder>(WorkingBasis, U21_u8indexed);
    SHOW_BIXNUM(U21_u8indexed);

    StreamSet * const WorkingU8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(WorkingBasis, WorkingU8index);
    SHOW_STREAM(WorkingU8index);

    StreamSet * const U21_focus = P.CreateStreamSet(21, 1);
    FilterByMask(P, WorkingU8index, U21_u8indexed, U21_focus);
    SHOW_BIXNUM(U21_focus);

    StreamSet * NFD_U21_Results = NFD_U21_Pipeline(P, NFD_Data, U21_focus);

    U21_to_UTF8(P, NFD_U21_Results, TransformedBasis);
    SHOW_BIXNUM(TransformedBasis);
}

typedef void (*NFD_TransformFunctionType)(const StreamSetPtr &, StreamSetPtr &);

NFD_TransformFunctionType NFD_transform_pipeline(CPUDriver & driver, NFD_BixData & NFD_Data) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkingBasis", 8, 1),
                            Output<streamset_t>("TransformedBasis", 8, 1, ReturnedBuffer(1))
                            );
    StreamSet * const WorkingBasis = P.getInputStreamSet("WorkingBasis");
    StreamSet * const TransformedBasis = P.getOutputStreamSet("TransformedBasis");

    NFD_Transform_Stage(P, NFD_Data, WorkingBasis, TransformedBasis);
    return P.compile();
}

void OutputAssemblyStage(PipelineBuilder & P, StreamSet * WorkSelectionMask, StreamSet * FinalWorkPlacementMask, StreamSet * Source, StreamSet * TransformedBasis) {

    StreamSet * const NonModifiedMask = P.CreateStreamSet(1, 1);
    Invert(P, WorkSelectionMask, NonModifiedMask);
    SHOW_STREAM(NonModifiedMask);

    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    if (ByteMerging) {
        StreamSet * const NonModified = P.CreateStreamSet(1, 8);
        FilterByMask(P, NonModifiedMask, Source, NonModified);
        SHOW_BYTES(NonModified);
        
        StreamSet * const NonModifiedPlacementMask = P.CreateStreamSet(1, 1);
        Invert(P, FinalWorkPlacementMask, NonModifiedPlacementMask);
        SHOW_STREAM(NonModifiedPlacementMask);
        
        StreamSet * const NonModifiedPlaced = P.CreateStreamSet(1, 8);
        SpreadByMask(P, NonModifiedPlacementMask, NonModified, NonModifiedPlaced);
        
        StreamSet * const TransformedBytes = P.CreateStreamSet(1, 8);
        P.CreateKernelCall<P2SKernel>(TransformedBasis, TransformedBytes);

        if (ByteReplace) {
            P.CreateKernelCall<ByteReplaceByMask>(FinalWorkPlacementMask, NonModifiedPlaced, TransformedBytes, OutputBytes);
        } else {
            StreamSet * const TransformedPlaced = P.CreateStreamSet(1, 8);
            SpreadByMask(P, FinalWorkPlacementMask, TransformedBytes, TransformedPlaced);

            P.CreateKernelCall<ByteCombine>(NonModifiedPlaced, TransformedPlaced, OutputBytes);
        }
    } else {
        StreamSet * const NonModifiedBasis = P.CreateStreamSet(8);
        FilterByMask(P, NonModifiedMask, Source, NonModifiedBasis);
        SHOW_BIXNUM(NonModifiedBasis);

        StreamSet * const OutputBasis = P.CreateStreamSet(8);
        MergeByMask(P, FinalWorkPlacementMask, TransformedBasis, NonModifiedBasis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);

        P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
    }
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}

void SimpleOutputStage(PipelineBuilder & P, StreamSet * TransformedBasis) {
    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(TransformedBasis, OutputBytes);
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}

typedef void (*CombinedWorkFunctionType)(StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, uint32_t fd);

CombinedWorkFunctionType generate_combined_work_pipeline(CPUDriver & driver, NFD_BixData & NFD_Data) {
    auto P = CreatePipeline(driver,
                            Output<streamset_t>("WorkSelectionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("FinalWorkPlacementMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("ByteStream", 1, 8, ReturnedBuffer(1)),
                            Output<streamset_t>("TransformedBasis", 8, 1, ReturnedBuffer(1)),
                            Input<uint32_t>("inputFileDecriptor")
                            );
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * const WorkSelectionMask = P.getOutputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getOutputStreamSet("FinalWorkPlacementMask");
    StreamSet * const ByteStream = P.getOutputStreamSet("ByteStream");
    StreamSet * const TransformedBasis = P.getOutputStreamSet("TransformedBasis");

    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);

    StreamSet * WorkingBasis = P.CreateStreamSet(8, 1);
    NFD_FilterStage(P, NFD_Data, BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);

    NFD_Transform_Stage(P, NFD_Data, WorkingBasis, TransformedBasis);

    return P.compile();
}

typedef void (*OutputAssemblyFunctionType)(const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &);

OutputAssemblyFunctionType generate_output_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkSelectionMask", 1, 1),
                            Input<streamset_t>("FinalWorkPlacementMask", 1, 1),
                            Input<streamset_t>("Source", ByteMerging ? 1 : 8, ByteMerging ? 8 : 1),
                            Input<streamset_t>("TransformedBasis", 8, 1)
                            );
    StreamSet * const WorkSelectionMask = P.getInputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getInputStreamSet("FinalWorkPlacementMask");
    StreamSet * const Source = P.getInputStreamSet("Source");
    StreamSet * const TransformedBasis = P.getInputStreamSet("TransformedBasis");

    OutputAssemblyStage(P, WorkSelectionMask, FinalWorkPlacementMask, Source, TransformedBasis);
    return P.compile();
}

typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_unitary_pipeline(CPUDriver & driver, NFD_BixData & NFD_Data) {
    auto P = CreatePipeline(driver, Input<uint32_t>("inputFileDecriptor"));
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);

    if (NoFocus) {
        StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
        NFD_Transform_Stage(P, NFD_Data, BasisBits, TransformedBasis);

        SimpleOutputStage(P, TransformedBasis);
    } else {
        StreamSet * WorkSelectionMask = P.CreateStreamSet(1, 1);
        StreamSet * FinalWorkPlacementMask = P.CreateStreamSet(1, 1);
        StreamSet * WorkingBasis = P.CreateStreamSet(8, 1);
        NFD_FilterStage(P, NFD_Data, BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);
        
        StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
        NFD_Transform_Stage(P, NFD_Data, WorkingBasis, TransformedBasis);
        
        OutputAssemblyStage(P, WorkSelectionMask, FinalWorkPlacementMask, ByteMerging ? ByteStream : BasisBits, TransformedBasis);
    }
    return P.compile();
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
        exit(-1);
    }
    if (ByteReplace) ByteMerging = true;
    if (UseIndexedShiftBack) FilterViolations = true;
    UTF_Encoder U8_Encoder(8);
    NFD_BixData NFD_Data(U8_Encoder);
    CPUDriver driver("NFD_function");
    if (SeparatedPipelineStages == 2) {
        CombinedWorkFunctionType working_stages = generate_combined_work_pipeline(driver, NFD_Data);
        kernel::StreamSetPtr WorkSelectionMask;
        kernel::StreamSetPtr FinalWorkPlacementMask;
        kernel::StreamSetPtr ByteStream;
        kernel::StreamSetPtr TransformedBasis;
        working_stages(WorkSelectionMask, FinalWorkPlacementMask, ByteStream, TransformedBasis, fd);
        close(fd);

        OutputAssemblyFunctionType stage3 = generate_output_pipeline(driver);
        stage3(WorkSelectionMask, FinalWorkPlacementMask, ByteStream, TransformedBasis);
    } else if (SeparatedPipelineStages == 4) {
        SourceInputFunctionType stage0 = source_input_pipeline(driver);
        kernel::StreamSetPtr ByteStream;
        kernel::StreamSetPtr BasisBits;
        stage0(ByteStream, BasisBits, fd);
        close(fd);

        NFD_FilterFunctionType stage1 = NFD_filter_pipeline(driver, NFD_Data);
        kernel::StreamSetPtr WorkSelectionMask;
        kernel::StreamSetPtr FinalWorkPlacementMask;
        kernel::StreamSetPtr WorkingBasis;
        stage1(BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);

        NFD_TransformFunctionType stage2 = NFD_transform_pipeline(driver, NFD_Data);
        kernel::StreamSetPtr TransformedBasis;
        stage2(WorkingBasis, TransformedBasis);

        OutputAssemblyFunctionType stage3 = generate_output_pipeline(driver);
        stage3(WorkSelectionMask, FinalWorkPlacementMask, ByteMerging ? ByteStream : BasisBits, TransformedBasis);
    } else {
        XfrmFunctionType fn = generate_unitary_pipeline(driver, NFD_Data);
        fn(fd);
        close(fd);
    }
    return 0;
}
