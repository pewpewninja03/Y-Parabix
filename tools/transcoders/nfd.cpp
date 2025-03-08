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
#include <string>
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
    NFD_BixData();
    std::vector<re::CC *> NFD_Insertion_BixNumCCs();
    unicode::BitTranslationSets NFD_1st_BitXorCCs();
    unicode::BitTranslationSets NFD_2nd_BitCCs();
    unicode::BitTranslationSets NFD_3rd_BitCCs();
    unicode::BitTranslationSets NFD_4th_BitCCs();
private:
    std::unordered_map<codepoint_t, unsigned> mNFD_length;
    unicode::TranslationMap mNFD_CharMap[4];
    UCD::UnicodeSet mHangul_Precomposed_LV;
    UCD::UnicodeSet mHangul_Precomposed_LVT;
};

NFD_BixData::NFD_BixData() : UCD::NFD_Engine(UCD::DecompositionOptions::NFD) {
    auto cps = decompMappingObj->GetExplicitCps();
    for (auto cp : cps) {
        std::u32string decomp;
        NFD_append1(decomp, cp);
        mNFD_length.emplace(cp, decomp.length());
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
    for (auto p : mNFD_length) {
        BixNumCCs.push_back(UCD::UnicodeSet());
        BixNumCCs.push_back(UCD::UnicodeSet());
        auto insert_amt = p.second - 1;
        if ((insert_amt & 1) == 1) {
            BixNumCCs[0].insert(p.first);
        }
        if ((insert_amt & 2) == 2) {
            BixNumCCs[1].insert(p.first);
        }
    }
    BixNumCCs[0] = BixNumCCs[0] + mHangul_Precomposed_LV;
    BixNumCCs[1] = BixNumCCs[1] + mHangul_Precomposed_LVT;
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

    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * U21_u8indexed = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);

    StreamSet * U21 = P.CreateStreamSet(21, 1);
    FilterByMask(P, u8index, U21_u8indexed, U21);
    SHOW_BIXNUM(U21);

    NFD_BixData NFD_Data;
    auto insert_ccs = NFD_Data.NFD_Insertion_BixNumCCs();

    StreamSet * Insertion_BixNum = P.CreateStreamSet(insert_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(insert_ccs, U21, Insertion_BixNum);
    SHOW_BIXNUM(Insertion_BixNum);

    StreamSet * SpreadMask = InsertionSpreadMask(P, Insertion_BixNum, kernel::InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(21, 1);
    SpreadByMask(P, SpreadMask, U21, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    //  The Hangul decomposition algorithm calculates replacements for LV and
    //  LVT characters using calculations based on three 5-bit indexes for
    //  the L, V and T characters.
    StreamSet * LIndexBixNum = P.CreateStreamSet(L_Index_bits);
    StreamSet * VIndexBixNum = P.CreateStreamSet(V_Index_bits);
    StreamSet * TIndexBixNum = P.CreateStreamSet(T_Index_bits);
    P.CreateKernelCall<LVT_Indexes>(ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    // Given the L, V and T indexes, the replacements for LV and LVT characters
    // can be calculated to determine the correct 21-bit representations at
    // <L, V> and <L, V, T> positions.
    StreamSet * Hangul_NFD_Basis = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<LVT2NFD>(ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum, Hangul_NFD_Basis);
    SHOW_BIXNUM(Hangul_NFD_Basis);

    StreamSet * NFD_Basis = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<NFD_Translation>(NFD_Data, Hangul_NFD_Basis, NFD_Basis);
    SHOW_BIXNUM(NFD_Basis);

    UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
    StreamSet * CCC_Basis = P.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    P.CreateKernelCall<UnicodePropertyBasis>(enumObj, NFD_Basis, CCC_Basis);
    SHOW_BIXNUM(CCC_Basis);

    StreamSet * CCC_NonZero = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<bixnum::NEQ_immediate>(CCC_Basis, 0, CCC_NonZero);
    SHOW_STREAM(CCC_NonZero);

    StreamSets ToSort = {CCC_Basis, NFD_Basis};

    StreamSets SortResults = BitonicSortRuns(P, 8, CCC_NonZero, ToSort);
    SHOW_BIXNUM(SortResults[0]);
    SHOW_BIXNUM(SortResults[1]);

    StreamSet * const OutputBasis = P.CreateStreamSet(8);
    U21_to_UTF8(P, SortResults[1], OutputBasis);

    SHOW_BIXNUM(OutputBasis);

    StreamSet * OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
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
