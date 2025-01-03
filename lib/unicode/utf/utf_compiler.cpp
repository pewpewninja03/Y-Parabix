#include <unicode/utf/utf_compiler.h>
#include <unicode/utf/utf_encoder.h>
#include <array>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_var.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/printer_pablos.h>
#include <pablo/bixnum/bixnum.h>
#include <re/alphabet/alphabet.h>
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_compiler.h>
#include <re/alphabet/alphabet.h>
#include <re/adt/re_name.h>
#include <re/adt/re_cc.h>
#include <unicode/core/unicode_set.h>
#include <toolchain/toolchain.h>
#include <re/printer/re_printer.h>
#include <llvm/Support/CommandLine.h>
#include <boost/intrusive/detail/math.hpp>

using namespace cc;
using namespace re;
using namespace pablo;
using namespace llvm;
using namespace boost::container;

namespace UTF {

static cl::opt<unsigned> BinaryLogicCostPerByte("BinaryLogicCostPerByte", cl::init(2), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> TernaryLogicCostPerByte("TernaryLogicCostPerByte", cl::init(1), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> ShiftCostFactor("ShiftCostFactor", cl::init(10), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> IfEmbeddingCostThreshhold("IfEmbeddingCostThreshhold", cl::init(15), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> PartitioningCostThreshhold("PartitioningCostThreshhold", cl::init(12), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> PartitioningFactor("PartitioningFactor", cl::init(4), cl::cat(codegen::CodeGenOptions));
static cl::opt<unsigned> RangeMaskFactor("RangeMaskFactor", cl::init(4), cl::cat(codegen::CodeGenOptions));
enum class PartitioningKind {Linear, Logarithmic, Geometric, Gaps};
static cl::opt<PartitioningKind> Partitioning("Partitioning", cl::ValueOptional,
cl::values(
           clEnumValN(PartitioningKind::Linear, "Linear", "Partitioning based on linear division."),
           clEnumValN(PartitioningKind::Logarithmic, "Logarithmic", "Partitioning based on logarithmic division")),
                                                                               cl::cat(codegen::CodeGenOptions), cl::init(PartitioningKind::Linear));
static cl::opt<bool> PartitioningRevision("PartitioningRevision", cl::init(true), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> SuffixOptimization("SuffixOptimization", cl::init(false), cl::cat(codegen::CodeGenOptions));
static cl::opt<bool> AdvanceBasis("AdvanceBasis", cl::desc("Advance basis bits before compileCodeUnit"), cl::init(false), cl::cat(codegen::CodeGenOptions));
enum class InitialTestMode {PrefixCC, RangeCC, NonASCII};
static cl::opt<unsigned> PrefixTestMax("PrefixTestMax", cl::desc("Max prefix count to focus initial non-ASCII test to exact prefix CC"), cl::init(30), cl::cat(codegen::CodeGenOptions));
static cl::opt<InitialTestMode> InitialTest("InitialTest", cl::ValueOptional,
cl::values(
           clEnumValN(InitialTestMode::PrefixCC, "PrefixCC", "Initial test based on exact prefix CC."),
           clEnumValN(InitialTestMode::RangeCC, "RangeCC", "Initial test based on range of possible prefixes."),
           clEnumValN(InitialTestMode::NonASCII, "NonASCII", "Initial test based on non ASCII (any prefix/suffix).")),
                                            cl::cat(codegen::CodeGenOptions), cl::init(InitialTestMode::PrefixCC));
static cl::opt<bool> UTF_CompilationTracing("UTF_CompilationTracing", cl::init(false), cl::cat(codegen::CodeGenOptions));
enum class CC_Mode {BixNumCCs, SyntheticBasis, TruncatedBasis};
static cl::opt<CC_Mode> CCmode("CCmode", cl::ValueOptional,
cl::values(
           clEnumValN(CC_Mode::BixNumCCs, "BixNumCCs", "Use BixNum compiler for CC calculations."),
           clEnumValN(CC_Mode::SyntheticBasis, "SyntheticBasis", "Use a full synthetic basis for CCs."),
           clEnumValN(CC_Mode::TruncatedBasis, "TruncatedBasis", "Use truncated basis based on range.significant_bits for CCs.")),
                                            cl::cat(codegen::CodeGenOptions), cl::init(CC_Mode::TruncatedBasis));

std::string kernelAnnotation() {
    std::string a = "+b" + std::to_string(BinaryLogicCostPerByte);
    a += "t" + std::to_string(TernaryLogicCostPerByte);
    a += "s" + std::to_string(ShiftCostFactor);
    a += "i" + std::to_string(IfEmbeddingCostThreshhold);
    a += "p" + std::to_string(PartitioningCostThreshhold);
    a += "f" + std::to_string(PartitioningFactor);
    a += "x" + std::to_string(RangeMaskFactor);
    a += "r" + std::to_string(PartitioningRevision);
    a += "m" + std::to_string(PrefixTestMax);
    if (AdvanceBasis) {
        a += "a";
    }
    if (SuffixOptimization) {
        a += "x";
    }
    if (Partitioning == PartitioningKind::Linear) {
        a += "lP";
    } else if (Partitioning == PartitioningKind::Logarithmic) {
        a += "+gP";
    }
    if (InitialTest == InitialTestMode::NonASCII) {
        a += "+nA";
    } else if (InitialTest == InitialTestMode::RangeCC) {
        a += "+rCC";
    }
    if (CCmode == CC_Mode::SyntheticBasis) {
        a += "+sB";
    } else if (CCmode == CC_Mode::BixNumCCs) {
        a += "+bX";
    }
    return a;
}

using PabloAST = pablo::PabloAST;
using PabloBuilder = pablo::PabloBuilder;
using Basis_Set = std::vector<PabloAST *>;
using boost::intrusive::detail::ceil_log2;

struct Range {
    codepoint_t lo;
    codepoint_t hi;
    bool is_empty() {return lo > hi;}
    bool operator==(const Range & rhs) {
        return (lo == rhs.lo) && (hi == rhs.hi);
    }
    unsigned significant_bits() {
        auto differing_bits = lo ^ hi;
        return ceil_log2(differing_bits + 1);
    }
    unsigned total_bits() {
        return ceil_log2(hi+1);
    }
    std::string hex_string() {
        std::stringstream s;
        s << std::hex << lo << "_" << hi;
        return s.str();
    }
};


UCD::UnicodeSet computeEndpoints(const std::vector<re::CC *> & CCs) {
    UCD::UnicodeSet endpoints;
    for (unsigned i = 0; i < CCs.size(); i++) {
        for (const auto range : *CCs[i]) {
            const auto lo = re::lo_codepoint(range);
            const auto hi = re::hi_codepoint(range);
            endpoints.insert(lo);
            endpoints.insert(hi);
        }
    }
    return endpoints;
}

UCD::UnicodeSet computeEndpoints(const std::vector<const re::CC *> & CCs) {
    UCD::UnicodeSet endpoints;
    for (unsigned i = 0; i < CCs.size(); i++) {
        for (const auto range : *CCs[i]) {
            const auto lo = re::lo_codepoint(range);
            const auto hi = re::hi_codepoint(range);
            endpoints.insert(lo);
            endpoints.insert(hi);
        }
    }
    return endpoints;
}

re::CC * reduceCC(re::CC * cc, codepoint_t mask, cc::Alphabet & a) {
    UCD::UnicodeSet reduced;
    for (const auto range : *cc) {
        const auto lo = re::lo_codepoint(range) & mask;
        const auto hi = re::hi_codepoint(range) & mask;
        reduced.insert_range(lo, hi);
    }
    return re::makeCC(reduced, &a);
}
//
// Determine the actual range of codepoints encountered in a CC_List.
// If all the CCs are empty, return the impossible range {0x10FFFF, 0}.
Range CC_Set_Range(CC_List ccs) {
    codepoint_t lo = 0x10FFFF;
    codepoint_t hi = 0;
    for (unsigned i = 0; i < ccs.size(); i++) {
        if (!ccs[i]->empty()) {
            lo = std::min(lo_codepoint(ccs[i]->front()), lo);
            hi = std::max(hi_codepoint(ccs[i]->back()), hi);
        }
    }
    return Range{lo, hi};
}

void extract_CCs_by_range(Range r, CC_List & ccs, CC_List & in_range) {
    re::CC * rangeCC = re::makeCC(r.lo, r.hi, &Unicode);
    for (unsigned i = 0; i < ccs.size(); i++) {
        in_range[i] = re::intersectCC(ccs[i], rangeCC);
    }
}

struct EnclosingInfo {
    Range       range;
    PabloAST *  test;
    unsigned    testPosition;
    EnclosingInfo(Range & r, PabloAST * t, unsigned pos = 1) :
    range(r), test(t), testPosition(pos) {}
};

class Unicode_Range_Compiler {
public:
    Unicode_Range_Compiler(Basis_Set & basis, Target_List & targets, PabloBuilder & pb) :
        mBasis (basis), mTargets(targets), mPB(pb) {}
    void compile(CC_List & ccs, EnclosingInfo & enclosing);
protected:
    Basis_Set   &           mBasis;
    Target_List  &          mTargets;
    PabloBuilder &          mPB;
    unsigned costModel(CC_List & ccs);
    Basis_Set RangeBasis(Range enclosing_range, PabloBuilder & pb);
    std::vector<Range> selectPartitions(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb);
    void subrangePartitioning(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb);
    void compileUnguardedSubrange(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb);
    PabloAST * compileCodeRange(EnclosingInfo & enclosing, Range & codepointRange, PabloBuilder & pb);
};

void Unicode_Range_Compiler::compile(CC_List & ccs, EnclosingInfo & enclosing) {
    subrangePartitioning(ccs, enclosing, mPB);
}

unsigned Unicode_Range_Compiler::costModel(CC_List & ccs) {
    UCD::UnicodeSet endpoints = computeEndpoints(ccs);
    Range cc_span = CC_Set_Range(ccs);
    if (cc_span.is_empty()) return 0;
    unsigned bits_to_test = cc_span.significant_bits();
    unsigned total_codepoints = endpoints.count();
    return total_codepoints * bits_to_test * BinaryLogicCostPerByte / 8;
}

Basis_Set Unicode_Range_Compiler::RangeBasis(Range enclosing_range, PabloBuilder & pb) {
    Basis_Set rb(mBasis.size());
    unsigned significant_bits = enclosing_range.significant_bits();
    for (unsigned i = 0; i < significant_bits; i++) {
        rb[i] = mBasis[i];
    }
    for (unsigned i = significant_bits; i < mBasis.size(); i++) {
        if (((enclosing_range.lo >> i) & 1) == 1) {
                rb[i] = pb.createOnes();
        } else {
            rb[i] = pb.createZeroes();
        }
    }
    return rb;
}

std::vector<Range> Unicode_Range_Compiler::selectPartitions(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb) {
    std::vector<Range> partitions;
    Range actual_subrange = CC_Set_Range(ccs);
    if (UTF_CompilationTracing) {
        llvm::errs() << "URC::selectPartitions(" << enclosing.range.hex_string() << ")\n";
    }
    if (actual_subrange.is_empty()) return partitions;
    unsigned working_bits = actual_subrange.significant_bits();
    if (working_bits <= PartitioningFactor) {
        return {enclosing.range};
    }
    unsigned working_bitmask = (1u << working_bits) - 1u;
    if (Partitioning == PartitioningKind::Linear) {
        unsigned partition_bits = ceil_log2(PartitioningFactor);
        unsigned partition_size = 1u << (working_bits - partition_bits);
        unsigned partition_lo = actual_subrange.lo &~working_bitmask;
        for (unsigned i = 0; i < PartitioningFactor; i++) {
            unsigned partition_hi = std::min(partition_lo + partition_size - 1, actual_subrange.hi);
            partitions.push_back(Range{partition_lo, partition_hi});
            if (UTF_CompilationTracing) {
                llvm::errs() << "URC::selectPartitions: partition " << i << ": " << partitions[i].hex_string() << ")\n";
            }
            partition_lo = partition_hi + 1;
        }
        partitions[PartitioningFactor - 1].hi = actual_subrange.hi;
    } else if (Partitioning == PartitioningKind::Logarithmic) {
        unsigned pcount = working_bits > PartitioningFactor ? PartitioningFactor : 1;
        unsigned partition_lo = actual_subrange.lo;
        for (unsigned i = 0; i < pcount; i++) {
            unsigned shift = pcount - i - 1;
            unsigned partition_hi = partition_lo | (working_bitmask >> shift);
            partitions.push_back(Range{partition_lo, partition_hi});
            if (UTF_CompilationTracing) {
                llvm::errs() << "URC::selectPartitions: partition " << i << ": " << partitions[i].hex_string() << ")\n";
            }
            partition_lo = partition_hi + 1;
        }
    }    return partitions;
}

void Unicode_Range_Compiler::subrangePartitioning(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb) {
    std::vector<Range> partitions = selectPartitions(ccs, enclosing, pb);
    if (partitions.size() < 2) {
        compileUnguardedSubrange(ccs, enclosing, pb);
        return;
    }
    Basis_Set basis = RangeBasis(enclosing.range, pb);
    cc::Parabix_CC_Compiler_Builder rangeCompiler(basis);
    std::vector<EnclosingInfo> partitionInfo;
    std::vector<CC_List> partitionCCs(partitions.size());
    std::vector<unsigned> partitionCosts(partitions.size());
    for (unsigned i = 0; i < partitions.size(); i++) {
        partitionCCs[i].resize(ccs.size());
        extract_CCs_by_range(partitions[i], ccs, partitionCCs[i]);
        Range partition_subrange = CC_Set_Range(partitionCCs[i]);
        PabloAST * subrangeTest = nullptr;
        if (!partition_subrange.is_empty()) {
            unsigned partition_bits = partition_subrange.significant_bits();
            codepoint_t mask = (1u << partition_bits) - 1;
            mask >>= RangeMaskFactor;
            partition_subrange.lo &= ~mask;
            partition_subrange.hi |= mask;
            re::CC * subrangeCC = re::makeCC(partition_subrange.lo, partition_subrange.hi, &Unicode);
            subrangeTest = rangeCompiler.compileCC(subrangeCC, pb);
            subrangeTest = pb.createAnd(enclosing.test, subrangeTest, "Range_" + partition_subrange.hex_string());
            partitionCosts[i] = costModel(partitionCCs[i]);
        } else {
            subrangeTest = pb.createZeroes();
            partitionCosts[i] = 0;
        }
        partitionInfo.push_back(EnclosingInfo(partition_subrange, subrangeTest));
    }
    for (unsigned i = 0; i < partitions.size(); i++) {
        if (partitionCosts[i] < IfEmbeddingCostThreshhold) {
            compileUnguardedSubrange(partitionCCs[i], partitionInfo[i], pb);
        } else if (partitionCosts[i] >= IfEmbeddingCostThreshhold) {
            auto nested = pb.createScope();
            pb.createIf(partitionInfo[i].test, nested);
            subrangePartitioning(partitionCCs[i], partitionInfo[i], nested);
        }
    }
}

void Unicode_Range_Compiler::compileUnguardedSubrange(CC_List & ccs, EnclosingInfo & enclosing, PabloBuilder & pb) {
    Range actual_subrange = CC_Set_Range(ccs);
    if (actual_subrange.is_empty()) return;
    if (UTF_CompilationTracing) {
        llvm::errs() << "URC::compileUnguardedSubrange(" << enclosing.range.hex_string() << ")\n";
    }
    if (CCmode == CC_Mode::BixNumCCs) {
        for (unsigned i = 0; i < ccs.size(); i++) {
            for (const auto range : *ccs[i]) {
                Range r{re::lo_codepoint(range), re::hi_codepoint(range)};
                PabloAST * compiled = compileCodeRange(enclosing, r, pb);
                pb.createAssign(mTargets[i], pb.createOr(mTargets[i], compiled));
            }
        }
    } else if (CCmode == CC_Mode::SyntheticBasis) {
        Basis_Set basis = RangeBasis(enclosing.range, pb);
        cc::Parabix_CC_Compiler_Builder rangeCompiler(basis);
        for (unsigned i = 0; i < ccs.size(); i++) {
            PabloAST * compiled = rangeCompiler.compileCC(ccs[i], pb);
            pb.createAssign(mTargets[i], pb.createOr(mTargets[i], pb.createAnd(compiled, enclosing.test)));
        }
    } else {
        unsigned significant_bits = enclosing.range.significant_bits();
        std::string alphabet = "Low" + std::to_string(significant_bits);
        cc::CodeUnitAlphabet CodeAlpha(alphabet, alphabet, significant_bits);
        codepoint_t mask = (1U << significant_bits) - 1;
        pablo::BixNumCompiler bnc(pb);
        BixNum truncated = bnc.Truncate(mBasis, significant_bits);
        cc::Parabix_CC_Compiler_Builder truncatedCompiler(truncated);
        for (unsigned i = 0; i < ccs.size(); i++) {
            re::CC * reducedCC = reduceCC(ccs[i], mask, CodeAlpha);
            PabloAST * compiled = truncatedCompiler.compileCC(reducedCC, pb);
            pb.createAssign(mTargets[i], pb.createOr(mTargets[i], pb.createAnd(compiled, enclosing.test)));
        }
    }
}

PabloAST * Unicode_Range_Compiler::compileCodeRange(EnclosingInfo & enclosing, Range & codepointRange, PabloBuilder & pb) {
    pablo::BixNumCompiler bnc(pb);
    unsigned significant_bits = enclosing.range.significant_bits();
    codepoint_t mask = (1 << significant_bits) - 1;
    BixNum truncated = bnc.Truncate(mBasis, significant_bits);
    codepoint_t lo = codepointRange.lo & mask;
    codepoint_t hi = codepointRange.hi & mask;
    if (lo == hi) {
        return pb.createAnd(enclosing.test, bnc.EQ(truncated, lo, "EQ_" + std::to_string(lo)));
    } else {
        PabloAST * lo_test = bnc.UGE(truncated, lo, "UGE_" + std::to_string(lo));
        PabloAST * hi_test = bnc.ULE(truncated, hi, "ULE_" + std::to_string(hi));
        return pb.createAnd3(enclosing.test, lo_test, hi_test);
    }
}

PabloAST * combineOr(PabloAST * e1, PabloAST * e2, PabloBuilder & pb) {
    if (e1 == nullptr) return e2;
    if (e2 == nullptr) return e1;
    return pb.createOr(e1, e2);
}

PabloAST * combineAnd(PabloAST * e1, PabloAST * e2, PabloBuilder & pb) {
    if (e1 == nullptr) return e2;
    if (e2 == nullptr) return e1;
    return pb.createAnd(e1, e2);
}

class U21_Compiler {
public:
    const unsigned mCodeUnitBits = 21;
    U21_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) :
        mBasisVar(v), mPB(pb), mMask(mask) {
        mEncoder.setCodeUnitBits(mCodeUnitBits);
    }
    void compile(Target_List targets, CC_List ccs);
protected:
    pablo::Var *            mBasisVar;
    PabloBuilder &          mPB;
    pablo::PabloAST *       mMask;
    UTF_Encoder             mEncoder;
    Target_List             mTargets;
    Basis_Set               mBasis;
    Basis_Set prepareUnifiedBasis(Range basis_range);
};

Basis_Set U21_Compiler::prepareUnifiedBasis(Range basis_range) {
    Basis_Set basis(ceil_log2(basis_range.hi));
    for (unsigned i = 0; i < basis.size(); i++) {
        basis[i] = mBasis[i];
    }
    return basis;
}

void U21_Compiler::compile(Target_List targets, CC_List ccs) {
    //  Initialize all the target vars to 0.
    mTargets = targets;
    for (unsigned i = 0; i < targets.size(); i++) {
        mPB.createAssign(mTargets[i], mPB.createZeroes());
    }
    llvm::ArrayType * ty = cast<ArrayType>(mBasisVar->getType());
    mBasis.resize(mCodeUnitBits);
    unsigned streamCount = ty->getArrayNumElements();
    for (unsigned i = 0; i < streamCount; i++) {
        mBasis[i] = mPB.createExtract(mBasisVar, mPB.getInteger(i));
    }
    for (unsigned i = streamCount; i < mCodeUnitBits; i++) {
        mBasis[i] = mPB.createZeroes();
    }
    if (UTF_CompilationTracing) {
        llvm::errs() << "U21_Compiler\n";
    }
    if (InitialTest == InitialTestMode::NonASCII) {
        Range ASCII_Range{0, 0x7F};
        CC_List ASCII_ccs(ccs.size());
        extract_CCs_by_range(ASCII_Range, ccs, ASCII_ccs);
        Basis_Set ASCIIBasis = prepareUnifiedBasis(ASCII_Range);
        Unicode_Range_Compiler ASCII_compiler(ASCIIBasis, mTargets, mPB);
        PabloAST * e1 = mPB.createOr3(mBasis[7], mBasis[8], mBasis[9]);
        PabloAST * e2 = mPB.createOr3(mBasis[10], mBasis[11], mBasis[12]);
        PabloAST * e3 = mPB.createOr3(mBasis[13], mBasis[14], mBasis[15]);
        PabloAST * e4 = mPB.createOr3(mBasis[16], mBasis[17], mBasis[18]);
        PabloAST * e5 = mPB.createOr3(mBasis[19], mBasis[20], e1);
        PabloAST * e6 = mPB.createOr3(e2, e3, e4);
        PabloAST * nonASCII = mPB.createOr(e5, e6);
        EnclosingInfo ASCII_info(ASCII_Range, mPB.createNot(nonASCII));
        ASCII_compiler.compile(ccs, ASCII_info);
        auto nested = mPB.createScope();
        PabloAST * test = combineAnd(mMask, nonASCII, mPB);
        mPB.createIf(test, nested);
        Range nonASCII_Range{0x80, 0x10FFFF};
        CC_List nonASCII_ccs(ccs.size());
        extract_CCs_by_range(nonASCII_Range, ccs, nonASCII_ccs);
        Unicode_Range_Compiler range_compiler(mBasis, mTargets, nested);
        EnclosingInfo nonASCII_info(nonASCII_Range, test);
        range_compiler.compile(nonASCII_ccs, nonASCII_info);
    } else {
        Range UnicodeRange{0, 0x10FFFF};
        if (mMask) {
            auto nested = mPB.createScope();
            mPB.createIf(mMask, nested);
            Unicode_Range_Compiler range_compiler(mBasis, mTargets, nested);
            EnclosingInfo Unicode_info(UnicodeRange, mMask);
            range_compiler.compile(ccs, Unicode_info);
        } else {
            Unicode_Range_Compiler range_compiler(mBasis, mTargets, mPB);
            EnclosingInfo Unicode_info(UnicodeRange, mPB.createOnes());
            range_compiler.compile(ccs, Unicode_info);
        }
    }
}

enum U8_Seq_Kind : unsigned {ASCII, TwoByte, ThreeByte, FourByte};
std::vector<Range> UTF8_Range =
    {Range{0, 0x7F}, Range{0x80, 0x7FF}, Range{0x800, 0xFFFF}, Range{0x10000, 0x10FFFF}};

struct SeqData {
    CC_List                         seqCCs;
    Range                           actualRange;
    Range                           testRange;
    re::CC *                        byte1CC;
    PabloAST *                      test;
    PabloAST *                      combinedTest;
    PabloAST *                      suffixTest;
    std::vector<pablo::Var *>       targets;
};

class U8_Compiler {
public:
    const unsigned mCodeUnitBits = 8;
    const unsigned suffixDataBits = 6;
    U8_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) :
        mBasisVar(v), mPB(pb), mMask(mask) {
        mEncoder.setCodeUnitBits(mCodeUnitBits);
    }
    void compile(Target_List targets, CC_List ccs);
protected:
    pablo::Var *            mBasisVar;
    PabloBuilder &          mPB;
    pablo::PabloAST *       mMask;
    UTF_Encoder             mEncoder;
    Target_List             mTargets;
    SeqData                 mSeqData[4];
    Basis_Set               mScopeBasis[4];
    std::unique_ptr<cc::CC_Compiler> mCodeUnitCompiler[4];
    re::CC * codeUnitCC(re::CC *, unsigned pos = 1);
    re::CC * codeUnitCC(CC_List & ccs, unsigned pos = 1);
    virtual Basis_Set & getBasis(U8_Seq_Kind k, unsigned pos) = 0;
    virtual bool costModelExceedsThreshhold(CC_List & ccs, unsigned from_pos, unsigned threshhold);
    std::vector<UCD::UnicodeSet> computeFullBlockSets(CC_List & ccs, unsigned pos);
    void lengthAnalysis(CC_List & ccs);
    void preparePrefixTests(PabloBuilder & pb);
    void createInitialHierarchy(CC_List & ccs);
    void extendLengthHierarchy(EnclosingInfo & info, PabloBuilder & pb);
    void prepareFixedLengthHierarchy(U8_Seq_Kind k, EnclosingInfo & info, PabloBuilder & pb);
    CC_List prepareFullBlockSets(U8_Seq_Kind k, Range enclosing_range, CC_List ccs, PabloAST * enclosing_test, unsigned code_unit, PabloBuilder & pb);
    void compileFromCodeUnit(U8_Seq_Kind k, EnclosingInfo & if_parent, EnclosingInfo & enclosing, unsigned code_unit, PabloBuilder & pb);
    PabloAST * compilePrefix(re::CC * prefixCC, PabloBuilder & pb);
    PabloAST * compileCodeUnit(U8_Seq_Kind k, Range enclosing_range, re::CC * unitCC, unsigned pos, PabloBuilder & pb);
    virtual void prepareSuffix(unsigned scope, PabloBuilder & pb) = 0;
    virtual void prepareScope(unsigned scope, PabloBuilder & pb) = 0;
    virtual PabloAST * adjustPosition(PabloAST * t, unsigned from, unsigned to, PabloBuilder & pb) = 0;
};

bool U8_Compiler::costModelExceedsThreshhold(CC_List & ccs, unsigned from_pos, unsigned threshhold) {
    Range cc_span = CC_Set_Range(ccs);
    unsigned lgth = mEncoder.encoded_length(cc_span.hi);
    unsigned costSoFar = 0;
    for (auto cc : ccs) {
        unsigned ranges = cc->size();
        unsigned logic_cost = ranges * (lgth - from_pos + 1) * BinaryLogicCostPerByte;
        costSoFar += logic_cost;
        if (costSoFar > threshhold) return true;
    }
    return false;
}

void U8_Compiler::compile(Target_List targets, CC_List ccs) {
    //  Initialize all the target vars to 0.
    mTargets = targets;
    for (unsigned i = 0; i < targets.size(); i++) {
        mPB.createAssign(mTargets[i], mPB.createZeroes());
    }
    llvm::ArrayType * ty = cast<ArrayType>(mBasisVar->getType());
    unsigned streamCount = ty->getArrayNumElements();
    if (streamCount == mCodeUnitBits) {
        mScopeBasis[0].resize(mCodeUnitBits);
        for (unsigned i = 0; i < streamCount; i++) {
            mScopeBasis[0][i] = mPB.createExtract(mBasisVar, mPB.getInteger(i));
        }
        mCodeUnitCompiler[0] = std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[0]);
    } else {
        assert(streamCount == 1);  // A byte stream
        mScopeBasis[0].resize(1);
        mScopeBasis[0][0] = mPB.createExtract(mBasisVar, mPB.getInteger(0));
        mCodeUnitCompiler[0] = std::make_unique<cc::Direct_CC_Compiler>(mScopeBasis[0][0]);
    }
    createInitialHierarchy(ccs);
}

re::CC * U8_Compiler::codeUnitCC(re::CC * cc, unsigned pos) {
    CC_List singleton = {cc};
    return codeUnitCC(singleton, pos);
}

re::CC * U8_Compiler::codeUnitCC(CC_List & ccs, unsigned pos) {
    re::CC * unitCC = re::makeCC(&Byte);
    for (auto cc : ccs) {
        for (auto i : *cc) {
            unsigned lo_unit = mEncoder.nthCodeUnit(lo_codepoint(i), pos);
            unsigned hi_unit = mEncoder.nthCodeUnit(hi_codepoint(i), pos);
            unitCC = makeCC(unitCC, makeByte(lo_unit, hi_unit));
        }
    }
    return unitCC;
}

PabloAST * U8_Compiler::compileCodeUnit(U8_Seq_Kind k, Range enclosing_range, re::CC * unitCC, unsigned unitPos, PabloBuilder & pb) {
    Basis_Set & basis = getBasis(k, unitPos);
    if (basis.size() == 1) {
        // Byte stream - compile with direct CC compiler.
        return cc::Direct_CC_Compiler(basis[0]).compileCC(unitCC, pb);
    }
    if (SuffixOptimization && (unitPos > 1)) {
        unsigned significant_bits = enclosing_range.significant_bits();
        unsigned lgth = k + 1;
        unsigned code_unit_bit_offset = (lgth - unitPos) * suffixDataBits;
        unsigned significant_suffix_bits = significant_bits - code_unit_bit_offset;
        if (UTF_CompilationTracing) {
            llvm::errs() << "compileCodeUnit:" << enclosing_range.hex_string() << ", unitPos = " << unitPos;
            llvm::errs() << ", significant_suffix_bits = " << significant_suffix_bits << "\n";
        }
        Basis_Set suffixBasis(mCodeUnitBits);
        for (unsigned i = 0; i < significant_suffix_bits; i++) {
            suffixBasis[i] = basis[i];
        }
        for (unsigned i = significant_suffix_bits; i < suffixDataBits; i++) {
            if ((enclosing_range.lo >> (code_unit_bit_offset + i) & 1) == 1) {
                suffixBasis[i] = pb.createOnes();
            } else {
                suffixBasis[i] = pb.createZeroes();
            }
        }
        suffixBasis[6] = pb.createZeroes();
        suffixBasis[7] = pb.createOnes();
        return cc::Parabix_CC_Compiler_Builder(suffixBasis).compileCC(unitCC, pb);
    }
    return cc::Parabix_CC_Compiler_Builder(basis).compileCC(unitCC, pb);
}

std::vector<UCD::UnicodeSet> U8_Compiler::computeFullBlockSets(CC_List & ccs, unsigned pos) {
    std::vector<UCD::UnicodeSet> fullBlockSet(ccs.size());
    for (unsigned i = 0; i < ccs.size(); i++) {
        for (auto rg : *ccs[i]) {
            codepoint_t lo = lo_codepoint(rg);
            codepoint_t hi = hi_codepoint(rg);
            codepoint_t lo_base = mEncoder.minCodePointWithCommonCodeUnits(lo, pos);
            codepoint_t hi_ceil = mEncoder.maxCodePointWithCommonCodeUnits(hi, pos);
            if (UTF_CompilationTracing) {
                llvm::errs() << "pos = " << pos << "\n";
                llvm::errs() << "lo = " << lo << ", hi == " << hi << "\n";
                llvm::errs() << "lo_base = " << lo_base << ", hi_ceil == " << hi_ceil << "\n";
            }
            unsigned lo_unit = mEncoder.nthCodeUnit(lo, pos);
            unsigned hi_unit = mEncoder.nthCodeUnit(hi, pos);
            if (lo != lo_base) {
                lo_unit++;
                lo_base = mEncoder.maxCodePointWithCommonCodeUnits(lo, pos) + 1;
            }
            if (hi != hi_ceil) {
                hi_unit--;
                hi_ceil = mEncoder.minCodePointWithCommonCodeUnits(hi, pos) - 1;
            }
            if (lo_unit <= hi_unit) {
                fullBlockSet[i].insert_range(lo_base, hi_ceil);
            }
        }
    }
    return fullBlockSet;
}

void U8_Compiler::lengthAnalysis(CC_List & ccs) {
    for (unsigned k = ASCII; k <= FourByte; k++) {
        mSeqData[k].seqCCs.resize(ccs.size());
        extract_CCs_by_range(UTF8_Range[k], ccs, mSeqData[k].seqCCs);
        mSeqData[k].actualRange = CC_Set_Range(mSeqData[k].seqCCs);
        mSeqData[k].byte1CC = codeUnitCC(mSeqData[k].seqCCs, 1);
        mSeqData[k].test = nullptr;
        mSeqData[k].combinedTest = nullptr;
        mSeqData[k].suffixTest = nullptr;
        mSeqData[k].targets.resize(ccs.size());
        for (unsigned i = 0; i < ccs.size(); i++) {
            mSeqData[k].targets[i] = nullptr;
        }
    }
}

PabloAST * U8_Compiler::compilePrefix(re::CC * prefixCC, PabloBuilder & pb) {
    return mCodeUnitCompiler[0]->compileCC(prefixCC, pb);
}

void U8_Compiler::preparePrefixTests(PabloBuilder & pb) {
    for (unsigned k = TwoByte; k <= FourByte; k++) {
        if (mSeqData[k].actualRange.is_empty()) {
            mSeqData[k].test = nullptr;
        } else {
            mSeqData[k].test = compilePrefix(mSeqData[k].byte1CC, pb);
            mSeqData[k].testRange.lo = mEncoder.minCodePointWithCommonCodeUnits(mSeqData[k].actualRange.lo, 1);
            mSeqData[k].testRange.hi = mEncoder.maxCodePointWithCommonCodeUnits(mSeqData[k].actualRange.hi, 1);
        }
    }
    mSeqData[FourByte].combinedTest = mSeqData[FourByte].test;
    mSeqData[ThreeByte].combinedTest = combineOr(mSeqData[ThreeByte].test, mSeqData[FourByte].combinedTest, pb);
    mSeqData[TwoByte].combinedTest = combineOr(mSeqData[TwoByte].test, mSeqData[ThreeByte].combinedTest, pb);
}


void U8_Compiler::createInitialHierarchy(CC_List & ccs) {
    lengthAnalysis(ccs);
    for (unsigned k = ASCII; k <= FourByte; k++) {
        for (unsigned i = 0; i < ccs.size(); i++) {
            if (!mSeqData[k].actualRange.is_empty()) {
                std::string tname = "tgt" + std::to_string(i) + "_len" + std::to_string(k+1);
                mSeqData[k].targets[i] = mPB.createVar(tname, mPB.createZeroes());
            }
        }
    }
    if (!mSeqData[ASCII].actualRange.is_empty()) {
        re::CC * ASCII_CC = re::makeCC(0x0, 0x7F, &Byte);
        PabloAST * ASCII_test = mCodeUnitCompiler[0]->compileCC(ASCII_CC, mPB);
        Range ASCII_Range{0, 0x7F};
        EnclosingInfo ASCII_info(ASCII_Range, ASCII_test);
        compileFromCodeUnit(ASCII, ASCII_info, ASCII_info, 1, mPB);
        for (unsigned i = 0; i < ccs.size(); i++) {
            mPB.createAssign(mTargets[i], mSeqData[ASCII].targets[i]);
        }
    }
    CC_List nonASCII_ccs(ccs.size());
    Range nonASCII_Range{0x80, 0x10FFFF};
    extract_CCs_by_range(nonASCII_Range, ccs, nonASCII_ccs);
    Range actual_subrange = CC_Set_Range(nonASCII_ccs);
    if (!actual_subrange.is_empty()) {
        unsigned pfx_count = 0;
        for (unsigned k = TwoByte; k <= FourByte; k++) {
            pfx_count += mSeqData[k].byte1CC->count();
        }
        if ((InitialTest == InitialTestMode::PrefixCC) && (pfx_count > PrefixTestMax)) {
            InitialTest = InitialTestMode::NonASCII;
        }
        if (InitialTest == InitialTestMode::NonASCII) {
            re::CC * nonASCII = re::makeCC(0x80, 0xFF, &Byte);
            PabloAST * nonASCII_test = combineAnd(mMask, mCodeUnitCompiler[0]->compileCC(nonASCII, mPB), mPB);
            auto nested = mPB.createScope();
            mPB.createIf(nonASCII_test, nested);
            preparePrefixTests(nested);
            EnclosingInfo nonASCIIinfo(nonASCII_Range, nonASCII_test);
            extendLengthHierarchy(nonASCIIinfo, nested);
        } else {
            codepoint_t test_min = mEncoder.minCodePointWithCommonCodeUnits(actual_subrange.lo, 1);
            codepoint_t test_max = mEncoder.maxCodePointWithCommonCodeUnits(actual_subrange.hi, 1);
            Range testRange{test_min, test_max};
            PabloAST * rangeTest = nullptr;
            if (InitialTest == InitialTestMode::RangeCC) {
                unsigned lo_prefix = mEncoder.nthCodeUnit(actual_subrange.lo, 1);
                unsigned hi_prefix = mEncoder.nthCodeUnit(actual_subrange.hi, 1);
                re::CC * prefix_range_CC = re::makeCC(lo_prefix, hi_prefix, &Byte);
                rangeTest = combineAnd(mMask, compilePrefix(prefix_range_CC, mPB), mPB);
            } else { // InitialTest == InitialTestMode::PrefixCC
                preparePrefixTests(mPB);
                rangeTest = mSeqData[TwoByte].combinedTest;
            }
            auto nested = mPB.createScope();
            mPB.createIf(rangeTest, nested);
            if (InitialTest == InitialTestMode::RangeCC) {
                preparePrefixTests(nested);
            }
            EnclosingInfo multibyteInfo(testRange, rangeTest);
            extendLengthHierarchy(multibyteInfo, nested);
        }
    }
}

void U8_Compiler::extendLengthHierarchy(EnclosingInfo & initialInfo, PabloBuilder & pb) {
    prepareFixedLengthHierarchy(TwoByte, initialInfo, pb);
    PabloAST * prefix34_test = mSeqData[ThreeByte].combinedTest;
    if (prefix34_test == nullptr) return;  // No code generation required.
    if (prefix34_test == initialInfo.test) {
        // No nesting required.
        prepareFixedLengthHierarchy(ThreeByte, initialInfo, pb);
        prepareFixedLengthHierarchy(FourByte, initialInfo, pb);
    } else {
        auto nested = pb.createScope();
        pb.createIf(prefix34_test, nested);
        if (mSeqData[FourByte].actualRange.is_empty()) {
            EnclosingInfo threeByteInfo(mSeqData[ThreeByte].testRange, mSeqData[ThreeByte].test);
            prepareFixedLengthHierarchy(ThreeByte, threeByteInfo, nested);
        } else if (mSeqData[ThreeByte].actualRange.is_empty()) {
            EnclosingInfo fourByteInfo(mSeqData[FourByte].testRange, mSeqData[FourByte].test);
            prepareFixedLengthHierarchy(FourByte, fourByteInfo, nested);
        } else {
            Range range34{mSeqData[ThreeByte].testRange.lo, mSeqData[FourByte].testRange.hi};
            EnclosingInfo enclosing34info(range34, prefix34_test);
            prepareFixedLengthHierarchy(ThreeByte, enclosing34info, nested);
            prepareFixedLengthHierarchy(FourByte, enclosing34info, nested);
        }
    }
}

CC_List U8_Compiler::prepareFullBlockSets(U8_Seq_Kind k, Range enclosing_range, CC_List ccs, PabloAST * enclosing_test, unsigned code_unit, PabloBuilder & pb) {
    std::vector<UCD::UnicodeSet> fullBlockSets = computeFullBlockSets(ccs, code_unit);
    CC_List remainingCCs(ccs.size());
    for (unsigned i = 0; i < ccs.size(); i++) {
        if (fullBlockSets[i].empty()) {
            remainingCCs[i] = ccs[i];
            continue;
        }
        re::CC * fullBlockCC = re::makeCC(fullBlockSets[i], &Unicode);
        re::CC * unitCC = codeUnitCC(fullBlockCC, code_unit);
        unsigned lgth = k + 1;
        PabloAST * t = nullptr;
        if (code_unit == 1) {
            t = adjustPosition(compilePrefix(unitCC, pb), 1, lgth, pb);
        } else {
            t = compileCodeUnit(k, enclosing_range, unitCC, code_unit, pb);
            t = pb.createAnd(enclosing_test, t);
            if (!AdvanceBasis) {
                t = adjustPosition(t, code_unit, lgth, pb);
            }
        }
        if (UTF_CompilationTracing) {
            llvm::errs() << "prepareFullBlockSets[" ;
            llvm::errs().write_hex(fullBlockCC->min_codepoint());
            llvm::errs() << ", ";
            llvm::errs().write_hex(fullBlockCC->max_codepoint());
            llvm::errs() << "]\n";
        }
        pb.createAssign(mSeqData[k].targets[i], pb.createOr(mSeqData[k].targets[i], t));
        remainingCCs[i] = subtractCC(ccs[i], fullBlockCC);
    }
    return remainingCCs;
}

void U8_Compiler::prepareFixedLengthHierarchy(U8_Seq_Kind k, EnclosingInfo & if_parent, PabloBuilder & pb) {
    // No code generation if there are no CCs within this range.
    if (mSeqData[k].actualRange.is_empty()) return;
    //
    prepareSuffix(k, pb);
    CC_List narrowedCCs = prepareFullBlockSets(k, if_parent.range, mSeqData[k].seqCCs, mSeqData[k].test, 1, pb);
    re::CC * narrowedPrefixCC = codeUnitCC(narrowedCCs, 1);
    if (mSeqData[k].byte1CC->subset(*narrowedPrefixCC)) {
        prepareScope(k, pb);
        compileFromCodeUnit(k, if_parent, if_parent, 1, pb);
    } else {
        Range narrowed = CC_Set_Range(narrowedCCs);
        if (!narrowed.is_empty()) {
            PabloAST * t = compilePrefix(narrowedPrefixCC, pb);
            prepareScope(k, pb);
            auto lo_pfx = mSeqData[k].byte1CC->min_codepoint();
            auto hi_pfx = mSeqData[k].byte1CC->max_codepoint();
            Range tested{mEncoder.minCodePointWithPrefix(lo_pfx), mEncoder.maxCodePointWithPrefix(hi_pfx)};
            EnclosingInfo rangeInfo(tested, t);
            compileFromCodeUnit(k, if_parent, rangeInfo, 1, pb);
        }
    }
    for (unsigned i = 0; i < mSeqData[k].seqCCs.size(); i++) {
        PabloAST * CC_test = pb.createAnd(mSeqData[k].targets[i], mSeqData[k].suffixTest);
        pb.createAssign(mTargets[i], pb.createOr(mTargets[i], CC_test));
    }
}

//
// Precondition: partial compilation of CCs within the enclosing.range has
// been completed and is available as a PabloAST in the enclosing.test.
// Precondition: code_unit >= 1 and the enclosing.range has a partial
// UTF-8 sequence that is an exact sequence of bytes for prior units.
void U8_Compiler::compileFromCodeUnit(U8_Seq_Kind k, EnclosingInfo & if_parent, EnclosingInfo & enclosing, unsigned code_unit, PabloBuilder & pb) {
    unsigned lgth = mEncoder.encoded_length(enclosing.range.hi);
    CC_List rangeCCs(mSeqData[k].seqCCs.size());
    extract_CCs_by_range(enclosing.range, mSeqData[k].seqCCs, rangeCCs);
    PabloAST * enclosing_test = enclosing.test;
    if (!AdvanceBasis) {
        enclosing_test = adjustPosition(enclosing_test, enclosing.testPosition, code_unit, pb);
    }
    if (UTF_CompilationTracing) {
        llvm::errs() << "compileFromCodeUnit(" << enclosing.range.hex_string() << ")\n";
        llvm::errs() << "  enclosing.testPosition = " << enclosing.testPosition << "\n";
        llvm::errs() << "  code_unit = " << code_unit << "\n";
    }
    if (lgth == code_unit) {  // At the final code unit position
        for (unsigned i = 0; i < rangeCCs.size(); i++) {
            re::CC * unitCC = codeUnitCC(rangeCCs[i], code_unit);
            PabloAST * compiled = compileCodeUnit(k, enclosing.range, unitCC, code_unit, pb);
            if (code_unit > 1) {
                compiled = pb.createAnd(enclosing_test, compiled, "compiled");
            }
            pb.createAssign(mSeqData[k].targets[i], pb.createOr(mSeqData[k].targets[i], compiled));
        }
    } else {
        // First deal with the cases that some CCs are full for entire
        // ranges associated with some code units at this level.
        rangeCCs = prepareFullBlockSets(k, enclosing.range, rangeCCs, enclosing_test, code_unit, pb);
        //
        // Now process the individual code units and directly compile low cost sequences.
        re::CC * unionCC = rangeCCs[0];
        for (unsigned i = 1; i < rangeCCs.size(); i++) {
            unionCC = re::makeCC(unionCC, rangeCCs[i]);
        }
        codepoint_t lo = unionCC->min_codepoint();
        codepoint_t unit_range_base = mEncoder.minCodePointWithCommonCodeUnits(lo, code_unit);
        auto unit_range = mEncoder.maxCodePointWithCommonCodeUnits(lo, code_unit) - unit_range_base + 1;
        auto if_parent_range = if_parent.range.hi - if_parent.range.lo + 1;
        bool nesting_allowed = unit_range * PartitioningFactor <= if_parent_range;

        re::CC * unit_CC = codeUnitCC(unionCC, code_unit);
        auto lo_unit = unit_CC->min_codepoint();
        auto hi_unit = unit_CC->max_codepoint();
        unsigned units_to_process = unit_CC->count();
        unsigned partitions = PartitioningFactor;
        if (units_to_process < partitions) partitions = units_to_process;
        if (!PartitioningRevision) nesting_allowed = true;

        bool nesting = nesting_allowed && costModelExceedsThreshhold(rangeCCs, code_unit, IfEmbeddingCostThreshhold);

        codepoint_t partition_base = unit_range_base;
        unsigned units_this_partition = 0;
        unsigned unit_count = 0;
        re::CC * partitionCC = nullptr;
        for (unsigned unit = lo_unit; unit <= hi_unit; unit++) {
            codepoint_t unit_range_top = mEncoder.maxCodePointWithCommonCodeUnits(unit_range_base, code_unit);
            if (unit_CC->contains(unit)) {
                if (unit_count == 0) {
                    partition_base = unit_range_base;
                    units_this_partition = units_to_process/partitions;
                    partitionCC = re::makeCC(&Byte);
                }
                partitionCC = re::makeCC(partitionCC, re::makeCC(unit, &Byte));
                unit_count++;
                if (unit_count == units_this_partition) {
                    Range partitionRange{partition_base, unit_range_top};
                    PabloAST * t = compileCodeUnit(k, enclosing.range, partitionCC, code_unit, pb);
                    if ((code_unit > 1) || (mMask != nullptr)) {
                        t = pb.createAnd(enclosing_test, t, "rg_" + partitionRange.hex_string());
                    }
                    EnclosingInfo partitionInfo(partitionRange, t, code_unit);
                    if (nesting) {
                        auto nested = pb.createScope();
                        pb.createIf(t, nested);
                        if (units_this_partition == 1) {
                            compileFromCodeUnit(k, partitionInfo, partitionInfo, code_unit + 1, nested);
                        } else {
                            compileFromCodeUnit(k, partitionInfo, partitionInfo, code_unit, nested);
                        }
                    } else {
                        if (units_this_partition == 1) {
                            compileFromCodeUnit(k, if_parent, partitionInfo, code_unit + 1, pb);
                        } else {
                            compileFromCodeUnit(k, if_parent, partitionInfo, code_unit, pb);
                        }
                    }
                    // Partition is done; reset for next partition.
                    unit_count = 0;
                    units_to_process -= units_this_partition;
                    partitions--;
                }
            }
            unit_range_base = unit_range_top + 1;  // for next unit in loop
        }
    }
}

class U8_Lookahead_Compiler : public U8_Compiler {
public:
    U8_Lookahead_Compiler(pablo::Var * Var, PabloBuilder & pb, pablo::PabloAST * mask);
protected:
    void prepareSuffix(unsigned scope, PabloBuilder & pb) override;
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    Basis_Set & getBasis(U8_Seq_Kind k, unsigned pos) override;
    PabloAST * adjustPosition(PabloAST * t, unsigned from, unsigned to, PabloBuilder & pb) override;
};

class U8_Advance_Compiler : public U8_Compiler {
public:
    U8_Advance_Compiler(pablo::Var * Var, PabloBuilder & pb, pablo::PabloAST * mask);
protected:
    PabloAST * mSuffix;
    bool costModelExceedsThreshhold(CC_List & ccs, unsigned from_pos, unsigned threshhold) override;
    void prepareSuffix(unsigned scope, PabloBuilder & pb) override;
    void prepareScope(unsigned scope, PabloBuilder & pb) override;
    Basis_Set & getBasis(U8_Seq_Kind k, unsigned pos) override;
    PabloAST * adjustPosition(PabloAST * t, unsigned from, unsigned to, PabloBuilder & pb) override;
};

U8_Lookahead_Compiler::U8_Lookahead_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mask) :
U8_Compiler(v, pb, mask) {
}

void U8_Lookahead_Compiler::prepareSuffix(unsigned scope, PabloBuilder & pb) {
    if (mSeqData[scope].byte1CC->empty()) return;
    for (unsigned sfx = 1; sfx <= scope; sfx++) {
        if (mSeqData[sfx].suffixTest == nullptr) {
            if (mScopeBasis[0].size() == 8) {
                PabloAST * sfxbit7 = pb.createLookahead(mScopeBasis[0][7], sfx);
                PabloAST * sfxbit6 = pb.createLookahead(mScopeBasis[0][6], sfx);
                mSeqData[sfx].suffixTest = pb.createAnd(sfxbit7, pb.createNot(sfxbit6));
            } else {
                prepareScope(scope, pb);
                re::CC * suffixCC = re::makeCC(0x80, 0xBF, &Byte);
                mSeqData[sfx].suffixTest = mCodeUnitCompiler[scope]->compileCC(suffixCC, pb);
            }
            if (sfx > 1) {
                mSeqData[sfx].suffixTest = pb.createAnd(mSeqData[sfx-1].suffixTest, mSeqData[sfx].suffixTest);
            }
        }
    }
}

void U8_Lookahead_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    if (mSeqData[scope].byte1CC->empty()) return;
    for (unsigned sfx = 1; sfx <= scope; sfx++) {
        bool basis_needed = mScopeBasis[sfx].size() == 0;
        if (basis_needed) {
            mScopeBasis[sfx].resize(mScopeBasis[0].size());
            if (mScopeBasis[0].size() == 1) {
                mScopeBasis[sfx][0] = pb.createLookahead(mScopeBasis[0][0], sfx);
                mCodeUnitCompiler[sfx] = std::make_unique<cc::Direct_CC_Compiler>(mScopeBasis[sfx][0]);
            } else {
                for (unsigned i = 0; i < 6; i++) {
                    mScopeBasis[sfx][i] = pb.createLookahead(mScopeBasis[0][i], sfx);
                }
                // Set the expected suffix bits - the final suffixTest will confirm.
                mScopeBasis[sfx][6] = pb.createZeroes();
                mScopeBasis[sfx][7] = pb.createOnes();
                mCodeUnitCompiler[sfx] = std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[sfx]);
            }
        }
    }
}

PabloAST *  U8_Lookahead_Compiler::adjustPosition(PabloAST * t, unsigned from, unsigned to, PabloBuilder & pb) {
    // The lookahead compiler calculates everything at the first byte position,
    // so no adjustment is needed.
    return t;
}

Basis_Set & U8_Lookahead_Compiler::getBasis(U8_Seq_Kind k, unsigned unitPos) {
    return mScopeBasis[unitPos - 1];
}

U8_Advance_Compiler::U8_Advance_Compiler(pablo::Var * v, PabloBuilder & pb, pablo::PabloAST * mMask) :
U8_Compiler(v, pb, mMask), mSuffix(nullptr) {
}

bool U8_Advance_Compiler::costModelExceedsThreshhold(CC_List & ccs, unsigned from_pos, unsigned threshhold) {
    Range cc_span = CC_Set_Range(ccs);
    unsigned lgth = mEncoder.encoded_length(cc_span.hi);
    unsigned costSoFar = 0;
    for (auto cc : ccs) {
        unsigned ranges = cc->size();
        unsigned logic_cost = ranges * (lgth - from_pos + 1) * BinaryLogicCostPerByte;
        costSoFar += logic_cost;
        if ((from_pos < lgth) && !AdvanceBasis) {
            unsigned cc_range = cc->max_codepoint() - cc->min_codepoint();
            unsigned final_shifts = cc_range/64;
            unsigned shift_cost = final_shifts * ShiftCostFactor;
            costSoFar += shift_cost;
        }
        if (costSoFar > threshhold) return true;
    }
    return false;
}

void U8_Advance_Compiler::prepareSuffix(unsigned scope, PabloBuilder & pb) {
    if (mSeqData[scope].byte1CC->empty()) return;
    if (mSuffix == nullptr) {
        re::CC * suffixCC = re::makeCC(0x80, 0xBF, &Byte);
        mSuffix = mCodeUnitCompiler[0]->compileCC(suffixCC, pb);
        mSeqData[1].suffixTest = mSuffix;
    }
    for (unsigned sfx = 2; sfx <= scope; sfx++) {
        if (mSeqData[sfx].suffixTest == nullptr) {
            PabloAST * adv_suffix = pb.createAdvance(mSeqData[sfx-1].suffixTest, 1);
            mSeqData[sfx].suffixTest = pb.createAnd(adv_suffix, mSuffix, "scope" + std::to_string(sfx) + "_sfx_test");
        }
    }
}

void U8_Advance_Compiler::prepareScope(unsigned scope, PabloBuilder & pb) {
    if (mSeqData[scope].byte1CC->empty()) return;
    if (AdvanceBasis) {
        const unsigned suffixBits = 8;
        for (unsigned sfx = 1; sfx <= scope; sfx++) {
            if (UTF_CompilationTracing) {
                llvm::errs() << "preparing Advanced basis " << sfx << "\n";
            }
            unsigned sfx_bits = mScopeBasis[sfx].size();
            if (sfx_bits < suffixBits) {
                mScopeBasis[sfx].resize(suffixBits);
                for (unsigned i = sfx_bits; i < suffixBits; i++) {
                    mScopeBasis[sfx][i] = pb.createAdvance(mScopeBasis[sfx-1][i], 1);
                }
                mCodeUnitCompiler[sfx] = std::make_unique<cc::Parabix_CC_Compiler_Builder>(mScopeBasis[sfx]);
            }
        }
    }
}

Basis_Set & U8_Advance_Compiler::getBasis(U8_Seq_Kind k, unsigned unitPos) {
    if (AdvanceBasis) {
        unsigned lgth = k + 1;
        return mScopeBasis[lgth-unitPos];
    }
    return mScopeBasis[0];
}

PabloAST *  U8_Advance_Compiler::adjustPosition(PabloAST * t, unsigned from, unsigned to, PabloBuilder & pb) {
    if (from == to) return t;
    return pb.createAdvance(t, to - from, "adjust" + std::to_string(from) + "_" + std::to_string(to));
}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::PabloAST * mask,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(mask), mBitMovement(mode) {
}

UTF_Compiler::UTF_Compiler(pablo::Var * basisVar, pablo::PabloBuilder & pb,
             pablo::BitMovementMode mode) :
mVar(basisVar), mPB(pb), mMask(nullptr), mBitMovement(mode) {}

void UTF_Compiler::compile(Target_List targets, CC_List ccs) {
    llvm::ArrayType * ty = cast<ArrayType>(mVar->getType());
    unsigned streamCount = ty->getArrayNumElements();
    unsigned codeUnitBits;
    if (streamCount == 1) {
        VectorType * const vt = cast<VectorType>(ty->getArrayElementType());
        const auto streamWidth = vt->getElementType()->getIntegerBitWidth();
        codeUnitBits = streamWidth;
    } else {
        codeUnitBits = streamCount;
    }
    if (codeUnitBits == 8) {
        if (mBitMovement == pablo::BitMovementMode::LookAhead) {
            U8_Lookahead_Compiler utf_compiler(mVar, mPB, mMask);
            utf_compiler.compile(targets, ccs);
        } else {
            U8_Advance_Compiler utf_compiler(mVar, mPB, mMask);
            utf_compiler.compile(targets, ccs);
        }
    } else {
        U21_Compiler u21_compiler(mVar, mPB, mMask);
        u21_compiler.compile(targets, ccs);
    }
}

}
