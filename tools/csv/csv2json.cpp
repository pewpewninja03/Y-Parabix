/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */


#include <cstdio>
#include <vector>
#include <csv/csv_cmdline.h>
#include <csv/csv_parser.h>
#include <json/json_support.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/IR/Module.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/sentinel.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/util/linebreak_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/re/regexp_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/adt.h>
#include <re/adt/re_re.h>
#include <re/parse/parser.h>
#include <re/unicode/regex_passes.h>
#include <grep/grep_engine.h>
#include <grep/grep_kernel.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pablo_kernel.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::ceil_log2;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

using namespace kernel;
using namespace llvm;
using namespace pablo;

static cl::opt<bool> TestDynamicRepeatingFile("dyn", cl::desc("Test Dynamic Repeating StreamSet"), cl::init(true), cl::cat(csv::CSV_Options), cl::Hidden);
static cl::opt<bool> UseMergeByMaskKernel("merge-by-mask", cl::desc("Use MergeByMask kernel"), cl::init(false), cl::cat(csv::CSV_Options), cl::Hidden);

typedef void (*CSVFunctionType)(uint32_t fd);

inline void MergeByMask01(PipelineBuilder & P, StreamSet * mask, StreamSet * a, StreamSet * b, StreamSet * merged) {
    unsigned elems = merged->getNumElements();
    if ((a->getNumElements() != elems) || (b->getNumElements() != elems)) {
        llvm::report_fatal_error("MergeByMask called with incompatible element counts");
    }
    StreamSet * expandedA = P.CreateStreamSet(elems);
    SpreadByMask(P, mask, a, expandedA);
    StreamSet * inverted = P.CreateStreamSet(1);
    Invert(P, mask, inverted);
    StreamSet * expandedB = P.CreateStreamSet(elems);
    SpreadByMask(P, inverted, b, expandedB);
    OrCombine(P, expandedA, expandedB, merged);
}

std::vector<std::string> JSONfieldPrefixes(std::vector<std::string> fieldNames) {
    std::vector<std::string> tmp;
    if (fieldNames.size() == 0) return tmp;
    for (unsigned i = 0; i < fieldNames.size(); i++) {
        tmp.push_back("\"" + fieldNames[i] + "\":");
    }
    tmp[0] = "{" + tmp[0];
    return tmp;
}

class CSVdataFieldMask : public PabloKernel {
public:
    CSVdataFieldMask(LLVMTypeSystemInterface & ts, StreamSet * csvMarks, StreamSet * EOFbit, StreamSet * recordSeparators, StreamSet * quoteEscape,
        StreamSet * toKeep, bool deleteHeader = true)
        : PabloKernel(ts, "CSVdataFieldMask" + std::to_string(deleteHeader),
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)},
                       Binding{"EOFbit", EOFbit, FixedRate(), LookAhead(1)},
                       Binding{"recordSeparators", recordSeparators},
                       Binding{"quoteEscape", quoteEscape}},
                      {Binding{"toKeep", toKeep}})
    , mDeleteHeader(deleteHeader) {}
protected:
    void generatePabloMethod() override;
    bool mDeleteHeader;
};

void CSVdataFieldMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * recordMarks = pb.createExtract(getInputStreamVar("recordSeparators"), pb.getInteger(0));
    PabloAST * EOFmark = pb.createExtract(getInputStreamVar("EOFbit"), pb.getInteger(0));
    PabloAST * quoteEscape = pb.createExtract(getInputStreamVar("quoteEscape"), pb.getInteger(0));
    PabloAST * CRbeforeLF = pb.createAnd(csvMarks[csv::markCR], pb.createLookahead(csvMarks[csv::markLF], 1));
    PabloAST * escaped_quote = pb.createAdvance(quoteEscape, 1);
    PabloAST * fieldQuotes = pb.createAnd(csvMarks[csv::markDQ], pb.createNot(pb.createOr(quoteEscape, escaped_quote)));
    PabloAST * toDelete = CRbeforeLF;//pb.createOr(CRbeforeLF, fieldQuotes);
    if (mDeleteHeader) {
        PabloAST * afterHeader = pb.createMatchStar(pb.createAdvance(recordMarks, 1), pb.createOnes());
        toDelete = pb.createOr(toDelete, pb.createNot(afterHeader));
    }
    // Delete the final LF position, so that we won't generate a template string at EOF.
    //toDelete = pb.createOr(toDelete, pb.createAnd(recordMarks, pb.createLookahead(EOFmark, 1)));
    // Also delete the final EOFbit position generated by the Add1 attribute of the CSV lexer to avoid a null.
    //toDelete = pb.createOr(toDelete, EOFmark);
    PabloAST * toKeep = pb.createInFile(pb.createNot(toDelete));
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), toKeep);
}

class QuoteEscape2Backslash : public PabloKernel {
public:
    QuoteEscape2Backslash(LLVMTypeSystemInterface & ts, StreamSet * quoteEscape, StreamSet * basis,
                         StreamSet * translatedBasis)
        : PabloKernel(ts, std::string("QuoteEscape2Backslash") + (codegen::DebugOptionIsSet(codegen::DisableInOutAttributes) ? "-InOut" : ""),
                      {Binding{"quoteEscape", quoteEscape}, Binding{"basis", basis}},
                      {}) {
    mUseInOut = !codegen::DebugOptionIsSet(codegen::DisableInOutAttributes);
    if (mUseInOut) {
        mOutputStreamSets.push_back(Binding{"translatedBasis", translatedBasis, FixedRate(), InOut("basis")});
    } else {
        mOutputStreamSets.push_back(Binding{"translatedBasis", translatedBasis});
    }
}
protected:
    void generatePabloMethod() override;
private:
    bool mUseInOut;
};

void QuoteEscape2Backslash::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * quoteEscape = getInputStreamSet("quoteEscape")[0];
    auto nested = pb.createScope();
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    Var * outputVar = getOutputStreamVar("translatedBasis");
    pb.createIf(quoteEscape, nested);
    //
    // Replace the quote escape character with \ = 0x5C
    std::vector<PabloAST *> translated_basis(8, nullptr);
    PabloAST * notQuoteEscape = nested.createNot(quoteEscape);
    // Low 2 bits zeroed out whenever we have quoteEscape
    translated_basis[0] = nested.createAnd(basis[0], notQuoteEscape);
    translated_basis[1] = nested.createAnd(basis[1], notQuoteEscape);
    // Next 2 bits set whenever we have quoteEscape
    translated_basis[2] = nested.createOr(basis[2], quoteEscape);
    translated_basis[3] = nested.createOr(basis[3], quoteEscape);

    translated_basis[4] = nested.createOr(basis[4], quoteEscape);
    translated_basis[5] = nested.createAnd(basis[5], notQuoteEscape);
    translated_basis[6] = nested.createOr(basis[6], quoteEscape);
    translated_basis[7] = nested.createAnd(basis[7], notQuoteEscape);
    if (mUseInOut) {
        for (unsigned i = 0; i < 8; i++) {
            nested.createAssign(nested.createExtract(outputVar, nested.getInteger(i)), translated_basis[i]);
        } 
    } else {
        for (unsigned i = 0; i < 8; i++) {
            pb.createAssign(pb.createExtract(outputVar, pb.getInteger(i)), translated_basis[i]);
        } 
    }
}

class FinalCommaToRBracket : public PabloKernel {
public:
    FinalCommaToRBracket(LLVMTypeSystemInterface & ts, StreamSet * basis, StreamSet * EOFmark,
                         StreamSet * translatedBasis)
        : PabloKernel(ts, std::string("FinalCommaToRBracket") + (codegen::DebugOptionIsSet(codegen::DisableInOutAttributes) ? "-InOut" : ""),
                      {Binding{"basis", basis}, Binding{"EOFmark", EOFmark, FixedRate(), LookAhead(2)}},
                      {}) {
    mUseInOut = !codegen::DebugOptionIsSet(codegen::DisableInOutAttributes);
    if (mUseInOut) {
        mOutputStreamSets.push_back(Binding{"translatedBasis", translatedBasis, FixedRate(), InOut("basis")});
    } else {
        mOutputStreamSets.push_back(Binding{"translatedBasis", translatedBasis});
    }
}
protected:
    void generatePabloMethod() override;
    void translationLogic(PabloBuilder & pb, PabloAST * FinalCommaPos, std::vector<PabloAST *> & basis);
private:
    bool mUseInOut;
};

void FinalCommaToRBracket::translationLogic(PabloBuilder & pb, PabloAST * FinalCommaPos, std::vector<PabloAST *> & basis) {
    std::vector<PabloAST *> translated_basis(basis.size());
    Var * outputVar = getOutputStreamVar("translatedBasis");
    // Replace the comma (0x2C) character with ] = 0x5D
    pb.createAssign(pb.createExtract(outputVar, pb.getInteger(0)), pb.createOr(FinalCommaPos, basis[0]));
    for (unsigned i = 1; i < 4; i++) {
        pb.createAssign(pb.createExtract(outputVar, pb.getInteger(i)), basis[i]);
    }
    for (unsigned i = 4; i < 7; i++) {
        translated_basis[i] = pb.createXor(basis[i], FinalCommaPos, "xlated" + std::to_string(i));
        pb.createAssign(pb.createExtract(outputVar, pb.getInteger(i)), translated_basis[i]);
    }
    for (unsigned i = 7; i < basis.size(); i++) {
        pb.createAssign(pb.createExtract(outputVar, pb.getInteger(i)), basis[i]);
    }
}

void FinalCommaToRBracket::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * EOFmark = getInputStreamSet("EOFmark")[0];
    PabloAST * FinalCommaPos = pb.createLookahead(EOFmark, 2);

    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    if (mUseInOut) {
        auto nested = pb.createScope();
        pb.createIf(FinalCommaPos, nested);
        translationLogic(nested, FinalCommaPos, basis);
    } else {
        translationLogic(pb, FinalCommaPos, basis);
    }
}

class CalcQuoteBixNum : public PabloKernel {
public:
    CalcQuoteBixNum(LLVMTypeSystemInterface & ts, StreamSet * stringStarts, StreamSet * stringFollows,
                         StreamSet * quoteBixNum)
        : PabloKernel(ts, std::string("CalcQuoteBixNum"),
                      {Binding{"stringStarts", stringStarts}, Binding{"stringFollows", stringFollows}},
                      {Binding{"quoteBixNum", quoteBixNum}}) {}
protected:
    void generatePabloMethod() override;
};

void CalcQuoteBixNum::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * stringStarts = getInputStreamSet("stringStarts")[0];
    PabloAST * stringFollows = getInputStreamSet("stringFollows")[0];

    std::vector<PabloAST *> insertBixNum(2);
    insertBixNum[0] = pb.createXor(stringStarts, stringFollows);
    insertBixNum[1] = pb.createAnd(stringStarts, stringFollows);
    writeOutputStreamSet("quoteBixNum", insertBixNum);
}

class EnQuote : public PabloKernel {
public:
    EnQuote(LLVMTypeSystemInterface & ts, StreamSet * basis, StreamSet * spreadMask,
                         StreamSet * quotedBasis)
        : PabloKernel(ts, std::string("EnQuote") + basis->shapeString(),
                      {Binding{"basis", basis}, Binding{"spreadMask", spreadMask}},
                      {Binding{"quotedBasis", quotedBasis}}) {}
protected:
    void generatePabloMethod() override;
};

void EnQuote::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    PabloAST * spreadMask = getInputStreamSet("spreadMask")[0];
    PabloAST * quotePos = pb.createNot(spreadMask);
    //
    // Convert the nulls at quotePos to 0x22 (double quote)
    // This requires only setting bits 1 and 5
    basis[1] = pb.createOr(basis[1], quotePos);
    basis[5] = pb.createOr(basis[5], quotePos);
    writeOutputStreamSet("quotedBasis", basis);
}


void JSON_Value_Quoting(PipelineBuilder & P, StreamSet * BasisBits, StreamSet * fieldStarts, StreamSet * fieldFollows, StreamSet * QuotedBasis) {
    StreamSet * literalMatches = P.CreateStreamSet(1);
    json::JSON_ValueKind no_quotes_needed_set =
        static_cast<json::JSON_ValueKind>(json::NullLiteral|
                                          json::NumericLiteral|
                                          json::TrueLiteral|
                                          json::FalseLiteral|
                                          json::QuotedString);

    JSON_Value_Matching(P, no_quotes_needed_set, BasisBits, fieldStarts, fieldFollows, literalMatches);

    StreamSet * const literalFollows = P.CreateStreamSet(1);
    P.CreateKernelCall<MatchedLinesKernel>(literalMatches, fieldFollows, literalFollows);
    SHOW_STREAM(literalFollows);

    StreamSet * stringFollows = P.CreateStreamSet(1);
    XorCombine(P, literalFollows, fieldFollows, stringFollows);
    SHOW_STREAM(stringFollows);

    StreamSet * stringsByField = P.CreateStreamSet(1);
    FilterByMask(P, fieldFollows, stringFollows, stringsByField);
    SHOW_STREAM(stringsByField);

    StreamSet * const stringStarts = P.CreateStreamSet(1);
    SpreadByMask(P, fieldStarts, stringsByField, stringStarts);
    SHOW_STREAM(stringStarts);

    StreamSet * const quoteInsertBixNum = P.CreateStreamSet(2);
    P.CreateKernelCall<CalcQuoteBixNum>(stringStarts, stringFollows, quoteInsertBixNum);
    SHOW_BIXNUM(quoteInsertBixNum);

    StreamSet * const BasisSpreadMask = P.CreateStreamSet(1);
    InsertionSpreadMask(P, quoteInsertBixNum, BasisSpreadMask, kernel::InsertPosition::Before);
    SHOW_STREAM(BasisSpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(BasisBits->getNumElements());
    SpreadByMask(P, BasisSpreadMask, BasisBits, ExpandedBasis);

    P.CreateKernelCall<EnQuote>(ExpandedBasis, BasisSpreadMask, QuotedBasis);
    SHOW_BIXNUM(QuotedBasis);
}

CSVFunctionType generatePipeline(CPUDriver & driver, const std::vector<std::string> & templateStrs) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.
    auto P = CreatePipeline(driver, Input<uint32_t>{"inputFileDecriptor"});
    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");
    // File data from mmap
    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    //  ReadSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    SHOW_BYTES(ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P.CreateStreamSet(8);
    Selected_S2P(P, ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    //  We need to know which input positions are dquotes and which are not.
    StreamSet * csvCCs = P.CreateStreamSet(4);
    csv::CSV_Lexer(P, BasisBits, csvCCs);

    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldStarts = P.CreateStreamSet(1);
    StreamSet * fieldFollows = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldStarts, fieldFollows, quoteEscape);
    StreamSet * EOFmark = P.CreateStreamSet(1);
    P.CreateKernelCall<EOFbit>(BasisBits, EOFmark);

    StreamSet * toKeep = P.CreateStreamSet(1);
    P.CreateKernelCall<CSVdataFieldMask>(csvCCs, EOFmark, recordSeparators, quoteEscape, toKeep, csv::HeaderSpec == "");
    SHOW_STREAM(toKeep);
    //
    // Create a short stream which is 1-to-1 with the (field/record) separators,
    // having 0 bits for field separators and 1 bits for record separators.
    // Normally this will be a stream having exactly one bit set for every
    // N positions, where N is the number of entries per row.
    //StreamSet * recordsByField = P.CreateStreamSet(1);
    //FilterByMask(P, fieldSeparators, recordSeparators, recordsByField);
    //SHOW_STREAM(recordsByField);

    StreamSet * translatedBasis = P.CreateStreamSet(8);
    P.CreateKernelCall<QuoteEscape2Backslash>(quoteEscape, BasisBits, translatedBasis);
    SHOW_BIXNUM(translatedBasis);

    StreamSet * filteredBasis = P.CreateStreamSet(8);
    FilterByMask(P, toKeep, translatedBasis, filteredBasis);
    SHOW_BIXNUM(filteredBasis);

    // Reset Basis bits and reparse.
    BasisBits = filteredBasis;
    csvCCs = P.CreateStreamSet(4);
    csv::CSV_Lexer(P, BasisBits, csvCCs);

    recordSeparators = P.CreateStreamSet(1);
    fieldStarts = P.CreateStreamSet(1);
    fieldFollows = P.CreateStreamSet(1);
    quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldStarts, fieldFollows, quoteEscape);

    StreamSet * QuotedBasis = P.CreateStreamSet(BasisBits->getNumElements());
    JSON_Value_Quoting(P, BasisBits, fieldStarts, fieldFollows, QuotedBasis);

    // Reset Basis bits and reparse.
    BasisBits = QuotedBasis;
    csvCCs = P.CreateStreamSet(4);
    csv::CSV_Lexer(P, BasisBits, csvCCs);

    recordSeparators = P.CreateStreamSet(1);
    fieldStarts = P.CreateStreamSet(1);
    fieldFollows = P.CreateStreamSet(1);
    quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldStarts, fieldFollows, quoteEscape);

    std::vector<uint64_t> insertionAmts;
    unsigned maxInsertAmt = 0;
    for (auto & s : templateStrs) {
        unsigned insertAmt = s.size();
        insertionAmts.push_back(insertAmt);
        if (insertAmt > maxInsertAmt) maxInsertAmt = insertAmt;
    }
    const unsigned insertLengthBits = ceil_log2(maxInsertAmt+1);

    StreamSet * PrefixLgths = P.CreateRepeatingBixNum(insertLengthBits, insertionAmts, TestDynamicRepeatingFile);

    StreamSet * PrefixInsertBixNum = P.CreateStreamSet(insertLengthBits);
    SpreadByMask(P, fieldStarts, PrefixLgths, PrefixInsertBixNum);
    SHOW_BIXNUM(PrefixInsertBixNum);

    std::vector<uint64_t> fieldSuffixLgths;
    for (unsigned i = 0; i < templateStrs.size() - 1; i++) {
        fieldSuffixLgths.push_back(0);
    }
    // Insertion of |},| to terminate each record
    fieldSuffixLgths.push_back(2);

    const unsigned suffixLgthBits = 2;  // insert 1-3 characters.
    StreamSet * RepeatingSuffixLgths = P.CreateRepeatingBixNum(suffixLgthBits, fieldSuffixLgths, TestDynamicRepeatingFile);

    StreamSet * SuffixInsertBixNum = P.CreateStreamSet(suffixLgthBits);
    SpreadByMask(P, fieldFollows, RepeatingSuffixLgths, SuffixInsertBixNum);
    SHOW_BIXNUM(SuffixInsertBixNum);

    StreamSet * InsertBixNum = P.CreateStreamSet(ceil_log2(maxInsertAmt+suffixLgthBits+1));
    P.CreateKernelCall<bixnum::Add>(PrefixInsertBixNum, SuffixInsertBixNum, InsertBixNum);
    SHOW_BIXNUM(InsertBixNum);

    StreamSet * const BasisSpreadMask = P.CreateStreamSet(1);
    InsertionSpreadMask(P, InsertBixNum, BasisSpreadMask, kernel::InsertPosition::Before);
    SHOW_STREAM(BasisSpreadMask);
    
    std::vector<uint64_t> templateBytes;
    for (unsigned i = 0; i < templateStrs.size(); i++) {
        for (auto ch : templateStrs[i]) {
            templateBytes.push_back(static_cast<uint64_t>(ch));
        }
        if (i == templateStrs.size() - 1) {
            templateBytes.push_back(static_cast<uint64_t>('}'));
            templateBytes.push_back(static_cast<uint64_t>(','));
        }
    }
    StreamSet * TemplateBasis = P.CreateRepeatingBixNum(8, templateBytes, TestDynamicRepeatingFile);

    StreamSet * MergedBasis = P.CreateStreamSet(8);
    if (UseMergeByMaskKernel) {
        MergeByMask(P, BasisSpreadMask, BasisBits, TemplateBasis, MergedBasis);
    } else {
        MergeByMask01(P, BasisSpreadMask, BasisBits, TemplateBasis, MergedBasis);
    }
    SHOW_BIXNUM(MergedBasis);

    EOFmark = P.CreateStreamSet(1);
    P.CreateKernelCall<EOFbit>(MergedBasis, EOFmark);
    StreamSet * FinalBasis = P.CreateStreamSet(8);
    P.CreateKernelCall<FinalCommaToRBracket>(MergedBasis, EOFmark, FinalBasis);

    StreamSet * Instantiated = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(FinalBasis, Instantiated);
    P.CreateKernelCall<StdOutKernel>(Instantiated);
    return P.compile();
}

int main(int argc, char *argv[]) {
    llvm_shutdown_obj shutdown;
    csv::InitializeCommandLineInterface(argc, argv);

    std::vector<std::string> headers = csv::get_CSV_headers();

    const auto templateStrs = JSONfieldPrefixes(headers);
    std::string templatePrologue = "[\n";
    //  A CPU driver is capable of compiling and running Parabix programs on the CPU.
    CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    CSVFunctionType fn = generatePipeline(driver, templateStrs);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    const int fd = open(csv::inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::report_fatal_error(llvm::StringRef("Cannot open ") + csv::inputFile);
    } else {
        #ifdef REPORT_PAPI_TESTS
        papi::PapiCounter<4> jitExecution{{PAPI_L3_TCM, PAPI_L3_TCA, PAPI_TOT_INS, PAPI_TOT_CYC}};
        // papi::PapiCounter<3> jitExecution{{PAPI_FUL_ICY, PAPI_STL_CCY, PAPI_RES_STL}};
        jitExecution.start();
        #endif
        //  Run the pipeline.
        printf("%s", templatePrologue.c_str());
        fflush(stdout);
        fn(fd);
        close(fd);
        #ifdef REPORT_PAPI_TESTS
        jitExecution.stop();
        jitExecution.write(std::cerr);
        #endif
    }
    return csv::SuccessExitCode;
}
