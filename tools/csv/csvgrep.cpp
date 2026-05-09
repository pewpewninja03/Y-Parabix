/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <vector>
#include <csv/csv_cmdline.h>
#include <csv/csv_parser.h>
#include <csv/csv_support.h>
#include <ucd/utf/utf_encoder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ManagedStatic.h>
#include <ucd/core/unicode_set.h>
#include <re/adt/adt.h>
#include <re/adt/re_re.h>
#include <re/parse/parser.h>
#include <re/unicode/regex_passes.h>
#include <grep/grep_engine.h>
#include <grep/grep_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/re/regexp_kernel.h>
#include <kernel/unicode/utf8_support.h>
#include <kernel/unicode/utf8_decoder.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html

static cl::opt<std::string> Regex("regex", cl::desc("regular expression to search selected columns"), cl::cat(csv::CSV_Options));
static cl::alias RegexA("r", cl::desc("Alias for --regex"), cl::aliasopt(Regex), cl::cat(csv::CSV_Options), cl::NotHidden);
static cl::opt<bool> U21("u21", cl::desc("force work to be carried out in Unicode 21 bit space"), cl::cat(csv::CSV_Options));
static cl::opt<bool> FieldMatch("field-match", cl::desc("require that entire field be matched"), cl::init(false), cl::cat(csv::CSV_Options));
static cl::alias FieldMatchA("x", cl::desc("Alias for --field-match"), cl::aliasopt(FieldMatch), cl::cat(csv::CSV_Options), cl::NotHidden);

re::RE * csvRE(re::RE * re) {
    re::RE * xfrmedRE = resolveModesAndExternalSymbols(re, false, grep::lineNumGrep);
    xfrmedRE = csv::DoubleQuoteEscape(csv::QuoteChar).transformRE(xfrmedRE);
    if (FieldMatch) {
        xfrmedRE = re::makeSeq({re::makeStart(), xfrmedRE, re::makeEnd()});
    }
    return xfrmedRE;
}

class RegexRegions : public PabloKernel {
public:
    RegexRegions(LLVMTypeSystemInterface & ts, StreamSet * csvMarks, StreamSet * fieldSeparators, StreamSet * selected,
                 StreamSet * regionStart, StreamSet * regionFollow)
    : PabloKernel(ts, "RegexRegions",
                      {Binding{"csvMarks", csvMarks},
                       Binding{"fieldSeparators", fieldSeparators, FixedRate(), LookAhead(1)},
                       Binding{"selected", selected}},
                      {Binding{"regionStart", regionStart}, Binding{"regionFollow", regionFollow}}) {}
protected:
    void generatePabloMethod() override;
};

void RegexRegions::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * fieldSeparators = pb.createExtract(getInputStreamVar("fieldSeparators"), pb.getInteger(0));
    PabloAST * selectedFields = pb.createExtract(getInputStreamVar("selected"), pb.getInteger(0));
    PabloAST * regionStart = pb.createNot(pb.createAdvance(pb.createNot(fieldSeparators), 1));
    regionStart = pb.createAnd(regionStart, selectedFields);
    PabloAST * fieldStartQuote = pb.createAnd(regionStart, csvMarks[csv::markDQ]);
    regionStart = pb.createOr(pb.createXor(regionStart, fieldStartQuote), pb.createAdvance(fieldStartQuote, 1));
    PabloAST * regionFollow = pb.createAnd(fieldSeparators, selectedFields);
    PabloAST * fieldEndQuote = pb.createAnd(pb.createLookahead(fieldSeparators, 1), csvMarks[csv::markDQ]);
    regionFollow = pb.createOr(fieldEndQuote, pb.createXor(regionFollow, pb.createAdvance(fieldEndQuote, 1)));
    pb.createAssign(pb.createExtract(getOutputStreamVar("regionStart"), pb.getInteger(0)), regionStart);
    pb.createAssign(pb.createExtract(getOutputStreamVar("regionFollow"), pb.getInteger(0)), regionFollow);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

typedef void (*CSVFunctionType)(uint32_t fd);

CSVFunctionType generatePipeline(CPUDriver & driver, const std::vector<unsigned> & colNos) {

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

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BYTES(ByteStream);
    SHOW_BIXNUM(BasisBits);

    UTF_Encoder u8_encoder(8);

    re::RE * searchRE = csvRE(re::RE_Parser::parse(Regex));
    StreamSet * u8index = nullptr;

    unsigned DQ_u8bytes = u8_encoder.encoded_length(csv::QuoteChar);
    unsigned Delim_u8bytes = u8_encoder.encoded_length(csv::FieldDelimiter);

    RE_CompilerContext ctxt;
    if (!validateFixedUTF8(searchRE) || (DQ_u8bytes > 1) || (Delim_u8bytes > 1) || U21) {
        u8index = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
        SHOW_STREAM(u8index);

        // UTF-8 to U21 Conversion Pipeline
        // Creating U21 codepoint stream from UTF-8 basis bits (U21 Codepoint Generation)
        StreamSet * U21_u8indexed = P.CreateStreamSet(21, 1);
        P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);

        SHOW_BIXNUM(U21_u8indexed);

        StreamSet * U21codepoints = P.CreateStreamSet(21, 1);
        FilterByMask(P, u8index, U21_u8indexed, U21codepoints);
        SHOW_BIXNUM(U21codepoints);

        BasisBits = U21codepoints;

        ctxt.setCodeUnitContext(&cc::Unicode, BasisBits);
    } else {
        searchRE = toUTF8(searchRE);
        ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
    }

    StreamSet * csvCCs = P.CreateStreamSet(4);
    csv::CSV_Lexer(P, BasisBits, csvCCs);

    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldSeparators = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldSeparators, quoteEscape);

    StreamSet * Selected = P.CreateStreamSet(1);
    csv::ColumnSelectionMask(P, recordSeparators, fieldSeparators, Selected, colNos);
    SHOW_STREAM(Selected);

    StreamSet * fieldStarts = P.CreateStreamSet(1);
    StreamSet * fieldFollows = P.CreateStreamSet(1);
    P.CreateKernelCall<RegexRegions>(csvCCs, fieldSeparators, Selected, fieldStarts, fieldFollows);
    ctxt.setMatchRegions(fieldStarts, fieldFollows);

    StreamSet * Matches = P.CreateStreamSet(1);
    RE_PipelineBuilder RE_PB(P, ctxt);
    RE_PB.matchSearchPipeline(searchRE, Matches);

    StreamSet * MatchedLineEnds = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<MatchedLinesKernel>(Matches, recordSeparators, MatchedLineEnds);
    
    StreamSet * MatchesByLine = P.CreateStreamSet(1, 1);
    FilterByMask(P, recordSeparators, MatchedLineEnds, MatchesByLine);
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        P.captureBitstream("MatchesByLine", MatchesByLine);
    }

    StreamSet * MatchedLineStarts = P.CreateStreamSet(1, 1);
    StreamSet * lineStarts = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<LineStartsKernel>(recordSeparators, lineStarts);
    SpreadByMask(P, lineStarts, MatchesByLine, MatchedLineStarts);

    StreamSet * MatchedLineSpans = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<LineSpansKernel>(MatchedLineStarts, MatchedLineEnds, MatchedLineSpans);
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        P.captureBitstream("MatchedLineSpans", MatchedLineSpans);
    }

    if (u8index != nullptr) {
        StreamSet * SpreadSpans = P.CreateStreamSet(1, 1);
        SpreadByMask(P, u8index, MatchedLineSpans, SpreadSpans);

        StreamSet * ResultSpans = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<U8Spans>(SpreadSpans, u8index, ResultSpans);
        MatchedLineSpans = ResultSpans;
    }
    StreamSet * Filtered = P.CreateStreamSet(1, 8);
    FilterByMask(P, MatchedLineSpans, ByteStream, Filtered, 0, 64);

    SHOW_BYTES(Filtered);
    //  The StdOut kernel writes a byte stream to standard output.
    P.CreateKernelCall<StdOutKernel>(Filtered);
    return P.compile();
}

int main(int argc, char *argv[]) {
    llvm_shutdown_obj shutdown;
    csv::InitializeCommandLineInterface(argc, argv);

    std::vector<std::string> headers = csv::get_CSV_headers();

    std::vector<unsigned> colNos = csv::getColumnArgs(headers);

    CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    CSVFunctionType fn = generatePipeline(driver, colNos);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    const int fd = open(csv::inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::report_fatal_error(llvm::StringRef("Cannot open ") + csv::inputFile);
    } else {
        fn(fd);
        close(fd);
    }
    return csv::SuccessExitCode;
}
