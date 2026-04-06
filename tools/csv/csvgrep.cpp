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
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ManagedStatic.h>
#include <unicode/core/unicode_set.h>
#include <re/adt/adt.h>
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
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/transforms/re_transformer.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html

static cl::opt<std::string> Regex("regex", cl::desc("regular expression to search selected columns"), cl::cat(csv::CSV_Options));
static cl::alias RegexA("r", cl::desc("Alias for --regex"), cl::aliasopt(Regex), cl::cat(csv::CSV_Options), cl::NotHidden);

typedef void (*CSVFunctionType)(uint32_t fd);

using namespace re;

//
//  When match CSV fields, any double quote character within the
//  field will be doubled up to represent and escaped double quote.
//  So we transform the RE to match a pair of double quotes whenever
//  a single one is desired.

struct DoubleQuoteEscape : public RE_Transformer {
    DoubleQuoteEscape(codepoint_t dqChar) : RE_Transformer("DoubleQuoteEscape"),
       mDQ(dqChar), mDQ_CC(nullptr), mDoubleEscape(nullptr) {}
    CC * getDQ_CC() {
        if (mDQ_CC == nullptr) {
            mDQ_CC = re::makeCC(mDQ, &cc::Unicode);
        }
        return mDQ_CC;
    }
    RE * getDoubleEscape() {
        if (mDoubleEscape == nullptr) {
            CC * DQ_CC = getDQ_CC();
            mDoubleEscape = makeSeq({DQ_CC, DQ_CC});
        }
        return mDoubleEscape;
    }
    RE * transformCC (CC * cc) override {
        if (cc->contains(mDQ)) {
            auto dblEsc = getDoubleEscape();
            if (cc->count() == 1) {
                return dblEsc;
            }
            return makeAlt({dblEsc, subtractCC(cc, mDQ_CC)});
        }
        return cc;
    }
    RE * transformAny (Any * a) override {
        return makeDiff(a, getDQ_CC());
    }
    RE * transformName (Name * name) override {
        RE * defn = name->getDefinition();
        if (!defn || isa<Any>(defn)) return makeDiff(name, getDQ_CC());
        RE * d = transform(defn);
        if (d == defn) return name;
        return d;
    }
    RE * transformPropertyExpression (PropertyExpression * pe) override {
        RE * defn = pe->getResolvedRE();
        if (!defn) return makeDiff(pe, getDQ_CC());
        RE * d = transform(defn);
        if (d == defn) return pe;
        return d;
    }
private:
    codepoint_t mDQ;
    CC * mDQ_CC;
    RE * mDoubleEscape;
};

RE * csvRE(RE * re) {
    RE * xfrmedRE = resolveModesAndExternalSymbols(re, false, grep::lineNumGrep);
    xfrmedRE = DoubleQuoteEscape(csv::QuoteChar).transformRE(xfrmedRE);
    return xfrmedRE;
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

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

    StreamSet * csvCCs = P.CreateStreamSet(4);
    csv::CSV_Lexer(P, BasisBits, csvCCs);

    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldSeparators = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldSeparators, quoteEscape);

    StreamSet * Selected = P.CreateStreamSet(1);
    csv::ColumnSelectionMask(P, recordSeparators, fieldSeparators, Selected, colNos);
    SHOW_STREAM(Selected);

    StreamSet * NonSelected = P.CreateStreamSet(1);
    Invert(P, Selected, NonSelected);
    
    StreamSet * Barrier = P.CreateStreamSet(1);
    OrCombine(P, NonSelected, fieldSeparators, Barrier);

    RE * searchRE = csvRE(RE_Parser::parse(Regex));
    RE_CompilerContext ctxt;
    ctxt.setCodeUnitContext(&cc::UTF8, BasisBits);
    ctxt.setBarrier(Barrier);
    RE_PipelineBuilder RE_PB(P, ctxt);
    StreamSet * Matches = P.CreateStreamSet(1);
    RE_PB.matchSearchPipeline(searchRE, Matches);
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        P.captureBitstream("Matches", Matches);
    }
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
