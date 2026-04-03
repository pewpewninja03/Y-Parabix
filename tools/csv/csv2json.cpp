/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */


#include <cstdio>
#include <vector>
#include <csv/csv_cmdline.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
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
#include <kernel/scan/scanmatchgen.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include "csv_util.hpp"
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

using namespace kernel;
using namespace llvm;
using namespace pablo;

static cl::opt<bool> TestDynamicRepeatingFile("dyn", cl::desc("Test Dynamic Repeating StreamSet"), cl::init(true), cl::cat(csv::CSV_Options));
static cl::opt<bool> UseMergeByMaskKernel("merge-by-mask", cl::desc("Use MergeByMask kernel"), cl::init(false), cl::cat(csv::CSV_Options));

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

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P.CreateStreamSet(8);
    Selected_S2P(P, ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
    //  We need to know which input positions are dquotes and which are not.
    StreamSet * csvCCs = P.CreateStreamSet(5);
    P.CreateKernelCall<CSVlexer>(BasisBits, csvCCs);
    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldSeparators = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);

    P.CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape);
    SHOW_STREAM(recordSeparators);
    SHOW_STREAM(fieldSeparators);
    SHOW_STREAM(quoteEscape);
    StreamSet * toKeep = P.CreateStreamSet(1);
    P.CreateKernelCall<CSVdataFieldMask>(csvCCs, recordSeparators, quoteEscape, toKeep, csv::HeaderSpec == "");
    SHOW_STREAM(toKeep);
    //
    // Create a short stream which is 1-to-1 with the (field/record) separators,
    // having 0 bits for field separators and 1 bits for record separators.
    // Normally this will be a stream having exactly one bit set for every
    // N positions, where N is the number of entries per row.
    StreamSet * recordsByField = P.CreateStreamSet(1);
    FilterByMask(P, fieldSeparators, recordSeparators, recordsByField);
    SHOW_STREAM(recordsByField);

    StreamSet * translatedBasis = P.CreateStreamSet(8);
    P.CreateKernelCall<CSV_Char_Replacement>(quoteEscape, BasisBits, translatedBasis);
    SHOW_BIXNUM(translatedBasis);

    StreamSet * filteredBasis = P.CreateStreamSet(8);
    StreamSet * filteredFieldSeparators = P.CreateStreamSet(1);
    FilterByMask(P, toKeep, translatedBasis, filteredBasis);
    SHOW_BIXNUM(filteredBasis);

    FilterByMask(P, toKeep, fieldSeparators, filteredFieldSeparators);
    SHOW_STREAM(filteredFieldSeparators);

    std::vector<uint64_t> insertionAmts;
    unsigned maxInsertAmt = 0;
    for (auto & s : templateStrs) {
        unsigned insertAmt = s.size();
        insertionAmts.push_back(insertAmt);
        if (insertAmt > maxInsertAmt) maxInsertAmt = insertAmt;
    }
    const unsigned insertLengthBits = ceil_log2(maxInsertAmt+1);

    StreamSet * PrefixLgths = P.CreateRepeatingBixNum(insertLengthBits, insertionAmts, TestDynamicRepeatingFile);

    StreamSet * fieldStarts = P.CreateStreamSet(1);
    P.CreateKernelCall<LineStartsKernel>(filteredFieldSeparators, fieldStarts);
    SHOW_STREAM(fieldStarts);

    StreamSet * PrefixInsertBixNum = P.CreateStreamSet(insertLengthBits);
    SpreadByMask(P, fieldStarts, PrefixLgths, PrefixInsertBixNum);
    SHOW_BIXNUM(PrefixInsertBixNum);

    std::vector<uint64_t> fieldSuffixLgths;
    for (unsigned i = 0; i < templateStrs.size() - 1; i++) {
        // Insertion of a single quote to terminate each field.
        fieldSuffixLgths.push_back(1);
    }
    // Insertion of |"},| to terminate each record
    fieldSuffixLgths.push_back(3);

    const unsigned suffixLgthBits = 2;  // insert 1-3 characters.
    StreamSet * RepeatingSuffixLgths = P.CreateRepeatingBixNum(suffixLgthBits, fieldSuffixLgths, TestDynamicRepeatingFile);

    StreamSet * SuffixInsertBixNum = P.CreateStreamSet(suffixLgthBits);
    SpreadByMask(P, filteredFieldSeparators, RepeatingSuffixLgths, SuffixInsertBixNum);
    SHOW_BIXNUM(SuffixInsertBixNum);

    StreamSet * InsertBixNum = P.CreateStreamSet(insertLengthBits);
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
        templateBytes.push_back(static_cast<uint64_t>('"'));
        if (i == templateStrs.size() - 1) {
            templateBytes.push_back(static_cast<uint64_t>('}'));
            templateBytes.push_back(static_cast<uint64_t>(','));
        }
    }
    StreamSet * TemplateBasis = P.CreateRepeatingBixNum(8, templateBytes, TestDynamicRepeatingFile);

    StreamSet * FinalBasis = P.CreateStreamSet(8);
    if (UseMergeByMaskKernel) {
        MergeByMask(P, BasisSpreadMask, filteredBasis, TemplateBasis, FinalBasis);
    } else {
        MergeByMask01(P, BasisSpreadMask, filteredBasis, TemplateBasis, FinalBasis);
    }
    SHOW_BIXNUM(FinalBasis);
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
    //for (auto & s : templateStrs) {
    //    llvm::errs() << "template string: |" << s << "|\n";
    //}
    std::string templatePrologue = "[\n";
    std::string templateEpilogue = "\"}\n]\n";
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
        printf("%s", templateEpilogue.c_str());
        #ifdef REPORT_PAPI_TESTS
        jitExecution.stop();
        jitExecution.write(std::cerr);
        #endif
    }
    return 0;
}
