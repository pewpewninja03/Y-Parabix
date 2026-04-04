/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <cstdio>
#include <vector>
#include <csv/csv_cmdline.h>
#include <csv/csv_parser.h>
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
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
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
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::opt<bool> FilterBasisBits("FilterBasisBits", cl::desc("Perform filtering on basis bits rather than on byte stream"), 
    cl::init(false), cl::cat(csv::CSV_Options), cl::Hidden);

class SelectMultiField : public PabloKernel {
public:
    SelectMultiField(LLVMTypeSystemInterface & ts,
                              StreamSet * Record_separators,
                              StreamSet * Field_separators,
                              StreamSet * toKeep,
                              const std::vector<unsigned> & columnNos);
protected:
    std::string colNoString(const std::vector<unsigned> & cols);
    void generatePabloMethod() override;
    const std::vector<unsigned> & mColumnNos;
};

SelectMultiField::SelectMultiField(LLVMTypeSystemInterface & ts,
                                   StreamSet * Record_separators,
                                   StreamSet * Field_separators,
                                   StreamSet * toKeep,
                                   const std::vector<unsigned> & columnNos)
: PabloKernel(ts, "SelectMultiField_" + colNoString(columnNos),
  {Binding{"Record_separators", Record_separators},
   Binding{"Field_separators", Field_separators}},
  {Binding{"toKeep", toKeep}}), mColumnNos(columnNos)  {}

std::string SelectMultiField::colNoString(const std::vector<unsigned> & columnNos) {
    std::stringstream ss;
    ss << columnNos[0];
    for (unsigned i = 1; i < columnNos.size(); i++) {
        ss << "," << columnNos[i];
    }
    return ss.str();
}

void SelectMultiField::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Record_separators = pb.createExtract(getInputStreamVar("Record_separators"), pb.getInteger(0));
    PabloAST * Field_separators = pb.createExtract(getInputStreamVar("Field_separators"), pb.getInteger(0));
    PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(Record_separators), 1));
    PabloAST * fieldStarts = pb.createNot(pb.createAdvance(pb.createNot(Field_separators), 1));
    PabloAST * columnMark = recordStarts;
    PabloAST * toKeep = Record_separators;
    unsigned nextCol = 0;
    for (unsigned i = 0; i < mColumnNos.size(); i++) {
        if (mColumnNos[i] > nextCol) {
            columnMark = pb.createIndexedAdvance(columnMark, fieldStarts, mColumnNos[i] - nextCol);
        }
        PabloAST * columnFollow = pb.createScanTo(columnMark, Field_separators);
        PabloAST * columnMask  = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {columnMark, columnFollow});
        toKeep = pb.createOr(toKeep, columnMask);
        nextCol = mColumnNos[i] + 1;
        columnMark = pb.createAdvance(columnFollow, 1);
        if (i < mColumnNos.size() - 1) {
            toKeep = pb.createOr(toKeep, columnFollow);
        }
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), pb.createInFile(toKeep));
}

typedef void (*CSVFunctionType)(uint32_t fd);

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
    csv::CSV_Lexer(P, csv::FieldDelimiter, csv::QuoteChar, BasisBits, csvCCs);

    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldSeparators = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);
    csv::ParseCSV(P, csvCCs, recordSeparators, fieldSeparators, quoteEscape);

    StreamSet * Selected = P.CreateStreamSet(1);
    P.CreateKernelCall<SelectMultiField>(recordSeparators, fieldSeparators, Selected, colNos);
    SHOW_STREAM(Selected);

    StreamSet * Filtered = P.CreateStreamSet(1, 8);
    if (FilterBasisBits) {
        StreamSet * filteredBasis = P.CreateStreamSet(8);
        FilterByMask(P, Selected, BasisBits, filteredBasis);
        P.CreateKernelCall<P2SKernel>(filteredBasis, Filtered);
        SHOW_BIXNUM(filteredBasis);
    } else {
        FilterByMask(P, Selected, ByteStream, Filtered);
    }
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
