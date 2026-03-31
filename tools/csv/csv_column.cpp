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
#include "csv_util.hpp"

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory CSV_Options("CSV Processing Options", "CSV Processing Options.");
//static cl::opt<int> columnNo(cl::Positional, cl::desc("column number (1-based)"), cl::Required, cl::cat(CSV_Options));
static cl::list<std::string> Columns("columns",
                                     cl::desc("A comma-separated list of column names or indices"),
                                     cl::ValueRequired, cl::OneOrMore, cl::CommaSeparated, cl::cat(CSV_Options));
static cl::opt<bool> ZeroIndexing("zero", cl::desc("Use 0-based rather than 1-based indices for column numbers"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(CSV_Options));
static cl::opt<bool> HeaderSpecNamesFile("f", cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string> HeaderSpec("headers", cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));
static cl::opt<bool> FilterBasisBits("FilterBasisBits", cl::desc("Perform filtering on basis bits rather than on byte stream"), cl::init(false), cl::cat(CSV_Options));

class SelectField : public PabloKernel {
public:
    SelectField(LLVMTypeSystemInterface & ts, StreamSet * csvMarks,
                              StreamSet * Record_separators,
                              StreamSet * Field_separators,
                              StreamSet * toKeep,
                              unsigned columnNo);
protected:
    void generatePabloMethod() override;
    unsigned mColumnNo;
};

SelectField::SelectField(LLVMTypeSystemInterface & ts,  StreamSet * csvMarks,
                                        StreamSet * Record_separators,
                                        StreamSet * Field_separators,
                                        StreamSet * toKeep,
                                        unsigned columnNo)
: PabloKernel(ts, "SelectField" + std::to_string(columnNo),
  {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)},
   Binding{"Record_separators", Record_separators},
   Binding{"Field_separators", Field_separators}},
  {Binding{"toKeep", toKeep}}), mColumnNo(columnNo)  {}


void SelectField::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Record_separators = pb.createExtract(getInputStreamVar("Record_separators"), pb.getInteger(0));
    PabloAST * Field_separators = pb.createExtract(getInputStreamVar("Field_separators"), pb.getInteger(0));
    PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(Record_separators), 1));
    PabloAST * fieldStarts = pb.createNot(pb.createAdvance(pb.createNot(Field_separators), 1));
    PabloAST * columnMark = recordStarts;
    if (mColumnNo > 1) {
        columnMark = pb.createIndexedAdvance(columnMark, fieldStarts, mColumnNo - 1);
    }
    PabloAST * columnFollow = pb.createScanTo(columnMark, Field_separators);
    PabloAST * columnMask  = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {columnMark, columnFollow});
    PabloAST * notEOF = pb.createNot(pb.createExtract(getInputStreamVar("csvMarks"), pb.getInteger(markEOF)));
    PabloAST * toKeep = pb.createAnd(pb.createOr(columnMask, Record_separators), notEOF);
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), pb.createInFile(toKeep));
}

class SelectMultiField : public PabloKernel {
public:
    SelectMultiField(LLVMTypeSystemInterface & ts, StreamSet * csvMarks,
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
                                   StreamSet * csvMarks,
                                   StreamSet * Record_separators,
                                   StreamSet * Field_separators,
                                   StreamSet * toKeep,
                                   const std::vector<unsigned> & columnNos)
: PabloKernel(ts, "SelectMultiField_" + colNoString(columnNos),
  {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)},
   Binding{"Record_separators", Record_separators},
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
    PabloAST * notEOF = pb.createNot(pb.createExtract(getInputStreamVar("csvMarks"), pb.getInteger(markEOF)));
    PabloAST * toKeep = pb.createAnd(Record_separators, notEOF);
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

    //  We need to know which input positions are dquotes and which are not.
    StreamSet * csvCCs = P.CreateStreamSet(5);
    P.CreateKernelCall<CSVlexer>(BasisBits, csvCCs);

    StreamSet * recordSeparators = P.CreateStreamSet(1);
    StreamSet * fieldSeparators = P.CreateStreamSet(1);
    StreamSet * quoteEscape = P.CreateStreamSet(1);
    P.CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape);

    StreamSet * Selected = P.CreateStreamSet(1);
    P.CreateKernelCall<SelectMultiField>(csvCCs, recordSeparators, fieldSeparators, Selected, colNos);
    SHOW_STREAM(recordSeparators);
    SHOW_STREAM(fieldSeparators);
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

const unsigned MaxHeaderSize = 24;

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&CSV_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    std::vector<std::string> headers;
    if (HeaderSpec == "") {
        headers = get_CSV_headers(inputFile);
    } else if (HeaderSpecNamesFile) {
        headers = get_CSV_headers(HeaderSpec);
    } else {
        headers = parse_CSV_headers(HeaderSpec);
    }
    std::map<std::string, unsigned> header_map;
    for (unsigned i = 0; i < headers.size(); i++) {
        header_map.emplace(headers[i], i);
        if (headers[i].size() > MaxHeaderSize) {
            headers[i] = headers[i].substr(0, MaxHeaderSize);
        }
    }
    std::vector<unsigned> colNos(Columns.size());
    for (unsigned i = 0; i < Columns.size(); i++) {
        auto f = header_map.find(Columns[i]);
        if (f != header_map.end()) {
            colNos[i] = f->second;
        } else {
            std::stringstream ss(Columns[i]);
            unsigned colNo;
            ss >> colNo;
            if (ss.eof() && (colNo < headers.size())) {
                if (ZeroIndexing) {
                    colNos[i] = colNo;
                    continue;
                } else if (colNo > 0) {
                    colNos[i] = colNo - 1;
                    continue;
                }
            }
            llvm::report_fatal_error("Invalid column spec");
        }
    }
    
    CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    CSVFunctionType fn = generatePipeline(driver, colNos);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        fn(fd);
        close(fd);
    }
    return 0;
}
