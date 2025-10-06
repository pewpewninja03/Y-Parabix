/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>
#include <boost/filesystem.hpp>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/adt.h>
#include <re/parse/parser.h>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/core/streamset.h>
#include <kernel/unicode/utf8_support.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/pablo_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <grep/grep_kernel.h>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <map>

namespace fs = boost::filesystem;

using namespace llvm;
using namespace codegen;
using namespace kernel;
using namespace pablo;

static cl::OptionCategory tokFlags("Command Flags", "Tokenizer options");

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(tokFlags));

class AddLFs : public PabloKernel {
public:
    AddLFs(LLVMTypeSystemInterface & ts, StreamSet * insertMask, StreamSet * spreadBasis, StreamSet * finalBasis)
        : PabloKernel(ts, "addLFs",
                      {Binding{"insertMask", insertMask}, Binding{"spreadBasis", spreadBasis}},
                      {Binding{"finalBasis", finalBasis}}) {}
protected:
    void generatePabloMethod() override;
};


void AddLFs::generatePabloMethod() {
    //  pb is an object used for build Pablo language statements
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * insertMask = getInputStreamSet("insertMask")[0];
    std::vector<PabloAST *> spreadBasis = getInputStreamSet("spreadBasis");
    //
    // Null bytes have been inserted at the insertMask positions.
    // These must be converted to LF characters with the numeric code 0X0A.
    // This simply requires that bit positions 1 and 3 of each inserted
    // null byte are changed from zero bits into one bits, while keeping
    // other bits the same.
    //
    std::vector<PabloAST *> finalBasis(spreadBasis.size());
    for (unsigned i = 0; i < spreadBasis.size(); i++) {
        finalBasis[i] = spreadBasis[i];
    }
    PabloAST * nullPositions = pb.createInFile(pb.createNot(insertMask));
    finalBasis[1] = pb.createOr(finalBasis[1], nullPositions);
    finalBasis[3] = pb.createOr(finalBasis[3], nullPositions);

    writeOutputStreamSet("finalBasis", finalBasis);
}

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

typedef void (*TokenizerFunctionType)(uint32_t fd);

TokenizerFunctionType tokenizerPipeline(CPUDriver & driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"fileDescriptor"});

    Scalar * const fileDescriptor = P.getInputScalar("fileDescriptor");

    //  Create a stream set consisting of a single stream of 8-bit units (bytes).
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);

    //  Read the file into the ByteStream.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    //  Create a set of 8 parallel streams of 1-bit units (bits).
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);

    //  Transpose the ByteSteam into parallel bit stream for
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    
    StreamSet * GCB = P.CreateStreamSet(1, 1);
    GraphemeClusterLogic(P, BasisBits, u8index, GCB);
    SHOW_STREAM(GCB);

    StreamSet * tokenInsertMask = P.CreateStreamSet(1, 1);
    UnitInsertionSpreadMask(P, GCB, tokenInsertMask, kernel::InsertPosition::After);
    SHOW_STREAM(tokenInsertMask);

    StreamSet * spreadBasis = P.CreateStreamSet(8);
    SpreadByMask(P, tokenInsertMask, BasisBits, spreadBasis);
    SHOW_BIXNUM(spreadBasis);

    StreamSet * tokenBasis = P.CreateStreamSet(8);
    P.CreateKernelCall<AddLFs>(tokenInsertMask, spreadBasis, tokenBasis);
    SHOW_BIXNUM(tokenBasis);

    // The computed output can be converted back to byte stream form by the
    // P2S kernel (parallel-to-serial).
    StreamSet * tokenLines = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(tokenBasis, tokenLines);
    SHOW_BYTES(tokenLines);

    //  The StdOut kernel writes a byte stream to standard output.
    P.CreateKernelCall<StdOutKernel>(tokenLines);

    return reinterpret_cast<TokenizerFunctionType>(P.compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&tokFlags, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    CPUDriver driver("tok");

    TokenizerFunctionType tokFn = tokenizerPipeline(driver);
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        //  Run the pipeline.
        tokFn(fd);
        close(fd);
    }
    return 0;
}
