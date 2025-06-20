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
#include <kernel/io/source_kernel.h>
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
#include <fileselect/file_select.h>
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

static cl::OptionCategory gcFlags("Command Flags", "gcount options");

static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<input file ...>"), cl::OneOrMore, cl::cat(gcFlags));

std::vector<fs::path> allFiles;

typedef uint64_t (*GCountFunctionType)(uint32_t fd);

GCountFunctionType pipelineGen(CPUDriver & driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"fileDescriptor"}, Output<uint64_t>{"countResult"});

    Scalar * const fileDescriptor = P.getInputScalar("fileDescriptor");

    //  Create a stream set consisting of a single stream of 8-bit units (bytes).
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);

    //  Read the file into the ByteStream.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    //  Create a set of 8 parallel streams of 1-bit units (bits).
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);

    //  Transpose the ByteSteam into parallel bit stream for
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);

    StreamSet * u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    
    StreamSet * GCB = P.CreateStreamSet(1, 1);
    re::UTF8_Transformer U8xfrmer;
    GraphemeClusterLogic(P, BasisBits, u8index, GCB);

    P.CreateKernelCall<PopcountKernel>(GCB, P.getOutputScalar("countResult"));

    return reinterpret_cast<GCountFunctionType>(P.compile());
}

uint64_t gcount1(GCountFunctionType fn_ptr, const uint32_t fileIdx) {
    std::string fileName = allFiles[fileIdx].string();
    struct stat sb;
    const int fd = open(fileName.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        if (errno == EACCES) {
            std::cerr << "ucount: " << fileName << ": Permission denied.\n";
        }
        else if (errno == ENOENT) {
            std::cerr << "ucount: " << fileName << ": No such file.\n";
        }
        else {
            std::cerr << "ucount: " << fileName << ": Failed.\n";
        }
        return 0;
    }
    if (stat(fileName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        std::cerr << "ucount: " << fileName << ": Is a directory.\n";
        close(fd);
        return 0;
    }
    auto r = fn_ptr(fd);
    close(fd);
    return r;
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&gcFlags, codegen::codegen_flags()});
    if (argv::RecursiveFlag || argv::DereferenceRecursiveFlag) {
        argv::DirectoriesFlag = argv::Recurse;
    }
    CPUDriver driver("gc");

    allFiles = argv::getFullFileList(driver, inputFiles);
    const auto fileCount = allFiles.size();

    GCountFunctionType gCountFunctionPtr = nullptr;
    gCountFunctionPtr = pipelineGen(driver);
    
    std::vector<uint64_t> theCounts;
    
    theCounts.resize(fileCount);
    uint64_t totalCount = 0;

    for (unsigned i = 0; i < fileCount; ++i) {
        theCounts[i] = gcount1(gCountFunctionPtr, i);
        totalCount += theCounts[i];
    }
    
    const int defaultDisplayColumnWidth = 7;
    int displayColumnWidth = std::to_string(totalCount).size() + 1;
    if (displayColumnWidth < defaultDisplayColumnWidth) displayColumnWidth = defaultDisplayColumnWidth;

    for (unsigned i = 0; i < fileCount; ++i) {
        std::cout << std::setw(displayColumnWidth);
        std::cout << theCounts[i] << std::setw(displayColumnWidth);
        std::cout << " " << allFiles[i].string() << std::endl;
    }
    if (inputFiles.size() > 1) {
        std::cout << std::setw(displayColumnWidth-1);
        std::cout << totalCount;
        std::cout << " total" << std::endl;
    }

    return 0;
}
