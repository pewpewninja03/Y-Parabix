/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <boost/filesystem.hpp>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/core/streamset.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_zeroes.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <toolchain/toolchain.h>
#include <fileselect/file_select.h>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <string>
#include <sys/stat.h>

namespace fs = boost::filesystem;

using namespace llvm;
using namespace codegen;

static cl::OptionCategory matchpriorFlags("Command Flags", "matchprior options");

static cl::list<std::string> inputFiles(cl::Positional, cl::desc("<input file ...>"), cl::OneOrMore, cl::cat(matchpriorFlags));

static cl::opt<unsigned> priorDistance("distance", cl::desc("distance for prior match (default 2)"), cl::init(2),  cl::cat(matchpriorFlags));

static int defaultDisplayColumnWidth = 10;  // default field width

std::vector<fs::path> allFiles;

using namespace kernel;

class MatchPriorKernel final: public pablo::PabloKernel {
public:
    MatchPriorKernel(LLVMTypeSystemInterface & ts, StreamSet * const countable, Scalar * countResult);
protected:
    void generatePabloMethod() override;
};

MatchPriorKernel::MatchPriorKernel (LLVMTypeSystemInterface & ts, StreamSet * const countable, Scalar * countResult)
    : pablo::PabloKernel(ts, "matchprior_" + std::to_string(priorDistance),
    {Binding{"countable", countable}},
    {},
    {},
    {Binding{"matchCount", countResult}}) {
}

void MatchPriorKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<pablo::PabloAST *> basis = getInputStreamSet("countable");

    pablo::PabloAST * mismatches = pb.createZeroes();
    for (unsigned i = 0; i < 8; i++) {
        mismatches = pb.createOr(mismatches,
                                 pb.createXor(basis[i], pb.createAdvance(basis[i], priorDistance)),
                                 "mismatches_" + std::to_string(priorDistance) + "_to_bit" + std::to_string(i));
    }
    pablo::PabloAST * matchesprior = pb.createInFile(pb.createNot(mismatches), "matchesprior");
    pablo::Var * matchCount = getOutputScalarVar("matchCount");
    pb.createAssign(matchCount, pb.createCount(matchesprior));
}

typedef uint64_t (*MatchPriorFunctionType)(uint32_t fd);

MatchPriorFunctionType mpPipelineGen(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>{"fd"}, Output<uint64_t>{"countResult"});
    Scalar * const fileDescriptor = P.getInputScalar("fd");
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    P.CreateKernelCall<MatchPriorKernel>(BasisBits, P.getOutputScalar("countResult"));
    return P.compile();
}

size_t file_size(const int fd) {
    struct stat st;
    if (LLVM_UNLIKELY(fstat(fd, &st) != 0)) {
        st.st_size = 0;
    }
    return st.st_size;
}

void mp(MatchPriorFunctionType fn_ptr, const uint32_t fileIdx) {
    std::string fileName = allFiles[fileIdx].string();
    struct stat sb;
    const int fd = open(fileName.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        if (errno == EACCES) {
            std::cerr << "matchprior: " << fileName << ": Permission denied.\n";
        }
        else if (errno == ENOENT) {
            std::cerr << "matchprior: " << fileName << ": No such file.\n";
        }
        else {
            std::cerr << "matchprior: " << fileName << ": Failed.\n";
        }
        return;
    }
    if (stat(fileName.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        std::cerr << "matchprior: " << fileName << ": Is a directory.\n";
        close(fd);
        return;
    }
    uint64_t countResult = fn_ptr(fd);
    std::cout << std::setw(defaultDisplayColumnWidth);
    std::cout << countResult << std::setw(defaultDisplayColumnWidth);
    std::cout << file_size(fd);
    std::cout << " " << fileName << std::endl;
    close(fd);
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&matchpriorFlags, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    if (argv::RecursiveFlag || argv::DereferenceRecursiveFlag) {
        argv::DirectoriesFlag = argv::Recurse;
    }
    CPUDriver driver("mp");
    allFiles = argv::getFullFileList(driver, inputFiles);
    const auto fileCount = allFiles.size();

    auto matchPriorFunctionPtr = mpPipelineGen(driver);
    for (unsigned i = 0; i < fileCount; ++i) {
        mp(matchPriorFunctionPtr, i);
    }
    return 0;
}
