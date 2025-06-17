/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>
#include <boost/filesystem.hpp>
#include <grep/grep_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/adt.h>
#include <re/parse/parser.h>
#include <re/transforms/re_simplifier.h>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/core/streamset.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/utf8_support.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/pablo_kernel.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
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

static cl::OptionCategory ufFlags("Command Flags", "ufilter options");

static cl::opt<std::string> CC_expr(cl::Positional, cl::desc("<Unicode character class expression>"), cl::Required, cl::cat(ufFlags));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"),  cl::cat(ufFlags));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

typedef void (*UFiltertFunctionType)(uint32_t fd);

UFiltertFunctionType pipelineGen(CPUDriver & driver, re::Name * CC_name) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"fileDescriptor"});

    Scalar * const fileDescriptor = P.getInputScalar("fileDescriptor");

    //  Create a stream set consisting of a single stream of 8-bit units (bytes).
    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    SHOW_BYTES(ByteStream);

    //  Read the file into the ByteStream.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    //  Create a set of 8 parallel streams of 1-bit units (bits).
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    SHOW_BIXNUM(BasisBits);

    //  Transpose the ByteSteam into parallel bit stream form.
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);

    //  Create a character class bit stream.
    StreamSet * CCmask = P.CreateStreamSet(1, 1);

    std::map<std::string, StreamSet *> propertyStreamMap;
    auto nameString = CC_name->getFullName();
    propertyStreamMap.emplace(nameString, CCmask);
    P.CreateKernelFamilyCall<UnicodePropertyKernelBuilder>(CC_name, BasisBits, CCmask);
    SHOW_STREAM(CCmask);

    StreamSet * u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * CCspans = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<U8Spans>(CCmask, u8index, CCspans);
    SHOW_STREAM(CCspans);

    StreamSet * const FilteredBytes = P.CreateStreamSet(1, 8);

    FilterByMask(P, CCspans, ByteStream, FilteredBytes, 0, 64);

    SHOW_BYTES(FilteredBytes);

    P.CreateKernelCall<StdOutKernel>(FilteredBytes);

    return P.compile();
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&ufFlags, codegen::codegen_flags()});
    CPUDriver driver("ufilter");

    UFiltertFunctionType fnPtr = nullptr;
    re::RE * CC_re = re::simplifyRE(re::RE_Parser::parse(CC_expr));
    CC_re = UCD::linkAndResolve(CC_re);
    CC_re = UCD::externalizeProperties(CC_re);
    if (re::Name * UCD_property_name = dyn_cast<re::Name>(CC_re)) {
        fnPtr = pipelineGen(driver, UCD_property_name);
    } else if (re::CC * CC_ast = dyn_cast<re::CC>(CC_re)) {
        fnPtr = pipelineGen(driver, makeName(CC_ast));
    } else {
        std::cerr << "Input expression must be a Unicode property or CC but found: " << CC_expr << " instead.\n";
        exit(1);
    }

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        if (errno == EACCES) {
            std::cerr << "ufilter: " << inputFile << ": Permission denied.\n";
        }
        else if (errno == ENOENT) {
            std::cerr << "ufilter: " << inputFile << ": No such file.\n";
        }
        else {
            std::cerr << "ufilter: " << inputFile << ": Failed.\n";
        }
        exit(1);
    }
    struct stat sb;
    if (stat(inputFile.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        std::cerr << "ufilter: " << inputFile << ": Is a directory.\n";
        close(fd);
        exit(1);
    }

    fnPtr(fd);
    close(fd);
    return 0;
}
