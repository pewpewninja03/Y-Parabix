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
#include <pablo/codegenstate.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/sentinel.h>
#include <kernel/streamutils/sorting.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/bitwise/bixnum_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/normalization.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/utf8_support.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/unicode/resolve_properties.h>
#include <string>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/algo/normalization.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFKD_Options("Decompositon Options", "Decompositon Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFKD_Options));
#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

void source_input_stage(PipelineBuilder & P, Scalar *const fileDescriptor, StreamSet * ByteStream, StreamSet * BasisBits) {

    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
}

void SimpleOutputStage(PipelineBuilder & P, StreamSet * TransformedBasis) {
    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(TransformedBasis, OutputBytes);
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}

typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_NFKD_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>("inputFileDecriptor"));
    NFD_PipelineBuilder NFD_builder(P);
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);

    StreamSet * const U21_u8indexed = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);
    SHOW_BIXNUM(U21_u8indexed);

    StreamSet * const U8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, U8index);
    SHOW_STREAM(U8index);

    StreamSet * const U21_basis = P.CreateStreamSet(21, 1);
    FilterByMask(P, U8index, U21_u8indexed, U21_basis);
    SHOW_BIXNUM(U21_basis);

    StreamSet * NFKD_U21_Results = NFD_builder.NFKD_U21_Pipeline(U21_basis);

    StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
    U21_to_UTF8(P, NFKD_U21_Results, TransformedBasis);
    SHOW_BIXNUM(TransformedBasis);

    SimpleOutputStage(P, TransformedBasis);
    return P.compile();
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&NFKD_Options, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
        exit(-1);
    }
    CPUDriver driver("NFKD_function");
    XfrmFunctionType fn = generate_NFKD_pipeline(driver);
    fn(fd);
    close(fd);
    return 0;
}
