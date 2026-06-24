/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>                   // for GetIDISA_Builder
#include <re/cc/cc_compiler.h>                        // for CC_Compiler
#include <kernel/unicode/utf8gen.h>
#include <kernel/streamutils/deletion.h>                      // for DeletionKernel
#include <kernel/io/source_kernel.h>
#include <kernel/basis/p2s_kernel.h>                    // for P2S16KernelWithCom...
#include <kernel/basis/s2p_kernel.h>                    // for S2PKernel
#include <kernel/io/stdout_kernel.h>                 // for StdOutKernel_
#include <kernel/streamutils/pdep_kernel.h>
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <pablo/pablo_kernel.h>                    // for PabloKernel
#include <kernel/core/kernel_builder.h>
#include <pablo/pe_zeroes.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/util/hex_convert.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/builder.hpp>
#include <fcntl.h>
#include <kernel/pipeline/program_builder.h>
#ifdef ENABLE_PAPI
#include <util/papi_helper.hpp>
#endif

using namespace pablo;
using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory u32u8Options("u32u8 Options", "Transcoding control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(u32u8Options));

typedef void (*u32u8FunctionType)(uint32_t fd);

u32u8FunctionType u32u8_gen (CPUDriver & driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>("fd"));
    Scalar * const fileDescriptor = P.getInputScalar("fd");

    // Source data
    StreamSet * const codeUnitStream = P.CreateStreamSet(1, 32);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, codeUnitStream);

    // Source buffers for transposed UTF-32 basis bits.
    StreamSet * const u32basis = P.CreateStreamSet(21);
    P.CreateKernelCall<S2P_21Kernel>(codeUnitStream, u32basis);

    // Final buffers for computed UTF-8 basis bits and byte stream.
    StreamSet * const u8basis = P.CreateStreamSet(8);
    StreamSet * const u8bytes = P.CreateStreamSet(1, 8);

    U21_to_UTF8(P, u32basis, u8basis);

    P.CreateKernelCall<P2SKernel>(u8basis, u8bytes);

    P.CreateKernelCall<StdOutKernel>(u8bytes);

    return reinterpret_cast<u32u8FunctionType>(P.compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&u32u8Options, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    CPUDriver driver("u32u8");
    auto u32u8Function = u32u8_gen(driver);
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        #ifdef REPORT_PAPI_TESTS
        papi::PapiCounter<4> jitExecution{{PAPI_L3_TCM, PAPI_L3_TCA, PAPI_TOT_INS, PAPI_TOT_CYC}};
        jitExecution.start();
        #endif
        u32u8Function(fd);
        #ifdef REPORT_PAPI_TESTS
        jitExecution.stop();
        jitExecution.write(std::cerr);
        #endif
        close(fd);
    }
    return 0;
}
