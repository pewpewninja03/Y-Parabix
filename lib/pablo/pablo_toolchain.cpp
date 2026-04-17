/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <pablo/pablo_toolchain.h>

#include <toolchain/toolchain.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(7, 0, 0)
#define OF_None F_None
#endif

using namespace llvm;

namespace pablo {

static cl::OptionCategory PabloOptions("Pablo Options", "These options control printing, generation and instrumentation of Pablo intermediate code.");

const cl::OptionCategory * pablo_toolchain_flags() {
    return &PabloOptions;
}


std::string ShowPabloOption = codegen::OmittedOption;
static cl::opt<std::string, true> PabloOutputOption("ShowPablo", cl::location(ShowPabloOption), cl::ValueOptional,
    cl::desc("Print generated Pablo code to stderr (by omitting =<filename>) or a file"),
    cl::value_desc("filename"), cl::cat(codegen::JIT_InfoOptions));
sys::fs::OpenFlags PabloOutputFileFlag = sys::fs::OpenFlags::OF_None;
    
std::string ShowOptimizedPabloOption = codegen::OmittedOption;
static cl::opt<std::string, true> OptimizedPabloOutputOption("ShowOptimizedPablo", cl::location(ShowOptimizedPabloOption), cl::ValueOptional,
    cl::desc("Print optimized Pablo code to stderr (by omitting =<filename>) or a file"),
    cl::value_desc("filename"), cl::cat(codegen::JIT_InfoOptions));
sys::fs::OpenFlags PabloOptimizedOutputFileFlag = sys::fs::OpenFlags::OF_None;

static cl::bits<PabloCompilationFlags> 
    PabloOptimizationsOptions(cl::values(clEnumVal(Flatten, "Flatten all the Ifs in the Pablo AST"),
                                         clEnumVal(DisableSimplification, "Disable Pablo Simplification pass (not recommended)"),
                                         clEnumVal(DisableCodeMotion, "Moves statements into the innermost legal If-scope and moves invariants out of While-loops."),
                                         clEnumVal(EnableDistribution, "Apply distribution law optimization."),                                         
                                         clEnumVal(EnableSchedulingPrePass, "Pablo Statement Scheduling Pre-Pass"),
                                         clEnumVal(EnableProfiling, "Profile branch statistics."),
                                         clEnumVal(EnableTernaryOpt, "Enable ternary optimization."),
                                         clEnumVal(PabloUseLLVMOptimizationPasses, "Run aggressive LLVM optimization passes on compiled Pablo code."),
                                         clEnumVal(VerifyPablo, "Run the Pablo verifier")),
    cl::cat(codegen::CodeGenOptions), cl::Hidden);

bool CompileOptionIsSet(const PabloCompilationFlags flag) {return PabloOptimizationsOptions.isSet(flag);}

PabloCarryMode CarryMode;
static cl::opt<PabloCarryMode, true> PabloCarryModeOptions("CarryMode", cl::desc("Carry mode for pablo compiler (default BitBlock)"), 
    cl::location(CarryMode), cl::ValueOptional,
    cl::values(
        clEnumValN(PabloCarryMode::BitBlock, "BitBlock", "All carries are stored as bit blocks."),
        clEnumValN(PabloCarryMode::Compressed, "Compressed", "When possible, carries are stored as 64-bit integers.")),
    cl::cat(codegen::CodeGenOptions), cl::init(PabloCarryMode::Compressed));

std::string BitMovementMode_string(BitMovementMode m) {
    if (m == BitMovementMode::Advance) return "Advance";
    else return "LookAhead";
}

std::string PabloIllustrateKernelRegEx = "";
static cl::opt<std::string, true> PabloIllustrateKernelOption("pablo-illustrate-kernel", cl::location(PabloIllustrateBitstreamRegEx), cl::ValueOptional,
    cl::desc("RegEx describing Pablo kernel names to illustrate"), cl::value_desc("regex"), cl::cat(codegen::InstrumentationOptions));


std::string PabloIllustrateBitstreamRegEx = "";
static cl::opt<std::string, true> PabloIllustrateBitstreamOption("pablo-illustrate-bitstream", cl::location(PabloIllustrateBitstreamRegEx), cl::ValueOptional,
    cl::desc("RegEx describing Pablo statement names to illustrate"), cl::value_desc("regex"), cl::cat(codegen::InstrumentationOptions));
}
