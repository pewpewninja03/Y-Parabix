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
    
static cl::bits<PabloDebugFlags> 
DebugOptions(cl::desc("Pablo Debug Flags"),
             cl::values(clEnumVal(VerifyPablo, "Run the Pablo verifier"),
                        clEnumVal(DumpTrace, "Generate dynamic traces of executed Pablo assignments.")), cl::cat(PabloOptions));

std::string ShowPabloOption = codegen::OmittedOption;
static cl::opt<std::string, true> PabloOutputOption("ShowPablo", cl::location(ShowPabloOption), cl::ValueOptional,
                                                    cl::desc("Print generated Pablo code to stderr (by omitting =<filename>) or a file"), cl::value_desc("filename"), cl::cat(PabloOptions));
sys::fs::OpenFlags PabloOutputFileFlag = sys::fs::OpenFlags::OF_None;
    
std::string ShowOptimizedPabloOption = codegen::OmittedOption;
static cl::opt<std::string, true> OptimizedPabloOutputOption("ShowOptimizedPablo", cl::location(ShowOptimizedPabloOption), cl::ValueOptional,
                                                    cl::desc("Print optimized Pablo code to stderr (by omitting =<filename>) or a file"), cl::value_desc("filename"), cl::cat(PabloOptions));
sys::fs::OpenFlags PabloOptimizedOutputFileFlag = sys::fs::OpenFlags::OF_None;

static cl::bits<PabloCompilationFlags> 
    PabloOptimizationsOptions(cl::values(clEnumVal(Flatten, "Flatten all the Ifs in the Pablo AST"),
                                         clEnumVal(DisableSimplification, "Disable Pablo Simplification pass (not recommended)"),
                                         clEnumVal(DisableCodeMotion, "Moves statements into the innermost legal If-scope and moves invariants out of While-loops."),
                                         clEnumVal(EnableDistribution, "Apply distribution law optimization."),                                         
                                         clEnumVal(EnableSchedulingPrePass, "Pablo Statement Scheduling Pre-Pass"),
                                         clEnumVal(EnableProfiling, "Profile branch statistics."),
                                         clEnumVal(EnableTernaryOpt, "Enable ternary optimization.")), cl::cat(PabloOptions));

PabloCarryMode CarryMode;
static cl::opt<PabloCarryMode, true> PabloCarryModeOptions("CarryMode", cl::desc("Carry mode for pablo compiler (default BitBlock)"), 
    cl::location(CarryMode), cl::ValueOptional,
    cl::values(
        clEnumValN(PabloCarryMode::BitBlock, "BitBlock", "All carries are stored as bit blocks."),
        clEnumValN(PabloCarryMode::Compressed, "Compressed", "When possible, carries are stored as 64-bit integers.")),
                                                           cl::cat(PabloOptions), cl::init(PabloCarryMode::Compressed));

BitMovementMode MovementMode;
static cl::opt<BitMovementMode, true> BitMovementOptions("BitMovement", cl::desc("Bit movement option for pablo algorithms"),
    cl::location(MovementMode), cl::ValueOptional,
    cl::values(
        clEnumValN(BitMovementMode::Advance, "Advance", "Encode movements using Advance operations."),
        clEnumValN(BitMovementMode::LookAhead, "LookAhead", "When possible, encode movements using LookAhead operations.")),
                                                         cl::cat(PabloOptions), cl::init(BitMovementMode::Advance));

std::string BitMovementMode_string(BitMovementMode m) {
    if (m == BitMovementMode::Advance) return "Advance";
    else return "LookAhead";
}

std::string PabloIllustrateKernelRegEx = "";
static cl::opt<std::string, true> PabloIllustrateKernelOption("pablo-illustrate-kernel", cl::location(PabloIllustrateBitstreamRegEx), cl::ValueOptional,
                                                         cl::desc("RegEx describing Pablo kernel names to illustrate"), cl::value_desc("regex"), cl::cat(PabloOptions));


std::string PabloIllustrateBitstreamRegEx = "";
static cl::opt<std::string, true> PabloIllustrateBitstreamOption("pablo-illustrate-bitstream", cl::location(PabloIllustrateBitstreamRegEx), cl::ValueOptional,
                                                         cl::desc("RegEx describing Pablo statement names to illustrate"), cl::value_desc("regex"), cl::cat(PabloOptions));


bool PabloUseLLVMOptimizationPasses = false;
static cl::opt<bool, true> optPabloUseLLVMOptimizationPasses("pablo-llvm-optimization-passes", cl::location(PabloUseLLVMOptimizationPasses), cl::ValueOptional, cl::init(false),
                                                         cl::desc("Run aggressive LLVM optimization passes on compiled Pablo code"),  cl::cat(PabloOptions));


bool DebugOptionIsSet(const PabloDebugFlags flag) {return DebugOptions.isSet(flag);}
    
bool CompileOptionIsSet(const PabloCompilationFlags flag) {return PabloOptimizationsOptions.isSet(flag);}

}
