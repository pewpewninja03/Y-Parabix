/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>

#ifndef LLVM_VERSION_CODE
// #defines for comparison with LLVM_VERSION_INTEGER
#define LLVM_VERSION_CODE(major, minor, point) ((10000 * major) + (100 * minor) + point)
#endif

namespace llvm { namespace cl { class OptionCategory; } }

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(18, 0, 0)
namespace llvm {
using CodeGenOptLevel = CodeGenOpt::Level;
}
#endif

namespace codegen {

extern llvm::cl::OptionCategory CodeGenOptions;

const llvm::cl::OptionCategory * LLVM_READONLY codegen_flags();

// Command Parameters
enum DebugFlags {
    VerifyIR,
    SerializeThreads,
    TraceCounts,
    TraceDynamicBuffers,
    TraceDynamicMultithreading,
    TraceStridesPerSegment,
    TraceProducedItemCounts,
    TraceUnconsumedItemCounts,
    TraceBlockedIO,
    GenerateTransferredItemCountHistogram,
    GenerateDeferredItemCountHistogram,
    EnableAsserts,
    EnablePipelineAsserts,
    EnableMProtect,
    EnableCycleCounter,
    EnableBlockingIOCounter,
    #ifdef ENABLE_PAPI
    DisplayPAPICounterThreadTotalsOnly,
    #endif
    DisableIndirectBranch,
    DisableThreadLocalStreamSets,
    DisableCacheAlignedKernelStructs,
    DisableInOutAttributes,
    PrintPipelineGraph,
    PrintKernelSizes,
    ForcePipelineRecompilation,
    DebugFlagSentinel
};

enum PipelineCompilationModeOptions {
    DefaultFast
    , Expensive
};

extern bool PabloTransposition;
extern bool SplitTransposition;

bool LLVM_READONLY DebugOptionIsSet(const DebugFlags flag);

bool LLVM_READONLY AnyDebugOptionIsSet();

bool LLVM_READONLY AnyAssertionOptionIsSet();

// Options for generating IR or ASM to files
const std::string OmittedOption = ".";
extern std::string ShowUnoptimizedIROption;
extern std::string ShowIROption;
extern std::string TraceOption;
extern std::string CCCOption;
extern PipelineCompilationModeOptions PipelineCompilationMode;
#ifdef ENABLE_PAPI
extern std::string PapiCounterOptions;
#endif
extern std::string ShowASMOption;
extern const char * ObjectCacheDir;
extern unsigned CacheDaysLimit;  // set from command line
extern int FreeCallBisectLimit;  // set from command line
extern llvm::CodeGenOptLevel OptLevel;  // set from command line
extern llvm::CodeGenOptLevel BackEndOptLevel;  // set from command line
const unsigned LaneWidth = 64;
extern unsigned BlockSize;  // set from command line
extern unsigned SegmentSize; // set from command line
extern unsigned BufferSegments;
extern unsigned TaskThreads;
extern unsigned SegmentThreads;
extern unsigned ScanBlocks;
extern bool EnableObjectCache;
extern bool EnablePipelineObjectCache;
extern bool EnableDynamicMultithreading;
extern bool TraceObjectCache;
extern unsigned GroupNum;
extern const char * ProgramName;
extern llvm::TargetOptions target_Options;
extern bool TimeKernelsIsEnabled;
extern unsigned Z3_Timeout;
extern bool EnableIllustrator;
extern int IllustratorDisplay;
extern float DynamicMultithreadingAddThreshold;
extern float DynamicMultithreadingRemoveThreshold;
extern size_t DynamicMultithreadingPeriod;
extern bool EnableJumpGuidedSynchronizationVariables;
extern bool UseProcessThreadForIO;

void ParseCommandLineOptions(int argc, const char *const *argv, std::initializer_list<const llvm::cl::OptionCategory *> hiding = {});

void AddParabixVersionPrinter();

void setTaskThreads(unsigned taskThreads);
}

