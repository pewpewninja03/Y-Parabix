/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <toolchain/toolchain.h>
#include <unicode/core/UCD_Config.h>
#include <llvm/Support/CommandLine.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif
#include <llvm/Support/raw_ostream.h>
#include <boost/interprocess/mapped_region.hpp>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#include <thread>
#endif

using namespace llvm;

#ifndef NDEBUG
#define IN_DEBUG_MODE true
#else
#define IN_DEBUG_MODE false
#endif

// #define FORCE_ASSERTIONS

// #define DISABLE_OBJECT_CACHE

namespace codegen {

inline unsigned getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

cl::OptionCategory CodeGenOptions("Code Generation Options", "These options control code generation.");

static cl::bits<DebugFlags>
DebugOptions(cl::desc("Debugging Options"), cl::values(clEnumVal(VerifyIR, "Run the IR verification pass."),
                        clEnumVal(SerializeThreads, "Force segment threads to run sequentially."),
                        clEnumVal(TraceCounts, "Trace kernel processed, consumed and produced item counts."),
                        clEnumVal(TraceDynamicBuffers, "Trace dynamic buffer allocations and deallocations."),
                        clEnumVal(TraceDynamicMultithreading, "Trace dynamic multithreading thread count state."),
                        clEnumVal(TraceBlockedIO, "Trace kernels prevented from processing any strides "
                                                  "due to insufficient input items / output space."),
                        clEnumVal(TraceStridesPerSegment, "Trace number of strides executed over segments."),
                        clEnumVal(TraceProducedItemCounts, "Trace produced item count deltas over segments."),
                        clEnumVal(TraceUnconsumedItemCounts, "Trace unconsumed item counts over segments."),
                        clEnumVal(GenerateTransferredItemCountHistogram, "Generate a histogram CSV of each non-Fixed port detailing "
                                                                         "the transfered item count per executed stride."),
                        clEnumVal(GenerateDeferredItemCountHistogram, "Generate a histogram CSV of each deferred port detailing "
                                                                      "the difference between the deferred and total item count "
                                                                      "per executed stride."),
                        clEnumVal(EnableAsserts, "Enable built-in Parabix framework asserts in all generated IR."),
                        clEnumVal(EnablePipelineAsserts, "Enable built-in Parabix framework asserts in generated pipeline IR."),
                        clEnumVal(EnableMProtect, "Use mprotect to cause a write fault when erroneously "
                                                  "overwriting kernel state / stream space."),
                        clEnumVal(EnableCycleCounter, "Count and report CPU cycles per kernel."),
                        clEnumVal(EnableBlockingIOCounter, "Count and report the number of blocked kernel "
                                                           "executions due to insufficient data/space of a "
                                                           "particular stream."),
                        clEnumVal(DisableIndirectBranch, "Disable use of indirect branches in kernel code."),
                        clEnumVal(DisableThreadLocalStreamSets, "Disable use of thread-local memory for streamsets within the same partition."),

                        #ifdef ENABLE_PAPI
                        clEnumVal(DisplayPAPICounterThreadTotalsOnly, "Disable per-kernel PAPI counters when given a valid PapiCounters list."),
                        #endif

                        clEnumVal(DisableCacheAlignedKernelStructs, "Disable cache alignment of kernel state memory."),

                        clEnumVal(DisableInOutAttributes, "Disable In/Out attributes for streamset data buffers."),

                        clEnumVal(PrintKernelSizes, "Write kernel state object size in bytes to stderr."),
                        clEnumVal(PrintPipelineGraph, "Write PipelineKernel graph in dot file format to stderr."),
                        clEnumVal(ForcePipelineRecompilation, "Disable object cache lookup for any PipelineKernel.")), cl::cat(CodeGenOptions));

std::string ShowIROption = OmittedOption;
static cl::opt<std::string, true> IROutputOption("ShowIR", cl::location(ShowIROption), cl::ValueOptional,
                                                         cl::desc("Print optimized LLVM IR to stderr (by omitting =<filename>) or a file"), cl::value_desc("filename"), cl::cat(CodeGenOptions));

std::string ShowUnoptimizedIROption = OmittedOption;
static cl::opt<std::string, true> UnoptimizedIROutputOption("ShowUnoptimizedIR", cl::location(ShowUnoptimizedIROption), cl::ValueOptional,
                                                         cl::desc("Print generated LLVM IR to stderr (by omitting =<filename> or a file"), cl::value_desc("filename"), cl::cat(CodeGenOptions));

#ifdef ENABLE_PAPI
std::string PapiCounterOptions = OmittedOption;
static cl::opt<std::string, true> clPapiCounterOptions("PapiCounters", cl::location(PapiCounterOptions), cl::ValueOptional,
                                                       cl::desc("comma delimited list of PAPI event names (run papi_avail for options)"),
                                                       cl::value_desc("comma delimited list"), cl::cat(CodeGenOptions));
#endif

std::string ShowASMOption = OmittedOption;
static cl::opt<std::string, true> ASMOutputFilenameOption("ShowASM", cl::location(ShowASMOption), cl::ValueOptional,
                                                         cl::desc("Print generated assembly code to stderr (by omitting =<filename> or a file"), cl::value_desc("filename"), cl::cat(CodeGenOptions));

// Enable Debug Options to be specified on the command line

static cl::opt<CodeGenOptLevel, true>
OptimizationLevel("optimization-level", cl::location(OptLevel), cl::init(CodeGenOptLevel::None), cl::desc("Set the front-end optimization level:"),
                  cl::values(clEnumValN(CodeGenOptLevel::None, "none", "no optimizations (default)"),
                             clEnumValN(CodeGenOptLevel::Less, "less", "trivial optimizations"),
                             clEnumValN(CodeGenOptLevel::Default, "standard", "standard optimizations"),
                             clEnumValN(CodeGenOptLevel::Aggressive, "aggressive", "aggressive optimizations")), cl::cat(CodeGenOptions));
static cl::opt<CodeGenOptLevel, true>
BackEndOptOption("backend-optimization-level", cl::location(BackEndOptLevel), cl::init(CodeGenOptLevel::None), cl::desc("Set the back-end optimization level:"),
                  cl::values(clEnumValN(CodeGenOptLevel::None, "none", "no optimizations (default)"),
                             clEnumValN(CodeGenOptLevel::Less, "less", "trivial optimizations"),
                             clEnumValN(CodeGenOptLevel::Default, "standard", "standard optimizations"),
                             clEnumValN(CodeGenOptLevel::Aggressive, "aggressive", "aggressive optimizations")), cl::cat(CodeGenOptions));

PipelineCompilationModeOptions PipelineCompilationMode = PipelineCompilationModeOptions::DefaultFast;

static cl::opt<PipelineCompilationModeOptions, true>
PipelineCompilationModeOption("pipeline-optimization-level", cl::location(PipelineCompilationMode),
                  cl::init(PipelineCompilationModeOptions::DefaultFast),
                  cl::desc("Set the pipeline optimization level:"),
                  cl::values(clEnumValN(PipelineCompilationModeOptions::DefaultFast, "fast", "minimal analysis(default)"),
                             clEnumValN(PipelineCompilationModeOptions::Expensive, "aggressive", "full analysis")), cl::cat(CodeGenOptions));

static cl::opt<bool, true> EnableObjectCacheOption("enable-object-cache", cl::location(EnableObjectCache), cl::init(true),
                                                   cl::desc("Enable object caching"), cl::cat(CodeGenOptions));

static cl::opt<bool, true> TraceObjectCacheOption("trace-object-cache", cl::location(TraceObjectCache), cl::init(false),
                                                   cl::desc("Trace object cache retrieval."), cl::cat(CodeGenOptions));

static cl::opt<std::string> ObjectCacheDirOption("object-cache-dir", cl::init(""),
                                                 cl::desc("Path to the object cache diretory"), cl::cat(CodeGenOptions));

bool EnableDynamicMultithreading;
static cl::opt<bool, true> EnableDynamicMultithreadingOption("dynamic-multithreading", cl::location(EnableDynamicMultithreading), cl::init(false),
                                                   cl::desc("Dynamic multithreading."), cl::cat(CodeGenOptions));

float DynamicMultithreadingAddThreshold;
static cl::opt<float, true> DynamicMultithreadingAddThresholdOption("dynamic-multithreading-add-threshold", cl::location(DynamicMultithreadingAddThreshold), cl::init(10.0),
                                                   cl::desc("Dynamic multithreading."), cl::cat(CodeGenOptions));

float DynamicMultithreadingRemoveThreshold;
static cl::opt<float, true> DynamicMultithreadingRemoveThresholdOption("dynamic-multithreading-remove-threshold", cl::location(DynamicMultithreadingRemoveThreshold), cl::init(15.0),
                                                   cl::desc("Dynamic multithreading."), cl::cat(CodeGenOptions));

size_t DynamicMultithreadingPeriod;
static cl::opt<size_t, true> DynamicMultithreadingPeriodOption("dynamic-multithreading-period", cl::location(DynamicMultithreadingPeriod), cl::init(100),
                                                   cl::desc("Dynamic multithreading."), cl::cat(CodeGenOptions));


bool EnableJumpGuidedSynchronizationVariables;
static cl::opt<bool, true> EnableJumpGuidedSynchronizationVariablesOption("partition-guided-synchronization", cl::location(EnableJumpGuidedSynchronizationVariables), cl::init(false),
                                                   cl::desc("Enable partition jump guided synchronization variables."), cl::cat(CodeGenOptions));


static cl::opt<int, true> FreeCallBisectOption("free-bisect-value", cl::location(FreeCallBisectLimit), cl::init(-1),
                                                    cl::desc("The number of free calls to allow in bisecting"), cl::cat(CodeGenOptions));

static cl::opt<unsigned, true> BlockSizeOption("BlockSize", cl::location(BlockSize), cl::init(0),
                                          cl::desc("specify a block size (defaults to widest SIMD register width in bits)."), cl::cat(CodeGenOptions));


const unsigned DefaultSegmentSize = 16384;
static cl::opt<unsigned, true> SegmentSizeOption("segment-size", cl::location(SegmentSize),
                                               cl::init(DefaultSegmentSize),
                                               cl::desc("Expected amount of input data to process per segment"), cl::value_desc("positive integer"), cl::cat(CodeGenOptions));

static cl::opt<unsigned, true> BufferSegmentsOption("buffer-segments", cl::location(BufferSegments), cl::init(1),
                                               cl::desc("Buffer Segments"), cl::value_desc("positive integer"));


unsigned Z3_Timeout;
static cl::opt<unsigned, true> Z3_TimeoutOption("Z3-timeout", cl::location(Z3_Timeout), cl::init(3000),
                                               cl::desc("Z3 timeout"), cl::value_desc("positive integer"));

bool PabloTransposition;
static cl::opt<bool, true> OptPabloTransposition("enable-pablo-s2p", cl::location(PabloTransposition),
                                                 cl::desc("Enable experimental pablo transposition."), cl::init(false), cl::cat(CodeGenOptions));

bool SplitTransposition;
static cl::opt<bool, true> OptSplitTransposition("enable-split-s2p", cl::location(SplitTransposition),
                                                 cl::desc("Enable experimental split transposition."), cl::init(false), cl::cat(CodeGenOptions));

static cl::opt<unsigned, true>
MaxTaskThreadsOption("max-task-threads", cl::location(TaskThreads),
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
                     cl::init(std::thread::hardware_concurrency()),
#else
                     cl::init(llvm::sys::getHostNumPhysicalCores()),
#endif
                     cl::desc("Maximum number of threads to assign for separate pipeline tasks."),
                     cl::value_desc("positive integer"));

static cl::opt<unsigned, true>
ThreadNumOption("thread-num", cl::location(SegmentThreads),
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
                cl::init(std::thread::hardware_concurrency() - 1),
#else
                // If we have more than 2 cores, leave one for other processes,
                // and use the rest for multithreading of the pipeline.
                cl::init(std::max(llvm::sys::getHostNumPhysicalCores() - 1, 2)),
#endif
                cl::desc("Number of threads used for segment pipeline parallel"),
                cl::value_desc("positive integer"));

static cl::opt<unsigned, true> ScanBlocksOption("scan-blocks", cl::location(ScanBlocks), cl::init(4),
                                          cl::desc("Number of blocks per stride for scanning kernels"), cl::value_desc("positive initeger"));

static cl::opt<unsigned, true> GroupNumOption("group-num", cl::location(GroupNum), cl::init(256),
                                         cl::desc("NUmber of groups declared on GPU"), cl::value_desc("positive integer"), cl::cat(CodeGenOptions));

std::string TraceOption = "";
static cl::opt<std::string, true> TraceValueOption("trace", cl::location(TraceOption),
                                            cl::desc("Trace the values of variables beginning with the given prefix."), cl::value_desc("prefix"), cl::cat(CodeGenOptions));

bool EnableIllustrator;
static cl::opt<bool, true> OptEnableIllustrator("enable-illustrator", cl::location(EnableIllustrator),
                                                 cl::desc("Enable bitstream illustrator with the default display width."), cl::init(0), cl::cat(CodeGenOptions));

int IllustratorDisplay;
static cl::opt<int, true> OptIllustratorWidth("illustrator-width", cl::location(IllustratorDisplay),
                                                 cl::desc("Enable bitstream illustrator with the given display width."), cl::init(0), cl::cat(CodeGenOptions));

std::string CCCOption = "";
static cl::opt<std::string, true> CCTypeOption("ccc-type", cl::location(CCCOption), cl::init("binary"),
                                            cl::desc("The character class compiler"), cl::value_desc("[binary, ternary]"));

bool TimeKernelsIsEnabled;
static cl::opt<bool, true> OptCompileTime("time-kernels", cl::location(TimeKernelsIsEnabled),
                                        cl::desc("Times each kernel, printing elapsed time for each on exit"), cl::init(false));


bool UseProcessThreadForIO;
static cl::opt<bool, true> OptUseProcessThreadForIO("io-thread", cl::location(UseProcessThreadForIO),
                                        cl::desc("Only permit the process thread to perform IO"), cl::init(false));

CodeGenOptLevel OptLevel;
CodeGenOptLevel BackEndOptLevel;

const char * ObjectCacheDir;

unsigned BlockSize;

unsigned SegmentSize;

unsigned BufferSegments;
unsigned TaskThreads;
unsigned SegmentThreads;

unsigned ScanBlocks;

bool EnableObjectCache = true;
bool EnablePipelineObjectCache = true;
bool TraceObjectCache;

unsigned CacheDaysLimit;

int FreeCallBisectLimit;

unsigned GroupNum;

TargetOptions target_Options;

const cl::OptionCategory * LLVM_READONLY codegen_flags() {
    return &CodeGenOptions;
}

bool LLVM_READONLY DebugOptionIsSet(const DebugFlags flag) {
    #ifdef FORCE_ASSERTIONS
    if (flag == DebugFlags::EnableAsserts) return true;
    #endif
    return DebugOptions.isSet(flag);
}

bool LLVM_READONLY AnyDebugOptionIsSet() {
    #ifdef FORCE_ASSERTIONS
    return true;
    #endif
    return DebugOptions.getBits() != 0;
}

bool LLVM_READONLY AnyAssertionOptionIsSet() {
    #ifdef FORCE_ASSERTIONS
    return true;
    #endif
    return DebugOptions.isSet(DebugFlags::EnableAsserts) || DebugOptions.isSet(DebugFlags::EnablePipelineAsserts);
}

const char * ProgramName;

inline bool disableObjectCacheDueToCommandLineOptions() {
    if (!TraceOption.empty()) return true;
    if (DebugOptions.isSet(PrintKernelSizes)) return true;
    if (DebugOptions.isSet(PrintPipelineGraph)) return true;
    if (ShowIROption != OmittedOption) return true;
    if (ShowUnoptimizedIROption != OmittedOption) return true;
    if (ShowASMOption != OmittedOption) return true;
//    if (pablo::ShowPabloOption != OmittedOption) return true;
//    if (pablo::ShowOptimizedPabloOption != OmittedOption) return true;
    return false;
}

inline bool disablePipelineObjectCacheDueToCommandLineOptions() {
    if (DebugOptions.isSet(PrintPipelineGraph)) return true;
    if (DebugOptions.isSet(EnablePipelineAsserts)) return true;
    if (DebugOptions.isSet(DisableThreadLocalStreamSets)) return true;
    if (DebugOptions.isSet(ForcePipelineRecompilation)) return true;
    return false;
}


void ParseCommandLineOptions(int argc, const char * const *argv, std::initializer_list<const cl::OptionCategory *> hiding) {
    AddParabixVersionPrinter();

    codegen::ProgramName = argv[0];
    if (hiding.size() != 0) {
        cl::HideUnrelatedOptions(ArrayRef<const cl::OptionCategory *>(hiding));
    }
    cl::ParseCommandLineOptions(argc, argv);
//    if (LLVM_UNLIKELY(!PabloIllustrateBitstreamRegEx.empty() || IllustratorDisplay != 0)) {
//        EnableIllustrator = true;
//    }
    if (disableObjectCacheDueToCommandLineOptions()) {
        EnableObjectCache = false;
    } else if (disablePipelineObjectCacheDueToCommandLineOptions()) {
        EnablePipelineObjectCache = false;
    }
    ObjectCacheDir = ObjectCacheDirOption.empty() ? nullptr : ObjectCacheDirOption.data();
    target_Options.MCOptions.AsmVerbose = true;
}

void printParabixVersion (raw_ostream & outs) {
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
    llvm::sys::printDefaultTargetAndDetectedCPU(outs);
#endif
    outs << "Unicode version " << UCD::UnicodeVersion << "\n";
    outs << "Parabix (http://parabix.costar.sfu.ca/):\n  " << "Parabix revision " << PARABIX_VERSION << "\n";
}

void AddParabixVersionPrinter() {
    cl::AddExtraVersionPrinter(&printParabixVersion);
}

void setTaskThreads(unsigned taskThreads) {
    TaskThreads = std::max(taskThreads, 1u);
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
    unsigned coresPerTask = std::thread::hardware_concurrency()/TaskThreads;
#else
    unsigned coresPerTask = llvm::sys::getHostNumPhysicalCores()/TaskThreads;
#endif
    SegmentThreads = std::min(coresPerTask, SegmentThreads);
}

}
