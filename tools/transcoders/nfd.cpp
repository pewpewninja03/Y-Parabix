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
#include <ucd/algo/normalization.h>
#include <ucd/core/unicode_set.h>
#include <ucd/data/PropertyAliases.h>
#include <ucd/data/PropertyObjects.h>
#include <ucd/data/PropertyObjectTable.h>
#include <ucd/utf/utf_compiler.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decompositon Options", "Decompositon Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));
static cl::opt<bool> NoFocus("NoFocus", cl::desc("Process the entire file without filtering to narrow the focus of work"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteMerging("ByteMerging", cl::desc("Use byte stream merging of transformed and unmodified data"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> ByteReplace("ByteReplace", cl::desc("Perform byte merging using the ByteReplaceByMask kernel"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<int> SeparatedPipelineStages("SeparatedPipelineStages", cl::desc("Use multiple separated pipeline stages"), cl::init(0), cl::cat(NFD_Options));
static cl::opt<bool> FilterViolations("FilterViolations", cl::desc("Only include reorderable sequences in work items if they are misordered or start with a decomposable character"), cl::init(false), cl::cat(NFD_Options));
static cl::opt<bool> UseIndexedShiftBack("IndexedShiftBack", cl::desc("Use IndexedShiftBack in place of Filter/Spread combination"), cl::init(false), cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

void source_input_stage(PipelineBuilder & P, Scalar *const fileDescriptor, StreamSet * ByteStream, StreamSet * BasisBits) {

    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);

    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
}

typedef void (*SourceInputFunctionType)(StreamSetPtr &, StreamSetPtr &, uint32_t fd);

SourceInputFunctionType source_input_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Output<streamset_t>("ByteStream", 1, 8, ReturnedBuffer(1)),
                            Output<streamset_t>("BasisBits", 8, 1, ReturnedBuffer(1)),
                            Input<uint32_t>("inputFileDecriptor")
                            );
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * const ByteStream = P.getOutputStreamSet("ByteStream");
    StreamSet * const BasisBits = P.getOutputStreamSet("BasisBits");
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);
    return P.compile();
}

typedef void (*NFD_FilterFunctionType)(const StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, StreamSetPtr &);

NFD_FilterFunctionType NFD_filter_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("BasisBits", 8, 1),
                            Output<streamset_t>("WorkSelectionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("FinalWorkPlacementMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("WorkingBasis", 8, 1, ReturnedBuffer(1))
                            );
    NFD_PipelineBuilder NFD_builder(P);
    StreamSet * const BasisBits = P.getInputStreamSet("BasisBits");
    StreamSet * const WorkSelectionMask = P.getOutputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getOutputStreamSet("FinalWorkPlacementMask");
    StreamSet * const WorkingBasis = P.getOutputStreamSet("WorkingBasis");

    NFD_builder.NFD_FilterStage(BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);
    return P.compile();
}

typedef void (*NFD_TransformFunctionType)(const StreamSetPtr &, StreamSetPtr &);

NFD_TransformFunctionType NFD_transform_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkingBasis", 8, 1),
                            Output<streamset_t>("TransformedBasis", 8, 1, ReturnedBuffer(1))
                            );
    NFD_PipelineBuilder NFD_builder(P);
    StreamSet * const WorkingBasis = P.getInputStreamSet("WorkingBasis");
    StreamSet * const TransformedBasis = P.getOutputStreamSet("TransformedBasis");

    NFD_builder.NFD_U8_Pipeline(WorkingBasis, TransformedBasis);
    return P.compile();
}

void OutputAssemblyStage(PipelineBuilder & P, StreamSet * WorkSelectionMask, StreamSet * FinalWorkPlacementMask, StreamSet * Source, StreamSet * TransformedBasis) {

    StreamSet * const NonModifiedMask = P.CreateStreamSet(1, 1);
    Invert(P, WorkSelectionMask, NonModifiedMask);
    SHOW_STREAM(NonModifiedMask);

    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    if (ByteMerging) {
        StreamSet * const NonModified = P.CreateStreamSet(1, 8);
        FilterByMask(P, NonModifiedMask, Source, NonModified);
        SHOW_BYTES(NonModified);
        
        StreamSet * const NonModifiedPlacementMask = P.CreateStreamSet(1, 1);
        Invert(P, FinalWorkPlacementMask, NonModifiedPlacementMask);
        SHOW_STREAM(NonModifiedPlacementMask);
        
        StreamSet * const NonModifiedPlaced = P.CreateStreamSet(1, 8);
        SpreadByMask(P, NonModifiedPlacementMask, NonModified, NonModifiedPlaced);
        
        StreamSet * const TransformedBytes = P.CreateStreamSet(1, 8);
        P.CreateKernelCall<P2SKernel>(TransformedBasis, TransformedBytes);

        if (ByteReplace) {
            P.CreateKernelCall<ByteReplaceByMask>(FinalWorkPlacementMask, NonModifiedPlaced, TransformedBytes, OutputBytes);
        } else {
            StreamSet * const TransformedPlaced = P.CreateStreamSet(1, 8);
            SpreadByMask(P, FinalWorkPlacementMask, TransformedBytes, TransformedPlaced);

            P.CreateKernelCall<ByteCombine>(NonModifiedPlaced, TransformedPlaced, OutputBytes);
        }
    } else {
        StreamSet * const NonModifiedBasis = P.CreateStreamSet(8);
        FilterByMask(P, NonModifiedMask, Source, NonModifiedBasis);
        SHOW_BIXNUM(NonModifiedBasis);

        StreamSet * const OutputBasis = P.CreateStreamSet(8);
        MergeByMask(P, FinalWorkPlacementMask, TransformedBasis, NonModifiedBasis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);

        P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
    }
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}

void SimpleOutputStage(PipelineBuilder & P, StreamSet * TransformedBasis) {
    StreamSet * const OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(TransformedBasis, OutputBytes);
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}

typedef void (*CombinedWorkFunctionType)(StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, uint32_t fd);

CombinedWorkFunctionType generate_combined_work_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Output<streamset_t>("WorkSelectionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("FinalWorkPlacementMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("ByteStream", 1, 8, ReturnedBuffer(1)),
                            Output<streamset_t>("TransformedBasis", 8, 1, ReturnedBuffer(1)),
                            Input<uint32_t>("inputFileDecriptor")
                            );
    NFD_PipelineBuilder NFD_builder(P);
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * const WorkSelectionMask = P.getOutputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getOutputStreamSet("FinalWorkPlacementMask");
    StreamSet * const ByteStream = P.getOutputStreamSet("ByteStream");
    StreamSet * const TransformedBasis = P.getOutputStreamSet("TransformedBasis");

    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);

    StreamSet * WorkingBasis = P.CreateStreamSet(8, 1);
    NFD_builder.NFD_FilterStage(BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);

    NFD_builder.NFD_U8_Pipeline(WorkingBasis, TransformedBasis);

    return P.compile();
}

typedef void (*OutputAssemblyFunctionType)(const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &);

OutputAssemblyFunctionType generate_output_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkSelectionMask", 1, 1),
                            Input<streamset_t>("FinalWorkPlacementMask", 1, 1),
                            Input<streamset_t>("Source", ByteMerging ? 1 : 8, ByteMerging ? 8 : 1),
                            Input<streamset_t>("TransformedBasis", 8, 1)
                            );
    StreamSet * const WorkSelectionMask = P.getInputStreamSet("WorkSelectionMask");
    StreamSet * const FinalWorkPlacementMask = P.getInputStreamSet("FinalWorkPlacementMask");
    StreamSet * const Source = P.getInputStreamSet("Source");
    StreamSet * const TransformedBasis = P.getInputStreamSet("TransformedBasis");

    OutputAssemblyStage(P, WorkSelectionMask, FinalWorkPlacementMask, Source, TransformedBasis);
    return P.compile();
}

typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_unitary_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver, Input<uint32_t>("inputFileDecriptor"));
    NFD_PipelineBuilder NFD_builder(P);
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    StreamSet * const BasisBits = P.CreateStreamSet(8, 1);
    source_input_stage(P, fileDescriptor, ByteStream, BasisBits);

    if (NoFocus) {
        StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
        NFD_builder.NFD_U8_Pipeline(BasisBits, TransformedBasis);

        SimpleOutputStage(P, TransformedBasis);
    } else {
        StreamSet * WorkSelectionMask = P.CreateStreamSet(1, 1);
        StreamSet * FinalWorkPlacementMask = P.CreateStreamSet(1, 1);
        StreamSet * WorkingBasis = P.CreateStreamSet(8, 1);
        NFD_builder.NFD_FilterStage(BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);
        
        StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
        NFD_builder.NFD_U8_Pipeline(WorkingBasis, TransformedBasis);
        
        OutputAssemblyStage(P, WorkSelectionMask, FinalWorkPlacementMask, ByteMerging ? ByteStream : BasisBits, TransformedBasis);
    }
    return P.compile();
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
        exit(-1);
    }
    if (ByteReplace) ByteMerging = true;
    if (UseIndexedShiftBack) FilterViolations = true;
    CPUDriver driver("NFD_function");
    if (SeparatedPipelineStages == 2) {
        CombinedWorkFunctionType working_stages = generate_combined_work_pipeline(driver);
        kernel::StreamSetPtr WorkSelectionMask;
        kernel::StreamSetPtr FinalWorkPlacementMask;
        kernel::StreamSetPtr ByteStream;
        kernel::StreamSetPtr TransformedBasis;
        working_stages(WorkSelectionMask, FinalWorkPlacementMask, ByteStream, TransformedBasis, fd);
        close(fd);

        OutputAssemblyFunctionType stage3 = generate_output_pipeline(driver);
        stage3(WorkSelectionMask, FinalWorkPlacementMask, ByteStream, TransformedBasis);
    } else if (SeparatedPipelineStages == 4) {
        SourceInputFunctionType stage0 = source_input_pipeline(driver);
        kernel::StreamSetPtr ByteStream;
        kernel::StreamSetPtr BasisBits;
        stage0(ByteStream, BasisBits, fd);
        close(fd);

        NFD_FilterFunctionType stage1 = NFD_filter_pipeline(driver);
        kernel::StreamSetPtr WorkSelectionMask;
        kernel::StreamSetPtr FinalWorkPlacementMask;
        kernel::StreamSetPtr WorkingBasis;
        stage1(BasisBits, WorkSelectionMask, FinalWorkPlacementMask, WorkingBasis);

        NFD_TransformFunctionType stage2 = NFD_transform_pipeline(driver);
        kernel::StreamSetPtr TransformedBasis;
        stage2(WorkingBasis, TransformedBasis);

        OutputAssemblyFunctionType stage3 = generate_output_pipeline(driver);
        stage3(WorkSelectionMask, FinalWorkPlacementMask, ByteMerging ? ByteStream : BasisBits, TransformedBasis);
    } else {
        XfrmFunctionType fn = generate_unitary_pipeline(driver);
        fn(fd);
        close(fd);
    }
    return 0;
}
