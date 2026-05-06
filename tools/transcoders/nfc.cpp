/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <fcntl.h>
#include <string>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/codegenstate.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/bixnum/bixnum.h>
#include <grep/grep_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/sorting.h>
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
#include <toolchain/fileutil.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <ucd/core/unicode_set.h>
#include <ucd/algo/normalization.h>
#include <ucd/utf/utf_compiler.h>
#include <ucd/data/PropertyObjectTable.h>
#include <re/toolchain/toolchain.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/unicode/UCD_property_kernel.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFC_Options("Decomposition Options", "Decomposition Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFC_Options));
static cl::opt<bool> PreferMMap("useMMap", cl::desc("Use mmap for file input, if possible"), cl::init(false), cl::cat(NFC_Options));
static cl::opt<bool> NoFocus("NoFocus", cl::desc("Process the entire file without filtering to narrow the focus of work"), cl::init(false), cl::cat(NFC_Options));
static cl::opt<bool> MultiStage("MultiStage", cl::desc("Use two pipeline stages"), cl::init(false), cl::cat(NFC_Options));
static cl::opt<bool> ByteFiltering("ByteFiltering", cl::desc("Use byte filtering for focused work"), cl::init(false), cl::cat(NFC_Options));

static cl::opt<bool> UseLayers("UseLayers", cl::desc("Use pipeline layers"), cl::init(false), cl::cat(NFC_Options));


//static cl::opt<bool> U21("U21", cl::desc("perform character translation via 21-bit Unicode"),  cl::cat(NFC_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

class NFC_Focus : public pablo::PabloKernel {
public:
    NFC_Focus(LLVMTypeSystemInterface & ts,
              StreamSet * Basis, StreamSet * Composable2nds, StreamSet * Focus);
protected:
    void generatePabloMethod() override;
};

NFC_Focus::NFC_Focus(LLVMTypeSystemInterface & ts, 
                     StreamSet * Basis, StreamSet * NFC_candidates,
                     StreamSet * Focus)
: PabloKernel(ts, "NFC_Focus",
// inputs
{Binding{"Basis", Basis},
 Binding{"NFC_candidates", NFC_candidates, FixedRate(), LookAhead(4)}},
// output
{Binding{"Focus", Focus}}) {
}

void NFC_Focus::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    std::vector<PabloAST *> NFC_candidates = getInputStreamSet("NFC_candidates");
    PabloAST * composable_seconds = NFC_candidates[0];
    PabloAST * excluded_composites = NFC_candidates[1];
    BixNumCompiler bnc(pb);
    PabloAST * pfx4 = bnc.UGE(Basis, 0xF0);
    PabloAST * pfx3or4 = bnc.UGE(Basis, 0xE0);
    PabloAST * pfx = bnc.UGE(Basis, 0xC2);
    PabloAST * pfx3 = pb.createXor(pfx3or4, pfx4);
    PabloAST * pfx2 = pb.createXor(pfx, pfx3or4);
    PabloAST * ASCII = bnc.ULE(Basis, 0x7F);
    PabloAST * focus = pb.createOr(composable_seconds, pb.createAnd(ASCII, pb.createLookahead(composable_seconds, 1)));
    focus = pb.createOr(focus, pb.createAnd(pfx2, pb.createLookahead(composable_seconds, 2)));
    focus = pb.createOr(focus, pb.createAnd(pfx3, pb.createLookahead(composable_seconds, 3)));
    focus = pb.createOr(focus, pb.createAnd(pfx4, pb.createLookahead(composable_seconds, 4)));
    focus = pb.createOr(focus, excluded_composites);
    pb.createAssign(pb.createExtract(getOutputStreamVar("Focus"), pb.getInteger(0)), focus);
}

using namespace UCD;

void DetermineNFC_WorkSpans(PipelineBuilder & P, StreamSet * U8_Basis, StreamSet * WorkSelectionMask) {
    StreamSet * NFC_Candidates = P.CreateStreamSet(2, 1);
    P.CreateKernelCall<NFC_CandidateClass>(U8_Basis, NFC_Candidates);
    SHOW_BIXNUM(NFC_Candidates);

    StreamSet * NFC_WorkItems = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<NFC_Focus>(U8_Basis, NFC_Candidates, NFC_WorkItems);
    SHOW_STREAM(NFC_WorkItems);

    StreamSet * const u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(U8_Basis, u8index);
    SHOW_STREAM(u8index);

    P.CreateKernelCall<U8Spans>(NFC_WorkItems, u8index, WorkSelectionMask, BitMovementMode::Advance);
    SHOW_STREAM(WorkSelectionMask);
}

void NFC_U8_logic(PipelineBuilder & P, StreamSet * ExpansionMask, StreamSet * U8_Basis, StreamSet * FinalSelectionMask, StreamSet * TransformedBytes) {

    // The SourceNull stream identifies null bytes in the original source
    // stream, rather than ones generated by insertion or by zeroing out
    // characters to be deleted.
    StreamSet * const NullStream = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<bixnum::EQ_immediate>(U8_Basis, 0, NullStream);
    StreamSet * const SourceNull = P.CreateStreamSet(1, 1);
    AndCombine(P, NullStream, ExpansionMask, SourceNull);
    SHOW_STREAM(SourceNull);

    StreamSet * EC_Basis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<ExcludedCompositeStage>(U8_Basis, EC_Basis);
    SHOW_BIXNUM(EC_Basis);

    re::PropertyExpression * CCC0_Prop = re::makePropertyExpression("CCC", "NR");
    CCC0_Prop = cast<re::PropertyExpression>(UCD::linkAndResolve(CCC0_Prop));

    StreamSet * const ccc_NR0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Prop, EC_Basis, ccc_NR0, BitMovementMode::LookAhead);
    SHOW_STREAM(ccc_NR0);

    StreamSet * const ccc_NR = P.CreateStreamSet(1, 1);
    AndCombine(P, ccc_NR0, ExpansionMask, ccc_NR);
    SHOW_STREAM(ccc_NR);

    StreamSet * CanonBasis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<SingletonCanonicalization>(EC_Basis, CanonBasis);
    SHOW_BIXNUM(CanonBasis);

    StreamSet * ShortBasis = P.CreateStreamSet(8, 1);
    ShortComposablePipeline(P, CanonBasis, ShortBasis);
    SHOW_BIXNUM(ShortBasis);

    StreamSet * FinalBasis = P.CreateStreamSet(8, 1);
    LongComposablePipeline(P, ShortBasis, ccc_NR, FinalBasis);
    SHOW_BIXNUM(FinalBasis);

    StreamSet * L_V_T =  P.CreateStreamSet(Hangul_Composables::HC_Kind::Count);
    P.CreateKernelCall<Hangul_Composables>(FinalBasis, L_V_T, pablo::BitMovementMode::LookAhead);
    //P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, FilteredBasis, L_V_T, pablo::BitMovementMode::LookAhead);
    SHOW_BIXNUM(L_V_T);

    StreamSet * TranslatedBasis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<Hangul_Composition>(FinalBasis, L_V_T, TranslatedBasis);
    SHOW_BIXNUM(TranslatedBasis);

    StreamSet * NonZeroResults = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<bixnum::NEQ_immediate>(TranslatedBasis, 0, NonZeroResults);

    OrCombine(P, NonZeroResults, SourceNull, FinalSelectionMask);
    SHOW_STREAM(FinalSelectionMask);

    StreamSet * TransformedBasis = P.CreateStreamSet(8, 1);
    FilterByMask(P, FinalSelectionMask, TranslatedBasis, TransformedBasis);

    UCD::EnumeratedPropertyObject * enumObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(UCD::ccc));
    StreamSet * const CCC_Basis = P.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    P.CreateKernelCall<UnicodePropertyBasis>(enumObj, TransformedBasis, CCC_Basis);
    SHOW_BIXNUM(CCC_Basis);

    StreamSet * const u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(TransformedBasis, u8index);
    SHOW_STREAM(u8index);

    StreamSet * const CCC_Spans = P.CreateStreamSet(enumObj->GetEnumerationBasisSets().size(), 1);
    P.CreateKernelCall<U8Spans>(CCC_Basis, u8index, CCC_Spans);
    SHOW_BIXNUM(CCC_Spans);

    StreamSet * const CCC_NonZero = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<bixnum::NEQ_immediate>(CCC_Spans, 0, CCC_NonZero);
    SHOW_STREAM(CCC_NonZero);

    StreamSets ToSort = {CCC_Spans, TransformedBasis};

    StreamSets SortResults = BitonicSortRuns(P, 32, CCC_NonZero, ToSort);
    SHOW_BIXNUM(SortResults[0]);
    SHOW_BIXNUM(SortResults[1]);

    P.CreateKernelCall<P2SKernel>(SortResults[1], TransformedBytes);
    SHOW_BYTES(TransformedBytes);
}



void source_logic(PipelineBuilder & P, StreamSet * ByteStream, StreamSet * BasisBits) {
    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);

    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);
}


void focus_stage_logic(PipelineBuilder & P, StreamSet * ByteStream, StreamSet * BasisBits, 
    StreamSet * WorkSelectionMask, StreamSet * FocusedWorkBasis, StreamSet * NonModifiedBytes) {

    DetermineNFC_WorkSpans(P, BasisBits, WorkSelectionMask);

    if (ByteFiltering) {
        FilterByMask(P, WorkSelectionMask, ByteStream, FocusedWorkBasis);
    } else {
        FilterByMask(P, WorkSelectionMask, BasisBits, FocusedWorkBasis);
    }

    if (UseLayers) {
        P.InsertPhaseBoundary();
    }

    StreamSet * const NonModifiedMask = P.CreateStreamSet(1, 1);
    Invert(P, WorkSelectionMask, NonModifiedMask);
    SHOW_STREAM(NonModifiedMask);

    FilterByMask(P, NonModifiedMask, ByteStream, NonModifiedBytes);
}

typedef void (*FocusStageFunctionType)(StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, StreamSetPtr &, const char * buffer, const size_t length);

FocusStageFunctionType focus_stage_pipeline(CPUDriver & driver) {
    const unsigned selected_work_elems = ByteFiltering ? 1 : 8;
    const unsigned selected_work_width = ByteFiltering ? 8 : 1;

    auto P = CreatePipeline(driver,
                            Output<streamset_t>("ByteStream", 1, 8, ReturnedBuffer(1)),
                            Output<streamset_t>("WorkSelectionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("FocusedWorkBasis", selected_work_elems, selected_work_width, ReturnedBuffer(1)),
                            Output<streamset_t>("NonModifiedBytes", 1, 8, ReturnedBuffer(1)),
                            Input<const char*>{"buffer"}, Input<size_t>{"length"}
                            );

    StreamSet * const ByteStream = P.getOutputStreamSet("ByteStream");
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    source_logic(P, ByteStream, BasisBits);

    StreamSet * const WorkSelectionMask = P.getOutputStreamSet("WorkSelectionMask");
    StreamSet * const FocusedWorkBasis = P.getOutputStreamSet("FocusedWorkBasis");
    StreamSet * const NonModifiedBytes = P.getOutputStreamSet("NonModifiedBytes");
    focus_stage_logic(P, ByteStream, BasisBits, WorkSelectionMask, FocusedWorkBasis, NonModifiedBytes);
    return P.compile();
}

void working_space_logic(PipelineBuilder & P, StreamSet * SelectedWorkBasis, StreamSet * WorkingExpansionMask, StreamSet * WorkingBasis) {

    StreamSet * BasisBits = nullptr;
    if (ByteFiltering) {
        BasisBits = P.CreateStreamSet(8, 1);
        P.CreateKernelCall<S2PKernel>(SelectedWorkBasis, BasisBits);
        SHOW_BIXNUM(BasisBits);
    } else {
        BasisBits = SelectedWorkBasis;
    }

    StreamSet * WorkingInsertionBixNum = P.CreateStreamSet(4, 1);
    P.CreateKernelCall<NFC_Initial_Insertion>(BasisBits, WorkingInsertionBixNum);
    SHOW_BIXNUM(WorkingInsertionBixNum);

    InsertionSpreadMask(P, WorkingInsertionBixNum, WorkingExpansionMask, kernel::InsertPosition::After);
    SHOW_STREAM(WorkingExpansionMask);

    SpreadByMask(P, WorkingExpansionMask, BasisBits, WorkingBasis);
}

typedef void (*WorkingSpaceFunctionType)(const StreamSetPtr &, StreamSetPtr &, StreamSetPtr &);

WorkingSpaceFunctionType working_space_pipeline(CPUDriver & driver) {
    const unsigned selected_work_elems = ByteFiltering ? 1 : 8;
    const unsigned selected_work_width = ByteFiltering ? 8 : 1;

    auto P = CreatePipeline(driver,
                            Input<streamset_t>("SelectedWorkBasis", selected_work_elems, selected_work_width),
                            Output<streamset_t>("WorkingExpansionMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("WorkingBasis", 8, 1, ReturnedBuffer(1))
                            );
    StreamSet * const SelectedWorkBasis = P.getInputStreamSet("SelectedWorkBasis");
    StreamSet * const WorkingExpansionMask = P.getOutputStreamSet("WorkingExpansionMask");
    StreamSet * const WorkingBasis = P.getOutputStreamSet("WorkingBasis");
    working_space_logic(P, SelectedWorkBasis, WorkingExpansionMask, WorkingBasis);
    return P.compile();
}

typedef void (*NFCFunctionType)(const StreamSetPtr &, const StreamSetPtr &, StreamSetPtr &, StreamSetPtr &);

NFCFunctionType NFC_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("WorkingExpansionMask", 1, 1),
                            Input<streamset_t>("WorkingBasis", 8, 1),
                            Output<streamset_t>("ValidWorkMask", 1, 1, ReturnedBuffer(1)),
                            Output<streamset_t>("TransformedBytes", 1, 8, ReturnedBuffer(1))
                            );
    StreamSet * const WorkingExpansionMask = P.getInputStreamSet("WorkingExpansionMask");
    StreamSet * const WorkingBasis = P.getInputStreamSet("WorkingBasis");
    StreamSet * const ValidWorkMask = P.getOutputStreamSet("ValidWorkMask");
    StreamSet * const TransformedBytes = P.getOutputStreamSet("TransformedBytes");

    NFC_U8_logic(P, WorkingExpansionMask, WorkingBasis, ValidWorkMask, TransformedBytes);
    return P.compile();
}

void final_stage_logic(PipelineBuilder & P, StreamSet * ByteStream, StreamSet * BasisBits, 
    StreamSet * WorkSelectionMask, StreamSet * ValidWorkMask, StreamSet * NonModifiedBytes,
    StreamSet * TransformedBytes) {

    if (BasisBits == nullptr) {
        BasisBits = P.CreateStreamSet(8, 1);
        P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    }

    StreamSet * FinalInsertionBixNum = P.CreateStreamSet(4, 1);
    P.CreateKernelCall<NFC_Initial_Insertion>(BasisBits, FinalInsertionBixNum, WorkSelectionMask);
    SHOW_BIXNUM(FinalInsertionBixNum);

    StreamSet * SourceExpansionMask = P.CreateStreamSet(1, 1);
    InsertionSpreadMask(P, FinalInsertionBixNum, SourceExpansionMask, kernel::InsertPosition::After);
    SHOW_STREAM(SourceExpansionMask);

    StreamSet * const ExpandedWorkMask = P.CreateStreamSet(1, 1);
    ExpandFilter(P, SourceExpansionMask, WorkSelectionMask, ExpandedWorkMask);
    SHOW_STREAM(ExpandedWorkMask);

    StreamSet * const FinalWorkSelectionMask = P.CreateStreamSet(1, 1);
    ExpandFilter(P, ExpandedWorkMask, ValidWorkMask, FinalWorkSelectionMask);
    SHOW_STREAM(FinalWorkSelectionMask);

    StreamSet * const FinalWorkPlacementMask = P.CreateStreamSet(1, 1);
    FilterByMask(P, FinalWorkSelectionMask, ExpandedWorkMask, FinalWorkPlacementMask);

    StreamSet * OutputBytes = P.CreateStreamSet(1, 8);
    MergeByMask(P, FinalWorkPlacementMask, TransformedBytes, NonModifiedBytes, OutputBytes);
    P.CreateKernelCall<StdOutKernel>(OutputBytes);
}


typedef void (*FinalStageFunctionType)(const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &);
//typedef void (*FinalStageFunctionType)(const StreamSetPtr &, const StreamSetPtr &, const StreamSetPtr &, const char * buffer, const size_t length);

FinalStageFunctionType final_stage_pipeline(CPUDriver & driver) {
    auto P = CreatePipeline(driver,
                            Input<streamset_t>("ByteStream", 1, 8),
                            Input<streamset_t>("WorkSelectionMask", 1, 1),
                            Input<streamset_t>("ValidWorkMask", 1, 1),
                            Input<streamset_t>("NonModifiedBytes", 1, 8),
                            Input<streamset_t>("TransformedBytes", 1, 8)
                            //Input<const char*>{"buffer"}, Input<size_t>{"length"}
                            );
    //Scalar * const buffer = P.getInputScalar("buffer");
    //Scalar * const length = P.getInputScalar("length");
    StreamSet * const ByteStream = P.getInputStreamSet("ByteStream");
    StreamSet * const WorkSelectionMask = P.getInputStreamSet("WorkSelectionMask");
    StreamSet * const ValidWorkMask = P.getInputStreamSet("ValidWorkMask");
    StreamSet * const NonModifiedBytes = P.getInputStreamSet("NonModifiedBytes");
    StreamSet * const TransformedBytes = P.getInputStreamSet("TransformedBytes");
    final_stage_logic(P, ByteStream, nullptr, WorkSelectionMask, ValidWorkMask, NonModifiedBytes, TransformedBytes);
    return P.compile();
}

typedef void (*XfrmFunctionType)(const char * buffer, const size_t length);

XfrmFunctionType generate_pipeline(CPUDriver & driver) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.

    auto P = CreatePipeline(driver, Input<const char*>{"buffer"}, Input<size_t>{"length"});

    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    source_logic(P, ByteStream, BasisBits);
    if (NoFocus) {
        StreamSet * WorkSelectionMask = P.CreateStreamSet(1, 1);
        DetermineNFC_WorkSpans(P, BasisBits, WorkSelectionMask);

        StreamSet * WorkingInsertionBixNum = P.CreateStreamSet(4, 1);
        P.CreateKernelCall<NFC_Initial_Insertion>(BasisBits, WorkingInsertionBixNum, WorkSelectionMask);
        SHOW_BIXNUM(WorkingInsertionBixNum);

        StreamSet * SourceExpansionMask = P.CreateStreamSet(1, 1);
        InsertionSpreadMask(P, WorkingInsertionBixNum, SourceExpansionMask, kernel::InsertPosition::After);
        SHOW_STREAM(SourceExpansionMask);

        StreamSet * ExpandedBasis = P.CreateStreamSet(8, 1);
        SpreadByMask(P, SourceExpansionMask, BasisBits, ExpandedBasis);

        StreamSet * ValidWorkMask = P.CreateStreamSet(1, 1);
        StreamSet * OutputBytes = P.CreateStreamSet(1, 8);
        NFC_U8_logic(P, SourceExpansionMask, ExpandedBasis, ValidWorkMask, OutputBytes);
        P.CreateKernelCall<StdOutKernel>(OutputBytes);

    } else {
        StreamSet * WorkSelectionMask = P.CreateStreamSet(1, 1);
        StreamSet * NonModifiedBytes = P.CreateStreamSet(1, 8);
        StreamSet * SelectedWorkBasis = nullptr;
        if (ByteFiltering) {
            SelectedWorkBasis = P.CreateStreamSet(1, 8);
        } else {
            SelectedWorkBasis = P.CreateStreamSet(8, 1);
        }
        focus_stage_logic(P, ByteStream, BasisBits, WorkSelectionMask, SelectedWorkBasis, NonModifiedBytes);

        StreamSet * WorkingExpansionMask = P.CreateStreamSet(1, 1);
        StreamSet * WorkingBasis = P.CreateStreamSet(8, 1);
        working_space_logic(P, SelectedWorkBasis, WorkingExpansionMask, WorkingBasis);

        StreamSet * ValidWorkMask = P.CreateStreamSet(1, 1);
        StreamSet * TransformedBytes = P.CreateStreamSet(1, 8);
        NFC_U8_logic(P, WorkingExpansionMask, WorkingBasis, ValidWorkMask, TransformedBytes);
        SHOW_STREAM(ValidWorkMask);

        final_stage_logic(P, ByteStream, BasisBits, WorkSelectionMask, ValidWorkMask, NonModifiedBytes, TransformedBytes);
    }

    return P.compile();
}

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&NFC_Options, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});

    bool useMMap = PreferMMap && canMMap(inputFile);
    AlignedFileBuffer buf;
    buf.load(inputFile, useMMap);
    size_t bytes_read = buf.getBufSize();
    if (bytes_read <= 0) {
        llvm::errs() << "Error: unable to read " << inputFile << " for processing.\n";
        exit(-1);
    }

    CPUDriver driver("NFC_function");
    if (MultiStage) {
        FocusStageFunctionType focus_stage = focus_stage_pipeline(driver);
        WorkingSpaceFunctionType working_space_stage = working_space_pipeline(driver);
        NFCFunctionType NFC_stage = NFC_pipeline(driver);
        FinalStageFunctionType final_stage = final_stage_pipeline(driver);
        //
        kernel::StreamSetPtr ByteStream;
        kernel::StreamSetPtr WorkSelectionMask;
        kernel::StreamSetPtr SelectedWorkBasis;
        kernel::StreamSetPtr NonModifiedBytes;
        kernel::StreamSetPtr WorkingExpansionMask;
        kernel::StreamSetPtr WorkingBasis;
        kernel::StreamSetPtr ValidWork;
        kernel::StreamSetPtr TransformedBytes;
        focus_stage(ByteStream, WorkSelectionMask, SelectedWorkBasis, NonModifiedBytes, buf.getBuf(), buf.getBufSize());
        working_space_stage(SelectedWorkBasis, WorkingExpansionMask, WorkingBasis);
        NFC_stage(WorkingExpansionMask, WorkingBasis, ValidWork, TransformedBytes);
        final_stage(ByteStream, WorkSelectionMask, ValidWork, NonModifiedBytes, TransformedBytes);
    } else {
        XfrmFunctionType fn;
        fn = generate_pipeline(driver);
        //
        fn(buf.getBuf(), buf.getBufSize());
    }
    buf.release();
    return 0;
}
