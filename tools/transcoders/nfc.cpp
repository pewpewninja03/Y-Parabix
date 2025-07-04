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
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/normalization.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/utf8_support.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/core/unicode_set.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
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
static cl::opt<bool> U21("U21", cl::desc("perform character translation via 21-bit Unicode"),  cl::cat(NFC_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

class NFC_Focus : public pablo::PabloKernel {
public:
    NFC_Focus(LLVMTypeSystemInterface & ts,
              StreamSet * Basis, StreamSet * ccc_NR, StreamSet * Composable2nds, StreamSet * Focus);
protected:
    void generatePabloMethod() override;
};

NFC_Focus::NFC_Focus(LLVMTypeSystemInterface & ts, 
                     StreamSet * Basis, StreamSet * ccc_NR, StreamSet * NFC_candidates,
                     StreamSet * Focus)
: PabloKernel(ts, "NFC_Focus",
// inputs
{Binding{"Basis", Basis},
 Binding{"ccc_NR", ccc_NR},
 Binding{"NFC_candidates", NFC_candidates, FixedRate(), LookAhead(4)}},
// output
{Binding{"Focus", Focus}}) {
}

void NFC_Focus::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    PabloAST * ccc_NR = getInputStreamSet("ccc_NR")[0];
    std::vector<PabloAST *> NFC_candidates = getInputStreamSet("NFC_candidates");
    PabloAST * composable_seconds = NFC_candidates[0];
    PabloAST * excluded_composites = NFC_candidates[1];
    BixNumCompiler bnc(pb);
    PabloAST * pfx4 = bnc.UGE(Basis, 0xF0);
    PabloAST * pfx3or4 = bnc.UGE(Basis, 0xE0);
    PabloAST * pfx = bnc.UGE(Basis, 0xC2);
    PabloAST * pfx3 = pb.createXor(pfx3or4, pfx4);
    PabloAST * pfx2 = pb.createXor(pfx, pfx3or4);
    PabloAST * thirdlast = pb.createOr(pfx3, pb.createAdvance(pfx4, 1));
    PabloAST * secondlast = pb.createOr(pfx2, pb.createAdvance(thirdlast, 1));
    PabloAST * focus = pb.createLookahead(composable_seconds, 1);
    focus = pb.createOr(focus, pb.createAnd(secondlast, pb.createLookahead(composable_seconds, 2)));
    focus = pb.createOr(focus, pb.createAnd(thirdlast, pb.createLookahead(composable_seconds, 3)));
    focus = pb.createOr(focus, pb.createAnd(pfx4, pb.createLookahead(composable_seconds, 4)));
    // Now find the next starter from each candidate run.
    focus = pb.createMatchStar(focus, pb.createNot(ccc_NR));
    focus = pb.createOr(focus, excluded_composites);
    focus = pb.createOr(focus, pb.createAdvance(pb.createAnd(pfx, excluded_composites), 1));
    focus = pb.createOr(focus, pb.createAdvance(pb.createAnd(pfx3or4, excluded_composites), 2));
    focus = pb.createOr(focus, pb.createAdvance(pb.createAnd(pfx4, excluded_composites), 3));
    pb.createAssign(pb.createExtract(getOutputStreamVar("Focus"), pb.getInteger(0)), focus);
}

class DelPriorToSelectMask : public PabloKernel {
public:
    DelPriorToSelectMask
        (LLVMTypeSystemInterface & ts,
         StreamSet * Basis, StreamSet * InputMask, StreamSet * MarkDeletion, StreamSet * DeletePrior,
         StreamSet * SelectMask);
protected:
    void generatePabloMethod() override;
};

DelPriorToSelectMask::DelPriorToSelectMask
    (LLVMTypeSystemInterface & ts,
     StreamSet * Basis, StreamSet * InputMask, StreamSet * MarkDeletion, StreamSet * DeletePrior,
     StreamSet * SelectMask)
: PabloKernel(ts, "DelPriorToSelectMask",
{Binding{"Basis", Basis}, Binding{"InputMask", InputMask}, Binding{"MarkDeletion", MarkDeletion}, Binding{"DeletePrior", DeletePrior, FixedRate(), LookAhead(4)}},
{Binding{"SelectMask", SelectMask}}) {}

void DelPriorToSelectMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    PabloAST * InputMask = getInputStreamSet("InputMask")[0];
    PabloAST * MarkDeletion = getInputStreamSet("MarkDeletion")[0];
    PabloAST * DelPrior = getInputStreamSet("DeletePrior")[0];
    BixNumCompiler bnc(pb);
    PabloAST * nullDeletion = bnc.EQ(Basis, 0);
    // TODO: keep Nulls in original basis
    PabloAST * pfx4 = bnc.UGE(Basis, 0xF0);
    PabloAST * pfx3or4 = bnc.UGE(Basis, 0xE0);
    PabloAST * pfx = bnc.UGE(Basis, 0xC2);
    PabloAST * pfx3 = pb.createXor(pfx3or4, pfx4);
    PabloAST * pfx2 = pb.createXor(pfx, pfx3or4);
    PabloAST * delpfx4 = pb.createAnd(pfx4, pb.createLookahead(DelPrior, 4));
    PabloAST * thirdlast = pb.createOr(pfx3, pb.createAdvance(pfx4, 1));
    PabloAST * del3 = pb.createAnd(thirdlast, pb.createLookahead(DelPrior, 3));
    PabloAST * secondlast = pb.createOr(pfx2, pb.createAdvance(thirdlast, 1));
    PabloAST * del2 = pb.createAnd(secondlast, pb.createLookahead(DelPrior, 2));
    PabloAST * del1 = pb.createLookahead(DelPrior, 1);
    PabloAST * del = pb.createOr(pb.createOr(delpfx4, del3), pb.createOr(del2, del1));
    del = pb.createOr3(nullDeletion, MarkDeletion, del);
    std::vector<PabloAST *> select_streamset = {pb.createAnd(InputMask, pb.createInFile(pb.createNot(del)))};
    writeOutputStreamSet("SelectMask", select_streamset);
}
using namespace UCD;

void NFC_U8_Pipeline(PipelineBuilder & P, re::Name * CCC0_Name, StreamSet * U8_Basis, StreamSet * OutputBasis, StreamSet * FinalSelectionMask) {
    StreamSet * InsertionBixNum = P.CreateStreamSet(4, 1);
    P.CreateKernelCall<NFC_Initial_Insertion>(U8_Basis, InsertionBixNum);
    SHOW_BIXNUM(InsertionBixNum);

    StreamSet * SpreadMask = InsertionSpreadMask(P, InsertionBixNum, kernel::InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(8, 1);
    SpreadByMask(P, SpreadMask, U8_Basis, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    StreamSet * EC_Basis = P.CreateStreamSet(8, 1);
    StreamSet * EC_SelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<ExcludedCompositeStage>(ExpandedBasis, EC_SelectionMask, EC_Basis);
    SHOW_BIXNUM(EC_Basis);

    StreamSet * const ccc_NR = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Name, EC_Basis, ccc_NR, BitMovementMode::LookAhead);
    SHOW_STREAM(ccc_NR);

    StreamSet * CanonBasis = P.CreateStreamSet(8, 1);
    StreamSet * CanonSelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<SingletonCanonicalization>(EC_Basis, CanonSelectionMask, CanonBasis);
    SHOW_BIXNUM(CanonBasis);
    SHOW_STREAM(CanonSelectionMask);

    StreamSet * XfrmBasis = P.CreateStreamSet(8, 1);
    StreamSet * DelPrior = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<ShortComposableTranslation>(CanonBasis, DelPrior, XfrmBasis);
    SHOW_BIXNUM(XfrmBasis);
    SHOW_STREAM(DelPrior);

    StreamSet * XfrmedBasis = P.CreateStreamSet(8, 1);
    XorCombine(P, CanonBasis, XfrmBasis, XfrmedBasis);
    SHOW_BIXNUM(XfrmedBasis);

    StreamSet * FinalBasis = P.CreateStreamSet(8, 1);
    StreamSet * MarkDeletion = P.CreateStreamSet(1, 1);
    LongComposablePipeline(P, XfrmedBasis, ccc_NR, FinalBasis, MarkDeletion);
    SHOW_STREAM(MarkDeletion);

    StreamSet * SelectionMask0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<DelPriorToSelectMask>(FinalBasis, CanonSelectionMask, MarkDeletion, DelPrior, SelectionMask0);
    SHOW_STREAM(SelectionMask0);

    StreamSet * L_V_T =  P.CreateStreamSet(Hangul_Composables::HC_Kind::Count);
    P.CreateKernelCall<Hangul_Composables>(FinalBasis, L_V_T, pablo::BitMovementMode::LookAhead);
    //P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, FilteredBasis, L_V_T, pablo::BitMovementMode::LookAhead);
    SHOW_BIXNUM(L_V_T);

    StreamSet * TranslatedBasis = P.CreateStreamSet(8, 1);
    StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Hangul_Composition>(FinalBasis, L_V_T, TranslatedBasis, SelectionMask);
    SHOW_BIXNUM(TranslatedBasis);
    SHOW_STREAM(SelectionMask);

    AndCombine(P, SelectionMask0, SelectionMask, FinalSelectionMask);
    SHOW_STREAM(FinalSelectionMask);

    FilterByMask(P, FinalSelectionMask, TranslatedBasis, OutputBasis);
    SHOW_BIXNUM(OutputBasis);
}

typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_pipeline(CPUDriver & driver) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.

    re::RE * CCC0_Prop = re::makePropertyExpression("CCC", "NR");
    CCC0_Prop = UCD::linkAndResolve(CCC0_Prop);
    re::Name * CCC0_Name = re::makeName("CCC", "NR");
    CCC0_Name->setDefinition(CCC0_Prop);

    auto P = CreatePipeline(driver, Input<uint32_t>("inputFileDecriptor"));

    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");
    StreamSet * ByteStream = P.CreateStreamSet(1, 8);
    //  ReadSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, ByteStream);
    SHOW_BYTES(ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * const ccc_NR0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UnicodePropertyKernelBuilder>(CCC0_Name, BasisBits, ccc_NR0, BitMovementMode::LookAhead);
    SHOW_STREAM(ccc_NR0);

    StreamSet * NFC_Candidates = P.CreateStreamSet(2, 1);
    P.CreateKernelCall<NFC_CandidateClass>(BasisBits, NFC_Candidates);
    SHOW_BIXNUM(NFC_Candidates);

    StreamSet * NFC_WorkItems = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<NFC_Focus>(BasisBits, ccc_NR0, NFC_Candidates, NFC_WorkItems);
    SHOW_STREAM(NFC_WorkItems);

    StreamSet * const OutputBasis = P.CreateStreamSet(8);
    StreamSet * FinalSelectionMask = P.CreateStreamSet(1, 1);
    NFC_U8_Pipeline(P, CCC0_Name, BasisBits, OutputBasis, FinalSelectionMask);

    //  The P2SKernel transforms the basis bit streams into the corresponding
    //  byte stream, inverting the S2P process.
    StreamSet * OutputBytes = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);

    // The program output is generated by writing the output bytes to stdout.
    P.CreateKernelCall<StdOutKernel>(OutputBytes);

    return P.compile();
}


int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&NFC_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    CPUDriver driver("NFC_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    XfrmFunctionType fn;
    fn = generate_pipeline(driver);
    //
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
    } else {
        fn(fd);
        close(fd);
    }
    return 0;
}
