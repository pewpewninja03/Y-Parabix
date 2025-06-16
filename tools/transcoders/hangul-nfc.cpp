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
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/normalization.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/core/unicode_set.h>
#include <unicode/algo/normalization.h>
#include <unicode/utf/utf_compiler.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decomposition Options", "Decomposition Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));
static cl::opt<bool> U21("U21", cl::desc("perform character translation via 21-bit Unicode"),  cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)


class ApplyTransform : public pablo::PabloKernel {
public:
    ApplyTransform(LLVMTypeSystemInterface & ts,
                   StreamSet * Basis, StreamSet * Xfrms, StreamSet * Output);
protected:
    void generatePabloMethod() override;
};

ApplyTransform::ApplyTransform (LLVMTypeSystemInterface & ts,
                                StreamSet * Basis, StreamSet * Xfrms, StreamSet * Output)
: PabloKernel(ts, "xfrm_" + std::to_string(Basis->getNumElements()) + "x1_" + std::to_string(Xfrms->getNumElements()),
// inputs
{Binding{"basis", Basis},
 Binding{"xfrms", Xfrms}
},
// output
{Binding{"output_basis", Output}}) {
}

void ApplyTransform::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> xfrms = getInputStreamSet("xfrms");
    std::vector<PabloAST *> transformed(basis.size());
    for (unsigned i = 0; i < basis.size(); i++) {
        if (i < xfrms.size()) {
            transformed[i] = pb.createXor(xfrms[i], basis[i]);
        } else {
            transformed[i] = basis[i];
        }
    }
    writeOutputStreamSet("output_basis", transformed);
}


class DelPriorToSelectMask : public PabloKernel {
public:
    DelPriorToSelectMask
        (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * InputMask, StreamSet * DeletePrior,
                                       StreamSet * SelectMask);
protected:
    void generatePabloMethod() override;
};

DelPriorToSelectMask::DelPriorToSelectMask
    (LLVMTypeSystemInterface & ts, StreamSet * Basis, StreamSet * InputMask, StreamSet * DeletePrior,
                                   StreamSet * SelectMask)
: PabloKernel(ts, "DelPriorToSelectMask",
{Binding{"Basis", Basis}, Binding{"InputMask", InputMask}, Binding{"DeletePrior", DeletePrior, FixedRate(), LookAhead(4)}},
{Binding{"SelectMask", SelectMask}}) {}

void DelPriorToSelectMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    PabloAST * InputMask = getInputStreamSet("InputMask")[0];
    PabloAST * DelPrior = getInputStreamSet("DeletePrior")[0];
    BixNumCompiler bnc(pb);
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
    std::vector<PabloAST *> select_streamset = {pb.createAnd(InputMask, pb.createInFile(pb.createNot(del)))};
    writeOutputStreamSet("SelectMask", select_streamset);
}
using namespace UCD;


typedef void (*XfrmFunctionType)(uint32_t fd);

XfrmFunctionType generate_pipeline(CPUDriver & driver) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.

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

    StreamSet * InsertionBixNum = P.CreateStreamSet(2, 1);
    P.CreateKernelCall<NFC_Initial_Insertion>(BasisBits, InsertionBixNum);
    SHOW_BIXNUM(InsertionBixNum);

    StreamSet * SpreadMask = InsertionSpreadMask(P, InsertionBixNum, kernel::InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    StreamSet * ExpandedBasis = P.CreateStreamSet(8, 1);
    SpreadByMask(P, SpreadMask, BasisBits, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    StreamSet * NSD_Basis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<NonStarterDecomposition>(ExpandedBasis, NSD_Basis);
    SHOW_BIXNUM(NSD_Basis);

    StreamSet * Canon_Basis = P.CreateStreamSet(8, 1);
    StreamSet * CanonSelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<SingletonCanonicalization>(NSD_Basis, CanonSelectionMask, Canon_Basis);
    SHOW_BIXNUM(Canon_Basis);
    SHOW_STREAM(CanonSelectionMask);

    StreamSet * XfrmBasis = P.CreateStreamSet(8, 1);
    StreamSet * DelPrior = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<ShortComposableTranslation>(Canon_Basis, DelPrior, XfrmBasis);
    SHOW_BIXNUM(XfrmBasis);
    SHOW_STREAM(DelPrior);

    StreamSet * XfrmedBasis = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<ApplyTransform>(Canon_Basis, XfrmBasis, XfrmedBasis);
    SHOW_BIXNUM(XfrmedBasis);

    StreamSet * SelectionMask0 = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<DelPriorToSelectMask>(ExpandedBasis, CanonSelectionMask, DelPrior, SelectionMask0);
    SHOW_STREAM(SelectionMask0);

    StreamSet * FilteredBasis = P.CreateStreamSet(8, 1);
    FilterByMask(P, SelectionMask0, XfrmedBasis, FilteredBasis);
    SHOW_BIXNUM(FilteredBasis);

    StreamSet * const OutputBasis = P.CreateStreamSet(8);

    if (U21) {
        //  The u8index stream marks the final byte of each UTF-8 character sequence.
        StreamSet * u8index = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<UTF8_index>(FilteredBasis, u8index);
        SHOW_STREAM(u8index);

        //  To make Unicode calculations simpler, we will construct a stream set of
        //  all 21 Unicode bits.   As a first step we calculate these bits at
        //  the u8index positions.
        StreamSet * U21_u8indexed = P.CreateStreamSet(21, 1);
        P.CreateKernelCall<UTF8_Decoder>(FilteredBasis, U21_u8indexed);

        //  Now we construct a compressed stream set which is one-to-one with
        //  Unicode characters, by filtering out non u8index positions.
        StreamSet * U21 = P.CreateStreamSet(21, 1);
        FilterByMask(P, u8index, U21_u8indexed, U21);
        SHOW_BIXNUM(U21);

        //  Hangul composable characters are of three types, L, V, LV and T.
        //  Compute a set of 4 parallel bit streams, one for each character class.
        //auto L_V_T_sets = Hangul_Composables();
        StreamSet * L_V_T =  P.CreateStreamSet(Hangul_Composables::Kind::Count);
        P.CreateKernelCall<Hangul_Composables>(U21, L_V_T);
        //P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, U21, L_V_T);
        SHOW_BIXNUM(L_V_T);

        StreamSet * TranslatedBasis = P.CreateStreamSet(21, 1);
        StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<Hangul_Composition>(U21, L_V_T, TranslatedBasis, SelectionMask);
        SHOW_BIXNUM(TranslatedBasis);
        SHOW_STREAM(SelectionMask);

        StreamSet * NFC_Basis = P.CreateStreamSet(21, 1);
        FilterByMask(P, SelectionMask, TranslatedBasis, NFC_Basis);
        SHOW_BIXNUM(NFC_Basis);

        // Given the 21-bit basis representation, we can now transform back
        // to a UTF-8 representation.
        U21_to_UTF8(P, NFC_Basis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);
    } else {
        
        //auto L_V_T_sets = Hangul_Composables();
        StreamSet * L_V_T =  P.CreateStreamSet(Hangul_Composables::Kind::Count);
        P.CreateKernelCall<Hangul_Composables>(FilteredBasis, L_V_T, pablo::BitMovementMode::LookAhead);
        //P.CreateKernelCall<CharClassesKernel>(L_V_T_sets, FilteredBasis, L_V_T, pablo::BitMovementMode::LookAhead);
        SHOW_BIXNUM(L_V_T);

        StreamSet * TranslatedBasis = P.CreateStreamSet(8, 1);
        StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
        P.CreateKernelCall<Hangul_Composition>(FilteredBasis, L_V_T, TranslatedBasis, SelectionMask);
        SHOW_BIXNUM(TranslatedBasis);
        SHOW_STREAM(SelectionMask);

        FilterByMask(P, SelectionMask, TranslatedBasis, OutputBasis);
        SHOW_BIXNUM(OutputBasis);
    }
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
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    CPUDriver driver("NFD_function");
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
