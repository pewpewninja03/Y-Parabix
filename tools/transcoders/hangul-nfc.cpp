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
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/core/unicode_set.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decomposition Options", "Decomposition Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

const UCD::codepoint_t Hangul_SBase = 0xAC00;
const UCD::codepoint_t Hangul_LBase = 0x1100;
const UCD::codepoint_t Hangul_VBase = 0x1161;
const UCD::codepoint_t Hangul_TBase = 0x11A7;
const unsigned Hangul_LCount = 19;
const unsigned Hangul_VCount = 21;
const unsigned Hangul_TCount = 28;
const unsigned Hangul_NCount = 588;
const unsigned Hangul_SCount = 11172;

const unsigned S_Index_bits = 14;
const unsigned L_Index_bits = 5;
const unsigned V_Index_bits = 5;
const unsigned T_Index_bits = 5;

std::vector<re::CC *> Hangul_Decomposed_CCs() {
    UCD::UnicodeSet Hangul_L(Hangul_LBase, Hangul_LBase + Hangul_LCount - 1);
    UCD::UnicodeSet Hangul_V(Hangul_LBase, Hangul_VBase + Hangul_VCount - 1);
    UCD::UnicodeSet Hangul_T(Hangul_LBase, Hangul_TBase + Hangul_TCount - 1);
    return {re::makeCC(Hangul_L, &cc::Unicode),
            re::makeCC(Hangul_V, &cc::Unicode),
            re::makeCC(Hangul_T, &cc::Unicode)};
}

class Hangul_Precomposition : public pablo::PabloKernel {
public:
    Hangul_Precomposition(LLVMTypeSystemInterface & ts,
                          StreamSet * Basis, StreamSet * L_V_T_Decomposed, 
                          StreamSet * Output_Basis, StreamSet * SelectionMask);
protected:
    void generatePabloMethod() override;
};

Hangul_Precomposition::Hangul_Precomposition (LLVMTypeSystemInterface & ts,
                                              StreamSet * Basis, StreamSet * L_V_T_Decomposed,
                                              StreamSet * Output_Basis, StreamSet * SelectionMask)
: PabloKernel(ts, "Hangul_Precomposition" + Basis->shapeString(),
// inputs
    {Binding{"Basis", Basis, FixedRate(1), LookAhead(8)}, Binding{"L_V_T_Decomposed", L_V_T_Decomposed, FixedRate(1), LookAhead(8)}},
// output
    {Binding{"Output_Basis", Output_Basis}, Binding{"SelectionMask", SelectionMask}}) {
}

void Hangul_Precomposition::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> Basis = getInputStreamSet("Basis");
    // For UTF-8 each L, V, T are 3 code units in length.
    // For Unicode or UTF-16, only a single code unit is needed.
    unsigned codeUnitsPerChar = Basis.size() > 8 ? 1 : 3;
    std::vector<PabloAST *> L_V_T_Decomposed = getInputStreamSet("L_V_T_Decomposed");
    PabloAST * Hangul_L = L_V_T_Decomposed[0];
    PabloAST * Hangul_V = L_V_T_Decomposed[1];
    PabloAST * Hangul_T = L_V_T_Decomposed[2];
    //
    //  Set up variables to receive the output basis bit streams.
    std::vector<Var *> outputVar(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        outputVar[i] = pb.createVar("outputVar" + std::to_string(i), Basis[i]);
    }
    Var * maskVar = pb.createVar("maskVar", pb.createOnes());
    //
    // All calculations depend on having an initial L, if there are none,
    // we can skip.
    auto nested = pb.createScope();
    
    pb.createIf(Hangul_L, nested);
    BixNumCompiler bnc(nested);
    //
    // Only the 5 low bits of the basis are needed for index calculations.
    const unsigned index_bits = 5;
    BixNum IndexBasis = bnc.Truncate(Basis, index_bits);
    // All index calculations will be performed at the first character position
    BixNum IndexAhead1(index_bits);
    BixNum IndexAhead2(index_bits);
    for (unsigned i = 0; i < index_bits; i++) {
        IndexAhead1[i] = nested.createLookahead(IndexBasis[i], codeUnitsPerChar * 1);
        IndexAhead2[i] = nested.createLookahead(IndexBasis[i], codeUnitsPerChar * 2);
    }
    PabloAST * V_ahead = nested.createLookahead(Hangul_V, codeUnitsPerChar * 1);
    PabloAST * T_ahead2 = nested.createLookahead(Hangul_T, codeUnitsPerChar * 2);
    PabloAST * LV_combo = nested.createAnd(Hangul_L, V_ahead);
    PabloAST * LVT_sequence = nested.createAnd(LV_combo, T_ahead2);
    PabloAST * V_suffix = nested.createAdvance(LV_combo, codeUnitsPerChar * 1);
    PabloAST * T_suffix = nested.createAdvance(LVT_sequence, codeUnitsPerChar * 2);
    PabloAST * VT_suffixes = nested.createOr(V_suffix, T_suffix);
    nested.createAssign(maskVar, nested.createInFile(nested.createNot(VT_suffixes)));
    BixNum ZeroIndex(index_bits, nested.createZeroes());
    BixNum L_index = bnc.Select(LV_combo, IndexBasis, ZeroIndex);
    unsigned indexMask = (1 << index_bits) - 1;  // mask to select low bits of index only
    BixNum V_index = bnc.Select(LV_combo, bnc.SubModular(IndexAhead1, Hangul_VBase & indexMask), ZeroIndex);
    BixNum T_index = bnc.Select(LVT_sequence, bnc.SubModular(IndexAhead2, Hangul_TBase & indexMask), ZeroIndex);
    BixNum LV_index = bnc.AddFull(bnc.MulFull(L_index, Hangul_VCount), V_index);
    BixNum S_Index = bnc.AddFull(bnc.MulFull(LV_index, Hangul_TCount), T_index);
    BixNum Precomposed = bnc.Select(LV_combo, bnc.ZeroExtend(bnc.AddFull(S_Index, Hangul_SBase), 21), Basis);
    for (unsigned i = 0; i < Basis.size(); i++) {
        nested.createAssign(outputVar[i], Precomposed[i]);
    }
    writeOutputStreamSet("Output_Basis", outputVar);
    writeOutputStreamSet("SelectionMask", std::vector<Var *>{maskVar});
}

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

    //  The u8index stream marks the final byte of each UTF-8 character sequence.
    StreamSet * u8index = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    //  To make Unicode calculations simpler, we will construct a stream set of
    //  all 21 Unicode bits.   As a first step we calculate these bits at
    //  the u8index positions.
    StreamSet * U21_u8indexed = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);

    //  Now we construct a compressed stream set which is one-to-one with
    //  Unicode characters, by filtering out non u8index positions.
    StreamSet * U21 = P.CreateStreamSet(21, 1);
    FilterByMask(P, u8index, U21_u8indexed, U21);
    SHOW_BIXNUM(U21);

    //  Hangul decomposed characters are of three types, L, V and T.
    //  Compute a set of 3 parallel bit streams, one for each character class.
    auto L_V_T_ccs = Hangul_Decomposed_CCs();
    StreamSet * L_V_T =  P.CreateStreamSet(L_V_T_ccs.size());
    P.CreateKernelCall<CharClassesKernel>(L_V_T_ccs, U21, L_V_T);
    SHOW_BIXNUM(L_V_T);
    
    StreamSet * TranslatedBasis = P.CreateStreamSet(21, 1);
    StreamSet * SelectionMask = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<Hangul_Precomposition>(U21, L_V_T, TranslatedBasis, SelectionMask);
    SHOW_BIXNUM(TranslatedBasis);
    SHOW_STREAM(SelectionMask);

    StreamSet * NFC_Basis = P.CreateStreamSet(21, 1);
    FilterByMask(P, SelectionMask, TranslatedBasis, NFC_Basis);
    SHOW_BIXNUM(NFC_Basis);

    // Given the 21-bit basis representation, we can now transform back
    // to a UTF-8 representation.
    StreamSet * const OutputBasis = P.CreateStreamSet(8);
    U21_to_UTF8(P, NFC_Basis, OutputBasis);

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
