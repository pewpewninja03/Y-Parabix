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

std::vector<re::CC *> NFD_Hangul_LV_LVT_CCs() {
    UCD::codepoint_t Max_Hangul_Precomposed = Hangul_SBase + Hangul_SCount - 1;
    UCD::UnicodeSet Hangul_Precomposed(Hangul_SBase, Max_Hangul_Precomposed);
    UCD::UnicodeSet Hangul_Precomposed_LV;
    UCD::UnicodeSet Hangul_Precomposed_LVT;
    for (UCD::codepoint_t cp = Hangul_SBase; cp <= Max_Hangul_Precomposed; cp += Hangul_TCount) {
        Hangul_Precomposed_LV.insert(cp);
    }
    Hangul_Precomposed_LVT = Hangul_Precomposed - Hangul_Precomposed_LV;
    return {re::makeCC(Hangul_Precomposed_LV, &cc::Unicode),
            re::makeCC(Hangul_Precomposed_LVT, &cc::Unicode)};
}

class LVT_Indexes : public pablo::PabloKernel {
public:
    LVT_Indexes(LLVMTypeSystemInterface & ts,
                StreamSet * U21_Basis, StreamSet * LIndex, StreamSet * VIndex, StreamSet * TIndex);
protected:
    void generatePabloMethod() override;
};

LVT_Indexes::LVT_Indexes (LLVMTypeSystemInterface & ts,
                          StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index)
: PabloKernel(ts, "Hangul::LVT_Indexes21x1",
// inputs
{Binding{"basis", Basis}},
// output
    {Binding{"L_index", L_index}, Binding{"V_index", V_index}, Binding{"T_index", T_index}}) {
}

void LVT_Indexes::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    BixNumCompiler bnc0(pb);
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul_SBase), bnc0.ULT(basis, Hangul_SBase + Hangul_SCount));
    BixNum ZeroBasis(basis.size(), pb.createZeroes());

    std::vector<Var *> L_indexVar(L_Index_bits);
    for (unsigned i = 0; i < L_Index_bits; i++) {
        L_indexVar[i] = pb.createVar("L_var" + std::to_string(i), pb.createZeroes());
    }
    std::vector<Var *> V_indexVar(V_Index_bits);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_indexVar[i] = pb.createVar("V_var" + std::to_string(i), pb.createZeroes());
    }
    std::vector<Var *> T_indexVar(T_Index_bits);
    for (unsigned i = 0; i < T_Index_bits; i++) {
        T_indexVar[i] = pb.createVar("T_var" + std::to_string(i), pb.createZeroes());
    }

    auto nested = pb.createScope();
    pb.createIf(Hangul_precomposed, nested);
    BixNumCompiler bnc(nested);

    BixNum S_Index = bnc.Select(Hangul_precomposed, bnc.SubModular(basis, Hangul_SBase), ZeroBasis);
    S_Index = bnc.Truncate(S_Index, S_Index_bits);
    BixNum LV_index;
    BixNum T_index;
    bnc.Div(S_Index, Hangul_TCount, LV_index, T_index);
    LV_index = bnc.Truncate(LV_index, L_Index_bits + V_Index_bits);
    BixNum L_index;
    BixNum V_index;
    bnc.Div(LV_index, Hangul_VCount, L_index, V_index);
    
    for (unsigned i = 0; i < L_Index_bits; i++) {
        nested.createAssign(L_indexVar[i], L_index[i]);
    }
    for (unsigned i = 0; i < V_Index_bits; i++) {
        nested.createAssign(V_indexVar[i], V_index[i]);
    }
    for (unsigned i = 0; i < T_Index_bits; i++) {
        nested.createAssign(T_indexVar[i], T_index[i]);
    }
    writeOutputStreamSet("L_index", L_indexVar);
    writeOutputStreamSet("V_index", V_indexVar);
    writeOutputStreamSet("T_index", T_indexVar);
}

class LVT2NFD : public pablo::PabloKernel {
public:
    LVT2NFD(LLVMTypeSystemInterface & ts,
               StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index,
               StreamSet * NFD_Basis);
protected:
    void generatePabloMethod() override;
};

LVT2NFD::LVT2NFD (LLVMTypeSystemInterface & ts,
                        StreamSet * Basis, StreamSet * L_index, StreamSet * V_index, StreamSet * T_index,
                        StreamSet * NFD_Basis)
: PabloKernel(ts, "Hangul::LVT2NFD_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis},
    Binding{"L_index", L_index},
    Binding{"V_index", V_index},
    Binding{"T_index", T_index}},
// output
{Binding{"NFD_Basis", NFD_Basis}}) {
}

void LVT2NFD::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> L_index = getInputStreamSet("L_index");
    std::vector<PabloAST *> V_index = getInputStreamSet("V_index");
    std::vector<PabloAST *> T_index = getInputStreamSet("T_index");
    BixNumCompiler bnc0(pb);
    PabloAST * Hangul_precomposed = pb.createAnd(bnc0.UGE(basis, Hangul_SBase), bnc0.ULT(basis, Hangul_SBase + Hangul_SCount));
    // Set up Vars to receive the generated basis values.
    std::vector<Var *> basisVar(21);
    for (unsigned i = 0; i < 21; i++) {
        basisVar[i] = pb.createVar("basisVar" + std::to_string(i), basis[i]);
    }
    auto nested = pb.createScope();
    pb.createIf(Hangul_precomposed, nested);
    BixNumCompiler bnc(nested);
    BixNum LPart = bnc.ZeroExtend(L_index, 21);
    // The LPart will be encoded at the original precomposed position.
    LPart = bnc.AddModular(LPart, Hangul_LBase);
    // The V Part, when it exists, is one position after the opening L Part.
    PabloAST * V_position = nested.createAdvance(Hangul_precomposed, 1);
    for (unsigned i = 0; i < V_Index_bits; i++) {
        V_index[i] = nested.createAdvance(V_index[i], 1);
    }
    BixNum VPart = bnc.ZeroExtend(V_index, 21);
    VPart = bnc.AddModular(VPart, Hangul_VBase);
    // The T Part, if it exists is two positions after the opening
    // L Part.  A T Part only exists for LVT initials (T != 0).
    PabloAST * T_position = nested.createAdvance(Hangul_precomposed, 2);
    for (unsigned i = 0; i < T_Index_bits; i++) {
        T_index[i] = nested.createAdvance(T_index[i], 2);
    }
    T_position = nested.createAnd(T_position, bnc.NEQ(T_index, 0));
    BixNum TPart = bnc.ZeroExtend(T_index, 21);
    TPart = bnc.AddModular(TPart, Hangul_TBase);
    for (unsigned i = 0; i < 21; i++) {
        PabloAST * bit = nested.createSel(Hangul_precomposed, LPart[i], basis[i]);
        bit = nested.createSel(V_position, VPart[i], bit);
        bit = nested.createSel(T_position, TPart[i], bit);
        nested.createAssign(basisVar[i], bit);
    }
    writeOutputStreamSet("NFD_Basis", basisVar);
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

    //  Hangul precomposed characters are of two types, LV and LVT, which
    //  decompose respectively into <L, V> and <L, V, T> sequences.
    //  We determine a 2-bit bixnum having the value 1 at LV positions and
    //  2 at LVT positions, in order to identify the number of additional
    //  positions that are needed in creating the decomposed representations.
    auto LV_LVT_ccs = NFD_Hangul_LV_LVT_CCs();
    StreamSet * LV_LVT =  P.CreateStreamSet(2);
    P.CreateKernelCall<CharClassesKernel>(LV_LVT_ccs, U21, LV_LVT);
    SHOW_BIXNUM(LV_LVT);

    //  We now calculate a spreadmask that can be used to insert positions
    //  after each LV and LVT character.   This spreadmask will have an
    //  extra 0 bit inserted after each LV character and two extra 0 bits
    //  inserted after each LVT character.
    StreamSet * SpreadMask = InsertionSpreadMask(P, LV_LVT, kernel::InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    //  The SpreadByMask operation uses the calculated spreadmask to produce
    //  a final set of 21 basis bits, with zeroes where V and T positions will be
    //  inserted.
    StreamSet * ExpandedBasis = P.CreateStreamSet(21, 1);
    SpreadByMask(P, SpreadMask, U21, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    //  The Hangul decomposition algorithm calculates replacements for LV and
    //  LVT characters using calculations based on three 5-bit indexes for
    //  the L, V and T characters.
    StreamSet * LIndexBixNum = P.CreateStreamSet(L_Index_bits);
    StreamSet * VIndexBixNum = P.CreateStreamSet(V_Index_bits);
    StreamSet * TIndexBixNum = P.CreateStreamSet(T_Index_bits);
    P.CreateKernelCall<LVT_Indexes>(ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    // Given the L, V and T indexes, the replacements for LV and LVT characters
    // can be calculated to determine the correct 21-bit representations at
    // <L, V> and <L, V, T> positions.
    StreamSet * Hangul_NFD_Basis = P.CreateStreamSet(21, 1);
    P.CreateKernelCall<LVT2NFD>(ExpandedBasis, LIndexBixNum, VIndexBixNum, TIndexBixNum, Hangul_NFD_Basis);
    SHOW_BIXNUM(Hangul_NFD_Basis);

    // Given the 21-bit basis representation, we can now transform back
    // to a UTF-8 representation.
    StreamSet * const OutputBasis = P.CreateStreamSet(8);
    U21_to_UTF8(P, Hangul_NFD_Basis, OutputBasis);

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
