/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>               // for StdOutKernel_
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <llvm/ADT/StringRef.h>
#include <kernel/pipeline/program_builder.h>
#include <fcntl.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::floor_log2;
#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory PackDemoOptions("Pack Demo Options", "Pack demo options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(PackDemoOptions));

enum class PackOption {packh, packl};
class PackKernel final : public MultiBlockKernel {
public:
    PackKernel(LLVMTypeSystemInterface & ts,
              StreamSet * const i16Stream,
              StreamSet * const i8Stream,
              PackOption opt);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
private:
    PackOption mOption;
};

std::string packOptionString(PackOption opt) {
    if (opt == PackOption::packh) return "packh";
    return "packl";
}

PackKernel::PackKernel(LLVMTypeSystemInterface & ts, StreamSet * const i16Stream, StreamSet * const i8Stream, PackOption opt)
: MultiBlockKernel(ts, packOptionString(opt) ,
{Binding{"i16Stream", i16Stream}},
    {Binding{"i8Stream", i8Stream}}, {}, {}, {}), mOption(opt)  {}


void PackKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const unsigned fw = 8;
    const unsigned inputPacksPerStride = 2*fw;
    const unsigned outputPacksPerStride = fw;

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const ZERO = b.getSize(0);
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
        llvm::errs() << "stride = " << getStride() << "\n";
    }
    b.CreateBr(packLoop);
    b.SetInsertPoint(packLoop);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);
    Value * i16pack[inputPacksPerStride];
    for (unsigned i = 0; i < inputPacksPerStride; i++) {
        i16pack[i] = b.loadInputStreamPack("i16Stream", ZERO, b.getInt32(i), blockOffsetPhi);
    }
    Value * packed[outputPacksPerStride];
    for (unsigned i = 0; i < outputPacksPerStride; i++) {
        if (mOption == PackOption::packh) {
            packed[i] = b.hsimd_packh(2*fw, i16pack[2*i], i16pack[2*i+1]);
        } else {
            packed[i] = b.hsimd_packl(2*fw, i16pack[2*i], i16pack[2*i+1]);
        }
        b.storeOutputStreamPack("i8Stream", ZERO, b.getInt32(i), blockOffsetPhi, packed[i]);
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, packLoop);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

    b.CreateCondBr(moreToDo, packLoop, packFinalize);
    b.SetInsertPoint(packFinalize);
}

typedef void (*PackDemoFunctionType)(uint32_t fd);

PackDemoFunctionType packdemo_gen (CPUDriver & driver) {

    auto P = CreatePipeline(driver, Input<uint32_t>{"inputFileDecriptor"});

    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");

    // Source data
    StreamSet * const i16Stream = P.CreateStreamSet(1, 16);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, i16Stream);

    StreamSet * const packedStreamL = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<PackKernel>(i16Stream, packedStreamL, PackOption::packl);
    SHOW_BYTES(packedStreamL);

    StreamSet * const BasisBitsL = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(packedStreamL, BasisBitsL);
    SHOW_BIXNUM(BasisBitsL);

    StreamSet * const packedStreamH = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<PackKernel>(i16Stream, packedStreamH, PackOption::packh);
    SHOW_BYTES(packedStreamH);

    StreamSet * const BasisBitsH = P.CreateStreamSet(8, 1);
    P.CreateKernelCall<S2PKernel>(packedStreamH, BasisBitsH);
    SHOW_BIXNUM(BasisBitsH);

    P.CreateKernelCall<StdOutKernel>(packedStreamH);

    return P.compile();
}



int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&PackDemoOptions, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    CPUDriver driver("packdemo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        PackDemoFunctionType func = nullptr;
        func = packdemo_gen(driver);
        func(fd);
        close(fd);
    }
    return 0;
}
