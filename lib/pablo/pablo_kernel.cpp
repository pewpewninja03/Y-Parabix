/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <pablo/pablo_kernel.h>
#include <pablo/codegenstate.h>
#include <pablo/pablo_compiler.h>
#include <pablo/pe_var.h>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/passes.h>

#include <pablo/pablo_toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <toolchain/toolchain.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorHandling.h>
#include <pablo/branch.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <llvm/Support/raw_ostream.h>

#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>
#include <llvm/Transforms/Scalar/DCE.h>
// #include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/MemorySSA.h>

#include <llvm/IR/PassManager.h>
#ifndef NDEBUG
#include <llvm/IR/Verifier.h>
#endif

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

using namespace kernel;
using namespace IDISA;
using namespace llvm;

namespace pablo {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief instantiateKernelCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<KernelCompiler> PabloKernel::instantiateKernelCompiler(KernelBuilder & /* b */) const {
    return std::make_unique<PabloCompiler>(const_cast<PabloKernel *>(this));
}

Var * PabloKernel::getInputStreamVar(const std::string & name) {
    const auto port = mPabloCompiler->getStreamPort(name);
    assert (port.Type == PortType::Input);
    return mInputs[port.Number];
}

std::vector<PabloAST *> PabloKernel::getInputStreamSet(const std::string & name) {
    const auto port = mPabloCompiler->getStreamPort(name);
    assert (port.Type == PortType::Input);
    const Binding & input = getInputStreamSetBinding(port.Number);
    const auto numOfStreams = IDISA::getNumOfStreams(input.getType());
    std::vector<PabloAST *> inputSet(numOfStreams);
    for (unsigned i = 0; i < numOfStreams; i++) {
        inputSet[i] = mEntryScope->createExtract(mInputs[port.Number], mEntryScope->getInteger(i));
    }
    return inputSet;
}

template void PabloKernel::writeOutputStreamSet(const std::string & name, std::vector<Var *> s);
template void PabloKernel::writeOutputStreamSet(const std::string & name, std::vector<PabloAST *> s);

template <class T> void PabloKernel::writeOutputStreamSet(const std::string & name, std::vector<T *> s) {
    const auto port = mPabloCompiler->getStreamPort(name);
    assert (port.Type == PortType::Output);
    Var * outputVar = mOutputs[port.Number];
    for (unsigned i = 0; i < s.size(); i++) {
        mEntryScope->createAssign(mEntryScope->createExtract(outputVar, mEntryScope->getInteger(i)), s[i]);
    }
}

Var * PabloKernel::getOutputStreamVar(const std::string & name) {
    const auto port = mPabloCompiler->getStreamPort(name);
    assert (port.Type == PortType::Output);
    return mOutputs[port.Number];
}

Var * PabloKernel::getOutputScalarVar(const std::string & name) {
    for (Var * out : mScalarOutputVars) {
        if (out->getName().compare(name) == 0) {
            return out;
        }
    }
    report_fatal_error("Kernel does not contain scalar " + StringRef(name));
}

Var * PabloKernel::makeVariable(const String * const name, Type * const type) {
    for (Var * const var : mVariables) {
        if (var->getClassTypeId() == ClassTypeId::Var) {
            if (LLVM_UNLIKELY(&var->getName() == name)) {
                return var;
            }
        }
    }
    Var * const var = new (mAllocator) Var(type, name, mAllocator);
    mVariables.push_back(var);
    return var;
}

Extract * PabloKernel::makeExtract(Var * const array, PabloAST * const index) {
    for (Var * const var : mVariables) {
        if (var->getClassTypeId() == ClassTypeId::Extract) {
            Extract * const ext = cast<Extract>(var);
            if (ext->getArray() == array && ext->getIndex() == index) {
                return ext;
            }
        }
    }
    Type * type = array->getType(); assert (type);
    if (LLVM_LIKELY(isa<ArrayType>(type))) {
        type = cast<ArrayType>(type)->getArrayElementType();
    } else {
        std::string tmp;
        raw_string_ostream out(tmp);
        out << "cannot extract element from ";
        array->print(out);
        out << ": ";
        type->print(out);
        out << " is not an array type";
        throw std::runtime_error(out.str());
    }
    Extract * const ext = new (mAllocator) Extract(type, array, index, mAllocator);
    for (auto const & user : array->users()) {
        if (isa<PabloKernel>(user)) {
            ext->addUser(this);
            break;
        }
    }
    mVariables.push_back(ext);
    return ext;
}

Zeroes * PabloKernel::getNullValue(Type * type) {
    if (LLVM_LIKELY(type == nullptr)) {
        type = getStreamTy();
    }
    for (PabloAST * constant : mConstants) {
        if (isa<Zeroes>(constant) && constant->getType() == type) {
            return cast<Zeroes>(constant);
        }
    }
    Zeroes * value = new (mAllocator) Zeroes(type, mAllocator);
    mConstants.push_back(value);
    return value;
}

Ones * PabloKernel::getAllOnesValue(Type * type) {
    if (LLVM_LIKELY(type == nullptr)) {
        type = getStreamTy();
    }
    for (PabloAST * constant : mConstants) {
        if (isa<Ones>(constant) && constant->getType() == type) {
            return cast<Ones>(constant);
        }
    }
    Ones * value = new (mAllocator) Ones(type, mAllocator);
    mConstants.push_back(value);
    return value;
}

void PabloKernel::addInternalProperties(KernelBuilder & b) {
    mPabloCompiler = reinterpret_cast<PabloCompiler *>(b.getCompiler());
    mSizeTy = b.getSizeTy();
    mStreamTy = b.getStreamTy();
    mSymbolTable.reset(new SymbolGenerator(b.getContext(), mAllocator));
    mEntryScope = new (mAllocator) PabloBlock(this, mAllocator);
    mContext = &b.getContext();
    for (const Binding & ss : mInputStreamSets) {
        Var * param = new (mAllocator) Var(ss.getType(), makeName(ss.getName()), mAllocator, Var::KernelInputStream);
        param->addUser(this);
        mInputs.push_back(param);
        mVariables.push_back(param);
    }
    for (const Binding & ss : mOutputStreamSets) {
        Var * result = new (mAllocator) Var(ss.getType(), makeName(ss.getName()), mAllocator, Var::KernelOutputStream);
        result->addUser(this);
        mOutputs.push_back(result);
        mVariables.push_back(result);
    }
    for (const Binding & ss : mOutputScalars) {
        Var * result = new (mAllocator) Var(ss.getType(), makeName(ss.getName()), mAllocator, Var::KernelOutputScalar);
        result->addUser(this);
        mOutputs.push_back(result);
        mVariables.push_back(result);
        mScalarOutputVars.push_back(result);
    }
    generatePabloMethod();
    pablo_function_passes(this);
    mPabloCompiler->initializeKernelData(b);
    mPabloCompiler = nullptr;
    mSizeTy = nullptr;
    mStreamTy = nullptr;
}

bool PabloKernel::isCachable() const {
    return !codegen::EnableIllustrator;
}

void PabloKernel::linkExternalMethods(KernelBuilder & b) {
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        assert (mSharedStateType);
        BEGIN_SCOPED_REGION
        FixedArray<Type *, 2> params;
        params[0] = b.getVoidPtrTy();
        params[1] = b.getVoidPtrTy();
        FunctionType * funTy = FunctionType::get(b.getVoidTy(), params, false);
        b.LinkFunction(KERNEL_ILLUSTRATOR_ENTER_KERNEL, funTy, (void*)&illustratorEnterKernel);
        b.LinkFunction(KERNEL_ILLUSTRATOR_ENTER_LOOP, funTy, (void*)&illustratorEnterLoop);
        b.LinkFunction(KERNEL_ILLUSTRATOR_ITERATE_LOOP, funTy, (void*)&illustratorIterateLoop);
        b.LinkFunction(KERNEL_ILLUSTRATOR_EXIT_LOOP, funTy, (void*)&illustratorExitLoop);
        b.LinkFunction(KERNEL_ILLUSTRATOR_EXIT_KERNEL, funTy, (void*)&illustratorExitKernel);
        END_SCOPED_REGION
    }
    Kernel::linkExternalMethods(b);
}

void PabloKernel::generateInitializeMethod(KernelBuilder & b) {
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        mPabloCompiler = reinterpret_cast<PabloCompiler *>(b.getCompiler());
        mPabloCompiler->initializeIllustrator(b);
        mPabloCompiler = nullptr;
    }
}

void PabloKernel::generateDoBlockMethod(KernelBuilder & b) {
    mPabloCompiler = reinterpret_cast<PabloCompiler *>(b.getCompiler());
    mSizeTy = b.getSizeTy();
    mStreamTy = b.getStreamTy();
    mPabloCompiler->compile(b);
    mPabloCompiler = nullptr;
    mSizeTy = nullptr;
    mStreamTy = nullptr;
}

void PabloKernel::generateFinalBlockMethod(KernelBuilder & b, Value * const remainingBytes) {
    // Standard Pablo convention for final block processing: set a bit marking
    // the position just past EOF, as well as a mask marking all positions past EOF.
    assert (remainingBytes);
    assert (remainingBytes->getType()->isIntegerTy());
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        b.setScalarField("EOFUnnecessaryData", b.CreateSub(b.getSize(b.getBitBlockWidth()), remainingBytes));
    }
    b.setScalarField("EOFbit", b.bitblock_set_bit(remainingBytes));
    b.setScalarField("EOFmask", b.bitblock_mask_from(remainingBytes));
    RepeatDoBlockLogic(b);
}

void PabloKernel::generateFinalizeMethod(KernelBuilder & b) {
    mPabloCompiler = reinterpret_cast<PabloCompiler *>(b.getCompiler());
    mPabloCompiler->releaseKernelData(b);
    if (CompileOptionIsSet(PabloCompilationFlags::EnableProfiling)) {

        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "./" << getName() << ".profile";

        Value * const fd = b.CreateOpenCall(b.GetString(out.str()),
                                             b.getInt32(O_WRONLY | O_CREAT | O_TRUNC),
                                             b.getInt32(S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH));

        Function * dprintf = b.GetDprintf();

        Value * profile; Type * profileTy;

        std::tie(profile, profileTy) = b.getScalarFieldPtr("profile");


        unsigned branchCount = 0;

        for (const auto bb : mPabloCompiler->mBasicBlock) {

            std::string tmp;
            raw_string_ostream str(tmp);
            str << "%lu\t";
            str << bb->getName();
            str << "\n";

            Value * taken = b.CreateLoad(profileTy, b.CreateGEP(profileTy, profile, {b.getInt32(0), b.getInt32(branchCount++)}));
            b.CreateCall(dprintf, {fd, b.GetString(str.str()), taken});

        }

        b.CreateCloseCall(fd);
    }
    mPabloCompiler = nullptr;
}

bool PabloKernel::requiresExplicitPartialFinalStride() const {
    return true;
}

String * PabloKernel::makeName(const llvm::StringRef prefix) const {
    return mSymbolTable->makeString(prefix);
}

Integer * PabloKernel::getInteger(const int64_t value, unsigned intWidth) const {
    return mSymbolTable->getInteger(value, intWidth);
}

llvm::IntegerType * PabloKernel::getInt1Ty() const {
    return IntegerType::getInt1Ty(getModule()->getContext());
}

std::string && annotateKernelNameWithPabloDebugFlags(std::string && name) {
    assert (name.length() > 0);
    if (DebugOptionIsSet(DumpTrace)) {
        name += "+Dump";
    }
    if (CompileOptionIsSet(Flatten)) {
        name += "+Flatten";
    }
    if (CompileOptionIsSet(EnableProfiling)) {
        name += "+BranchP";
    }
    if (CompileOptionIsSet(DisableSimplification)) {
        name += "-Simp";
    }
    if (CompileOptionIsSet(DisableCodeMotion)) {
        name += "-CodeM";
    }
    if (CompileOptionIsSet(EnableDistribution)) {
        name += "+Dist";
    }
    if (CompileOptionIsSet(EnableSchedulingPrePass)) {
        name += "+Sched";
    }
    if (CompileOptionIsSet(EnableTernaryOpt)) {
        name += "+TerOpt";
    }
    if (codegen::CCCOption == "ternary") {
        name += "+ternary";
    }
    switch (CarryMode) {
    case PabloCarryMode::BitBlock:
        name += "+CMBitBlock";
        break;
    case PabloCarryMode::Compressed:
        name += "+CMCompressed";
        break;
    default:
        llvm_unreachable("Illegal PabloCarryMode");
    }
    if (PabloUseLLVMOptimizationPasses) {
        name += "+LLVM-opt";
    }
    return std::move(name);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void PabloKernel::addOptimizationPasses(KernelBuilder & b, SelectedOptimizationPasses & passes) const {
    if (PabloUseLLVMOptimizationPasses) {
        passes.push_back(OptimizationPass::EarlyCSEPass);
        passes.push_back(OptimizationPass::AggressiveInstCombinePass);
        passes.push_back(OptimizationPass::NewGVNPass);
        passes.push_back(OptimizationPass::DCEPass);
    }
}


PabloKernel::PabloKernel(LLVMTypeSystemInterface & ts,
                         std::string && kernelName,
                         std::vector<Binding> stream_inputs,
                         std::vector<Binding> stream_outputs,
                         std::vector<Binding> scalar_parameters,
                         std::vector<Binding> scalar_outputs)
: BlockOrientedKernel(ts, annotateKernelNameWithPabloDebugFlags(std::move(kernelName)),
                      std::move(stream_inputs), std::move(stream_outputs),
                      std::move(scalar_parameters), std::move(scalar_outputs), {})
, PabloAST(PabloAST::ClassTypeId::Kernel, nullptr, mAllocator)
, mPabloCompiler()
, mSymbolTable()
, mEntryScope(nullptr)
, mSizeTy(nullptr)
, mStreamTy(nullptr)
, mContext(nullptr) {
    addNonPersistentScalar(ts.getBitBlockType(), "EOFbit");
    addNonPersistentScalar(ts.getBitBlockType(), "EOFmask");
    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        addNonPersistentScalar(ts.getSizeTy(), "EOFUnnecessaryData");
        addInternalScalar(ts.getSizeTy(), KERNEL_ILLUSTRATOR_STRIDE_NUM);
    }
}

PabloKernel::~PabloKernel() { }

}
