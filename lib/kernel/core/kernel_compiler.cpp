#include <kernel/core/kernel_compiler.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>
#include <llvm/IR/Dominators.h>
#include <llvm/ADT/Twine.h>
#include <boost/intrusive/detail/math.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/container/flat_map.hpp>
#include <kernel/core/streamsetptr.h>
#include <codegen/TypeBuilder.h>
#include <kernel/illustrator/illustrator.h>
#include <llvm/IR/Verifier.h>
#include <boost/regex.hpp>

#include <llvm/IR/PassManager.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/AssumptionCache.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/Analysis/OptimizationRemarkEmitter.h>
#include <llvm/Analysis/PhiValues.h>
#include <llvm/Analysis/ProfileSummaryInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/Reassociate.h>
#include <llvm/Transforms/Scalar/MemCpyOptimizer.h>
#include <llvm/Transforms/Scalar/NewGVN.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/Local.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(14, 0, 0)
#include <llvm/IRPrinter/IRPrintingPasses.h>
#else
#include <llvm/IR/IRPrintingPasses.h>
#endif
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(19, 0, 0)
#include <llvm/IR/PassInstrumentation.h>
#endif

#include <llvm/Support/FileSystem.h>

#include <llvm/Target/TargetMachine.h>             // for TargetMachine, Tar...
#include <llvm/Target/TargetOptions.h>             // for TargetOptions
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Scalar/GVN.h>
#include <llvm/Transforms/Scalar/SROA.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(7, 0, 0)
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Utils.h>
#endif

#if BOOST_VERSION >= 107600
#include <boost/core/bit.hpp>
#endif

#if BOOST_VERSION < 107600
template <typename T> int scan_forward_zeroes(const T x) noexcept;
template <> inline int scan_forward_zeroes<unsigned int>(const unsigned int x) noexcept { return __builtin_ctz(x); }
template <> inline int scan_forward_zeroes<unsigned long>(const unsigned long x) noexcept { return __builtin_ctzl(x); }
template <> inline int scan_forward_zeroes<unsigned long long>(const unsigned long long x) noexcept { return __builtin_ctzll(x); }
#else
template <typename T> int scan_forward_zeroes(const T x) noexcept {
    return boost::core::countr_zero<T>(x);
}
#endif

using namespace llvm;
using namespace boost;
using boost::intrusive::detail::floor_log2;
using boost::container::flat_set;
using boost::container::flat_map;

namespace kernel {

class RemoveRedundantAllocaAndGEPInstructions : public PassInfoMixin<RemoveRedundantAllocaAndGEPInstructions> {
public:
    /// Run the pass over the function.
    PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};

class PHICanonicalizerPass : public PassInfoMixin<PHICanonicalizerPass> {
public:
    /// Run the pass over the function.
    PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
};


class TracePass : public PassInfoMixin<TracePass> {
public:
    TracePass(KernelBuilder & b) : b(b) {}
    PreservedAnalyses run(Function &F, AnalysisManager<Function> &AM);
private:
    KernelBuilder & b;
};

using AttrId = Attribute::KindId;
using Rational = ProcessingRate::Rational;
using RateId = ProcessingRate::KindId;
using StreamSetPort = Kernel::StreamSetPort;
using PortType = Kernel::PortType;

constexpr static auto BUFFER_HANDLE_SUFFIX = "_buffer";
constexpr static auto TERMINATION_SIGNAL = "__termination_signal";

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

// TODO: this check is a bit too strict in general; if the pipeline could request data/
// EOF padding from the MemorySource kernel, it would be possible to re-enable.

// #define CHECK_IO_ADDRESS_RANGE

// TODO: split the init/final into two methods each, one to do allocation/init, and the
// other final/deallocate? Would potentially allow us to reuse the kernel/stream set
// memory in the nested engine if each init method memzero'ed them. Would need to change
// the "main" method.

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::generateKernel(KernelBuilder & b) {
    // NOTE: make sure to keep and reset the original compiler here. A kernel could generate new kernels and
    // reuse the same KernelBuilder to do so; this could result in unexpected behaviour if the this function
    // exits without restoring the original compiler state.
    assert (mTarget->getCompilationStatus() == Kernel::CompilationStatus::FullyInitialized);
    assert (mTarget->getModule() == b.getModule());
    auto const oc = b.getCompiler();
    b.setCompiler(this);
    b.linkAllNecessaryExternalFunctions();
    constructStreamSetBuffers(b);
    #ifndef NDEBUG
    for (const auto & buffer : mStreamSetInputBuffers) {
        assert ("input buffer not set by constructStreamSetBuffers" && buffer.get());
    }
    for (const auto & buffer : mStreamSetOutputBuffers) {
        assert ("output buffer not set by constructStreamSetBuffers" && buffer.get());
    }
    #endif

    addBaseInternalProperties(b);
    mTarget->addInternalProperties(b);
    mTarget->constructStateTypes(b);
    assert (mTarget->getCompilationStatus() == Kernel::CompilationStatus::StateConstructed);
    mTarget->addKernelDeclarations(b);
    callGenerateInitializeMethod(b);
    if (LLVM_UNLIKELY(mStreamSetInputBuffers.empty())) {
        callGenerateExpectedOutputSizeMethod(b);
    }
    callGenerateAllocateSharedInternalStreamSets(b);
    callGenerateInitializeThreadLocalMethod(b);
    callGenerateAllocateThreadLocalInternalStreamSets(b);
    callGenerateDoSegmentMethod(b);
    callGenerateFinalizeThreadLocalMethod(b);
    callGenerateFinalizeMethod(b);
    mTarget->addAdditionalFunctions(b);

    // TODO: we could create a LLVM optimization pass manager here and execute it on this kernel;
    // it would allow the programmer to define a set of optimizations they want executed on the
    // kernel code. However, if compilers are intended to be short lived, we wouldn't be able to
    // easily share it amongst the same type of kernel compiler.

    // What is the cost of generating a pass manager instance for each compiled kernel vs.
    // the complexity of using a factory?

    #ifndef NDEBUG
    SmallVector<char, 256> tmp;
    raw_svector_ostream msg(tmp);
    bool BrokenDebugInfo = false;
    if (LLVM_UNLIKELY(llvm::verifyModule(*b.getModule(), &msg, &BrokenDebugInfo))) {
        b.getModule()->print(errs(), nullptr);
        report_fatal_error(StringRef(msg.str()));
    }
    #endif

    Kernel::SelectedOptimizationPasses passes;
    mTarget->addOptimizationPasses(b, passes);
    runAllOptimizationPasses(b, passes);
    b.setCompiler(oc);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runAllOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::runAllOptimizationPasses(KernelBuilder & b, Kernel::SelectedOptimizationPasses & passes) {

    enum AnalysisPass : uint64_t {
//        _AAManager,
//        _PhiValuesAnalysis,
        _PostDominatorTreeAnalysis,
//        _MemoryDependenceAnalysis,
//        _MemorySSAAnalysis,
//        _LoopAnalysis,
//        _OptimizationRemarkEmitterAnalysis,
        _VerifierAnalysis
    };



    ModuleAnalysisManager MAM;
    MAM.registerPass([&] { return ProfileSummaryAnalysis(); });

    FunctionAnalysisManager FAM;
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    FAM.registerPass([&] { return TargetIRAnalysis(); });

    FAM.registerPass([&] { return AssumptionAnalysis(); });
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return TargetLibraryAnalysis(); });
    FAM.registerPass([&] { return AAManager(); });

    FAM.registerPass([&] { return LoopAnalysis(); });
    FAM.registerPass([&] { return PhiValuesAnalysis(); });
    FAM.registerPass([&] { return MemoryDependenceAnalysis(); });
    FAM.registerPass([&] { return MemorySSAAnalysis(); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

    FAM.registerPass([&] { return OptimizationRemarkEmitterAnalysis(); });

    FunctionPassManager FPM;

    #ifndef NDEBUG
    #define ADD_VERIFY_IR_PASS true
    #else
    const auto __addVerifyPass = codegen::DebugOptionIsSet(codegen::VerifyIR);
    #define ADD_VERIFY_IR_PASS  LLVM_UNLIKELY(__addVerifyPass)
    #endif

    #define FLAG(x) (1ULL << (x))

    uint64_t requiredPasses = 0;

    std::unique_ptr<raw_fd_ostream> unoptimizedOut;

    if (LLVM_UNLIKELY(codegen::ShowUnoptimizedIROption != codegen::OmittedOption)) {
        const auto & options = codegen::ShowUnoptimizedIROption;
        if (options.empty()) {
            unoptimizedOut = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
        } else {
            std::error_code unoptimizedErr;
            unoptimizedOut = std::make_unique<raw_fd_ostream>(options, unoptimizedErr, sys::fs::OpenFlags::OF_None);
        }
        FPM.addPass(PrintFunctionPass(*unoptimizedOut));
    }

    if (ADD_VERIFY_IR_PASS) {
        FPM.addPass(VerifierPass());
        requiredPasses = FLAG(_VerifierAnalysis);
    }
    FPM.addPass(RemoveRedundantAllocaAndGEPInstructions());
    if (LLVM_UNLIKELY(!codegen::TraceOption.empty())) {
        FPM.addPass(TracePass(b));
    }

    #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));
    #elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(14, 0, 0)
    FPM.addPass(SROAPass());
    #else
    FPM.addPass(SROA());
    #endif
    FPM.addPass(InstCombinePass());
    FPM.addPass(DCEPass());
    FPM.addPass(ReassociatePass());
    #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(14, 0, 0)
    FPM.addPass(GVNPass());
    #else
    FPM.addPass(GVN());
    #endif

    using P = Kernel::OptimizationPass;

    for (const P pass : passes) {
        switch (pass) {
             case P::AggressiveInstCombinePass:
                FPM.addPass(AggressiveInstCombinePass());
                break;
            case P::DCEPass:
                FPM.addPass(DCEPass());
                break;
            case P::EarlyCSEPass:
                FPM.addPass(EarlyCSEPass());
                break;
            case P::MemCpyOptPass:
                requiredPasses |= FLAG(_PostDominatorTreeAnalysis);
                FPM.addPass(MemCpyOptPass());
                break;
            case P::NewGVNPass:
                FPM.addPass(NewGVNPass());
                break;
            case P::SimplifyCFGPass:
                FPM.addPass(SimplifyCFGPass());
                break;
            case P::PHICanonicalizerPass:
                FPM.addPass(PHICanonicalizerPass());
                break;
        }
    }

    #define CASE_(PASS) case _##PASS: FAM.registerPass([&] { return PASS(); }); break

    while (requiredPasses) {
        const auto k = scan_forward_zeroes(requiredPasses);
        assert ((requiredPasses & FLAG(k)) == FLAG(k));
        requiredPasses ^= FLAG(k);
        switch (k) {
            CASE_(PostDominatorTreeAnalysis);
            CASE_(VerifierAnalysis);
        default:
            llvm_unreachable("error! unknown analysis pass");
        }
    }

    #undef CASE_
    #undef FLAG

    std::unique_ptr<raw_fd_ostream> optimizedOut;

    if (LLVM_UNLIKELY(codegen::ShowIROption != codegen::OmittedOption)) {
        const auto & options = codegen::ShowIROption;
        if (options.empty()) {
            optimizedOut = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
        } else {
            std::error_code optimizedErr;
            optimizedOut = std::make_unique<raw_fd_ostream>(options, optimizedErr, sys::fs::OpenFlags::OF_None);
        }
        FPM.addPass(PrintFunctionPass(*optimizedOut));
    }

    if (ADD_VERIFY_IR_PASS) {
        FPM.addPass(VerifierPass());
    }

    #undef ADD_VERIFY_IR_PASS

    for (Function & F : *b.getModule()) {
        if (F.empty()) continue;
        FPM.run(F, FAM);
    }

}

#if 0

inline void CPUDriver::preparePassManager() {

    if (mPassManager) return;

    mPassManager = std::make_unique<legacy::PassManager>();

    PassRegistry * Registry = PassRegistry::getPassRegistry();
    initializeCore(*Registry);
    initializeCodeGen(*Registry);
    initializeLowerIntrinsicsPass(*Registry);
    if (LLVM_UNLIKELY(codegen::ShowUnoptimizedIROption != codegen::OmittedOption)) {
        if (LLVM_LIKELY(mIROutputStream == nullptr)) {
            if (!codegen::ShowUnoptimizedIROption.empty()) {
                std::error_code error;
                mUnoptimizedIROutputStream = std::make_unique<raw_fd_ostream>(codegen::ShowUnoptimizedIROption, error, sys::fs::OpenFlags::OF_None);
            } else {
                mUnoptimizedIROutputStream = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
            }
        }
        mPassManager->add(createPrintModulePass(*mUnoptimizedIROutputStream));
    }
    if (IN_DEBUG_MODE || LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::VerifyIR))) {
        mPassManager->add(createVerifierPass());
    }
    if (LLVM_UNLIKELY(!codegen::TraceOption.empty())) {
        mPassManager->add(createTracePass(mBuilder.get(), codegen::TraceOption));
    }
    if (LLVM_UNLIKELY(codegen::ShowIROption != codegen::OmittedOption)) {
        if (LLVM_LIKELY(mIROutputStream == nullptr)) {
            if (!codegen::ShowIROption.empty()) {
                std::error_code error;
                mIROutputStream = std::make_unique<raw_fd_ostream>(codegen::ShowIROption, error, sys::fs::OpenFlags::OF_None);
            } else {
                mIROutputStream = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
            }
        }
        mPassManager->add(createPrintModulePass(*mIROutputStream));
    }
    mPassManager->add(createPromoteMemoryToRegisterPass());    // Promote stack variables to constants or PHI nodes
    mPassManager->add(createSROAPass());                       // Promote elements of aggregate allocas whose addresses are not taken to registers.



    mPassManager->add(createCFGSimplificationPass());          // Remove dead basic blocks and unnecessary branch statements / phi nodes
    mPassManager->add(createEarlyCSEPass());                   // Simple common subexpression elimination pass
    mPassManager->add(createInstructionCombiningPass());       // Simple peephole optimizations and bit-twiddling.
    mPassManager->add(createReassociatePass());                // Canonicalizes commutative expressions
    mPassManager->add(createGVNPass());                        // Global value numbering redundant expression elimination pass
    mPassManager->add(createCFGSimplificationPass());          // Repeat CFG Simplification to "clean up" any newly found redundant phi nodes
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        mPassManager->add(createRemoveRedundantAssertionsPass());
    }
    if (LLVM_UNLIKELY(codegen::ShowASMOption != codegen::OmittedOption)) {
        if (!codegen::ShowASMOption.empty()) {
            std::error_code error;
            mASMOutputStream = std::make_unique<raw_fd_ostream>(codegen::ShowASMOption, error, sys::fs::OpenFlags::OF_None);
        } else {
            mASMOutputStream = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
        }
        #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(10, 0, 0)
        const auto r = mTarget->addPassesToEmitFile(*mPassManager, *mASMOutputStream, nullptr, CGFT_AssemblyFile);
        #elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(7, 0, 0)
        const auto r = mTarget->addPassesToEmitFile(*mPassManager, *mASMOutputStream, nullptr, TargetMachine::CGFT_AssemblyFile);
        #else
        const auto r = mTarget->addPassesToEmitFile(*mPassManager, *mASMOutputStream, TargetMachine::CGFT_AssemblyFile);
        #endif
        if (r) {
            report_fatal_error("LLVM error: could not add emit assembly pass");
        }
    }
    if (IN_DEBUG_MODE || LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::VerifyIR))) {
        mPassManager->add(createVerifierPass());
    }
}

#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructStreamSetBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::constructStreamSetBuffers(KernelBuilder & b) {
    mStreamSetInputBuffers.clear();
    const auto numOfInputStreams = mInputStreamSets.size();
    mStreamSetInputBuffers.resize(numOfInputStreams);
    for (unsigned i = 0; i < numOfInputStreams; ++i) {
        const Binding & input = mInputStreamSets[i];
        mStreamSetInputBuffers[i].reset(new ExternalBuffer(i, b, input.getType(), true, 0));
    }
    mStreamSetOutputBuffers.clear();
    const auto numOfOutputStreams = mOutputStreamSets.size();
    mStreamSetOutputBuffers.resize(numOfOutputStreams);
    for (unsigned i = 0; i < numOfOutputStreams; ++i) {
        const Binding & output = mOutputStreamSets[i];
        mStreamSetOutputBuffers[i].reset(new ExternalBuffer(i + numOfInputStreams, b, output.getType(), true, 0));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addBaseInternalProperties
  ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::addBaseInternalProperties(KernelBuilder & b) {
     // If an output is a managed buffer, store its handle.
    const auto n = mOutputStreamSets.size();
    for (unsigned i = 0; i < n; ++i) {
        const Binding & output = mOutputStreamSets[i];
        Type * const handleTy = mStreamSetOutputBuffers[i]->getHandleType(b);
        assert (handleTy && !handleTy->isPointerTy());
        bool isShared = false;
        bool isManaged = false;
        bool isReturned = false;
        const auto isLocal = Kernel::isLocalBuffer(output, isShared, isManaged, isReturned);
        if (LLVM_UNLIKELY(isLocal)) {
            mTarget->addInternalScalar(handleTy, output.getName() + BUFFER_HANDLE_SUFFIX);
        } else {
            mTarget->addNonPersistentScalar(handleTy, output.getName() + BUFFER_HANDLE_SUFFIX);
        }
    }
    IntegerType * const sizeTy = b.getSizeTy();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {

        // In multi-threaded mode, given a small file, the pipeline could finish before all of the
        // threads are constructed. Since we cannot detect when this occurs without additional
        // book keeping and the behaviour is safe, we do not guard against double termination.
        // All other kernels are checked to ensure that there are no pipeline errors.

        if (mTarget->getTypeId() != Kernel::TypeId::Pipeline || mTarget->hasAttribute(AttrId::InternallySynchronized)) {
            mTarget->addInternalScalar(sizeTy, TERMINATION_SIGNAL);
        } else {
            mTarget->addNonPersistentScalar(sizeTy, TERMINATION_SIGNAL);
        }
    } else {
        mTarget->addNonPersistentScalar(sizeTy, TERMINATION_SIGNAL);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief reset
 ** ------------------------------------------------------------------------------------------------------------- */
template <typename Vec>
inline void reset(Vec & vec, const size_t n) {
    vec.resize(n);
    std::fill_n(vec.begin(), n, nullptr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateInitializeMethod(KernelBuilder & b) {
    mCurrentMethod = mTarget->getInitializeFunction(b);
    mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
    b.SetInsertPoint(mEntryPoint);
    auto arg = mCurrentMethod->arg_begin();
    const auto arg_end = mCurrentMethod->arg_end();
    auto nextArg = [&]() {
        assert (arg != arg_end);
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    assert (getHandle() == nullptr);
    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
    }
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    for (const auto & binding : mInputScalars) {
        b.setScalarField(binding.getName(), nextArg());
    }
    bindAdditionalInitializationArguments(b, arg, arg_end);
    assert (arg == arg_end);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect) && mTarget->isStateful())) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::WRITE);
    }
    // TODO: we could permit shared managed buffers here if we passed in the buffer
    // into the init method. However, since there are no uses of this in any written
    // program, we currently prohibit it.
    initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    // any kernel can set termination on initialization
    Type * termSignalTy;
    std::tie(mTerminationSignalPtr, termSignalTy) = getScalarFieldPtr(b, TERMINATION_SIGNAL);
    b.CreateStore(b.getSize(KernelBuilder::TerminationCode::None), mTerminationSignalPtr);
    mTarget->generateInitializeMethod(b);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect) && mTarget->isStateful())) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::READ);
    }
    b.CreateRet(b.CreateLoad(termSignalTy, mTerminationSignalPtr));
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindFamilyInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::callGenerateExpectedOutputSizeMethod(KernelBuilder & b) {
    assert (mTarget->getNumOfStreamInputs() == 0);
    mCurrentMethod = mTarget->getExpectedOutputSizeFunction(b);
    mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
    b.SetInsertPoint(mEntryPoint);
    auto arg = mCurrentMethod->arg_begin();
    #ifndef NDEBUG
    const auto arg_end = mCurrentMethod->arg_end();
    #endif
    auto nextArg = [&]() {
        assert (arg != arg_end);
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
    }
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect) && mTarget->isStateful())) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::WRITE);
    }
    Value * const retVal = mTarget->generateExpectedOutputSizeMethod(b);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect) && mTarget->isStateful())) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::READ);
    }
    assert (retVal);
    b.CreateRet(retVal);
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindFamilyInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::bindAdditionalInitializationArguments(KernelBuilder & /* b */, ArgIterator & /* arg */, const ArgIterator & /* arg_end */) {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateInitializeThreadLocalMethod(KernelBuilder & b) {
    if (mTarget->hasThreadLocal()) {
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getInitializeThreadLocalFunction(b);
        mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
        b.SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        StructType * const threadLocalTy = mTarget->getThreadLocalStateType();
        Value * const providedState = b.CreatePointerCast(nextArg(), threadLocalTy->getPointerTo());
        BasicBlock * const allocThreadLocal = BasicBlock::Create(b.getContext(), "allocThreadLocalState", mCurrentMethod);
        BasicBlock * const initThreadLocal = BasicBlock::Create(b.getContext(), "initThreadLocalState", mCurrentMethod);
        b.CreateCondBr(b.CreateIsNull(providedState), allocThreadLocal, initThreadLocal);

        b.SetInsertPoint(allocThreadLocal);
        Value * const allocedState = b.CreatePageAlignedMalloc(mTarget->getThreadLocalStateType());
        assert (providedState->getType() == allocedState->getType());
        b.CreateBr(initThreadLocal);

        b.SetInsertPoint(initThreadLocal);
        PHINode * const threadLocal = b.CreatePHI(providedState->getType(), 2);
        threadLocal->addIncoming(providedState, mEntryPoint);
        threadLocal->addIncoming(allocedState, allocThreadLocal);
        mThreadLocalHandle = threadLocal;
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
        mTarget->generateInitializeThreadLocalMethod(b);
        b.CreateRet(threadLocal);
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callAllocateSharedInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateAllocateSharedInternalStreamSets(KernelBuilder & b) {
    if (LLVM_UNLIKELY(mTarget->allocatesInternalStreamSets())) {
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getAllocateSharedInternalStreamSetsFunction(b);
        mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
        b.SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        Value * const expectedNumOfStrides = nextArg();
        initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
        initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
        mTarget->generateAllocateSharedInternalStreamSetsMethod(b, expectedNumOfStrides);
        b.CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callAllocateThreadLocalInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateAllocateThreadLocalInternalStreamSets(KernelBuilder & b) {
    if (LLVM_UNLIKELY(mTarget->allocatesInternalStreamSets() && mTarget->hasThreadLocal())) {
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b);
        mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
        b.SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        setThreadLocalHandle(nextArg());
        Value * const expectedNumOfStrides = nextArg();
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
        initializeOwnedBufferHandles(b, InitializeOptions::IncludeThreadLocalScalars);
        mTarget->generateAllocateThreadLocalInternalStreamSetsMethod(b, expectedNumOfStrides);
        b.CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getLCMOfFixedRateInputs
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ Rational KernelCompiler::getLCMOfFixedRateInputs(const Kernel * const target) {
    Rational rateLCM(1);
    bool first = true;
    for (const Binding & input : target->getInputStreamSetBindings()) {
        const ProcessingRate & rate = input.getRate();
        if (LLVM_LIKELY(rate.isFixed())) {
            if (first) {
                rateLCM = rate.getRate();
                first = false;
            } else {
                rateLCM = lcm(rateLCM, rate.getRate());
            }
        }
    }
    return rateLCM;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setDoSegmentProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::setDoSegmentProperties(KernelBuilder & b, const ArrayRef<Value *> args) {

    // WARNING: any change to this must be reflected in Kernel::addDoSegmentDeclaration,
    // Kernel::getDoSegmentFields, KernelCompiler::getDoSegmentProperties,
    // PipelineCompiler::buildKernelCallArgumentList and PipelineKernel::addOrDeclareMainFunction

    auto arg = args.begin();
    auto nextArg = [&]() {
        assert (arg != args.end());
        Value * const v = *arg; assert (v);
        std::advance(arg, 1);
        return v;
    };

    const auto enableAsserts = codegen::DebugOptionIsSet(codegen::EnableAsserts);

    clearInternalStateAfterCodeGen();

    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
        if (LLVM_UNLIKELY(enableAsserts)) {
            b.CreateAssert(getHandle(), "%s: shared handle cannot be null", b.GetString(getName()));
        }
    }
    if (LLVM_UNLIKELY(mTarget->hasThreadLocal())) {
        setThreadLocalHandle(nextArg());
        if (LLVM_UNLIKELY(enableAsserts)) {
            b.CreateAssert(getThreadLocalHandle(), "%s: thread local handle cannot be null", b.GetString(getName()));
        }
    }    
    const auto internallySynchronized = mTarget->hasAttribute(AttrId::InternallySynchronized);
    // TODO: the simplest way of ensuring we can allow external I/O to be passed though the main pipeline
    // even if there are multiple consumers of the input with differing processing rates is to special
    // case the outermost pipeline such that all I/O is always addressible. This creates a problem,
    // however, in that we will not be able to optimize a single kernel program by having the main
    // function call it directly instead of a pipeline. Given that this is not a realistic use-case,
    // we're ignoring that limitation for now.
    const auto isPipeline = (mTarget->getTypeId() == Kernel::TypeId::Pipeline);
    const auto isMainPipeline = isPipeline && !internallySynchronized;

    Rational fixedRateLCM{0};
    mFixedRateFactor = nullptr;

    if (LLVM_UNLIKELY(isMainPipeline)) {
        mIsFinal = b.getTrue();
        mRawNumOfStrides = nullptr;
        mNumOfStrides = nullptr;
    } else {
        if (LLVM_UNLIKELY(internallySynchronized)) {
            mExternalSegNo = nextArg();
        }
        mRawNumOfStrides = nextArg();
        if (LLVM_UNLIKELY(mTarget->hasAttribute(AttrId::MustExplicitlyTerminate))) {
            mIsFinal = nullptr;
            mNumOfStrides = mRawNumOfStrides;
        } else {
            mIsFinal = b.CreateIsNull(mRawNumOfStrides);
            mNumOfStrides = b.CreateSelect(mIsFinal, b.getSize(1), mRawNumOfStrides);
        }
        if (LLVM_LIKELY(mTarget->hasFixedRateIO())) {
            fixedRateLCM = getLCMOfFixedRateInputs(mTarget);
            mFixedRateFactor = nextArg();
        }
        #ifdef ENABLE_PAPI
        if (LLVM_UNLIKELY(codegen::PapiCounterOptions.compare(codegen::OmittedOption) != 0)) {
            mPAPIEventSetId = nextArg();
        }
        #endif
    }

    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);

    // NOTE: the disadvantage of passing the stream pointers as a parameter is that it becomes more difficult
    // to access a stream set from a LLVM function call. We could create a stream-set aware function creation
    // and call system here but that is not an ideal way of handling this.

    #ifdef CHECK_IO_ADDRESS_RANGE
    auto checkStreamRange = [&](const StreamSetBuffer * const buffer, const Binding & binding, Value * const startItemCount) {

        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "StreamSet " << getName() << ":" << binding.getName();

        DataLayout DL(b.getModule());
        Type * const intPtrTy = DL.getIntPtrType(b.getInt8PtrTy());

        ConstantInt * const ZERO = b.getSize(0);
        ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());

        Value * const fromIndex = b.CreateUDiv(startItemCount, BLOCK_WIDTH);
        Value * const baseAddress = buffer->getBaseAddress(b);
        Value * const startPtr = buffer->getStreamBlockPtr(b, baseAddress, ZERO, fromIndex);
        Value * const start = b.CreatePtrToInt(startPtr, intPtrTy);

        Value * const endPos = b.CreateAdd(startItemCount, buffer->getCapacity(b));
        Value * const toIndex = b.CreateCeilUDiv(endPos, BLOCK_WIDTH);
        Value * const endPtr = buffer->getStreamBlockPtr(b, baseAddress, ZERO, toIndex);
        Value * const end = b.CreatePtrToInt(endPtr, intPtrTy);

        Value * const length = b.CreateSub(end, start);

        b.CreateAssert(b.CreateICmpULE(start, end),
                        "%s: illegal kernel I/O address range [0x%" PRIx64 ", 0x%" PRIx64 ")",
                        b.GetString(out.str()), start, end);

        b.CheckAddress(startPtr, length, out.str());


    };
    #endif

    const auto numOfInputs = getNumOfStreamInputs();

    IntegerType * const sizeTy = b.getSizeTy();
    for (unsigned i = 0; i < numOfInputs; i++) {

        /// ----------------------------------------------------
        /// virtual base address
        /// ----------------------------------------------------
        StreamSetBuffer * const buffer = mStreamSetInputBuffers[i].get();

        const Binding & input = mInputStreamSets[i];
        Value * const virtualBaseAddress = b.CreatePointerCast(nextArg(), buffer->getPointerType());
        Value * const localHandle = b.CreateAllocaAtEntryPoint(buffer->getHandleType(b));
        buffer->setHandle(localHandle); assert (localHandle);
        buffer->setBaseAddress(b, virtualBaseAddress);

        if (LLVM_UNLIKELY(internallySynchronized)) {
            mInputIsClosed[i] = nextArg();
            assert (mInputIsClosed[i]->getType() == b.getInt1Ty());
        } else {
            mInputIsClosed[i] = mIsFinal;
        }

        /// ----------------------------------------------------
        /// processed item count
        /// ----------------------------------------------------

        const ProcessingRate & rate = input.getRate();
        Value * processed = nullptr;
        if (isMainPipeline || isAddressable(input)) {
            mProcessedInputItemPtr[i] = nextArg();
            processed = b.CreateLoad(sizeTy, mProcessedInputItemPtr[i]);
        } else {
            if (LLVM_LIKELY(isCountable(input))) {
                processed = nextArg();
            } else { // isRelative
                const auto port = getStreamPort(rate.getReference());
                assert (port.Type == PortType::Input && port.Number < i);
                assert (mProcessedInputItemPtr[port.Number]);
                Value * const ref = b.CreateLoad(sizeTy, mProcessedInputItemPtr[port.Number]);
                processed = b.CreateMulRational(ref, rate.getRate());
            }
            // NOTE: we create a redundant alloca to store the input param so that
            // Mem2Reg can convert it into a PHINode if the item count is updated in
            // a loop; otherwise, it will be discarded in favor of the param itself.
            Value * const processedItems = b.CreateAllocaAtEntryPoint(sizeTy);
            b.CreateStore(processed, processedItems);
            mProcessedInputItemPtr[i] = processedItems;
        }
        /// ----------------------------------------------------
        /// accessible item count
        /// ----------------------------------------------------
        Value * accessible = nullptr;
        if (LLVM_UNLIKELY(isMainPipeline || requiresItemCount(input))) {
            accessible = nextArg();
        } else {
            accessible = b.CreateCeilUMulRational(mFixedRateFactor, rate.getRate() / fixedRateLCM);
        }
        assert (accessible);
        assert (accessible->getType() == sizeTy);
        mAccessibleInputItems[i] = accessible;

        Value * avail = b.CreateAdd(processed, accessible);
        mAvailableInputItems[i] = avail;
        if (input.hasLookahead()) {
            avail = b.CreateAdd(avail, b.getSize(input.getLookahead()));
        }
        buffer->setCapacity(b, avail);
        #ifdef CHECK_IO_ADDRESS_RANGE
        if (LLVM_UNLIKELY(enableAsserts)) {
            checkStreamRange(buffer, input, processed);
        }
        #endif
    }

    // set all of the output buffers
    const auto numOfOutputs = getNumOfStreamOutputs();

    const auto canTerminate = canSetTerminateSignal();

    for (unsigned i = 0; i < numOfOutputs; i++) {

        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        StreamSetBuffer * const buffer = mStreamSetOutputBuffers[i].get();

        const Binding & output = mOutputStreamSets[i];
        bool isShared = false;
        bool isManaged = false;
        bool isReturned = false;
        const auto isLocal =  Kernel::isLocalBuffer(output, isShared, isManaged, isReturned);
        if (LLVM_UNLIKELY(isShared)) {
            Value * const handle = nextArg();
            assert (isa<DynamicBuffer>(buffer));
            buffer->setHandle(b.CreatePointerCast(handle, buffer->getHandlePointerType(b)));
        } else if (LLVM_UNLIKELY(isMainPipeline || isLocal)) {
            // If an output is a managed buffer, the address is stored within the state instead
            // of being passed in through the function call.
            mUpdatableOutputBaseVirtualAddressPtr[i] = nextArg();
            Value * handle = getScalarFieldPtr(b, output.getName() + BUFFER_HANDLE_SUFFIX).first;
            buffer->setHandle(handle);
        } else {
            Value * const virtualBaseAddress = b.CreatePointerCast(nextArg(), buffer->getPointerType());
            Value * const localHandle = b.CreateAllocaAtEntryPoint(buffer->getHandleType(b));
            buffer->setHandle(localHandle);
            buffer->setBaseAddress(b, virtualBaseAddress);
            assert (isa<ExternalBuffer>(buffer));
        }

        /// ----------------------------------------------------
        /// produced item count
        /// ----------------------------------------------------
        const ProcessingRate & rate = output.getRate();
        Value * produced = nullptr;
        if (LLVM_LIKELY(canTerminate || isMainPipeline || isAddressable(output))) {
            mProducedOutputItemPtr[i] = nextArg();
            produced = b.CreateLoad(sizeTy, mProducedOutputItemPtr[i]);
        } else {
            if (LLVM_LIKELY(isCountable(output))) {
                produced = nextArg();
            } else { // isRelative
                // For now, if something is produced at a relative rate to another stream in a kernel that
                // may terminate, its final item count is inherited from its reference stream and cannot
                // be set independently. Should they be independent at early termination?
                const auto port = getStreamPort(rate.getReference());
                assert (port.Type == PortType::Input || (port.Type == PortType::Output && port.Number < i));
                const auto & items = (port.Type == PortType::Input) ? mProcessedInputItemPtr : mProducedOutputItemPtr;
                Value * const ref = b.CreateLoad(sizeTy, items[port.Number]);
                produced = b.CreateMulRational(ref, rate.getRate());
            }
            Value * const producedItems = b.CreateAllocaAtEntryPoint(sizeTy);
            b.CreateStore(produced, producedItems);
            mProducedOutputItemPtr[i] = producedItems;
        }
        assert (produced);
        assert (produced->getType() == sizeTy);
        mInitiallyProducedOutputItems[i] = produced;

        /// ----------------------------------------------------
        /// writable / consumed item count
        /// ----------------------------------------------------
        Value * writable = nullptr;
        if (isShared || isLocal) {
            Value * const consumed = nextArg();
            assert (consumed->getType() == sizeTy);
            mConsumedOutputItems[i] = consumed;
            writable = buffer->getLinearlyWritableItems(b, produced, consumed);
            assert (writable && writable->getType() == sizeTy);
        } else {
            if (isMainPipeline || requiresItemCount(output)) {
                writable = nextArg();
                assert (writable && writable->getType() == sizeTy);
            } else if (mFixedRateFactor) {
                writable = b.CreateCeilUMulRational(mFixedRateFactor, rate.getRate() / fixedRateLCM);
                assert (writable && writable->getType() == sizeTy);
            }
            Value * capacity = nullptr;
            if (writable) {
                capacity = b.CreateAdd(produced, writable);
                buffer->setCapacity(b, capacity);
                #ifdef CHECK_IO_ADDRESS_RANGE
                if (LLVM_UNLIKELY(enableAsserts)) {
                    checkStreamRange(buffer, output, produced);
                }
                #endif
            } else {
                capacity = ConstantExpr::getNeg(b.getSize(1));
                buffer->setCapacity(b, capacity);
            }
        }
        mWritableOutputItems[i] = writable;
    }
    assert (arg == args.end());

    // initialize the termination signal if this kernel can set it
    mTerminationSignalPtr = nullptr;
    if (canTerminate) {
        mTerminationSignalPtr = getScalarFieldPtr(b, TERMINATION_SIGNAL).first;
        if (LLVM_UNLIKELY(enableAsserts)) {
            Value * const unterminated =
                b.CreateICmpEQ(b.CreateLoad(sizeTy, mTerminationSignalPtr), b.getSize(KernelBuilder::TerminationCode::None));
            b.CreateAssert(unterminated, getName() + ".doSegment was called after termination?");
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getDoSegmentProperties
 *
 * Reverse of the setDoSegmentProperties operation; used by the PipelineKernel when constructing internal threads
 * to simplify passing of the state data.
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> KernelCompiler::getDoSegmentProperties(KernelBuilder & b) const {

    // WARNING: any change to this must be reflected in addDoSegmentDeclaration, getDoSegmentFields,
    // setDoSegmentProperties, and PipelineCompiler::writeKernelCall

    std::vector<Value *> props;
    if (LLVM_LIKELY(mTarget->isStateful())) {
        props.push_back(mSharedHandle); assert (mSharedHandle);
    }
    if (LLVM_UNLIKELY(mTarget->hasThreadLocal())) {
        props.push_back(mThreadLocalHandle); assert (mThreadLocalHandle);
    }
    const auto internallySynchronized = mTarget->hasAttribute(AttrId::InternallySynchronized);
    const auto isPipeline = (mTarget->getTypeId() == Kernel::TypeId::Pipeline);
    const auto isMainPipeline = isPipeline && !internallySynchronized;

    if (LLVM_UNLIKELY(internallySynchronized)) {
        props.push_back(mExternalSegNo);
    }

    if (LLVM_LIKELY(!isMainPipeline)) {
        props.push_back(mNumOfStrides); assert (mNumOfStrides);
        if (LLVM_LIKELY(mTarget->hasFixedRateIO())) {
            props.push_back(mFixedRateFactor);
        }
        #ifdef ENABLE_PAPI
        if (LLVM_UNLIKELY(codegen::PapiCounterOptions.compare(codegen::OmittedOption) != 0)) {
            props.push_back(mPAPIEventSetId);
        }
        #endif
    }

    PointerType * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const sizeTy = b.getSizeTy();
    const auto numOfInputs = getNumOfStreamInputs();
    for (unsigned i = 0; i < numOfInputs; i++) {
        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        const auto & buffer = mStreamSetInputBuffers[i];
        props.push_back(b.CreatePointerCast(buffer->getBaseAddress(b), voidPtrTy));
        /// ----------------------------------------------------
        /// is closed
        /// ----------------------------------------------------
        if (LLVM_UNLIKELY(internallySynchronized)) {
            props.push_back(mInputIsClosed[i]);
        }
        /// ----------------------------------------------------
        /// processed item count
        /// ----------------------------------------------------
        const Binding & input = mInputStreamSets[i];
        if (isMainPipeline || isAddressable(input)) {
            props.push_back(mProcessedInputItemPtr[i]);
        } else if (LLVM_LIKELY(isCountable(input))) {
            props.push_back(b.CreateLoad(sizeTy, mProcessedInputItemPtr[i]));
        }
        /// ----------------------------------------------------
        /// accessible item count
        /// ----------------------------------------------------
        if (isMainPipeline || requiresItemCount(input)) {
            props.push_back(mAccessibleInputItems[i]);
        }
    }

    // set all of the output buffers
    const auto numOfOutputs = getNumOfStreamOutputs();
    const auto canTerminate = canSetTerminateSignal();

    for (unsigned i = 0; i < numOfOutputs; i++) {
        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        const auto & buffer = mStreamSetOutputBuffers[i];
        const Binding & output = mOutputStreamSets[i];

        bool isShared = false;
        bool isManaged = false;
        bool isReturned = false;
        const auto isLocal = Kernel::isLocalBuffer(output, isShared, isManaged, isReturned);

        Value * handle = nullptr;
        if (LLVM_UNLIKELY(isShared)) {            
            handle = b.CreatePointerCast(buffer->getHandle(), voidPtrTy);
        } else if (LLVM_UNLIKELY(isMainPipeline || isLocal)) {
            // If an output is a managed buffer, the address is stored within the state instead
            // of being passed in through the function call.
            PointerType * const voidPtrPtrTy = voidPtrTy->getPointerTo();
            handle = b.CreatePointerCast(mUpdatableOutputBaseVirtualAddressPtr[i], voidPtrPtrTy);
        } else {
            handle = b.CreatePointerCast(buffer->getBaseAddress(b), voidPtrTy);
        }
        props.push_back(handle);

        /// ----------------------------------------------------
        /// produced item count
        /// ----------------------------------------------------
        if (LLVM_LIKELY(canTerminate || isMainPipeline ||isAddressable(output))) {
            props.push_back(mProducedOutputItemPtr[i]);
        } else if (LLVM_LIKELY(isCountable(output))) {
            props.push_back(b.CreateLoad(sizeTy, mProducedOutputItemPtr[i]));
        }
        /// ----------------------------------------------------
        /// writable / consumed item count
        /// ----------------------------------------------------
        if (LLVM_UNLIKELY(isShared || isLocal)) {
            props.push_back(mConsumedOutputItems[i]);
        } else if (isMainPipeline || requiresItemCount(output)) {
            props.push_back(mWritableOutputItems[i]);
        }
    }
    return props;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateDoSegmentMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateDoSegmentMethod(KernelBuilder & b) {

    assert (mInputStreamSets.size() == mStreamSetInputBuffers.size());
    assert (mOutputStreamSets.size() == mStreamSetOutputBuffers.size());

    mCurrentMethod = mTarget->getDoSegmentFunction(b);
    mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
    b.SetInsertPoint(mEntryPoint);

    BEGIN_SCOPED_REGION
    Vec<Value *, 64> args;
    args.reserve(mCurrentMethod->arg_size());
    for (auto ArgI = mCurrentMethod->arg_begin(); ArgI != mCurrentMethod->arg_end(); ++ArgI) {
        args.push_back(&(*ArgI));
    }
    setDoSegmentProperties(b, args);
    END_SCOPED_REGION

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::WRITE);
    }

    mTarget->generateKernelMethod(b);

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle, CBuilder::Protect::READ);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();

    IntegerType * const sizeTy = b.getSizeTy();

    for (unsigned i = 0; i < numOfOutputs; i++) {
        // Write the virtual base address out to inform the pipeline of any changes
        const auto & buffer = mStreamSetOutputBuffers[i];
        if (mUpdatableOutputBaseVirtualAddressPtr[i]) {
            Value * const baseAddress = buffer->getBaseAddress(b);
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                const Binding & output = mOutputStreamSets[i];
                out << getName() << ":%s is returning a virtual base address "
                                    "computed from a null base address.";
                b.CreateAssert(baseAddress, out.str(), b.GetString(output.getName()));
            }
            Value * produced = mInitiallyProducedOutputItems[i];
            assert (isFromCurrentFunction(b, produced, false));
            // TODO: will LLVM optimizations replace the following with the already loaded value?
            // If not, re-loading it here may reduce register pressure / compilation time.
            if (mProducedOutputItemPtr[i]) {
                assert (isFromCurrentFunction(b, mProducedOutputItemPtr[i], false));
                produced = b.CreateAlignedLoad(sizeTy, mProducedOutputItemPtr[i], sizeof(size_t));
            }
            assert (isFromCurrentFunction(b, produced, true));
            Value * vba = buffer->getVirtualBasePtr(b, baseAddress, produced);
            vba = b.CreatePointerCast(vba, b.getVoidPtrTy());

            assert (isFromCurrentFunction(b, mUpdatableOutputBaseVirtualAddressPtr[i], true));

            b.CreateStore(vba, mUpdatableOutputBaseVirtualAddressPtr[i]);
        }
    }

    // return the termination signal (if one exists)
    if (mTerminationSignalPtr) {
        assert (isFromCurrentFunction(b, mTerminationSignalPtr, true));
        b.CreateRet(b.CreateAlignedLoad(sizeTy, mTerminationSignalPtr, sizeof(size_t)));
        mTerminationSignalPtr = nullptr;
    } else {
        b.CreateRetVoid();
    }
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateFinalizeThreadLocalMethod(KernelBuilder & b) {
    if (mTarget->hasThreadLocal()) {
        mCurrentMethod = mTarget->getFinalizeThreadLocalFunction(b);
        mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
        b.SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        mCommonThreadLocalHandle = nextArg();
        mThreadLocalHandle = nextArg();
        initializeScalarMap(b, InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars);
        mTarget->generateFinalizeThreadLocalMethod(b);
        b.CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateFinalizeMethod(KernelBuilder & b) {
    mCurrentMethod = mTarget->getFinalizeFunction(b);
    mEntryPoint = BasicBlock::Create(b.getContext(), "entry", mCurrentMethod);
    b.SetInsertPoint(mEntryPoint);
    auto arg = mCurrentMethod->arg_begin();
    auto nextArg = [&]() {
        assert (arg != mCurrentMethod->arg_end());
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        setThreadLocalHandle(nextArg());
    }
    assert (arg == mCurrentMethod->arg_end());
    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b.CreateMProtect(mTarget->getSharedStateType(), mSharedHandle,CBuilder::Protect::WRITE);
    }
    initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    mTarget->generateFinalizeMethod(b); // may be overridden by the Kernel subtype
    const auto outputs = getFinalOutputScalars(b);
    if (outputs.empty()) {
        b.CreateRetVoid();
    } else {
        const auto n = outputs.size();
        if (n == 1) {
            b.CreateRet(outputs[0]);
        } else {
            b.CreateAggregateRet(outputs.data(), n);
        }
    }
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getFinalOutputScalars
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> KernelCompiler::getFinalOutputScalars(KernelBuilder & b) {
    const auto n = mOutputScalars.size();
    std::vector<Value *> outputs(n);
    for (unsigned i = 0; i < n; ++i) {
        auto ref = getScalarFieldPtr(b, mOutputScalars[i].getName());
        outputs[i] = b.CreateLoad(ref.second, ref.first);
    }
    return outputs;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeScalarMap
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeScalarMap(KernelBuilder & b, const InitializeOptions options) {

    FixedArray<Value *, 3> indices;
    indices[0] = b.getInt32(0);

    StructType * const sharedTy = mTarget->getSharedStateType();

    StructType * const threadLocalTy = mTarget->getThreadLocalStateType();

    DataLayout DL(b.getModule());

    #ifndef NDEBUG
    auto verifyStateType = [](Value * const handle, StructType * const stateType) {
        if (handle == nullptr && stateType == nullptr) {
            return true;
        }
        if (handle == nullptr || stateType == nullptr) {
            return false;
        }
        if (handle->getType() != stateType->getPointerTo()) {
            return false;
        }
        assert (!stateType->isOpaque());
        assert (stateType->isSized());
        const auto n = stateType->getStructNumElements();
        for (unsigned i = 0; i < n; ++i) {
            assert (isa<StructType>(stateType->getStructElementType(i)));
        }
        return true;
    };
    assert ("incorrect shared handle/type!" && verifyStateType(mSharedHandle, sharedTy));
    if (options == InitializeOptions::IncludeThreadLocalScalars) {
        assert ("incorrect thread local handle/type!" && verifyStateType(mThreadLocalHandle, threadLocalTy));
    }
    #endif

    mScalarFieldMap.clear();

    auto addToScalarFieldMap = [&](StringRef bindingName, Value * const scalar, Type * const expectedType, Type * const actualType) {
        const auto i = mScalarFieldMap.insert(std::make_pair(bindingName, std::make_pair(scalar, expectedType)));
        if (LLVM_UNLIKELY(!i.second)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Kernel " << getName() << " contains two scalar or alias fields named " << bindingName;
            report_fatal_error(Twine(out.str()));
        }
        if (LLVM_UNLIKELY(actualType != expectedType && expectedType)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Scalar " << getName() << '.' << bindingName << " was expected to be a ";
            expectedType->print(out);
            out << " but was stored as a ";
            actualType->print(out);
            report_fatal_error(Twine(out.str()));
        }
    };

    flat_set<unsigned> sharedGroups;
    flat_set<unsigned> threadLocalGroups;

    bool hasThreadLocalAccum = false;

    for (const auto & scalar : mInternalScalars) {
        assert (scalar.getValueType());
        switch (scalar.getScalarType()) {
            case ScalarType::Internal:
                sharedGroups.insert(scalar.getGroup());
                break;
            case ScalarType::ThreadLocal:
                if (options == InitializeOptions::DoNotIncludeThreadLocalScalars) continue;
                threadLocalGroups.insert(scalar.getGroup());
                if (options != InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars) continue;
                if (scalar.getAccumulationRule() != Kernel::ThreadLocalScalarAccumulationRule::DoNothing) {
                    assert (mCommonThreadLocalHandle && "no main thread local given?");
                    hasThreadLocalAccum = true;
                }
                break;
            default: break;
        }
    }

    std::vector<unsigned> sharedIndex(sharedGroups.size() + 2, 0);
    std::vector<unsigned> threadLocalIndex(threadLocalGroups.size(), 0);

    BasicBlock * combineToMainThreadLocal = nullptr;

    if (LLVM_UNLIKELY(hasThreadLocalAccum)) {
        combineToMainThreadLocal = b.CreateBasicBlock("combineToMainThreadLocal");
    }

    auto enumerate = [&](const Bindings & bindings, const unsigned groupId) {
        indices[1] = b.getInt32(groupId);
        auto & k = sharedIndex[groupId];
        for (const auto & binding : bindings) {
            assert (sharedTy);
            assert ((groupId) < sharedTy->getStructNumElements());
            assert (k < sharedTy->getStructElementType(groupId)->getStructNumElements());
            assert (sharedTy->getStructElementType(groupId)->getStructElementType(k) == binding.getType());
            Type * actualType = sharedTy->getStructElementType(groupId)->getStructElementType(k);
            indices[2] = b.getInt32(k); k += 2;
            Value * const scalar = b.CreateGEP(sharedTy, mSharedHandle, indices);
            addToScalarFieldMap(binding.getName(), scalar, binding.getType(), actualType);
        }
    };

    enumerate(mInputScalars, 0);

    BasicBlock * combineExit = combineToMainThreadLocal;

    for (const auto & binding : mInternalScalars) {
        Value * scalar = nullptr;
        Type * scalarType = nullptr;
        auto getGroupIndex = [&](const flat_set<unsigned> & groups) {
            const auto f = groups.find(binding.getGroup());
            assert (f != groups.end());
            return std::distance(groups.begin(), f);
        };

        switch (binding.getScalarType()) {
            case ScalarType::Internal:
                assert (mSharedHandle);
                BEGIN_SCOPED_REGION
                const auto j = getGroupIndex(sharedGroups) + 1;
                indices[1] = b.getInt32(j);
                auto & k = sharedIndex[j];
                assert ((j) < sharedTy->getStructNumElements());
                assert (k < sharedTy->getStructElementType(j)->getStructNumElements());
                scalarType = sharedTy->getStructElementType(j)->getStructElementType(k);
                assert (scalarType == binding.getValueType());
                indices[2] = b.getInt32(k); k += 2;
                scalar = b.CreateGEP(sharedTy, mSharedHandle, indices);
                END_SCOPED_REGION
                break;
            case ScalarType::ThreadLocal:
                if (options == InitializeOptions::DoNotIncludeThreadLocalScalars) continue;
                assert (mThreadLocalHandle);
                BEGIN_SCOPED_REGION
                const auto j = getGroupIndex(threadLocalGroups);
                indices[1] = b.getInt32(j);
                auto & k = threadLocalIndex[j];
                assert ((j) < threadLocalTy->getStructNumElements());
                assert (k < threadLocalTy->getStructElementType(j)->getStructNumElements());
                scalarType = threadLocalTy->getStructElementType(j)->getStructElementType(k);
                assert (scalarType == binding.getValueType());
                indices[2] = b.getInt32(k); k += 2;
                scalar = b.CreateGEP(threadLocalTy, mThreadLocalHandle, indices);

                if (LLVM_UNLIKELY(options == InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars)) {

                    Value * const mainScalar = b.CreateGEP(threadLocalTy, mCommonThreadLocalHandle, indices);

                    using AccumRule = Kernel::ThreadLocalScalarAccumulationRule;

                    if (binding.getAccumulationRule() != AccumRule::DoNothing) {

                        const auto ip = b.saveIP();
                        b.SetInsertPoint(combineExit);

                        if (isa<ArrayType>(scalarType)) {
                            ArrayType * const arrayTy = cast<ArrayType>(scalarType);

                            unsigned depth = 2;
                            for (ArrayType * aTy = arrayTy;;) {
                                Type * const eTy = aTy->getArrayElementType();
                                if (eTy->isArrayTy()) {
                                    aTy = cast<ArrayType>(eTy);
                                    ++depth;
                                } else {
                                    assert (eTy->isIntOrIntVectorTy());
                                    break;
                                }
                            }

                            const auto size = depth;

                            ConstantInt * const i32_ZERO = b.getInt32(0);
                            ConstantInt * const i32_ONE = b.getInt32(1);

                            SmallVector<Value *, 4> indices(size);
                            indices[0] = i32_ZERO;


                            std::function<BasicBlock *(unsigned, Type *)> recursiveAccum = [&](const unsigned idx, Type * const elemTy) {
                                assert (idx <= size);
                                assert (indices.size() == size);

                                BasicBlock * const entry = b.GetInsertBlock();

                                if (idx == size) {
                                    Value * const scalarPtr = b.CreateGEP(scalarType, scalar, indices);
                                    const auto align = b.getTypeSize(DL, elemTy);
                                    Value * const scalarVal = b.CreateAlignedLoad(elemTy, scalarPtr, align);
                                    assert (scalarVal->getType()->isIntOrIntVectorTy());
                                    Value * const mainScalarPtr = b.CreateGEP(scalarType, mainScalar, indices);
                                    Value * mainScalarVal = b.CreateAlignedLoad(elemTy, mainScalarPtr, align);
                                    assert (scalarVal->getType() == mainScalarVal->getType());
                                    switch (binding.getAccumulationRule()) {
                                        case AccumRule::Sum:
                                            mainScalarVal = b.CreateAdd(scalarVal, mainScalarVal, "sum");
                                            break;
                                        default: llvm_unreachable("unexpected thread-local scalar accumulation rule");
                                    }
                                    b.CreateStore(mainScalarVal, mainScalarPtr);
                                    return entry;
                                } else {

                                    BasicBlock * const loop = b.CreateBasicBlock();
                                    b.CreateBr(loop);

                                    b.SetInsertPoint(loop);
                                    PHINode * const idxPhi = b.CreatePHI(b.getInt32Ty(), 2);
                                    idxPhi->addIncoming(i32_ZERO, entry);
                                    assert (idx < indices.size());
                                    indices[idx] = idxPhi;

                                    BasicBlock * const loopExit =
                                        recursiveAccum(idx + 1U, cast<ArrayType>(elemTy)->getArrayElementType());

                                    BasicBlock * const exit = b.CreateBasicBlock();
                                    Value * const nextIdx = b.CreateAdd(idxPhi, i32_ONE);
                                    idxPhi->addIncoming(nextIdx, loopExit);

                                    const auto m = cast<ArrayType>(elemTy)->getNumElements();
                                    if (LLVM_UNLIKELY(m == 0)) {
                                        report_fatal_error(Twine(getName()) + ": cannot automatically accumulate a 0-element scalar");
                                    }

                                    b.CreateCondBr(b.CreateICmpNE(nextIdx, b.getInt32(m)), loop, exit);

                                    b.SetInsertPoint(exit);
                                    return exit;
                                }
                            };

                            combineExit = recursiveAccum(1, arrayTy);
                        } else {
                            Value * const scalarVal = b.CreateLoad(scalarType, scalar);
                            Value * mainScalarVal = b.CreateLoad(scalarType, mainScalar);
                            switch (binding.getAccumulationRule()) {
                                case Kernel::ThreadLocalScalarAccumulationRule::Sum:
                                    mainScalarVal = b.CreateAdd(scalarVal, mainScalarVal);
                                    break;
                                default: llvm_unreachable("unexpected thread-local scalar accumulation rule");
                            }
                            b.CreateStore(mainScalarVal, mainScalar);
                        }
                        b.restoreIP(ip);
                    }



                }
                END_SCOPED_REGION
                break;
            case ScalarType::NonPersistent:
                scalarType = binding.getValueType(); assert (scalarType);
                scalar = b.CreateAllocaAtEntryPoint(scalarType);
                b.CreateStore(Constant::getNullValue(scalarType), scalar);
                break;
            default: llvm_unreachable("I/O scalars cannot be internal");
        }

        addToScalarFieldMap(binding.getName(), scalar, binding.getValueType(), scalarType);
    }

    enumerate(mOutputScalars, sharedGroups.size() + 1);

    // finally add any aliases
    for (const auto & alias : mScalarAliasMap) {
        const auto f = mScalarFieldMap.find(alias.second);
        if (f != mScalarFieldMap.end()) {
            addToScalarFieldMap(alias.first, f->second.first, f->second.second, f->second.second);
        }
    }

    if (LLVM_UNLIKELY(hasThreadLocalAccum)) {
        BasicBlock * const exit = b.CreateBasicBlock("afterThreadLocalAccumulation");
        Value * const cond = b.CreateICmpEQ(mThreadLocalHandle, mCommonThreadLocalHandle);
        b.CreateCondBr(cond, exit, combineToMainThreadLocal);
        b.SetInsertPoint(combineExit);
        b.CreateBr(exit);
        b.SetInsertPoint(exit);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAlias
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::addAlias(llvm::StringRef alias, llvm::StringRef scalarName) {
    mScalarAliasMap.emplace_back(alias, scalarName);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeBindingMap
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeIOBindingMap() {

    auto enumerate = [&](const Bindings & bindings, const BindingType type) {
        const auto n = bindings.size();
        for (unsigned i = 0; i < n; ++i) {
            const auto & binding = bindings[i];
            mBindingMap.insert(std::make_pair(binding.getName(), BindingMapEntry{type, i}));
        }
    };

    enumerate(mInputScalars, BindingType::ScalarInput);
    enumerate(mOutputScalars, BindingType::ScalarOutput);
    enumerate(mInputStreamSets, BindingType::StreamInput);
    enumerate(mOutputStreamSets, BindingType::StreamOutput);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeOwnedBufferHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeOwnedBufferHandles(KernelBuilder & b, const InitializeOptions /* options */) {
    const auto numOfOutputs = getNumOfStreamOutputs();
    for (unsigned i = 0; i < numOfOutputs; i++) {
        const Binding & output = mOutputStreamSets[i];
        bool isShared = false;
        bool isManaged = false;
        bool isReturned = false;
        if (LLVM_UNLIKELY(Kernel::isLocalBuffer(output, isShared, isManaged, isReturned))) {
            auto handle = getScalarFieldPtr(b, output.getName() + BUFFER_HANDLE_SUFFIX);
            const auto & buffer = mStreamSetOutputBuffers[i]; assert (buffer.get());
            buffer->setHandle(handle.first);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const BindingMapEntry & KernelCompiler::getBinding(const BindingType type, const llvm::StringRef name) const {

    const auto f = mBindingMap.find(name);
    if (f != mBindingMap.end()) {
        const BindingMapEntry & entry = f->second;
        assert (entry.Type == type);
        return entry;
    }

    SmallVector<char, 256> tmp;
    raw_svector_ostream out(tmp);
    out << "Kernel " << getName() << " does not contain an ";
    switch (type) {
        case BindingType::ScalarInput:
        case BindingType::StreamInput:
            out << "input"; break;
        case BindingType::ScalarOutput:
        case BindingType::StreamOutput:
            out << "output"; break;
    }
    out << ' ';
    switch (type) {
        case BindingType::ScalarInput:
        case BindingType::ScalarOutput:
            out << "scalar"; break;
        case BindingType::StreamInput:
        case BindingType::StreamOutput:
            out << "streamset"; break;
    }
    out << " named \"" << name << "\"\n"
           "Currently contains:";


    auto listAvailableBindings = [&](const Bindings & bindings) {
        if (LLVM_UNLIKELY(bindings.empty())) {
            out << "<no bindings>";
        } else {
            char joiner = ' ';
            for (const auto & binding : bindings) {
                out << joiner << binding.getName();
                joiner = ',';
            }
        }
        out << '\n';
    };

    switch (type) {
        case BindingType::ScalarInput:
            listAvailableBindings(mInputScalars); break;
        case BindingType::ScalarOutput:
            listAvailableBindings(mOutputScalars); break;
        case BindingType::StreamInput:
            listAvailableBindings(mInputStreamSets); break;
        case BindingType::StreamOutput:
            listAvailableBindings(mOutputStreamSets); break;
    }

    report_fatal_error(Twine(out.str()));
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getStreamPort
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetPort KernelCompiler::getStreamPort(const StringRef name) const {

    // NOTE: temporary refactoring step to limit changes outside of the kernel class

    static_assert(static_cast<unsigned>(BindingType::StreamInput) == static_cast<unsigned>(PortType::Input), "");
    static_assert(static_cast<unsigned>(BindingType::StreamOutput) == static_cast<unsigned>(PortType::Output), "");

    const auto f = mBindingMap.find(name);
    if (LLVM_LIKELY(f != mBindingMap.end())) {

        const BindingMapEntry & entry = f->second;
        switch (entry.Type) {
            case BindingType::StreamInput:
            case BindingType::StreamOutput:
                return StreamSetPort(static_cast<PortType>(entry.Type), entry.Index);
            default: break;
        }
    }

    SmallVector<char, 256> tmp;
    raw_svector_ostream out(tmp);
    out << "Kernel " << getName() << " does not contain a streamset named: \"" << name << "\"";
    report_fatal_error(Twine(out.str()));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarFieldPtr
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::ScalarRef KernelCompiler::getScalarFieldPtr(KernelBuilder & b, const StringRef name) const {
    assert (this);
    if (LLVM_UNLIKELY(mScalarFieldMap.empty())) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "Scalar map for " << getName() << " was not initialized prior to calling getScalarFieldPtr";
        assert (false);
        report_fatal_error(Twine(out.str()));
    } else {
        const auto f = mScalarFieldMap.find(name);
        if (LLVM_UNLIKELY(f == mScalarFieldMap.end())) {
            #ifdef NDEBUG
            SmallVector<char, 1024> tmp;
            raw_svector_ostream out(tmp);
            #else
            auto & out = errs();
            #endif
            out << "Scalar map for " << getName() << " does not contain " << name << "\n\n"
                "Currently contains:";
            char spacer = ' ';
            for (const auto & entry : mScalarFieldMap) {
                out << spacer << entry.getKey();
                spacer = ',';
            }
            #ifdef NDEBUG
            report_fatal_error(Twine(out.str()));
            #else
            out << "\n";
            assert (false);
            #endif
        }
        ScalarRef result = f->second;
        assert (isFromCurrentFunction(b, result.first, false));
        return result;
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadLocalScalarFieldPtr
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::ScalarRef KernelCompiler::getThreadLocalScalarFieldPtr(KernelBuilder & b, Value * handle, const StringRef name) const {

    const auto count = mInternalScalars.size();

    flat_map<size_t, size_t> threadLocalGroups;

    size_t i = 0;
    size_t groupIndex = 0;
    size_t scalarIndex = 0;
    for (; i < count; ++i) {
        const InternalScalar & scalar = mInternalScalars[i];
        if (scalar.getScalarType() == ScalarType::ThreadLocal) {
            const auto g = scalar.getGroup();
            auto f = threadLocalGroups.find(g);
            if (LLVM_UNLIKELY(f == threadLocalGroups.end())) {
                f = threadLocalGroups.emplace(g, 0).first;
            }
            if (scalar.getName() == name) {
                groupIndex = g;
                scalarIndex = f->second;
                break;
            }
            f->second++;
        }
    }



    for (; i < count; ++i) {
        const InternalScalar & scalar = mInternalScalars[i];
        if (scalar.getScalarType() == ScalarType::ThreadLocal) {
            threadLocalGroups.emplace(scalar.getGroup(), 0);
        }
    }

    StructType * const threadLocalTy = mTarget->getThreadLocalStateType(); assert (threadLocalTy);

    const auto f = threadLocalGroups.find(groupIndex);
    const auto groupPos = std::distance(threadLocalGroups.begin(), f);

    assert (groupPos < threadLocalTy->getStructNumElements());
    assert ((scalarIndex * 2) < threadLocalTy->getStructElementType(groupPos)->getStructNumElements());


    FixedArray<Value *, 3> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(groupPos);
    indices[2] = b.getInt32(scalarIndex * 2);

    Value * ptr = b.CreateGEP(threadLocalTy, handle, indices); assert (ptr);
    Type * ty = threadLocalTy->getStructElementType(groupPos)->getStructElementType(scalarIndex * 2);
    return ScalarRef{ptr, ty};

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarValuePtr
 ** ------------------------------------------------------------------------------------------------------------- */
bool KernelCompiler::hasScalarField(const llvm::StringRef name) const {
    return mScalarFieldMap.count(name) != 0;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getLowerBound
 ** ------------------------------------------------------------------------------------------------------------- */
Rational KernelCompiler::getLowerBound(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.hasReference()) {
        return rate.getLowerBound() * getLowerBound(getStreamBinding(rate.getReference()));
    } else {
        return rate.getLowerBound();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getUpperBound
 ** ------------------------------------------------------------------------------------------------------------- */
Rational KernelCompiler::getUpperBound(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.hasReference()) {
        return rate.getUpperBound() * getUpperBound(getStreamBinding(rate.getReference()));
    } else {
        return rate.getUpperBound();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief requiresOverflow
 ** ------------------------------------------------------------------------------------------------------------- */
bool KernelCompiler::requiresOverflow(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.isFixed() || binding.hasAttribute(AttrId::BlockSize)) {
        return false;
    } else if (rate.isRelative()) {
        return requiresOverflow(getStreamBinding(rate.getReference()));
    } else {
        return true;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearInternalStateAfterCodeGen
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::clearInternalStateAfterCodeGen() {
    // TODO: this function is more of a sanity check to ensure we don't have a pointer
    // to an out-of-scope LLVM-value. It should be possible to remove it.
    mScalarFieldMap.clear();
    mSharedHandle = nullptr;
    mThreadLocalHandle = nullptr;
    mCommonThreadLocalHandle = nullptr;
    mExternalSegNo = nullptr;
    mCurrentMethod = nullptr;
    mEntryPoint = nullptr;
    mIsFinal = nullptr;
    mNumOfStrides = nullptr;
    mTerminationSignalPtr = nullptr;
    const auto numOfInputs = getNumOfStreamInputs();
    reset(mInputIsClosed, numOfInputs);
    reset(mProcessedInputItemPtr, numOfInputs);
    reset(mAccessibleInputItems, numOfInputs);
    reset(mAvailableInputItems, numOfInputs);
    const auto numOfOutputs = getNumOfStreamOutputs();
    reset(mProducedOutputItemPtr, numOfOutputs);
    reset(mInitiallyProducedOutputItems, numOfOutputs);
    reset(mWritableOutputItems, numOfOutputs);
    reset(mConsumedOutputItems, numOfOutputs);
    reset(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    for (const auto & buffer : mStreamSetInputBuffers) {
        buffer->setHandle(nullptr);
    }
    for (const auto & buffer : mStreamSetOutputBuffers) {
        buffer->setHandle(nullptr);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerIllustrator
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::registerIllustrator(KernelBuilder & b,
                                         llvm::Constant * kernelName, llvm::Constant * streamName,
                                         const size_t rows, const size_t cols, const size_t itemWidth, const MemoryOrdering ordering,
                                         IllustratorTypeId illustratorTypeId, const char replacement0, const char replacement1,
                                         const ArrayRef<size_t> loopIds) const {

    auto init = mTarget->getInitializeFunction(b);
    assert (init);
    auto arg = init->arg_begin();
    auto nextArg = [&]() {
        assert (arg != init->arg_end());
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    assert (mTarget->isStateful());
    Value * handle = nextArg();
    Instruction * ret = nullptr;
    for (auto & bb : *init) {
        assert (bb.getTerminator());
        if (isa<ReturnInst>(bb.getTerminator())) {
            ret = bb.getTerminator();
            break;
        }
    }
    assert (ret && "no return statement found in initialize kernel function?");
    Value * illustratorObject = nullptr;
    for (const auto & binding : mInputScalars) {
        Value * inputArg = nextArg();
        if (binding.getName() == KERNEL_ILLUSTRATOR_CALLBACK_OBJECT) {
            illustratorObject = inputArg;
            break;
        }
    }
    assert (illustratorObject && "no illustrator object found?");

    auto ip = b.saveIP();

    b.SetInsertPoint(ret->getPrevNode());

    registerIllustrator(b, illustratorObject, kernelName, streamName, handle, rows, cols, itemWidth, ordering, illustratorTypeId, replacement0, replacement1, loopIds);

    b.restoreIP(ip);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerIllustrator
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::registerIllustrator(KernelBuilder & b,
                                         Value * illustratorObject,
                                         Constant * kernelName, Constant * streamName, Value * handle,
                                         const size_t rows, const size_t cols, const size_t itemWidth, const MemoryOrdering ordering,
                                         IllustratorTypeId illustratorTypeId,
                                         const char replacement0, const char replacement1,
                                         const ArrayRef<size_t> loopIds) const {

    assert (isFromCurrentFunction(b, illustratorObject, false));
    assert (isFromCurrentFunction(b, handle, false));

    Function * regFunc = b.getModule()->getFunction(KERNEL_REGISTER_ILLUSTRATOR_CALLBACK); assert (regFunc);
    FixedArray<Value *, 12> args;
    args[0] = b.CreatePointerCast(illustratorObject, b.getVoidPtrTy());
    args[1] = kernelName;
    args[2] = streamName;
    args[3] = b.CreatePointerCast(handle, b.getVoidPtrTy());
    args[4] = b.getSize(rows);
    args[5] = b.getSize(cols);
    args[6] = b.getSize(itemWidth);
    args[7] = b.getInt8((unsigned)ordering);
    args[8] = b.getInt8((unsigned)illustratorTypeId);
    args[9] = b.getInt8(replacement0);
    args[10] = b.getInt8(replacement1);

    IntegerType * const sizeTy = b.getSizeTy();
    PointerType * const sizePtrTy = sizeTy->getPointerTo();
    Constant * loopIdConstant = nullptr;
    if (loopIds.empty()) {
        loopIdConstant = ConstantPointerNull::get(sizePtrTy);
    } else {
        const auto n = loopIds.size();
        SmallVector<Constant *, 8> ids(n + 1);
        for (size_t i = 0; i < n; ++i) {
            assert (loopIds[i] != 0);
            ids[i] = b.getSize(loopIds[i]);
        }
        ids[n] = b.getSize(0);
        ArrayType * arTy = ArrayType::get(sizeTy, n + 1);
        Constant * ar = ConstantArray::get(arTy, ids);
        GlobalVariable * const gv = new GlobalVariable(*b.getModule(), arTy, true, GlobalValue::ExternalLinkage, ar);
        loopIdConstant = ConstantExpr::getPointerCast(gv, sizePtrTy);
    }
    args[11] = loopIdConstant;
    b.CreateCall(regFunc->getFunctionType(), regFunc, args);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief captureStreamData
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::captureStreamData(KernelBuilder & b, Constant * kernelName, Constant * streamName, Value * handle,
                                       Value * strideNum,
                                       Type * type, const MemoryOrdering ordering,
                                       Value * streamData, Value * from, Value * to)  const {

    FixedArray<Value *, 9> args;
    args[0] = b.CreatePointerCast(b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT), b.getVoidPtrTy());
    args[1] = kernelName;
    args[2] = streamName;
    args[3] = b.CreatePointerCast(handle, b.getVoidPtrTy());
    args[4] = strideNum;
    args[5] = b.CreatePointerCast(streamData, b.getVoidPtrTy());
    args[6] = from;
    args[7] = to;
    args[8] = b.getSize(b.getBitBlockWidth());

    Function * func = b.getModule()->getFunction(KERNEL_ILLUSTRATOR_CAPTURE_CALLBACK); assert (func);
    b.CreateCall(func->getFunctionType(), func, args);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::KernelCompiler(not_null<Kernel *> kernel) noexcept
: mTarget(kernel)
, mInputStreamSets(kernel->mInputStreamSets)
, mOutputStreamSets(kernel->mOutputStreamSets)
, mInputScalars(kernel->mInputScalars)
, mOutputScalars(kernel->mOutputScalars)
, mInternalScalars(kernel->mInternalScalars) {
    initializeIOBindingMap();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief destructor
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::~KernelCompiler() {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief RemoveRedundantAllocaAndGEPInstructions::run
 ** ------------------------------------------------------------------------------------------------------------- */
PreservedAnalyses RemoveRedundantAllocaAndGEPInstructions::run(Function &F,
                                                 FunctionAnalysisManager &AM) {

    assert (!F.empty());


    SmallVector<AllocaInst *, 32> allocas;

    BasicBlock & bb = F.getEntryBlock();

    Instruction * inst = bb.getFirstNonPHIOrDbgOrLifetime();
    while (inst) {
        #ifndef NDEBUG
        for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
            Value * const op = inst->getOperand(i);
            if (op == nullptr) {
                report_fatal_error("null operand");
            }
        }
        #endif
        Instruction * const nextNode = inst->getNextNode();
        if (isa<AllocaInst>(inst) || isa<GetElementPtrInst>(inst)) {
            if (LLVM_UNLIKELY(inst->getNumUses() == 0)) {
                inst->eraseFromParent();
                inst = nextNode;
                continue;
            }
        }
        if (isa<AllocaInst>(inst)) {
            if (isAllocaPromotable(cast<AllocaInst>(inst))) {
                allocas.push_back(cast<AllocaInst>(inst));
            }
        }
        inst = nextNode;
    }

    if (!allocas.empty()) {
        auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
        PromoteMemToReg(allocas, DT);
    }

    return PreservedAnalyses::all();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief PHICanonicalizerPass::run
 ** ------------------------------------------------------------------------------------------------------------- */
PreservedAnalyses PHICanonicalizerPass::run(Function &F, FunctionAnalysisManager &AM) {

    assert (!F.empty());

    // LLVM is not aggressive enough with how it deals with phi nodes. To ensure that
    // we collapse every phi node in which all incoming values are identical into the
    // incoming value, we execute the following mini optimization pass.

    // TODO: check the newer versions of LLVM to see if any can do this now.

    SmallVector<BasicBlock *, 16> preds;
    SmallVector<Value *, 16> value;

    bool anyPhis = false;

    for (BasicBlock & bb : F) {

        preds.assign(pred_begin(&bb), pred_end(&bb));
        const auto n = preds.size();
        value.resize(n);

        Instruction * inst = &bb.front();
        while (isa<PHINode>(inst)) {
            PHINode * const phi = cast<PHINode>(inst);
            #ifndef NDEBUG
            if (LLVM_UNLIKELY(phi->getNumIncomingValues() != n || n == 0)) {
                bb.print(errs(), nullptr, true, false);
                errs() << "\n\nIllegal PHINode: ";
                phi->print(errs(), true);
            }
            #endif
            inst = inst->getNextNode();
            if (LLVM_LIKELY(phi->hasNUsesOrMore(1))) {
                Value * const value = phi->getIncomingValue(0);
                assert (value);
                const auto n = phi->getNumIncomingValues();
                for (unsigned i = 1; i != n; ++i) {
                    Value * const op = phi->getIncomingValue(i);
                    assert (op);
                    if (LLVM_LIKELY(op != value)) {
                        goto keep_phi_node;
                    }
                }
                phi->replaceAllUsesWith(value);
            }

            RecursivelyDeleteDeadPHINode(phi);
            continue;
            // ----------------------------------------------------------------------------------
            //  canonicalize the phi node ordering for the eliminate duplicate phi node function
            // ----------------------------------------------------------------------------------
keep_phi_node:
            bool canonicalize = false;
            for (unsigned i = 0; i != n; ++i) {
                const auto f = std::find(preds.begin(), preds.end(), phi->getIncomingBlock(i));
                assert ("phi-node has invalid incoming block?" && f != preds.end());
                const auto j = std::distance(preds.begin(), f);
                canonicalize |= (j != i);
                value[j] = phi->getIncomingValue(i);
            }
            if (canonicalize) {
                for (unsigned i = 0; i != n; ++i) {
                    phi->setIncomingBlock(i, preds[i]);
                    phi->setIncomingValue(i, value[i]);
                }
            }
            anyPhis = true;
        }
        if (LLVM_LIKELY(anyPhis)) {
            EliminateDuplicatePHINodes(&bb);
        }
    }

    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief TracePass::run
 ** ------------------------------------------------------------------------------------------------------------- */
PreservedAnalyses TracePass::run(Function &F, FunctionAnalysisManager & AM) {

    const boost::regex ex(codegen::TraceOption);

    SmallVector<Instruction *, 16> toTrace;
    for (auto & B : F) {

        assert (toTrace.empty());

        for (Instruction & inst : B) {
            if (LLVM_UNLIKELY(boost::regex_match(inst.getName().data(), ex))) {
                toTrace.push_back(&inst);
            }
        }

        if (LLVM_LIKELY(toTrace.empty())) {
            continue;
        }

        for (Instruction * I : toTrace) {
            Instruction * N = I;
            if (LLVM_UNLIKELY(isa<PHINode>(I))) {
                N = B.getFirstNonPHIOrDbgOrLifetime();
            } else if (LLVM_LIKELY(I != B.getTerminator())) {
                assert (I->getNextNode());
                N = I->getNextNode();
            }
            b.SetInsertPoint(N);
            const Type * ty = I->getType();
            if (ty->isIntOrPtrTy()) {
                b.CallPrintInt(I->getName(), I);
            } else if (ty->isVectorTy()) {
                b.CallPrintRegister(I->getName(), I);
            }
        }

        toTrace.clear();
    }


    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();

}


}
