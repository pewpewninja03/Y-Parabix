#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/kernel_compiler.h>
#include <kernel/core/idisa_target.h>
#include <toolchain/toolchain.h>
#include <llvm/Support/DynamicLibrary.h>           // for LoadLibraryPermanently
#include <llvm/ExecutionEngine/ExecutionEngine.h>  // for EngineBuilder
#include <llvm/ExecutionEngine/RTDyldMemoryManager.h>
#include <llvm/InitializePasses.h>                 // for initializeCodeGencd .
#include <llvm/PassRegistry.h>                     // for PassRegistry
#include <llvm/Support/CodeGen.h>                  // for Level, Level::None
#include <llvm/Support/Compiler.h>                 // for LLVM_UNLIKELY
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Timer.h>
#include <objcache/object_cache.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/IR/Verifier.h>
#include "llvm/IR/Mangler.h"
#include <llvm/ExecutionEngine/MCJIT.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Support/CommandLine.h>

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(7, 0, 0)
#define OF_None F_None
#endif
#include <llvm/ADT/Statistic.h>
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(8, 0, 0)
#include <llvm/IR/LegacyPassManager.h>
#else
#include <llvm/IR/PassTimingInfo.h>
#endif
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
#include <llvm/Support/Host.h>
#endif

#ifndef NDEBUG
#define IN_DEBUG_MODE true
#else
#define IN_DEBUG_MODE false
#endif

#if defined(__clang__) || defined (__GNUC__)
    #define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
    #define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif

using namespace llvm;
using namespace kernel;

using AttrId = kernel::Attribute::KindId;

ATTRIBUTE_NO_SANITIZE_ADDRESS
CPUDriver::CPUDriver(std::string && moduleName)
: BaseDriver(std::move(moduleName))
, mUnoptimizedIROutputStream{}
, mIROutputStream{}
, mASMOutputStream{}
, mEngine(nullptr)
, mTarget(nullptr) {

    InitializeNativeTarget();
    InitializeNativeTargetAsmPrinter();
    llvm::sys::DynamicLibrary::LoadLibraryPermanently(nullptr);

    std::string errMessage;
    EngineBuilder builder{std::unique_ptr<Module>(mMainModule)};
    builder.setErrorStr(&errMessage);
    builder.setVerifyModules(false);
    builder.setEngineKind(EngineKind::JIT);
    builder.setTargetOptions(codegen::target_Options);
    builder.setOptLevel(codegen::BackEndOptLevel);

    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
    StringMap<bool> features;
    sys::getHostCPUFeatures(features);
    #else
    const StringMap<bool> features = sys::getHostCPUFeatures();
    #endif

    std::vector<std::string> attrs;
    for (auto & flag : features) {
        if (flag.second) {
            attrs.push_back("+" + flag.first().str());
        }
    }
    builder.setMAttrs(attrs);

    mTarget.reset(builder.selectTarget());
    if (mTarget == nullptr) {
        throw std::runtime_error("Could not selectTarget");
    }
    mEngine.reset(builder.create());
    if (mEngine == nullptr) {
        throw std::runtime_error("Could not create ExecutionEngine: " + errMessage);
    }
    if (mObjectCache) {
        mEngine->setObjectCache(mObjectCache.get());
    }
    mEngine->DisableSymbolSearching(false);
    mEngine->DisableLazyCompilation(true);
    mEngine->DisableGVCompilation(true);

    auto triple = mTarget->getTargetTriple().getTriple();
    const DataLayout DL(mTarget->createDataLayout());
    mMainModule->setTargetTriple(triple);
    mMainModule->setDataLayout(DL);
    mBuilder.reset(IDISA::GetIDISA_Builder(*mContext, features));
    mBuilder->setDriver(*this);
    mBuilder->setModule(mMainModule);
}

Function * CPUDriver::addLinkFunction(Module * mod, llvm::StringRef name, FunctionType * type, void * functionPtr) const {
    if (LLVM_UNLIKELY(mod == nullptr)) {
        report_fatal_error("addLinkFunction(" + name + ") cannot be called until after addKernel");
    }
    Function * f = mod->getFunction(name);
    if (LLVM_UNLIKELY(f == nullptr)) {
        f = Function::Create(type, Function::ExternalLinkage, name, mod);
        #ifndef ORCJIT
        mEngine->updateGlobalMapping(f, functionPtr);
        #endif
    } else if (LLVM_UNLIKELY(f->getType() != type->getPointerTo())) {
        report_fatal_error("Cannot link " + name + ": a function with a different signature already exists with that name in " + mod->getName());
    }
    return f;
}

void CPUDriver::generateUncachedKernels() {
    if (mUncachedKernel.empty()) return;

    // TODO: we may be able to reduce unnecessary optimization work by having kernel specific optimization passes.

    // NOTE: we currently require DCE and Mem2Reg for each kernel to eliminate any unnecessary scalar -> value
    // mappings made by the base KernelCompiler. That could be done in a more focused manner, however, as each
    // mapping is known.

    mCachedKernel.reserve(mUncachedKernel.size());
    for (unsigned i = 0; i < mUncachedKernel.size(); ++i) {
        auto & kernel = mUncachedKernel[i];
        NamedRegionTimer T(kernel->getSignature(), kernel->getName(),
                           "kernel", "Kernel Generation",
                           codegen::TimeKernelsIsEnabled);
        kernel->generateKernel(getBuilder());
        Module * const module = kernel->getModule(); assert (module);
        module->setTargetTriple(mMainModule->getTargetTriple());
        module->setDataLayout(mMainModule->getDataLayout());
        mCachedKernel.emplace_back(kernel.release());
    }

    mUncachedKernel.clear();

    llvm::reportAndResetTimings();
    llvm::PrintStatistics();
}

void * CPUDriver::finalizeObject(kernel::Kernel * const pk) {

    using ModuleSet = llvm::SmallVector<Module *, 32>;

    ModuleSet Infrequent;
    ModuleSet Normal;

    // TODO: As far as I can tell, the PassRegistry intrinsics below do not affect the final assembly but I
    // cannot locate the implementation code for initializeLowerIntrinsicsPass in any file within the llvm
    // code base. It's called by multiple LLVM tools, however, so must get generated somehow? Look further
    // into this case.

//    PassRegistry * Registry = PassRegistry::getPassRegistry();
//    initializeCore(*Registry);
//    initializeCodeGen(*Registry);
//    initializeLowerIntrinsicsPass(*Registry);

    for (const auto & kernel : mCompiledKernel) {
        kernel->ensureLoaded();
    }

    for (const auto & kernel : mCachedKernel) {
        if (LLVM_UNLIKELY(kernel->getModule() == nullptr)) {
            report_fatal_error(llvm::StringRef(kernel->getName()) + " was neither loaded from cache nor generated prior to finalizeObject");
        }
        Module * const m = kernel->getModule();
        assert ("cached kernel has no module?" && m);
        if (LLVM_UNLIKELY(kernel->hasAttribute(AttrId::InfrequentlyUsed))) {
            assert ("pipeline cannot be infrequently compiled" && !isa<PipelineKernel>(kernel));
            Infrequent.emplace_back(m);
        } else {
            Normal.emplace_back(m);
        }
    }

    auto addModules = [&](const ModuleSet & S, const CodeGenOptLevel level) {
        if (S.empty()) return;
        mEngine->getTargetMachine()->setOptLevel(level);
        for (Module * M : S) {
            mEngine->addModule(std::unique_ptr<Module>(M));
        }
        mEngine->finalizeObject();
    };

    auto removeModules = [&](const ModuleSet & S) {
        for (Module * M : S) {
            mEngine->removeModule(M);
        }
    };

    // compile any uncompiled kernels
    addModules(Infrequent, codegen::BackEndOptLevel);
    addModules(Normal, CodeGenOptLevel::Default);

    // write/declare the "main" method
    auto mainModule = std::make_unique<Module>("main", *mContext);
    mainModule->setTargetTriple(mMainModule->getTargetTriple());
    mainModule->setDataLayout(mMainModule->getDataLayout());
    mBuilder->setModule(mainModule.get());
    pk->addKernelDeclarations(getBuilder());

    // TODO: to ensure that we can pass the correct num of threads, we cannot statically compile the
    // main method until we add the thread count as a parameter. Investigate whether we can make a
    // better "wrapper" method for that that allows easier access to the output scalars.

    const auto e = true; // pk->containsKernelFamilyCalls() || pk->generatesDynamicRepeatingStreamSets();
    const auto method = e ? Kernel::AddInternal : Kernel::DeclareExternal;
    Function * const main = pk->addOrDeclareMainFunction(getBuilder(), method);
    mBuilder->setModule(mMainModule);

    // NOTE: the pipeline kernel is destructed after calling clear unless this driver preserves kernels!
    if (getPreservesKernels()) {
        for (auto & kernel : mCachedKernel) {
            mPreservedKernel.emplace_back(kernel.release());
        }
        for (auto & kernel : mCompiledKernel) {
            mPreservedKernel.emplace_back(kernel.release());
        }
    } else {
        mPreservedKernel.clear();
    }

    // return the compiled main method
    mEngine->getTargetMachine()->setOptLevel(CodeGenOptLevel::None);
    const auto mainModulePtr = mainModule.get();
    mEngine->addModule(std::move(mainModule));

    if (LLVM_UNLIKELY(codegen::ShowASMOption != codegen::OmittedOption)) {
        if (!codegen::ShowASMOption.empty()) {
            std::error_code error;
            mASMOutputStream = std::make_unique<raw_fd_ostream>(codegen::ShowASMOption, error, sys::fs::OpenFlags::OF_None);
        } else {
            mASMOutputStream = std::make_unique<raw_fd_ostream>(STDERR_FILENO, false, true);
        }

        // TODO: there does not seem to be an ASM printer for the new PassManager?
        auto pm = std::make_unique<legacy::PassManager>();

        #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(18, 0, 0)
        const auto r = mTarget->addPassesToEmitFile(*pm, *mASMOutputStream, nullptr, CodeGenFileType::AssemblyFile);
        #elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(10, 0, 0)
        const auto r = mTarget->addPassesToEmitFile(*pm, *mASMOutputStream, nullptr, CGFT_AssemblyFile);
        #elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(7, 0, 0)
        const auto r = mTarget->addPassesToEmitFile(*mPassManager, *mASMOutputStream, nullptr, TargetMachine::CGFT_AssemblyFile);
        #else
        const auto r = mTarget->addPassesToEmitFile(*mPassManager, *mASMOutputStream, TargetMachine::CGFT_AssemblyFile);
        #endif
        if (r) {
            report_fatal_error("LLVM error: could not add emit assembly pass");
        }

        for (const auto & kernel : mCachedKernel) {
            pm->run(*kernel->getModule());
        }

        for (const auto & kernel : mCompiledKernel) {
            pm->run(*kernel->getModule());
        }

    }

    mCachedKernel.clear();
    mCompiledKernel.clear();

    mEngine->finalizeObject();
    #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
    auto mainFnPtr = mEngine->getFunctionAddress(main->getName().str());
    #else
    auto mainFnPtr = mEngine->getFunctionAddress(main->getName());
    #endif
    removeModules(Normal);
    removeModules(Infrequent);
    //#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(12, 0, 0)
    mEngine->removeModule(mainModulePtr);
    mEngine->removeModule(mMainModule);
    //#endif
    return reinterpret_cast<void *>(mainFnPtr);
}

bool CPUDriver::hasExternalFunction(llvm::StringRef functionName) const {
    #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
    return RTDyldMemoryManager::getSymbolAddressInProcess(functionName.str());
    #else
    return RTDyldMemoryManager::getSymbolAddressInProcess(functionName);
    #endif
}

CPUDriver::~CPUDriver() {

}

