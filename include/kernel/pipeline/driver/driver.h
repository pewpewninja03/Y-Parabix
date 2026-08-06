#pragma once

#include <codegen/FunctionTypeBuilder.h>
#include <codegen/LLVMTypeSystemInterface.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <kernel/core/kernel.h>
#include <kernel/core/relationship.h>
#include <util/slab_allocator.h>
#include <llvm/IR/Constants.h>
#include <kernel/illustrator/illustrator.h>
#include <string>
#include <vector>
#include <memory>

namespace llvm { class Function; }
namespace kernel { class KernelBuilder; }
namespace kernel { class PipelineAnalysis; }
namespace kernel { class PipelineBuilder; }
namespace kernel { class ProgramBuilder; }
namespace kernel {template<typename ... Args> class TypedProgramBuilder; }

class CBuilder;
class ParabixObjectCache;

class BaseDriver : public LLVMTypeSystemInterface {
    friend class CBuilder;
    friend class kernel::PipelineAnalysis;
    friend class kernel::PipelineBuilder;
    friend class kernel::ProgramBuilder;
    friend class kernel::Kernel;
    template<typename ... Args> friend class kernel::TypedProgramBuilder;

public:

    using Kernel = kernel::Kernel;
    using Relationship = kernel::Relationship;
    using Bindings = kernel::Bindings;
    using KernelSet = std::vector<std::unique_ptr<Kernel>>;

    void addKernel(not_null<Kernel *> kernel);

    virtual bool hasExternalFunction(const llvm::StringRef functionName) const = 0;

    virtual void generateUncachedKernels() = 0;

    virtual void * finalizeObject(kernel::Kernel * pipeline) = 0;

    virtual ~BaseDriver();

    llvm::LLVMContext & getContext() const final {
        return *mContext.get();
    }

    bool getPreservesKernels() const {
        return mPreservesKernels;
    }

    void setPreserveKernels(const bool value = true) {
        mPreservesKernels = value;
    }

    unsigned getBitBlockWidth() const final;

protected:

    kernel::StreamSet * CreateStreamSet(const unsigned NumElements = 1, const unsigned FieldWidth = 1) noexcept;

    kernel::RepeatingStreamSet * CreateRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t> > &&stringSet, const bool isDynamic = true) noexcept;

    kernel::TruncatedStreamSet * CreateTruncatedStreamSet(const kernel::StreamSet * data) noexcept;

    kernel::RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t> > &&stringSet, const bool isDynamic = true) noexcept;

    kernel::Scalar * CreateScalar(not_null<llvm::Type *> scalarType) noexcept;

    kernel::Scalar * CreateConstant(not_null<llvm::Constant *> value) noexcept;

    kernel::Scalar * CreateCommandLineScalar(kernel::CommandLineScalarType type) noexcept;

    llvm::VectorType * getBitBlockType() const final;

    llvm::VectorType * getStreamTy(const unsigned FieldWidth = 1) final;

    llvm::ArrayType * getStreamSetTy(const unsigned NumElements = 1, const unsigned FieldWidth = 1) final;

protected:

    BaseDriver(std::string && moduleName);

    template <typename ExternalFunctionType>
    void LinkFunction(not_null<Kernel *> kernel, llvm::StringRef name, ExternalFunctionType & functionPtr) const;

    virtual llvm::Function * addLinkFunction(llvm::Module * mod, llvm::StringRef name, llvm::FunctionType * type, void * functionPtr) const = 0;

    kernel::KernelBuilder & getBuilder() {
        return *mBuilder;
    }

protected:

    std::unique_ptr<llvm::LLVMContext>                      mContext;
    llvm::Module * const                                    mMainModule;
    std::unique_ptr<kernel::KernelBuilder>                  mBuilder;
    std::unique_ptr<ParabixObjectCache>                     mObjectCache;

    bool                                                    mPreservesKernels = false;
    KernelSet                                               mUncachedKernel;
    KernelSet                                               mCachedKernel;
    KernelSet                                               mCompiledKernel;
    KernelSet                                               mPreservedKernel;
    std::vector<kernel::RepeatingStreamSet *>               mRepeatingStreamSets;
    SlabAllocator<>                                         mAllocator;
};

template <typename ExternalFunctionType>
void BaseDriver::LinkFunction(not_null<Kernel *> kernel, llvm::StringRef name, ExternalFunctionType & functionPtr) const {
    kernel->link<ExternalFunctionType>(name, functionPtr);
}

