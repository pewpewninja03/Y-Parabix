#pragma once

#include <kernel/core/kernel.h>
#include <type_traits>
#include <functional>
#include <kernel/pipeline/driver/driver.h>
#include <boost/container/flat_map.hpp>
#include <kernel/illustrator/illustrator_binding.h>

namespace llvm { class Value; }

class BaseDriver;

namespace kernel {

const static std::string INITIALIZE_FUNCTION_POINTER_SUFFIX = "_IFP";
const static std::string ALLOCATE_SHARED_INTERNAL_STREAMSETS_FUNCTION_POINTER_SUFFIX = "_AFP";
const static std::string INITIALIZE_THREAD_LOCAL_FUNCTION_POINTER_SUFFIX = "_ITFP";
const static std::string ALLOCATE_THREAD_LOCAL_INTERNAL_STREAMSETS_FUNCTION_POINTER_SUFFIX = "_ATFP";
const static std::string DO_SEGMENT_FUNCTION_POINTER_SUFFIX = "_SFP";
const static std::string FINALIZE_THREAD_LOCAL_FUNCTION_POINTER_SUFFIX = "_FTIP";
const static std::string FINALIZE_FUNCTION_POINTER_SUFFIX = "_FIP";

class PipelineAnalysis;
class PipelineBuilder;
class PipelineCompiler;

class PipelineKernel final : public Kernel {
    friend class Kernel;
    friend class PipelineCompiler;
    friend class PipelineAnalysis;
    friend class PipelineBuilder;
    friend class ProgramBuilder;
    template<typename ... Args>
    friend class TypedProgramBuilder;
    friend class ::BaseDriver;
public:

    static bool classof(const Kernel * const k) {
        return k->getTypeId() == TypeId::Pipeline;
    }

    static bool classof(const void *) { return false; }

public:

    using Scalars = std::vector<Scalar *>;

    using Relationships = std::vector<const Relationship *>;

    enum KernelBindingFlag {
        None = 0
        , Family = 1
    };

    struct KernelBinding {
        Kernel * Object = nullptr;
        unsigned Flags = KernelBindingFlag::None;

        bool isFamilyCall() const {
            return (Flags & KernelBindingFlag::Family) != 0;
        }

        KernelBinding(Kernel * kernel, unsigned flags)
        : Object(kernel)
        , Flags(flags) {

        }
    };

    using Kernels = std::vector<KernelBinding>;

    struct CallBinding {
        std::string Name;
        llvm::FunctionType * Type;
        void * FunctionPointer;
        Scalars Args;

        mutable llvm::Constant * Callee;

        CallBinding(std::string Name, llvm::FunctionType * Type, void * FunctionPointer, std::initializer_list<Scalar *> && Args)
        : Name(std::move(Name)), Type(Type), FunctionPointer(FunctionPointer), Args(Args.begin(), Args.end()), Callee(nullptr) { }    
    };

    using CallBindings = std::vector<CallBinding>;

    using IllustratorBindings = std::vector<IllustratorBinding>;

    using LengthAssertion = std::array<const StreamSet *, 2>;

    using LengthAssertions = std::vector<LengthAssertion>;

    bool hasSignature() const final { return true; }

    bool isCachable() const override;

    LLVM_READNONE bool allocatesInternalStreamSets() const final;

    void setInputStreamSetAt(const unsigned i, StreamSet * const value) final;

    void setOutputStreamSetAt(const unsigned i, StreamSet * const value) final;

    void setInputScalarAt(const unsigned i, Scalar * const value) final;

    void setOutputScalarAt(const unsigned i, Scalar * const value) final;

    llvm::StringRef getSignature() const final {
        return mSignature;
    }

    const Kernels & getKernels() const {
        return mKernels;
    }

    const CallBindings & getCallBindings() const {
        return mCallBindings;
    }

    const LengthAssertions & getLengthAssertions() const {
        return mLengthAssertions;
    }

    void addKernelDeclarations(KernelBuilder & b) final;

    std::unique_ptr<KernelCompiler> instantiateKernelCompiler(KernelBuilder & b) const final;

    ~PipelineKernel() override;

    llvm::Function * addOrDeclareMainFunction(KernelBuilder & b, const MainMethodGenerationType method) const final;

protected:

    PipelineKernel(LLVMTypeSystemInterface & ts,
                   std::string && signature,
                   const unsigned numOfKernelFamilyCalls,
                   Kernels && kernels, CallBindings && callBindings,
                   Bindings && stream_inputs, Bindings && stream_outputs,
                   Bindings && scalar_inputs, Bindings && scalar_outputs,
                   Relationships && internallyGenerated,
                   LengthAssertions && lengthAssertions);

    PipelineKernel(LLVMTypeSystemInterface & ts,
                   std::string && signature,
                   AttributeSet && attributes,
                   Bindings && stream_inputs, Bindings && stream_outputs,
                   Bindings && scalar_inputs, Bindings && scalar_outputs);

    static std::string annotateSignatureWithPipelineFlags(std::string && name);

    static std::string makePipelineHashName(const std::string & signature);

    unsigned getNumOfNestedKernelFamilyCalls() const override;

private:

    struct Internal {};

    PipelineKernel(Internal, LLVMTypeSystemInterface & ts,
                   std::string && signature,
                   const unsigned numOfKernelFamilyCalls,
                   Kernels && kernels, CallBindings && callBindings,
                   Bindings && stream_inputs, Bindings && stream_outputs,
                   Bindings && scalar_inputs, Bindings && scalar_outputs,
                   Relationships && internallyGenerated,
                   LengthAssertions && lengthAssertions);


private:

    void addAdditionalInitializationArgTypes(KernelBuilder & b, InitArgTypes & argTypes) const final;

    void recursivelyConstructFamilyKernels(KernelBuilder & b, InitArgs & args, ParamMap & params, NestedStateObjs & toFree) const final;

    void recursivelyListFamilyKernels(llvm::raw_ostream & familyName) const final;

    void linkExternalMethods(KernelBuilder & b) final;

    void generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, llvm::Value * expectedNumOfStrides) final;

    void generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, llvm::Value * expectedNumOfStrides) final;

    void addAdditionalFunctions(KernelBuilder & b) final;

    void addInternalProperties(KernelBuilder & b) final;

    void generateInitializeMethod(KernelBuilder & b) final;

    void generateInitializeThreadLocalMethod(KernelBuilder & b) final;

    void generateKernelMethod(KernelBuilder & b) final;

    void generateFinalizeThreadLocalMethod(KernelBuilder & b) final;

    void generateFinalizeMethod(KernelBuilder & b) final;

    void addOptimizationPasses(KernelBuilder & b, SelectedOptimizationPasses & passes) const final;

protected:

    bool hasInternallyGeneratedStreamSets() const final {
        return !mInternallyGeneratedStreamSets.empty();
    }

    const Relationships & getInternallyGeneratedStreamSets() const final {
        return mInternallyGeneratedStreamSets;
    }

    void writeInternallyGeneratedStreamSetScaleVector(const Relationships & R, MetadataScaleVector & V, const size_t scale) const final;

    ParamMap::PairEntry createRepeatingStreamSet(KernelBuilder & b, const RepeatingStreamSet * streamSet, const size_t maxStrideLength) const;

    const IllustratorBindings & getIllustratorBindings() const {
        return mIllustratorBindings;
    }

protected:

    unsigned                            mNumOfKernelFamilyCalls;
    std::string                         mSignature;
    Relationships                       mInternallyGeneratedStreamSets;
    Kernels                             mKernels;
    CallBindings                        mCallBindings;
    LengthAssertions                    mLengthAssertions;
    IllustratorBindings                 mIllustratorBindings;

};

}

