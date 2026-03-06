#pragma once

#include <kernel/core/kernel.h>

namespace llvm { class Value; }

namespace kernel {

class PipelineCompiler;
class OptimizationBranchCompiler;

class OptimizationBranch final : public Kernel {
    friend class OptimizationBranchBuilder;
    friend class OptimizationBranchCompiler;
    friend class PipelineCompiler;
public:

    static bool classof(const Kernel * const k) {
        switch (k->getTypeId()) {
            case TypeId::OptimizationBranch:
                return true;
            default:
                return false;
        }
    }

    const Kernel * getAllZeroKernel() const {
        return mAllZeroKernel;
    }

    const Kernel * getNonZeroKernel() const {
        return mNonZeroKernel;
    }

    Relationship * getCondition() const {
        return mCondition;
    }

    std::unique_ptr<KernelCompiler> instantiateKernelCompiler(KernelBuilder & b) const final;

protected:

    OptimizationBranch(LLVMTypeSystemInterface & ts,
                       std::string && signature,
                       not_null<Relationship *> condition,
                       Kernel * const nonZeroKernel,
                       Kernel * const allZeroKernel,
                       Bindings && stream_inputs,
                       Bindings && stream_outputs,
                       Bindings && scalar_inputs,
                       Bindings && scalar_outputs);

    void addKernelDeclarations(KernelBuilder & b) override;

    void addInternalProperties(KernelBuilder & b) override;

    void generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, llvm::Value * expectedNumOfStrides) override;

    void generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, llvm::Value * expectedNumOfStrides) override;

    void generateInitializeMethod(KernelBuilder & b) override;

    void generateInitializeThreadLocalMethod(KernelBuilder & b) override;

    void generateKernelMethod(KernelBuilder & b) override;

    void generateFinalizeThreadLocalMethod(KernelBuilder & b) override;

    void generateFinalizeMethod(KernelBuilder & b) override;

private:

    Relationship * const                        mCondition;
    Kernel * const                              mNonZeroKernel;
    Kernel * const                              mAllZeroKernel;

};

}

