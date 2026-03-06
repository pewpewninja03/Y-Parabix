#include <kernel/pipeline/optimizationbranch.h>
#include "optimizationbranch_compiler.hpp"
#include <kernel/core/kernel_builder.h>

namespace kernel {

#define COMPILER (reinterpret_cast<OptimizationBranchCompiler *>(b.getCompiler()))

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief instantiateKernelCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<KernelCompiler> OptimizationBranch::instantiateKernelCompiler(KernelBuilder & b) const {
    return std::make_unique<OptimizationBranchCompiler>(b, const_cast<OptimizationBranch *>(this));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addKernelDeclarations
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::addKernelDeclarations(KernelBuilder & b) {
    mAllZeroKernel->addKernelDeclarations(b);
    mNonZeroKernel->addKernelDeclarations(b);
    Kernel::addKernelDeclarations(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::addInternalProperties(KernelBuilder & b) {
    COMPILER->addBranchProperties(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateSharedInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateAllocateSharedInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    COMPILER->generateAllocateSharedInternalStreamSetsMethod(b, expectedNumOfStrides);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateAllocateThreadLocalInternalStreamSetsMethod(KernelBuilder & b, Value * const expectedNumOfStrides) {
    COMPILER->generateAllocateThreadLocalInternalStreamSetsMethod(b, expectedNumOfStrides);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateInitializeMethod(KernelBuilder & b) {
    COMPILER->generateInitializeMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateInitializeThreadLocalMethod(KernelBuilder & b) {
    COMPILER->generateInitializeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateDoSegmentMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateKernelMethod(KernelBuilder & b) {
    COMPILER->generateKernelMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateFinalizeThreadLocalMethod(KernelBuilder & b) {
    COMPILER->generateFinalizeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void OptimizationBranch::generateFinalizeMethod(KernelBuilder & b) {
    COMPILER->generateFinalizeMethod(b);
}

OptimizationBranch::OptimizationBranch(LLVMTypeSystemInterface & ts,
    std::string && signature,
    not_null<Relationship *> condition,
    Kernel * const nonZeroKernel,
    Kernel * const allZeroKernel,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_inputs,
    Bindings && scalar_outputs)
: Kernel(ts, TypeId::OptimizationBranch, std::move(signature),
         std::move(stream_inputs), std::move(stream_outputs),
         std::move(scalar_inputs), std::move(scalar_outputs),
        {} /* Internal scalars are generated by the OptimizationBranchCompiler */,
        CompilationStatus::FullyInitialized,
        nonZeroKernel->getKernelFlags() | allZeroKernel->getKernelFlags())
, mCondition(condition)
, mNonZeroKernel(nonZeroKernel)
, mAllZeroKernel(allZeroKernel) {

}

}
