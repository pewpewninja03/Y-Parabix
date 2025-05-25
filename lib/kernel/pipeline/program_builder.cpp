#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/program_builder.h>
#include "compiler/pipeline_compiler.hpp"
#include <kernel/core/kernel_builder.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Timer.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <toolchain/toolchain.h>

using namespace llvm;

#define ADD_CL_SCALAR(Id,Type) \
    mTarget->mInputScalars.emplace_back(Id, mDriver.CreateCommandLineScalar(CommandLineScalarType::Type))

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief compile()
 ** ------------------------------------------------------------------------------------------------------------- */
void * ProgramBuilder::compile() {
    Kernel * const kernel = makeKernel();
    if (LLVM_UNLIKELY(kernel == nullptr)) {
        report_fatal_error("Main pipeline contains no kernels nor function calls.");
    }
    NamedRegionTimer T(kernel->getSignature(), kernel->getName(), "pipeline", "Pipeline Compilation", codegen::TimeKernelsIsEnabled);
    return compileKernel(kernel);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief compileKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void * ProgramBuilder::compileKernel(Kernel * const kernel) {
    mDriver.addKernel(kernel);
    mDriver.generateUncachedKernels();
    return mDriver.finalizeObject(kernel);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * ProgramBuilder::makeKernel() {
    if (codegen::EnableDynamicMultithreading) {
        ADD_CL_SCALAR(MINIMUM_NUM_OF_THREADS, MinThreadCount);
        ADD_CL_SCALAR(MAXIMUM_NUM_OF_THREADS, MaxThreadCount);
        ADD_CL_SCALAR(DYNAMIC_MULTITHREADING_SEGMENT_PERIOD, DynamicMultithreadingPeriod);
        ADD_CL_SCALAR(DYNAMIC_MULTITHREADING_ADDITIONAL_THREAD_SYNCHRONIZATION_THRESHOLD, DynamicMultithreadingAddSynchronizationThreshold);
        ADD_CL_SCALAR(DYNAMIC_MULTITHREADING_REMOVE_THREAD_SYNCHRONIZATION_THRESHOLD, DynamicMultithreadingRemoveSynchronizationThreshold);
    } else {
        ADD_CL_SCALAR(MAXIMUM_NUM_OF_THREADS, MaxThreadCount);
    }
    // ADD_CL_SCALAR(BUFFER_SEGMENT_LENGTH, BufferSegmentLength);
    mDriver.generateUncachedKernels();
    return PipelineBuilder::makeKernel();

}

ProgramBuilder::ProgramBuilder(BaseDriver & driver, PipelineKernel * const kernel)
: PipelineBuilder(driver, kernel) {

}

}
