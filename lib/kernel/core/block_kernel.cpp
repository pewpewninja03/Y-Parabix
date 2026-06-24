/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/kernel.h>
#include <kernel/core/block_kernel_compiler.h>

namespace kernel {

// TODO: Break the BlockOrientedKernel into two classes, one with an explicit DoFinal block and another that
// calls the DoBlock method with optional preamble and postamble hooks. By doing so, we can remove the indirect
// branches (or function calls) from the following kernel and simplify the cognitive load for the kernel
// programmer. This is less general than the current method but no evidence that being able to reenter the
// DoBlock method multiple times from the DoFinal block would ever be useful.

#define COMPILER (static_cast<BlockKernelCompiler *>(b.getCompiler()))

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief instantiateKernelCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<KernelCompiler> BlockOrientedKernel::instantiateKernelCompiler(KernelBuilder & /* b */) const {
    return std::make_unique<BlockKernelCompiler>(const_cast<BlockOrientedKernel *>(this));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMultiBlockLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void BlockOrientedKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfBlocks) {
    COMPILER->generateMultiBlockLogic(b, numOfBlocks);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief RepeatDoBlockLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void BlockOrientedKernel::RepeatDoBlockLogic(KernelBuilder & b) {
    COMPILER->generateDefaultFinalBlockMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalBlockMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void BlockOrientedKernel::generateFinalBlockMethod(KernelBuilder & b, llvm::Value * /* remainingItems */) {
    COMPILER->generateDefaultFinalBlockMethod(b);
}

// CONSTRUCTOR
BlockOrientedKernel::BlockOrientedKernel(LLVMTypeSystemInterface & ts,
    std::string && kernelName,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_parameters,
    Bindings && scalar_outputs,
    InternalScalars && internal_scalars,
    const unsigned flags)
: MultiBlockKernel(ts,
    TypeId::BlockOriented,
    std::move(kernelName),
    std::move(stream_inputs),
    std::move(stream_outputs),
    std::move(scalar_parameters),
    std::move(scalar_outputs),
    std::move(internal_scalars),
    flags) {

}


}
