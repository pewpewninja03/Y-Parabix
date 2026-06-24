/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/core/kernel.h>
#include <kernel/core/streamset.h>
#include <kernel/core/kernel_builder.h>

using namespace llvm;

namespace kernel {

using PortType = Kernel::PortType;
using StreamSetPort = Kernel::StreamSetPort;
using AttrId = Attribute::KindId;
using RateId = ProcessingRate::KindId;
using Rational = ProcessingRate::Rational;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void MultiBlockKernel::generateKernelMethod(KernelBuilder & b) {
    generateMultiBlockLogic(b, b.getNumOfStrides());
}

// MULTI-BLOCK KERNEL CONSTRUCTOR
MultiBlockKernel::MultiBlockKernel(LLVMTypeSystemInterface & ts,
    std::string && kernelName,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_parameters,
    Bindings && scalar_outputs,
    InternalScalars && internal_scalars, unsigned flags)
: MultiBlockKernel(ts,
    TypeId::MultiBlock,
    std::move(kernelName),
    std::move(stream_inputs),
    std::move(stream_outputs),
    std::move(scalar_parameters),
    std::move(scalar_outputs),
    std::move(internal_scalars),
    flags) {

}

MultiBlockKernel::MultiBlockKernel(LLVMTypeSystemInterface & ts,
    const TypeId typeId,
    std::string && kernelName,
    Bindings && stream_inputs,
    Bindings && stream_outputs,
    Bindings && scalar_parameters,
    Bindings && scalar_outputs,
    InternalScalars && internal_scalars,
    unsigned flags)
: Kernel(ts, typeId,
     std::move(kernelName),
     std::move(stream_inputs),
     std::move(stream_outputs),
     std::move(scalar_parameters),
     std::move(scalar_outputs),
     std::move(internal_scalars),
     CompilationStatus::FullyInitialized, flags) {

}

}
