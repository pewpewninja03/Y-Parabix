/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

/*
   The UntilNkernel copies the initial prefix of a marker bitstream
   up to and including the position of the Nth one bit.   In the mode
   ZeroAfterN, the remainder of the output bit stream is zeroed out,
   up to the size of the input marker stream.   In the mode TerminateAtN,
   the output stream is terminated at the position of the Nth one bit.
*/

namespace kernel {

class UntilNkernel final : public MultiBlockKernel {
public:
    enum Mode {ZeroAfterN, TerminateAtN, ReportAcceptedLengthAtAndBeforeN};
    UntilNkernel(LLVMTypeSystemInterface & ts, Scalar * N, StreamSet * Markers, StreamSet * FirstN,
                 UntilNkernel::Mode m = UntilNkernel::Mode::ZeroAfterN);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;
    bool mUseInOut;
    UntilNkernel::Mode mMode;
};

}
