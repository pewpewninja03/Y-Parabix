/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
#include <pablo/pablo_kernel.h>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

//
//  Shift bit streams forward by 1 or more positions given by shiftAmount.
//  Output streams will be the same length as input streams; bits shifted
//  past the end of the input stream are lost.  Zeroes are shifted in to
//  the beginning of each input stream.
//
class ShiftForward final : public pablo::PabloKernel {
public:
    ShiftForward(LLVMTypeSystemInterface & ts, StreamSet * inputs, StreamSet * shifted, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

//
//  Shift bit streams backward by 1 or more positions given by shiftAmount.
//  Output streams will be the same length as input streams; bits shifted
//  past the beginning of the input stream are lost.  Zeroes are shifted in
//  to the end of each input stream.
//
class ShiftBack final : public pablo::PabloKernel {
public:
    ShiftBack(LLVMTypeSystemInterface & ts, StreamSet * inputs, StreamSet * shifted, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

//
//  Shift bit streams forward according to an index stream.   The index stream
//  marks positions that participate in the shift.  Bits at positions marked
//  by 1 bits in the index stream are shifted forward to the nth following
//  index position, where n is the shiftAmount.   Output streams are the
//  same length as input streams.   Zeroes are shifted in and bits shifted
//  past the last marked index position are lost.   All positions not marked
//  by one bits in the index stream are zeroed out.
//
class IndexedAdvance final : public pablo::PabloKernel {
public:
    IndexedAdvance(LLVMTypeSystemInterface & ts, StreamSet * index, StreamSet * inputs, StreamSet * shifted, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

//
//  Shift a marker bit stream backward one position according to an index stream.
//  The index stream marks positions that participate in the shift.  Bits at
//  positions marked by 1 bits in the index stream are shifted backward to
//  the immediately preceding index position.   The shifted output stream is the
//  same length as the input stream.   Input stream bits at the first marker
//  position are shifted out, while zeroes are shifted in to the last
//  marker position.  All positions not marked by one bits in the index
//  stream are zeroed out.
//
class IndexedShiftBack final : public MultiBlockKernel {
public:
    IndexedShiftBack(LLVMTypeSystemInterface & ts,
                     StreamSet * index, StreamSet * markers, StreamSet * shifted);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}

