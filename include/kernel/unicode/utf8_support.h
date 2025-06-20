/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>

namespace kernel {

//
// Compute a stream marking UTF-8 character positions.   Each valid
// character is marked at the position of its final UTF-8 byte.
// If the optional linebreak parameter is included, also include
// any positions marked in this stream, including a possible extra
// bit just past EOF if the file is unterminated.
//
class UTF8_index : public pablo::PabloKernel {
public:
    UTF8_index(LLVMTypeSystemInterface & ts, StreamSet * Source, StreamSet * u8index, StreamSet * linebreak = nullptr);
protected:
    void generatePabloMethod() override;
};

//  Given selected UTF-8 characters identified in the marks stream, and
//  a u8index stream marking the last position of each UTF-8 character,
//  produce a stream of UTF-8 character spans, in which each position of
//  marked UTF-8 characters are identified with 1 bits.   The marks
//  stream may be marking the first byte of each UTF-8 sequence in the
//  BitMovementMode::Advance mode or the last byte of each UTF-8 sequence
//  in the BitMovementMode::LookAhead mode (default).
class U8Spans : public pablo::PabloKernel {
public:
    U8Spans(LLVMTypeSystemInterface & ts, StreamSet * marks, StreamSet * u8index, StreamSet * spans,
            pablo::BitMovementMode m = pablo::BitMovementMode::LookAhead);
protected:
    void generatePabloMethod() override;
private:
    pablo::BitMovementMode mBitMovement;
};
}

