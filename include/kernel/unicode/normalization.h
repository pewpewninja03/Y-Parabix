/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>

using StreamSet = kernel::StreamSet;

// Hangul Composition Kernels for NFC (See Unicode section 3.12).
//
//  Hangul composables are individual Hangul L, V and T characters that
//  can be combined to produce precomposed LV or LVT combinations.   Note
//  also that LV combinations are composables, as they can potentially be
//  combined with trailing T character.
//  The Hangul_Composables kernel returns a set of 4 streams respectively
//  marking composable L, V, LV and T characters given a set of basis
//  streams.
//
//  This kernel may be used with a basis set of 21 bit streams for
//  Unicode-indexed data, 16 bit streams for UTF-16 indexed data or
//  of 8 bit streams for UTF-8 indexed data.   For UTF-8, calculations
//  are produced at the position of the first code unit with LookAhead
//  type bit movement (the default).

class Hangul_Composables : public pablo::PabloKernel {
public:
    enum Kind : unsigned {L, V, LV, T, Count};

    Hangul_Composables(LLVMTypeSystemInterface & ts,
                       StreamSet * Basis, StreamSet * L_V_T_Composables,
                       pablo::BitMovementMode m = pablo::BitMovementMode::LookAhead);
protected:
    void generatePabloMethod() override;
private:
    pablo::BitMovementMode mBitMovement;
};

//
//  Transform sequences of Hangul composable characters into the equivalent
//  precomposed characters.  Three character L-V-T sequences are
//  transformed into the corresponding precomposed character at the V
//  position.   A sequence consisting of an LV precomposed character
//  followed by a T is transformed int the combined LVT character at
//  the LV position.   Finally, in the case of a two-character L-V
//  sequence not followed by a T, a transformed LV character is produced
//  at the LV position.
//
//  As a result of this transformation, leading L and trailing T
//  characters become redundant.   This is reflected by zeroing out
//  such positions in the produced SelectionMask bit stream.
//  The produced Output Basis will still contain the leading L and
//  trailing T characters, but they may be subsequently removed using
//  a FilterByMask operation with the SelectionMask.
//
//  This kernel may be used with basis stream sets having 21 streams
//  (Unicode indexing), 16 streams (UTF-16 indexing) or 8 streams
//  (UTF-8 indexing).  In the case of UTF-8 indexing, this kernel
//  requires Hangul that the Hangul Composables stream be produced
//  at the first byte position of each such composable character.
//
class Hangul_Composition : public pablo::PabloKernel {
public:
    Hangul_Composition(LLVMTypeSystemInterface & ts,
                       StreamSet * Basis, StreamSet * Hangul_Composables,
                       StreamSet * Output_Basis, StreamSet * SelectionMask);
protected:
    void generatePabloMethod() override;
};


