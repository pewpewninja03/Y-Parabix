/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>

using StreamSet = kernel::StreamSet;
//
//  Hangul composables are individual Hangul L, V and T characters that
//  can be combined to produce precomposed LV or LVT combinations.   Note
//  also that LV combinations are composables, as they can potentially be
//  combined with trailing T character.
//  The Hangul_Composables kernel returns a set of 4 streams respectively
//  marking composable L, V, LV and T characters given a set of basis
//  streams.
//

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
//  precomposed characters.  
//
class Hangul_Composition : public pablo::PabloKernel {
public:
    Hangul_Composition(LLVMTypeSystemInterface & ts,
                          StreamSet * Basis, StreamSet * L_V_T_Composables, 
                          StreamSet * Output_Basis, StreamSet * SelectionMask);
protected:
    void generatePabloMethod() override;
};


