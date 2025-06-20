/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <pablo/pablo_toolchain.h>
#include <kernel/pipeline/pipeline_builder.h>

using StreamSet = kernel::StreamSet;
using PipelineBuilder = kernel::PipelineBuilder;

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

//
//  Although NFC generally compresses text by replacing decomposed
//  character sequences by their precomposed equivalents, there are
//  two cases where expansion is required.   The first is that any
//  precomposed characters with non-starter decompositions must be
//  decomposed and left in decomposed form.   The second is that
//  when the UTF-8 length of a precomposed character is longer than
//  the starter of its decomposed sequence.   The NFC_Initial_Insertion
//  kernel produces an insertion bixnum that indicates the maximum
//  insertion required at each position.
//
class NFC_Initial_Insertion : public pablo::PabloKernel {
public:
    NFC_Initial_Insertion
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * InsertionBixNum);
protected:
    void generatePabloMethod() override;
};

//
//  Apply the logic of nonstarter decomposition to any relevant
//  precomposed characters.   This kernel assumes that the Basis
//  bit streams have had sufficient zeroes inserted for the decomposed
//  sequence.
//
class NonStarterDecomposition : public pablo::PabloKernel {
public:
    NonStarterDecomposition
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * NSD_Basis);
protected:
    void generatePabloMethod() override;
};

//
//  Excluded composites are those that are always normalized to
//  decomposed form with both NFD and NFC.
//
class ExcludedCompositeStage : public pablo::PabloKernel {
public:
    ExcludedCompositeStage
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
         StreamSet * SelectMask, StreamSet * EC_Basis);
protected:
    void generatePabloMethod() override;
};

//
//  Short composable sequences are those involving non reorderable
//  characters.   In this case, precomposition is only applied when
//  the characters are adjacent.   This kernel replaces the
//  second character of such short composable sequences with
//  the resulting precomposed character.   In addition a marker
//  bit stream DeletePrior is produced identifying that the
//  previous character must be deleted.
//
class ShortComposableTranslation : public pablo::PabloKernel {
public:
    ShortComposableTranslation
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * DeletePrior, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

//
//  Transform any characters with a singleton decomposition to their
//  canonical form.   In the case of UTF-8, if the singleton decomposition
//  produces a shorter transformed sequence, the extra positions will be
//  marked with zeroes for deletion using FilterByMask.
//
class SingletonCanonicalization : public pablo::PabloKernel {
public:
    SingletonCanonicalization
        (LLVMTypeSystemInterface & ts, StreamSet * Basis,
                                       StreamSet * SelectMask, StreamSet * XfrmBasis);
protected:
    void generatePabloMethod() override;
};

void LongComposablePipeline(PipelineBuilder & P,
                            StreamSet * Basis, StreamSet * ccc_NR,
                            StreamSet * FinalBasis, StreamSet * DeletionMask);
