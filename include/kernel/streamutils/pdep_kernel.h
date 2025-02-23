/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
#include <llvm/IR/Value.h>
#include <string>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/pipeline/pipeline_builder.h>

namespace kernel {

enum class StreamExpandOptimization {None, NullCheck};

/*  SpreadByMask - spread out the bits of input streams according to a mask.
    Outputs are 1-to-1 with mask positions.   For a mask bit of 0, the corresponding
    output bits are always 0.   Otherwise, the nth 1 bit in the mask selects
    an input bit from position n in the input stream.

    Input stream:    abcdef
    Mask stream:     ...1.1...111..1..     . represents 0
    Output stream:   ...a.b...cde..f..     . represents 0

    The input stream is processed at the Popcount rate of the mask stream.

    The number of streams to process is governed by the size of the output stream set.
    The input streams to process are selected sequentially from the stream set toSpread,
    starting from the position indicated by the streamOffset value.

    If zeroExtend is set, the input stream will be zeroExtended as required
    to match the mask stream.

    Depending on the expected density of the mask stream, different optimizations
    can be chosen.   The expansionFieldWidth parameter may also affect performance.

    SpreadByMask employs two kernels in sequence:  StreamExpandKernel followed by
    FieldDepositKernel.

 */
void SpreadByMask(PipelineBuilder & P,
                  StreamSet * mask, StreamSet * toSpread, StreamSet * outputs,
                  unsigned streamOffset = 0,
                  bool zeroExtend = false,
                  StreamExpandOptimization opt = StreamExpandOptimization::None,
                  unsigned expansionFieldWidth = 64,
                  ProcessingRateProbabilityDistribution itemsPerOutputUnit = GammaDistribution(5.0f, 0.1f));

void MergeByMask(PipelineBuilder & P, StreamSet * mask, StreamSet * a, StreamSet *b, StreamSet * merged);

/*  Create a spread mask for inserting a single item into a stream for each position
    in the given insertion mask that is nonzero.   The insertion mask may be
    a bixnum; in this case the spread mask will have a single insert position
    for each position at which the bixnum is greater than zero.    */

enum class InsertPosition {Before, After};

StreamSet * UnitInsertionSpreadMask(PipelineBuilder & P,
                                    StreamSet * insertion_mask,
                                    InsertPosition p = InsertPosition::Before,
                                    ProcessingRateProbabilityDistribution insertionProbabilityDistribution = UniformDistribution());


/*   Prepare a spread mask for inserting data into bit streams.
     At each stream position, a bixnum encodes the number of items
     to be inserted at that position.   The resulting spread mask
     will have a 1 bit for every original position in the stream,
     plus a 0 bit for every position at which data is inserted.
     The total number of 0 bits is thus the sum of the numbers
     of items to be inserted at each position.   Let the number
     of positions to be inserted at a given position n.   For each
     source position sequentially, the spread mask will consist
     of either n 0 bits followed by a 1 bit (InsertPostion::Before)
     or a 1 bit followed by n 0 bits (InsertPosition::After).    */

StreamSet * InsertionSpreadMask(PipelineBuilder & P,
                                StreamSet * bixNumInsertCount,
                                InsertPosition p = InsertPosition::Before,
                                ProcessingRateProbabilityDistribution itemsPerOutputUnit = UniformDistribution(),
                                ProcessingRateProbabilityDistribution expansionRate = UniformDistribution());



class ByteCombine final : public MultiBlockKernel {
public:
    ByteCombine(LLVMTypeSystemInterface & ts,
                StreamSet * const byteStream1,
                StreamSet * const byteStream2,
                StreamSet * const outputBytes);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfBlocks) override;
};

/* The following kernels are used by SpreadByMask internally. */

class StreamExpandKernel final : public MultiBlockKernel {
public:
    StreamExpandKernel(LLVMTypeSystemInterface & ts,
                       StreamSet * mask,
                       StreamSet * source,
                       StreamSet * expanded,
                       Scalar * base,
                       bool zeroExtend,
                       const StreamExpandOptimization = StreamExpandOptimization::None,
                       const unsigned FieldWidth = sizeof(size_t) * 8,
                       ProcessingRateProbabilityDistribution itemsPerOutputUnitProbability = UniformDistribution());
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfBlocks) override;
private:
    const unsigned mFieldWidth;
    const unsigned mSelectedStreamCount;
    const StreamExpandOptimization mOptimization;
};

/**********************************/
class StreamMergeKernel final : public MultiBlockKernel {
public:
    StreamMergeKernel(LLVMTypeSystemInterface & ts,
                       StreamSet * mask,
                       StreamSet * source1,
                       StreamSet * source2,
                       StreamSet * merged,
                       Scalar * base,
                       const unsigned FieldWidth = sizeof(size_t) * 8);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfBlocks) override;
private:
    const unsigned mFieldWidth;
    const unsigned mSelectedStreamCount;
};

/*******************************************************/
class FieldDepositKernel final : public MultiBlockKernel {
public:
    FieldDepositKernel(LLVMTypeSystemInterface & ts, StreamSet * mask, StreamSet * input, StreamSet * output, const unsigned fieldWidth = sizeof(size_t) * 8);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const unsigned mFieldWidth;
    const unsigned mStreamCount;
};


class PDEPFieldDepositKernel final : public MultiBlockKernel {
public:
    PDEPFieldDepositKernel(LLVMTypeSystemInterface & ts, StreamSet * mask, StreamSet * expandedA, StreamSet * outputs, const unsigned fieldWidth = sizeof(size_t) * 8);
protected:
    void generateMultiBlockLogic(KernelBuilder & kb, llvm::Value * const numOfStrides) override;
private:
    const unsigned mPDEPWidth;
    const unsigned mStreamCount;
};

/*
 Swizzled PDEP kernel - now deprecated.

 Conceptually, given an unbounded input stream set of k streams and a marker stream, this kernel uses the
 Parallel Bits Deposit (PDEP) instruction to copy the input items from the i-th input stream to the i-th
 output stream the positions indicated by the marker bits. All other output items are set to zero. E.g.,

 SOURCE >  abcdefgh i0000000 00000000 00000000
 MARKER >  ...1.1.1 .....11. ..1...1. ...1.1..
 OUTPUT >  ...a.b.c .....de. ..f...g. ...h.i..

 The complicating factor of this Kernel is that it assumes the input streams are *swizzled*. I.e., it
 "divides" each block of the marker stream into k elements, M_1 ... M_k, and applies the PDEP operation
 using M_i to the each of the k elements in the i-th input (swizzled) stream.

 CONCEPTUAL VIEW OF INPUT STREAM SET                    ACTUAL LAYOUT OF INPUT STREAM SET

 STREAM 0  abcde...  fg......  hijklm..  nopqrst.     SWIZZLE 0  abcde...  uvwxy...  OPQRS...  89abc...
 STREAM 1  uvwxy...  zA......  BCDEFG..  HIJKLMN.     SWIZZLE 1  fg......  zA......  TU......  de......
 STREAM 2  OPQRS...  TU......  VWXYZ0..  1234567.     SWIZZLE 2  hijklm..  BCDEFG..  VWXYZ0..  fghijk..
 STREAM 3  89abc...  de......  fghijk..  lmnopqr.     SWIZZLE 3  nopqrst.  HIJKLMN.  1234567.  lmnopqr.


 NOTE: this kernel does *NOT* unswizzle the output. This will eventually be the responsibility of the
 pipeline to ensure it is done when needed.

 */
class PDEPkernel final : public MultiBlockKernel {
public:
    PDEPkernel(LLVMTypeSystemInterface & ts, const unsigned swizzleFactor = 4, std::string name = "PDEP");
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;
private:
    const unsigned mSwizzleFactor;
};

class ByteSpreadByMaskKernel final : public MultiBlockKernel {
public:
    ByteSpreadByMaskKernel(LLVMTypeSystemInterface & b, StreamSet * const byteStream, StreamSet * const spread, StreamSet * const output, Scalar * streamOffset = nullptr);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

class ByteReplaceByMask final : public MultiBlockKernel {
public:
    ByteReplaceByMask(LLVMTypeSystemInterface & b, StreamSet * mask, StreamSet * ToFill, StreamSet * Filler, StreamSet * Filled);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}

