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
#include <kernel/scan/base.h>

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

/*   Applying the given spreadmask, spread a filter mask, filling in 1 bits at the
     inserted positions.  */
void ExpandFilter(PipelineBuilder & P,
                  StreamSet * spreadMask, StreamSet * filter, StreamSet * expanded);

void MergeByMask(PipelineBuilder & P, StreamSet * mask, StreamSet * a, StreamSet *b, StreamSet * merged);

/*  Create a spread mask for inserting a single item into a stream for each position
    in the given insertion mask that is nonzero.   The insertion mask may be
    a bixnum; in this case the spread mask will have a single insert position
    for each position at which the bixnum is greater than zero.    */

enum class InsertPosition {Before, After};

void UnitInsertionSpreadMask(PipelineBuilder & P,
                             StreamSet * insertion_mask,
                             StreamSet * spread_mask,
                             InsertPosition p = InsertPosition::Before);

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

void InsertionSpreadMask(PipelineBuilder & P,
                         StreamSet * bixNumInsertCount,
                         StreamSet * spread_mask,
                         InsertPosition p = InsertPosition::Before);

class UnitInsertionSpreadMaskKernel final : public BlockOrientedKernel {
public:
    UnitInsertionSpreadMaskKernel(LLVMTypeSystemInterface & ts,
                                 StreamSet * insertion_mask, StreamSet * spread_mask, InsertPosition p = InsertPosition::Before);
protected:
    const unsigned pack_width = 64;
    void generateDoBlockMethod(KernelBuilder & b) override;
    void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * const remainingBytes) override;
private:
    const InsertPosition mInsertPos;
};

class InsertionSpreadMaskKernel final : public TwoLevelScanKernel {
public:
    InsertionSpreadMaskKernel(LLVMTypeSystemInterface & ts,
                              StreamSet * insertion_counts, StreamSet * spread_mask,
                              InsertPosition p = InsertPosition::Before);
protected:
    static const unsigned ScanWordWidth = 64;
    void initialize(KernelBuilder & b) override;
    void wordPrologueLogic(KernelBuilder & b,
                           llvm::Value * absWordPos,
                           std::vector<llvm::Value *> indexWord,
                           std::vector<llvm::Value *> & loopVars) override;
    void generateProcessingLogic(KernelBuilder & b,
                                 llvm::Value * absItemPos,
                                 std::vector<llvm::Value *> & loopVars) override;
    void finalize(KernelBuilder & b, std::vector<llvm::Value *> & loopVarFinalValues) override;

private:
    const unsigned mBixBits;
    const unsigned mExpansionWidth;
    const InsertPosition mInsertPos;
    enum LoopVars {bn_processed = 0, sm_produced = 1, sm_pending = 2};
    // Values initialized in scan word prologue for use
    // throughout item processing logic.
    std::vector<llvm::Value *> mMasks;
};

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

