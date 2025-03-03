/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/sorting.h>
#include <kernel/streamutils/run_index.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_var.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::ceil_log2;
using namespace kernel;
using namespace pablo;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)


//
//  Given a bit stream marking Runs to be sorted and a
//  SortOrder BixNum to be used within each Run, find any
//  violations within the Runs and mark them at the Run end
//  position.
//
class Misorder_Check : public pablo::PabloKernel {
public:
    Misorder_Check(LLVMTypeSystemInterface & ts, StreamSet * Runs, StreamSet * SortOrder, StreamSet * Misordered);
protected:
    void generatePabloMethod() override;
};

Misorder_Check::Misorder_Check (LLVMTypeSystemInterface & ts, StreamSet * Runs, StreamSet * SortOrder, StreamSet * Misordered)
: PabloKernel(ts, "Misorder_Check_" + SortOrder->shapeString(),
// inputs
{Binding{"Runs", Runs, FixedRate(1), LookAhead(1)}, Binding{"SortOrder", SortOrder, FixedRate(1), LookAhead(1)}},
// output
{Binding{"Misordered", Misordered}}) {
}

void Misorder_Check::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * Runs = getInputStreamSet("Runs")[0];
    BixNum SortOrder = getInputStreamSet("SortOrder");
    BixNum SortOrder_ahead(SortOrder.size());
    for (unsigned i = 0; i < SortOrder.size(); i++) {
        SortOrder_ahead[i] = pb.createLookahead(SortOrder[i], 1);
    }
    PabloAST * RunEnds = pb.createAnd(Runs, pb.createNot(pb.createLookahead(Runs, 1)));
    PabloAST * violation = pb.createAnd(bnc.UGT(SortOrder, SortOrder_ahead), pb.createNot(RunEnds));
    PabloAST * violationInSeq = pb.createAnd(pb.createMatchStar(violation, Runs), RunEnds);
    pb.createAssign(pb.createExtract(getOutputStreamVar("Misordered"), pb.getInteger(0)), violationInSeq);
}

class AdjustRunsAndIndexes : public pablo::PabloKernel {
public:
    AdjustRunsAndIndexes(LLVMTypeSystemInterface & ts,
                         StreamSet * Runs, StreamSet * Misordered, StreamSet * SeqIndex,
                         StreamSet * FilteredRuns, StreamSet * AdjustedIndex);
protected:
    void generatePabloMethod() override;
    unsigned mLgth;
};

AdjustRunsAndIndexes::AdjustRunsAndIndexes (LLVMTypeSystemInterface & ts,
                                            StreamSet * Runs, StreamSet * Misordered, StreamSet * SeqIndex,
                                            StreamSet * FilteredRuns, StreamSet * AdjustedIndex)
: PabloKernel(ts, "AdjustRunsAndIndexes_" + SeqIndex->shapeString(),
// inputs
{Binding{"Runs", Runs, FixedRate(1), LookAhead(64)}, Binding{"Misordered", Misordered, FixedRate(1), LookAhead(64)}, Binding{"SeqIndex", SeqIndex, FixedRate(1), LookAhead(64)}},
// output
{Binding{"FilteredRuns", FilteredRuns}, Binding{"AdjustedIndex", AdjustedIndex}}) {
}

void AdjustRunsAndIndexes::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * Runs = getInputStreamSet("Runs")[0];
    PabloAST * RunStart = pb.createAnd(Runs, pb.createNot(pb.createAdvance(Runs, 1)));
    PabloAST * Misordered = getInputStreamSet("Misordered")[0];
    BixNum SeqIndex = getInputStreamSet("SeqIndex");
    unsigned maxLgth = 1<<SeqIndex.size();
    BixNum SeqIndexAhead(SeqIndex.size());
    PabloAST * FilteredRuns = Runs;
    BixNum AdjustedIndex = SeqIndex;
    for (unsigned lgth = 3; lgth < maxLgth; lgth++) {
        for (unsigned i = 0; i < SeqIndex.size(); i++) {
            SeqIndexAhead[i] = pb.createLookahead(SeqIndex[i], lgth-1);
        }
        PabloAST * atRunStart = pb.createAnd(RunStart, pb.createNot(pb.createLookahead(Runs, lgth)));
        atRunStart = pb.createAnd(bnc.EQ(SeqIndexAhead, lgth-1), atRunStart);
        PabloAST * MisorderedAtStart = pb.createAnd(atRunStart, pb.createLookahead(Misordered, lgth-1), "MisorderedAtStart");
        PabloAST * OrderedRunStart = pb.createAnd(atRunStart, pb.createNot(MisorderedAtStart));
        PabloAST * OrderedRun = pb.createAnd(pb.createMatchStar(OrderedRunStart, Runs), Runs);
        FilteredRuns = pb.createAnd(FilteredRuns, pb.createNot(OrderedRun));
        unsigned lgth_ceil = 1 << ceil_log2(lgth);
        unsigned lgth_offset = lgth_ceil - lgth;
        if (lgth_offset != 0) {
            PabloAST * lgthRun = pb.createAnd(pb.createMatchStar(atRunStart, Runs), Runs);
            AdjustedIndex = bnc.Select(lgthRun, bnc.AddModular(AdjustedIndex, lgth_offset), AdjustedIndex);
        }
    }
    for (unsigned i = 0; i < SeqIndex.size(); i++) {
        AdjustedIndex[i] = pb.createAnd(AdjustedIndex[i], FilteredRuns);
    }
    writeOutputStreamSet("FilteredRuns", std::vector<PabloAST *>{FilteredRuns});
    writeOutputStreamSet("AdjustedIndex", AdjustedIndex);
}

BitonicCompareStep::BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned distance, unsigned region_size,
                                       StreamSet * Runs, StreamSet * SeqIndex, StreamSet * Basis, StreamSet * SwapMarks, StreamSet * Debug)
: PabloKernel(ts, "BitonicCompareStep<" + std::to_string(region_size) + "," + std::to_string(distance) + ">" +
              SeqIndex->shapeString() + "_" + Basis->shapeString(),
// inputs
{Binding{"Runs", Runs}, Binding{"SeqIndex", SeqIndex}, Binding{"Basis", Basis}},
// output
{Binding{"SwapMarks", SwapMarks}, Binding{"Debug", Debug}}), mCompareDistance(distance), mRegionSize(region_size) {
}

void BitonicCompareStep::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Runs = getInputStreamSet("Runs")[0];
    BixNum SeqIndex = getInputStreamSet("SeqIndex");
    BixNum Basis = getInputStreamSet("Basis");
    auto advance_amt = pb.getInteger(mCompareDistance);
    Var * SwapVar = pb.createVar("SwapVar", pb.createZeroes());
    // Bitonic swapping:
    // At step N (N = 0, 1, ...):
    // Comparison distance is mCompareDistance = 1 << N.
    // Input is divided into groups of size 1 << (N + 2) (group size 4, for N = 0).
    // Each input group is split into 2 subgroups of size (1 << (N + 1)) (size 2 for N = 0)
    BixNumCompiler bnc0(pb);
    PabloAST * DistN = bnc0.UGE(SeqIndex, mCompareDistance, "DistN");
    // If no instance has sequential index reaching the comparison distance,
    // there will be nothing to compare.
    auto nested = pb.createScope();
    pb.createIf(DistN, nested);
    BixNumCompiler bnc(nested);
    BixNum Forward_Basis(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        Forward_Basis[i] = nested.createAdvance(Basis[i], advance_amt, "Fwd_basis" + std::to_string(i));
    }
    // Identify the separate regions.
    unsigned bit_identifying_hi_region = ceil_log2(mRegionSize);
    PabloAST * descending_regions = nested.createZeroes();
    if (SeqIndex.size() > bit_identifying_hi_region) {
        descending_regions = SeqIndex[bit_identifying_hi_region];
    }
    unsigned bit_identifying_subgroup_hi_elements = ceil_log2(mCompareDistance);
    PabloAST * hi_elements_in_comparisons = SeqIndex[bit_identifying_subgroup_hi_elements];
    PabloAST * gt_forward = bnc.UGT(Forward_Basis, Basis, "gt_forward");
    PabloAST * compare = nested.createXor(gt_forward, descending_regions, "compare");
    // Negation of > is <=, exclude the = case.
    compare = nested.createAnd(compare, bnc.NEQ(Forward_Basis, Basis), "compare3");
    PabloAST * swap_mark = nested.createAnd(compare, hi_elements_in_comparisons);
    PabloAST * consecutiveRuns = Runs;
    for (unsigned i = 1; i < mCompareDistance; i*=2) {
        consecutiveRuns = nested.createAnd(consecutiveRuns, nested.createAdvance(consecutiveRuns, i));
    }
    swap_mark = nested.createAnd(swap_mark, consecutiveRuns);
    nested.createAssign(SwapVar, swap_mark);
    pb.createAssign(pb.createExtract(getOutputStreamVar("SwapMarks"), pb.getInteger(0)), SwapVar);
}


SwapBack_N::SwapBack_N(LLVMTypeSystemInterface & ts, unsigned n, StreamSet * SwapMarks, StreamSet * Source, StreamSet * Swapped)
: PabloKernel(ts, "SwapBack" + std::to_string(n) + "_" + Source->shapeString(),
// inputs
{Binding{"SwapMarks", SwapMarks, FixedRate(1), LookAhead(n)},
 Binding{"Source", Source, FixedRate(1), LookAhead(n)}},
// output
{Binding{"Swapped", Swapped}}), mN(n) {
}

void SwapBack_N::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * SwapMarks = getInputStreamSet("SwapMarks")[0];
    PabloAST * PriorMark = pb.createLookahead(SwapMarks, mN);
    std::vector<PabloAST *> SourceSet = getInputStreamSet("Source");
    std::vector<Var *> SwappedVar(SourceSet.size());
    for (unsigned i = 0; i < SourceSet.size(); i++) {
        SwappedVar[i] = pb.createVar("SwapVar" + std::to_string(i), SourceSet[i]);
    }
    auto nested = pb.createScope();
    pb.createIf(pb.createOr(PriorMark, SwapMarks), nested);
    for (unsigned i = 0; i < SourceSet.size(); i++) {
        PabloAST * compare = nested.createXor(nested.createLookahead(SourceSet[i], mN), SourceSet[i]);
        compare = nested.createAnd(compare, PriorMark);
        PabloAST * flip = nested.createOr(compare, nested.createAdvance(compare, pb.getInteger(mN)));
        nested.createAssign(SwappedVar[i], nested.createXor(SourceSet[i], flip));
    }
    writeOutputStreamSet("Swapped", SwappedVar);
}

class AppendStreamSets : public PabloKernel {
public:
    AppendStreamSets(LLVMTypeSystemInterface & ts, StreamSet * A, StreamSet * B, StreamSet * Combined);
protected:
    void generatePabloMethod() override;
};

AppendStreamSets::AppendStreamSets(LLVMTypeSystemInterface & ts, StreamSet * A, StreamSet * B, StreamSet * Combined)
: PabloKernel(ts, "AppendStreamSets_" + A->shapeString() + "_" + B->shapeString(),
// inputs
{Binding{"A", A}, Binding{"B", B}},
// output
{Binding{"Combined", Combined}}) {
}

void AppendStreamSets::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> A = getInputStreamSet("A");
    std::vector<PabloAST *> B = getInputStreamSet("B");
    std::vector<PabloAST *> Combined(A.size() + B.size());
    for (unsigned i = 0; i < A.size(); i++) {
        Combined[i + B.size()] = A[i];
    }
    for (unsigned i = 0; i < B.size(); i++) {
        Combined[i] = B[i];
    }
    writeOutputStreamSet("Combined", Combined);
}

class RunTails : public PabloKernel {
public:
    RunTails(LLVMTypeSystemInterface & ts, unsigned lgth, StreamSet * Runs, StreamSet * SeqIndex, StreamSet * Tails);
protected:
    void generatePabloMethod() override;
private:
    unsigned mLgth;
};

RunTails::RunTails(LLVMTypeSystemInterface & ts, unsigned lgth, StreamSet * Runs, StreamSet * SeqIndex, StreamSet * Tails)
: PabloKernel(ts, "RunTail" + std::to_string(lgth) + "_" + SeqIndex->shapeString(),
// inputs
{Binding{"Runs", Runs, FixedRate(1), LookAhead(lgth+1)},
 Binding{"SeqIndex", SeqIndex, FixedRate(1), LookAhead(lgth)}},
// output
{Binding{"Tails", Tails}}), mLgth(lgth) {
}

void RunTails::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Runs = getInputStreamSet("Runs")[0];
    std::vector<PabloAST *> SeqIndex = getInputStreamSet("SeqIndex");
    Var * tailVar = pb.createVar("Tails", pb.createZeroes());
    BixNumCompiler bnc0(pb);
    // First determine a potential tail candidate start position.
    PabloAST * ahead = pb.createLookahead(Runs, mLgth);
    PabloAST * tail_prior = pb.createAnd(Runs, ahead);
    auto nested = pb.createScope();
    pb.createIf(tail_prior, nested);
    BixNumCompiler bnc(nested);
    std::vector<PabloAST *> SeqIndexAhead(SeqIndex.size());
    for (unsigned i = 0; i < SeqIndex.size(); i++) {
        SeqIndexAhead[i] = nested.createLookahead(SeqIndex[i], mLgth);
    }
    PabloAST * confirm = bnc.EQ(bnc.AddModular(SeqIndex, mLgth), SeqIndexAhead);
    PabloAST * tail1 = nested.createAdvance(nested.createAnd(tail_prior, confirm), 1);
    tail1 = nested.createAnd(tail1, nested.createNot(ahead));
    PabloAST * tails = nested.createAnd(nested.createMatchStar(tail1, Runs), Runs);
    nested.createAssign(tailVar, tails);
    pb.createAssign(pb.createExtract(getOutputStreamVar("Tails"), pb.getInteger(0)), tailVar);
}

StreamSets  BitonicSortRuns(PipelineBuilder & P, unsigned instance_size, StreamSet * Runs, StreamSets & ToSort) {
    unsigned steps = ceil_log2(instance_size);
    StreamSet * SeqIndex = P.CreateStreamSet(steps);
    P.CreateKernelCall<RunIndex>(Runs, SeqIndex);
    SHOW_BIXNUM(SeqIndex);
    StreamSet * Misordered = P.CreateStreamSet(1);
    P.CreateKernelCall<Misorder_Check>(Runs, ToSort[0], Misordered);
    SHOW_STREAM(Misordered);
    StreamSet * FilteredRuns = P.CreateStreamSet(1);
    StreamSet * AdjustedIndex = P.CreateStreamSet(SeqIndex->getNumElements());
    P.CreateKernelCall<AdjustRunsAndIndexes>(Runs, Misordered, SeqIndex, FilteredRuns, AdjustedIndex);
    SHOW_STREAM(FilteredRuns);
    SHOW_BIXNUM(AdjustedIndex);
    StreamSet * SortOrder = P.CreateStreamSet(SeqIndex->getNumElements() + ToSort[0]->getNumElements());
    P.CreateKernelCall<AppendStreamSets>(ToSort[0], AdjustedIndex, SortOrder);
    ToSort[0] = SortOrder;
    return BitonicSort(P, instance_size, FilteredRuns, AdjustedIndex, ToSort);
}

StreamSets BitonicSort(PipelineBuilder & P, unsigned instance_size, StreamSet * Runs, StreamSet * SeqIndex, StreamSets & ToSort) {
    unsigned region_size = instance_size/2;
    unsigned compare_distance = region_size/2;

    StreamSets PartiallySorted;
    if (compare_distance > 1) {
        PartiallySorted = BitonicSort(P, region_size, Runs, SeqIndex, ToSort);
    } else {
        PartiallySorted = ToSort;
    }

    StreamSet * Debug = P.CreateStreamSet(1, 1);
    StreamSet * SwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(compare_distance, region_size, Runs, SeqIndex, PartiallySorted[0], SwapMarks, Debug);
    SHOW_STREAM(Debug);
    SHOW_STREAM(SwapMarks);

    StreamSets Sorted(ToSort.size());
    for (unsigned i = 0; i < ToSort.size(); i++) {
        Sorted[i] = P.CreateStreamSet(ToSort[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(compare_distance, SwapMarks, PartiallySorted[i], Sorted[i]);
        SHOW_BIXNUM(Sorted[i]);
    }

    if (instance_size <=2 ) {
        return Sorted;
    } else {
        return BitonicMerge(P, instance_size, instance_size, Runs, SeqIndex, Sorted);
    }
}

StreamSets BitonicMerge(PipelineBuilder & P, unsigned region_size, unsigned instance_size, StreamSet * Runs, StreamSet * RunIndex, StreamSets & ToMerge) {
    
    StreamSet * MergeSwapMarks = P.CreateStreamSet(1, 1);
    StreamSet * Debug = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(region_size/2, instance_size, Runs, RunIndex, ToMerge[0], MergeSwapMarks, Debug);
    SHOW_STREAM(MergeSwapMarks);
    SHOW_STREAM(Debug);

    StreamSets Merged(ToMerge.size());
    for (unsigned i = 0; i < ToMerge.size(); i++) {
        Merged[i] = P.CreateStreamSet(ToMerge[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(region_size/2, MergeSwapMarks, ToMerge[i], Merged[i]);
        SHOW_BIXNUM(Merged[i]);
    }
    if (region_size <= 2) {
        return Merged;
    } else {
        return BitonicMerge(P, region_size/2, instance_size, Runs, RunIndex, Merged);
    }
}
