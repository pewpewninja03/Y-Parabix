#include "pipeline_analysis.hpp"
#include "evolutionary_algorithm.hpp"
#include <boost/icl/interval_set.hpp>

using boost::icl::interval_set;

namespace kernel {

// TODO: nested pipeline kernels could report how much internal memory they require
// and reason about that here (and in the scheduling phase)

constexpr static unsigned BUFFER_SIZE_INIT_POPULATION_SIZE = 15;

constexpr static unsigned BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS = 2;

constexpr static unsigned BUFFER_SIZE_POPULATION_SIZE = 30;

constexpr static unsigned BUFFER_SIZE_GA_MAX_TIME_SECONDS = 15;

constexpr static unsigned BUFFER_SIZE_GA_STALLS = 50;

// Intel spatial prefetcher pulls cache line pairs, aligned to 128 bytes.

using IntervalGraph = adjacency_list<hash_setS, vecS, undirectedS>;

using IntervalSet = interval_set<unsigned>;

using Interval = IntervalSet::interval_type; // std::pair<unsigned, unsigned>;

using Vertex = unsigned;

struct BufferLayoutOptimizerWorker final : public PermutationBasedEvolutionaryAlgorithmWorker {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repair
     ** ------------------------------------------------------------------------------------------------------------- */
    void repair(Candidate & /* candidate */, pipeline_random_engine & rng) final { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    size_t fitness(const Candidate & candidate, pipeline_random_engine & rng) final {

        const auto candidateLength = candidate.size();

        size_t max_colours = 0;
        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto a = candidate[i];
            assert (a < candidateLength);
            size_t w = weight[a];

            assert (GC_IntervalSet.empty());

            for (unsigned j = 0; j != i; ++j) {
                assert (j < candidateLength);
                const auto b = candidate[j];
                assert (b < candidateLength);
                if (edge(a, b, I).second) {
                    const auto & interval = GC_Intervals[b];
                    auto l = interval.lower();
                    auto r = interval.upper();
                    GC_IntervalSet.insert(Interval::right_open(l, r));
                }
            }

            size_t start = 0;
            auto end = w;
            if (!GC_IntervalSet.empty()) {
//                auto d = w;
                for (const auto & interval : GC_IntervalSet) {
                    if (end < interval.lower()) {
                        break;
                    } else {
//                        const auto l = interval.lower();
                        const auto r = interval.upper();
                        // We want memory to be laid out s.t. when we expand it at run time,
                        // we're guaranteed that we won't overlap another buffer and ideally
                        // optimize to a solution that won't require a huge amount of
                        // additional space. To do so, we increase the weight (bytes required)
                        // so that the size of each placement in sequence is non-decreasing.

                        // NOTE: this is not the final size of the placement.

                        // TODO: is max sufficient? do we need a LCM?
//                        const auto m = r - l;
//                        if (d < m) {
//                            d = m;
//                        }
                        start = r;
                        end = r + w;
                    }
                }
                GC_IntervalSet.clear();
            }
            assert (a < candidateLength);
//            const auto end = start + w;
            GC_Intervals[a] = Interval::right_open(start, end);
            max_colours = std::max(max_colours, end);
        }

        return max_colours;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    const std::vector<Interval> & getIntervals(const OrderingDAWG & O, const unsigned candidateLength, pipeline_random_engine & rng) {
        Candidate chosen;
        chosen.reserve(candidateLength);
        Vertex u = 0;
        while (out_degree(u, O) != 0) {
            const auto e = first_out_edge(u, O);
            const auto k = O[e];
            chosen.push_back(k);
            u = target(e, O);
        }
        fitness(chosen, rng);
        return GC_Intervals;
    }

    BufferLayoutOptimizerWorker(const IntervalGraph & I, const std::vector<unsigned> & weight,
                                const unsigned candidateLength, pipeline_random_engine & rng)
    : I(I), weight(weight), GC_Intervals(candidateLength) {
        assert (num_vertices(I) == candidateLength);
        assert (weight.size() >= candidateLength);
    }

private:
    const IntervalGraph & I;
    const std::vector<unsigned> & weight;

    IntervalSet GC_IntervalSet;
    std::vector<Interval> GC_Intervals;
};

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    const std::vector<Interval> & getIntervals(const OrderingDAWG & O, pipeline_random_engine & rng) {
        auto w = (BufferLayoutOptimizerWorker *)mainWorker.get();
        return w->getIntervals(O, candidateLength, rng);
    }

    std::unique_ptr<PermutationBasedEvolutionaryAlgorithmWorker> makeWorker(pipeline_random_engine & rng) final {
        return std::make_unique<BufferLayoutOptimizerWorker>(I, weight, candidateLength, rng);
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned numOfLocalStreamSets
                         , IntervalGraph && I
                         , std::vector<unsigned> && weight
                         , pipeline_random_engine & srcRng)
    : PermutationBasedEvolutionaryAlgorithm (numOfLocalStreamSets,
                                             BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS,
                                             BUFFER_SIZE_INIT_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_MAX_TIME_SECONDS,                                             
                                             BUFFER_SIZE_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_STALLS,
                                             std::max(codegen::SegmentThreads, codegen::TaskThreads),
                                             srcRng)
    , I(std::move(I))
    , weight(weight) {

    }


private:

    const IntervalGraph I;
    const std::vector<unsigned> weight;

};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determineInitialThreadLocalBufferLayout
 *
 * Given our buffer graph, we want to identify the best placement to maximize sequential prefetching behavior with
 * the minimal total memory required. Although this assumes that the memory-aware scheduling algorithm was first
 * called, it does not actually use any data from it. The reason for this disconnection is to enable us to explore
 * the impact of static memory allocation independent of the chosen scheduling algorithm.
 *
 * Because the Intel L2 streamer prefetcher has one forward and one reverse monitor per page, a streamset will
 * only be placed in a page in which no other streamset accesses it during the same kernel invocation.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determineInitialThreadLocalBufferLayout(KernelBuilder & b, pipeline_random_engine & rng) {

    // This process serves two purposes: (1) generate the initial memory layout for our thread-local
    // streamsets. (2) determine how many the number of pages to assign each streamset based on the
    // number of strides executed by the parition root.

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }

    const auto n = LastStreamSet - FirstStreamSet + 1U;

    // TODO: can we insert a zero-extension region rather than having a secondary buffer?

    std::vector<unsigned> mapping(n, -1U);

    RequiredThreadLocalStreamSetMemory = 0;

//    PartitionRootStridesPerThreadLocalPage.resize(PartitionCount);

//    NumOfPartialOverflowStridesPerPartitionRootStride.resize(PartitionCount);

    unsigned numOfThreadLocalStreamSets = 0U;

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
            mapping[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
            ++numOfThreadLocalStreamSets;
        }
    }

    if (LLVM_UNLIKELY(numOfThreadLocalStreamSets == 0)) {
        return;
    }

    DataLayout DL(b.getModule());

    const auto blockWidth = b.getBitBlockWidth();

    const size_t pageSize = b.getPageSize();

    IntervalGraph I(numOfThreadLocalStreamSets);

    std::vector<unsigned> weight(numOfThreadLocalStreamSets, 0);
    std::vector<int> remaining(numOfThreadLocalStreamSets, 0); // NOTE: signed int type is necessary here
    std::vector<Rational> streamSetFactor(numOfThreadLocalStreamSets);

    for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto firstKernel = FirstKernelInPartition[partitionId];
        const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

        bool hasThreadLocal = false;

        for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

            const auto strideLength = getKernel(kernel)->getStride();

            const Rational rateFactor{strideLength * MaximumNumOfStrides[kernel], blockWidth};

            // Because data is layed out in a "strip mined" format within streamsets, the type of
            // each "chunk" will be blockwidth items in length.

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                if (LLVM_UNLIKELY(in_degree(streamSet, InOutStreamSetReplacement) != 0)) {
                    continue;
                }

                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                    // determine the number of bytes this streamset requires per *root kernel* stride
                    const BufferPort & producerRate = mBufferGraph[output];
                    const Binding & outputRate = producerRate.Binding;
                    Type * const type = StreamSetBuffer::resolveType(b, outputRate.getType());
                    const auto typeSize = b.getTypeSize(DL, type);
                    const auto j = mapping[streamSet - FirstStreamSet];
                    assert (j != -1U);
                    const auto size = typeSize * bn.RequiredCapacity;

                    weight[j] = round_up_to(size, pageSize); // .numerator()
                    assert ((weight[j] % pageSize) == 0);

                    // record how many consumers exist before the streamset memory can be reused
                    // (NOTE: the +1 is to indicate this kernel requires each output streamset
                    // to be distinct even if one or more of the outputs is not used later.)
                    remaining[j] = out_degree(streamSet, mBufferGraph) + 1U;

                    hasThreadLocal = true;
                }
            }
        }

        if (hasThreadLocal) {
            // Mark any overlapping allocations in our interval graph.
            for (unsigned i = 0; i != numOfThreadLocalStreamSets; ++i) {
                if (remaining[i] > 0) {
                    for (unsigned j = 0; j != i; ++j) {
                        if (remaining[j] > 0) {
                            add_edge(j, i, I);
                        }
                    }
                }
            }

            // Determine which streamsets are no longer alive
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }
                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }
            }
        }
    }

    BufferLayoutOptimizer BA(numOfThreadLocalStreamSets, std::move(I), std::move(weight), rng);
    BA.runGA();

    auto requiredMemory = BA.getBestFitnessValue();
    assert ((requiredMemory % pageSize) == 0);
    auto O = BA.getResult();

    // TODO: apart from total memory, when would one layout be better than another?
    // Can we quantify it based on the buffer graph order? Currently, we just take
    // the first one.
    const auto intervals = BA.getIntervals(O, rng);

    bool hasThreadLocalInOut = false;

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const auto i = streamSet - FirstStreamSet;
        BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {
            if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
                hasThreadLocalInOut = true;
            } else {
                const auto j = mapping[i];
                const auto & interval = intervals[j];
                bn.BufferStart = interval.lower();
                assert ((bn.BufferStart % pageSize) == 0);
                bn.BufferEnd = interval.upper();
                assert ((bn.BufferEnd % pageSize) == 0);
                assert (bn.BufferEnd <= requiredMemory);
            }

        }
    }
    if (LLVM_UNLIKELY(hasThreadLocalInOut)) {
        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            BufferNode & bn = mBufferGraph[streamSet];
            assert (bn.isInOutRedirect() ^ (in_degree(streamSet, InOutStreamSetReplacement) == 0));
            if (LLVM_UNLIKELY(bn.isThreadLocal() && bn.isInOutRedirect())) {
                auto src = streamSet;
                do {
                    src = parent(src, InOutStreamSetReplacement);
                } while (in_degree(src, InOutStreamSetReplacement) != 0);
                const BufferNode & bs = mBufferGraph[src];
                bn.BufferStart = bs.BufferStart;
                bn.BufferEnd = bs.BufferEnd;
            }
        }
    }

    RequiredThreadLocalStreamSetMemory = requiredMemory;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateInterPartitionThreadLocalBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::updateInterPartitionThreadLocalBuffers() {

    // update threadlocal status of sources for truncated buffers

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) {
            mNonThreadLocalStreamSets.insert(streamSet);
            unsigned srcStreamSet = 0;
            for (auto ref : make_iterator_range(in_edges(streamSet, mStreamGraph))) {
                const auto & v = mStreamGraph[ref];
                if (v.Reason == ReasonType::Reference) {
                    srcStreamSet = source(ref, mBufferGraph);
                    assert (srcStreamSet >= FirstStreamSet && srcStreamSet <= LastStreamSet);
                    break;
                }
            }
            assert (srcStreamSet);
            const BufferNode & bn = mBufferGraph[srcStreamSet];
            if (LLVM_UNLIKELY(bn.isConstant() || bn.hasZeroElementsOrWidth())) {
                continue;
            }

            const auto producer = parent(srcStreamSet, mBufferGraph);
            const auto partId = KernelPartitionId[producer];

            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(input, mBufferGraph);
                if (partId != KernelPartitionId[consumer]) {
                    mNonThreadLocalStreamSets.insert(srcStreamSet);
                    break;
                }
            }
        }

        if (LLVM_UNLIKELY(bn.isConstant() || bn.hasZeroElementsOrWidth())) {
            continue;
        }

        const auto producer = parent(streamSet, mBufferGraph);
        const auto partId = KernelPartitionId[producer];

        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(input, mBufferGraph);
            if (partId != KernelPartitionId[consumer]) {
                mNonThreadLocalStreamSets.insert(streamSet);
                break;
            }
        }
    }

    if (num_edges(InOutStreamSetReplacement) > 0) {
        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            if (LLVM_UNLIKELY(out_degree(streamSet, InOutStreamSetReplacement) >0 && in_degree(streamSet, InOutStreamSetReplacement) == 0)) {
                auto toCheck = streamSet;
                bool isNonThreadLocal = false;
                for (;;) {
                    assert (FirstStreamSet <= toCheck && toCheck <= LastStreamSet);
                    if (mNonThreadLocalStreamSets.count(toCheck)) {
                        isNonThreadLocal = true;
                        break;
                    }
                    if (out_degree(toCheck, InOutStreamSetReplacement) == 0) {
                        break;
                    }
                    toCheck = child(toCheck, InOutStreamSetReplacement);
                }

                if (isNonThreadLocal) {
                    auto toUpdate = streamSet;
                    for (;;) {
                        assert (FirstStreamSet <= toUpdate && toUpdate <= LastStreamSet);
                        mNonThreadLocalStreamSets.insert(toUpdate);
                        if (out_degree(toUpdate, InOutStreamSetReplacement) == 0) {
                            break;
                        }
                        toUpdate = child(toUpdate, InOutStreamSetReplacement);
                    }
                }

            }
        }
    }

    for (;;) {

        for (const auto streamSet : mNonThreadLocalStreamSets) {
            BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.hasZeroElementsOrWidth())) {
                continue;
            }
            const auto producer = parent(streamSet, mBufferGraph);
            const auto partId = KernelPartitionId[producer];
            auto type = BufferLocality::PartitionLocal;
            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(e, mBufferGraph);
                if (KernelPartitionId[consumer] != partId) {
                    type = BufferLocality::GloballyShared;
                    break;
                }
            }

            bn.Locality = type;
        }

        mNonThreadLocalStreamSets.clear();

        // If any inter-partition input to a kernel is not thread local, none of its
        // inter-partition inputs can be safely made to be thread local.
        for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
            bool hasNonThreadLocalInput = false;
            for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isNonThreadLocal()) {
                    hasNonThreadLocalInput = true;
                    break;
                }
            }
            if (hasNonThreadLocalInput) {
                const auto partId = KernelPartitionId[kernel];
                for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(e, mBufferGraph);
                    BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto producer = parent(streamSet, mBufferGraph);
                        if (partId != KernelPartitionId[producer]) {
                            mNonThreadLocalStreamSets.insert(streamSet);
                        }
                    }
                }
            }
        }

        if (mNonThreadLocalStreamSets.empty()) {
            break;
        }

    }

}


} // end of kernel namespace
