#include "pipeline_analysis.hpp"
#include "evolutionary_algorithm.hpp"
#include "lexographic_ordering.hpp"
#include <boost/icl/interval_set.hpp>
//#include <boost/graph/breadth_first_search.hpp>
#include <toolchain/toolchain.h>
#include <z3.h>
#include <stack>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

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

using IntervalSet = boost::icl::interval_set<unsigned>;

using Interval = IntervalSet::interval_type; // std::pair<unsigned, unsigned>;

using Vertex = unsigned;

struct BufferLayoutOptimizerWorker final : public PermutationBasedEvolutionaryAlgorithmWorker<double> {


    struct PartitionData {
        unsigned StreamSetCount = 0;
        unsigned PageCount = 0;
        Rational MinStridesPerSegment{};
        Rational SumOfStridesPerSegment;
    };

    constexpr static auto MAX_INT = std::numeric_limits<Rational::int_type>::max();

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repair
     ** ------------------------------------------------------------------------------------------------------------- */
    void repair(Candidate & /* candidate */, pipeline_random_engine & /* rng */) final { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    double fitness(const Candidate & candidate, pipeline_random_engine & /* rng */) final {

        const auto partitionCount = MaxMemorySize.size();

        assert (partitionCount > 0);

        for (unsigned i = 0; i < partitionCount; ++i) {
            MaxMemorySize[i] = 0;
        }

        const auto candidateLength = candidate.size();

        assert (candidateLength >= partitionCount);

        #ifndef NDEBUG
        for (unsigned i = 0; i < candidateLength; ++i) {
            GC_Intervals[i] = Interval::right_open(0, 0);
        }
        #endif

        for (unsigned i = 0; i < candidateLength; ++i) {

            const auto a = candidate[i];
            assert (a < candidateLength);
            assert (GC_Intervals[a].lower() == 0);
            assert (GC_Intervals[a].upper() == 0);

            assert (GC_IntervalSet.empty());
            for (unsigned j = 0; j < i; ++j) {
                const auto b = candidate[j];
                assert (b < candidateLength);
                if (edge(a, b, I).second) {
                    assert (edge(b, a, I).second);
                    assert (GC_Intervals[b].lower() < GC_Intervals[b].upper());
                    GC_IntervalSet.insert(GC_Intervals[b]);
                }
            }

            const auto w = weight[a];

            size_t start = 0;
            size_t end = w;
            if (!GC_IntervalSet.empty()) {
                for (const auto & interval : GC_IntervalSet) {
                    if (end <= interval.lower()) {
                        break;
                    } else {
                        const auto r = interval.upper();
                        start = r;
                        end = r + w;
                    }
                }
                GC_IntervalSet.clear();
            }
            assert (GC_IntervalSet.iterative_size() == 0);

            assert (end > start);
            GC_Intervals[a] = Interval::right_open(start, end);

            const auto p = partitionId[a];
            assert (p < partitionId.size());
            auto & M = MaxMemorySize[p];
            M = std::max(M, end);
        }

        double cost = 0;
        size_t max = 0;
        for (unsigned i = 0; i < partitionCount; ++i) {
            const size_t M = MaxMemorySize[i];
            cost += std::log(M * M);
            if (max < M) {
                max = M;
            }
        }
        cost = std::exp(cost / ((double)partitionCount)) * (max * max);
        return cost;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief translate
     ** ------------------------------------------------------------------------------------------------------------- */
    std::vector<Interval> translate(const OrderingDAWG & O, const unsigned candidateLength,
                   pipeline_random_engine & rng) {
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

//        for (unsigned i = 0; i < candidateLength; ++i) {
//            const Interval & interval = GC_Intervals[i];
//            const auto j = reverse_mapping[i];
//            BufferNode & bn = G[j];
//            bn.BufferStart = interval.lower();
//            bn.BufferEnd = interval.upper();
//        }

//        SepIntervalSet selected;
//        for (unsigned i = 0; i < candidateLength; ++i) {

//            const auto a = chosen[i];
//            const Interval & interval = GC_Intervals[a];
//            const auto lowest = interval.lower();
//            assert (selected.empty());
//            for (const auto u : make_iterator_range(adjacent_vertices(a, I))) {
//                const Interval & other = GC_Intervals[u];
//                if (other.upper() <= interval.lower()) {
//                    selected.insert(other);
//                }
//            }
//            for (auto & S : selected) {
//                #ifndef NDEBUG
//                bool found = false;
//                for (const auto u : make_iterator_range(adjacent_vertices(a, I))) {
//                    // while two streamsets may overlap in memory if they are not alive
//                    // at the same moment,
//                    const Interval & other = GC_Intervals[u];
//                    if (other.lower() == S.lower() && other.upper() == S.upper()) {
//                        found = true;
//                        break;
//                    }
//                }
//                assert (found);
//                #endif


//            }




//        }

//        size_t max = 0;

//        for (unsigned i = 0; i < candidateLength; ++i) {

//            const auto a = chosen[i];
//            assert (a < candidateLength);
//            assert (GC_IntervalSet.empty());
//            for (unsigned j = 0; j != i; ++j) {
//                const auto b = chosen[j];
//                assert (b < candidateLength);
//                if (edge(a, b, I).second) {
//                    errs() << "count0: " << GC_IntervalSet.iterative_size() << "\n";
//                    #ifndef NDEBUG
//                    const auto count = GC_IntervalSet.iterative_size();
//                    #endif
//                    GC_IntervalSet.insert(GC_Intervals[b]);
//                    errs() << "count1: " << GC_IntervalSet.iterative_size() << "\n";
//                    assert (GC_IntervalSet.iterative_size() == count + 1);
//                }
//            }

//            const auto w = weight[a];

//            size_t start = 0;
//            size_t end = w;
//            if (!GC_IntervalSet.empty()) {
//                const auto v = reverse_mapping[a];
//                BufferNode & bn = G[v];
//                auto & vec = bn.PreceedingThreadLocalStrideSize;
//                for (const auto & interval : GC_IntervalSet) {
//                    if (end <= interval.lower()) {
//                        break;
//                    } else {
//                        const auto r = interval.upper();
//                        assert (r > interval.lower());
//                        vec.push_back(r - interval.lower());
//                        start = r;
//                        end = r + w;
//                    }
//                }
//                GC_IntervalSet.clear();
//                std::sort(vec.begin(), vec.end());
//                bn.StrideSize = w;

//                auto u = v;
//                while (LLVM_UNLIKELY(out_degree(u, inOutGraph) != 0)) {
//                    assert (out_degree(u, inOutGraph) == 0);
//                    u = parent(u, inOutGraph);
//                    BufferNode & bn = G[u];
//                    auto & inoutVec = bn.PreceedingThreadLocalStrideSize;
//                    inoutVec.reserve(vec.size());
//                    for (auto val : vec) {
//                        inoutVec.push_back(val);
//                    }
//                    bn.StrideSize = w;
//                }

//            }
//            GC_Intervals[a] = Interval::right_open(start, end);
//            max = std::max(max, end);
//        }
//        return max;
    }

    BufferLayoutOptimizerWorker(const unsigned candidateLength
                               , const size_t partitionCount
                               , const IntervalGraph & I
                               , const std::vector<size_t> & weight
                               , const std::vector<unsigned> & partitionId
                               , pipeline_random_engine & rng)
    : I(I)
    , weight(weight)
    , partitionId(partitionId)
    , GC_IntervalSet()
    , GC_Intervals(weight.size())
    , MaxMemorySize(partitionCount) {

    }

private:

    const IntervalGraph & I;
    const std::vector<size_t> & weight;
    const std::vector<unsigned> & partitionId;

    IntervalSet GC_IntervalSet;
    std::vector<Interval> GC_Intervals;
    std::vector<size_t> MaxMemorySize;

};

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm<double> {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    std::vector<Interval> translate(const OrderingDAWG & O, pipeline_random_engine & rng) {
        auto w = (BufferLayoutOptimizerWorker *)mainWorker.get();
        return w->translate(O, candidateLength, rng);
    }

    WorkerPtr makeWorker(pipeline_random_engine & rng) final {
        return std::make_unique<BufferLayoutOptimizerWorker>(candidateLength, partitionCount, I, weight, partitionId, rng);
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned numOfLocalStreamSets
                         , const size_t partitionCount
                         , const IntervalGraph & I
                         , const std::vector<size_t> & weight
                         , const std::vector<unsigned> & partitionId
                         , pipeline_random_engine & srcRng)
    : PermutationBasedEvolutionaryAlgorithm (numOfLocalStreamSets,
                                             BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS,
                                             BUFFER_SIZE_INIT_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_MAX_TIME_SECONDS,
                                             BUFFER_SIZE_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_STALLS,
                                             std::max(codegen::SegmentThreads, codegen::TaskThreads),
                                             srcRng)
    , I(I)
    , weight(weight)
    , partitionId(partitionId)
    , partitionCount(partitionCount) {
        assert (num_vertices(I) == numOfLocalStreamSets);
        assert (weight.size() >= numOfLocalStreamSets);
        assert (numOfLocalStreamSets >= partitionCount);
    }


private:

    const IntervalGraph & I;
    const std::vector<size_t> & weight;
    const std::vector<unsigned> & partitionId;
    const size_t partitionCount;

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

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }

    // This process serves two purposes: (1) generate the initial memory layout for our thread-local
    // streamsets. (2) determine how many the number of pages to assign each streamset based on the
    // number of strides executed by the parition root.


    const auto n = LastStreamSet - FirstStreamSet + 1U;

    std::vector<unsigned> mapping(n);
    std::vector<size_t> unitWeight(n);
    std::vector<unsigned> streamSetPartitionId(n);

    auto & dl = b.getModule()->getDataLayout();

    size_t numOfThreadLocalStreamSets = 0U;
    size_t packedPartitionCount = 0;

    const size_t pageSize = getPageSize();

    const Rational PS{pageSize * 1, 10};

    for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto firstKernel = FirstKernelInPartition[partitionId];
        const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

        const auto startThreadLocalStreamSetCount = numOfThreadLocalStreamSets;

        for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];

                if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                    mapping[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
                    streamSetPartitionId[numOfThreadLocalStreamSets] = packedPartitionCount;
                    Type * const type = bn.Buffer->getType();
                    const size_t typeSize = b.getTypeSize(dl, type);
                    const BufferPort & bp = mBufferGraph[output];
                    const auto W = bp.Maximum * Rational{typeSize * StrideRepetitionVector[kernel], b.getBitBlockWidth()};
                    assert (W.numerator() > 0);
                    assert (W.denominator() == 1);
                    unitWeight[numOfThreadLocalStreamSets] = W.numerator();
                    ++numOfThreadLocalStreamSets;
                }
            }
        }

        if (startThreadLocalStreamSetCount != numOfThreadLocalStreamSets) {
            ++packedPartitionCount;
        }

    }

    const auto m = PartitionCount + LastStreamSet - FirstStreamSet + 1;

    ThreadLocalPlacementGraph T(m);

    if (numOfThreadLocalStreamSets) {

        IntervalGraph I(numOfThreadLocalStreamSets);

        std::vector<int> remaining(numOfThreadLocalStreamSets); // NOTE: signed int type is necessary here

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

            // Determine which streamsets are no longer alive
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        // We add +1 here because some outputs might be unused but still cannot
                        // reuse memory of an input buffer
                        remaining[j] = out_degree(streamSet, mBufferGraph) + 1;
                    }
                }

                // Mark any overlapping allocations in our interval graph.
                for (unsigned i = 1; i != numOfThreadLocalStreamSets; ++i) {
                    if (remaining[i] > 0) {
                        for (unsigned j = 0; j != i; ++j) {
                            if (remaining[j] > 0) {
                                add_edge(j, i, I);
                            }
                        }
                    }
                }

                // Undo the +1 for the produced buffers
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
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
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
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

        BufferLayoutOptimizer BA(numOfThreadLocalStreamSets,
                                 packedPartitionCount,
                                 I, unitWeight, streamSetPartitionId,
                                 rng);

        BA.runGA<false>();

        auto O = BA.getResult();

        const auto intervals = BA.translate(O, rng);

        std::vector<unsigned> streamSets;
        std::vector<unsigned> reverse_mapping(numOfThreadLocalStreamSets, 0);

        size_t gcd = 0;

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
                assert (streamSets.empty());

                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (reverse_mapping[j] == (PartitionCount + streamSet - FirstStreamSet));
                        streamSets.push_back(j);
                    }
                }

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        reverse_mapping[j] = PartitionCount + streamSet - FirstStreamSet;
                        streamSets.push_back(j);
                    }
                }

                std::sort(streamSets.begin(), streamSets.end());
                const auto end = std::unique(streamSets.begin(), streamSets.end());

                std::sort(streamSets.begin(), end, [&](unsigned i, unsigned j) {
                    const auto & A = intervals[i];
                    const auto & B = intervals[j];
                    return (B.lower() >= A.lower()) && (B.upper() >= A.upper());
                });

                size_t current = partitionId;
                const auto m = std::distance(streamSets.begin(), end);
                for (size_t i = 0; i < m; ++i) {
                    const auto a = streamSets[i];
                    assert (a < numOfThreadLocalStreamSets);
                    const auto & A = intervals[a];
                    const auto dist = A.upper() - A.lower();
                    const auto next = reverse_mapping[a];
                    assert (PartitionCount <= next && next <= PartitionCount + LastStreamSet);
                    for (auto e : make_iterator_range(out_edges(current, T))) {
                        if (target(e, T) == next) {
                            assert(T[e] == dist);
                            goto skip_adding;
                        }
                    }
                    if (gcd == 0) {
                        gcd = dist;
                    } else {
                        gcd = boost::gcd<size_t>(gcd, dist);
                    }
                    add_edge(current, next, dist, T);
skip_adding:        current = next;
                }
                streamSets.clear();
            }
        }

        transitive_reduction_dag(T);

#if 0

        // TODO: we need to determine if the every path has an equal weight to simplify the max
        // memory required formula at run time. Conversely if we idenitfy the highest weight path
        // then we can simplify the complexity here.

        std::vector<unsigned> toVisit(m);
        for (unsigned i = PartitionCount; i < m; ++i) {
            const auto d = in_degree(i, T);
            toVisit[i] = d;
        }

        std::queue<unsigned> Q;

        std::vector<size_t> distance(m, 0);
        size_t maxpathval = 0;
        for (unsigned x = 0; x < PartitionCount; ++x) {
            for (auto u = x;;) {
                for (auto e : make_iterator_range(out_edges(u, T))) {
                    const auto v = target(e, T);
                    if (--toVisit[v] == 0) {
                        size_t d = 0;
                        for (auto e : make_iterator_range(in_edges(v, T))) {
                            d = std::max(d, distance[source(e, T)]);
                        }
                        assert (T[e] > 0);
                        assert (T[e] % gcd == 0);
                        const size_t v = T[e] / gcd;
                        assert (v < (v * v) || v == 1);
                        const auto pathval = d + v * v;
                        assert ((v * v) < pathval || d == 0);
                        maxpathval = std::max(maxpathval, pathval);
                        distance[v] = pathval;
                        Q.push(v);
                    }
                }
                if (Q.empty()) {
                    break;
                }
                u = Q.back();
                Q.pop();
            }
        }

        for (unsigned i = PartitionCount; i < m; ++i) {
            if (distance[i] == maxpathval) {
                add_edge(i, m, 0, T);
            }
        }

#endif

#if 0

        struct visitor : boost::default_bfs_visitor {
            using Vertex = ThreadLocalPlacementGraph::vertex_descriptor;
            void discover_vertex(const Vertex & s, const ThreadLocalPlacementGraph & g) {
                size_t weight = 0;
                unsigned i = 0;
                for (auto e : make_iterator_range(in_edges(s, g))) {
                    const auto w = g[e];
                    assert ((w % gcd) == 0);
                    const auto w2 = w / gcd;
                    pathWeights[i++] = g[source(e, g)] + (w2 * w2);
                }


            }
            const size_t gcd;
            std::vector<size_t> pathWeights;
        };

        std::vector<size_t> p1;
        std::vector<size_t> p2;

        for (unsigned v = PartitionCount; v < m; ++v) {
            const auto d = in_degree(v, T);
            if (d > 1) {
                // determine if the weights of both paths are identical
                for (unsigned i = 1; i < d; ++i) {
                    for (unsigned j = 0; j < i; ++j) {

                    }
                }


            }
        }

#endif

//        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
//            const BufferNode & bn = mBufferGraph[streamSet];
//            if (bn.isThreadLocal()) {
//                const auto producer = parent(streamSet, mBufferGraph);

//                add_edge(producer, streamSet, )

//                auto src = streamSet;
//                if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
//                    src = parent(streamSet, InOutStreamSetReplacement);
//                    assert (src < streamSet);
//                }

//                const BufferNode & bs = mBufferGraph[src];





//            }
//        }



//        const auto maxRequired = BA.translate(O, mBufferGraph, reverse_mapping, InOutStreamSetReplacement);

//        Rational outRate{std::numeric_limits<unsigned>::max()};

//        for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
//            if (in_degree(kernel, mBufferGraph) == 0) {
//                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
//                    outRate = std::min(outRate, mBufferGraph[output].Maximum);
//                }
//            }
//        }

//        RequiredThreadLocalStreamSetMemory = Rational{maxRequired * b.getBitBlockWidth()} / outRate;






    #if 0

        const std::vector<Interval> intervals{std::move(BA.translate(O, rng))};

        std::vector<unsigned> maxPartitionMemory(numOfThreadLocalStreamSets, 0);
        unsigned maxMemory = 0;
        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            const auto p = streamSetPartitionId[i];
            assert (p < packedPartitionCount);
            const auto m = intervals[i].upper();
            if (maxPartitionMemory[p] < m) {
                maxPartitionMemory[p] = m;
                if (maxMemory < m) {
                    maxMemory = m;
                }
            }
        }

        errs() << "maxMemory: " << maxMemory << "\n";

        flat_map<size_t, size_t> W;

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isThreadLocal()) {
                if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
                    const auto src = parent(streamSet, InOutStreamSetReplacement);
                    assert (src < streamSet);
                    const BufferNode & bs = mBufferGraph[src];
                    bn.BufferStart = bs.BufferStart;
                    bn.BufferEnd = bs.BufferEnd;
                } else {
                    const auto j = mapping[streamSet - FirstStreamSet];
                    assert (j < numOfThreadLocalStreamSets);
                    const auto & interval = intervals[j];

                    for (auto v : make_iterator_range(adjacent_vertices(j, I))) {
                        const auto & other = intervals[v];
                        if (other.upper() <= interval.lower()) {
                            W.emplace(other.lower(), unitWeight[v]);
                        }
                    }

                    for (auto w : W) {
                        bn.PreceedingThreadLocalWeights.push_back(w.second);
                    }



                    bn.BufferStart = interval.lower();
                    bn.BufferEnd = interval.upper();
                }
            }
        }

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isThreadLocal()) {
                BufferNode & bn = mBufferGraph[streamSet];
                auto src = streamSet;
                if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
                    src = parent(streamSet, InOutStreamSetReplacement);
                    assert (src < streamSet);
                }
                const auto j = mapping[streamSet - FirstStreamSet];
                for (auto v : make_iterator_range(adjacent_vertices(j, I))) {
                    const auto other = reverse_mapping[v];
                    const BufferNode & bo = mBufferGraph[other];
                    if (bo.BufferEnd <= bn.BufferStart) {
                        bn.PreceedingThreadLocalBuffers.push_back(other);
                    }
                }

            }
        }



        ThreadLocalExpansionThresholdFactor.resize(PartitionCount, Rational{0});

        Rational maxRequired{0};

//        unsigned packedPartitionIndex = 0;
        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        // CEIL(num of strides * sf) * bn.BufferStart
                        assert (bn.BufferEnd > bn.BufferStart);
                        const auto j = mapping[streamSet - FirstStreamSet];
                        Rational sf{unitWeight[j], bn.BufferEnd - bn.BufferStart};
                        assert (ThreadLocalExpansionThresholdFactor[partitionId].numerator() == 0 || ThreadLocalExpansionThresholdFactor[partitionId] == sf);
                        assert (sf.denominator() >= sf.numerator());

                        errs() << "StreamSetTL_" << streamSet << " := " << sf.numerator() << "/" << sf.denominator() << "\n";

                        ThreadLocalExpansionThresholdFactor[partitionId] = sf;
                        maxRequired = std::max(maxRequired, sf);
//                        ++packedPartitionIndex;
//                        goto next_partition;
                    }
                }
            }
next_partition:
            continue;
        }
//        assert (packedPartitionIndex == packedPartitionCount);



        Rational rsm{maxMemory * pageSize * mPipelineKernel->getStride(), 1};

        RequiredThreadLocalStreamSetMemory = maxRequired * rsm;

        errs() << mPipelineKernel->getName() <<  ".RequiredThreadLocalStreamSetMemory: " << RequiredThreadLocalStreamSetMemory.numerator() << "/" << RequiredThreadLocalStreamSetMemory.denominator() << "\n";

#endif

#if 0


        using OrderedBeforeGraph = adjacency_list<vecS, vecS, bidirectionalS>;

        OrderedBeforeGraph B(numOfThreadLocalStreamSets + 1);

        printGraph(I, errs(), "I");



        for (auto e : make_iterator_range(edges(I))) {
            auto i = source(e, I);
            const auto & a = intervals[i];
            auto j = target(e, I);
            const auto & b = intervals[j];
            assert (streamSetPartitionId[i] == streamSetPartitionId[j]);
            assert (a.lower() != b.lower());
            if (a.lower() > b.lower()) {
                std::swap(i, j);
            }
            assert (intervals[i].upper() <= intervals[j].lower());
            add_edge(i, j, B);


        }

        std::vector<unsigned> maxPartitionMemory(numOfThreadLocalStreamSets, 0);
        unsigned maxMemory = 0;
        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            if (out_degree(i, B) == 0) {
                add_edge(i, numOfThreadLocalStreamSets, B);
            }
            const auto m = intervals[i].upper();
            const auto p = streamSetPartitionId[i];
            assert (p < packedPartitionCount);
            maxPartitionMemory[p] = std::max(maxPartitionMemory[p], m);
            maxMemory = std::max(maxMemory, m);
        }

        transitive_reduction_dag(B);

        printGraph(B, errs(), "B");

        const auto cfg = Z3_mk_config();
        Z3_set_param_value(cfg, "model", "true");
        Z3_set_param_value(cfg, "proof", "false");
        Z3_set_param_value(cfg, "timeout", "2000");
        const auto ctx = Z3_mk_context(cfg);
        Z3_del_config(cfg);
        const auto solver = Z3_mk_optimize(ctx);
        Z3_optimize_inc_ref(ctx, solver);

        const auto varType = Z3_mk_int_sort(ctx);

        auto constant = [&](const Z3_int64 value) {
            return Z3_mk_int(ctx, value, varType);
        };

        auto hard_assert = [&](Z3_ast c) {
            Z3_optimize_assert(ctx, solver, c);
        };

        auto add =[&](Z3_ast X, Z3_ast Y) {
            assert (X && Y);
            std::array<Z3_ast, 2> args{ X, Y };
            return Z3_mk_add(ctx, 2, args.data());
        };

        auto sub =[&](Z3_ast X, Z3_ast Y) {
            assert (X && Y);
            std::array<Z3_ast, 2> args{ X, Y };
            return Z3_mk_sub(ctx, 2, args.data());
        };

        auto multiply =[&](Z3_ast X, Z3_ast Y) {
            assert (X && Y);
            std::array<Z3_ast, 2> args{ X, Y };
            return Z3_mk_mul(ctx, 2, args.data());
        };

        auto check = [&]() -> Z3_lbool {
            #if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 5, 0)
            return Z3_optimize_check(ctx, solver, 0, nullptr);
            #else
            return Z3_optimize_check(ctx, solver);
            #endif
        };

        const Z3_ast z3_ZERO = constant(0);

//        std::vector<Z3_ast> partitionScalingVars(packedPartitionCount, nullptr);
//        for (unsigned i = 0; i < packedPartitionCount; ++i) {
//            Z3_ast var = Z3_mk_fresh_const(ctx, nullptr, varType);
//            hard_assert(Z3_mk_gt(ctx, var, z3_ZERO));
//            partitionScalingVars[i] = var;




//        }

        std::vector<Z3_ast> partitionScalingVars(packedPartitionCount, nullptr);
        for (unsigned i = 0; i < packedPartitionCount; ++i) {
            const auto d = maxMemory / maxPartitionMemory[i];
            assert (d > 0);
            maxPartitionMemory[i] = d;
            partitionScalingVars[i] = constant(d);
        }

        std::vector<Z3_ast> streamSetStartOffset(numOfThreadLocalStreamSets + 1, nullptr);
        std::vector<Z3_ast> streamSetEndOffset(numOfThreadLocalStreamSets, nullptr);


        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            const auto p = streamSetPartitionId[i];
            assert (p < packedPartitionCount);
            auto sv = partitionScalingVars[p];
            const auto & a = intervals[i];
            Z3_ast start = z3_ZERO;
            assert (a.upper() > a.lower());
            assert (a.upper() <= maxMemory);
            Z3_ast end = constant(maxPartitionMemory[p] * (a.upper() - a.lower()));
            if (in_degree(i, B) > 0) {
                Z3_ast off = Z3_mk_fresh_const(ctx, nullptr, varType);
                hard_assert(Z3_mk_gt(ctx, off, z3_ZERO));
                start = multiply(partitionScalingVars[p], off);
                end = add(start, end);
            }
            streamSetStartOffset[i] = start;
            streamSetEndOffset[i] = end;
        }

        streamSetStartOffset[numOfThreadLocalStreamSets] = constant(maxMemory);

        std::vector<std::vector<Z3_ast>> spacers(packedPartitionCount);

        for (auto e : make_iterator_range(edges(B))) {
            auto i = source(e, B);
            assert (i < numOfThreadLocalStreamSets);
            auto j = target(e, B);
            assert (j == numOfThreadLocalStreamSets || streamSetPartitionId[i] == streamSetPartitionId[j]);
            assert (streamSetEndOffset[i]);
            assert (streamSetStartOffset[j]);
            hard_assert(Z3_mk_le(ctx, streamSetEndOffset[i], streamSetStartOffset[j]));
            Z3_ast dist = sub(streamSetStartOffset[j], streamSetEndOffset[i]);
            const auto p = streamSetPartitionId[i];
            assert (p < packedPartitionCount);
            spacers[p].push_back(multiply(dist, dist));
        }

        for (unsigned i = 0; i < packedPartitionCount; ++i) {
            const auto & S = spacers[i];
            if (LLVM_LIKELY(S.size() > 1)) {
                Z3_ast sumOfSpacers = Z3_mk_add(ctx, S.size(), S.data());
                Z3_optimize_minimize(ctx, solver, sumOfSpacers);
            }
        }



        if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
            report_fatal_error("Z3 failed to find a solution to the thread local layout problem");
        }

        const auto model = Z3_optimize_get_model(ctx, solver);
        Z3_model_inc_ref(ctx, model);

        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {

            Z3_ast const startOffset = streamSetStartOffset[i];
            Z3_ast value;
            if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, startOffset, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
            }

            Z3_int64 num;
            if (LLVM_UNLIKELY(Z3_get_numeral_int64(ctx, value, &num) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
            }

            errs() << "S_" << i << " := " << num << "\n";

        }

        Z3_model_dec_ref(ctx, model);
        Z3_optimize_dec_ref(ctx, solver);
        Z3_del_context(ctx);
        Z3_reset_memory();
#endif

//        const size_t pageSize = getPageSize();
//        const auto pageScalingFactor = boost::lcm<size_t>(gcd, pageSize);
//        maxMemory *= pageScalingFactor;

//        size_t packedPartitionIndex = 0;

//        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
//            const auto firstKernel = FirstKernelInPartition[partitionId];
//            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];
//            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
//                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
//                    const auto streamSet = target(output, mBufferGraph);
//                    const BufferNode & bn = mBufferGraph[streamSet];
//                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
//                        assert (packedPartitionIndex < packedPartitionCount);
//                        const Rational & S = result.ScalingFactors[packedPartitionIndex];
//                        ThreadLocalExpansionThresholdFactor[partitionId] = S / StrideRepetitionVector[firstKernel];
//                        ++packedPartitionIndex;
//                        goto next_partition;
//                    }
//                }
//            }
//next_partition:
//            continue;
//        }
//        assert (packedPartitionIndex == packedPartitionCount);



//        RequiredThreadLocalStreamSetMemory = requiredMemory;

    }

    ThreadLocalPlacement = T;
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


#if 0

struct BufferLayoutOptimizerWorker final : public PermutationBasedEvolutionaryAlgorithmWorker<double> {


    struct PartitionData {
        unsigned StreamSetCount = 0;
        unsigned PageCount = 0;
        Rational MinStridesPerSegment{};
        Rational SumOfStridesPerSegment;
    };

    struct ResultVal {
        std::vector<Interval> Intervals;
        std::vector<Rational> ScalingFactors;

        ResultVal(std::vector<Interval> && intervals, std::vector<Rational> && sf)
        : Intervals(intervals), ScalingFactors(sf) {

        }
    };

    constexpr static auto MAX_INT = std::numeric_limits<Rational::int_type>::max();

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief newCandidate
     ** ------------------------------------------------------------------------------------------------------------- */
    void newCandidate(Candidate & candidate, pipeline_random_engine & rng) final {
        const auto l = candidate.size();
        assert (numOfThreadLocalStreamSets <= l);
        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            candidate[i] = i;
        }
        const auto c = unusedNodeRepetitions.size();
        auto i = numOfThreadLocalStreamSets;
        for (unsigned j = 0; j < c; ++j) {
            const auto m = unusedNodeRepetitions[j];
            for (auto k = m; k; --k) {
                assert (i < l);
                candidate[i++] = numOfThreadLocalStreamSets + j;
            }
        }
        assert (i == l);
        std::shuffle(candidate.begin(), candidate.begin() + l, rng);
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repair
     ** ------------------------------------------------------------------------------------------------------------- */
    void repair(Candidate & /* candidate */, pipeline_random_engine & /* rng */) final { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    double fitness(const Candidate & candidate, pipeline_random_engine & /* rng */) final {

        const auto partitionCount = PartitionData.size();

        for (unsigned i = 0; i < partitionCount; ++i) {
            auto & P = PartitionData[i];
            P.PageCount = 0;
            P.StreamSetCount = 0;
            P.MinStridesPerSegment = Rational{0};
            P.SumOfStridesPerSegment = Rational{0};
            GC_EmptyIntervals[i].clear();
        }

        const auto candidateLength = candidate.size();

        size_t maxEnd = 0;

        for (unsigned i = 0; i < candidateLength; ++i) {

            const auto a = candidate[i];
            assert (a < candidateLength);
            assert (GC_IntervalSet.empty());
            for (unsigned j = 0; j != i; ++j) {
                const auto b = candidate[j];
                assert (b < candidateLength);
                if (edge(a, b, I).second) {
                    if (b < numOfThreadLocalStreamSets) {
                        GC_IntervalSet.insert(GC_Intervals[b]);
                    } else {
                        const auto p = partitionId[a];
                        for (const auto & S : GC_EmptyIntervals[p]) {
                            GC_IntervalSet.insert(S);
                        }
                    }
                }
            }

            const auto w = minimumNumOfPages[a];

            size_t start = 0;
            size_t end = w;
            if (!GC_IntervalSet.empty()) {
                for (const auto & interval : GC_IntervalSet) {
                    if (end <= interval.lower()) {
                        break;
                    } else {
                        const auto r = interval.upper();
                        start = r;
                        end = r + w;
                    }
                }
                GC_IntervalSet.clear();
            }

            if (a < numOfThreadLocalStreamSets) {
                GC_Intervals[a] = Interval::right_open(start, end);
                // we allow "fake" nodes to permit the algorithm to insert unused pages
                if (a < numOfThreadLocalStreamSets) {
                    maxEnd = std::max(maxEnd, end);
                }
            } else {
                const auto p = partitionId[a];
                GC_EmptyIntervals[p].insert(Interval::right_open(start, end));
            }


        }


        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto a = candidate[i];
            assert (a < candidateLength);
            if (a < numOfThreadLocalStreamSets) {
                unsigned closestStartAfterEnd = maxEnd;
                auto & Ia = GC_Intervals[a];
                const auto endOfCurrentInterval = Ia.upper();
                for (const auto b : make_iterator_range(adjacent_vertices(a, I))) {
                    const auto startOfAdjacentInterval = GC_Intervals[b].lower();
                    if (startOfAdjacentInterval >= endOfCurrentInterval) {
                        if (b < numOfThreadLocalStreamSets) {
                            closestStartAfterEnd = std::min(startOfAdjacentInterval, closestStartAfterEnd);
                        }
                    }
                }
                assert (closestStartAfterEnd >= Ia.upper());
                const auto dist = closestStartAfterEnd - Ia.lower();
                assert (DataSizePerMinimumRepetition[a].numerator() > 0);
                const auto max = Rational{dist} * DataSizePerMinimumRepetition[a];
                const auto p = partitionId[a];
                assert (p < PartitionData.size());
                auto & P = PartitionData[p];
                if (P.StreamSetCount == 0) {
                    P.MinStridesPerSegment = max;
                    P.SumOfStridesPerSegment = max;
                } else {
                    P.MinStridesPerSegment = std::max(P.MinStridesPerSegment, max);
                    P.SumOfStridesPerSegment += max;
                }

                const auto d = Ia.upper() - Ia.lower();
                assert (d > 0);
                P.PageCount += d;
                P.StreamSetCount++;

                Ia = Interval::right_open(Ia.lower(), closestStartAfterEnd);
            }
        }


        Rational C{maxEnd * maxEnd};

        for (unsigned i = 0; i < partitionCount; ++i) {
            const auto & P = PartitionData[i];
            assert (P.StreamSetCount > 0);
            assert (P.MinStridesPerSegment.numerator() > 0);
            // R = total amount of unusuable space when M is the partition scaling factor
            const auto D = (P.MinStridesPerSegment * P.StreamSetCount);
            assert (P.SumOfStridesPerSegment.numerator() > 0);
            assert (D >= P.SumOfStridesPerSegment);
            const auto R = ((D - P.SumOfStridesPerSegment) / (P.SumOfStridesPerSegment + D)) + Rational{1};
            assert (R.numerator() > 0);
            C *= R;
        }

        const auto result = (double)C.numerator() / (double)C.denominator();

        errs() << " := " << result << "\n";

        return result;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief translate
     ** ------------------------------------------------------------------------------------------------------------- */
    ResultVal translate(const OrderingDAWG & O, const unsigned candidateLength, pipeline_random_engine & rng) {
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
        const auto partitionCount = PartitionData.size();
        std::vector<Rational> scalingFactors(partitionCount, Rational{-1U, 1});
        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto a = chosen[i];
            assert (a < candidateLength);
            if (a < numOfThreadLocalStreamSets) {
                auto & Ia = GC_Intervals[a];
                const auto p = partitionId[a];
                assert (p < PartitionData.size());
                const auto length = Ia.upper() - Ia.lower();
                assert (length > 0);
                assert (DataSizePerMinimumRepetition[a].numerator() > 0);
                const auto sf = Rational{length} / DataSizePerMinimumRepetition[a];
                scalingFactors[p] = std::min(scalingFactors[p], sf);
            }
        }

        return ResultVal{std::move(GC_Intervals), std::move(scalingFactors)};
    }



    BufferLayoutOptimizerWorker(const IntervalGraph & I
                               , const std::vector<Rational> & segmentsPerPage
                               , const std::vector<unsigned> & minimumNumOfPages
                               , const std::vector<unsigned> & partitionId
                               , const std::vector<unsigned> & unusedNodeRepetitions
                               , const size_t numOfThreadLocalStreamSets
                               , const size_t partitionCount
                               , pipeline_random_engine & rng)
    : I(I)
    , DataSizePerMinimumRepetition(segmentsPerPage), minimumNumOfPages(minimumNumOfPages)
    , partitionId(partitionId), unusedNodeRepetitions(unusedNodeRepetitions)
    , numOfThreadLocalStreamSets(numOfThreadLocalStreamSets)
    , GC_IntervalSet()
    , GC_Intervals(numOfThreadLocalStreamSets)
    , GC_EmptyIntervals(partitionCount)
    , PartitionData(partitionCount) {
        assert (numOfThreadLocalStreamSets <= num_vertices(I));
        assert (numOfThreadLocalStreamSets <= segmentsPerPage.size());
        assert (numOfThreadLocalStreamSets <= minimumNumOfPages.size());
        assert (numOfThreadLocalStreamSets <= partitionId.size());
    }

private:

    const IntervalGraph & I;
    const std::vector<Rational> & DataSizePerMinimumRepetition;
    const std::vector<unsigned> & minimumNumOfPages;
    const std::vector<unsigned> & partitionId;
    const std::vector<unsigned> & unusedNodeRepetitions;

    const size_t numOfThreadLocalStreamSets;


    IntervalSet GC_IntervalSet;
    std::vector<Interval> GC_Intervals;
    std::vector<IntervalSet> GC_EmptyIntervals;

    std::vector<Rational> MaxStridesPerStreamSet;

    std::vector<PartitionData> PartitionData;

};

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm<double> {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizerWorker::ResultVal translate(const OrderingDAWG & O, pipeline_random_engine & rng) {
        auto w = (BufferLayoutOptimizerWorker *)mainWorker.get();
        return w->translate(O, candidateLength, rng);
    }

    WorkerPtr makeWorker(pipeline_random_engine & rng) final {
        return std::make_unique<BufferLayoutOptimizerWorker>(I, segmentsPerPage, minimumNumOfPages, partitionId, unusedNodeRepetitions, numOfLocalStreamSets, partitionCount, rng);
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned candidateLength
                         , const unsigned numOfLocalStreamSets
                         , const size_t partitionCount
                         , const IntervalGraph & I
                         , const std::vector<Rational> & segmentsPerPage
                         , const std::vector<unsigned> & minimumNumOfPages
                         , const std::vector<unsigned> & partitionId
                         , const std::vector<unsigned> & unusedNodeRepetitions
                         , pipeline_random_engine & srcRng)
    : PermutationBasedEvolutionaryAlgorithm (candidateLength,
                                             BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS,
                                             BUFFER_SIZE_INIT_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_MAX_TIME_SECONDS,
                                             BUFFER_SIZE_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_STALLS,
                                             std::max(codegen::SegmentThreads, codegen::TaskThreads),
                                             srcRng)
    , numOfLocalStreamSets(numOfLocalStreamSets)
    , partitionCount(partitionCount)
    , I(I)
    , segmentsPerPage(segmentsPerPage)
    , minimumNumOfPages(minimumNumOfPages)
    , partitionId(partitionId)
    , unusedNodeRepetitions(unusedNodeRepetitions) {

    }


private:

    const size_t numOfLocalStreamSets;
    const size_t partitionCount;
    const IntervalGraph & I;
    const std::vector<Rational> & segmentsPerPage;
    const std::vector<unsigned> & minimumNumOfPages;
    const std::vector<unsigned> & partitionId;
    const std::vector<unsigned> & unusedNodeRepetitions;


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

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }

    // This process serves two purposes: (1) generate the initial memory layout for our thread-local
    // streamsets. (2) determine how many the number of pages to assign each streamset based on the
    // number of strides executed by the parition root.

    const auto n = LastStreamSet - FirstStreamSet + 1U;

    std::vector<unsigned> mapping(n);
    std::vector<Rational> unitWeight(n);
    std::vector<unsigned> minimumNumOfPages(n);
    std::vector<unsigned> streamSetPartitionId(n);

    auto & dl = b.getModule()->getDataLayout();

    const size_t pageSize = getPageSize();

    size_t numOfThreadLocalStreamSets = 0U;
    size_t packedPartitionCount = 0;
    unsigned largestStreamSet = 0;

    for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto firstKernel = FirstKernelInPartition[partitionId];
        const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

        const auto startThreadLocalStreamSetCount = numOfThreadLocalStreamSets;

        for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];

                if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                    mapping[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
                    streamSetPartitionId[numOfThreadLocalStreamSets] = packedPartitionCount;
                    Type * const type = bn.Buffer->getType();
                    const size_t typeSize = b.getTypeSize(dl, type);
                    const BufferPort & bp = mBufferGraph[output];
                    const auto & rate = bp.getRate().getUpperBound();
                    const auto W = rate * Rational{typeSize * StrideRepetitionVector[kernel], pageSize};
                    assert (W.numerator() > 0);
                    unitWeight[numOfThreadLocalStreamSets] = W;
                    assert ((MaximumNumOfStrides[kernel] % StrideRepetitionVector[kernel]) == 0);
                    assert (MaximumNumOfStrides[kernel] >= StrideRepetitionVector[kernel]);
                    const auto scale = MaximumNumOfStrides[kernel] / StrideRepetitionVector[kernel];
                    const auto d = ceiling(W * scale);
                    largestStreamSet = std::max(largestStreamSet, d);
                    minimumNumOfPages[numOfThreadLocalStreamSets] = d;
                    ++numOfThreadLocalStreamSets;
                }
            }
        }

        if (startThreadLocalStreamSetCount != numOfThreadLocalStreamSets) {
            ++packedPartitionCount;
        }

    }

    ThreadLocalExpansionThresholdFactor.resize(PartitionCount, Rational{0});

    if (numOfThreadLocalStreamSets) {

        const auto total = numOfThreadLocalStreamSets + packedPartitionCount;

        size_t candidateLength = numOfThreadLocalStreamSets;

        std::vector<unsigned> unusedNodeRepetitions(packedPartitionCount);

        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            const auto toAdd = largestStreamSet - minimumNumOfPages[i];
            const auto p = streamSetPartitionId[i];
            assert (p < packedPartitionCount);
            unusedNodeRepetitions[p] += toAdd;
            candidateLength += toAdd;
        }

        std::vector<int> remaining(n); // NOTE: signed int type is necessary here

        IntervalGraph I(total);

        unitWeight.resize(total, Rational{0});

        minimumNumOfPages.resize(total, 1);

        streamSetPartitionId.resize(total);

        for (unsigned i = numOfThreadLocalStreamSets; i < total; ++i) {
            streamSetPartitionId[i] = i - numOfThreadLocalStreamSets;
        }

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

            // Determine which streamsets are no longer alive
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        // We add +1 here because some outputs might be unused but still cannot
                        // reuse memory of an input buffer
                        remaining[j] = out_degree(streamSet, mBufferGraph) + 1;
                    }
                }

                // Mark any overlapping allocations in our interval graph.
                for (unsigned i = 0; i != numOfThreadLocalStreamSets; ++i) {
                    if (remaining[i] > 0) {
                        for (unsigned j = 0; j != i; ++j) {
                            if (remaining[j] > 0) {
                                add_edge(j, i, I);
                            }
                        }
                        const auto partId = streamSetPartitionId[i];
                        if (unusedNodeRepetitions[partId] > 0) {
                            add_edge(i, numOfThreadLocalStreamSets + partId, I);
                        }
                    }
                }



                // Undo the +1 for the produced buffers
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

        BufferLayoutOptimizer BA(candidateLength, numOfThreadLocalStreamSets, packedPartitionCount,
                                 I, unitWeight, minimumNumOfPages, streamSetPartitionId, unusedNodeRepetitions,
                                 rng);

        BA.runGA<true>();

        auto O = BA.getResult();

        const auto result = BA.translate(O, rng);

        size_t packedPartitionIndex = 0;

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                        assert (packedPartitionIndex < packedPartitionCount);
                        const Rational & S = result.ScalingFactors[packedPartitionIndex];
                        ThreadLocalExpansionThresholdFactor[partitionId] = S / StrideRepetitionVector[firstKernel];
                        ++packedPartitionIndex;
                        goto next_partition;
                    }
                }
            }
next_partition:
            continue;
        }
        assert (packedPartitionIndex == packedPartitionCount);

        bool hasThreadLocalInOut = false;

        unsigned requiredMemory{0};

        const auto & intervals = result.Intervals;

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const auto i = streamSet - FirstStreamSet;
            BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isThreadLocal()) {
                if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
                    hasThreadLocalInOut = true;
                } else {
                    const auto j = mapping[i];
                    assert (j < numOfThreadLocalStreamSets);
                    const auto & interval = intervals[j];
                    bn.BufferStart = interval.lower() * pageSize;
                    bn.BufferEnd = interval.upper() * pageSize;
                    requiredMemory = std::max(requiredMemory, bn.BufferEnd);
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

#endif

#if 0


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


                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal() && !bn.isInOutRedirect()) {
                    assert (in_degree(streamSet, InOutStreamSetReplacement) == 0);

                    // determine the number of bytes this streamset requires per *root kernel* stride
                    const BufferPort & producerRate = mBufferGraph[output];
                    const Binding & outputRate = producerRate.Binding;
                    Type * const type = StreamSetBuffer::resolveType(b, outputRate.getType());
                    const auto typeSize = b.getTypeSize(DL, type);
                    const auto j = mapping[streamSet - FirstStreamSet];
                    assert (j != -1U);
                    weight[j] = typeSize * bn.ExtimatedPerStrideCapacity;
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
    // TODO: track how many are alive at any one point and allocate that many pages of additional space
    // to ensure page aligned streamsets.
    auto requiredMemory = BA.getBestFitnessValue();
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
                bn.BufferEnd = interval.upper();
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

#endif

} // end of kernel namespace
