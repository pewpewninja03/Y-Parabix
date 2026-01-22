#include "pipeline_analysis.hpp"
#include "evolutionary_algorithm.hpp"
#include "lexographic_ordering.hpp"
#include <boost/icl/interval_set.hpp>
#include <boost/integer/common_factor.hpp>
#include <toolchain/toolchain.h>
#include <stack>

// #define PRINT_Z3_OPTIMIZATION

using boost::icl::interval_set;

namespace kernel {

// TODO: nested pipeline kernels could report how much internal memory they require
// and reason about that here (and in the scheduling phase)

constexpr static unsigned BUFFER_SIZE_INIT_POPULATION_SIZE = 15;

constexpr static unsigned BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS = 2;

constexpr static unsigned BUFFER_SIZE_POPULATION_SIZE = 30;

constexpr static unsigned BUFFER_SIZE_GA_MAX_TIME_SECONDS = 15;

constexpr static unsigned BUFFER_SIZE_GA_STALLS = 50;

using ConflictGraph = adjacency_list<hash_setS, vecS, undirectedS>;

using IntervalSet = interval_set<unsigned>;

using Interval = IntervalSet::interval_type;

using Vertex = unsigned;

struct BufferLayoutOptimizerWorker final : public PermutationBasedEvolutionaryAlgorithmWorker {


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
    size_t fitness(const Candidate & candidate, pipeline_random_engine & /* rng */) final {

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
                    GC_IntervalSet.add(GC_Intervals[b]);
                } else {
                    assert ("sanity check for undirected graph failed?" && !edge(b, a, I).second);
                }
            }

            const auto w = weight[a];

            size_t start = 0;
            size_t end = w;
            if (!GC_IntervalSet.empty()) {
                #ifndef NDEBUG
                auto ii = GC_IntervalSet.begin();
                assert (ii->lower() < ii->upper());
                auto lastUpper = ii->upper();
                while (++ii != GC_IntervalSet.end()) {
                    assert (lastUpper < ii->lower());
                    assert (ii->lower() < ii->upper());
                    lastUpper = ii->upper();
                }
                #endif
                for (const auto & interval : GC_IntervalSet) {
                    if (end <= interval.lower()) {
                        for (auto i = start; i < end; ++i) {
                            assert (!boost::icl::contains(GC_IntervalSet, i));
                        }
                        break;
                    } else {
                        const auto r = interval.upper();
                        start = r;
                        end = r + w;
                    }
                }
                GC_IntervalSet.clear();
            }

            assert (end > start);
            GC_Intervals[a] = Interval::right_open(start, end);

            const auto p = partitionId[a];
            assert (p < partitionId.size());
            auto & M = MaxMemorySize[p];
            M = std::max(M, end);
        }

        std::sort(MaxMemorySize.begin(), MaxMemorySize.end());

        const auto idx = (partitionCount / 2);
        auto median = MaxMemorySize[idx];
        if ((partitionCount % 2) == 0) {
            const auto idx2 = ((partitionCount - 1) / 2);
            assert (idx != idx2);
            median = (median + MaxMemorySize[idx2] + 1) / 2;
        }
        const auto max = MaxMemorySize[partitionCount - 1];
        return median + (max * max);
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
        assert (chosen.size() == candidateLength);
        fitness(chosen, rng);
        return GC_Intervals;
    }

    BufferLayoutOptimizerWorker(const unsigned candidateLength
                               , const size_t partitionCount
                               , const ConflictGraph & I
                               , const std::vector<size_t> & weight
                               , const std::vector<unsigned> & partitionId
                               , pipeline_random_engine & rng)
    : I(I)
    , weight(weight)
    , partitionId(partitionId)
    , GC_IntervalSet()
    , GC_Intervals(candidateLength)
    , MaxMemorySize(partitionCount) {

    }

private:

    const ConflictGraph & I;
    const std::vector<size_t> & weight;
    const std::vector<unsigned> & partitionId;

    IntervalSet GC_IntervalSet;
    std::vector<Interval> GC_Intervals;
    std::vector<size_t> MaxMemorySize;

};

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm {

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
                         , const ConflictGraph & I
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

    const ConflictGraph & I;
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

    std::vector<unsigned> mapStreamSetToThreadLocal(n);
    std::vector<size_t> unitWeight(n);
    std::vector<size_t> overflowWeight(n);
    std::vector<unsigned> streamSetPartitionId(n);

    auto & dl = b.getModule()->getDataLayout();

    size_t numOfThreadLocalStreamSets = 0U;
    size_t packedPartitionCount = 0;

    #ifdef PRINT_Z3_OPTIMIZATION
    errs() << " -- starting thread local layout\n";
    #endif

    for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto firstKernel = FirstKernelInPartition[partitionId];
        const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

        const auto startThreadLocalStreamSetCount = numOfThreadLocalStreamSets;

        for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];

                if (bn.isThreadLocal()) {
                    if (LLVM_UNLIKELY(bn.isInOutRedirect())) {
                        auto src = parent(streamSet, InOutStreamSetReplacement);
                        while (LLVM_UNLIKELY(in_degree(src, InOutStreamSetReplacement) != 0)) {
                            src = parent(src, InOutStreamSetReplacement);
                            assert (FirstStreamSet <= src && src <= LastStreamSet);
                        }
                        assert (FirstStreamSet < src && src < streamSet);
                        const auto k = mapStreamSetToThreadLocal[src - FirstStreamSet];
                        assert (unitWeight[k] > 0);
                        mapStreamSetToThreadLocal[streamSet - FirstStreamSet] = k;
                    } else {
                        mapStreamSetToThreadLocal[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
                        streamSetPartitionId[numOfThreadLocalStreamSets] = packedPartitionCount;
                        Type * const type = bn.Buffer->getType();
                        const size_t typeSize = b.getTypeSize(dl, type);
                        const BufferPort & bp = mBufferGraph[output];
                        const auto W = bp.Maximum * typeSize * StrideRepetitionVector[kernel];
                        assert (W.denominator() == 1);
                        unitWeight[numOfThreadLocalStreamSets] = W.numerator();
                        overflowWeight[numOfThreadLocalStreamSets] = bn.NumOfOverflowStrides;
                        ++numOfThreadLocalStreamSets;
                    }

                }
            }
        }

        if (startThreadLocalStreamSetCount != numOfThreadLocalStreamSets) {
            ++packedPartitionCount;
        }

    }

    const auto m = PartitionCount + n;

    ThreadLocalPlacementGraph T(m + 1U);

    if (numOfThreadLocalStreamSets) {

        ConflictGraph I(numOfThreadLocalStreamSets);

        std::vector<unsigned> remaining(numOfThreadLocalStreamSets, 0);
        std::vector<unsigned> mapThreadLocalToStreamSet(numOfThreadLocalStreamSets);

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

            #ifdef PREVENT_THREAD_LOCAL_BUFFERS_FROM_SHARING_MEMORY
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        if (LLVM_LIKELY(!bn.isInOutRedirect())) {
                            const auto k = PartitionCount + streamSet - FirstStreamSet;
                            reverse_mapping[j] = k;  assert (k > 0);
                        }
                        remaining[j] = firstKernel + 1;
                    }
                }
            }
            for (unsigned i = 1; i < numOfThreadLocalStreamSets; ++i) {
                const auto id = remaining[i];
                if (id == (firstKernel + 1)) {
                    for (unsigned j = 0; j < i; ++j) {
                        if (remaining[j] == id) {
                            add_edge(j, i, I);
                        }
                    }
                }
            }
            #else

            // Determine which streamsets are no longer alive
            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapStreamSetToThreadLocal[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (remaining[j] == (bn.isInOutRedirect() ? 1U : 0));
                        remaining[j] += 1U;
                        if (LLVM_LIKELY(!bn.isInOutRedirect())) {
                            mapThreadLocalToStreamSet[j] = streamSet;
                        }
                    }
                }

                // Mark any overlapping allocations in our interval graph.
                for (unsigned i = 1; i != numOfThreadLocalStreamSets; ++i) {
                    if (remaining[i]) {
                        for (unsigned j = 0; j != i; ++j) {
                            if (remaining[j]) {
                                add_edge(j, i, I);
                            }
                        }
                    }
                }

                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapStreamSetToThreadLocal[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapStreamSetToThreadLocal[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (remaining[j] > 0);
                        remaining[j] += out_degree(streamSet, mBufferGraph) - 1U;
                    }
                }
            }
            #endif
        }

        #if !defined(NDEBUG) && !defined(PREVENT_THREAD_LOCAL_BUFFERS_FROM_SHARING_MEMORY)
        for (size_t i = 0; i < numOfThreadLocalStreamSets; ++i) {
            assert (remaining[i] == 0);
            assert (in_degree(mapThreadLocalToStreamSet[i], InOutStreamSetReplacement) == 0);
        }
        #endif

        ThreadLocalConflictGraph = ThreadLocalConflictGraphType(n);

        for (auto e : make_iterator_range(edges(I))) {

            std::function<void(size_t, size_t)> add_conflict_edge = [&](const size_t u, const size_t v) {
                assert (FirstStreamSet <= u && u <= LastStreamSet);
                assert (FirstStreamSet <= v && v <= LastStreamSet);
                assert (in_degree(u, InOutStreamSetReplacement) == 0);
                assert (in_degree(v, InOutStreamSetReplacement) == 0);
                add_edge(u - FirstStreamSet, v - FirstStreamSet, ThreadLocalConflictGraph);
                if (LLVM_UNLIKELY(out_degree(u, InOutStreamSetReplacement) > 0)) {
                    add_conflict_edge(child(u, InOutStreamSetReplacement), v);
                }
                if (LLVM_UNLIKELY(out_degree(v, InOutStreamSetReplacement) > 0)) {
                    add_conflict_edge(u, child(v, InOutStreamSetReplacement));
                }
            };

            add_conflict_edge(mapThreadLocalToStreamSet[source(e, I)], mapThreadLocalToStreamSet[target(e, I)]);

        }

        #ifdef PRINT_Z3_OPTIMIZATION
        BEGIN_SCOPED_REGION
            auto & out = errs();
            out << "digraph \"" << "I" << "\" {\n";
            for (unsigned i = 0; i < n; ++i) {
                out << "v" << i << " [label=\"";
                out << "S_" << (FirstStreamSet + i);
                out << "\"];\n";
            }
            for (const auto e : make_iterator_range(edges(ThreadLocalConflictGraph))) {
                const auto s = source(e, ThreadLocalConflictGraph);
                const auto t = target(e, ThreadLocalConflictGraph);
                out << "v" << s << " -> v" << t << ";\n";
            }
            out << "}\n\n";
        END_SCOPED_REGION
        #endif

        BufferLayoutOptimizer BA(numOfThreadLocalStreamSets,
                                 packedPartitionCount,
                                 I, unitWeight, streamSetPartitionId,
                                 rng);

        BA.runGA();

        auto O = BA.getResult();

        #ifdef PRINT_Z3_OPTIMIZATION
        errs() << " -- finished thread local layout genetic algorithm phase\n";
        #endif

        const auto intervals = BA.translate(O, rng);
        assert (intervals.size() == numOfThreadLocalStreamSets);

        const auto pageSize = getPageSize();
        const auto bw = b.getBitBlockWidth();


        Rational::int_type denomLCM = 1U;

        auto add_edge_to_T = [&](const size_t u, const size_t v, const size_t weight) {

            assert (u < PartitionCount + n);
            assert (FirstStreamSet <= v && v <= LastStreamSet);

            const auto producer = parent(v, mBufferGraph);
            assert (FirstKernel <= producer && producer <= LastKernel);
            const auto partId = KernelPartitionId[producer];
            assert (partId < PartitionCount);
            const auto firstKernel = FirstKernelInPartition[partId];

            Rational percentOfPagePerStride{weight, StrideRepetitionVector[firstKernel] * pageSize * bw};
            denomLCM = boost::integer::lcm(denomLCM, percentOfPagePerStride.denominator());
            const auto w = PartitionCount + v - FirstStreamSet;
            assert (w < PartitionCount + n);
            assert (!edge(u, w, T).second);
            add_edge(u, w, percentOfPagePerStride, T);

            #ifndef NDEBUG
            for (auto e : make_iterator_range(in_edges(w, T))) {
                assert (T[e] == percentOfPagePerStride);
            }
            #endif
        };

        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            const auto streamSet = mapThreadLocalToStreamSet[i];
            assert (!mBufferGraph[streamSet].isInOutRedirect());
            const auto & C = intervals[i];
            #ifndef NDEBUG
            assert (C.upper() > C.lower());
            #endif
            if (C.lower() == 0) {
                assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                const auto producer = parent(streamSet, mBufferGraph);
                assert (FirstKernel <= producer && producer <= LastKernel);
                const auto partId = KernelPartitionId[producer];
                assert (partId < PartitionCount);
                add_edge_to_T(partId, streamSet, unitWeight[i]);
            }
        }

        for (auto e : make_iterator_range(edges(I))) {
            const auto a = source(e, I);
            assert (a < numOfThreadLocalStreamSets);
            const auto & A = intervals[a];
            const auto b = target(e, I);
            assert (b < numOfThreadLocalStreamSets);
            const auto & B = intervals[b];

            assert (disjoint(A, B));

            auto make_edge = [&](const size_t i, const size_t j) {
                const auto u = mapThreadLocalToStreamSet[i];
                const auto v = mapThreadLocalToStreamSet[j];
                add_edge_to_T(PartitionCount + u - FirstStreamSet, v, unitWeight[j]);
            };

            if (A.lower() < B.lower()) {
                assert (A.upper() <= B.lower());
                assert (B.lower() > 0);
                make_edge(a, b);
            } else {
                assert (B.upper() <= A.lower());
                assert (A.lower() > 0);
                make_edge(b, a);
            }
        }

        #ifdef PRINT_Z3_OPTIMIZATION
        const ThreadLocalPlacementGraph T0(T);
        #endif

        // We need to compute both the most-expensive *longest* partition -> sink path and the most-expensive path,
        // which are not necessarily the same path.

        std::stack<Vertex> S;
        std::vector<size_t> unvistedAncestors(m + 1);
        std::vector<size_t> depth(m + 1);
        std::vector<Rational::int_type> pathCost(m + 1);
        std::vector<size_t> partitionId(m + 1);
        for (unsigned i = 0; i < PartitionCount; ++i) {
            depth[i] = 1;
            unvistedAncestors[i] = 0;
        }
        for (unsigned i = PartitionCount; i < m; ++i) {
            #ifndef NDEBUG
            depth[i] = 0;
            #endif
            const auto a = in_degree(i, T);
            unvistedAncestors[i] = a;
            if (a != 0 && out_degree(i, T) == 0) {
                add_edge(i, m, Rational{0}, T);
            }
        }
        unvistedAncestors[m] = in_degree(m, T);
        assert (unvistedAncestors[m] > 0);
        for (unsigned partId = 0; partId < PartitionCount; ++partId) {
            assert (unvistedAncestors[partId] == 0);
            partitionId[partId] = partId;
            pathCost[partId] = 0;
            if (out_degree(partId, T) > 0) {
                assert (in_degree(partId, T) == 0);
                for (auto u = partId;;) {
                    for (auto e : make_iterator_range(out_edges(u, T))) {
                        const auto v = target(e, T);
                        auto & U = unvistedAncestors[v];
                        assert (U > 0);
                        if (--U == 0) {
                            S.push(v);
                        }
                    }
                    if (S.empty()) {
                        break;
                    }
                    u = S.top();
                    assert (PartitionCount <= u && u <= m);
                    assert (unvistedAncestors[u] == 0);
                    S.pop();
                    assert (in_degree(u, T) > 0);
                    size_t d = 0;
                    Rational::int_type pc{0};
                    for (auto e : make_iterator_range(in_edges(u, T))) {
                        const auto v = source(e, T);
                        assert (depth[v] > 0);
                        d = std::max(d, depth[v]);
                        const auto c = T[e] * denomLCM;
                        assert (c.denominator() == 1);
                        const auto x = c.numerator() * c.numerator();
                        assert ("overflow?" && (x > c.numerator() || c.numerator() <= 1));
                        pc = std::max(pc, pathCost[v] + x);
                    }
                    assert (pc > 0);
                    depth[u] = d + 1U;
                    pathCost[u] = pc;

                    partitionId[u] = partId;
                }
            }
        }

        #ifndef NDEBUG
        for (unsigned i = PartitionCount; i <= m; ++i) {
            assert (unvistedAncestors[i] == 0);
        }
        #endif
        // Before we prune the graph, mark which "sinks" for each partition component that
        // will be used to calculate the total memory required

        for (unsigned i = 0; i <= m; ++i) {
            T[i] = false;
        }

        for (unsigned partId = 0; partId < PartitionCount; ++partId) {
            if (out_degree(partId, T) > 0) {
                size_t maxWeight{0};
                size_t maxWeightDepth = 0;
                size_t maxDepth = 0;
                size_t maxDepthWeight{0};
                size_t sink1 = m;
                size_t sink2 = m;

                for (auto e : make_iterator_range(in_edges(m, T))) {
                    const auto u = source(e, T);
                    assert (u >= PartitionCount);
                    if (partId != partitionId[u]) {
                        continue;
                    }
                    const auto C = pathCost[u];
                    const auto d = depth[u];
                    if (maxWeight <= C) {
                        if (maxWeightDepth < d || maxWeight < C) {
                            maxWeightDepth = d;
                            maxWeight = C;
                            sink1 = u;
                        }
                    }
                    if (maxDepth <= d) {
                        if (maxDepth < d || maxDepthWeight < C) {
                            maxDepth = d;
                            maxDepthWeight = C;
                            sink2 = u;
                        }
                    }
                }

                assert (depth[sink1] <= depth[sink2]);
                assert (pathCost[sink2] <= pathCost[sink1]);

                T[sink1] = true;
                T[sink2] = true;
            }
        }

        for (auto u = m; u >= PartitionCount; --u) {
            if (in_degree(u, T) > 1U) {
                const auto W = pathCost[u];
                const auto du = depth[u];
                size_t maxWeightDepth = 0;
                size_t maxDepthWeight{0};
                size_t keep1 = -1U;
                size_t keep2 = -1U;

                for (auto e : make_iterator_range(in_edges(u, T))) {
                    const auto v = source(e, T);
                    const auto & C = pathCost[v];
                    const auto dv = depth[v];
                    assert (C < W || u == m);

                    const auto c = T[e] * denomLCM;
                    assert (c.denominator() == 1);
                    const auto x = c.numerator() * c.numerator();
                    const auto X = C + x;
                    assert (X <= W);
                    // is this edge on a heaviest path?
                    if (X == W) {
                        // if so is it a longest-heaviest path?
                        if (maxWeightDepth < dv) {
                            maxWeightDepth = dv;
                            keep1 = v;
                        }
                    }
                    // is this edge on a longest path?
                    if ((dv + 1U) == du) {
                        // if so is it a heaviest-longest path?
                        if (maxDepthWeight < X) {
                            maxDepthWeight = X;
                            keep2 = v;
                        }
                    }
                }
                assert (keep1 != -1U || keep2 != -1U);
                remove_in_edge_if(u, [&](const ThreadLocalPlacementGraph::edge_descriptor e) -> bool {
                    const auto v = source(e, T);
                    return (v >= PartitionCount) && (v != keep1) && (v != keep2);
                }, T);
            }
        }

        #ifdef PRINT_Z3_OPTIMIZATION
        BEGIN_SCOPED_REGION
        auto & out = errs();
        out << "digraph \"" << "T" << "\" {\n";
        for (unsigned i = 0; i < PartitionCount + n; ++i) {
            if (degree(i, T) > 0) {
                out << "v" << i << " [label=\"";
                if (i < PartitionCount) {
                    out << "P_" << i;
                } else {
                    out << "S_" << (FirstStreamSet + i - PartitionCount);

                    const Rational C{pathCost[i], denomLCM};

                    out << " (W:" << C.numerator() << "/" << C.denominator() << " d:" << depth[i] << ")";

                    if (T[i]) {
                        out << '*';
                    }


                }


                out << "\"];\n";
            }
        }

        for (const auto e : make_iterator_range(edges(T0))) {
            const auto s = source(e, T0);
            const auto t = target(e, T0);
            const auto & V = T0[e];
            out << "v" << s << " -> v" << t <<
                   " [label=\"" << V.numerator() << "/" << V.denominator() << "\"";
            if (!edge(s, t, T).second) {
                out << ", color=\"red\"";
            }
            out << "];\n";
        }

        for (const auto e : make_iterator_range(edges(T))) {
            const auto s = source(e, T);
            const auto t = target(e, T);
            if (!edge(s, t, T0).second) {
                const auto & V = T[e];
                out << "v" << s << " -> v" << t <<
                       " [label=\"" << V.numerator() << "/" << V.denominator() << "\"";
                out << ", color=\"green\"";
                out << "];\n";

            }
        }

        out << "}\n\n";
        END_SCOPED_REGION
        #endif
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
            assert (bn.Locality != BufferLocality::ConstantShared);
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
