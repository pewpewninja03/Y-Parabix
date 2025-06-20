#include "pipeline_analysis.hpp"
#include "evolutionary_algorithm.hpp"
#include "lexographic_ordering.hpp"
#include <boost/icl/interval_set.hpp>
#include <boost/integer/common_factor.hpp>
#include <toolchain/toolchain.h>
#include <z3.h>
#include <queue>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

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

        auto median = MaxMemorySize[partitionCount / 2];
        if ((partitionCount & 2) == 0) {
            median = (median + MaxMemorySize[(partitionCount / 2) + 1] + 1) / 2;
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

    std::vector<unsigned> mapping(n);
    std::vector<size_t> unitWeight(n);
    std::vector<unsigned> streamSetPartitionId(n);

    auto & dl = b.getModule()->getDataLayout();

    size_t numOfThreadLocalStreamSets = 0U;
    size_t packedPartitionCount = 0;

    const auto bw = b.getBitBlockWidth();

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
                        const auto k = mapping[src - FirstStreamSet];
                        assert (unitWeight[k] > 0);
                        mapping[streamSet - FirstStreamSet] = k;
                    } else {
                        mapping[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
                        streamSetPartitionId[numOfThreadLocalStreamSets] = packedPartitionCount;
                        Type * const type = bn.Buffer->getType();
                        const size_t typeSize = b.getTypeSize(dl, type);
                        const BufferPort & bp = mBufferGraph[output];
                        const auto & M = bp.Maximum;
                        const auto W = Rational{M.numerator() * typeSize * StrideRepetitionVector[kernel], M.denominator() * bw};
                        assert (W.numerator() > 0);
                        assert (W.denominator() == 1);
                        unitWeight[numOfThreadLocalStreamSets] = W.numerator();
                        ++numOfThreadLocalStreamSets;
                    }

                }
            }
        }

        if (startThreadLocalStreamSetCount != numOfThreadLocalStreamSets) {
            ++packedPartitionCount;
        }

    }

    ThreadLocalPlacementGraph T(PartitionCount + n + 1);

    if (numOfThreadLocalStreamSets) {

        ConflictGraph I(numOfThreadLocalStreamSets);

        std::vector<unsigned> remaining(numOfThreadLocalStreamSets, 0);
        std::vector<unsigned> reverse_mapping(numOfThreadLocalStreamSets, 0);

        for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
            const auto firstKernel = FirstKernelInPartition[partitionId];
            const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];

            // Determine which streamsets are no longer alive
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
                        // We add +1 here because some outputs might be unused but still cannot
                        // reuse memory of an input buffer
                        assert ((remaining[j] == 0) ^ bn.isInOutRedirect());
                        remaining[j] += out_degree(streamSet, mBufferGraph) + 1;
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

                // Undo the +1 for the produced buffers
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }

                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j < numOfThreadLocalStreamSets);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }
            }
        }

        ThreadLocalConflictGraph = ThreadLocalConflictGraphType(LastStreamSet - FirstStreamSet + 1);

        for (auto e : make_iterator_range(edges(I))) {
            auto sid = [&](const unsigned i) {
                return reverse_mapping[i] - PartitionCount;
            };
            add_edge(sid(source(e, I)), sid(target(e, I)), ThreadLocalConflictGraph);
        }

        #ifndef NDEBUG
        for (size_t i = 0; i < numOfThreadLocalStreamSets; ++i) {
            assert (remaining[i] == 0);
        }
        #endif

        BufferLayoutOptimizer BA(numOfThreadLocalStreamSets,
                                 packedPartitionCount,
                                 I, unitWeight, streamSetPartitionId,
                                 rng);

        BA.runGA();

        auto O = BA.getResult();



        const auto intervals = BA.translate(O, rng);
        assert (intervals.size() == numOfThreadLocalStreamSets);

        const auto pageSize = getPageSize();

        size_t lcmOfDenom = 1U;

        auto add_edge_to_T = [&](const size_t u, const size_t v, const unsigned num, const unsigned denom) {
            assert (!edge(u, v, T).second);
            Rational percentOfPagePerStride{num, denom * pageSize};
            lcmOfDenom = boost::integer::lcm(lcmOfDenom, percentOfPagePerStride.denominator());
            add_edge(u, v, percentOfPagePerStride, T);
            #ifndef NDEBUG
            for (auto e : make_iterator_range(in_edges(v, T))) {
                assert (T[e] == percentOfPagePerStride);
            }
            #endif
        };

        for (unsigned i = 0; i < numOfThreadLocalStreamSets; ++i) {
            const auto u = reverse_mapping[i];
            const auto & C = intervals[i];
            #ifndef NDEBUG
            assert (C.upper() > C.lower());
            #endif
            if (C.lower() == 0) {
                const auto w = FirstStreamSet + u - PartitionCount;
                assert (FirstStreamSet <= w && w <= LastStreamSet);
                const auto producer = parent(w, mBufferGraph);
                assert (FirstKernel <= producer && producer <= LastKernel);
                const auto partId = KernelPartitionId[producer];
                assert (partId < PartitionCount);
                const auto firstKernel = FirstKernelInPartition[partId];
                add_edge_to_T(partId, u, C.upper(), StrideRepetitionVector[firstKernel]);
            }
            const auto & S = StreamSetIORate[u - PartitionCount];
            lcmOfDenom = boost::integer::lcm(lcmOfDenom, S.denominator());
        }

        for (auto e : make_iterator_range(edges(I))) {
            const auto a = source(e, I);
            assert (a < numOfThreadLocalStreamSets);
            const auto & A = intervals[a];
            const auto b = target(e, I);
            assert (b < numOfThreadLocalStreamSets);
            const auto & B = intervals[b];

            assert (disjoint(A, B));

            auto make_edge = [&](const size_t i, const size_t j, const Interval & C) {
                const auto u = reverse_mapping[i];
                const auto v = reverse_mapping[j];
                const auto w = FirstStreamSet + v - PartitionCount;
                assert (FirstStreamSet <= w && w <= LastStreamSet);
                const auto producer = parent(w, mBufferGraph);
                assert (FirstKernel <= producer && producer <= LastKernel);
                const auto partId = KernelPartitionId[producer];
                assert (partId < PartitionCount);
                const auto firstKernel = FirstKernelInPartition[partId];
                add_edge_to_T(u, v, C.upper() - C.lower(), StrideRepetitionVector[firstKernel]);
            };

            if (A.lower() < B.lower()) {
                assert (A.upper() <= B.lower());
                make_edge(a, b, B);
            } else {
                assert (B.upper() <= A.lower());
                make_edge(b, a, A);
            }
        }

        transitive_reduction_dag(T);

        #ifdef PRINT_Z3_OPTIMIZATION
        const ThreadLocalPlacementGraph T0(T);
        #endif

        // TODO: this is probably too complex a solution. is it not equivalent to simply check the path lengths
        // and conclude if one is longer, then with a small segment length, it'd be preferred over the other in
        // that case? Similarly, if we take two paths and sort their weights, we could say one is bigger than
        // with a large segment size whenever any i-th edge weight is larger in one than the other.

        // TODO: just have them loop back to the partition root to indicate final cost?

        std::vector<unsigned> sinks;

        for (unsigned i = 0; i < PartitionCount; ++i) {
            T[i] = false;
        }
        for (unsigned i = 0; i < n; ++i) {
            const auto u = PartitionCount + i;
            const bool isSink = (out_degree(u, T) == 0 && in_degree(u, T) != 0);
            T[u] = isSink;
            if (isSink) {
                sinks.push_back(u);
            }
        }

        const auto cfg = Z3_mk_config();
        Z3_set_param_value(cfg, "model", "false");
        Z3_set_param_value(cfg, "proof", "false");
        Z3_set_param_value(cfg, "timeout", "2000");
        const auto ctx = Z3_mk_context(cfg);
        Z3_del_config(cfg);
        const auto solver = Z3_mk_solver(ctx);
        Z3_solver_inc_ref(ctx, solver);


        const auto varType = Z3_mk_int_sort(ctx);

        auto hard_assert = [&](Z3_ast c) {
            Z3_solver_assert(ctx, solver, c);
        };

        auto check = [&]() -> Z3_lbool {
            return Z3_solver_check(ctx, solver);
        };

        auto constant = [&](const size_t value) {
            return Z3_mk_int(ctx, value, varType);
        };

        const Z3_ast z3_ZERO = constant(0);

        const Z3_ast z3_LCM_OF_DENOM = constant(lcmOfDenom);

        std::queue<unsigned> Q;

        std::vector<unsigned> pending(n);

        for (unsigned i = 0; i < n; ++i) {
            pending[i] = in_degree(PartitionCount + i, T);
        }

        std::vector<Z3_ast> val(PartitionCount + n);

        std::vector<Rational> totalPathSum(n);
        std::vector<unsigned> totalPathLength(n);

        flat_map<unsigned, Z3_ast> M;

        auto to_int = [&](const Rational & R) -> unsigned {
            const auto V = R * lcmOfDenom;
            assert (V.denominator() == 1);
            return V.numerator();
        };

        // TODO: preprocess the io scaling vars to determine if two partition vars are equivalent?

        using Vertex = ThreadLocalPlacementGraph::vertex_descriptor;

        auto identifyLowerValueAncestors = [&](auto & ancestors, auto & result) {
            // sort ancestors in decending sum/length cost
            std::sort(ancestors.begin(), ancestors.end(), [&](const Vertex a, const Vertex b) {
                assert (a >= PartitionCount);
                const auto & A = totalPathSum[a - PartitionCount];
                assert (b >= PartitionCount);
                const auto & B = totalPathSum[b - PartitionCount];
                if (A > B) {
                    return true;
                } else if (A == B) {
                    return totalPathLength[a - PartitionCount] > totalPathLength[b - PartitionCount];
                } else {
                    return false;
                }
            });

            const auto m = ancestors.size();

            assert (result.size() == m);

            for (unsigned i = 0; i != m; ++i) {
                result[i] = val[ancestors[i]]; assert (result[i]);
            }

            for (unsigned i = 1; i != m; ++i) {
                for (unsigned j = 0; j != i; ++j) {

                    if (result[j] == nullptr) {
                        continue; // path_j was already pruned as being strictly less than some other path
                    }

                    Z3_solver_push(ctx, solver);
                    hard_assert(Z3_mk_lt(ctx, result[j], result[i]));
                    const Z3_lbool JltI = check();
                    Z3_solver_pop(ctx, solver, 1);

                    Z3_solver_push(ctx, solver);
                    hard_assert(Z3_mk_lt(ctx, result[i], result[j]));
                    const Z3_lbool IltJ = check();
                    Z3_solver_pop(ctx, solver, 1);

                    // If there exists an assignment of var s.t. path_j >= path_i and
                    // some different assignment of var s.t. path_i >= path_j, we must
                    // test both paths at runtime to determine the capacity required.

                    if (JltI == Z3_L_FALSE && IltJ != Z3_L_UNDEF) {
                        // for all assignments of var, path_j >= path_i
                        result[i] = nullptr;
                        break;
                    } else if (IltJ == Z3_L_FALSE && JltI == Z3_L_TRUE) {
                        // for all assignments of var, path_i > path_j
                        result[j] = nullptr;
                    }
                }
            }

        };

        for (unsigned i = 0; i < PartitionCount; ++i) {
            if (out_degree(i, T) > 0) {

                const Z3_ast var = Z3_mk_fresh_const(ctx, nullptr, varType);
                hard_assert(Z3_mk_gt(ctx, var, z3_ZERO));
                val[i] = var;

                assert (M.empty());

                auto mk_ceil_var_mul = [&](const unsigned V) {
                    auto f = M.find(V);
                    if (f != M.end()) {
                        return f->second;
                    }
                    Z3_ast v = nullptr;
                    FixedArray<Z3_ast, 2> args;
                    args[0] = var;
                    args[1] = constant(V);
                    v = Z3_mk_mul(ctx, 2, args.data());
                    if (LLVM_LIKELY(V != lcmOfDenom)) {
                        const Z3_ast r = Z3_mk_mod(ctx, v, z3_LCM_OF_DENOM);
                        args[0] = v;
                        args[1] = r;
                        const Z3_ast a = Z3_mk_sub(ctx, 2, args.data());
                        args[0] = a;
                        args[1] = z3_LCM_OF_DENOM;
                        const Z3_ast b = Z3_mk_add(ctx, 2, args.data());
                        v = Z3_mk_ite(ctx, Z3_mk_eq(ctx, r, z3_ZERO), a, b);
                    }
                    M.emplace(V, v);
                    return v;
                };

                assert (Q.empty());
                for (auto e : make_iterator_range(out_edges(i, T))) {
                    const auto v = target(e, T);
                    assert (v >= PartitionCount);
                    const auto k = v - PartitionCount;
                    assert (pending[k] == 1);
                    pending[k] = 0;
                    const Rational & R = T[e];
                    totalPathSum[k] = R;
                    totalPathLength[k] = 0;
                    val[v] = mk_ceil_var_mul(to_int(R));
                    Q.push(v);
                }

                for (;;) {
                    assert (!Q.empty());
                    const auto u = Q.front(); Q.pop();
                    for (auto e : make_iterator_range(out_edges(u, T))) {
                        const auto v = target(e, T);
                        assert (v >= PartitionCount);
                        const auto k = v - PartitionCount;
                        assert (pending[k] > 0);
                        if (--pending[k] == 0) {
                            const auto d = in_degree(v, T);
                            assert (d > 0);
                            Z3_ast priorOffset = nullptr;

                            Rational priorPathSum{0};
                            unsigned priorPathLength{0};

                            if (LLVM_LIKELY(d == 1)) {
                                priorPathSum = totalPathSum[u - PartitionCount];
                                priorPathLength = totalPathLength[u - PartitionCount];
                                priorOffset = val[u];
                            } else {
                                ThreadLocalPlacementGraph::in_edge_iterator begin, end;
                                std::tie(begin, end) = in_edges(v, T);
                                SmallVector<Vertex, 4> ancestor(d);
                                auto ei = begin;
                                for (unsigned i = 0; i != d; ++i, ++ei) {
                                    ancestor[i] = source(*ei, T);
                                    assert (ancestor[i] >= PartitionCount);
                                }

                                SmallVector<Z3_ast, 4> prior(d);

                                identifyLowerValueAncestors(ancestor, prior);

                                auto ej = begin;
                                for (unsigned i = 0; i != d; ++i) {
                                    auto ek = ej++;
                                    if (prior[i]) {
                                        if (priorOffset) {
                                            priorOffset = Z3_mk_ite(ctx, Z3_mk_gt(ctx, prior[i], priorOffset), prior[i], priorOffset);
                                        } else {
                                            priorOffset = prior[i];
                                        }
                                        const auto w = ancestor[i] - PartitionCount;
                                        priorPathSum = std::max(priorPathSum, totalPathSum[w]);
                                        priorPathLength = std::max(priorPathLength, totalPathLength[w]);
                                    } else {
                                        // prune the edge from the graph since we do not need to consider
                                        // this path when determining the memory placement of this buffer
                                        remove_edge(ancestor[i], v, T);
                                    }
                                }
                            }

                            FixedArray<Z3_ast, 2> args;
                            args[0] = priorOffset;
                            args[1] = mk_ceil_var_mul(to_int(T[e]));
                            val[v] = Z3_mk_add(ctx, 2, args.data());
                            totalPathSum[k] = priorPathSum + T[e];
                            totalPathLength[k] = priorPathLength + 1U;
                            Q.push(v);
                        }
                    }
                    if (Q.empty()) {
                        break;
                    }
                }

                M.clear();
            }
        }

        const auto l = sinks.size();
        if (LLVM_UNLIKELY(l == 1)) {

            add_edge(sinks[0], PartitionCount + n, Rational{0}, T);

        } else { assert (l > 0);

            const Z3_ast ioScalingVar = Z3_mk_fresh_const(ctx, nullptr, varType);

            for (unsigned i = 0; i < PartitionCount; ++i) {
                if (out_degree(i, T) > 0) {
                    const auto firstKernel = FirstKernelInPartition[i];
                    Z3_ast scalingVar = ioScalingVar;
                    if (in_degree(firstKernel, mBufferGraph) > 0) {
                        Rational maxInputScale{0};
                        for (auto input : make_iterator_range(in_edges(firstKernel, mBufferGraph))) {
                            const auto streamSet = source(input, mBufferGraph);
                            maxInputScale = std::max(maxInputScale, StreamSetIORate[streamSet - FirstStreamSet]);
                        }
                        if (maxInputScale.numerator() != 1 || maxInputScale.denominator() != 1) {
                            FixedArray<Z3_ast, 2> args;
                            args[0] = ioScalingVar;
                            args[1] = constant(to_int(maxInputScale));
                            scalingVar = Z3_mk_mul(ctx, 2, args.data());
                        }
                    }
                    hard_assert(Z3_mk_ge(ctx, scalingVar, constant(MaximumNumOfStrides[firstKernel] * lcmOfDenom)));
                    hard_assert(Z3_mk_eq(ctx, scalingVar, val[i]));
                }
            }

            std::vector<Z3_ast> sinkval(l);

            identifyLowerValueAncestors(sinks, sinkval);

            for (unsigned i = 0; i != l; ++i) {
                if (sinkval[i]) {
                    add_edge(sinks[i], PartitionCount + n, Rational{0}, T);
                }
            }

            assert (in_degree(PartitionCount + n, T) > 0);
        }

        Z3_solver_dec_ref(ctx, solver);
        Z3_del_context(ctx);
        Z3_reset_memory();

        #ifdef PRINT_Z3_OPTIMIZATION
        auto print_T = [&](StringRef name) {
            auto & out = errs();

            out << "digraph \"" << name << "\" {\n";
            for (unsigned i = 0; i < PartitionCount + n; ++i) {
                if (degree(i, T) > 0) {
                    out << "v" << i << " [label=\"";
                    if (i < PartitionCount) {
                        out << "P_" << i;
                    } else {
                        out << "S_" << (FirstStreamSet + i - PartitionCount);
                    }
                    out << "\"];\n";
                }
            }

            for (const auto e : make_iterator_range(edges(T0))) {
                const auto s = source(e, T);
                const auto t = target(e, T);
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
        };
        print_T("T");
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
            #warning why are some of the initial streamsets in a partition not thread local?
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
