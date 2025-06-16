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

using ConflictGraph = adjacency_list<hash_setS, vecS, undirectedS>;

using IntervalSet = interval_set<unsigned>;

using Interval = IntervalSet::interval_type;

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

    ThreadLocalPlacementGraph T(PartitionCount + n);

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

        #ifndef NDEBUG
        for (size_t i = 0; i < numOfThreadLocalStreamSets; ++i) {
            assert (remaining[i] == 0);
        }
        #endif

        BufferLayoutOptimizer BA(numOfThreadLocalStreamSets,
                                 packedPartitionCount,
                                 I, unitWeight, streamSetPartitionId,
                                 rng);

        BA.runGA<false>();

        auto O = BA.getResult();



        const auto intervals = BA.translate(O, rng);
        assert (intervals.size() == numOfThreadLocalStreamSets);

        const auto pageSize = getPageSize();

        auto add_edge_to_T = [&](const size_t u, const size_t v, const unsigned num, const unsigned denom) {
            assert (!edge(u, v, T).second);
            Rational percentOfPagePerStride{num, denom * pageSize};
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

#if 0

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
        }

        Z3_model_dec_ref(ctx, model);
        Z3_optimize_dec_ref(ctx, solver);
        Z3_del_context(ctx);
        Z3_reset_memory();
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
