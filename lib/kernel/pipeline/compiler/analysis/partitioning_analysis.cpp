#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <toolchain/toolchain.h>
#include <util/slab_allocator.h>

// #define PRINT_GRAPH_BITSETS

namespace kernel {

using BitSet = dynamic_bitset<>;

using BindingVertex = RelationshipGraph::vertex_descriptor;

using Graph = adjacency_list<vecS, vecS, bidirectionalS, BitSet, BindingVertex>;

using DepGraph = adjacency_list<hash_setS, vecS, bidirectionalS>;

using PartitionMap = std::map<BitSet, unsigned>;

using Vertex = PartitionGraph::vertex_descriptor;

struct BindingInfo {
    Vertex Producer = 0;
    Vertex Consumer = 0;

    BindingInfo() = default;

    BindingInfo(const BindingInfo &) = default;

    BindingInfo(Vertex prod, Vertex consumer)
        : Producer(prod),  Consumer(consumer) {

    }
};

using BindingQueueItem = std::pair<unsigned, unsigned>;

class BindingQueueComparator {
public:
    bool operator()(const BindingQueueItem & a, const BindingQueueItem & b) {
        return (a.second < b.second);
    }
};

#if 0

class BindingQueue : public std::priority_queue<BindingQueueItem, std::vector<BindingQueueItem>, BindingQueueComparator> {
public:

    bool remove(const size_t value) {
        auto it = std::find(this->c.begin(), this->c.end(), value);

        if (it == this->c.end()) {
            return false;
        }
        if (it == this->c.begin()) {
            // deque the top element
            this->pop();
        }
        else {
            // remove element and re-heap
            this->c.erase(it);
            std::make_heap(this->c.begin(), this->c.end(), this->comp);
        }
        return true;
    }
};

#endif

struct PartitionBindingNode {
    std::vector<std::pair<unsigned, unsigned>> Transferable;
    std::vector<unsigned> PotentialRoots;
    std::vector<unsigned> AllKernels;

    PartitionBindingNode & operator=(const PartitionBindingNode & other) {
        AllKernels.assign(other.AllKernels.begin(), other.AllKernels.end());
        PotentialRoots.assign(other.PotentialRoots.begin(), other.PotentialRoots.end());
        Transferable.assign(other.Transferable.begin(), other.Transferable.end());
        return *this;
    }
};

using InternalPartitionGraph = adjacency_list<vecS, vecS, bidirectionalS, PartitionBindingNode, BindingInfo>;


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generatePartitionGraph
 ** ------------------------------------------------------------------------------------------------------------- */
PartitionGraph PipelineAnalysis::generatePartitionGraph() {

    const unsigned n = num_vertices(Relationships);

    std::vector<unsigned> sequence;
    sequence.reserve(n);

    std::vector<unsigned> allKernels;
    allKernels.reserve(n / 2);

    std::vector<unsigned> sequencePosition(n); // , -1U

    BEGIN_SCOPED_REGION

    std::vector<unsigned> ordering;
    ordering.reserve(n);
    if (LLVM_UNLIKELY(!lexical_ordering(Relationships, ordering))) {
        report_fatal_error("Failed to generate acyclic partition graph from kernel ordering");
    }

    // Convert the relationship graph into a simpler graph G that we can annotate.
    // For simplicity, force the pipeline input to be the first and the pipeline output
    // to be the last one.

    // For some reason, the Mac C++ compiler cannot link the constexpr PipelineInput value?
    // Hardcoding 0 here as a temporary workaround.
    sequencePosition[0] = 0;
    sequence.push_back(0);
    allKernels.push_back(0);
    for (unsigned u : ordering) {
        const RelationshipNode & node = Relationships[u];
        switch (node.Type) {
            case RelationshipNode::IsKernel:
                BEGIN_SCOPED_REGION
                #ifndef NDEBUG
                const auto & R = Relationships[u];
                #endif
                if (u == PipelineInput || u == PipelineOutput) {
                    assert (R.Kernel == mPipelineKernel);
                } else {
                    assert (R.Kernel != mPipelineKernel);
                    const auto k = sequence.size();
                    sequencePosition[u] = k;
                    allKernels.push_back(k);
                    sequence.push_back(u);
                }
                END_SCOPED_REGION
                break;
            case RelationshipNode::IsStreamSet:
                BEGIN_SCOPED_REGION
                const Relationship * const ss = Relationships[u].Relationship;
                // NOTE: We explicitly ignore RepeatingStreamSets here; trying to reason
                // about them will only complicate the analysis.
                if (LLVM_LIKELY(isa<StreamSet>(ss) || isa<TruncatedStreamSet>(ss))) {
                    sequencePosition[u] = sequence.size();
                    sequence.push_back(u);
                }
                END_SCOPED_REGION
                break;
            default: break;
        }
    }

    const auto k = sequence.size();
    sequencePosition[PipelineOutput] = k;
    allKernels.push_back(k);
    sequence.push_back(PipelineOutput);
    END_SCOPED_REGION
    const auto m = sequence.size();

    Graph G(m);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            addKernelRelationshipsInReferenceOrdering(u, Relationships,
                [&](const PortType type, const unsigned binding, const unsigned streamSet) {
                    const auto r = Relationships[streamSet].Relationship;
                    if (LLVM_LIKELY(isa<StreamSet>(r) || isa<TruncatedStreamSet>(r))) {
                        const auto j = sequencePosition[streamSet];
                        assert (j < m);
                        assert (sequence[j] == streamSet);
                        auto a = i, b = j;
                        if (type == PortType::Input) {
                            a = j; b = i;
                        }
                        assert (a < b);
                        assert (Relationships[binding].Type == RelationshipNode::IsBinding);
                        add_edge(a, b, binding, G);
                    }
                }
            );
        }
    }

    // Stage 1: identify synchronous components

    // wcan through the graph and determine where every non-Fixed relationship exists
    // so that we can construct our initial set of partitions. The goal here is to act
    // as a naive first pass to simplify the problem before using Z3.

    // NOTE: any decisions made during this pass *must* be provably correct for any
    // situation because the choices will *not* be verified.

    for (unsigned i = 0; i < m; ++i) {
        BitSet & V = G[i];
        V.resize(n);
        assert (V.none());
    }

    const auto numOfKernels = allKernels.size();
    unsigned nextRateId = 0;

    std::map<std::pair<BitSet, size_t>, size_t> L;
    std::vector<size_t> LookAheadIds;

    std::vector<unsigned> potentiallyMergable;
    potentiallyMergable.reserve(numOfKernels);

    std::vector<unsigned> forcedPartitionRoot;

    flat_map<Rational, unsigned> fixedRateDemarcationIds;

    for (unsigned i = 0; i < m; ++i) {

        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];

        BitSet & V = G[i];

        if (node.Type == RelationshipNode::IsKernel) {

            bool hasInputRateChange = false;

            const auto initialRateId = nextRateId;

            assert (node.Kernel);

            if (in_degree(i, G) == 0) {
                if (out_degree(i, G) != 0) {
                    hasInputRateChange = true;
                    // We isolate the source kernels so that the simulator will be able to
                    // infer the actual outputs of the source and correctly judge how to
                    // scale the segment length of the source kernels.
                    // demarcateOutputs = true;
                } else {
                    assert (node.Kernel == mPipelineKernel);
                }
            } else {

                assert (LookAheadIds.empty());

                for (const auto e : make_iterator_range(in_edges(i, G))) {

                    const auto bindingId = G[e];
                    const RelationshipNode & rn = Relationships[bindingId];
                    assert (rn.Type == RelationshipNode::IsBinding);
                    const Binding & bind = rn.Binding;
                    const ProcessingRate & rate = bind.getRate();

                    if (rate.isFixed()) {
                        // Check the attributes to see whether any impose a partition change
                        size_t fixedLookAhead = 0;
                        for (const Attribute & attr : bind.getAttributes()) {
                            switch (attr.getKind()) {
                                case AttrId::LookAhead:
                                    fixedLookAhead = std::max(fixedLookAhead, attr.amount());
                                    break;
                                case AttrId::Add:
                                case AttrId::BlockSize:
                                    hasInputRateChange = true;
                                default: break;
                            }
                        }
                        if (fixedLookAhead) {
                            const auto stride = node.Kernel->getStride();
                            const auto k = fixedLookAhead % stride;
                            if (k) {
                                fixedLookAhead += stride - k;
                            }
                            assert ((fixedLookAhead % stride) == 0);
                            const auto m = L.find(std::make_pair(V, fixedLookAhead));
                            size_t rateId = 0;
                            if (m == L.end()) {
                                rateId = nextRateId++;
                                L.emplace(std::make_pair(V, fixedLookAhead), rateId);
                            } else {
                                rateId = m->second;
                            }
                            LookAheadIds.push_back(rateId);
                        }
                    } else {
                        hasInputRateChange = true;
                        break;
                    }
                }
            }

            if (hasInputRateChange) {
                V.set(nextRateId++);
            }

            if (!LookAheadIds.empty()) {
                for (auto l : LookAheadIds) {
                    V.set(l);
                }
                LookAheadIds.clear();
                hasInputRateChange = true;
            }

            if (hasInputRateChange) {
                forcedPartitionRoot.push_back(i);
            }

            bool demarcateOutputs = false;

            if (node.Kernel != mPipelineKernel) {
                for (const Attribute & attr : node.Kernel->getAttributes()) {
                    switch (attr.getKind()) {
                        case AttrId::InternallySynchronized:
                            // although in some cases, an internally synchronized kernel does not need to be
                            // isolated to its own partition, we must guarantee that the kernel does not
                            // require multiple invocations to execute any full segment even in the case of
                            // a final partial block. To avoid this complication for now, we always isolate
                            // these kernels.
                            hasInputRateChange = true;
                        case AttrId::CanTerminateEarly:
                        case AttrId::MayFatallyTerminate:
                        case AttrId::MustExplicitlyTerminate:
                            fixedRateDemarcationIds.clear();
                            demarcateOutputs = true;
                        default: break;
                    }
                }
            }

            assert (V.any() || node.Kernel == mPipelineKernel);

            // Now iterate through the outputs
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto bindingId = G[e];
                const RelationshipNode & rn = Relationships[bindingId];
                assert (rn.Type == RelationshipNode::IsBinding);
                const Binding & b = rn.Binding;
                const ProcessingRate & rate = b.getRate();
                BitSet & O = G[target(e, G)];
                O |= V;

                if (rate.isFixed()) {

                    if (demarcateOutputs) {
                        auto f = fixedRateDemarcationIds.find(rate.getRate());
                        unsigned demarcationId = 0;
                        if (f == fixedRateDemarcationIds.end()) {
                            demarcationId = nextRateId++;
                            fixedRateDemarcationIds.emplace(rate.getRate(), demarcationId);
                        } else {
                            demarcationId = f->second;
                        }
                        O.set(demarcationId);
                    }

                    // Check the attributes to see whether any impose a partition change
                    for (const Attribute & attr : b.getAttributes()) {
                        switch (attr.getKind()) {
                            case AttrId::Add:
                            // TODO: if we computed the transitive add/subtracts prior to partitioning,
                            // we could more safely associate them with the dataflow
                            case AttrId::Delayed:
                            case AttrId::Deferred:
                            // A deferred output rate is closer to an bounded rate than a
                            // countable rate but a deferred input rate simply means the
                            // buffer must be dynamic.
                            case AttrId::BlockSize:
                                goto add_output_rate;
                            default: break;
                        }
                    }
                } else {
add_output_rate:    O.set(nextRateId++);
                }

            }

            if (initialRateId == nextRateId) {
                potentiallyMergable.push_back(i);
            }

        } else { // just propagate the bitsets

            for (const auto e : make_iterator_range(out_edges(i, G))) {
                BitSet & R = G[target(e, G)];
                R |= V;
            }

        }
    }

    assert (Relationships[sequence[0]].Kernel == mPipelineKernel);
    assert (Relationships[sequence[m - 1]].Kernel == mPipelineKernel);

    G[0].reset();
    G[m - 1].set(nextRateId);

    std::vector<unsigned> partitionIds(m);

    auto convertUniqueNodeBitSetsToUniquePartitionIds = [&]() -> unsigned {
        PartitionMap partitionSets;
        unsigned nextPartitionId = 1;
        for (auto i : allKernels) {
            assert (Relationships[sequence[i]].Type == RelationshipNode::IsKernel);
            BitSet & V = G[i];
            unsigned partitionId = 0;
            if (LLVM_LIKELY(V.any())) {
                auto f = partitionSets.find(V);
                if (f == partitionSets.end()) {
                    partitionId = nextPartitionId++;
                    partitionSets.emplace(V, partitionId);
                } else {
                    partitionId = f->second;
                }
                assert (partitionId > 0);
            } else {
                assert (Relationships[sequence[i]].Kernel == mPipelineKernel);
            }
            partitionIds[i] = partitionId;
        }
        return nextPartitionId;
    };

    auto synchronousPartitionCount = convertUniqueNodeBitSetsToUniquePartitionIds();

    assert (synchronousPartitionCount > 0);

    assert (std::is_sorted(forcedPartitionRoot.begin(), forcedPartitionRoot.end()));

    assert (std::is_sorted(potentiallyMergable.begin(), potentiallyMergable.end()));

    // Stage 6: split disconnected components within a partition into separate partitions

    BEGIN_SCOPED_REGION

    std::vector<unsigned> componentId(m);
    std::iota(componentId.begin(), componentId.end(), 0);

    std::function<unsigned(unsigned)> find = [&](unsigned x) {
        assert (x < m);
        if (componentId[x] != x) {
            componentId[x] = find(componentId[x]);
        }
        return componentId[x];
    };

    auto union_find = [&](unsigned x, unsigned y) {
        x = find(x);
        y = find(y);
        if (x != y) {
            if (y < x) {
                std::swap(y, x);
            }
            componentId[y] = x;
        }
    };

    for (auto i : allKernels) {
        assert (Relationships[sequence[i]].Type == RelationshipNode::IsKernel);
        const auto prodPartId = partitionIds[i];
        for (const auto e : make_iterator_range(out_edges(i, G))) {
            const auto j = target(e, G);
            assert (Relationships[sequence[j]].Type == RelationshipNode::IsStreamSet);
            Graph::out_edge_iterator begin, end;
            std::tie(begin, end) = out_edges(j, G);
            for (auto ei = begin; ei != end; ++ei) {
                const auto ki = target(*ei, G);
                assert (Relationships[sequence[ki]].Type == RelationshipNode::IsKernel);
                const auto consPartIdI = partitionIds[ki];
                assert (consPartIdI > 0);
                if (prodPartId == consPartIdI) {
                    union_find(i, ki);
                }
                for (auto ej = begin; ej != ei; ++ej) {
                    const auto kj = target(*ej, G);
                    assert (Relationships[sequence[kj]].Type == RelationshipNode::IsKernel);
                    const auto consPartIdJ = partitionIds[kj];
                    assert (consPartIdJ > 0);
                    if (consPartIdJ == consPartIdI) {
                        union_find(kj, ki);
                    }
                }
            }
        }
    }

    flat_set<unsigned> componentIds;
    componentIds.reserve(synchronousPartitionCount * 2);

    for (auto i : allKernels) {
        // find(x) updates and returns componentId[x]
        componentIds.insert(find(i));
    }

    for (auto i : allKernels) {
        const auto f = componentIds.find(componentId[i]);
        assert (f != componentIds.end());
        partitionIds[i] = std::distance(componentIds.begin(), f);
    }

    synchronousPartitionCount = componentIds.size();

    END_SCOPED_REGION

    // Stage 7: attempt to transfer kernels from earlier to later partitions when it is safe to do so

    InternalPartitionGraph P(synchronousPartitionCount);
    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto producer = allKernels[i];
        assert (Relationships[sequence[producer]].Type == RelationshipNode::IsKernel);
        const auto prodPartId = partitionIds[producer];
        unsigned localConsumers = 0;
        for (const auto e : make_iterator_range(out_edges(producer, G))) {
            const auto streamSet = target(e, G);
            assert (Relationships[sequence[streamSet]].Type == RelationshipNode::IsStreamSet);
            for (const auto f : make_iterator_range(out_edges(streamSet, G))) {
                const auto consumer = target(f, G);
                assert (Relationships[sequence[consumer]].Type == RelationshipNode::IsKernel);
                for (auto g : make_iterator_range(out_edges(prodPartId, P))) {
                    const auto & bi = P[g];
                    if (bi.Producer == producer && bi.Consumer == consumer) {
                        assert (target(g, P) == partitionIds[consumer]);
                        goto already_added;
                    }
                }
                BEGIN_SCOPED_REGION
                const auto consPartId = partitionIds[consumer];
                // we may only migrate kernels with no local users from a partition to the next
                // but as we migrate them, more opportunities may arise.
                if (prodPartId == consPartId) {
                    ++localConsumers;
                }
                add_edge(prodPartId, consPartId, BindingInfo{producer, consumer}, P);
                END_SCOPED_REGION
already_added:  continue;
            }
        }

        auto & X = P[prodPartId];
        X.AllKernels.emplace_back(producer);

        BEGIN_SCOPED_REGION
        const auto m = std::lower_bound(potentiallyMergable.begin(), potentiallyMergable.end(), producer);
        if (m != potentiallyMergable.end() && *m == producer) {
            X.Transferable.emplace_back(producer, localConsumers);
        }
        END_SCOPED_REGION

        BEGIN_SCOPED_REGION
        const auto m = std::lower_bound(forcedPartitionRoot.begin(), forcedPartitionRoot.end(), producer);
        if (m != forcedPartitionRoot.end() && *m == producer) {
            X.PotentialRoots.emplace_back(producer);
        }
        END_SCOPED_REGION
    }

    for (unsigned partId = 0; partId < synchronousPartitionCount; ++partId) {
        auto & X = P[partId];
        auto & roots = X.PotentialRoots;
        if (roots.empty()) {
            const auto & kernels = X.AllKernels;
            assert (kernels.size() > 0);
            assert (std::is_sorted(kernels.begin(), kernels.end()));

            if (kernels.size() == 1) {
                roots.emplace_back(kernels[0]);
            } else {
                auto indexOf = [&](const size_t kernel) {
                    const auto m = std::lower_bound(kernels.begin(), kernels.end(), kernel);
                    assert (m != kernels.end() && *m == kernel);
                    return std::distance(kernels.begin(), m);
                };

                dynamic_bitset<size_t> D(kernels.size());

                for (auto e : make_iterator_range(out_edges(partId, P))) {
                    assert (partitionIds[P[e].Producer] == partId);
                    if (target(e, P) == partId) {
                        assert (partitionIds[P[e].Consumer] == partId);
                        D.set(indexOf(P[e].Consumer));
                    } else {
                        assert (partitionIds[P[e].Consumer] != partId);
                    }
                }

                D.flip();

                assert (D.any());

                for (auto j = D.find_first(); j != dynamic_bitset<size_t>::npos; j = D.find_next(j)) {
                    const auto k = kernels[j];
                    #ifndef NDEBUG
                    assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                    for (const auto e : make_iterator_range(in_edges(k, G))) {
                        const auto streamSet = source(e, G);
                        assert (Relationships[sequence[streamSet]].Type == RelationshipNode::IsStreamSet);
                        for (const auto f : make_iterator_range(in_edges(streamSet, G))) {
                            const auto producer = source(f, G);
                            assert (Relationships[sequence[producer]].Type == RelationshipNode::IsKernel);
                            assert (partitionIds[producer] != partitionIds[k]);
                        }
                    }
                    #endif
                    roots.emplace_back(k);
                }
            }
        }
        assert (roots.size() > 0);
    }

    std::queue<unsigned> Q;
    std::vector<dynamic_bitset<size_t>> partitionPostDominators(synchronousPartitionCount);
    std::vector<Vertex> ordering(synchronousPartitionCount);

    for (unsigned v = synchronousPartitionCount; v--; ) {
        Q.push(v);
    }
    for (;;) {
start_of_loop:
        const auto v = Q.front();
        Q.pop();
        for (const auto e : make_iterator_range(out_edges(v, P))) {
            const auto u = target(e, P);
            if (LLVM_UNLIKELY(u != v && partitionPostDominators[u].empty())) {
                assert (!Q.empty());
                Q.push(v);
                goto start_of_loop;
            }
        }
        auto & D = partitionPostDominators[v];
        assert (D.empty());
        // ordering is written in reverse order that we process the queue in so
        // that the result is a topological ordering of the initial partitions.
        ordering[Q.size()] = v;
        D.resize(synchronousPartitionCount, false);
        D.set(v);
        for (const auto e : make_iterator_range(out_edges(v, P))) {
            const auto u = target(e, P);
            if (u != v) {
                const auto & X = partitionPostDominators[u];
                assert (X.size() == synchronousPartitionCount);
                assert (X.test(v) == 0);
                D |= X;
            }
        }
        if (Q.empty()) {
            break;
        }
    }

    // to maximize our ability to move a kernel to the deepest partition, we move according
    // to the ordering position.
    std::vector<unsigned> ordinal(m, -1U);
    for (unsigned i = 0; i < synchronousPartitionCount; ++i) {
        ordinal[ordering[i]] = i;
    }

    auto transferKernelsToAdjacentPartitions = [&](InternalPartitionGraph & partGraph) {

//        errs() << "---------\n";

        size_t transferCount = 0;

        for (auto currentPartId : ordering) {

            auto & CurrentPart = partGraph[currentPartId];
            auto & Transfer = CurrentPart.Transferable;
            assert (Transfer.size() <= CurrentPart.AllKernels.size());
            assert (CurrentPart.PotentialRoots.size() == 1 || CurrentPart.AllKernels.size() == 0);
            if (Transfer.empty()) {
                continue;
            }
            const auto root = CurrentPart.PotentialRoots[0];

            for (;;) {

                std::sort(Transfer.begin(), Transfer.end(), [](std::pair<unsigned, unsigned> & A, std::pair<unsigned, unsigned> & B) {
                    return A.second < B.second;
                });

                bool noChange = true;

                auto itr = Transfer.begin();
start_of_transfer_loop:
                while (itr != Transfer.end()) {

                    const auto potentiallyTransferedKernel = itr->first;
                    assert (potentiallyTransferedKernel < m);
                    assert (Relationships[sequence[potentiallyTransferedKernel]].Type == RelationshipNode::IsKernel);

                    #ifndef NDEBUG
                    size_t knownLocalConsumers = 0;
                    for (const auto e : make_iterator_range(out_edges(currentPartId, partGraph))) {
                        const BindingInfo & bi = partGraph[e];
                        if (bi.Producer == potentiallyTransferedKernel) {
                            assert (bi.Consumer != -1);
                            if (target(e, partGraph) == currentPartId) {
                                knownLocalConsumers++;
                            }
                        }
                    }
                    assert (knownLocalConsumers == itr->second);
                    #endif
                    if (itr->second > 0 || (itr->first == root && CurrentPart.AllKernels.size() != 1)) {
                        ++itr;
                        goto start_of_transfer_loop;
                    }

                    for (auto e : make_iterator_range(out_edges(currentPartId, partGraph))) {
                        auto & bi = partGraph[e];
                        if (bi.Producer == potentiallyTransferedKernel) {
                            const auto c = bi.Consumer;
                            assert (c != -1U);
                            const auto consumerPartId = target(e, partGraph);
                            assert (consumerPartId != currentPartId);
                            auto & roots = partGraph[consumerPartId].PotentialRoots;
                            assert (roots.size() == 1);
                            const auto selectedRoot = roots[0];
                            if (c == selectedRoot) {
                                const auto & H = partitionPostDominators[consumerPartId];
                                assert (H.test(consumerPartId) == 1);
                                assert (H.test(currentPartId) == 0);
                                const auto producer = bi.Producer; // we may end up "deleting" the bi edge in the loop below; keep the producer id
                                bool prunedAllPotentialDestinations = true;
                                for (auto f : make_iterator_range(out_edges(currentPartId, partGraph))) {
                                    auto & bj = partGraph[f];
                                    if (bj.Producer == producer) {
                                        assert (bj.Consumer != -1U);
                                        const auto consPartId = target(f, partGraph);
                                        if (H.test(consPartId)) {
                                            bj.Producer = -1U;
                                            bj.Consumer = -1U;
                                        } else {
                                            prunedAllPotentialDestinations = false;
                                        }
                                    }
                                }
                                assert (bi.Producer == -1U && bi.Consumer == -1U);

                                if (prunedAllPotentialDestinations) {
                                    itr = Transfer.erase(itr);
                                    goto start_of_transfer_loop;
                                }

                            }
                        }
                    }

                    unsigned transferPos = -1U;
                    for (const auto e : make_iterator_range(out_edges(currentPartId, partGraph))) {
                        const BindingInfo & bi = partGraph[e];
                        if (bi.Producer == potentiallyTransferedKernel) {
                            const auto u = target(e, partGraph);
                            assert (u != currentPartId);
                            assert (bi.Consumer != partGraph[u].PotentialRoots[0]);
                            const auto pos = ordinal[u];
                            assert (pos < synchronousPartitionCount);
                            assert (ordering[pos] == u);
                            if (pos < transferPos) {
                                transferPos = pos;
                            }
                        }
                    }

                    if (transferPos != -1U) {

                        const auto transferPartId = ordering[transferPos];

//                        errs() << "transfer " << potentiallyTransferedKernel << " from " << currentPartId
//                               << " to " << transferPartId
//                               << " [" << partGraph[transferPartId].PotentialRoots[0] << "]\n";

                        #ifndef NDEBUG
                        bool foundTransferInEntry = false;
                        for (const auto e : make_iterator_range(in_edges(transferPartId, partGraph))) {
                            const auto & bi = partGraph[e];
                            if (bi.Producer == potentiallyTransferedKernel) {
                                assert (source(e, partGraph) == currentPartId);
                                assert (bi.Consumer != partGraph[transferPartId].PotentialRoots[0]);
                                foundTransferInEntry = true;
                                break;
                            }
                        }
                        assert (foundTransferInEntry);
                        #endif

                        for (const auto e : make_iterator_range(in_edges(currentPartId, partGraph))) {
                            BindingInfo & bi = partGraph[e];
                            if (bi.Consumer == potentiallyTransferedKernel) {
                                const auto p = bi.Producer;
                                const auto prodPartId = source(e, partGraph);
                                if (prodPartId == currentPartId) {
                                    for (auto i = Transfer.begin(); i != Transfer.end(); ++i) {
                                        if (i->first == p) {
                                            assert (i->second > 0);
                                            i->second--;
                                            break;
                                        }
                                    }
                                }
                                add_edge(prodPartId, transferPartId, BindingInfo{p, potentiallyTransferedKernel}, partGraph);
                                bi.Producer = -1U;
                                bi.Consumer = -1U;
                            }
                        }

                        size_t localConsumers = 0;

                        for (const auto e : make_iterator_range(out_edges(currentPartId, partGraph))) {
                            BindingInfo & bi = partGraph[e];
                            if (bi.Producer == potentiallyTransferedKernel) {
                                const auto consPartId = target(e, partGraph);
                                assert (consPartId != currentPartId);
                                if (transferPartId == consPartId) {
                                    ++localConsumers;
                                }
                                add_edge(transferPartId, consPartId, BindingInfo{potentiallyTransferedKernel, bi.Consumer}, partGraph);
                                bi.Producer = -1U;
                                bi.Consumer = -1U;
                            }
                        }

                        assert (itr->first == potentiallyTransferedKernel);
                        itr = Transfer.erase(itr);
                        auto & TransferPart = partGraph[transferPartId];

                        TransferPart.Transferable.emplace_back(potentiallyTransferedKernel, localConsumers);

                        auto & T = TransferPart.AllKernels;
                        auto toInsert = std::lower_bound(T.begin(), T.end(), potentiallyTransferedKernel);
                        assert (toInsert == T.end() || *toInsert != potentiallyTransferedKernel);
                        T.insert(toInsert, potentiallyTransferedKernel);

                        auto & C = CurrentPart.AllKernels;
                        auto toErase = std::lower_bound(C.begin(), C.end(), potentiallyTransferedKernel);
                        assert (toErase != T.end() && *toErase == potentiallyTransferedKernel);
                        C.erase(toErase);

                        assert (potentiallyTransferedKernel != root || C.empty());
                        if (C.empty()) {
                            assert (Transfer.empty());
                            CurrentPart.PotentialRoots.clear();
                        }

                        transferCount++;

                        noChange = false;

                    } else { // can't transfer this kernel
                        ++itr;
                    }
                }
                if (noChange) {
                    break;
                }
            }
        }

//        errs() << " -- transferCount = " << transferCount << "\n";

        return transferCount;
    };

    bool requiresEnumeration = false;

    for (unsigned i = 0; i < synchronousPartitionCount; ++i) {
        auto & roots = P[i].PotentialRoots;
        assert (roots.size() > 0);
        if (roots.size() > 1) {
            // we have at least one selection to enumerate
            requiresEnumeration = true;
            break;
        }
    }

    if (requiresEnumeration) {

        size_t bestScore = 0;
        InternalPartitionGraph bestSelection;

        std::function<void(std::vector<unsigned> &)> enumerateAllOptions = [&](std::vector<unsigned> & vec) {
            for (auto i = vec.size(); i < synchronousPartitionCount; ++i) {
                auto & roots = P[i].PotentialRoots;
                assert (roots.size() > 0);
                if (roots.size() == 1) {
                    vec.push_back(roots[0]);
                } else {
                    const auto l = vec.size();
                    for (auto j = 0; j < roots.size(); ++j) {
                        vec.push_back(roots[j]);
                        enumerateAllOptions(vec);
                        vec.resize(l);
                    }
                    return;
                }
            }
            InternalPartitionGraph filteredClone(synchronousPartitionCount);
            for (unsigned i = 0; i < synchronousPartitionCount; ++i) {
                const auto & Pi = P[i];
                auto & Gi = filteredClone[i];
                Gi.AllKernels.assign(Pi.AllKernels.begin(), Pi.AllKernels.end());
                Gi.PotentialRoots.push_back(vec[i]);
                Gi.Transferable.assign(Pi.Transferable.begin(), Pi.Transferable.end());
            }
            for (auto e : make_iterator_range(edges(P))) {
                const auto & bi = P[e];
                assert (bi.Producer != -1U && bi.Consumer != -1U);
                add_edge(source(e, P), target(e, P), BindingInfo{bi.Producer, bi.Consumer}, filteredClone);
            }
            const auto score = transferKernelsToAdjacentPartitions(filteredClone) + 1UL;
            if (score > bestScore) {
                bestScore = score;
                bestSelection = filteredClone;
            }
        };

        std::vector<unsigned> selection;
        selection.reserve(synchronousPartitionCount);
        enumerateAllOptions(selection);
        assert (bestScore > 0);
        P = bestSelection;
    } else {
        transferKernelsToAdjacentPartitions(P);
    }



    // Stage 8: renumber the partition ids

    // To simplify processing later, renumber the partitions such that the partition id
    // of any predecessor of a kernel K is <= the partition id of K.

    size_t finalPartitionCount = 0;

    for (unsigned i = 0; i < synchronousPartitionCount; ++i) {
        const auto & CurrentPart = P[i];
        if (LLVM_LIKELY(CurrentPart.AllKernels.size() > 0)) {
            assert (CurrentPart.PotentialRoots.size() == 1);
            finalPartitionCount++;
        }
    }

    PartitionIds.clear();
    PartitionIds.reserve(allKernels.size());

    PartitionGraph partGraph(finalPartitionCount);

    forcedPartitionRoot.clear();
    forcedPartitionRoot.reserve(finalPartitionCount);

    for (unsigned i = 0, partId = 0; i < synchronousPartitionCount; ++i) {
        const auto currentPartId = ordering[i];
        const auto & CurrentPart = P[currentPartId];
        const auto & K = CurrentPart.AllKernels;
        if (LLVM_LIKELY(K.size() > 0)) {
            assert ("should start lexographically sorted" && std::is_sorted(K.begin(), K.end()));
            assert (CurrentPart.PotentialRoots.size() == 1);
            forcedPartitionRoot.push_back(CurrentPart.PotentialRoots[0]);
            auto & PjK = partGraph[partId].Kernels;
            PjK.reserve(K.size());

            for (auto k : K) {
                assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                assert (sequencePosition[sequence[k]] == k);
                PjK.push_back(k);
                partitionIds[k] = partId;

                PartitionIds.emplace(sequence[k], partId);
            }
            assert (PjK.size() == K.size());
            ++partId;
        }
    }

    assert (forcedPartitionRoot.size() == finalPartitionCount);

    synchronousPartitionCount = finalPartitionCount;

    std::sort(forcedPartitionRoot.begin(), forcedPartitionRoot.end());

    for (unsigned i = 0; i < synchronousPartitionCount; ++i) {

        const auto currentPartId = i;
        PartitionData & pd = partGraph[currentPartId];
        auto & K = pd.Kernels;
        const auto partitionSize = K.size();

        for (unsigned j = 0; j < partitionSize; ++j) {
            const auto v = K[j];
            assert (partitionIds[v] == currentPartId);
            for (const auto e : make_iterator_range(out_edges(v, G))) {
                const auto streamSet = target(e, G);
                assert (Relationships[sequence[streamSet]].Type == RelationshipNode::IsStreamSet);
                for (const auto f : make_iterator_range(out_edges(streamSet, G))) {
                    const auto u = target(f, G);
                    assert (Relationships[sequence[u]].Type == RelationshipNode::IsKernel);
                    const auto targetPartId = partitionIds[u];
                    assert (targetPartId < synchronousPartitionCount);
                    if (targetPartId != currentPartId) {
                        const auto ss = sequence[streamSet];
                        for (auto g : make_iterator_range(out_edges(currentPartId, partGraph))) {
                            if (target(g, partGraph) == targetPartId && partGraph[g] == ss) {
                                goto found_interpartition_edge;
                            }
                        }
                        assert (currentPartId < targetPartId);
                        add_edge(currentPartId, targetPartId, ss, partGraph);
found_interpartition_edge:
                        continue;
                    }
                }
            }
        }
    }

    std::vector<dynamic_bitset<size_t>> postdominators;

    for (unsigned currentPartId = 0; currentPartId < synchronousPartitionCount; ++currentPartId) {
        PartitionData & pd = partGraph[currentPartId];
        auto & K = pd.Kernels;
        const auto partitionSize = K.size();
        assert (partitionSize > 0);

        if (K.size() > 1) {

            #ifndef NDEBUG
            std::fill_n(ordinal.begin(), m, -1U);
            size_t rootCount = 0;
            #endif
            for (unsigned i = 0; i < partitionSize; ++i) {
                const auto k = K[i];
                assert (k < m);
                assert (ordinal[k] == -1U);
                #ifndef NDEBUG
                const auto x = std::lower_bound(forcedPartitionRoot.begin(), forcedPartitionRoot.end(), k);
                if (x != forcedPartitionRoot.end() && *x == k) {
                    ++rootCount;
                }
                assert (partitionIds[k] == currentPartId);
                #endif
                ordinal[k] = i;
            }

            assert ("partition must have exactly one root" && (rootCount == 1));

            assert ("should start lexographically sorted" && std::is_sorted(K.begin(), K.end()));

            postdominators.resize(partitionSize);

            assert (Q.empty());

            for (unsigned j = partitionSize; j--; ) {
                postdominators[j].clear();
                assert (postdominators[j].empty());
                Q.push(K[j]);
            }

            for (;;) {
start_of_partition_sort_loop:
                const auto v = Q.front();
                Q.pop();
                assert (Relationships[sequence[v]].Type == RelationshipNode::IsKernel);
                #ifndef NDEBUG
                assert (partitionIds[v] == currentPartId);
                #endif
                for (const auto e : make_iterator_range(out_edges(v, G))) {
                    const auto streamSet = target(e, G);
                    assert (Relationships[sequence[streamSet]].Type == RelationshipNode::IsStreamSet);
                    for (const auto f : make_iterator_range(out_edges(streamSet, G))) {
                        const auto u = target(f, G);
                        assert (Relationships[sequence[u]].Type == RelationshipNode::IsKernel);
                        const auto targetPartId = partitionIds[u];
                        assert (targetPartId < synchronousPartitionCount);
                        if (targetPartId == currentPartId) {
                            assert (u < m && ordinal[u] < partitionSize);
                            const auto & X = postdominators[ordinal[u]];
                            if (LLVM_UNLIKELY(X.empty())) {
                                assert (!Q.empty());
                                Q.push(v);
                                goto start_of_partition_sort_loop;
                            }
                        }
                    }
                }

                assert (v < m && ordinal[v] < partitionSize);
                auto & D = postdominators[ordinal[v]];
                assert (D.empty());
                D.resize(partitionSize, false);
                assert (D.none());

                const auto x = std::lower_bound(forcedPartitionRoot.begin(), forcedPartitionRoot.end(), v);
                if (x != forcedPartitionRoot.end() && *x == v) {
                    D.set();
                } else {
                    D.set(ordinal[v]);
                    for (const auto e : make_iterator_range(out_edges(v, G))) {
                        const auto streamSet = target(e, G);
                        assert (Relationships[sequence[streamSet]].Type == RelationshipNode::IsStreamSet);
                        for (const auto f : make_iterator_range(out_edges(streamSet, G))) {
                            const auto u = target(f, G);
                            assert (Relationships[sequence[u]].Type == RelationshipNode::IsKernel);
                            const auto targetPartId = partitionIds[u];
                            assert (targetPartId < synchronousPartitionCount);
                            if (targetPartId == currentPartId) {
                                assert (u < m && ordinal[u] < partitionSize);
                                const auto & X = postdominators[ordinal[u]];
                                assert (X.size() == partitionSize);
                                assert (X.test(ordinal[v]) == 0);
                                D |= X;
                            }
                        }
                    }
                    assert ("only the partition root can have all depedencies set" && !D.all());
                }

                if (Q.empty()) {
                    break;
                }
            }

            for (unsigned i = 0; i < partitionSize; ++i) {
                const auto k = K[i];
                assert (k < m);
                ordinal[k] = i;
            }

            std::stable_sort(K.begin(), K.end(), [&](const unsigned a, const unsigned b) {
                const auto & A = postdominators[ordinal[a]];
                const auto & B = postdominators[ordinal[b]];
                const auto C = A & B;
                return B.is_subset_of(C);
            });

        }

        for (unsigned i = 0; i < partitionSize; ++i) {
            const auto k = K[i];
            #ifndef NDEBUG
            if (i > 0) {
                const auto m = std::lower_bound(forcedPartitionRoot.begin(), forcedPartitionRoot.end(), k);
                assert ("partition root occurred min-partition? " && m == forcedPartitionRoot.end() || *m != k);
            }
            #endif
            K[i] = sequence[k];
            assert (Relationships[K[i]].Type == RelationshipNode::IsKernel);
        }
    }

    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    flat_set<unsigned> included;
    included.reserve(numOfKernels);
    for (const auto u : partGraph[0].Kernels) {
        assert ("kernel is in multiple partitions?" && included.insert(u).second);
        const auto & R = Relationships[u];
        assert (R.Type == RelationshipNode::IsKernel);
        assert (R.Kernel == mPipelineKernel);
    }
    auto numOfPartitionedKernels = partGraph[0].Kernels.size();
    for (unsigned i = 1; i < synchronousPartitionCount; ++i) {
        numOfPartitionedKernels += partGraph[i].Kernels.size();
        for (const auto u : partGraph[i].Kernels) {
            assert ("kernel is in multiple partitions?" && included.insert(u).second);
            const auto & R = Relationships[u];
            assert (R.Type == RelationshipNode::IsKernel);
        }
    }
    assert (numOfPartitionedKernels == numOfKernels);
    END_SCOPED_REGION
    #endif

    assert (synchronousPartitionCount > 0);

    for (unsigned i = 1; i < (synchronousPartitionCount - 1); ++i) {
        if (in_degree(i, P) == 0) {
            add_edge(0, i, 0, partGraph);
        }
        if (out_degree(i, P) == 0) {
            add_edge(i, synchronousPartitionCount - 1, 0, partGraph);
        }
    }

    PartitionCount = synchronousPartitionCount;


    #ifdef PRINT_GRAPH_BITSETS
    BEGIN_SCOPED_REGION
    auto & out = errs();

    out << "digraph \"H\" {\n";
    for (auto v : make_iterator_range(vertices(partGraph))) {
        const PartitionData & D = partGraph[v];
        out << "v" << v << " [label=\"";
        bool addNewLine = false;
        for (const auto k : D.Kernels) {
            const RelationshipNode & node = Relationships[k];
            assert (node.Type == RelationshipNode::IsKernel);
            out << k << ". " << node.Kernel->getName();
            if (addNewLine) {
                out << "\\n";
            }
            addNewLine = true;
        }
        out << "\",shape=rect];\n";
    }
    for (auto e : make_iterator_range(edges(P))) {
        const auto s = source(e, P);
        const auto t = target(e, P);
        out << "v" << s << " -> v" << t << " [label=\"" << partGraph[e] << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    END_SCOPED_REGION
    #endif

    return partGraph;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determinePartitionJumpIndices
 *
 * If a partition determines it has insufficient data to execute, identify which partition is the next one to test.
 * I.e., the one with input from some disjoint path. If none exists, we'll begin jump to "PartitionCount", which
 * marks the end of the processing loop.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determinePartitionJumpIndices() {
     PartitionJumpTargetId.resize(PartitionCount);
#ifdef DISABLE_PARTITION_JUMPING
    for (unsigned i = FirstComputePartitionId; i <= LastComputePartitionId; ++i) {
        mPartitionJumpIndex[i] = i + 1;
    }
    mPartitionJumpIndex[(PartitionCount - 1)] = (PartitionCount - 1);
#else

    using BitSet = dynamic_bitset<>;

    using PartitionGraph = adjacency_list<hash_setS, vecS, bidirectionalS, no_property, no_property, no_property>;

    std::vector<BitSet> rateDomSet(PartitionCount);

    unsigned nextRateId = 0;

    auto expandCapacity = [&](BitSet & bs) {
        const auto size = (nextRateId + 127U) & (~63U);
        if (bs.size() != size) {
            bs.resize(size, false);
        }
    };

    auto addRateId = [&](const unsigned id, const unsigned rateId) {
        auto & bs = rateDomSet[id];
        expandCapacity(bs);
        bs.set(rateId);
    };

    PartitionGraph J(PartitionCount);

    for (auto streamSet = FirstStreamSet; streamSet < LastStreamSet; ++streamSet) {

        if (LLVM_UNLIKELY(in_degree(streamSet, mBufferGraph) == 0)) {
            assert (mStreamGraph[streamSet].Type == RelationshipNode::IsStreamSet);
            assert (isa<RepeatingStreamSet>(mStreamGraph[streamSet].Relationship));
        } else {
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto & outputPort = mBufferGraph[output];
            const Binding & binding = outputPort.Binding;

            const auto hasVarOutput = isNonSynchronousRate(binding);

            const auto producer = source(output, mBufferGraph);
            const auto pid = KernelPartitionId[producer];

            auto rateId = nextRateId;
            nextRateId += hasVarOutput ? 1 : 0;
            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(input, mBufferGraph);
                const auto cid = KernelPartitionId[consumer];
                if (cid != pid) {
                    add_edge(pid, cid, J);
                    if (hasVarOutput) {
                        addRateId(cid, rateId);
                    }
                }
            }
        }


    }

    // Now compute the transitive reduction of the partition relationships
    BEGIN_SCOPED_REGION
    const reverse_traversal ordering(PartitionCount);
    assert (is_valid_topological_sorting(ordering, J));
    transitive_closure_dag(ordering, J);
    transitive_reduction_dag(ordering, J);
    END_SCOPED_REGION

    if (in_degree(PartitionCount - 1, J) == 0) {
        for (auto partitionId = 1U; partitionId < PartitionCount; ++partitionId) {
            if (LLVM_UNLIKELY(out_degree(partitionId, J) == 0)) {
                add_edge(partitionId, PartitionCount - 1, J);
            }
        }
    }

    for (unsigned i = 0; i < PartitionCount; ++i) {
        expandCapacity(rateDomSet[i]);
    }

    BitSet intersection;
    expandCapacity(intersection);

    for (auto partitionId = 1U; partitionId < PartitionCount; ++partitionId) { // topological ordering
        auto & ds = rateDomSet[partitionId];

        if (out_degree(partitionId, J) == 0) {
            ds.reset();
        } else {
            if (in_degree(partitionId, J) > 0) {
                intersection.set();
                for (const auto e : make_iterator_range(in_edges(partitionId, J))) {
                    const unsigned producerId = source(e, J);
                    assert (producerId <= partitionId);
                    intersection &= rateDomSet[producerId];
                }
                ds |= intersection;
            }
        }

        if (ds.none()) {
            const auto rateId = nextRateId++;
            for (unsigned i = 0; i < PartitionCount; ++i) {
                expandCapacity(rateDomSet[i]);
            }
            expandCapacity(intersection);
            ds.set(rateId);
        }
    }


    for (size_t i = 1U; i < PartitionCount - 1; ++i) {
        const BitSet & prior =  rateDomSet[i - 1];
        const BitSet & current =  rateDomSet[i];
        auto j = i + 1U;
        if (prior != current && in_degree(i, J) > 0) {
            assert (current.any());
            for (; j < (PartitionCount - 1); ++j) {
                const BitSet & next =  rateDomSet[j];
                if (!current.is_subset_of(next)) {
                    break;
                }
            }
        }
        assert (j > i);

        PartitionJumpTargetId[i] = j;
    }

    if (LLVM_UNLIKELY(!IsNestedPipeline && codegen::EnableJumpGuidedSynchronizationVariables)) {
        const auto lastComputeKernel = FirstKernelInPartition[LastComputePartitionId + 1U] - 1U;
        for (auto partId = FirstComputePartitionId; partId <= LastComputePartitionId; ++partId) {
            if (LLVM_UNLIKELY(PartitionJumpTargetId[partId] == (PartitionCount - 1))) {
                const auto kernelId = FirstKernelInPartition[partId];
                if (kernelId < lastComputeKernel) {
                    mBufferGraph[kernelId].Type |= StartsNestedSynchronizationRegion;
                }
            }
        }
    }



    PartitionJumpTargetId[0] = 0;
    if (LLVM_LIKELY(FirstComputePartitionId > 0)) {
        for (size_t i = 2; i < FirstComputePartitionId; ++i) {
            PartitionJumpTargetId[i - 1] = i;
        }
        PartitionJumpTargetId[(FirstComputePartitionId - 1)] = (LastComputePartitionId + 1);
        for (size_t i = FirstComputePartitionId; i <= LastComputePartitionId; ++i) {
            if (PartitionJumpTargetId[i] > LastComputePartitionId) {
                PartitionJumpTargetId[i] = (PartitionCount - 1);
            }
        }
    }
    assert (PartitionCount > 1);
    for (auto i = (LastComputePartitionId + 1); i < (PartitionCount - 1); ++i) {
        PartitionJumpTargetId[i] = (i + 1);
    }
    PartitionJumpTargetId[(PartitionCount - 1)] = (PartitionCount - 1);

#endif
}

} // end of namespace kernel
