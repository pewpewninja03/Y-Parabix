#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <toolchain/toolchain.h>
#include <util/slab_allocator.h>

// #define PRINT_GRAPH_BITSETS

namespace kernel {

using BitSet = dynamic_bitset<>;

using BindingVertex = RelationshipGraph::vertex_descriptor;

using Graph = adjacency_list<vecS, vecS, bidirectionalS, BitSet, BindingVertex>;

using PartitionMap = std::map<BitSet, unsigned>;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initialPartitioningPass
 ** ------------------------------------------------------------------------------------------------------------- */
PartitionGraph PipelineAnalysis::initialPartitioningPass() {

    const unsigned n = num_vertices(Relationships);

    std::vector<unsigned> sequence;
    sequence.reserve(n);

    std::vector<unsigned> mapping(n, -1U);

    unsigned numOfKernels = 2;

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
    mapping[0] = 0;
    sequence.push_back(0);
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
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                    ++numOfKernels;
                }
                END_SCOPED_REGION
                break;
            case RelationshipNode::IsStreamSet:
                BEGIN_SCOPED_REGION
                const Relationship * const ss = Relationships[u].Relationship;
                // NOTE: We explicitly ignore RepeatingStreamSets here; trying to reason
                // about them will only complicate the analysis.
                if (LLVM_LIKELY(isa<StreamSet>(ss) || isa<TruncatedStreamSet>(ss))) {
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                }
                END_SCOPED_REGION
                break;
            default: break;
        }
    }

    mapping[PipelineOutput] = sequence.size();
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
                        const auto j = mapping[streamSet];
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

    unsigned nextRateId = 0;

    for (unsigned i = 0; i < m; ++i) {

        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];

        BitSet & V = G[i];

        if (node.Type == RelationshipNode::IsKernel) {

            bool hasInputRateChange = false;

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
                for (const auto e : make_iterator_range(in_edges(i, G))) {

                    const auto bindingId = G[e];
                    const RelationshipNode & rn = Relationships[bindingId];
                    assert (rn.Type == RelationshipNode::IsBinding);
                    const Binding & b = rn.Binding;
                    const ProcessingRate & rate = b.getRate();
                    if (rate.isFixed()) {
                        // Check the attributes to see whether any impose a partition change
                        for (const Attribute & attr : b.getAttributes()) {
                            switch (attr.getKind()) {
                                case AttrId::Add:
                                case AttrId::LookAhead:
                                case AttrId::BlockSize:
                                    hasInputRateChange = true;
                                default: break;
                            }
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
                    // Check the attributes to see whether any impose a partition change
                    for (const Attribute & attr : b.getAttributes()) {
                        switch (attr.getKind()) {
                            case AttrId::Add:
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

    auto convertUniqueNodeBitSetsToUniquePartitionIds = [&]() {
        PartitionMap partitionSets;
        unsigned nextPartitionId = 1;
        for (unsigned i = 0; i < m; ++i) {
            const auto u = sequence[i];
            const RelationshipNode & node = Relationships[u];
            if (node.Type == RelationshipNode::IsKernel) {
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
                    assert (node.Kernel == mPipelineKernel);
                }
                partitionIds[i] = partitionId;
            }
        }
        return nextPartitionId;
    };

    const auto synchronousPartitionCount = convertUniqueNodeBitSetsToUniquePartitionIds();

    assert (synchronousPartitionCount > 0);

    // Stage 6: split (weakly) disconnected components within a partition into separate partitions

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
        assert (x < y);
        x = find(x);
        y = find(y);
        if (x != y) {
            componentId[y] = x;
        }
    };

    for (unsigned i = 1; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsStreamSet) {
            const auto j = parent(i, G);
            assert (Relationships[sequence[j]].Type == RelationshipNode::IsKernel);
            const auto prodPartId = partitionIds[j];
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto k = target(e, G);
                assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                const auto consPartId = partitionIds[k];
                assert (consPartId > 0);
                if (prodPartId == consPartId) {
                    union_find(j, k);
                }
            }
        }
    }

    flat_set<unsigned> componentIds;
    componentIds.reserve(synchronousPartitionCount * 2);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            // find(x) updates and returns componentId[x]
            componentIds.insert(find(i));
        }
    }

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            const auto f = componentIds.find(componentId[i]);
            assert (f != componentIds.end());
            componentId[i] = std::distance(componentIds.begin(), f);
        }
    }

    const auto partitionCount = componentIds.size();
    assert (partitionCount >= synchronousPartitionCount);

    using RenumberingGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

    // Stage 7: renumber the partition ids

    // To simplify processing later, renumber the partitions such that the partition id
    // of any predecessor of a kernel K is <= the partition id of K.

    RenumberingGraph T(partitionCount);

    for (unsigned i = 1; i < partitionCount; ++i) {
        add_edge(0, i, 0, T);
    }

    for (unsigned i = 1; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsStreamSet) {
            const auto j = parent(i, G);
            assert (Relationships[sequence[j]].Type == RelationshipNode::IsKernel);
            const auto prodPartId = componentId[j];
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto k = target(e, G);
                assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                const auto consPartId = componentId[k];
                if (prodPartId != consPartId) {
                    assert (consPartId > 0);
                    add_edge(prodPartId, consPartId, u, T);
                }
            }
        }
    }

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (out_degree(i, T) == 0) {
            add_edge(i, partitionCount - 1, 0, T);
        }
    }

    std::vector<unsigned> renumberingSeq;
    renumberingSeq.reserve(partitionCount);

    if (LLVM_UNLIKELY(!lexical_ordering(T, renumberingSeq))) {
        report_fatal_error("Internal error: failed to generate initial acyclic partition graph");
    }

    assert (renumberingSeq[0] == 0);

    std::vector<unsigned> renumbered(partitionCount);

    for (unsigned i = 0; i < partitionCount; ++i) {
        const auto j = renumberingSeq[i];
        assert (j < partitionCount);
        renumbered[j] = i;
    }

    assert (renumbered[0] == 0);

    PartitionGraph P(partitionCount);

    PartitionIds.clear();

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            assert (componentId[i] < partitionCount);
            const auto j = renumbered[componentId[i]];
            assert (j < partitionCount);
            assert ((j > 0 && (j + 1) < partitionCount) ^ (node.Kernel == mPipelineKernel));
            PartitionData & pd = P[j];
            pd.Kernels.push_back(u);
            PartitionIds.emplace(u, j);
        }
    }

    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    flat_set<unsigned> included;
    included.reserve(numOfKernels);
    for (const auto u : P[0].Kernels) {
        assert ("kernel is in multiple partitions?" && included.insert(u).second);
        const auto & R = Relationships[u];
        assert (R.Type == RelationshipNode::IsKernel);
        assert (R.Kernel == mPipelineKernel);
    }
    auto numOfPartitionedKernels = P[0].Kernels.size();
    for (unsigned i = 1; i < partitionCount; ++i) {
        numOfPartitionedKernels += P[i].Kernels.size();
        for (const auto u : P[i].Kernels) {
            assert ("kernel is in multiple partitions?" && included.insert(u).second);
            const auto & R = Relationships[u];
            assert (R.Type == RelationshipNode::IsKernel);
        }
    }
    assert (numOfPartitionedKernels == numOfKernels);
    END_SCOPED_REGION
    #endif

    flat_set<std::pair<unsigned, unsigned>> duplicateFilter;

    for (unsigned i = 0; i < partitionCount; ++i) {
        assert (P[i].Kernels.size() > 0);
        const auto j = renumbered[i];
        assert (duplicateFilter.empty());
        for (const auto e : make_iterator_range(out_edges(i, T))) {
            const auto k = renumbered[target(e, T)];
            assert (k > j);
            const auto streamSet = T[e];
            if (LLVM_UNLIKELY(streamSet == 0)) continue;
            assert (streamSet < num_vertices(Relationships));
            assert (Relationships[streamSet].Type == RelationshipNode::IsStreamSet);
            if (duplicateFilter.emplace(k, streamSet).second) {
                add_edge(j, k, streamSet, P);
            }
        }
        duplicateFilter.clear();
    }

    assert (partitionCount > 0);

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (in_degree(i, P) == 0) {
            add_edge(0, i, 0, P);
        }
        if (out_degree(i, P) == 0) {
            add_edge(i, partitionCount - 1, 0, P);
        }
    }

    PartitionCount = partitionCount;

    #ifdef PRINT_GRAPH_BITSETS
    BEGIN_SCOPED_REGION
    auto & out = errs();

    out << "digraph \"H\" {\n";
    for (auto v : make_iterator_range(vertices(P))) {
        const PartitionData & D = P[v];
        out << "v" << v << " [label=\"";
        for (const auto k : D.Kernels) {
            const RelationshipNode & node = Relationships[k];
            assert (node.Type == RelationshipNode::IsKernel);
            out << k << ". " << node.Kernel->getName() << "\\n";
        }
        out << " -- linkId=" << D.LinkedGroupId << "\",shape=rect];\n";
    }
    for (auto e : make_iterator_range(edges(P))) {
        const auto s = source(e, P);
        const auto t = target(e, P);
        out << "v" << s << " -> v" << t << " [label=\"" << P[e] << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    END_SCOPED_REGION
    #endif

    return P;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief postDataflowAnalysisPartitioningPass
 ** ------------------------------------------------------------------------------------------------------------- */
PartitionGraph PipelineAnalysis::postDataflowAnalysisPartitioningPass(PartitionGraph & initial) {

    const unsigned n = num_vertices(Relationships);

    std::vector<unsigned> sequence;
    sequence.reserve(n);

    std::vector<unsigned> mapping(n, -1U);

    unsigned numOfKernels = 2;

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
    mapping[0] = 0;
    sequence.push_back(0);

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
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                    ++numOfKernels;
                }
                END_SCOPED_REGION
                break;
            case RelationshipNode::IsStreamSet:
                BEGIN_SCOPED_REGION
                const Relationship * const ss = Relationships[u].Relationship;
                if (LLVM_LIKELY(isa<StreamSet>(ss) || isa<TruncatedStreamSet>(ss))) {
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                }
                END_SCOPED_REGION
                break;
            default: break;
        }
    }

    mapping[PipelineOutput] = sequence.size();
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
                        const auto j = mapping[streamSet];
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
        V.resize((n * 2) + 1);
        assert (V.none());
    }

    const auto l = num_vertices(initial);

    for (unsigned i = 0; i < l; ++i) {
        const auto & P = initial[i];
        for (const auto u : P.Kernels) {
            assert (u < mapping.size());
            assert (Relationships[u].Type == RelationshipNode::IsKernel);
            const auto j = mapping[u];
            assert (j < m);
            assert (P.LinkedGroupId <= l);
            G[j].set(P.LinkedGroupId);
        }
    }

    auto nextRateId = l;

    const auto useIOProcessThread = codegen::UseProcessThreadForIO && !IsNestedPipeline;

    flat_map<unsigned, unsigned> zeroExtendMap;

    for (unsigned i = 0; i < m; ++i) {

        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];

        BitSet & V = G[i];

        for (const auto e : make_iterator_range(in_edges(i, G))) {
            const auto u = source(e, G);
            V |= G[u];
        }

        if (node.Type == RelationshipNode::IsKernel) {

            const Kernel * const kernelObj = node.Kernel;

            assert (V.any() || kernelObj == mPipelineKernel);

            // Check whether this (internal) kernel could terminate early
            bool useNewRateId = (in_degree(i, G) == 0) || (useIOProcessThread && out_degree(i, G) == 0);
            bool demarcateOutputs = (kernelObj == mPipelineKernel) || useNewRateId;

            if (kernelObj != mPipelineKernel) {
                for (const Attribute & attr : kernelObj->getAttributes()) {
                    switch (attr.getKind()) {
                        case AttrId::InternallySynchronized:
                            // although in some cases, an internally synchronized kernel does not need to be
                            // isolated to its own partition, we must guarantee that the kernel does not
                            // require multiple invocations to execute any full segment even in the case of
                            // a final partial block. To avoid this complication for now, we always isolate
                            // these kernels.
                            useNewRateId = true;
                        case AttrId::CanTerminateEarly:
                        case AttrId::MayFatallyTerminate:
                        case AttrId::MustExplicitlyTerminate:
                            demarcateOutputs = true;
                        default: break;
                    }
                }

            }

            if (useNewRateId) {
                assert (!V.test(nextRateId));
                V.set(nextRateId++);
            }

            unsigned demarcationId = 0;
            if (LLVM_UNLIKELY(demarcateOutputs)) {
                assert (!V.test(nextRateId));
                demarcationId = nextRateId++;
            }

            // Now iterate through the outputs
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                BitSet & O = G[target(e, G)];
                O |= V;
                if (LLVM_UNLIKELY(demarcateOutputs)) {
                    assert (!O.test(demarcationId));
                    O.set(demarcationId);
                }
            }

        } else { // just propagate the bitsets

            assert (node.Type == RelationshipNode::IsStreamSet || node.Type == RelationshipNode::IsScalar);

            // If a streamset has consumers that zero-extend it but also some
            // that do not, we may end up kernels who view the streamset as
            // having different lengths. However, if the producer and consumers
            // belong to the same partition, we do not need to worry about
            // zero-extension.

            bool isZeroExtendedInput = false;

            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto w = target(e, G);
                assert (Relationships[sequence[w]].Type == RelationshipNode::IsKernel);
                BitSet & R = G[w];
                R |= V;
                const auto r = G[e];
                const RelationshipNode & rn = Relationships[r];
                assert (rn.Type == RelationshipNode::IsBinding);
                const Binding & bn = rn.Binding;
                isZeroExtendedInput |= bn.hasAttribute(AttrId::ZeroExtended);
            }

            if (isZeroExtendedInput) {

                assert (zeroExtendMap.empty());

                auto findPartionId = [&](const size_t vertex) {
                    const auto v = sequence[vertex];
                    assert (Relationships[v].Type == RelationshipNode::IsKernel);
                    const auto f = PartitionIds.find(v);
                    assert (f != PartitionIds.end());
                    return f->second;
                };

                const auto prodPartId = findPartionId(parent(i, G));

                for (const auto e : make_iterator_range(out_edges(i, G))) {
                    const auto w = target(e, G);
                    assert (Relationships[sequence[w]].Type == RelationshipNode::IsKernel);
                    BitSet & R = G[w];

                    const auto conPartId = findPartionId(w);
                    if (prodPartId != conPartId) {
                        const auto g = zeroExtendMap.find(conPartId);
                        unsigned rateId = 0U;
                        if (g == zeroExtendMap.end()) {
                            rateId = nextRateId++;
                            zeroExtendMap.emplace(conPartId, rateId);
                        } else {
                            rateId = g->second;
                        }
                        R.set(rateId);
                    }
                }
                zeroExtendMap.clear();

            }


        }
    }

    assert (Relationships[sequence[0]].Kernel == mPipelineKernel);
    assert (Relationships[sequence[m - 1]].Kernel == mPipelineKernel);

    G[0].reset();
    G[m - 1].set(nextRateId);

    std::vector<unsigned> partitionIds(m);

    auto convertUniqueNodeBitSetsToUniquePartitionIds = [&]() {
        PartitionMap partitionSets;
        unsigned nextPartitionId = 1;
        for (unsigned i = 0; i < m; ++i) {
            const auto u = sequence[i];
            const RelationshipNode & node = Relationships[u];
            if (node.Type == RelationshipNode::IsKernel) {
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
                    assert (node.Kernel == mPipelineKernel);
                }
                partitionIds[i] = partitionId;
            }
        }
        return nextPartitionId;
    };

    const auto synchronousPartitionCount = convertUniqueNodeBitSetsToUniquePartitionIds();

    assert (synchronousPartitionCount > 0);

    // Stage 6: split (weakly) disconnected components within a partition into separate partitions

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
        assert (x < y);
        x = find(x);
        y = find(y);
        if (x != y) {
            componentId[y] = x;
        }
    };

    for (unsigned i = 1; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsStreamSet) {
            const auto j = parent(i, G);
            assert (Relationships[sequence[j]].Type == RelationshipNode::IsKernel);
            const auto prodPartId = partitionIds[j];
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto k = target(e, G);
                assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                const auto consPartId = partitionIds[k];
                assert (consPartId > 0);
                if (prodPartId == consPartId) {
                    union_find(j, k);
                }
            }
        }
    }

    flat_set<unsigned> componentIds;
    componentIds.reserve(synchronousPartitionCount * 2);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            // find(x) updates and returns componentId[x]
            componentIds.insert(find(i));
        }
    }

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            const auto f = componentIds.find(componentId[i]);
            assert (f != componentIds.end());
            componentId[i] = std::distance(componentIds.begin(), f);
        }
    }

    const auto partitionCount = componentIds.size();
    assert (partitionCount >= synchronousPartitionCount);

    using RenumberingGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

    // Stage 7: renumber the partition ids

    // To simplify processing later, renumber the partitions such that the partition id
    // of any predecessor of a kernel K is <= the partition id of K.

    RenumberingGraph T(partitionCount);

    for (unsigned i = 1; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsStreamSet) {
            const auto j = parent(i, G);
            assert (Relationships[sequence[j]].Type == RelationshipNode::IsKernel);
            const auto prodPartId = componentId[j];
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto k = target(e, G);
                assert (Relationships[sequence[k]].Type == RelationshipNode::IsKernel);
                const auto consPartId = componentId[k];
                if (prodPartId != consPartId) {
                    assert (consPartId > 0);
                    add_edge(prodPartId, consPartId, u, T);
                }
            }
        }
    }

    llvm::BitVector V(partitionCount);

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (in_degree(i, T) == 0) {
            add_edge(0, i, 0, T);
            for (const auto e : make_iterator_range(out_edges(i, T))) {
                V.set(target(e, T));
            }
        }
    }

    for (const auto e : make_iterator_range(out_edges(0, T))) {
        const auto s = target(e, T);
        for (auto j = V.find_first(); j != -1; j = V.find_next(j)) {
            if (!edge(s, j, T).second) {
                add_edge(s, j, 0, T);
            }
        }
    }

    V.reset();

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (out_degree(i, T) == 0) {
            add_edge(i, partitionCount - 1, 0, T);
        }
    }

    std::vector<unsigned> renumberingSeq;
    renumberingSeq.reserve(partitionCount);

    if (LLVM_UNLIKELY(!lexical_ordering(T, renumberingSeq))) {
        report_fatal_error("Internal error: failed to generate final acyclic partition graph");
    }

    assert (renumberingSeq[0] == 0);

    std::vector<unsigned> renumbered(partitionCount);

    for (unsigned i = 0; i < partitionCount; ++i) {
        const auto j = renumberingSeq[i];
        assert (j < partitionCount);
        renumbered[j] = i;
    }

    assert (renumbered[0] == 0);

    PartitionGraph P(partitionCount);

    flat_set<unsigned> linkedPartitionGroups;

    PartitionIds.clear();

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            assert (componentId[i] < partitionCount);
            const auto j = renumbered[componentId[i]];
            assert (j < partitionCount);
            assert ((j > 0 && (j + 1) < partitionCount) ^ (node.Kernel == mPipelineKernel));
            PartitionData & pd = P[j];
            const auto pid = partitionIds[i];
            assert (pd.LinkedGroupId == 0 || pd.LinkedGroupId == pid);
            pd.LinkedGroupId = pid;
            linkedPartitionGroups.insert(pid);
            pd.Kernels.push_back(u);
            PartitionIds.emplace(u, j);
        }
    }

    for (unsigned partitionId = 0; partitionId < partitionCount; ++partitionId) {
        PartitionData & N = P[partitionId];
        const auto f = linkedPartitionGroups.find(N.LinkedGroupId);
        assert (f != linkedPartitionGroups.end());
        N.LinkedGroupId = std::distance(linkedPartitionGroups.begin(), f);
    }


    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    flat_set<unsigned> included;
    included.reserve(numOfKernels);
    for (const auto u : P[0].Kernels) {
        assert ("kernel is in multiple partitions?" && included.insert(u).second);
        const auto & R = Relationships[u];
        assert (R.Type == RelationshipNode::IsKernel);
        assert (R.Kernel == mPipelineKernel);
    }
    auto numOfPartitionedKernels = P[0].Kernels.size();
    for (unsigned i = 1; i < partitionCount; ++i) {
        numOfPartitionedKernels += P[i].Kernels.size();
        for (const auto u : P[i].Kernels) {
            assert ("kernel is in multiple partitions?" && included.insert(u).second);
            const auto & R = Relationships[u];
            assert (R.Type == RelationshipNode::IsKernel);
        }
    }
    assert (numOfPartitionedKernels == numOfKernels);
    END_SCOPED_REGION
    #endif

    flat_set<std::pair<unsigned, unsigned>> duplicateFilter;

    for (unsigned i = 0; i < partitionCount; ++i) {
        assert (P[i].Kernels.size() > 0);
        const auto j = renumbered[i];
        assert (duplicateFilter.empty());
        for (const auto e : make_iterator_range(out_edges(i, T))) {
            const auto k = renumbered[target(e, T)];
            assert (k > j);
            const auto streamSet = T[e];
            if (LLVM_UNLIKELY(streamSet == 0)) continue;
            assert (streamSet < num_vertices(Relationships));
            assert (Relationships[streamSet].Type == RelationshipNode::IsStreamSet);
            if (duplicateFilter.emplace(k, streamSet).second) {
                add_edge(j, k, streamSet, P);
            }
        }
        duplicateFilter.clear();
    }

    assert (partitionCount > 0);

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (in_degree(i, P) == 0) {
            add_edge(0, i, 0, P);
        }
        if (out_degree(i, P) == 0) {
            add_edge(i, partitionCount - 1, 0, P);
        }
    }

    PartitionCount = partitionCount;

    // Convert the dataflow expectations to the new partitioning graph.

    // TODO: there is almost certainly a more efficient way to do this

    struct Expected {
        Rational Reps;
        Rational CoV;
        Expected(Rational reps, Rational cov) : Reps(reps), CoV(cov) { }
    };

    flat_map<unsigned, Expected> update;
    update.reserve(numOfKernels);

    for (auto i = 0UL, l = num_vertices(initial); i < l; ++i) {
        const PartitionData & D = initial[i];
        const auto m = D.Kernels.size();
        assert (D.Repetitions.size() == m);
        const auto exp = std::max(floor(D.ExpectedStridesPerSegment + Rational{1, 2}), 1U);
        for (unsigned j = 0; j < m; ++j) {
            const auto k = D.Kernels[j];
            assert (Relationships[k].Type == RelationshipNode::IsKernel);
            const auto reps = D.Repetitions[j] * exp;
            assert (Relationships[k].Kernel == mPipelineKernel || reps > Rational{0});
            update.emplace(std::make_pair(k, Expected{reps, D.StridesPerSegmentCoV}));
        }
    }

    assert (update.size() == numOfKernels);

    for (unsigned partitionId = 0; partitionId < partitionCount; ++partitionId) {
        PartitionData & N = P[partitionId];
        const auto m = N.Kernels.size();
        N.Repetitions.resize(m);
        for (unsigned j = 0; j < m; ++j) {
            const auto f = update.find(N.Kernels[j]);
            assert (f != update.end());
            const Expected & E = f->second;
            N.Repetitions[j] = E.Reps;
            N.StridesPerSegmentCoV += E.CoV;
            #ifndef NDEBUG
            update.erase(f);
            #endif
        }
        N.StridesPerSegmentCoV /= m;
    }
    assert (update.empty());


    #ifdef PRINT_GRAPH_BITSETS
    BEGIN_SCOPED_REGION
    auto & out = errs();

    out << "digraph \"H2\" {\n";
    for (auto v : make_iterator_range(vertices(P))) {
        const PartitionData & D = P[v];
        const auto n = D.Kernels.size();

        out << "v" << v << " [label=\"";
        for (unsigned i = 0; i < n; ++i) {
            const auto k = D.Kernels[i];
            const RelationshipNode & node = Relationships[k];
            assert (node.Type == RelationshipNode::IsKernel);
            const auto & V = D.Repetitions[i];
            out << k << ". " << node.Kernel->getName()
                << " (" << V.numerator() << "/" << V.denominator() << ")\\n";
        }
        out << " -- linkId=" << D.LinkedGroupId << "\",shape=rect];\n";
    }
    for (auto e : make_iterator_range(edges(P))) {
        const auto s = source(e, P);
        const auto t = target(e, P);
        out << "v" << s << " -> v" << t << " [label=\"" << P[e] << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    END_SCOPED_REGION
    #endif

    return P;
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
