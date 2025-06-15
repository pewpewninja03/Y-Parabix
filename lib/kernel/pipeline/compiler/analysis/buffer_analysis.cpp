#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <boost/container/flat_set.hpp>
#include <unistd.h>
#include <z3.h>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

// TODO: any buffers that exist only to satisfy the output dependencies are unnecessary.
// We could prune away kernels if none of their outputs are needed but we'd want some
// form of "fake" buffer for output streams in which only some are unnecessary. Making a
// single static thread local buffer thats large enough for one segment.

// TODO: can we "combine" static stream sets that are used together and use fixed offsets
// from the first set? Would this improve data locality or prefetching?

// TODO: generate thread local buffers when we can guarantee all produced data is consumed
// within the same segment "iteration"? We can eliminate synchronization for kernels that
// consume purely local data.

// TODO: if an external buffer is marked as managed, have it allocate and manage the
// buffer but not deallocate it.

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitialBufferGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::generateInitialBufferGraph(KernelBuilder & b) {

    mBufferGraph = BufferGraph(LastStreamSet + 1U);

    using Graph = adjacency_list<hash_setS, vecS, bidirectionalS, RelationshipGraph::edge_descriptor>;
    using Vertex = graph_traits<Graph>::vertex_descriptor;

    const auto disableThreadLocalMemory = DebugOptionIsSet(codegen::DisableThreadLocalStreamSets);

    InOutStreamSetReplacement = InOutGraph(LastStreamSet + 1U);


    Rational lowestSourceStreamSetIORate{std::numeric_limits<unsigned>::max()};

    for (auto kernel = PipelineInput; kernel <= PipelineOutput; ++kernel) {

        const RelationshipNode & node = mStreamGraph[kernel];
        const Kernel * const kernelObj = node.Kernel; assert (kernelObj);

        unsigned numOfZeroBoundGreedyInputs = 0;

        auto makeBufferPort = [&](const RelationshipType port,
                                  const RelationshipNode & bindingNode,
                                  const bool noThreadLocal,
                                  const unsigned streamSet) -> BufferPort {
            assert (bindingNode.Type == RelationshipNode::IsBinding);
            const Binding & binding = bindingNode.Binding;

            const ProcessingRate & rate = binding.getRate();
            Rational lb{rate.getLowerBound()};
            Rational ub{rate.getUpperBound()};
            if (LLVM_UNLIKELY(rate.isGreedy())) {
                if (LLVM_UNLIKELY(port.Type == PortType::Output)) {
                    if (LLVM_LIKELY(kernel == PipelineInput)) {
                        ub = std::max(std::max(ub, Rational{1}), lb);
                    } else {
                        SmallVector<char, 0> tmp;
                        raw_svector_ostream out(tmp);
                        out << "Greedy rate cannot be applied an output port: "
                            << kernelObj->getName() << "." << binding.getName();
                        report_fatal_error(out.str());
                    }
                } else {
                    const auto e = in_edge(streamSet, mBufferGraph);
                    const BufferPort & producerBr = mBufferGraph[e];
                    ub = std::max(lb, producerBr.Maximum);
                    if (lb.numerator() == 0) {
                        numOfZeroBoundGreedyInputs++;
                    }
                }
            } else {
                const auto strideLength = kernelObj->getStride();
                if (LLVM_UNLIKELY(rate.isRelative())) {
                    const Binding & ref = getBinding(kernel, getReference(kernel, port));
                    const ProcessingRate & refRate = ref.getRate();
                    lb *= refRate.getLowerBound();
                    ub *= refRate.getUpperBound();
                }
                lb *= strideLength;
                ub *= strideLength;
            }

            BufferPort bp(port, binding, lb, ub);

            auto cannotBePlacedIntoThreadLocalMemory = disableThreadLocalMemory || noThreadLocal;

            if (rate.isFixed()) {
                bp.Flags |= BufferPortType::IsFixed;
            } else if (LLVM_UNLIKELY(rate.isUnknown())) {
                bp.Flags |= BufferPortType::IsManaged;
                cannotBePlacedIntoThreadLocalMemory = true;
            } else if (LLVM_UNLIKELY(rate.isRelative())) {

                const auto refPort = getReference(kernel, port);

                auto adoptRefProperties = [&](const BufferPort & src) {
                    bp.Flags |= src.Flags;
                    bp.Add = src.Add;
                    bp.Delay = src.Delay;
                };

                if (refPort.Type == PortType::Input) {
                    for (auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                        const BufferPort & ref = mBufferGraph[input];
                        if (ref.Port.Number == refPort.Number) {
                            adoptRefProperties(ref);
                            break;
                        }
                    }
                } else {
                    for (auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                        const auto & ref = mBufferGraph[output];
                        if (ref.Port.Number == refPort.Number) {
                            adoptRefProperties(ref);
                            break;
                        }
                    }
                }
                bp.Flags |= BufferPortType::IsRelative;
            }

            BufferNode & bn = mBufferGraph[streamSet];

            for (const Attribute & attr : binding.getAttributes()) {
                switch (attr.getKind()) {
                    case AttrId::Add:
                        bp.Add = std::max<unsigned>(bp.Add, attr.amount());
                        break;
                    case AttrId::Delayed:
                        bp.Delay = std::max<unsigned>(bp.Delay, attr.amount());
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::LookAhead:
                        bp.LookAhead = std::max<unsigned>(bp.LookAhead, attr.amount());
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::LookBehind:
                        bp.LookBehind = std::max<unsigned>(bp.LookBehind, attr.amount());
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::Truncate:
                        bp.Truncate = std::max<unsigned>(bp.Truncate, attr.amount());
                        break;
                    case AttrId::Principal:
                        bp.Flags |= BufferPortType::IsPrincipal;
                        if (LLVM_UNLIKELY(!rate.isFixed() || port.Type == PortType::Output)) {
                            SmallVector<char, 0> tmp;
                            raw_svector_ostream out(tmp);
                            out << "Principal attribute may only be applied to a Fixed rate input port: "
                                << kernelObj->getName() << "." << binding.getName();
                            report_fatal_error(out.str());
                        }
                        break;
                    case AttrId::Deferred:
                        bp.Flags |= BufferPortType::IsDeferred;
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::SharedManagedBuffer:
                        bp.Flags |= BufferPortType::IsShared;
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::ManagedBuffer:
                        bp.Flags |= BufferPortType::IsManaged;
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::ReturnedBuffer:
                        bn.Type |= BufferType::Returned;
                        cannotBePlacedIntoThreadLocalMemory = true;
                        break;
                    case AttrId::EmptyReadOverflow:
                        // TODO: thread local buffers could technically read into the next buffer here as long
                        // as the final buffer has one block padding.
                    case AttrId::EmptyWriteOverflow:
                        BEGIN_SCOPED_REGION
                        auto width = b.getBitBlockWidth();
                        if (binding.getNumElements() == 1) {
                            const auto fw = binding.getFieldWidth();
                            assert ((width % fw) == 0);
                            width /= fw;
                        }
                        bp.EmptyOverflow = width;
                        END_SCOPED_REGION
                        bn.Type |= BufferType::RequiresEmptyOverflow;
                        bn.Overflow = std::max(bn.Overflow, bp.EmptyOverflow);
                        break;
                    case AttrId::InOut:
                        if (LLVM_LIKELY(!codegen::DebugOptionIsSet(codegen::DisableInOutAttributes))) {
                            if (LLVM_UNLIKELY(port.Type == PortType::Input)) {
                                SmallVector<char, 256> tmp;
                                raw_svector_ostream msg(tmp);
                                msg << "InOut attribute on " << kernelObj->getName() << "." << bindingNode.Binding.get().getName()
                                    << " may only be applied to an output binding.";
                                report_fatal_error(msg.str());
                            } else {
                                const auto & inputs = kernelObj->getInputStreamSetBindings();
                                for (unsigned j = 0; j < inputs.size(); ++j) {
                                    if (attr.label().compare(inputs[j].getName())==0) {
                                        for (auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                                            const BufferPort & refPort = mBufferGraph[input];
                                            if (refPort.Port.Number == j) {
                                                const auto refStreamSet = source(input, mBufferGraph);
                                                assert (FirstStreamSet <= refStreamSet && refStreamSet < streamSet);
                                                if (LLVM_UNLIKELY(out_degree(refStreamSet, InOutStreamSetReplacement) != 0)) {
                                                    SmallVector<char, 256> tmp;
                                                    raw_svector_ostream msg(tmp);
                                                    msg << "InOut attribute on " << kernelObj->getName() << "." << bindingNode.Binding.get().getName()
                                                        << " may not be applied to the same streamset more than once.";
                                                    report_fatal_error(msg.str());
                                                }
                                                add_edge(refStreamSet, streamSet, InOutStreamSetReplacement);
                                                bp.Flags |= (refPort.Flags & BufferPortType::IsDeferred);
                                                bn.Type |= BufferType::InOutRedirect;
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    default: break;
                }
            }

            if (LLVM_UNLIKELY(bn.Type & BufferType::RequiresEmptyOverflow)) {
                auto id = streamSet;
                for (;;) {
                    if (LLVM_LIKELY(in_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = parent(id, InOutStreamSetReplacement);
                    BufferNode & bi = mBufferGraph[id];
                    bi.Type |= BufferType::RequiresEmptyOverflow;
                }
            }

            const RelationshipNode & sn = mStreamGraph[streamSet];
            assert (sn.Type == RelationshipNode::IsStreamSet);
            assert (sn.Relationship);
            const StreamSet * ss = static_cast<const StreamSet *>(sn.Relationship);
            if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(ss))) {
                bn.Locality = BufferLocality::ConstantShared;
                bn.IsLinear = true;
            } else if (LLVM_UNLIKELY(ss->getNumElements() == 0 || ss->getFieldWidth() == 0)) {
                bn.Locality = BufferLocality::ZeroElementsOrWidth;
                bn.IsLinear = true;
            } else {
                if (LLVM_UNLIKELY(isa<TruncatedStreamSet>(ss))) {
                    bn.Type |= BufferType::Truncated;
                    cannotBePlacedIntoThreadLocalMemory = true;
                }
                if (cannotBePlacedIntoThreadLocalMemory) {
                    mNonThreadLocalStreamSets.insert(streamSet);
                }
            }
            return bp;
        };

        // TODO: replace this with abstracted function

        // Evaluate the input/output ordering here and ensure that any reference port is stored first.
        const auto numOfInputs = in_degree(kernel, mStreamGraph);
        const auto numOfOutputs = out_degree(kernel, mStreamGraph);

        const auto numOfPorts = numOfInputs + numOfOutputs;

        if (LLVM_UNLIKELY(numOfPorts == 0)) {
            continue;
        }

        Graph E(numOfPorts);

        #ifndef NDEBUG
        RelationshipType prior_in{};
        #endif

        size_t principalInputPort = 0;
        bool hasPrincipalInput = false;
        bool nonGuaranteedInputRate = false;

        for (auto e : make_iterator_range(in_edges(kernel, mStreamGraph))) {
            const RelationshipType & port = mStreamGraph[e];
            #ifndef NDEBUG
            assert (prior_in < port);
            prior_in = port;
            #endif
            const auto binding = source(e, mStreamGraph);
            const RelationshipNode & rn = mStreamGraph[binding];
            assert (rn.Type == RelationshipNode::IsBinding);
            const Binding & bn = rn.Binding;
            if (bn.hasAttribute(AttrId::Principal)) {
                principalInputPort = port.Number;
                hasPrincipalInput = true;
            }
            const Binding & bd = rn.Binding;
            const ProcessingRate & rate = bd.getRate();


            // The following targets a StreamCompress/StreamExpand edge case in icgrep. StreamExpand
            // has a popcount input, fixed input and fixed output. StreamCompress produces data at a
            // popcount rate (C) and StreamExpand consumes it with its popcount rate (E) but the
            // reference stream of each is not always equal. If E >>> C, this can lead to StreamExpand
            // suddenly producing an unpredictable amount of output (equal to its input fixed rate)
            // during the final stride.

            switch (rate.getKind()) {
                case RateId::Fixed:
                    break;
                case RateId::Greedy:
                    if (rate.getLowerBound() == Rational{0}) {
                        break;
                    }
                default:
                    nonGuaranteedInputRate = true;
            }

            E[port.Number] = e;
            if (LLVM_UNLIKELY(in_degree(binding, mStreamGraph) != 1)) {
                for (const auto f : make_iterator_range(in_edges(binding, mStreamGraph))) {
                    const RelationshipType & ref = mStreamGraph[f];
                    if (ref.Reason == ReasonType::Reference) {
                        if (LLVM_UNLIKELY(port.Type == PortType::Output)) {
                            SmallVector<char, 256> tmp;
                            raw_svector_ostream out(tmp);
                            out << "Error: input reference for binding " <<
                                   kernelObj->getName() << "." << rn.Binding.get().getName() <<
                                   " refers to an output stream.";
                            report_fatal_error(out.str());
                        }
                        add_edge(ref.Number, port.Number, E);
                        break;
                    }
                }
            }
        }

        #ifndef NDEBUG
        RelationshipType prior_out{};
        #endif
        for (auto e : make_iterator_range(out_edges(kernel, mStreamGraph))) {
            const RelationshipType & port = mStreamGraph[e];
            #ifndef NDEBUG
            assert (prior_out < port);
            prior_out = port;
            #endif
            const auto binding = target(e, mStreamGraph);
            assert (mStreamGraph[binding].Type == RelationshipNode::IsBinding);
            const auto portNum = port.Number + numOfInputs;
            E[portNum] = e;
            if (LLVM_UNLIKELY(in_degree(binding, mStreamGraph) != 1)) {
                for (const auto f : make_iterator_range(in_edges(binding, mStreamGraph))) {
                    const RelationshipType & ref = mStreamGraph[f];
                    if (ref.Reason == ReasonType::Reference) {
                        auto refPort = ref.Number;
                        if (LLVM_UNLIKELY(ref.Type == PortType::Output)) {
                            refPort += numOfInputs;
                        }
                        add_edge(refPort, portNum, E);
                        break;
                    }
                }
            }
        }

        BitVector V(numOfPorts);
        std::queue<Vertex> Q;

        auto add_edge_if_no_induced_cycle = [&](const Vertex s, const Vertex t) {
            // If s-t exists, skip adding this edge
            if (LLVM_UNLIKELY(edge(s, t, E).second || s == t)) {
                return;
            }

            // If G is a DAG and there is a t-s path, adding s-t will induce a cycle.
            if (in_degree(s, E) > 0) {
                // do a BFS to search for a t-s path
                V.reset();
                assert (Q.empty());
                Q.push(t);
                for (;;) {
                    const auto u = Q.front();
                    Q.pop();
                    for (auto e : make_iterator_range(out_edges(u, E))) {
                        const auto v = target(e, E);
                        if (LLVM_UNLIKELY(v == s)) {
                            // we found a t-s path
                            return;
                        }
                        if (LLVM_LIKELY(!V.test(v))) {
                            V.set(v);
                            Q.push(v);
                        }
                    }
                    if (Q.empty()) {
                        break;
                    }
                }
            }
            add_edge(s, t, E);
        };

        // Order the graph so a principal input is reasoned about as early as possible
        if (hasPrincipalInput) {
            for (unsigned j = 0; j < numOfInputs; ++j) {
                if (principalInputPort != j) {
                    add_edge(principalInputPort, j, E);
                }
            }
        }

        for (unsigned j = 1; j < numOfPorts; ++j) {
            add_edge_if_no_induced_cycle(j - 1, j);
        }

        SmallVector<Graph::vertex_descriptor, 16> ordering;
        ordering.reserve(numOfPorts);
        lexical_ordering(E, ordering);

        for (const auto k : ordering) {
            const auto e = E[k];
            const RelationshipType & port = mStreamGraph[e];
            if (port.Type == PortType::Input) {
                const auto binding = source(e, mStreamGraph);
                const RelationshipNode & rn = mStreamGraph[binding];
                assert (rn.Type == RelationshipNode::IsBinding);
                const auto f = first_in_edge(binding, mStreamGraph);
                assert (mStreamGraph[f].Reason != ReasonType::Reference);
                const auto streamSet = source(f, mStreamGraph);
                assert (mStreamGraph[streamSet].Type == RelationshipNode::IsStreamSet);
                add_edge(streamSet, kernel, makeBufferPort(port, rn, false, streamSet), mBufferGraph);
            } else {
                const auto binding = target(e, mStreamGraph);
                const RelationshipNode & rn = mStreamGraph[binding];
                assert (rn.Type == RelationshipNode::IsBinding);
                const auto f = first_out_edge(binding, mStreamGraph);
                assert (mStreamGraph[f].Reason != ReasonType::Reference);
                const auto streamSet = target(f, mStreamGraph);
                assert (mStreamGraph[streamSet].Type == RelationshipNode::IsStreamSet);
                add_edge(kernel, streamSet, makeBufferPort(port, rn, nonGuaranteedInputRate, streamSet), mBufferGraph);
                if (numOfInputs == 0) {
                    const auto & R = StreamSetIORate[streamSet - FirstStreamSet];
                    lowestSourceStreamSetIORate = std::min(lowestSourceStreamSetIORate, R);
                }
            }
        }

        // If this kernel is not a source kernel but all inputs have a zero lower bound, it doesnot have
        // explicit termination condition. Report an error if this is the case.

        if (LLVM_UNLIKELY(numOfZeroBoundGreedyInputs > 0 && numOfZeroBoundGreedyInputs == in_degree(kernel, mBufferGraph))) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << kernelObj->getName() << " must have at least one input port with a non-zero lowerbound"
                   " to have an explicit termination condition.";
            report_fatal_error(out.str());
        }
    }

    if (LLVM_LIKELY(FirstKernel != PipelineInput)) {
        for (auto i = FirstStreamSet; i <= LastStreamSet; ++i) {
            StreamSetIORate[i - FirstStreamSet] /= lowestSourceStreamSetIORate;
        }
    }

    for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        if (LLVM_UNLIKELY(mBufferGraph[output].isManaged())) {
            mBufferGraph[source(output, mBufferGraph)].Type |= BufferType::ManagedOutput;
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyLinearBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyOutputNodeIds() {

    const auto & lengthAssertions = mPipelineKernel->getLengthAssertions();

    if (lengthAssertions.empty()) {

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            BufferNode & bn = mBufferGraph[streamSet];
            bn.OutputItemCountId = streamSet;
        }

    } else {

        const auto n = LastStreamSet - FirstStreamSet + 1;

        flat_map<const StreamSet *, unsigned> StreamSetToNodeIdMap;
        StreamSetToNodeIdMap.reserve(n);

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            assert (mStreamGraph[streamSet].Type == RelationshipNode::IsStreamSet);
            const StreamSet * const ss = cast<StreamSet>(mStreamGraph[streamSet].Relationship);
            StreamSetToNodeIdMap.emplace(ss, streamSet - FirstStreamSet);
        }

        std::vector<unsigned> component(n);
        std::iota(component.begin(), component.end(), 0);

        std::function<unsigned(unsigned)> find = [&](unsigned x) {
            assert (x < n);
            if (component[x] != x) {
                component[x] = find(component[x]);
            }
            return component[x];
        };

        auto union_find = [&](unsigned x, unsigned y) {

            x = find(x);
            y = find(y);

            if (x < y) {
                component[y] = x;
            } else {
                component[x] = y;
            }

        };

        for (const auto & pair : lengthAssertions) {
            unsigned id[2];
            for (unsigned i = 0; i < 2; ++i) {
                const auto f = StreamSetToNodeIdMap.find(pair[i]);
                if (f == StreamSetToNodeIdMap.end()) {
                    report_fatal_error("Length equality assertions contains an unknown streamset");
                }
                id[i] = f->second;
            }
            auto a = id[0], b = id[1];
            if (b > a) {
                std::swap(a, b);
            }
            union_find(a, b);
        }

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            BufferNode & bn = mBufferGraph[streamSet];
            const auto id = FirstStreamSet + find(component[streamSet - FirstStreamSet]);
            assert (id >= FirstStreamSet && id <= streamSet);
            bn.OutputItemCountId = id;
        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyOwnedBuffers() {

    // fill in any unmanaged pipeline input buffers
    for (const auto input : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[input];
        if (LLVM_LIKELY(bp.Port.Reason == ReasonType::Explicit)) {
            const auto streamSet = target(input, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.Type |= BufferType::External | BufferType::Unowned;
            bn.Locality = BufferLocality::GloballyShared;
        }
    }

    // fill in any known managed buffers
    for (auto kernel = FirstKernel; kernel <= PipelineOutput; ++kernel) {
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            if (LLVM_UNLIKELY(rate.isManaged())) {
                // Every managed buffer is considered linear to the pipeline
                const auto streamSet = target(e, mBufferGraph);
                BufferNode & bn = mBufferGraph[streamSet];
                bn.Type |= BufferType::Unowned;
                if (rate.isShared()) {
                    bn.Type |= BufferType::Shared;
                }
            }
        }
    }

    // and pipeline output buffers ...
    for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[output];
        if (LLVM_LIKELY(bp.Port.Reason == ReasonType::Explicit)) {
            const auto streamSet = source(output, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.Type |= BufferType::External;
            if (LLVM_LIKELY(!IsNestedPipeline)) {
                bn.Type |= BufferType::Returned;
            }
            assert (bn.Locality != BufferLocality::ConstantShared);
            bn.Locality = BufferLocality::GloballyShared;
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyLinearBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyLinearBuffers() {

    // All pipeline I/O must be linear
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        BufferNode & N = mBufferGraph[streamSet];
        if (!N.isConstant()) {
            N.IsLinear = true;
        }
        assert (!N.isReturned());
    }

    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        BufferNode & N = mBufferGraph[streamSet];
        if (N.isReturned()) {
            N.IsLinear = true;
        }
    }

    #if defined(FORCE_ALL_INTRA_PARTITION_STREAMSETS_TO_BE_LINEAR) || defined(FORCE_ALL_INTER_PARTITION_STREAMSETS_TO_BE_LINEAR)
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        BufferNode & N = mBufferGraph[streamSet];
        if (N.isConstant()) continue;
        const auto producer = parent(streamSet, mBufferGraph);
        const auto partId = KernelPartitionId[producer];
        bool isIntraPartition = true;
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(input, mBufferGraph);
            if (KernelPartitionId[consumer] != partId) {
                isIntraPartition = false;
                break;
            }
        }
        if (isIntraPartition) {
        #ifdef FORCE_ALL_INTRA_PARTITION_STREAMSETS_TO_BE_LINEAR
            N.IsLinear = true;
        #endif
        } else {
        #ifdef FORCE_ALL_INTER_PARTITION_STREAMSETS_TO_BE_LINEAR
            N.IsLinear = true;
        #endif
        }
    }
    #endif

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyPortsThatModifySegmentLength
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyPortsThatModifySegmentLength() {

//    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
//    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;
    #ifndef TEST_ALL_KERNEL_INPUTS
    auto currentPartitionId = -1U;
    #endif

    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        #ifndef TEST_ALL_KERNEL_INPUTS
        const auto partitionId = KernelPartitionId[kernel];
        const bool isPartitionRoot = (partitionId != currentPartitionId);
        currentPartitionId = partitionId;
        #endif
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);            
            BufferPort & inputRate = mBufferGraph[e];
            #ifdef TEST_ALL_KERNEL_INPUTS
            inputRate.Flags |= BufferPortType::CanModifySegmentLength;
            #else
            const BufferNode & N = mBufferGraph[streamSet];
            if (isPartitionRoot || !N.IsLinear || N.isConstant()) {
                inputRate.Flags |= BufferPortType::CanModifySegmentLength;
            }
            #endif
        }
        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            BufferPort & outputRate = mBufferGraph[output];
            const auto streamSet = target(output, mBufferGraph);
            const BufferNode & N = mBufferGraph[streamSet];

            if (N.isUnowned()) {
                outputRate.Flags |= BufferPortType::CanModifySegmentLength;
            } else if (LLVM_UNLIKELY(in_degree(streamSet, InOutStreamSetReplacement) != 0)) {
                const auto srcStreamSet = parent(streamSet, InOutStreamSetReplacement);
                bool canModifyLength = false;
                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    if (source(input, mBufferGraph) == srcStreamSet) {
                        const BufferPort & inputRate = mBufferGraph[input];
                        const ProcessingRate & I = inputRate.getRate();
                        const ProcessingRate & O = outputRate.getRate();
                        if (O.isFixed()) {
                            if (LLVM_LIKELY(I.isFixed() && I.getRate() == O.getRate())) {
                                continue;
                            }
                        } else if (O.isRelative()) {
                            if (LLVM_LIKELY(O.getRate() == Rational{1})) {
                                const auto ref = getReference(kernel, outputRate.Port);
                                if (LLVM_LIKELY(ref == inputRate.Port)) {
                                    continue;
                                }
                            }
                        } else if (O.isPartialSum()) {
                            if (I.isPartialSum()) {
                                if (LLVM_LIKELY(getReference(kernel, inputRate.Port) == getReference(kernel, outputRate.Port))) {
                                    continue;
                                }
                            }
                        }
                        canModifyLength = true;
                        break;
                    }
                }
                if (canModifyLength) {
                    outputRate.Flags |= BufferPortType::CanModifySegmentLength;
                }
            }
        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief estimateInitialBufferSizes
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::estimateInitialBufferSizes(KernelBuilder & b) {

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        return;
    }

    assert (MinimumNumOfStrides[PipelineInput] == MaximumNumOfStrides[PipelineInput]);
    assert (MinimumNumOfStrides[PipelineInput] > 0);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        BufferNode & currentNode = mBufferGraph[streamSet];

        if (LLVM_UNLIKELY(in_degree(streamSet, InOutStreamSetReplacement) != 0)) {
            continue;
        }

        unsigned maxLookBehind = 0;
        unsigned maxOverflow = 0;
        Rational minVal{std::numeric_limits<unsigned>::max()};
        Rational maxVal{std::numeric_limits<unsigned>::min()};

        auto id = streamSet;

        for (;;) {

            const BufferNode & bn = mBufferGraph[id];

            if (LLVM_UNLIKELY(bn.isConstant() || bn.isUnowned())) {
                goto unhandled_streamset;
            }

            if (LLVM_LIKELY(!bn.isConstant())) {

                const auto producerOutput = in_edge(id, mBufferGraph);
                const BufferPort & producerRate = mBufferGraph[producerOutput];
                maxLookBehind = producerRate.LookBehind;
                const auto producer = source(producerOutput, mBufferGraph);
                const auto bMin = producerRate.Minimum * MinimumNumOfStrides[producer];
                assert (bMin.denominator() == 1);
                const auto bMax = producerRate.Maximum * MaximumNumOfStrides[producer];
                assert (bMax.denominator() == 1);
                assert (bMax >= bMin);
                const auto extra = std::max(producerRate.LookAhead, producerRate.Add);
                maxOverflow = std::max(maxOverflow, extra);

                minVal = std::min(minVal, bMin);
                maxVal = std::max(maxVal, bMax);
            }

            for (const auto e : make_iterator_range(out_edges(id, mBufferGraph))) {

                const BufferPort & consumerRate = mBufferGraph[e];

                const auto consumer = target(e, mBufferGraph);

                const auto cMin = consumerRate.Minimum * MinimumNumOfStrides[consumer];
                assert (cMin.denominator() == 1);
                const auto cMax = consumerRate.Maximum * MaximumNumOfStrides[consumer];
                assert (cMax.denominator() == 1);
                assert (cMax >= cMin);

                minVal = std::min(minVal, cMin);
                maxVal = std::max(maxVal, cMax);

                maxLookBehind = std::max(maxLookBehind, consumerRate.LookBehind);
                const auto extra = std::max(consumerRate.LookAhead, consumerRate.Add);
                maxOverflow = std::max(maxOverflow, extra);
                maxOverflow = std::max(maxOverflow, bn.PartialSumSpanLength);
            }

            if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                break;
            }
            id = child(id, InOutStreamSetReplacement);
        }

        BEGIN_SCOPED_REGION

        auto amount = (maxVal * Rational{2}) - minVal;
        assert (amount.denominator() == 1);
        assert (amount.numerator() > 0);

        if (maxLookBehind) {
            currentNode.RequiresUnderflow = !currentNode.IsLinear;
        }

        currentNode.MaxQuantityPerSegment = std::max<unsigned>(amount.numerator(), maxOverflow + maxLookBehind);
        assert (currentNode.MaxQuantityPerSegment > 0);
        currentNode.Overflow = std::max(currentNode.Overflow, maxOverflow);

        END_SCOPED_REGION

unhandled_streamset:

        continue;

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addStreamSetsToBufferGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::addStreamSetsToBufferGraph(KernelBuilder & b) {

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        return;
    }

    mInternalBuffers.resize(LastStreamSet - FirstStreamSet + 1);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        BufferNode & bn = mBufferGraph[streamSet];

        if (LLVM_UNLIKELY(bn.Buffer != nullptr)) {
            continue;
        }

        StreamSetBuffer * buffer = nullptr;
        if (LLVM_UNLIKELY(in_degree(streamSet, InOutStreamSetReplacement) > 0)) {
            const auto src = parent(streamSet, InOutStreamSetReplacement);
            assert (FirstStreamSet <= src && src < streamSet);
            bn.Buffer = mBufferGraph[src].Buffer;
            continue;
        } else if (LLVM_UNLIKELY(bn.isTruncated())) {
            continue;
        } else if (LLVM_UNLIKELY(bn.isConstant())) {
            const auto ss = cast<RepeatingStreamSet>(mStreamGraph[streamSet].Relationship);
            buffer = new RepeatingBuffer(streamSet, b, ss->getType(), ss->isUnaligned());
        } else {
            assert (!isa<RepeatingStreamSet>(mStreamGraph[streamSet].Relationship));
            const auto producerOutput = in_edge(streamSet, mBufferGraph);
            const BufferPort & producerRate = mBufferGraph[producerOutput];
            const Binding & output = producerRate.Binding;

            if (LLVM_UNLIKELY(bn.isUnowned() || bn.isThreadLocal() || bn.hasZeroElementsOrWidth())) {
                assert (!bn.isManagedOutput());
                buffer = new ExternalBuffer(streamSet, b, output.getType(), 0);
            } else { // is internal buffer

                // A DynamicBuffer is necessary when we cannot bound the amount of unconsumed data a priori.
                // E.g., if this buffer is externally used, we cannot analyze the dataflow rate of
                // external consumers.  Similarly if any internal consumer has a deferred rate, we cannot
                // analyze any consumption rates.

                if (LLVM_UNLIKELY(bn.isManagedOutput())) {
                    buffer = new ManagedDynamicBuffer(streamSet, b, output.getType(), 0U);
                } else {
                    buffer = new DynamicBuffer(streamSet, b, output.getType(), bn.RequiresUnderflow, bn.IsLinear, 0U);
                }
            }
        }

        assert ("missing buffer?" && buffer);
        mInternalBuffers[streamSet - FirstStreamSet].reset(buffer);
        bn.Buffer = buffer;
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) {
            StreamSetBuffer * buffer = nullptr;
            for (const auto e : make_iterator_range(in_edges(streamSet, mStreamGraph))) {
                if (mStreamGraph[e].Reason == ReasonType::Reference) {
                    const auto sourceStreamSet = source(e, mStreamGraph);
                    const auto & src = mBufferGraph[sourceStreamSet];
                    if (src.isConstant()) {
                        bn.Locality = BufferLocality::ConstantShared;
                    }
                    buffer = src.Buffer;
                    break;
                }
            }
            assert ("missing source buffer for truncated streamset?" && buffer);
            bn.Buffer = buffer;
        }
        assert (bn.Buffer);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyIllustratedStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyIllustratedStreamSets() {
    const auto & illustratorBindings = mPipelineKernel->getIllustratorBindings();

    if (LLVM_LIKELY(illustratorBindings.empty())) return;

    // TODO: we need to move this up in the analysis phase and mark kernels with illustrated bindings as implicitly
    // side effecting. Otherwise we may end up not reporting streamsets that the user expects.

    const auto n = illustratorBindings.size();
    assert (mIllustratedStreamSetBindings.empty());
    mIllustratedStreamSetBindings.reserve(n);

    for (auto & p : illustratorBindings) {
        StreamSet * ss = p.StreamSetObj;
check_for_additional_remapping:
        auto f = RedundantStreamSets.find(ss);
        if (LLVM_UNLIKELY(f != RedundantStreamSets.end())) {
            ss = f->second;
            assert (ss != p.StreamSetObj);
            goto check_for_additional_remapping;
        }

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const auto & node = mStreamGraph[streamSet];
            assert (node.Type == RelationshipNode::IsStreamSet);
            if (node.Relationship == ss) {
                const auto output = in_edge(streamSet, mBufferGraph);
                auto & bp = mBufferGraph[output];
                if (LLVM_UNLIKELY((bp.Flags & BufferPortType::Illustrated) != 0)) {
                    for (const auto & E : mIllustratedStreamSetBindings) {
                        if (E.Name.compare(p.Name) == 0) {
                            goto ignore_duplicate_entry;
                        }
                    }
                }
                bp.Flags |= BufferPortType::Illustrated;
                mBufferGraph[streamSet].Type |= HasIllustratedStreamset;
                BEGIN_SCOPED_REGION
                const auto producer = parent(streamSet, mBufferGraph);
                mBufferGraph[producer].Type |= HasIllustratedStreamset;
                END_SCOPED_REGION
                mIllustratedStreamSetBindings.emplace_back(streamSet, p);
ignore_duplicate_entry:
                break;
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief scanZeroInputAfterFinalItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::buildZeroInputGraph() {

    SmallVector<std::pair<size_t, unsigned>, 8> entries;

    const auto n = LastKernel - FirstKernel + 1;

    ZeroInputGraph G(n);

    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {

        assert (entries.empty());

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {

            const auto streamSet = source(e, mBufferGraph);
            assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
            const BufferNode & bn = mBufferGraph[streamSet];

            if (LLVM_UNLIKELY(bn.hasZeroElementsOrWidth())) {
                continue;
            }

            if (LLVM_UNLIKELY(!(bn.isTruncated() || bn.isConstant()))) {
                const auto producer = parent(streamSet, mBufferGraph);
                if (KernelPartitionId[producer] == KernelPartitionId[kernel]) {
                    continue;
                }
            }


            const BufferPort & port = mBufferGraph[e];
            assert (port.Port.Type == PortType::Input);
            const Binding & input = port.Binding;
            const ProcessingRate & rate = input.getRate();

            // TODO: have an "unsafe" override attribute for unowned ones? this isn't needed for
            // nested pipelines but could replace the source output.

            if (LLVM_UNLIKELY(rate.isGreedy() && !(bn.isUnowned() || bn.isTruncated() || bn.isConstant()))) {
                continue;
            }

            size_t w = 0;
            if (port.isDeferred()) {
                // we won't know how big a deferred entry; we still allocate based on need at run-time
                // but this will at least minimize the potential reallocs.
                w = std::numeric_limits<size_t>::max();
            } else {
                assert (port.Maximum.denominator() == 1);
                w = port.Maximum.denominator();
            }

            entries.emplace_back(w, port.Port.Number);
        }

        if (entries.empty()) {
            continue;
        }

        // sort primarily by size so we can merge any larger ones as needed.
        std::sort(entries.begin(), entries.end());

        const auto l = entries.size();

        assert (num_vertices(G) >= n);

        for (size_t k = num_vertices(G) - n; k < l; ++k) {
            add_vertex(G);
        }

        for (size_t k = 0U; k < l; ++k) {
            const auto portNum = entries[k].second;
            add_edge(kernel - FirstKernel, n + k, portNum, G);
        }

        entries.clear();
    }

    mZeroInputGraph = G;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setStreamSetLockIds
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::setStreamSetLockIds() {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        BufferNode & bn = mBufferGraph[streamSet];

        if (bn.isThreadLocal() || bn.isInOutRedirect()) {
            continue;
        }

        if (LLVM_UNLIKELY(out_degree(streamSet, InOutStreamSetReplacement) > 0)) {
            auto id = streamSet;
            for (;;) {
                id = child(id, InOutStreamSetReplacement);
                if (out_degree(id, InOutStreamSetReplacement) == 0) {
                    break;
                }
            }
            const auto kernelLock = parent(id, mBufferGraph);
            assert (FirstKernel <= kernelLock && kernelLock <= LastKernel);
            bn.LockId = kernelLock;
        }
    }
}

}
