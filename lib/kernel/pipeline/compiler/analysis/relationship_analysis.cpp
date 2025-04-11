#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include "../internal/popcount_kernel.h"
#include "../internal/regionselectionkernel.h"
#include <boost/graph/topological_sort.hpp>
#include <llvm/Support/ErrorHandling.h>
#include <toolchain/toolchain.h>

namespace kernel {

// TODO: support call bindings that produce output that are inputs of
// other call bindings or become scalar outputs of the pipeline

// TODO: with a better model of stride rates, we could determine whether
// being unable to execute a kernel implies we won't be able to execute
// another and "skip" over the unnecessary kernels.

using RefVector = SmallVector<ProgramGraph::Vertex, 4>;

using KernelVertexVec = SmallVector<ProgramGraph::Vertex, 64>;

struct TruncatedStreamSetData {

    TruncatedStreamSetData(const unsigned producer, const unsigned binding, StreamSet * const src, const unsigned streamSet)
    : Producer(producer)
    , Binding(binding)
    , SourceData(src)
    , TruncatedStreamSetVertex(streamSet) {

    }

    const unsigned Producer;
    const unsigned Binding;
    StreamSet * const SourceData;
    const unsigned TruncatedStreamSetVertex;
};

using TruncatedStreamSetVec = SmallVector<TruncatedStreamSetData, 2>;

using CommandLineScalarVec = std::array<Relationship *, (unsigned)CommandLineScalarType::CommandLineScalarCount>;

using RedundantStreamSetMap = PipelineAnalysis::RedundantStreamSetMap;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitialPipelineGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::generateInitialPipelineGraph(KernelBuilder & b) {

//TODO: change enum tag to distinguish relationships and streamsets

struct RelationshipGraphBuilder {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addProducerRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addProducerStreamSets(const PortType portType, const unsigned producer, const Bindings & array) {
        const auto n = array.size();
        for (unsigned i = 0; i < n; ++i) {
            const Binding & item = array[i];
            const auto binding = G.add(RelationshipNode::IsBinding, &item);
            add_edge(producer, binding, RelationshipType{portType, i}, G);
            const auto rel = item.getRelationship();
            assert (isa<StreamSet>(rel) || isa<TruncatedStreamSet>(rel));
            const auto relationship = G.addOrFind(RelationshipNode::IsStreamSet, rel);
            add_edge(binding, relationship, RelationshipType{portType, i}, G);
            if (isa<TruncatedStreamSet>(rel)) {
                const Relationship * d = rel;
                for (;;) {
                    d = cast<TruncatedStreamSet>(d)->getData();
                    if (LLVM_LIKELY(!isa<TruncatedStreamSet>(d))) {
                        break;
                    }
                }
                assert (isa<StreamSet>(d) || isa<RepeatingStreamSet>(d));
                const auto s = static_cast<const StreamSet *>(d);
                TruncatedStreamSets.emplace_back(producer, binding, const_cast<StreamSet *>(s), relationship);
            }
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief canonicalizeAnyCommandLineScalar
     ** ------------------------------------------------------------------------------------------------------------- */
    inline Relationship * canonicalizeAnyCommandLineScalar(Relationship * const rel) { assert (rel);
        if (isa<CommandLineScalar>(rel)) {
            const unsigned k = (unsigned)cast<CommandLineScalar>(rel)->getCLType();
            if (CommandLineScalars[k] == nullptr) {
                CommandLineScalars[k] = rel;
            } else {
                return CommandLineScalars[k];
            }
        }
        return rel;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addProducerRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addProducerScalars(const PortType portType, const unsigned producer, const Bindings & array) {
        const auto n = array.size();
        for (unsigned i = 0; i < n; ++i) {
            Relationship * const r = canonicalizeAnyCommandLineScalar(array[i].getRelationship());
            assert (isa<Scalar>(r) || isa<CommandLineScalar>(r));
            const auto relationship = G.addOrFind(RelationshipNode::IsScalar, r);
            add_edge(producer, relationship, RelationshipType{portType, i}, G);
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addConsumerRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addConsumerStreamSets(const PortType portType, const unsigned consumer, const Bindings & array, const bool addRelationship) {
        const auto n = array.size();
        for (unsigned i = 0; i < n; ++i) {
            const Binding & item = array[i];
            const auto binding = G.add(RelationshipNode::IsBinding, &item);
            add_edge(binding, consumer, RelationshipType{portType, i}, G);
            const auto rel = item.getRelationship();
            assert (isa<RepeatingStreamSet>(rel) || isa<StreamSet>(rel) || isa<TruncatedStreamSet>(rel));
            auto relationship = G.addOrFind(RelationshipNode::IsStreamSet, rel, addRelationship || isa<RepeatingStreamSet>(rel));
            add_edge(relationship, binding, RelationshipType{portType, i}, G);
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addConsumerRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addConsumerScalars(const PortType portType, const unsigned consumer, const Bindings & array, const bool addRelationship) {
        const auto n = array.size();
        for (unsigned i = 0; i < n; ++i) {
            Relationship * const rel = canonicalizeAnyCommandLineScalar(array[i].getRelationship());
            assert (isa<Scalar>(rel) || isa<ScalarConstant>(rel) || isa<CommandLineScalar>(rel));
            const auto relationship = G.addOrFind(RelationshipNode::IsScalar, rel, addRelationship);
            add_edge(relationship, consumer, RelationshipType{portType, i}, G);
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addConsumerRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addConsumerCalls(const PortType portType, const CallBinding & call) {
        const auto & array = call.Args;
        const auto n = array.size();
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        const auto callFunc = G.addOrFind(RelationshipNode::IsCallee, &call);
        for (unsigned i = 0; i < n; ++i) {
            const auto relationship = G.find(RelationshipNode::IsScalar, array[i]);
            add_edge(relationship, callFunc, RelationshipType{portType, i}, G);
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getReferencePort
     ** ------------------------------------------------------------------------------------------------------------- */
    StreamSetPort getReferencePort(const Kernel * const kernel, const StringRef ref) {
        const Bindings & inputs = kernel->getInputStreamSetBindings();
        const auto n = inputs.size();
        for (unsigned i = 0; i != n; ++i) {
            if (ref.compare(inputs[i].getName()) == 0) {
                return StreamSetPort{PortType::Input, i};
            }
        }
        const Bindings & outputs = kernel->getOutputStreamSetBindings();
        const auto m = outputs.size();
        for (unsigned i = 0; i != m; ++i) {
            if (ref.compare(outputs[i].getName()) == 0) {
                return StreamSetPort{PortType::Output, i};
            }
        }
        SmallVector<char, 256> tmp;
        raw_svector_ostream msg(tmp);
        msg << "Invalid reference name: "
            << kernel->getName()
            << " does not contain a StreamSet called "
            << ref;
        report_fatal_error(StringRef(msg.str()));
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getReferenceBinding
     ** ------------------------------------------------------------------------------------------------------------- */
    inline const Binding & getReferenceBinding(const Kernel * const kernel, const StreamSetPort port) {
        if (port.Type == PortType::Input) {
            return kernel->getInputStreamSetBinding(port.Number);
        } else if (port.Type == PortType::Output) {
            return kernel->getOutputStreamSetBinding(port.Number);
        }
        llvm_unreachable("unknown port type?");
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addReferenceRelationships
     ** ------------------------------------------------------------------------------------------------------------- */
    void addReferenceRelationships(const PortType portType, const unsigned index, const Bindings & array) {
        const auto n = array.size();
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        for (unsigned i = 0; i != n; ++i) {
            const Binding & item = array[i];
            const ProcessingRate & rate = item.getRate();
            if (LLVM_UNLIKELY(rate.hasReference())) {
                const Kernel * const kernel = G[index].Kernel;
                const StreamSetPort refPort = getReferencePort(kernel, rate.getReference());
                if (LLVM_UNLIKELY(portType == PortType::Input && refPort.Type == PortType::Output)) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream msg(tmp);
                    msg << "Reference of input stream "
                        << kernel->getName()
                        << "."
                        << item.getName()
                        << " cannot refer to an output stream";
                    report_fatal_error(StringRef(msg.str()));
                }
                const Binding & ref = getReferenceBinding(kernel, refPort);
                assert (ref.getRelationship()->isStreamSet());
                if (LLVM_UNLIKELY(rate.isRelative() && ref.getRate().isFixed())) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream msg(tmp);
                    msg << "Reference of a Relative-rate stream "
                        << kernel->getName()
                        << "."
                        << item.getName()
                        << " cannot refer to a Fixed-rate stream";
                    report_fatal_error(StringRef(msg.str()));
                }
                // To preserve acyclicity, reference bindings always point to the binding that refers to it.
                // To simplify later I/O lookup, the edge stores the info of the reference port.
                auto I = G.find(RelationshipNode::IsBinding, &item);
                assert (in_degree(I, G) == 1);
                add_edge(G.find(RelationshipNode::IsBinding, &ref), I, RelationshipType{refPort, ReasonType::Reference}, G);
                assert (G[first_in_edge(I, G)].Reason != ReasonType::Reference);
                assert (in_degree(I, G) == 2);
            }
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addTruncatedStreamSetContraints
     ** ------------------------------------------------------------------------------------------------------------- */
    void addTruncatedStreamSetContraints() {
        for (const auto & c : TruncatedStreamSets) {
            const auto src = G.find(RelationshipNode::IsStreamSet, c.SourceData);
            const auto dst = c.TruncatedStreamSetVertex;
            assert (in_degree(dst, G) > 0);
            add_edge(src, dst, RelationshipType{ReasonType::Reference}, G);
            assert (G[first_in_edge(dst, G)].Reason != ReasonType::Reference);

            assert (G[c.Binding].Type == RelationshipNode::IsBinding);
            const Binding & output = G[c.Binding].Binding;
            const auto producer = c.Producer;
            assert (G[producer].Type == RelationshipNode::IsKernel);

            // TODO: if we take this as an input with an equivalent rate, do not add it as a check?
            // Or make general check to prevent that for any port drawing from the same streamset?

            Binding * const trunc = new Binding("#trunc", c.SourceData, output.getRate());
            mInternalBindings.emplace_back(trunc);
            const auto truncBinding = G.add(RelationshipNode::IsBinding, trunc, RelationshipNodeFlag::ImplicitlyAdded);

            const unsigned portNum = in_degree(producer, G);
            add_edge(src, truncBinding, RelationshipType{PortType::Input, portNum, ReasonType::ImplicitTruncatedSource}, G);
            add_edge(truncBinding, producer, RelationshipType{PortType::Input, portNum, ReasonType::ImplicitTruncatedSource}, G);



        }
    }



    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief mapInOutStreamSets
     ** ------------------------------------------------------------------------------------------------------------- */
    void mapInOutStreamSets(KernelVertexVec & kernelList) {


        assert (!codegen::DebugOptionIsSet(codegen::DisableInOutAttributes));
        assert (kernelList.size() == mKernels.size());

        const auto n = kernelList.size();

        flat_map<Relationship *, size_t> InOutRemap;

        for (unsigned i = 0; i < n; ++i) {
            const Kernel * const kernel = mKernels[i].Object;

            const auto vertex = kernelList[i];

            const auto & outputs = kernel->getOutputStreamSetBindings();

            for (unsigned i = 0; i < outputs.size(); ++i) {
                const Binding & output = outputs[i];

                for (const auto & attr : output.getAttributes()) {
                    if (LLVM_UNLIKELY(attr.getKind() == AttrId::InOut)) {
                        // If we discover an output has an InOut attribute, determine the matching input
                        // port and record the producing kernel of the original streamset. Later, we'll
                        // annotate the graph to ensure (and enforce) an ordering that does not allow the
                        // original streamset from being used after it has been modified by this kernel.

                        const auto & inputs = kernel->getInputStreamSetBindings();

                        bool notFound = true;
                        for (unsigned j = 0; j < inputs.size(); ++j) {
                            const Binding & input = inputs[j];
                            if (attr.label().compare(input.getName())==0) {

                                if (LLVM_UNLIKELY(InOutRemap.count(input.getRelationship()) != 0)) {
                                    SmallVector<char, 256> tmp;
                                    raw_svector_ostream msg(tmp);
                                    msg << kernel->getName() << "." << output.getName()
                                        << " is an InOut value for a streamset that is already an InOut for a prior kernel.";
                                }

                                InOutRemap.emplace(input.getRelationship(), vertex);

                                notFound = false;
                                break;
                            }
                        }

                        if (notFound) {
                            SmallVector<char, 256> tmp;
                            raw_svector_ostream msg(tmp);
                            msg << kernel->getName() << "." << output.getName()
                                << " is an InOut that does not refer to an input binding name.";
                        }
                    }
                }
            }

        }

        if (LLVM_LIKELY(InOutRemap.empty())) return;


        for (unsigned i = 0; i < n; ++i) {
            const Kernel * const kernel = mKernels[i].Object;

            const auto vertex = kernelList[i];

            const auto & inputs = kernel->getInputStreamSetBindings();
            for (unsigned j = 0; j < inputs.size(); ++j) {
                const Binding & input = inputs[j];
                const auto f = InOutRemap.find(input.getRelationship());
                if (LLVM_UNLIKELY(f != InOutRemap.end())) {
                    const auto remapKernel = f->second;
                    if (remapKernel != vertex) {
                        add_edge(vertex, remapKernel, RelationshipType{ReasonType::OrderingConstraint}, G);
                    }
                }
            }
        }


    }




    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief addPopCountKernels
     ** ------------------------------------------------------------------------------------------------------------- */
    void addPopCountKernels(KernelBuilder & b, Kernels & kernels, KernelVertexVec & vertex) {

        struct Edge {
            CountingType    Type;
            StreamSetPort   Port;
            size_t          StrideLength;

            Edge() : Type(Unknown), Port(), StrideLength() { }
            Edge(const CountingType type, const StreamSetPort port, size_t stepFactor) : Type(type), Port(port), StrideLength(stepFactor) { }
        };

        using Graph = adjacency_list<vecS, vecS, directedS, Relationship *, Edge>;
        using Vertex = Graph::vertex_descriptor;
        using Map = flat_map<Relationship *, Vertex>;

        const auto numOfKernels = kernels.size();

        Graph H(numOfKernels);
        Map M;

        for (unsigned i = 0; i < numOfKernels; ++i) {

            const Kernel * const kernel = kernels[i].Object;

            auto addPopCountDependency = [&](const ProgramGraph::vertex_descriptor bindingVertex,
                                             const RelationshipType & port) {

                const RelationshipNode & rn = G[bindingVertex];
                assert (rn.Type == RelationshipNode::IsBinding);
                const Binding & binding = rn.Binding;
                const ProcessingRate & rate = binding.getRate();
                if (LLVM_UNLIKELY(rate.isPopCount() || rate.isNegatedPopCount())) {
                    // determine which port this I/O port refers to
                    for (const auto e : make_iterator_range(in_edges(bindingVertex, G))) {
                        const RelationshipType & rt = G[e];
                        if (rt.Reason == ReasonType::Reference) {
                            const auto refStreamVertex = source(e, G);
                            const RelationshipNode & rn = G[refStreamVertex];
                            assert (rn.Type == RelationshipNode::IsBinding);
                            const Binding & refBinding = rn.Binding;
                            const ProcessingRate & refRate = refBinding.getRate();
                            Relationship * const refStream = refBinding.getRelationship();
                            const auto f = M.find(refStream);
                            Vertex refVertex = 0;
                            if (LLVM_UNLIKELY(f != M.end())) {
                                refVertex = f->second;
                            } else {
                                if (LLVM_UNLIKELY(refBinding.isDeferred() || !refRate.isFixed())) {
                                    SmallVector<char, 0> tmp;
                                    raw_svector_ostream msg(tmp);
                                    msg << kernel->getName();
                                    msg << ": pop count reference ";
                                    msg << refBinding.getName();
                                    msg << " must refer to a non-deferred Fixed rate stream";
                                    report_fatal_error(StringRef(msg.str()));
                                }
                                refVertex = add_vertex(refStream, H);
                                M.emplace(refStream, refVertex);
                            }
                            const Rational strideLength = refRate.getRate() * kernel->getStride();
                            if (LLVM_UNLIKELY(strideLength.denominator() != 1)) {
                                SmallVector<char, 0> tmp;
                                raw_svector_ostream msg(tmp);
                                msg << kernel->getName();
                                msg << ": pop count reference ";
                                msg << refBinding.getName();
                                msg << " cannot have a rational rate";
                                report_fatal_error(StringRef(msg.str()));
                            }
                            const auto type = rate.isPopCount() ? CountingType::Positive : CountingType::Negative;
                            add_edge(refVertex, i, Edge{type, port, strideLength.numerator()}, H);
                            return;
                        }
                    }
                    llvm_unreachable("could not find reference for popcount rate?");
                }
            };

            const auto j = G.find(RelationshipNode::IsKernel, kernel);

            for (const auto e : make_iterator_range(in_edges(j, G))) {
                addPopCountDependency(source(e, G), G[e]);
            }
            for (const auto e : make_iterator_range(out_edges(j, G))) {
                addPopCountDependency(target(e, G), G[e]);
            }
        }

        const auto n = num_vertices(H);
        if (LLVM_LIKELY(n == numOfKernels)) {
            return;
        }

        BaseDriver & driver = reinterpret_cast<BaseDriver &>(b.getDriver());

        IntegerType * const sizeTy = b.getSizeTy();

        assert (n > numOfKernels);

        kernels.reserve(n - numOfKernels);

        for (auto i = numOfKernels; i < n; ++i) {

            size_t strideLength = 0;
            #ifdef FORCE_POP_COUNTS_TO_BE_BITBLOCK_STEPS
            strideLength = b.getBitBlockWidth();
            #endif
            CountingType type = CountingType::Unknown;
            for (const auto e : make_iterator_range(out_edges(i, H))) {
                const Edge & ed = H[e];
                type |= ed.Type;
                if (strideLength == 0) {
                    strideLength = ed.StrideLength;
                } else {
                    strideLength = boost::gcd(strideLength, ed.StrideLength);
                }
            }
            assert (strideLength != 1);
            assert (type != CountingType::Unknown);

            StreamSet * positive = nullptr;
            if (LLVM_LIKELY(type & CountingType::Positive)) {
                positive = driver.CreateStreamSet(1, sizeTy->getBitWidth());
            }

            StreamSet * negative = nullptr;
            if (LLVM_UNLIKELY(type & CountingType::Negative)) {
                negative = driver.CreateStreamSet(1, sizeTy->getBitWidth());
            }
            assert (H[i]->isStreamSet());
            StreamSet * const input = static_cast<StreamSet *>(H[i]); assert (input);
            PopCountKernel * popCountKernel = nullptr;
            switch (type) {
                case CountingType::Positive:
                    popCountKernel = new PopCountKernel(b.getDriver(), PopCountKernel::POSITIVE, strideLength, input, positive);
                    break;
                case CountingType::Negative:
                    popCountKernel = new PopCountKernel(b.getDriver(), PopCountKernel::NEGATIVE, strideLength, input, negative);
                    break;
                case CountingType::Both:
                    popCountKernel = new PopCountKernel(b.getDriver(), PopCountKernel::BOTH, strideLength, input, positive, negative);
                    break;
                default: llvm_unreachable("unknown counting type?");
            }
            // Add the popcount kernel to the pipeline
            kernels.emplace_back(popCountKernel, 0U);
            mInternalKernels.emplace_back(popCountKernel);

            const auto k = G.add(RelationshipNode::IsKernel, popCountKernel, RelationshipNodeFlag::ImplicitlyAdded);
            vertex.push_back(k);
            addConsumerStreamSets(PortType::Input, k, popCountKernel->getInputStreamSetBindings(), false);
            addProducerStreamSets(PortType::Output, k, popCountKernel->getOutputStreamSetBindings());

            // subsitute the popcount relationships
            for (const auto e : make_iterator_range(out_edges(i, H))) {
                const Edge & ed = H[e];
                const Kernel * const kernel = kernels[target(e, H)].Object;
                const auto consumer = G.find(RelationshipNode::IsKernel, kernel);
                assert (ed.Type == CountingType::Positive || ed.Type == CountingType::Negative);
                StreamSet * const stream = ed.Type == CountingType::Positive ? positive : negative; assert (stream);
                const auto streamVertex = G.find(RelationshipNode::IsStreamSet, stream);

                // append the popcount rate stream to the kernel
                Rational stepRate{ed.StrideLength, strideLength * kernel->getStride()};
                Binding * const popCount = new Binding("#popcount" + std::to_string(ed.Port.Number), stream, FixedRate(stepRate));
                mInternalBindings.emplace_back(popCount);
                const auto popCountBinding = G.add(RelationshipNode::IsBinding, popCount, RelationshipNodeFlag::ImplicitlyAdded);

                const unsigned portNum = in_degree(consumer, G);
                add_edge(streamVertex, popCountBinding, RelationshipType{PortType::Input, portNum, ReasonType::ImplicitPopCount}, G);
                add_edge(popCountBinding, consumer, RelationshipType{PortType::Input, portNum, ReasonType::ImplicitPopCount}, G);

                auto rebind_reference = [&](const unsigned binding) {

                    RelationshipNode & rn = G[binding];
                    assert (rn.Type == RelationshipNode::IsBinding);

                    graph_traits<ProgramGraph>::in_edge_iterator ei, ei_end;
                    std::tie(ei, ei_end) = in_edges(binding, G);
                    assert (std::distance(ei, ei_end) == 2);

                    for (;;) {
                        const RelationshipType & type = G[*ei];
                        if (type.Reason == ReasonType::Reference) {
                            remove_edge(*ei, G);
                            break;
                        }
                        ++ei;
                        assert (ei != ei_end);
                    }

                    // create a new binding with the partial sum rate.
                    const Binding & orig = rn.Binding;
                    assert (orig.getRate().isPopCount() || orig.getRate().isNegatedPopCount());
                    Binding * const replacement = new Binding(orig, PartialSum(popCount->getName()));
                    mInternalBindings.emplace_back(replacement);
                    rn.Binding = replacement;

                    assert (in_degree(binding, G) > 0);
                    add_edge(popCountBinding, binding, RelationshipType{PortType::Input, portNum, ReasonType::Reference}, G);
                    assert (G[first_in_edge(binding, G)].Reason != ReasonType::Reference);
                };

                bool notFound = true;
                if (ed.Port.Type == PortType::Input) {
                    for (const auto e : make_iterator_range(in_edges(consumer, G))) {
                        const RelationshipType & type = G[e];
                        if (type.Number == ed.Port.Number) {
                            assert (type.Type == PortType::Input);
                            rebind_reference(source(e, G));
                            notFound = false;
                            break;
                        }
                    }
                } else { // if (ed.Port.Type == PortType::Output) {
                    for (const auto e : make_iterator_range(out_edges(consumer, G))) {
                        const RelationshipType & type = G[e];
                        if (type.Number == ed.Port.Number) {
                            assert (type.Type == PortType::Output);
                            rebind_reference(target(e, G));
                            notFound = false;
                            break;
                        }
                    }
                }
                if (LLVM_UNLIKELY(notFound)) {
                    report_fatal_error("Internal error: failed to locate PopCount binding.");
                }
            }
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief combineDuplicateKernels
     ** ------------------------------------------------------------------------------------------------------------- */
    void combineDuplicateKernels(KernelVertexVec & kernelList) {

        using StreamSetVector = std::vector<std::pair<unsigned, StreamSetPort>>;
        using ScalarVector = std::vector<unsigned>;

        struct KernelId {
            const std::string Id;
            const StreamSetVector Streams;
            const ScalarVector Scalars;

            KernelId(const std::string & id, const StreamSetVector & streams, const ScalarVector & scalars)
            : Id(id), Streams(streams), Scalars(scalars) {

            }
            bool operator<(const KernelId & other) const {
                const auto diff = Id.compare(other.Id);
                if (LLVM_LIKELY(diff != 0)) {
                    return diff < 0;
                } else {
                    return (Scalars < other.Scalars) || (Streams < other.Streams);
                }
            }
        };

        std::map<KernelId, unsigned> Ids;

        ScalarVector scalars;
        StreamSetVector inputs;
        ScalarVector outputs;

        for (;;) {
            bool unmodified = true;
            Ids.clear();

            for (const auto i : kernelList) {

                RelationshipNode & bn = G[i];
                if (bn.Type == RelationshipNode::IsKernel) {
                    const Kernel * const kernel = bn.Kernel;
                    // We cannot reason about a family of kernels nor safely combine two
                    // side-effecting kernels.
                    if ((bn.Flags & RelationshipNodeFlag::IndirectFamily) || kernel->hasAttribute(AttrId::SideEffecting)) {
                        continue;
                    }

                    const auto n = in_degree(i, G);
                    inputs.resize(n);
                    scalars.resize(n);
                    unsigned numOfStreams = 0;

                    for (const auto e : make_iterator_range(in_edges(i, G))) {
                        const RelationshipType & port = G[e];
                        const auto input = source(e, G);
                        const RelationshipNode & node = G[input];
                        if (node.Type == RelationshipNode::IsBinding) {
                            unsigned relationship = 0;
                            StreamSetPort ref{};
                            for (const auto e : make_iterator_range(in_edges(input, G))) {
                                RelationshipType & rt = G[e];
                                if (rt.Reason == ReasonType::Reference) {
                                    ref = rt;
                                    assert (G[source(e, G)].Type == RelationshipNode::IsBinding);
                                } else {
                                    relationship = source(e, G);
                                    assert (G[relationship].Type == RelationshipNode::IsStreamSet);
                                }
                            }
                            inputs[port.Number] = std::make_pair(relationship, ref);
                            ++numOfStreams;
                        } else if (node.Type == RelationshipNode::IsScalar) {
                            scalars[port.Number] = input;
                        }
                    }

                    inputs.resize(numOfStreams);
                    scalars.resize(n - numOfStreams);

                    KernelId id(kernel->getName(), inputs, scalars);

                    const auto f = Ids.emplace(std::move(id), i);
                    if (LLVM_UNLIKELY(!f.second)) {
                        // We already have an identical kernel; replace kernel i with kernel j
                        bool error = false;
                        const auto j = f.first->second;
                        const auto m = out_degree(j, G);
                        if (LLVM_UNLIKELY(out_degree(i, G) != m)) {
                            error = true;
                        } else {

                            // Collect all of the output information from kernel j.
                            outputs.resize(m);
                            scalars.resize(m);
                            unsigned numOfStreams = 0;
                            for (const auto e : make_iterator_range(out_edges(j, G))) {

                                const RelationshipType & port = G[e];
                                const auto output = target(e, G);
                                const RelationshipNode & node = G[output];
                                if (node.Type == RelationshipNode::IsBinding) {
                                    const auto relationship = child(output, G);
                                    assert (G[relationship].Type == RelationshipNode::IsStreamSet);
                                    assert (isa<StreamSet>(G[relationship].Relationship));
                                    outputs[port.Number] = relationship;
                                    ++numOfStreams;
                                } else if (node.Type == RelationshipNode::IsScalar) {
                                    assert (isa<Scalar>(G[output].Relationship));
                                    scalars[port.Number] = output;
                                }
                            }
                            outputs.resize(numOfStreams);
                            scalars.resize(m - numOfStreams);

                            // Replace the consumers of kernel i's outputs with j's.
                            for (const auto e : make_iterator_range(out_edges(i, G))) {
                                const StreamSetPort & port = G[e];
                                const auto output = target(e, G);
                                const RelationshipNode & node = G[output];
                                unsigned original = 0;
                                if (node.Type == RelationshipNode::IsBinding) {
                                    const auto relationship = child(output, G);
                                    assert (G[relationship].Type == RelationshipNode::IsStreamSet);
                                    assert (isa<StreamSet>(G[relationship].Relationship));
                                    original = relationship;
                                } else if (node.Type == RelationshipNode::IsScalar) {
                                    original = output;
                                }

                                unsigned replacement = 0;
                                if (node.Type == RelationshipNode::IsBinding) {
                                    assert (port.Number < outputs.size());
                                    replacement = outputs[port.Number];
                                } else {
                                    assert (port.Number < scalars.size());
                                    replacement = scalars[port.Number];
                                }
                                assert (G[replacement].Type == G[original].Type);

                                Relationship * const a = G[original].Relationship;
                                Relationship * const b = G[replacement].Relationship;
                                if (LLVM_UNLIKELY(a->getType() != b->getType())) {
                                    error = true;
                                    break;
                                }

                                RedundantStreamSets.emplace(cast<StreamSet>(a), cast<StreamSet>(b));

                                struct EdgeReplacement {
                                    ProgramGraph::vertex_descriptor Source;
                                    ProgramGraph::vertex_descriptor Target;
                                    ProgramGraph::edge_property_type EdgeType;

                                    EdgeReplacement(ProgramGraph::vertex_descriptor s, ProgramGraph::vertex_descriptor t, ProgramGraph::edge_property_type et)
                                    : Source(s), Target(t), EdgeType(et) {

                                    }
                                };

                                SmallVector<EdgeReplacement, 16> toCopy;

                                for (const auto e : make_iterator_range(out_edges(original, G))) {
                                    const auto t = target(e, G);
                                    assert (G[t].Type == RelationshipNode::IsBinding);
                                    assert (G[e].Reason != ReasonType::Reference);
                                    assert (in_degree(t, G) == 1 || in_degree(t, G) == 2);
                                    toCopy.emplace_back(replacement, t, G[e]);
                                    // we need to ensure a canonical ordering. since boost won't let me insert
                                    // edges at a specific position using their exposed API, we remove the ref
                                    // edge and reinsert it in the correct position.
                                    if (LLVM_UNLIKELY(in_degree(t, G) == 2)) {
                                        auto ei = *(++in_edges(t, G).first);
                                        assert (G[ei].Reason == ReasonType::Reference);
                                        const auto s = source(ei, G);
                                        assert (G[s].Type == RelationshipNode::IsBinding);
                                        toCopy.emplace_back(s, t, G[ei]);
                                        remove_edge(s, t, G);
                                    }
                                }

                                clear_vertex(original, G);
                                RelationshipNode & rn = G[original];
                                rn.Type = RelationshipNode::IsNil;
                                rn.Relationship = nullptr;

                                for (auto & E : toCopy) {
                                    add_edge(E.Source, E.Target, E.EdgeType, G);
                                }

                            }
                            clear_vertex(i, G);
                            RelationshipNode & rn = G[i];
                            rn.Type = RelationshipNode::IsNil;
                            rn.Kernel = nullptr;
                            unmodified = false;
                        }

                        if (LLVM_UNLIKELY(error)) {
                            report_fatal_error(StringRef(kernel->getName()) + " is ambiguous: multiple I/O layouts have the same signature");
                        }
                    }
                }
            }
            if (unmodified) {
                break;
            }
        }
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief removeUnusedKernels
     ** ------------------------------------------------------------------------------------------------------------- */
    void removeUnusedKernels(const unsigned p_in, const unsigned p_out) {

        flat_set<unsigned> visited;
        std::queue<unsigned> pending;
        pending.push(p_out);
        assert (p_in < p_out);
        visited.insert_unique(p_in);
        visited.insert_unique(p_out);

        // identify all nodes that must be in the final pipeline
        for (const Binding & output : mPipelineKernel->getOutputScalarBindings()) {
            const auto p = G.find(RelationshipNode::IsScalar, output.getRelationship());
            pending.push(p);
            visited.insert_unique(p);
        }
        for (const CallBinding & C : mPipelineKernel->getCallBindings()) {
            const auto c = G.find(RelationshipNode::IsCallee, &C);
            pending.push(c);
            visited.insert_unique(c);
        }
        for (const auto & K : mKernels) {
            const Kernel * kernel = K.Object;
            if (LLVM_UNLIKELY(kernel->hasAttribute(AttrId::SideEffecting))) {
                const auto k = G.find(RelationshipNode::IsKernel, kernel);
                pending.push(k);
                visited.insert_unique(k);
            }
        }

        // determine the inputs for each of the required nodes
        for (;;) {
            const auto v = pending.front(); pending.pop();
            for (const auto e : make_iterator_range(in_edges(v, G))) {
                const auto input = source(e, G);
                if (visited.insert(input).second) {
                    pending.push(input);
                }
            }
            if (pending.empty()) {
                break;
            }
        }

        // To cut any non-required kernel from G, we cannot simply
        // remove every unvisited node as we still need to keep the
        // unused outputs of a kernel in G. Instead we make two
        // passes: (1) marks the outputs of all used kernels as
        // live. (2) deletes every dead node.

        for (const auto v : make_iterator_range(vertices(G))) {
            const RelationshipNode & rn = G[v];
            if (rn.Type == RelationshipNode::IsKernel) {
                if (LLVM_LIKELY(visited.count(v) != 0)) {
                    for (const auto e : make_iterator_range(out_edges(v, G))) {
                        const auto b = target(e, G);
                        const RelationshipNode & rb = G[b];
                        assert (rb.Type == RelationshipNode::IsBinding || rb.Type == RelationshipNode::IsScalar);
                        visited.insert(b); // output binding/scalar
                        if (LLVM_LIKELY(rb.Type == RelationshipNode::IsBinding)) {
                            if (LLVM_LIKELY(out_degree(b, G) > 0)) {
                                const auto f = first_out_edge(b, G);
                                assert (G[f].Reason != ReasonType::Reference);
                                visited.insert(target(f, G)); // output stream
                            }
                        }
                    }
                }
            }
        }

        for (const auto v : make_iterator_range(vertices(G))) {
            if (LLVM_UNLIKELY(visited.count(v) == 0)) {
                RelationshipNode & rn = G[v];
                clear_vertex(v, G);
                rn.Type = RelationshipNode::IsNil;
                rn.Kernel = nullptr;
            }
        }

    }


    RelationshipGraphBuilder(ProgramGraph & G, PipelineAnalysis & P)
    : G(G)
    , mPipelineKernel(P.mPipelineKernel)
    , mKernels(P.mKernels)
    , mInternalKernels(P.mInternalKernels)
    , mInternalBindings(P.mInternalBindings)
    , mInternalBuffers(P.mInternalBuffers)
    , RedundantStreamSets(P.RedundantStreamSets) {
        std::fill_n(CommandLineScalars.begin(), CommandLineScalars.size(), nullptr);
    }

    ProgramGraph & G;
    PipelineKernel * const mPipelineKernel;
    Kernels & mKernels;
    OwningVector<Kernel> &          mInternalKernels;
    OwningVector<Binding> &         mInternalBindings;
    OwningVector<StreamSetBuffer> & mInternalBuffers;
    RedundantStreamSetMap &         RedundantStreamSets;
    CommandLineScalarVec            CommandLineScalars;
    TruncatedStreamSetVec           TruncatedStreamSets;

};


    RelationshipGraphBuilder B(Relationships, *this);

    // Copy the list of kernels and add in any internal kernels
    assert (num_vertices(Relationships) == 0);
    const unsigned p_in = add_vertex(RelationshipNode(RelationshipNode::IsKernel, mPipelineKernel), Relationships);
    assert (p_in == PipelineInput);
    const auto n = mKernels.size();
    KernelVertexVec vertex(n);
    for (unsigned i = 0; i < n; ++i) {
        const auto & P = mKernels[i];
        const Kernel * K = P.Object;
        if (LLVM_UNLIKELY(K == mPipelineKernel)) {
            std::string tmp;
            raw_string_ostream msg(tmp);
            msg << mPipelineKernel->getName()
                << " contains itself in its pipeline";
            report_fatal_error(StringRef(msg.str()));
        }

        const auto flags = P.isFamilyCall() ? RelationshipNodeFlag::IndirectFamily : 0U;
        vertex[i] = Relationships.add(RelationshipNode::IsKernel, K, flags);
    }
    const unsigned p_out = add_vertex(RelationshipNode(RelationshipNode::IsKernel, mPipelineKernel), Relationships);
    PipelineOutput = p_out;

    // From the pipeline's perspective, a pipeline input node "produces" the inputs of the pipeline and a
    // pipeline output node "consumes" its outputs. Internally this means the inputs and outputs of the
    // pipeline are inverted from its external view but this change simplifies the analysis considerably
    // by permitting the compiler's internal graphs to acyclic.

    B.addProducerStreamSets(PortType::Output, p_in, mPipelineKernel->getInputStreamSetBindings());
    B.addConsumerStreamSets(PortType::Input, p_out, mPipelineKernel->getOutputStreamSetBindings(), true);

    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addProducerStreamSets(PortType::Output, vertex[i], K->getOutputStreamSetBindings());
    }
    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addConsumerStreamSets(PortType::Input, vertex[i], K->getInputStreamSetBindings(), false);
    }
    if (LLVM_LIKELY(!codegen::DebugOptionIsSet(codegen::DisableInOutAttributes))) {
        B.mapInOutStreamSets(vertex);
    }
    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addReferenceRelationships(PortType::Input, vertex[i], K->getInputStreamSetBindings());
    }
    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addReferenceRelationships(PortType::Output, vertex[i], K->getOutputStreamSetBindings());
    }
    B.addTruncatedStreamSetContraints();
    B.addPopCountKernels(b, mKernels, vertex);
    B.addProducerScalars(PortType::Output, p_in, mPipelineKernel->getInputScalarBindings());
    B.addConsumerScalars(PortType::Input, p_out, mPipelineKernel->getOutputScalarBindings(), true);
    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addProducerScalars(PortType::Output, vertex[i], K->getOutputScalarBindings());
    }

    for (unsigned i = 0; i < n; ++i) {
        const Kernel * const K = mKernels[i].Object;
        B.addConsumerScalars(PortType::Input, vertex[i], K->getInputScalarBindings(), true);
    }

    for (const CallBinding & C : mPipelineKernel->getCallBindings()) {
        B.addConsumerCalls(PortType::Input, C);
    }

    #ifndef NDEBUG
    for (auto v : make_iterator_range(vertices(B.G))) {
        const auto & vi = B.G[v];
        if (vi.Type == RelationshipNode::IsBinding) {
            assert (in_degree(v, B.G) == 1 || in_degree(v, B.G) == 2);
            const auto f = first_in_edge(v, B.G);
            assert (B.G[f].Reason != ReasonType::Reference);
        }
    }
    #endif

    // Pipeline optimizations
    if (LLVM_UNLIKELY(!codegen::EnableIllustrator)) {
        B.combineDuplicateKernels(vertex);
        B.removeUnusedKernels(p_in, p_out);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief transcribeRelationshipGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::transcribeRelationshipGraph(const PartitionGraph & initialGraph, const PartitionGraph & partitionGraph) {

    using Vertices = Vec<unsigned, 64>;

    // Compute the lexographical ordering of G
    std::vector<unsigned> O;
    if (LLVM_UNLIKELY(!lexical_ordering(Relationships, O))) {
        // TODO: inspect G to determine what type of cycle. E.g., do we have circular references in the binding of
        // a kernel or is it a problem with the I/O relationships?
        #ifndef NDEBUG
        printRelationshipGraph(Relationships, errs(), "Relationships");
        #endif
        report_fatal_error("Pipeline contains a cycle");
    }

    // TODO: in u32u8, calls to the StreamExpand/FieldDeposit kernels could be "merged" if we had the ability to
    // "re-execute" pipeline code with a different input kernel & I/O state. However, we may not necessarily want
    // to just re-execute the same kernel and may instead want to do the full sequence before repeating.

    Vertices kernels;
    Vertices bindings;
    Vertices streamSets;
    Vertices callees;
    Vertices scalars;

    for (const auto i : O) {
        const RelationshipNode & rn = Relationships[i];
        switch (rn.Type) {
            case RelationshipNode::IsKernel:
                assert (rn.Kernel);
                kernels.push_back(i);
                break;
            case RelationshipNode::IsScalar:
                scalars.push_back(i);
                break;
            case RelationshipNode::IsStreamSet:
                streamSets.push_back(i);
                break;
            case RelationshipNode::IsCallee:
                assert (&rn.Callee);
                callees.push_back(i);
                break;
            case RelationshipNode::IsBinding:
                assert (&rn.Binding);
                bindings.push_back(i);
                break;
            default:
                break;
        }
    }

    // Transcribe the pipeline graph based on the lexical ordering, accounting for any auxillary
    // kernels and subsituted kernels/relationships.

    const auto numOfKernels = kernels.size();
    assert (numOfKernels >= 2);
    const auto numOfStreamSets = streamSets.size();
    const auto numOfBindings = bindings.size();
    const auto numOfCallees = callees.size();
    const auto numOfScalars = scalars.size();

    SmallVector<unsigned, 256> subsitution(num_vertices(Relationships), -1U);

    FirstKernel = (numOfKernels == 2) ? PipelineInput : 1;
    LastKernel = PipelineInput + (numOfKernels - 2);

    // Now fill in all of the remaining kernels subsitute position
    KernelPartitionId.resize(numOfKernels);

    MinimumNumOfStrides.resize(numOfKernels);
    MaximumNumOfStrides.resize(numOfKernels);
    StrideRepetitionVector.resize(numOfKernels);

    KernelPartitionId[PipelineInput] = 0;

    unsigned inputPartitionId = -1U;
    unsigned outputPartitionId = -1U;
    unsigned currentGroupId = -1U;

    assert (kernels[0] == PipelineInput);
    assert (kernels[numOfKernels - 1] == PipelineOutput);

    auto calculateSegmentLengthBounds = [&](const unsigned kernelId, const unsigned partitionId, const unsigned newKernelId) {
        const PartitionData & P = partitionGraph[partitionId];
        const KernelIdVector & K = P.Kernels;
        assert (P.Repetitions.size() == K.size());
        const auto k = std::find(K.begin(), K.end(), kernelId);
        assert (k != K.end());
        const auto j = std::distance(K.begin(), k);
        const Rational & rep = P.Repetitions[j];
        assert (rep.denominator() == 1);
        const auto sl = rep.numerator();
        StrideRepetitionVector[newKernelId] = sl;
        const auto cov3 = P.StridesPerSegmentCoV * Rational{3};
        Rational ONE{1};
        const auto min = (cov3 > ONE) ? 0U: floor(ONE - cov3);
        const auto max = ceiling(ONE + cov3);
        assert (min <= max);


        MinimumNumOfStrides[newKernelId] = sl * min;
        MaximumNumOfStrides[newKernelId] = sl * max;
    };

    for (unsigned i = 0; i < (numOfKernels - 1); ++i) {
        const auto in = kernels[i];
        assert (subsitution[in] == -1U);
        const auto out = PipelineInput + i;
        subsitution[in] = out;

        const auto f = PartitionIds.find(in);
        assert (f != PartitionIds.end());
        const auto origPartitionId = f->second;

        calculateSegmentLengthBounds(in, origPartitionId, out);

        // renumber the partitions to reflect the selected ordering
        #ifdef FORCE_EACH_KERNEL_INTO_UNIQUE_PARTITION
        ++outputPartitionId;
        #else
        if (origPartitionId != inputPartitionId) {
            inputPartitionId = origPartitionId;
            const PartitionData & P = partitionGraph[origPartitionId];
            const auto groupId = P.LinkedGroupId;
            if (groupId != currentGroupId) {
                ++outputPartitionId;
                currentGroupId = groupId;
            }
        }
        #endif
        KernelPartitionId[out] = outputPartitionId;
    }

    const auto newPipelineOutput = LastKernel + 1U;
    KernelPartitionId[newPipelineOutput] = ++outputPartitionId;
    subsitution[PipelineOutput] = newPipelineOutput;

    BEGIN_SCOPED_REGION

    const auto f = PartitionIds.find(PipelineOutput);
    assert (f != PartitionIds.end());
    const auto origPartitionId = f->second;
    calculateSegmentLengthBounds(PipelineOutput, origPartitionId, newPipelineOutput);
    END_SCOPED_REGION

    PartitionCount = outputPartitionId + 1U;

    FirstKernelInPartition.resize(PartitionCount + 1U);
    FirstKernelInPartition[0] = PipelineInput;

    auto currentPartitionId = KernelPartitionId[FirstKernel];
    auto firstKernelInPartition = FirstKernel;
    FirstKernelInPartition[currentPartitionId] = firstKernelInPartition;
    for (auto kernel = (FirstKernel + 1U); kernel <= LastKernel; ++kernel) {
        const auto partitionId = KernelPartitionId[kernel];
        if (partitionId != currentPartitionId) {
            assert (partitionId == currentPartitionId + 1);
            // set the first kernel for the next partition
            firstKernelInPartition = kernel;
            currentPartitionId = partitionId;
            FirstKernelInPartition[partitionId] = kernel;
        }
    }

    FirstKernelInPartition[PartitionCount - 1] = newPipelineOutput;
    FirstKernelInPartition[PartitionCount] = newPipelineOutput;
#ifndef NDEBUG
    if (LLVM_UNLIKELY(IsNestedPipeline && (MinimumNumOfStrides[PipelineInput] != 1))) {
        auto checkIO = [](const Bindings & bindings) -> bool {
            for (const Binding & binding : bindings) {
                if (isCountable(binding) && !binding.isDeferred()) {
                    return true;
                }
            }
            return false;
        };
        if (checkIO(mPipelineKernel->getInputStreamSetBindings()) || checkIO(mPipelineKernel->getOutputStreamSetBindings())) {
            errs() << "WARNING! nested pipeline "
                   << mPipelineKernel->getName() <<
                      " requires more than one stride of input but has at least one "
                      "non-deferred Countable I/O rate. This may cause I/O errors "
                      "with the outer pipeline.\n\n"
                      "Check -PrintPipelineGraph for details.\n";
        }
    }
#endif


    // Originally, if the pipeline kernel does not have external I/O, both the pipeline in/out
    // nodes would be placed into the same (ignored) set but this won't be true after scheduling.
    // Similarly, if we disable partitioning, every kernel will be placed into its own partition.
    // Accept whatever the prior loops determines is the new partition count.

    PipelineOutput = newPipelineOutput;

    assert (FirstKernel < PipelineOutput);

    if (LLVM_UNLIKELY(numOfStreamSets == 0)) {
        FirstStreamSet = PipelineOutput;
        LastStreamSet = PipelineOutput;
    } else {
        FirstStreamSet = PipelineOutput + 1U;
        LastStreamSet = PipelineOutput + numOfStreamSets;
    }

    assert (FirstStreamSet <= LastStreamSet);

    FirstBinding = LastStreamSet + 1U;
    LastBinding = LastStreamSet + numOfBindings;

    FirstCall = PipelineOutput + 1U;
    LastCall = PipelineOutput + numOfCallees;
    FirstScalar = LastCall + 1U;
    LastScalar = LastCall + numOfScalars;

    assert (KernelPartitionId[PipelineInput] == 0);

    assert (Relationships[kernels[PipelineInput]].Kernel == mPipelineKernel);
    assert (Relationships[kernels[PipelineOutput]].Kernel == mPipelineKernel);

    for (unsigned i = 0; i < numOfStreamSets; ++i) {
        assert (subsitution[streamSets[i]] == -1U);
        subsitution[streamSets[i]] = FirstStreamSet + i;
    }
    for (unsigned i = 0; i < numOfBindings; ++i) {
        assert (subsitution[bindings[i]] == -1U);
        subsitution[bindings[i]] = FirstBinding  + i;
    }
    for (unsigned i = 0; i < numOfCallees; ++i) {
        assert (subsitution[callees[i]] == -1U);
        subsitution[callees[i]] = FirstCall + i;
    }
    for (unsigned i = 0; i < numOfScalars; ++i) {
        assert (subsitution[scalars[i]] == -1U);
        subsitution[scalars[i]] = FirstScalar + i;
    }
    // When constructing the initial partition graph, we identified which streamsets were
    // thread-local before we considered termination properties.
    mNonThreadLocalStreamSets.reserve(num_edges(initialGraph));
    for (auto e : make_iterator_range(edges(initialGraph))) {
        const auto streamSet = initialGraph[e];
        if (streamSet) {
            mNonThreadLocalStreamSets.insert(subsitution[streamSet]);
        }
    }

    SmallVector<std::pair<RelationshipType, unsigned>, 64> temp;

    auto transcribe = [&](const Vertices & V, RelationshipGraph & H) {
        for (const auto j : V) {
            assert (j < subsitution.size());
            const auto v = subsitution[j];
            assert (j < num_vertices(Relationships));
            assert (v < num_vertices(H));
            H[v] = Relationships[j];
        }
    };

    auto copy_in_edges = [&](const Vertices & V, RelationshipGraph & H,
            const RelationshipNode::RelationshipNodeType type) {
        for (const auto j : V) {
            const auto v = subsitution[j];
            for (const auto e : make_iterator_range(in_edges(j, Relationships))) {
                const auto i = source(e, Relationships);
                if (Relationships[i].Type == type) {
                    assert (Relationships[e].Reason != ReasonType::OrderingConstraint);
                    const auto u = subsitution[i];
                    assert (u < num_vertices(H));
                    temp.emplace_back(Relationships[e], u);
                }
            }
            std::sort(temp.begin(), temp.end());
            for (const auto & e : temp) {
                add_edge(e.second, v, e.first, H);
            }
            temp.clear();
        }
    };

    auto copy_out_edges = [&](const Vertices & V, RelationshipGraph & H,
            const RelationshipNode::RelationshipNodeType type) {
        for (const auto j : V) {
            const auto v = subsitution[j];
            for (const auto e : make_iterator_range(out_edges(j, Relationships))) {
                const auto i = target(e, Relationships);
                if (Relationships[i].Type == type) {
                    assert (Relationships[e].Reason != ReasonType::OrderingConstraint);
                    const auto w = subsitution[i];
                    assert (w < num_vertices(H));
                    temp.emplace_back(Relationships[e], w);
                }
            }
            std::sort(temp.begin(), temp.end());
            for (const auto & e : temp) {
                add_edge(v, e.second, e.first, H);
            }
            temp.clear();
        }
    };

    // create the stream graph
    mStreamGraph = RelationshipGraph{numOfKernels + numOfBindings + numOfStreamSets};

    transcribe(kernels, mStreamGraph);
    copy_in_edges(kernels, mStreamGraph, RelationshipNode::IsBinding);
    copy_out_edges(kernels, mStreamGraph, RelationshipNode::IsBinding);

    transcribe(streamSets, mStreamGraph);
    copy_in_edges(streamSets, mStreamGraph, RelationshipNode::IsStreamSet);
    copy_in_edges(streamSets, mStreamGraph, RelationshipNode::IsBinding);
    copy_out_edges(streamSets, mStreamGraph, RelationshipNode::IsBinding);

    transcribe(bindings, mStreamGraph);
    copy_out_edges(bindings, mStreamGraph, RelationshipNode::IsBinding);

     // create the scalar graph
    mScalarGraph = RelationshipGraph{numOfKernels + numOfCallees + numOfScalars};

    transcribe(kernels, mScalarGraph);
    copy_in_edges(kernels, mScalarGraph, RelationshipNode::IsScalar);
    copy_out_edges(kernels, mScalarGraph, RelationshipNode::IsScalar);

    transcribe(callees, mScalarGraph);
    copy_in_edges(callees, mScalarGraph, RelationshipNode::IsScalar);
    copy_out_edges(callees, mScalarGraph, RelationshipNode::IsScalar);

    transcribe(scalars, mScalarGraph);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addKernelRelationshipsInReferenceOrdering
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::addKernelRelationshipsInReferenceOrdering(const unsigned kernel, const RelationshipGraph & G,
                                                                 std::function<void(PortType, unsigned, unsigned)> insertionFunction) {

    using Graph = adjacency_list<hash_setS, vecS, bidirectionalS, RelationshipGraph::edge_descriptor>;
    using Vertex = graph_traits<Graph>::vertex_descriptor;

    const RelationshipNode & node = G[kernel];
    assert (node.Type == RelationshipNode::IsKernel);
    const Kernel * const kernelObj = node.Kernel; assert (kernelObj);

    unsigned maxInputPort = -1U;
    for (auto e : reverse(make_iterator_range(in_edges(kernel, G)))) {
        const auto binding = source(e, G);
        const RelationshipNode & rn = G[binding];
        if (LLVM_LIKELY(rn.Type == RelationshipNode::IsBinding)) {
            const RelationshipType & port = G[e];
            maxInputPort = port.Number;
            break;
        }
    }
    const auto numOfInputs = maxInputPort + 1U;

    unsigned maxOutputPort = -1U;
    for (auto e : reverse(make_iterator_range(out_edges(kernel, G)))) {
        const auto binding = target(e, G);
        const RelationshipNode & rn = G[binding];
        if (LLVM_LIKELY(rn.Type == RelationshipNode::IsBinding)) {
            const RelationshipType & port = G[e];
            maxOutputPort = port.Number;
            break;
        }
    }
    const auto numOfOutputs = maxOutputPort + 1U;

    // Evaluate the input/output ordering here and ensure that any reference port is stored first.
    const auto numOfPorts = numOfInputs + numOfOutputs;

    if (LLVM_UNLIKELY(numOfPorts == 0)) {
        return;
    }

    Graph E(numOfPorts);

    for (auto e : make_iterator_range(in_edges(kernel, G))) {
        const auto binding = source(e, G);
        const RelationshipNode & rn = G[binding];
        if (LLVM_LIKELY(rn.Type == RelationshipNode::IsBinding)) {
            const RelationshipType & port = G[e];
            assert (port.Number < numOfInputs);
            E[port.Number] = e;
            if (LLVM_UNLIKELY(in_degree(binding, G) != 1)) {
                for (const auto f : make_iterator_range(in_edges(binding, G))) {
                    const RelationshipType & ref = G[f];
                    if (ref.Reason == ReasonType::Reference) {
                        if (LLVM_UNLIKELY(port.Type == PortType::Output)) {
                            SmallVector<char, 256> tmp;
                            raw_svector_ostream out(tmp);
                            out << "Error: input reference for binding " <<
                                   kernelObj->getName() << "." << rn.Binding.get().getName() <<
                                   " refers to an output stream.";
                            report_fatal_error(StringRef(out.str()));
                        }
                        add_edge(ref.Number, port.Number, E);
                        break;
                    }
                }
            }
        }

    }

    for (auto e : make_iterator_range(out_edges(kernel, G))) {
        const auto binding = target(e, G);
        const RelationshipNode & rn = G[binding];
        if (LLVM_LIKELY(rn.Type == RelationshipNode::IsBinding)) {
            const RelationshipType & port = G[e];
            assert (port.Number < numOfOutputs);
            const auto portNum = port.Number + numOfInputs;
            E[portNum] = e;
            if (LLVM_UNLIKELY(in_degree(binding, G) != 1)) {
                for (const auto f : make_iterator_range(in_edges(binding, G))) {
                    const RelationshipType & ref = G[f];
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

    for (unsigned j = 1; j < numOfPorts; ++j) {
        add_edge_if_no_induced_cycle(j - 1, j);
    }

    SmallVector<Graph::vertex_descriptor, 16> ordering;
    ordering.reserve(numOfPorts);
    lexical_ordering(E, ordering);
    assert (ordering.size() == numOfPorts);

    for (const auto k : ordering) {
        const auto e = E[k];
        const RelationshipType & port = G[e];
        if (port.Type == PortType::Input) {
            const auto binding = source(e, G);
            assert(G[binding].Type == RelationshipNode::IsBinding);
            const auto f = first_in_edge(binding, G);
            assert (G[f].Reason != ReasonType::Reference);
            const auto streamSet = source(f, G);
            assert (G[streamSet].Type == RelationshipNode::IsStreamSet);
            insertionFunction(PortType::Input, binding, streamSet);
        } else {
            const auto binding = target(e, G);
            assert(G[binding].Type == RelationshipNode::IsBinding);
            const auto f = first_out_edge(binding, G);
            assert (G[f].Reason != ReasonType::Reference);
            const auto streamSet = target(f, G);
            assert (G[streamSet].Type == RelationshipNode::IsStreamSet);
            insertionFunction(PortType::Output, binding, streamSet);
        }
    }

}

} // end of namespace
