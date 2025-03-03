#include "common.hpp"
#include "../config.h"
namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReferenceVertex
 ** ------------------------------------------------------------------------------------------------------------- */
BOOST_NOINLINE
RelationshipGraph::edge_descriptor PipelineCommonGraphFunctions::getReferenceEdge(const size_t kernel, const StreamSetPort port) const {
    using InEdgeIterator = graph_traits<RelationshipGraph>::in_edge_iterator;
    using OutEdgeIterator = graph_traits<RelationshipGraph>::out_edge_iterator;
    RelationshipGraph::vertex_descriptor binding = 0;
    if (port.Type == PortType::Input) {
        InEdgeIterator ei, ei_end;
        std::tie(ei, ei_end) = in_edges(kernel, mStreamGraphRef);
        assert (port.Number < std::distance(ei, ei_end));
        const auto e = *(ei + port.Number);
        assert (mStreamGraphRef[e].Number == port.Number);
        binding = source(e, mStreamGraphRef);
    } else { // if (port.Type == PortType::Output) {
        OutEdgeIterator ei, ei_end;
        std::tie(ei, ei_end) = out_edges(kernel, mStreamGraphRef);
        assert (port.Number < std::distance(ei, ei_end));
        const auto e = *(ei + port.Number);
        assert (mStreamGraphRef[e].Number == port.Number);
        binding = target(e, mStreamGraphRef);
    }

    assert (mStreamGraphRef[binding].Type == RelationshipNode::IsBinding);
    assert (in_degree(binding, mStreamGraphRef) == 2);

    InEdgeIterator ei, ei_end;
    std::tie(ei, ei_end) = in_edges(binding, mStreamGraphRef);
    assert (std::distance(ei, ei_end) == 2);
    const auto e = *(ei + 1);
    assert (mStreamGraphRef[e].Reason == ReasonType::Reference);
    return e;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCommonGraphFunctions::getInputBinding(const size_t kernel, const StreamSetPort inputPort) const {

    RelationshipGraph::vertex_descriptor v;
    RelationshipGraph::edge_descriptor e;

    graph_traits<RelationshipGraph>::in_edge_iterator ei, ei_end;
    std::tie(ei, ei_end) = in_edges(kernel, mStreamGraphRef);
    assert (inputPort.Number < static_cast<size_t>(std::distance(ei, ei_end)));
    e = *(ei + inputPort.Number);
    v = source(e, mStreamGraphRef);

    assert (static_cast<StreamSetPort>(mStreamGraphRef[e]) == inputPort);
    const RelationshipNode & rn = mStreamGraphRef[v];
    assert (rn.Type == RelationshipNode::IsBinding);
    return rn.Binding;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInput
 ** ------------------------------------------------------------------------------------------------------------- */
const BufferPort & PipelineCommonGraphFunctions::getInputPort(const size_t kernel, const StreamSetPort inputPort) const {
    assert (inputPort.Type == PortType::Input);
    assert (inputPort.Number < in_degree(kernel, mBufferGraphRef));
    for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraphRef))) {
        const BufferPort & br = mBufferGraphRef[e];
        if (br.Port.Number == inputPort.Number) {
            return br;
        }
    }
    llvm_unreachable("could not find input port");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInput
 ** ------------------------------------------------------------------------------------------------------------- */
const BufferGraph::edge_descriptor PipelineCommonGraphFunctions::getInput(const size_t kernel, const StreamSetPort inputPort) const {
    assert (inputPort.Type == PortType::Input);
    assert (inputPort.Number < in_degree(kernel, mBufferGraphRef));
    for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraphRef))) {
        const BufferPort & br = mBufferGraphRef[e];
        if (br.Port.Number == inputPort.Number) {
            return e;
        }
    }
    llvm_unreachable("could not find input port");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCommonGraphFunctions::getOutputBufferVertex(const size_t kernel, const StreamSetPort outputPort) const {
    assert (outputPort.Type == PortType::Output);
    return target(getOutput(kernel, outputPort), mBufferGraphRef);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCommonGraphFunctions::getOutputBinding(const size_t kernel, const StreamSetPort outputPort) const {

    assert (outputPort.Type == PortType::Output);

    RelationshipGraph::vertex_descriptor v;
    RelationshipGraph::edge_descriptor e;

    graph_traits<RelationshipGraph>::out_edge_iterator ei, ei_end;
    std::tie(ei, ei_end) = out_edges(kernel, mStreamGraphRef);
    assert (outputPort.Number < static_cast<size_t>(std::distance(ei, ei_end)));
    e = *(ei + outputPort.Number);
    v = target(e, mStreamGraphRef);

    assert (static_cast<StreamSetPort>(mStreamGraphRef[e]) == outputPort);

    const RelationshipNode & rn = mStreamGraphRef[v];
    assert (rn.Type == RelationshipNode::IsBinding);
    return rn.Binding;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetBuffer * PipelineCommonGraphFunctions::getOutputBuffer(const size_t kernel, const StreamSetPort outputPort) const {
    assert (outputPort.Type == PortType::Output);
    return mBufferGraphRef[getOutputBufferVertex(kernel, outputPort)].Buffer;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutput
 ** ------------------------------------------------------------------------------------------------------------- */
const BufferGraph::edge_descriptor PipelineCommonGraphFunctions::getOutput(const size_t kernel, const StreamSetPort outputPort) const {
    assert (outputPort.Type == PortType::Output);
    assert (outputPort.Number < out_degree(kernel, mBufferGraphRef));
    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraphRef))) {
        const BufferPort & br = mBufferGraphRef[e];
        if (br.Port.Number == outputPort.Number) {
            return e;
        }
    }
    llvm_unreachable("could not find output port");
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputPort
 ** ------------------------------------------------------------------------------------------------------------- */
const BufferPort & PipelineCommonGraphFunctions::getOutputPort(const size_t kernel, const StreamSetPort outputPort) const {
    assert (outputPort.Type == PortType::Output);
    assert (outputPort.Number < out_degree(kernel, mBufferGraphRef));
    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraphRef))) {
        const BufferPort & br = mBufferGraphRef[e];
        if (br.Port.Number == outputPort.Number) {
            return br;
        }
    }
    llvm_unreachable("could not find output port");
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const Binding & PipelineCommonGraphFunctions::getBinding(const unsigned kernel, const StreamSetPort port) const {
    if (port.Type == PortType::Input) {
        return getInputBinding(kernel, port);
    } else if (port.Type == PortType::Output) {
        return getOutputBinding(kernel, port);
    }
    llvm_unreachable("unknown port binding type!");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief mayHaveNonLinearIO
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCommonGraphFunctions::mayHaveNonLinearIO(const size_t kernel) const {

    // If this kernel has I/O that crosses a partition boundary and the
    // buffer itself is not guaranteed to be linear then this kernel
    // may have non-linear I/O. A kernel with non-linear I/O may not be
    // able to execute its full segment without splitting the work across
    // two or more linear sub-segments.

    for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraphRef))) {
        const auto streamSet = source(input, mBufferGraphRef);
        const BufferNode & node = mBufferGraphRef[streamSet];
        if (!node.IsLinear) {
            return true;
        }
    }
    for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraphRef))) {
        const auto streamSet = target(output, mBufferGraphRef);
        const BufferNode & node = mBufferGraphRef[streamSet];
        if (!node.IsLinear) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isKernelStateFree
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCommonGraphFunctions::isKernelStateFree(const size_t kernel) const {
#ifdef DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION
    return false;
#else
    const Kernel * const kernelObj = getKernel(kernel);
    bool isExplicitlyMarkedAsStateFree = false;
    bool hasOverridableAttribute = false;
    bool hasForbiddenAttribute = false;

    for (const Attribute & attr : kernelObj->getAttributes()) {
        switch (attr.getKind()) {
            case AttrId::MayFatallyTerminate:
            case AttrId::CanTerminateEarly:
            case AttrId::MustExplicitlyTerminate:
            case AttrId::InternallySynchronized:
                hasForbiddenAttribute = true;
                break;
            case AttrId::SideEffecting:
                hasOverridableAttribute = true;
                break;
            case AttrId::Statefree:
                isExplicitlyMarkedAsStateFree = true;
                break;
            default: break;
        }
    }

    if ((mBufferGraphRef[kernel].Type & HasIllustratedStreamset) != 0) {
        return false;
    }

    if (hasForbiddenAttribute || kernelObj->getNumOfNestedKernelFamilyCalls()) {
        return false;
    }

    for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraphRef))) {
        const BufferPort & p = mBufferGraphRef[e];
        #ifdef PREVENT_CROSS_THREAD_KERNELS_FROM_BEING_STATEFREE
        const auto streamSet = source(e, mBufferGraphRef);
        if (mBufferGraphRef[streamSet].isCrossThreaded()) {
            return false;
        }
        #endif
        const Binding & b = p.Binding;
        const ProcessingRate & r = b.getRate();
        switch (r.getKind()) {
            case ProcessingRate::KindId::Fixed:
            case ProcessingRate::KindId::PartialSum:
            case ProcessingRate::KindId::Greedy:
                break;
            default:
                return false;
        }

        if (p.isDeferred()) {
            return false;
        }
    }

    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraphRef))) {
        const BufferPort & p = mBufferGraphRef[e];
        const Binding & b = p.Binding;
        const ProcessingRate & r = b.getRate();

        switch (r.getKind()) {
            case ProcessingRate::KindId::Fixed:
                break;
            case ProcessingRate::KindId::PartialSum:
                // We permit a partial sum output rate if and only if the kernel
                // was explicitly marked as statefree. Otherwise we cannot ensure
                // that the portion of a buffer that demarcates two invocations
                // will be correctly merged.
                if (isExplicitlyMarkedAsStateFree) break;
            default:
                return false;
        }

        if (p.isDeferred() || p.LookBehind) {
            return false;
        }
    }
    if (LLVM_UNLIKELY(isExplicitlyMarkedAsStateFree)) {
        return true;
    }
    if (LLVM_UNLIKELY(hasOverridableAttribute)) {
        return false;
    }
    StructType * const st = kernelObj->getSharedStateType();
    if (st == nullptr) {
        return true;
    }
    assert (st->getStructNumElements() >= kernelObj->getNumOfScalarInputs());
    return st->getStructNumElements() == kernelObj->getNumOfScalarInputs();
#endif
}

}
