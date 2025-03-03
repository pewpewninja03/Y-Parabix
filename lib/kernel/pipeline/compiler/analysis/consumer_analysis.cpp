#include "pipeline_analysis.hpp"
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeConsumerGraph
 *
 * Copy the buffer graph but amalgamate any multi-edges into a single one
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeConsumerGraph() {

    mConsumerGraph = ConsumerGraph(LastStreamSet + 1);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        // If we have no consumers, we do not want to update the consumer count on exit
        // as we would then have to retain a scalar for it. If this streamset is
        // returned to the outside environment, we cannot ever release data from it
        // even if it has an internal consumer.

        auto id = streamSet;

        const BufferNode & bn = mBufferGraph[id];
        if (bn.isThreadLocal() || bn.isConstant() || bn.isReturned() || in_degree(id, InOutStreamSetReplacement) != 0) {
            continue;
        }

        if (LLVM_UNLIKELY(bn.isTruncated())) {
            for (auto ref : make_iterator_range(in_edges(streamSet, mStreamGraph))) {
                const auto & v = mStreamGraph[ref];
                if (v.Reason == ReasonType::Reference) {
                    id = source(ref, mBufferGraph);
                    assert (mBufferGraph[streamSet].isNonThreadLocal());
                    assert (id >= FirstStreamSet && id <= LastStreamSet);
                    break;
                }
            }

            const BufferNode & sn = mBufferGraph[id];
            if (sn.isThreadLocal() || sn.isConstant() || sn.isReturned()) {
                continue;
            }

        } else {
            // copy the producing edge
            const auto pe = in_edge(streamSet, mBufferGraph);
            const BufferPort & output = mBufferGraph[pe];
            const auto producer = source(pe, mBufferGraph);
            assert (producer >= PipelineInput && producer <= LastKernel);
            add_edge(producer, streamSet, ConsumerEdge{output.Port, 0, ConsumerEdge::None}, mConsumerGraph);
        }
        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.


        unsigned index = out_degree(id, mConsumerGraph);

        for (const auto ce : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(ce, mBufferGraph);
            const BufferPort & input = mBufferGraph[ce];
            add_edge(id, consumer, ConsumerEdge{input.Port, ++index, ConsumerEdge::UpdateConsumedCount}, mConsumerGraph);
        }


        auto check = streamSet;
        while (out_degree(check, InOutStreamSetReplacement) > 0)  {
            check = child(check, InOutStreamSetReplacement);
            for (const auto ce : make_iterator_range(out_edges(check, mBufferGraph))) {
                const auto consumer = target(ce, mBufferGraph);
                const BufferPort & input = mBufferGraph[ce];
                add_edge(id, consumer, ConsumerEdge{input.Port, ++index, ConsumerEdge::UpdateConsumedCount}, mConsumerGraph);
            }
        }


    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        if (LLVM_UNLIKELY(out_degree(streamSet, mConsumerGraph) == 0)) {
            clear_in_edges(streamSet, mConsumerGraph);
            continue;
        }

        #ifndef NDEBUG
        const BufferNode & bn = mBufferGraph[streamSet];
        assert (!(bn.isThreadLocal() || bn.isConstant() || bn.isReturned() || bn.isTruncated()));
        assert (in_degree(streamSet, mConsumerGraph) == 1);
        #endif

        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.

        size_t lastConsumer = PipelineInput;

        for (const auto ce : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
            const auto consumer = target(ce, mConsumerGraph);
            lastConsumer = std::max<size_t>(lastConsumer, consumer);
        }

        assert (lastConsumer != 0);

        // Although we may already know the final consumed item count prior
        // to executing the last consumer, we need to defer writing the final
        // consumed item count until the very last consumer reads the data.

        ConsumerGraph::edge_descriptor e;
        bool exists;
        std::tie(e, exists) = edge(streamSet, lastConsumer, mConsumerGraph); assert (exists);
        ConsumerEdge & cn = mConsumerGraph[e];
        cn.Flags |= ConsumerEdge::WriteConsumedCount;
    }

    // If this is a pipeline input, we want to update the count at the end of the loop.
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        ConsumerGraph::edge_descriptor f;
        bool exists;
        std::tie(f, exists) = edge(streamSet, PipelineOutput, mConsumerGraph);
        if (exists) {
            ConsumerEdge & cn = mConsumerGraph[f];
            cn.Flags |= ConsumerEdge::UpdateExternalCount;
        } else {
            const BufferPort & br = mBufferGraph[e];
            add_edge(streamSet, PipelineOutput, ConsumerEdge{br.Port, 0, ConsumerEdge::UpdateExternalCount}, mConsumerGraph);
        }
    }

#if 0
    BEGIN_SCOPED_REGION
    auto & out = errs();
    out << "digraph \"ConsumerGraph\" {\n";
    for (auto v : make_iterator_range(vertices(mConsumerGraph))) {
        out << "v" << v << " [label=\"" << v << "\"];\n";
    }
    for (auto e : make_iterator_range(edges(mConsumerGraph))) {
        const auto s = source(e, mConsumerGraph);
        const auto t = target(e, mConsumerGraph);
        out << "v" << s << " -> v" << t <<
               " [label=\"";
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::WriteConsumedCount) {
            out << 'W';
        }
        if (c.Flags & ConsumerEdge::UpdateExternalCount) {
            out << 'E';
        }
        out << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    END_SCOPED_REGION
#endif

}

}
