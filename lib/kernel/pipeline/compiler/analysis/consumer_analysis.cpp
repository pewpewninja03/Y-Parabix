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

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        return;
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        auto id = streamSet;
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal() || bn.isConstant() || bn.hasZeroElementsOrWidth()) {
            id = 0;
        } else {

restart:    const auto & cn = mBufferGraph[id];
            if (LLVM_UNLIKELY(cn.isInOutRedirect())) {
                while (in_degree(id, InOutStreamSetReplacement) > 0) {
                    id = parent(id, InOutStreamSetReplacement);
                }
                goto restart;
            }

            if (LLVM_UNLIKELY(cn.isTruncated())) {
                for (auto ref : make_iterator_range(in_edges(id, mStreamGraph))) {
                    const auto & v = mStreamGraph[ref];
                    if (v.Reason == ReasonType::Reference) {
                        id = source(ref, mBufferGraph);
                        goto restart;
                    }
                }
            }

            if (LLVM_UNLIKELY(id != streamSet)) {
                assert (id >= FirstStreamSet && id < streamSet);
                id = mConsumerGraph[id];
            }
        }
        assert (id == 0 || (FirstStreamSet <= id && id <= streamSet));
        mConsumerGraph[streamSet] = id;
    }

    std::vector<size_t> lastConsumer(LastStreamSet - FirstStreamSet + 1U, 0U);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        // If we have no consumers, we do not want to update the consumer count on exit
        // as we would then have to retain a scalar for it. If this streamset is
        // returned to the outside environment, we cannot ever release data from it
        // even if it has an internal consumer.

        const auto id = mConsumerGraph[streamSet];
        if (id == 0) {
            continue;
        }
        assert (FirstStreamSet <= id && id <= streamSet);

        const auto pe = in_edge(streamSet, mBufferGraph);
        const auto producer = source(pe, mBufferGraph);

        const BufferPort & output = mBufferGraph[pe];
        assert (producer >= PipelineInput && producer <= LastKernel);
        add_edge(producer, streamSet, ConsumerEdge{output.Port, 0, ConsumerEdge::None}, mConsumerGraph);
        assert (in_degree(id, mConsumerGraph) == 1);

        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.

        bool allConsumersFixedRate = true;
        const auto partId = KernelPartitionId[producer];
        for (const auto ce : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(ce, mBufferGraph);
            const BufferPort & input = mBufferGraph[ce];
            const auto isFixed = input.isFixed();
            if (LLVM_UNLIKELY(KernelPartitionId[consumer] == partId)) {
                assert (isFixed);
                assert (id >= FirstStreamSet);
                auto & lc = lastConsumer[id - FirstStreamSet];
                assert (consumer > 0);
                lc = std::max(lc, consumer);
            } else {
                const unsigned index = out_degree(id, mConsumerGraph);
                allConsumersFixedRate &= isFixed;
                add_edge(id, consumer, ConsumerEdge{input.Port, index, ConsumerEdge::UpdateConsumedCount}, mConsumerGraph);
            }
        }

        assert (lastConsumer[streamSet - FirstStreamSet] == 0 || mConsumerGraph[streamSet] == streamSet);
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const auto lc = lastConsumer[streamSet - FirstStreamSet];
        if (lc != 0 && out_degree(streamSet, mConsumerGraph) == 0) {
            assert (mConsumerGraph[streamSet] == streamSet);
            for (auto ss = streamSet; ss <= LastStreamSet; ++ss) {
                if (mConsumerGraph[ss] == streamSet) {
                    for (const auto ce : make_iterator_range(out_edges(ss, mBufferGraph))) {
                        const auto consumer = target(ce, mBufferGraph);
                        if (consumer == lc) {
                            const unsigned index = out_degree(streamSet, mConsumerGraph);
                            const BufferPort & input = mBufferGraph[ce];
                            add_edge(streamSet, consumer, ConsumerEdge{input.Port, index, ConsumerEdge::UpdateConsumedCount}, mConsumerGraph);
                        }
                    }
                }
            }
        }


    }

    for (auto id = FirstStreamSet; id <= LastStreamSet; ++id) {

        if (id != mConsumerGraph[id]) {
            continue;
        }

        BufferNode & bn = mBufferGraph[id];

        if (LLVM_UNLIKELY(out_degree(id, mConsumerGraph) == 0)) {
            const auto producer = parent(id, mBufferGraph);
            if (producer == PipelineInput || mTraceDynamicBuffers) {
                bn.Type |= BufferType::RequiresConsumedItemCount;
                add_edge(id, PipelineOutput, ConsumerEdge{}, mConsumerGraph);
                continue;
            } else {
                clear_in_edges(id, mConsumerGraph);
                continue;
            }
        }

        #ifndef NDEBUG
        assert (!(bn.isThreadLocal() || bn.isConstant() || bn.isTruncated()));
        assert (in_degree(id, mConsumerGraph) == 1);
        #endif

        bn.Type |= BufferType::RequiresConsumedItemCount;

        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.

        size_t lastConsumer = PipelineInput;
        for (const auto input : make_iterator_range(out_edges(id, mConsumerGraph))) {
            const auto consumer = target(input, mConsumerGraph);
            lastConsumer = std::max<size_t>(lastConsumer, consumer);
        }

        assert (lastConsumer != 0);

        // Although we may already know the final consumed item count prior
        // to executing the last consumer, we need to defer writing the final
        // consumed item count until the very last consumer reads the data.

        ConsumerGraph::edge_descriptor e;
        bool exists;
        std::tie(e, exists) = edge(id, lastConsumer, mConsumerGraph); assert (exists);
        ConsumerEdge & cn = mConsumerGraph[e];
        cn.Flags |= ConsumerEdge::WriteConsumedCount;


    }

    // If this is a pipeline input, we want to update the count at the end of the loop.
    for (const auto input : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[input];
        if (LLVM_LIKELY(bp.Port.Reason == ReasonType::Explicit)) {
            const auto streamSet = target(input, mBufferGraph);
            ConsumerGraph::edge_descriptor f;
            bool exists;
            std::tie(f, exists) = edge(streamSet, PipelineOutput, mConsumerGraph);
            if (exists) {
                ConsumerEdge & cn = mConsumerGraph[f];
                cn.Flags |= ConsumerEdge::UpdateExternalCount;
            } else {
                BufferNode & bn = mBufferGraph[streamSet];
                bn.Type |= BufferType::RequiresConsumedItemCount;
                add_edge(streamSet, PipelineOutput, ConsumerEdge{bp.Port, 0, ConsumerEdge::UpdateExternalCount}, mConsumerGraph);
            }
        }
    }

#if 0
    BEGIN_SCOPED_REGION
    auto & out = errs();
    out << "digraph \"ConsumerGraph\" {\n";
    for (auto v : make_iterator_range(vertices(mConsumerGraph))) {
        out << "v" << v << " [label=\"";
        if (v == PipelineInput) {
            out << "Pin";
        } else if (v < PipelineOutput) {
            out << "K_" << v;
        } else if (v == PipelineOutput) {
            out << "Pout";
        } else {
            const auto id = mConsumerGraph[v];
            out << "S_" << v;
            if (id != v) {
                out << " (" << id << ')';
            }
        }



        out << "\"];\n";
    }
    for (auto e : make_iterator_range(edges(mConsumerGraph))) {
        const auto s = source(e, mConsumerGraph);
        const auto t = target(e, mConsumerGraph);
        out << "v" << s << " -> v" << t <<
               " [label=\"";
        const ConsumerEdge & c = mConsumerGraph[e];
        if ((c.Flags & ConsumerEdge::UpdateConsumedCount) == 0) {
            out << 'N';
        }
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
