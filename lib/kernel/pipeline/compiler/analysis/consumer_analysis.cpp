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
restart:    auto & sn = mBufferGraph[id];
            if (LLVM_UNLIKELY(sn.isInOutRedirect())) {
                id = parent(id, InOutStreamSetReplacement);
                assert (FirstStreamSet <= id && id < streamSet);
                id = mConsumerGraph[id];
                if (LLVM_LIKELY(id != 0)) goto restart;
                assert (mBufferGraph[id].isThreadLocal());
                goto skip_phase_check;
            }
            if (LLVM_UNLIKELY(sn.isTruncated())) {
                for (auto ref : make_iterator_range(in_edges(id, mStreamGraph))) {
                    const auto & v = mStreamGraph[ref];
                    if (v.Reason == ReasonType::Reference) {
                        id = source(ref, mBufferGraph);
                        assert (FirstStreamSet <= id && id < streamSet);
                        id = mConsumerGraph[id];
                        if (LLVM_LIKELY(id != 0)) goto restart;
                        assert (mBufferGraph[id].isThreadLocal());
                        goto skip_phase_check;
                    }
                }
            }
            assert (FirstStreamSet <= id && id <= streamSet);
            if (PartitionPhaseBoundaries.size() > 2) {
                const auto producer = parent(streamSet, mBufferGraph);
                assert (PipelineInput <= producer && producer <= PipelineOutput);
                const auto prodPhaseId = mBufferGraph[producer].ProducedPhaseId;
                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    assert (producer < consumer && consumer <= PipelineOutput);
                    const auto consPhaseId = mBufferGraph[consumer].ProducedPhaseId;
                    if (LLVM_UNLIKELY(consPhaseId != prodPhaseId)) {
                        #ifdef DISABLE_FD_BACKED_BUFFERS
                        constexpr auto flags = BufferType::CrossesPhaseBoundary;
                        #else
                        constexpr auto flags = BufferType::PreserveEntireStreamSet | BufferType::CrossesPhaseBoundary;
                        #endif
                        sn.Type |= flags;
                        break;
                    }
                }
            }
        }
skip_phase_check:
        mConsumerGraph[streamSet] = id;
    }

    // FIXME: for now we cannot safely release fd-backed buffers as output buffers.
    for (auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        auto streamSet = source(output, mBufferGraph);
        for (;;) {
            BufferNode & sn = mBufferGraph[streamSet];
            sn.Type &= ~BufferType::CrossesPhaseBoundary;
            if (LLVM_UNLIKELY(sn.isInOutRedirect())) {
                streamSet = parent(streamSet, InOutStreamSetReplacement);
            } else {
                break;
            }
        }
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
        BufferNode & sn = mBufferGraph[id];
        assert (sn.isNonThreadLocal());

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
        auto & lc = lastConsumer[id - FirstStreamSet];
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(input, mBufferGraph);
            assert (consumer > 0);
            lc = std::max(lc, consumer);
            const BufferPort & I = mBufferGraph[input];
            const unsigned index = out_degree(id, mConsumerGraph);
            allConsumersFixedRate &= I.isFixed();
            const auto flags = (KernelPartitionId[consumer] == partId) ? ConsumerEdge::None : ConsumerEdge::UpdateConsumedCount;
            add_edge(id, consumer, ConsumerEdge{I.Port, 0, flags}, mConsumerGraph);
        }

        sn.Type |= allConsumersFixedRate ? 0U : BufferType::HasNonFixedRateConsumer;
        assert (lastConsumer[streamSet - FirstStreamSet] == 0 || mConsumerGraph[streamSet] == streamSet);
    }

//    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
//        const auto lc = lastConsumer[streamSet - FirstStreamSet];
//        if (LLVM_UNLIKELY(lc == 0)) {
//            continue;
//        }
//        assert (mConsumerGraph[streamSet] == streamSet);
//        for (auto ss = streamSet; ss <= LastStreamSet; ++ss) {
//            if (mConsumerGraph[ss] == streamSet) {
//                for (const auto ce : make_iterator_range(out_edges(ss, mBufferGraph))) {
//                    const auto consumer = target(ce, mBufferGraph);
//                    if (consumer == lc) {
//                        const unsigned index = out_degree(streamSet, mConsumerGraph);
//                        const BufferPort & input = mBufferGraph[ce];
//                        add_edge(streamSet, consumer, ConsumerEdge{input.Port, index + 1, ConsumerEdge::UpdateConsumedCount}, mConsumerGraph);
//                    }
//                }
//            }
//        }
//    }

    constexpr auto propogatedFlagSet =
        BufferType::PreserveEntireStreamSet |
        BufferType::CrossesPhaseBoundary |
        BufferType::HasNonFixedRateConsumer;

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        const auto id = mConsumerGraph[streamSet];
        if (out_degree(id, mConsumerGraph) == 0) {
            continue;
        }
        assert (id > 0);
        assert (in_degree(id, mConsumerGraph) == 1);


        const BufferNode & sn = mBufferGraph[id];
        BufferNode & bn = mBufferGraph[streamSet];
        assert (bn.isNonThreadLocal());
        assert (id == streamSet || (bn.Type & propogatedFlagSet) == 0);
        bn.Type |= (sn.Type & propogatedFlagSet);

        assert (bn.isNonThreadLocal());

        const auto lc = lastConsumer[id - FirstStreamSet];

        if (LLVM_UNLIKELY(lc == 0)) {
            assert (out_degree(id, mConsumerGraph) == 0);
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

        bn.Type |= BufferType::RequiresConsumedItemCount;

        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.

        // Although we may already know the final consumed item count prior
        // to executing the last consumer, we need to defer writing the final
        // consumed item count until the very last consumer reads the data.


        ConsumerGraph::edge_descriptor e;
        bool exists;
        std::tie(e, exists) = edge(id, lc, mConsumerGraph); assert (exists);
        //  If there are consumers, update.
        ConsumerEdge & cn = mConsumerGraph[e];
        cn.Flags |= ConsumerEdge::UpdateConsumedCount | ConsumerEdge::WriteConsumedCount;

        remove_out_edge_if(id, [&](ConsumerGraph::edge_descriptor f) -> bool {
            #ifndef NDEBUG
            const auto consumer = target(f, mConsumerGraph);
            assert (FirstKernel <= consumer && consumer <= PipelineOutput);
            #endif
            return mConsumerGraph[f].Flags == ConsumerEdge::None;
        }, mConsumerGraph);

        size_t index = 0;
        for (auto f : make_iterator_range(out_edges(id, mConsumerGraph))) {
            auto & C = mConsumerGraph[f];
            assert (C.Flags != ConsumerEdge::None);
            C.Index = ++index;
        }
        assert (index == out_degree(id, mConsumerGraph));

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
        if (c.Index > 0) {
            out << ':' << c.Index;
        }
        out << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    END_SCOPED_REGION
#endif

}

}
