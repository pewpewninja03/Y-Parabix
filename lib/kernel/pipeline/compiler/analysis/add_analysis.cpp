#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        return;
    }

    SmallVector<int, 64> transitiveAdd(LastStreamSet - FirstStreamSet + 1);

    SmallVector<int, 16> inputAdd;

    for (auto i = PipelineInput; i <= PipelineOutput; ++i) {

        int minAddK = 0;

        const auto d = in_degree(i, mBufferGraph);


        if (LLVM_LIKELY(d > 0)) {
            bool noPrincipal = true;
            minAddK = std::numeric_limits<int>::max();
            int principalK = 0;
            if (inputAdd.size() < d) {
                inputAdd.resize(d);
            }

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                BufferPort & br = mBufferGraph[e];
                if (br.isFixed()) {
                    const int k = transitiveAdd[streamSet - FirstStreamSet] + br.Add;
                    minAddK = std::min(minAddK, k);
                    if (LLVM_UNLIKELY(br.isPrincipal())) {
                        principalK = k;
                        noPrincipal = false;
                        break;
                    }
                    inputAdd[br.Port.Number] = k;
                }
            }

            if (noPrincipal) {
                if (minAddK == std::numeric_limits<int>::max()) {
                    minAddK = 0;
                }
            } else {
                minAddK = principalK;
            }

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                BufferPort & br = mBufferGraph[e];
                if (br.isFixed()) {
                    assert (minAddK < std::numeric_limits<int>::max());
                    const int k = inputAdd[br.Port.Number] - minAddK;
                    br.OverflowSlackSpace = k;
                    BufferNode & bn = mBufferGraph[streamSet];
                    if (LLVM_LIKELY(k > bn.RequiredOverflowSpace)) {
                        bn.RequiredOverflowSpace = k;
                    }
                }
            }

        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            BufferPort & br = mBufferGraph[e];
            const auto k = minAddK + br.Add;
            const auto streamSet = target(e, mBufferGraph);
            transitiveAdd[streamSet - FirstStreamSet] = k;
            BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_LIKELY(k > bn.RequiredOverflowSpace)) {
                bn.RequiredOverflowSpace = k;
            }
        }
    }

    // Scan through the I/O for each kernel to see if some transitive
    // add relationship imposes an overflow requirement on a buffer

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_LIKELY(bn.isOwned() && !bn.isConstant())) {

            if (LLVM_LIKELY(in_degree(streamSet, InOutStreamSetReplacement) == 0)) {

                int maxOverflowSpace = 0;

                for (auto id = streamSet;;) {

                    const BufferNode & sn = mBufferGraph[id];
                    int required = sn.RequiredOverflowSpace;

                    const auto output = in_edge(id, mBufferGraph);
                    BufferPort & br = mBufferGraph[output];
                    const auto c = std::max<int>(br.OverflowSlackSpace, br.EmptyOverflow);
                    required = std::max<int>(required, c);

                    for (const auto input : make_iterator_range(out_edges(id, mBufferGraph))) {
                        const BufferPort & br = mBufferGraph[input];
                        const auto a = std::max<int>(br.OverflowSlackSpace, br.EmptyOverflow);
                        const auto b = std::max<int>(a, br.LookAhead);
                        required = std::max<int>(required, b);
                    }

                    const auto ta = transitiveAdd[id - FirstStreamSet];
                    maxOverflowSpace = std::max<int>(maxOverflowSpace, ta + required);

                    if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = child(id, InOutStreamSetReplacement);
                }

                for (auto id = streamSet;;) {

                    const auto output = in_edge(id, mBufferGraph);
                    BufferPort & br = mBufferGraph[output];
                    assert (maxOverflowSpace >= br.OverflowSlackSpace);
                    br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;
                    assert (br.OverflowSlackSpace >= 0);
                    maxOverflowSpace = std::max<int>(maxOverflowSpace, br.OverflowSlackSpace);

                    for (const auto input : make_iterator_range(out_edges(id, mBufferGraph))) {
                        BufferPort & br = mBufferGraph[input];
                        assert (maxOverflowSpace >= br.OverflowSlackSpace);
                        br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;
                        assert (br.OverflowSlackSpace >= 0);
                        maxOverflowSpace = std::max<int>(maxOverflowSpace, br.OverflowSlackSpace);
                    }

                    bn.RequiredOverflowSpace = maxOverflowSpace;

                    if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = child(id, InOutStreamSetReplacement);

                }

            }



        } else { // if (bn.isUnowned()) {

            bn.RequiredOverflowSpace = 0;
            const auto output = in_edge(streamSet, mBufferGraph);
            BufferPort & br = mBufferGraph[output];
            br.OverflowSlackSpace = 0;
            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                BufferPort & br = mBufferGraph[input];
                br.OverflowSlackSpace = 0;
            }

        }
    }

}

}



