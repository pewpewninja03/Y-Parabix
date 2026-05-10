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

            if (LLVM_UNLIKELY(inputAdd.size() < d)) {
                inputAdd.resize(d);
            }

            bool hasPrincipal = false;
            bool noFixed = true;
            minAddK = std::numeric_limits<int>::max();
            int principalK = 0;

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                BufferPort & br = mBufferGraph[e];
                int k = 0;
                if (br.isFixed()) {
                    noFixed = false;
                    k = transitiveAdd[streamSet - FirstStreamSet] + br.Add;
                    minAddK = std::min(minAddK, k);
                    if (LLVM_UNLIKELY(br.isPrincipal())) {
                        principalK = k;
                        hasPrincipal = true;
                        break;
                    }
                }
                inputAdd[br.Port.Number] = k;
            }

            if (noFixed) {
                minAddK = 0;
            } else if (hasPrincipal) {
                minAddK = principalK;
            }

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                BufferPort & br = mBufferGraph[e];
                if (br.isFixed()) {
                    assert (minAddK < std::numeric_limits<int>::max());
                    const int k = inputAdd[br.Port.Number] - minAddK;
                    br.RequiredOverflowSpace = k;
                }
            }

        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            BufferPort & br = mBufferGraph[e];
            const auto k = minAddK + br.Add;
            const auto streamSet = target(e, mBufferGraph);
            transitiveAdd[streamSet - FirstStreamSet] = k;
            br.RequiredOverflowSpace = k;
        }
    }

    // Scan through the I/O for each kernel to see if some transitive
    // add relationship imposes an overflow requirement on a buffer

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        BufferNode & bn = mBufferGraph[streamSet];

        int maxOverflowSpace = 0;

        if (LLVM_UNLIKELY(bn.isUnowned() || bn.isConstant())) {

            maxOverflowSpace = 0;

        } else if (LLVM_LIKELY(in_degree(streamSet, InOutStreamSetReplacement) == 0)) {

            for (auto id = streamSet;;) {

                const auto output = in_edge(id, mBufferGraph);
                BufferPort & br = mBufferGraph[output];
                const auto c = std::max<int>(br.RequiredOverflowSpace, br.EmptyOverflow);
                maxOverflowSpace = std::max<int>(maxOverflowSpace, c);

                for (const auto input : make_iterator_range(out_edges(id, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[input];
                    const auto a = std::max<int>(br.RequiredOverflowSpace, br.EmptyOverflow);
                    const auto b = std::max<int>(a, br.LookAhead);
                    br.RequiredOverflowSpace = b;
                    maxOverflowSpace = std::max<int>(maxOverflowSpace, b);
                }
                if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                    break;
                }
                id = child(id, InOutStreamSetReplacement);
            }

        }

        const auto output = in_edge(streamSet, mBufferGraph);
        BufferPort & br = mBufferGraph[output];
        br.RequiredOverflowSpace = maxOverflowSpace;

    }

}

}



