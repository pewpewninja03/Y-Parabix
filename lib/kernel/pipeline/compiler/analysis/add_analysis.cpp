#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

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

            minAddK = noPrincipal ? minAddK : principalK;

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                BufferPort & br = mBufferGraph[e];
                if (br.isFixed()) {
                    assert (minAddK < std::numeric_limits<int>::max());
                    const auto k = inputAdd[br.Port.Number] - minAddK;
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
        if (LLVM_LIKELY(bn.isOwned())) {

            int maxOverflowSpace = 0;

            if (LLVM_LIKELY(in_degree(streamSet, InOutStreamSetReplacement) == 0)) {

                for (auto id = streamSet;;) {

                    const BufferNode & sn = mBufferGraph[id];
                    int required = sn.RequiredOverflowSpace;

                    for (const auto input : make_iterator_range(out_edges(id, mBufferGraph))) {
                        const BufferPort & br = mBufferGraph[input];
                        const auto a = std::max<int>(br.OverflowSlackSpace, br.EmptyOverflow);
                        const auto b = std::max<int>(a, br.LookAhead);
                        required = std::max<int>(required, b);
                    }

                    const auto output = in_edge(id, mBufferGraph);
                    BufferPort & br = mBufferGraph[output];
                    const auto c = std::max<int>(br.OverflowSlackSpace, br.EmptyOverflow);
                    required = std::max<int>(required, c);

                    const auto ta = transitiveAdd[id - FirstStreamSet];
                    maxOverflowSpace = std::max<unsigned>(maxOverflowSpace, ta + required);

                    if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = child(id, InOutStreamSetReplacement);
                }

            } else {

                for (auto id = streamSet;;) {
                    id = parent(id, InOutStreamSetReplacement);
                    if (LLVM_LIKELY(in_degree(id, InOutStreamSetReplacement) == 0)) {
                        const BufferNode & sn = mBufferGraph[id];
                        maxOverflowSpace = sn.RequiredOverflowSpace;
                        break;
                    }
                }

            }

            const auto output = in_edge(streamSet, mBufferGraph);
            BufferPort & br = mBufferGraph[output];
            assert (maxOverflowSpace >= br.OverflowSlackSpace);
            br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;
            assert (br.OverflowSlackSpace >= 0);
            maxOverflowSpace = std::max<int>(maxOverflowSpace, br.OverflowSlackSpace);

            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                BufferPort & br = mBufferGraph[input];
                assert (maxOverflowSpace >= br.OverflowSlackSpace);
                br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;
                assert (br.OverflowSlackSpace >= 0);
                maxOverflowSpace = std::max<int>(maxOverflowSpace, br.OverflowSlackSpace);
            }

            bn.RequiredOverflowSpace = maxOverflowSpace;

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


#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

    SmallVector<int, 64> transitiveAdd(LastStreamSet - FirstStreamSet + 1, 0);


    size_t lastPartId = -1U;

    for (auto kernel = PipelineInput; kernel <= PipelineOutput; ++kernel) {

        int minAddK = 0;

        if (LLVM_LIKELY(in_degree(kernel, mBufferGraph) != 0)) {
            minAddK = std::numeric_limits<int>::max();
            int principalAdd = 0;
            bool hasPrincipal = false;
            for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                BufferPort & br = mBufferGraph[e];
                assert (!br.isPrincipal() || br.isFixed());
                if (br.isFixed()) {
                    const auto streamSet = source(e, mBufferGraph);
                    const int k = transitiveAdd[streamSet - FirstStreamSet] + br.Add - br.Truncate;
                    minAddK = std::min(minAddK, k);
                    if (LLVM_UNLIKELY(br.isPrincipal())) {
                        principalAdd = k;
                        hasPrincipal = true;
                    } else {
                        br.TransitiveAdd = k;
                    }
                }
            }

            if (minAddK == std::numeric_limits<int>::max()) {
                minAddK = 0;
            } else {
                assert (minAddK >= 0);
                assert (!hasPrincipal || minAddK <= principalAdd);
                if (hasPrincipal) {
                    minAddK = principalAdd - minAddK;
                    assert (minAddK >= 0);
                }


                for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    if (br.isFixed()) {
                        if (LLVM_UNLIKELY(br.isPrincipal())) {
                            br.TransitiveAdd = 0;
                        } else {
                            br.TransitiveAdd += minAddK;
                        }
                    }
                }
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            BufferPort & br = mBufferGraph[e];
            if (br.isFixed()) {
                const int k = br.Add - br.Truncate + minAddK;
                const auto streamSet = target(e, mBufferGraph);
                transitiveAdd[streamSet - FirstStreamSet] = k;
            }
        }

    }


    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        auto maxAddK = std::numeric_limits<int>::min();
        auto minPrincipalAddK = std::numeric_limits<int>::max();
        bool hasPrincipal = false;
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[input];
            assert (br.isFixed() || br.TransitiveAdd == 0);
            if (br.isFixed()) {
                maxAddK = std::max(maxAddK, br.TransitiveAdd);
                if (LLVM_UNLIKELY(br.isPrincipal())) {
                    minPrincipalAddK = std::min(minPrincipalAddK, br.TransitiveAdd);
                    hasPrincipal = true;
                }
            }
        }

        if (maxAddK != std::numeric_limits<int>::min()) {
            maxAddK = hasPrincipal ? minPrincipalAddK : maxAddK;
            const auto output = in_edge(streamSet, mBufferGraph);
            BufferPort & br = mBufferGraph[output];
            br.TransitiveAdd += maxAddK;
            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                BufferPort & br = mBufferGraph[input];
                if (br.isFixed()) {
                    br.TransitiveAdd -= maxAddK;
                }
            }
        }

    }

    // Scan through the I/O for each kernel to see if some transitive
    // add relationship imposes an overflow requirement on a buffer


    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        int maxK = 0;
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
    }

}

#endif

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

    SmallVector<int, 64> transitiveAdd(LastStreamSet - FirstStreamSet + 1);

    for (auto i = PipelineInput; i <= PipelineOutput; ++i) {
        int minAddK = 0;
        if (LLVM_LIKELY(in_degree(i, mBufferGraph) > 0)) {
            bool noPrincipal = true;
            minAddK = std::numeric_limits<int>::max();
            int principalAddK = 0;
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                int k = transitiveAdd[streamSet - FirstStreamSet];
                BufferPort & br = mBufferGraph[e];
                assert (br.TransitiveAdd == 0);
                assert (!br.isPrincipal() || br.isFixed());
                if (br.isFixed()) {
                    k += br.Add;
                    k -= br.Truncate;
                    minAddK = std::min(minAddK, k);
                    if (LLVM_UNLIKELY(br.isPrincipal())) {
                        principalAddK = k;
                        noPrincipal = false;
                    }
                    br.TransitiveAdd = k;
                }
            }

            if (LLVM_LIKELY(noPrincipal)) {
                for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    br.TransitiveAdd -= minAddK;
                }
            } else {
                for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    br.TransitiveAdd = 0; // principalAddK - br.TransitiveAdd;
                }
                minAddK = principalAddK;
            }
        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            BufferPort & br = mBufferGraph[e];
            auto k = minAddK;
            k += br.Add;
            k -= br.Truncate;
            assert (br.TransitiveAdd == 0);
            br.TransitiveAdd = k;
            const auto streamSet = target(e, mBufferGraph);
            transitiveAdd[streamSet - FirstStreamSet] = k;
        }
    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        int maxAddK = 0;

        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[input];
//            assert (br.isFixed() || br.TransitiveAdd == 0);
            maxAddK = std::max<int>(maxAddK, br.TransitiveAdd);
        }

        const auto output = in_edge(streamSet, mBufferGraph);
        BufferPort & br = mBufferGraph[output];
        br.TransitiveAdd += maxAddK;
        if (br.TransitiveAdd > 0) {
            br.Add = br.TransitiveAdd;
        } else if (br.TransitiveAdd < 0) {
            br.Truncate = -br.TransitiveAdd;
        }
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            BufferPort & br = mBufferGraph[input];
            br.TransitiveAdd -= maxAddK;
//            if (br.TransitiveAdd > 0) {
//                br.Add = br.TransitiveAdd;
//            } else if (br.TransitiveAdd < 0) {
//                br.Truncate = -br.TransitiveAdd;
//            }
        }

    }

    // Scan through the I/O for each kernel to see if some transitive
    // add relationship imposes an overflow requirement on a buffer


    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        int maxK = 0;
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
    }



}

#endif

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

    SmallVector<int, 64> transitiveAdd(LastStreamSet - FirstStreamSet + 1);

    for (size_t p = 1; p < PartitionCount; ++p) {
        const auto firstKernel = FirstKernelInPartition[p - 1];
        const auto lastKernel = FirstKernelInPartition[p];
        for (auto i = firstKernel; i != lastKernel; ++i) {


            int minAddK = 0;

            if (LLVM_LIKELY(i != firstKernel && in_degree(i, mBufferGraph) > 0)) {
                minAddK = std::numeric_limits<int>::max();
                for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    assert (!br.isPrincipal() || br.isFixed());
                    if (br.isFixed()) {
                        const auto streamSet = source(e, mBufferGraph);
                        const int k = transitiveAdd[streamSet - FirstStreamSet] + br.Add;
                        br.OverflowSlackSpace = k;
                        if (LLVM_UNLIKELY(br.isPrincipal())) {
                            minAddK = k;
                            break;
                        }
                        minAddK = std::min(minAddK, k);
                    }
                }
                if (minAddK == std::numeric_limits<int>::max()) {
                    minAddK = 0;
                }
            }


            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                BufferPort & br = mBufferGraph[e];
                if (br.isFixed()) {
                    const auto streamSet = target(e, mBufferGraph);
                    const int k = minAddK + br.Add;
                    br.OverflowSlackSpace = k;
                    transitiveAdd[streamSet - FirstStreamSet] = k;
                }
            }


        }
    }

//    for (size_t i = PipelineInput; i < PipelineOutput; ++i) {

//        int minAddK = 0;

//        if (LLVM_LIKELY(in_degree(i, mBufferGraph) > 0)) {
//            minAddK = std::numeric_limits<int>::max();
//            int principalAddK = std::numeric_limits<int>::lowest();
//            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
//                BufferPort & br = mBufferGraph[e];
//                assert (!br.isPrincipal() || br.isFixed());
//                if (br.isFixed()) {
//                    const auto streamSet = source(e, mBufferGraph);
//                    const int k = transitiveAdd[streamSet - FirstStreamSet] + br.Add;
//                    br.OverflowSlackSpace = k;
//                    if (LLVM_UNLIKELY(br.isPrincipal())) {
//                        principalAddK = k;
//                    }
//                    minAddK = std::min(minAddK, k);
//                }
//            }
//            if (principalAddK != std::numeric_limits<int>::lowest()) {
//                minAddK = principalAddK;
//            } else if (minAddK == std::numeric_limits<int>::max()) {
//                minAddK = 0;
//            }
//        }


//        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
//            BufferPort & br = mBufferGraph[e];
//            if (br.isFixed()) {
//                const auto streamSet = target(e, mBufferGraph);
//                const int k = minAddK + br.Add;
//                br.OverflowSlackSpace = k;
//                transitiveAdd[streamSet - FirstStreamSet] = k;
//            }
//        }

//    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {

        BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_LIKELY(bn.isOwned())) {

            int maxOverflowSpace = 0;

            if (LLVM_LIKELY(in_degree(streamSet, InOutStreamSetReplacement) == 0)) {

                auto id = streamSet;
                for (;;) {

                    for (const auto input : make_iterator_range(out_edges(id, mBufferGraph))) {
                        const BufferPort & br = mBufferGraph[input];
                        const auto a = std::max<int>(br.Add, br.EmptyOverflow);
                        const auto b = std::max<int>(a, br.LookAhead);
                        maxOverflowSpace = std::max<int>(maxOverflowSpace, b);
                    }

                    const auto output = in_edge(id, mBufferGraph);
                    BufferPort & br = mBufferGraph[output];
                    const auto c = std::max<int>(br.Add, br.EmptyOverflow);
                    maxOverflowSpace = std::max<int>(maxOverflowSpace, c);

                    const BufferNode & sn = mBufferGraph[id];
                    const auto ta = transitiveAdd[id - FirstStreamSet];
                    maxOverflowSpace = std::max<unsigned>(sn.RequiredOverflowSpace, ta + maxOverflowSpace);

                    if (LLVM_LIKELY(out_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = child(id, InOutStreamSetReplacement);
                }

            } else {

                auto id = streamSet;
                for (;;) {
                    if (LLVM_LIKELY(in_degree(id, InOutStreamSetReplacement) == 0)) {
                        break;
                    }
                    id = parent(id, InOutStreamSetReplacement);
                }

                const BufferNode & sn = mBufferGraph[id];
                maxOverflowSpace = sn.RequiredOverflowSpace;

            }

            bn.RequiredOverflowSpace = maxOverflowSpace;

            const auto output = in_edge(streamSet, mBufferGraph);
            BufferPort & br = mBufferGraph[output];
            assert (br.OverflowSlackSpace <= maxOverflowSpace);
            br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;

            for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                BufferPort & br = mBufferGraph[input];
                assert (br.OverflowSlackSpace <= maxOverflowSpace);
                br.OverflowSlackSpace = maxOverflowSpace - br.OverflowSlackSpace;
            }

        } else { // if (bn.isUnowned()) {
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

#endif


#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeAddGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::annotateBufferGraphWithAddAttributes() {

    SmallVector<int, 64> transitiveAdd(LastStreamSet - FirstStreamSet + 1);

    for (auto i = PipelineInput; i <= PipelineOutput; ++i) {
        int minAddK = 0;
        if (LLVM_LIKELY(in_degree(i, mBufferGraph) > 0)) {
            bool noPrincipal = true;
            minAddK = std::numeric_limits<int>::max();
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                int k = transitiveAdd[streamSet - FirstStreamSet];
                BufferPort & br = mBufferGraph[e];
                k += br.Add;
                minAddK = std::min(minAddK, k);
                if (LLVM_UNLIKELY(br.isPrincipal())) {
                    minAddK = k;
                    noPrincipal = false;
                    break;
                }
                br.TransitiveAdd = k;
            }

            if (LLVM_LIKELY(noPrincipal)) {
                for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    br.TransitiveAdd -= minAddK;
                }
            } else {
                for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                    BufferPort & br = mBufferGraph[e];
                    br.TransitiveAdd = 0;
                }
            }
        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            BufferPort & br = mBufferGraph[e];
            auto k = minAddK;
            k += br.Add;
            br.TransitiveAdd = k;
            const auto streamSet = target(e, mBufferGraph);
            transitiveAdd[streamSet - FirstStreamSet] = k;
        }
    }

    // Scan through the I/O for each kernel to see if some transitive
    // add relationship imposes an overflow requirement on a buffer


    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        int maxK = 0;
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & rate = mBufferGraph[e];
            maxK = std::max(maxK, rate.TransitiveAdd);
        }
        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            BufferNode & bn = mBufferGraph[streamSet];
            bn.MaxAdd = std::max<unsigned>(bn.MaxAdd, maxK);
        }
    }



}

#endif

}



