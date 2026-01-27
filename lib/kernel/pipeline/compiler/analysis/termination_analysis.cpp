#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyTerminationChecks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyTerminationChecks() {

    using TerminationGraph = adjacency_list<hash_setS, vecS, bidirectionalS>;

    const auto numOfPhases = PartitionPhaseBoundaries.size(); assert (numOfPhases > 1);

    size_t requiredChecks = 0;
    for (size_t i = 1U; i < numOfPhases; ++i) {
        const auto firstPartition = PartitionPhaseBoundaries[i - 1];
        const auto oneAfterLastPartition = PartitionPhaseBoundaries[i];
        requiredChecks = std::max(requiredChecks, oneAfterLastPartition - firstPartition);
    }

    TerminationGraph G(requiredChecks + 1);

    ReverseTopologicalOrdering ordering;
    ordering.reserve(requiredChecks + 1);

    mTerminationCheck.resize(PartitionCount, 0U);

    for (size_t i = 1U; i < numOfPhases; ++i) {
        for (size_t j = 0; j <= requiredChecks; ++j) {
            clear_vertex(j, G);
        }

        const auto firstPartition = PartitionPhaseBoundaries[i - 1];
        const auto oneAfterLastPartition = PartitionPhaseBoundaries[i];
        assert (firstPartition < oneAfterLastPartition);
        const auto firstKernelInCurrentPhase = FirstKernelInPartition[firstPartition];
        const auto oneAfterLastKernelInCurrentPhase = FirstKernelInPartition[oneAfterLastPartition];\
        assert (firstKernelInCurrentPhase < oneAfterLastKernelInCurrentPhase);
        assert (oneAfterLastKernelInCurrentPhase <= PipelineOutput);
        const auto terminal = oneAfterLastPartition - firstPartition;

        assert (in_degree(terminal, G) == 0);
        for (auto kernel = firstKernelInCurrentPhase; kernel < oneAfterLastKernelInCurrentPhase; ++kernel) {
            const auto pid = KernelPartitionId[kernel];
            assert (firstPartition <= pid);
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal() || bn.isConstant()) continue;
                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    const auto cid = KernelPartitionId[consumer];
                    assert (firstPartition <= cid);
                    if (cid != pid) {
                        const auto id = std::min<size_t>(cid, oneAfterLastPartition);
                        add_edge(pid - firstPartition, id - firstPartition, G);
                    }
                }
            }
        }

        for (auto kernel = firstKernelInCurrentPhase; kernel < oneAfterLastKernelInCurrentPhase; ++kernel) {
            const Kernel * const kernelObj = getKernel(kernel); assert (kernelObj);
            if (LLVM_UNLIKELY(kernelObj->hasAttribute(AttrId::SideEffecting))) {
                const auto pid = KernelPartitionId[kernel];
                add_edge(pid, terminal, G);
            }

        }
        assert (in_degree(terminal, G) > 0);

        ordering.clear();
        topological_sort(G, std::back_inserter(ordering));
        transitive_closure_dag(ordering, G);
        transitive_reduction_dag(ordering, G);

        // we are only interested in the incoming edges of the terminal node
        for (const auto e : make_iterator_range(in_edges(terminal, G))) {
            const auto partId = source(e, G);
            assert (partId < PartitionCount);
            mTerminationCheck[partId] = TerminationCheckFlag::Soft;
        }

    }

    // hard terminations
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(getKernel(i)->hasAttribute(AttrId::MayFatallyTerminate))) {
            mTerminationCheck[KernelPartitionId[i]] |= TerminationCheckFlag::Hard;
        }
    }

}

#if 0


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyTerminationChecks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyTerminationChecks() {

    using TerminationGraph = adjacency_list<hash_setS, vecS, bidirectionalS>;

    TerminationGraph G(PartitionCount);

    const auto terminal = KernelPartitionId[PipelineOutput];

    const auto numOfPhases = PartitionPhaseBoundaries.size(); assert (numOfPhases > 1);      ;

    unsigned largestPhaseSize = 0;
    for (size_t i = 1U; i < numOfPhases; ++i) {
        const auto firstPartition = PartitionPhaseBoundaries[i - 1];
        const auto oneAfterLastPartition = PartitionPhaseBoundaries[i];
        const auto firstKernelInCurrentPhase = FirstKernelInPartition[firstPartition];
        const auto oneAfterLastKernelInCurrentPhase = FirstKernelInPartition[oneAfterLastPartition];
        const auto m = oneAfterLastKernelInCurrentPhase - firstKernelInCurrentPhase;
        largestPhaseSize = std::max(largestPhaseSize, m);
    }


    for (auto consumer = FirstKernel; consumer <= PipelineOutput; ++consumer) {
        const auto cid = KernelPartitionId[consumer];

        for (const auto e : make_iterator_range(in_edges(consumer, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const auto producer = parent(streamSet, mBufferGraph);
            const auto pid = KernelPartitionId[producer];
            assert (pid <= cid);
            if (pid != cid) {
                add_edge(pid, cid, G);
            }
        }
    }

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
        if (LLVM_UNLIKELY(kernelObj->hasAttribute(AttrId::SideEffecting))) {
            const auto pid = KernelPartitionId[i];
            add_edge(pid, terminal, G);
        }
    }

    assert (FirstCall == (PipelineOutput + 1U));

    for (auto consumer = PipelineOutput; consumer <= LastCall; ++consumer) {
        for (const auto relationship : make_iterator_range(in_edges(consumer, mScalarGraph))) {
            const auto scalar = source(relationship, mScalarGraph);
            for (const auto production : make_iterator_range(in_edges(scalar, mScalarGraph))) {
                const auto producer = source(production, mScalarGraph);
                const auto partitionId = KernelPartitionId[producer];
                assert ("cannot occur" && partitionId != terminal);
                add_edge(partitionId, terminal, G);
            }
        }
    }

    ReverseTopologicalOrdering ordering;
    ordering.reserve(num_vertices(G));
    topological_sort(G, std::back_inserter(ordering));
    transitive_closure_dag(ordering, G);

    if (1 < FirstComputePartitionId || LastComputePartitionId < (PartitionCount - 1)) {

        #define ON_COMPUTE(id) ((FirstComputePartitionId <= id) && (id <= LastComputePartitionId))

        #define ON_DIFFERENT_THREADS(a, b) (ON_COMPUTE(a) ^ ON_COMPUTE(b))

        remove_edge_if([&](TerminationGraph::edge_descriptor e){
            const auto a = source(e, G);
            const auto b = target(e, G);
            assert (a < b);
            return ON_DIFFERENT_THREADS(a, b) && (b != (PartitionCount - 1));
        }, G);

    }

    transitive_reduction_dag(ordering, G);

    mTerminationCheck.resize(PartitionCount, 0U);

    // we are only interested in the incoming edges of the pipeline output
    for (const auto e : make_iterator_range(in_edges(terminal, G))) {
        const auto partId = source(e, G);
        mTerminationCheck[partId] = TerminationCheckFlag::Soft;
    }

    // hard terminations
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(getKernel(i)->hasAttribute(AttrId::MayFatallyTerminate))) {
            mTerminationCheck[KernelPartitionId[i]] |= TerminationCheckFlag::Hard;
        }
    }

    mTerminationCheck[terminal] = 0;

}

#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeTerminationPropagationGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeTerminationPropagationGraph() {

    // When a partition terminates, we want to inform any kernels that supply information to it that
    // one of their consumers has finished processing data. In a pipeline with a single output, this
    // isn't necessary but if a pipeline has multiple outputs, we could end up needlessly producing
    // data that will never be consumed.

    mTerminationPropagationGraph = TerminationPropagationGraph(LastKernel + 1U);

    HasTerminationSignal.resize(PipelineOutput + 1U);

    BitVector marks(PipelineOutput + 1U);

    const auto numOfPhases = PartitionPhaseBoundaries.size(); assert (numOfPhases > 1);

    for (size_t i = 1U; i < numOfPhases; ++i) {
        const auto firstPartition = PartitionPhaseBoundaries[i - 1];
        const auto oneAfterLastPartition = PartitionPhaseBoundaries[i];
        const auto firstKernelInCurrentPhase = FirstKernelInPartition[firstPartition];
        for (auto pid = firstPartition; pid < oneAfterLastPartition; ++pid) {
            const auto start = FirstKernelInPartition[pid];
            const auto end = FirstKernelInPartition[pid + 1U];
            assert (start <= end);
            assert (end <= PipelineOutput);

            // regardless of whether a partition root can terminate, every root has
            // a terminated flag stored in the state that any kernel that cannot
            // explicitly terminate shares.
            HasTerminationSignal.set(start);

            for (const auto e : make_iterator_range(in_edges(start, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (LLVM_UNLIKELY(bn.isConstant())) continue;
                const auto producer = parent(streamSet, mBufferGraph);
                if (producer < firstKernelInCurrentPhase) continue;
                marks.set(KernelPartitionId[producer]);
            }

            const auto term = getKernel(start)->canSetTerminateSignal();
            for (const auto j : marks.set_bits()) {
                const auto k = FirstKernelInPartition[j];
                assert (k < start);
                add_edge(start, k, term, mTerminationPropagationGraph);
            }
            marks.reset();

            for (auto i = start + 1U; i < end; ++i) {
                const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
                if (kernelObj->canSetTerminateSignal()) {
                    add_edge(i, start, true, mTerminationPropagationGraph);
                    HasTerminationSignal.set(i);
                }
            }
        }
    }


    ReverseTopologicalOrdering ordering;
    ordering.reserve(num_vertices(mTerminationPropagationGraph));
    topological_sort(mTerminationPropagationGraph, std::back_inserter(ordering));

    auto first = ordering.begin();
    const auto end = ordering.end();

    for (auto i = first; i != end; ++i) {
        TerminationPropagationGraph::in_edge_iterator ei_begin, ei_end;
        std::tie(ei_begin, ei_end) = in_edges(*i, mTerminationPropagationGraph);
        bool anyPropagate = false;
        bool allPropagate = true;
        for (auto ei = ei_begin; ei != ei_end; ++ei) {
            const auto p = mTerminationPropagationGraph[*ei];
            anyPropagate |= p;
            allPropagate &= p;
        }
        if (anyPropagate) {
            const Kernel * const kernelObj = getKernel(*i); assert (kernelObj);
            for (const auto & attr : kernelObj->getAttributes()) {
                switch (attr.getKind()) {
                    case AttrId::MustExplicitlyTerminate:
                    case AttrId::SideEffecting:
                        goto disable_propagated_signals;
                    default:
                        break;
                }
            }
            if (allPropagate) {
                for (auto e : make_iterator_range(out_edges(*i, mTerminationPropagationGraph))) {
                    mTerminationPropagationGraph[e] = true;
                }
            } else {
disable_propagated_signals:
                for (auto ei = ei_begin; ei != ei_end; ++ei) {
                    mTerminationPropagationGraph[*ei] = false;
                }
            }
        }
    }

    remove_edge_if([&](const TerminationPropagationGraph::edge_descriptor e) {
        return !mTerminationPropagationGraph[e];
    }, mTerminationPropagationGraph);

    transitive_closure_dag(ordering, mTerminationPropagationGraph);
    transitive_reduction_dag(ordering, mTerminationPropagationGraph);

}

#if 0


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeTerminationPropagationGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeTerminationPropagationGraph() {

    // When a partition terminates, we want to inform any kernels that supply information to it that
    // one of their consumers has finished processing data. In a pipeline with a single output, this
    // isn't necessary but if a pipeline has multiple outputs, we could end up needlessly producing
    // data that will never be consumed.

    mTerminationPropagationGraph = TerminationPropagationGraph(LastKernel + 1U);

    HasTerminationSignal.resize(PipelineOutput + 1U);

    const auto firstComputeKernelId = FirstKernelInPartition[FirstComputePartitionId];
//    const auto afterLastComputeKernelId = FirstKernelInPartition[LastComputePartitionId + 1];

    for (auto kernel = FirstKernel; kernel < firstComputeKernelId; ++kernel) {
        HasTerminationSignal.set(kernel);
    }

    BitVector marks(PipelineOutput + 1U);

    for (auto pid = KernelPartitionId[FirstKernel]; pid < PartitionCount; ++pid) {
        const auto start = FirstKernelInPartition[pid];
        const auto end = FirstKernelInPartition[pid + 1U];
        assert (start <= end);
        assert (end <= PipelineOutput);

        for (const auto e : make_iterator_range(in_edges(start, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const auto producer = parent(streamSet, mBufferGraph);
            marks.set(KernelPartitionId[producer]);
        }

        const auto term = getKernel(start)->canSetTerminateSignal();
        for (const auto j : marks.set_bits()) {
            const auto k = FirstKernelInPartition[j];
            assert (k < start);
            add_edge(start, k, term, mTerminationPropagationGraph);
        }
        marks.reset();

        // regardless of whether a partition root can terminate, every root has
        // a terminated flag stored in the state that any kernel that cannot
        // explicitly terminate shares.
        HasTerminationSignal.set(start);
        for (auto i = start + 1U; i < end; ++i) {
            const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
            if (kernelObj->canSetTerminateSignal()) {
                add_edge(i, start, true, mTerminationPropagationGraph);
                HasTerminationSignal.set(i);
            } else {
                // If we have a cross threaded buffer, we cannot rely on only storing the root's
                // termination signal because we need to know exactly when the actual producer
                // is terminated. Threads executing the same code only need to know when the
                // partition is terminated.
                for (const auto e : make_iterator_range(out_edges(i , mBufferGraph))) {
                    const BufferNode & bn = mBufferGraph[target(e, mBufferGraph)];
                    if (LLVM_UNLIKELY(bn.isCrossThreaded())) {
                        HasTerminationSignal.set(i);
                        break;
                    }
                }
            }
        }
    }



    ReverseTopologicalOrdering ordering;
    ordering.reserve(num_vertices(mTerminationPropagationGraph));
    topological_sort(mTerminationPropagationGraph, std::back_inserter(ordering));

    auto first = ordering.begin();
    const auto end = ordering.end();

    for (auto i = first; i != end; ++i) {
        TerminationPropagationGraph::in_edge_iterator ei_begin, ei_end;
        std::tie(ei_begin, ei_end) = in_edges(*i, mTerminationPropagationGraph);
        bool anyPropagate = false;
        bool allPropagate = true;
        for (auto ei = ei_begin; ei != ei_end; ++ei) {
            const auto p = mTerminationPropagationGraph[*ei];
            anyPropagate |= p;
            allPropagate &= p;
        }
        if (anyPropagate) {
            const Kernel * const kernelObj = getKernel(*i); assert (kernelObj);
            for (const auto & attr : kernelObj->getAttributes()) {
                switch (attr.getKind()) {
                    case AttrId::MustExplicitlyTerminate:
                    case AttrId::SideEffecting:
                        goto disable_propagated_signals;
                    default:
                        break;
                }
            }
            if (allPropagate) {
                for (auto e : make_iterator_range(out_edges(*i, mTerminationPropagationGraph))) {
                    mTerminationPropagationGraph[e] = true;
                }
            } else {
disable_propagated_signals:
                for (auto ei = ei_begin; ei != ei_end; ++ei) {
                    mTerminationPropagationGraph[*ei] = false;
                }
            }
        }
    }

    remove_edge_if([&](const TerminationPropagationGraph::edge_descriptor e) {
        return !mTerminationPropagationGraph[e];
    }, mTerminationPropagationGraph);

    transitive_closure_dag(ordering, mTerminationPropagationGraph);
    transitive_reduction_dag(ordering, mTerminationPropagationGraph);

}

#endif


}
