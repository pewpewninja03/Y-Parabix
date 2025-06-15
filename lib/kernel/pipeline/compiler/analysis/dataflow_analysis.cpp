#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <toolchain/toolchain.h>
#include <z3.h>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

// #define PRINT_INTRA_PARTITION_VECTOR_GRAPH

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeIntraPartitionRepetitionVectors
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::computeIntraPartitionRepetitionVectors(PartitionGraph & P) {

    #ifdef PRINT_INTRA_PARTITION_VECTOR_GRAPH
    using Graph = adjacency_list<vecS, vecS, bidirectionalS, no_property, Rational>;
    #endif

    const auto numOfPartitions = num_vertices(P);

    const auto cfg = Z3_mk_config();
    Z3_set_param_value(cfg, "model", "true");
    Z3_set_param_value(cfg, "proof", "false");

    const auto ctx = Z3_mk_context(cfg);
    Z3_del_config(cfg);
    const auto solver = Z3_mk_solver(ctx);
    Z3_solver_inc_ref(ctx, solver);

    auto hard_assert = [&](Z3_ast c) {
        Z3_solver_assert(ctx, solver, c);
    };

    const auto intType = Z3_mk_int_sort(ctx);

    auto constant_real = [&](const Rational value) {
        assert (value.numerator() > 0);
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto multiply =[&](Z3_ast X, Z3_ast Y) {
        assert (X && Y);
        Z3_ast args[2] = { X, Y };
        return Z3_mk_mul(ctx, 2, args);
    };

    const auto ONE = Z3_mk_int(ctx, 1, intType);
    const auto m = num_vertices(Relationships);
    std::vector<Z3_ast> VarList(m);

    #ifdef PRINT_INTRA_PARTITION_VECTOR_GRAPH
    unsigned totalNumOfKernels = 0;
    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;
        totalNumOfKernels += K.size();
    }

    Graph G(totalNumOfKernels);
    flat_map<unsigned, unsigned> M;
    M.reserve(totalNumOfKernels);

    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;
        for (const auto u : K) {
            M.emplace(u, M.size());
        }
    }
    #endif

    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;

        for (const auto u : K) {
            auto repVar = Z3_mk_fresh_const(ctx, nullptr, intType);
            hard_assert(Z3_mk_ge(ctx, repVar, ONE));
            VarList[u] = repVar; // multiply(rootVar, repVar);
        }

        for (const auto producer : K) {
            assert (Relationships[producer].Type == RelationshipNode::IsKernel);
            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto binding = target(e, Relationships);
                if (Relationships[binding].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(binding, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    if (Relationships[streamSet].Type == RelationshipNode::IsStreamSet) {
                        const RelationshipNode & output = Relationships[binding];
                        assert (output.Type == RelationshipNode::IsBinding);

                        const Binding & outputBinding = output.Binding;
                        const ProcessingRate & outputRate = outputBinding.getRate();
                        // ignore unknown output rates; we cannot reason about them here.
                        if (LLVM_LIKELY(outputRate.isFixed())) {

                            assert (VarList[producer]);
                            Z3_ast expOutRate = nullptr;

                            for (const auto e : make_iterator_range(out_edges(streamSet, Relationships))) {
                                const auto binding = target(e, Relationships);
                                const RelationshipNode & input = Relationships[binding];
                                if (LLVM_LIKELY(input.Type == RelationshipNode::IsBinding)) {

                                    const Binding & inputBinding = input.Binding;
                                    const ProcessingRate & inputRate = inputBinding.getRate();

                                    if (LLVM_LIKELY(inputRate.isFixed())) {

                                        const auto f = first_out_edge(binding, Relationships);
                                        assert (Relationships[f].Reason != ReasonType::Reference);
                                        const unsigned consumer = target(f, Relationships);

                                        const auto c = PartitionIds.find(consumer);
                                        assert (c != PartitionIds.end());
                                        const auto consumerPartitionId = c->second;
                                        assert (producerPartitionId <= consumerPartitionId);

                                        if (producerPartitionId == consumerPartitionId) {

                                            if (expOutRate == nullptr) {
                                                const RelationshipNode & producerNode = Relationships[producer];
                                                assert (producerNode.Type == RelationshipNode::IsKernel);
                                                const auto expectedOutput = outputRate.getRate() * producerNode.Kernel->getStride();
                                                expOutRate = multiply(VarList[producer], constant_real(expectedOutput));
                                            }

                                            const RelationshipNode & consumerNode = Relationships[consumer];
                                            assert (consumerNode.Type == RelationshipNode::IsKernel);
                                            const auto expectedInput = inputRate.getRate() * consumerNode.Kernel->getStride();
                                            const Z3_ast expInRate = multiply(VarList[consumer], constant_real(expectedInput));

                                            #ifdef PRINT_INTRA_PARTITION_VECTOR_GRAPH
                                            const RelationshipNode & producerNode = Relationships[producer];
                                            assert (producerNode.Type == RelationshipNode::IsKernel);
                                            const auto expectedOutput = outputRate.getRate() * producerNode.Kernel->getStride();
                                            const auto a = M.find(producer)->second;
                                            const auto b = M.find(consumer)->second;
                                            add_edge(a, b, expectedOutput / expectedInput, G);
                                            #endif

                                            hard_assert(Z3_mk_eq(ctx, expOutRate, expInRate));
                                        }

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

    }


    if (LLVM_UNLIKELY(Z3_solver_check(ctx, solver) == Z3_L_FALSE)) {
        report_fatal_error("Z3 failed to find a solution to minimum expected dataflow problem");
    }

    const auto model = Z3_solver_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);

    // TODO: if the root of a partition has a "large" rep vector count, split the partition?

    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;

        const auto n = K.size();
        N.Repetitions.resize(n);
        assert (n > 0);

        for (unsigned i = 0; i < n; ++i) {
            Z3_ast var = VarList[K[i]]; assert (var);
            Z3_ast value;
            if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, var, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
            }

            Z3_int64 num, denom;
            if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
            }
            assert (num > 0 && denom == 1);
            N.Repetitions[i] = num;
        }

        auto gcdRepCount = N.Repetitions[0];
        for (unsigned i = 1; i < n; ++i) {
            gcdRepCount = boost::gcd(gcdRepCount, N.Repetitions[i]);
        }
        if (gcdRepCount != 1) {
            for (unsigned i = 0; i < n; ++i) {
                N.Repetitions[i] /= gcdRepCount;
            }
        }
    }

    Z3_model_dec_ref(ctx, model);

    Z3_solver_dec_ref(ctx, solver);
    Z3_del_context(ctx);

    #ifdef PRINT_INTRA_PARTITION_VECTOR_GRAPH
    auto & out = errs();

    out << "digraph \"V\" {\n";
    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;
        const auto n = K.size();
        for (unsigned i = 0; i < n; ++i) {
            const auto k = K[i];
            const RelationshipNode & node = Relationships[k];
            assert (node.Type == RelationshipNode::IsKernel);
            const auto id = M.find(k)->second;
            const auto & V = N.Repetitions[i];
            out << "v" << id << " [label=\""
                << k << ". " << node.Kernel->getName()
                << "  (" << V.numerator() << "/" << V.denominator()
                << ")\"];\n";
        }
    }

    for (const auto e : make_iterator_range(edges(G))) {
        const auto s = source(e, G);
        const auto t = target(e, G);
        const auto & V = G[e];
        out << "v" << s << " -> v" << t <<
               " [label=\"" << V.numerator() << "/" << V.denominator() << "\"];\n";
    }

    out << "}\n\n";
    out.flush();
    #endif



}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyInterPartitionSymbolicRates
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyInterPartitionSymbolicRates() {



}

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyInterPartitionSymbolicRates
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyInterPartitionSymbolicRates() {

    using BitSet = dynamic_bitset<>;

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    const auto m = num_edges(mBufferGraph);

    std::vector<BitSet> portRateSet(m + LastStreamSet - FirstStreamSet + 1U);

    unsigned portNum = 1;
    unsigned nextRateId = PartitionCount + 1;

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {
        auto updateEdgeRate = [&](const BufferGraph::edge_descriptor & e, const size_t streamSet) {
            BufferPort & port = mBufferGraph[e];
            assert (portNum < m);
            port.SymbolicRateId = portNum++;
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) {
                port.SymbolicRateId = 0;
                return;
            }
            if (bn.isNonThreadLocal()) {
                const Binding & binding = port.Binding;
                if (isNonSynchronousRate(binding)) {
                    BitSet & bs = portRateSet[port.SymbolicRateId];
                    const auto rateId = nextRateId++;
                    bs.resize(nextRateId);
                    bs.set(rateId);
                }
            }
        };

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, source(e, mBufferGraph));
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, target(e, mBufferGraph));
        }
    }

    BitSet accum(nextRateId);

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        accum.reset();

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_LIKELY(!bn.isConstant())) {
                const BufferPort & port = mBufferGraph[e];
                BitSet & bs = portRateSet[port.SymbolicRateId];
                bs.resize(nextRateId);
                bs.set(KernelPartitionId[kernel]);
                const auto k = m + streamSet - FirstStreamSet;
                assert (k < portRateSet.size());
                const BitSet & src = portRateSet[k];
                assert (src.size() == bs.size());
                bs |= src;
                accum |= bs;
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_LIKELY(!bn.isConstant())) {
                const BufferPort & port = mBufferGraph[e];
                BitSet & bs = portRateSet[port.SymbolicRateId];
                bs.resize(nextRateId);
                const auto k = m + streamSet - FirstStreamSet;
                assert (k < portRateSet.size());
                BitSet & dst = portRateSet[k];
                dst.resize(nextRateId);
                bs |= accum;
                dst |= bs;
            }
        }
    }

    std::map<BitSet, unsigned> rateMap;

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        auto updateEdgeRate = [&](const BufferGraph::edge_descriptor & e, const BufferGraph::vertex_descriptor streamSet) {
            BufferPort & port = mBufferGraph[e];
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_LIKELY(!bn.isConstant())) {
                BitSet & bs = portRateSet[port.SymbolicRateId];
                assert (bs.size() == nextRateId);
                const auto f = rateMap.find(bs);
                unsigned symRateId = 0;
                if (f == rateMap.end()) {
                    symRateId = (unsigned)rateMap.size() + 1U;
                    rateMap.emplace(std::move(bs), symRateId);
                } else {
                    symRateId = f->second;
                }
                port.SymbolicRateId = symRateId;
            }
        };

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, source(e, mBufferGraph));
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, target(e, mBufferGraph));
        }
    }

}

#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculatePartialSumStepFactors
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::calculatePartialSumStepFactors(KernelBuilder & b) {

    PartialSumStepFactorGraph G(LastStreamSet + 1);

    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        auto checkForPopCountRef = [&](const BufferGraph::edge_descriptor io) {
            const BufferPort & port = mBufferGraph[io];
            const Binding & binding = port.Binding;
            const ProcessingRate & rate = binding.getRate();
            if (LLVM_UNLIKELY(rate.isPartialSum())) {
                const auto inputRefPort = getReference(kernel, port.Port);
                const auto streamSet = getInputBufferVertex(kernel, inputRefPort);
                assert (streamSet > LastKernel);
                const auto refOuput = in_edge(streamSet, mBufferGraph);
                const BufferPort & refOutputRate = mBufferGraph[refOuput];
                const BufferPort & refInputRate = getInputPort(kernel, inputRefPort);
                const auto ratio = refInputRate.Minimum / refOutputRate.Minimum;
                assert (ratio.denominator() == 1 && ratio.numerator() > 0);
                assert (!edge(streamSet, kernel, G).second);
                add_edge(streamSet, kernel, ratio.numerator(), G);
            }
        };

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            checkForPopCountRef(input);
        }
        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            checkForPopCountRef(output);
        }
    }

    const auto bw = b.getBitBlockWidth();
    const auto fw = b.getSizeTy()->getIntegerBitWidth();
    assert ((bw % fw) == 0 && bw >= fw);
    const auto stepsPerBlock = bw / fw;

    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            if (out_degree(streamSet, G) != 0) {
                auto maxStepFactor = StrideRepetitionVector[kernel];
                for (const auto e : make_iterator_range(out_edges(streamSet, G))) {
                    const auto consumer = target(e, G);
                    assert (consumer > kernel && consumer <= LastKernel);
                    const auto k = G[e] * StrideRepetitionVector[consumer];
                    maxStepFactor = std::max(maxStepFactor, k);
                }
                maxStepFactor = round_up_to(maxStepFactor, stepsPerBlock);
                const auto spanLength = maxStepFactor / stepsPerBlock;
                add_edge(kernel, streamSet, spanLength, G);
                assert (spanLength > 0);
                BufferNode & bn = mBufferGraph[streamSet];
                bn.PartialSumSpanLength = std::max(bn.PartialSumSpanLength, spanLength + 1);
            }
        }

    }

    mPartialSumStepFactorGraph = G;
}

} // end of kernel namespace
