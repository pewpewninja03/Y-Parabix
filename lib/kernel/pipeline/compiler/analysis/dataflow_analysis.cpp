#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <toolchain/toolchain.h>
#include <boost/dynamic_bitset/dynamic_bitset.hpp>
#include <boost/graph/connected_components.hpp>
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

    }

    flat_set<size_t> joiningStreamSets;

    for (unsigned producerPartitionId = 0; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;

        assert (joiningStreamSets.empty());

        for (auto e : make_iterator_range(in_edges(producerPartitionId, P))) {
            const PartitionStreamSet & ss = P[e];
            if (ss.Type == 1) {
                joiningStreamSets.insert(ss.Id);
            }
        }

        for (auto e : make_iterator_range(out_edges(producerPartitionId, P))) {
            const PartitionStreamSet & ss = P[e];
            if (ss.Type == 1) {
                joiningStreamSets.insert(ss.Id);
            }
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

                                        if (producerPartitionId == consumerPartitionId || joiningStreamSets.count(streamSet)) {

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

        joiningStreamSets.clear();
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


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculateRelativeToInputDataTransferIORates
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::calculateRelativeToInputDataTransferIORates() {

    if (LLVM_UNLIKELY(FirstStreamSet == PipelineOutput)) {
        assert (LastStreamSet == PipelineOutput);
        return;
    }

    using NonFixedTaintGraph = adjacency_list<hash_setS, vecS, bidirectionalS, boost::dynamic_bitset<size_t>>;

    using Graph = adjacency_list<hash_setS, vecS, undirectedS>;

    const auto cfg = Z3_mk_config();
    Z3_set_param_value(cfg, "model", "true");
    Z3_set_param_value(cfg, "proof", "false");
//    Z3_set_param_value(cfg, "timeout", "2000");
    const auto ctx = Z3_mk_context(cfg);
    Z3_del_config(cfg);
    const auto solver = Z3_mk_optimize(ctx);
    Z3_optimize_inc_ref(ctx, solver);

    auto hard_assert = [&](Z3_ast c) {
        Z3_optimize_assert(ctx, solver, c);
    };

    auto soft_assert = [&](Z3_ast c) {
        Z3_optimize_assert_soft(ctx, solver, c, "1", nullptr);
    };

    auto check = [&]() -> Z3_lbool {
        #if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 5, 0)
        return Z3_optimize_check(ctx, solver, 0, nullptr);
        #else
        return Z3_optimize_check(ctx, solver);
        #endif
    };

    const auto varType = Z3_mk_real_sort(ctx);

    auto constant_real = [&](const Rational & value) {
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto multiply =[&](Z3_ast X, const Rational & value) {
        if ((value.numerator() == 1) && (value.denominator() == 1)) {
            return X;
        }
        Z3_ast args[2] = { X, constant_real(value) };
        return Z3_mk_mul(ctx, 2, args);
    };


    NonFixedTaintGraph T(PartitionCount);
    for (unsigned prodId = 0; prodId < PartitionCount; ++prodId) {
        T[prodId].resize(PartitionCount, false);
    }

    const auto z3_ZERO = constant_real(0);

    const auto z3_ONE = constant_real(1);

    std::vector<Z3_ast> VarList(LastStreamSet + 1U, nullptr);

    std::vector<Z3_ast> PartitionVarList(PartitionCount, nullptr);

    for (unsigned i = 0; i < PartitionCount; ++i) {
        auto rootVar = Z3_mk_fresh_const(ctx, nullptr, varType);
        hard_assert(Z3_mk_gt(ctx, rootVar, z3_ZERO));
        PartitionVarList[i] = rootVar;
    }

    // we cannot predict how much data will be passed to a pipeline
    if (out_degree(PipelineInput, mBufferGraph) > 0) {
        for (const auto input : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
            Z3_ast expOutRate = Z3_mk_fresh_const(ctx, nullptr, varType);
            hard_assert(Z3_mk_gt(ctx, expOutRate, z3_ZERO));
            const auto streamSet = target(input, mBufferGraph);
            VarList[streamSet] = expOutRate;
        }

        // if the pipeline has input, allow the pipeline to adjust the segment size
        assert (KernelPartitionId[PipelineInput] == 0);
        T[0].set(0);
    }


    assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);
    assert (PipelineOutput == FirstKernelInPartition[PartitionCount - 1]);

    for (unsigned i = PipelineInput; i < PipelineOutput; ++i) {
        const auto partId = KernelPartitionId[i];
        assert (partId < PartitionCount);
        auto rootVar = PartitionVarList[partId];
        VarList[i] = multiply(rootVar, StrideRepetitionVector[i]);
    }
    VarList[PipelineOutput] = z3_ONE;

    for (auto kernel = FirstKernel; kernel <= PipelineOutput; ++kernel) {

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(input, mBufferGraph);
            assert (VarList[streamSet] != nullptr);
            const auto producer = parent(streamSet, mBufferGraph);
            const auto prodPartId = KernelPartitionId[producer];
            const auto consPartId = KernelPartitionId[kernel];
            if (prodPartId != consPartId) {
                const BufferPort & port = mBufferGraph[input];
                const Binding & bind = port.Binding;
                const ProcessingRate & iRate = bind.getRate();
                if (LLVM_UNLIKELY(iRate.isGreedy())) {
                    // VarList[streamSet] == VarList[streamSet]
                } else {
                    const auto expectedInput = (port.Minimum + port.Maximum) * Rational{1, 2};
                    assert (expectedInput.numerator() > 0);
                    Z3_ast expInRate = multiply(VarList[kernel], expectedInput);
                    soft_assert(Z3_mk_eq(ctx, expInRate, VarList[streamSet]));
                }
                add_edge(prodPartId, consPartId, T);
                if (port.Minimum != port.Maximum) {
                    assert (consPartId > 0);
                    T[consPartId].set(consPartId);
                }
            }
        }

        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            assert (VarList[streamSet] == nullptr);
            const BufferPort & port = mBufferGraph[output];
            const Binding & bind = port.Binding;
            const ProcessingRate & oRate = bind.getRate();

            Z3_ast expOutRate = nullptr;

            if (LLVM_UNLIKELY(oRate.isUnknown())) {
                // we cannot predict how much data will be passed to a pipeline
                expOutRate = Z3_mk_fresh_const(ctx, nullptr, varType);
            } else {
                const auto expectedOutput = (port.Minimum + port.Maximum) * Rational{1, 2};
                assert (expectedOutput.numerator() > 0);
                expOutRate = multiply(VarList[kernel], expectedOutput);
            }

            hard_assert(Z3_mk_gt(ctx, expOutRate, z3_ZERO));
            VarList[streamSet] = expOutRate;
            if (port.Minimum != port.Maximum) {
                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    const auto consPartId = KernelPartitionId[consumer];
                    assert (consPartId != KernelPartitionId[kernel]);
                    assert (consPartId > 0);
                    T[consPartId].set(consPartId);
                }
            }
        }
    }

    SmallVector<Z3_ast, 2> fakeIOVars;
    for (unsigned kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        if (in_degree(kernel, mBufferGraph) == 0) {
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                fakeIOVars.push_back(VarList[streamSet]);
            }
        }
    }

    if (fakeIOVars.empty()) {
        for (const auto input : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
            const auto streamSet = target(input, mBufferGraph);
            fakeIOVars.push_back(VarList[streamSet]);
        }
    }

    const auto m = fakeIOVars.size(); assert (m > 0);

    SmallVector<Z3_ast, 2> fakeIOConstraint(m);
    for (unsigned i = 0; i < m; ++i) {
        fakeIOConstraint[i] = Z3_mk_eq(ctx, fakeIOVars[i], z3_ONE);
    }

    Z3_ast constraint;
    if (m == 1) {
        constraint = fakeIOConstraint[0];
    } else {
        constraint = Z3_mk_or(ctx, m, fakeIOConstraint.data());
    }
    hard_assert(constraint);

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        report_fatal_error("Z3 failed to find a solution to the maximum permitted dataflow problem");
    }

    const auto model = Z3_optimize_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);


//    Z3_ast value;
//    if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, VarList[PipelineInput], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
//        report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
//    }

//    Z3_int64 pipelineInputNum, pipelineInputDenom;
//    if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &pipelineInputNum, &pipelineInputDenom) != Z3_L_TRUE)) {
//        report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
//    }
//    assert (pipelineInputDenom > 0);
//    assert (pipelineInputNum > 0);

//    size_t lcmOfDenom = 1UL;

//    std::vector<Rational> partitionRepVector(PartitionCount);
//    const Rational inRatio{pipelineInputDenom, pipelineInputNum};

//    for (unsigned partId = 0; partId < PartitionCount; ++partId) {

//        Z3_ast value;
//        if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, PartitionVarList[partId], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
//            report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
//        }

//        Z3_int64 num, denom;
//        if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
//            report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
//        }

//        assert (num > 0);
//        assert (denom > 0);
//        // Sometimes Z3 may return extremely large pipeline input num/denoms and partiton num/denoms that cancel one
//        // another out. To mitigate potential 64-bit overflows, split the equation into two rational nums.
//        const auto rv = Rational{num, denom} * inRatio;
//        const auto firstKernel = FirstKernelInPartition[partId];
//        assert (KernelPartitionId[firstKernel] == partId);

//        assert (StrideRepetitionVector[firstKernel] > 0);

//        partitionRepVector[partId] = (rv / StrideRepetitionVector[firstKernel]);
//        assert (partitionRepVector[partId].numerator() > 0);
//        const auto m = partitionRepVector[partId].denominator();
//        if (m > 1) {
//            lcmOfDenom = boost::lcm(lcmOfDenom, m);
//        }
//    }

//    assert (lcmOfDenom > 0);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        Z3_ast value;
        if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, VarList[streamSet], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
        }

        Z3_int64 num, denom;
        if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
        }

        assert (denom > 0);
        assert (num > 0);

        BufferNode & bn = mBufferGraph[streamSet];
        bn.RelativeIORate = Rational{num, denom};
    }

    Z3_model_dec_ref(ctx, model);
    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    Z3_reset_memory();


    #ifndef NDEBUG
    const reverse_traversal ordering(PartitionCount);
    assert (is_valid_topological_sorting(ordering, T));
    #endif

    // iterate through the graph in topological order to determine what portions of
    // the program are not strictly fixed rate

    Graph P(PartitionCount);

    for (unsigned partId = 0; partId < PartitionCount; ++partId) {
        const auto & X = T[partId];
        if (X.any()) {
            const auto firstKernel = FirstKernelInPartition[partId];
            const auto firstKernelInNextPartition = FirstKernelInPartition[partId + 1];
            for (auto kernel = firstKernel + 1; kernel < firstKernelInNextPartition; ++kernel) {
                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    const auto producer = parent(streamSet, mBufferGraph);
                    const auto prodPartId = KernelPartitionId[producer];
                    if ((prodPartId != partId) && (X != T[prodPartId])) {
                        add_edge(partId, prodPartId, P);
                    }
                }
            }
            for (const auto output : make_iterator_range(out_edges(partId, T))) {
                T[target(output, T)] |= X;
            }
        }
    }

    for (auto kernel = PipelineInput; kernel <= PipelineOutput; ++kernel) {
        const auto partId = KernelPartitionId[kernel];
        assert (partId < PartitionCount);
        const auto R = StrideRepetitionVector[kernel]; // partitionRepVector[partId] * StrideRepetitionVector[kernel] * lcmOfDenom;
        const auto & X = T[partId];
        if (X.any()) {
            MinimumNumOfStrides[kernel] = 0;
            MaximumNumOfStrides[kernel] = R * 2;
        } else {
            MinimumNumOfStrides[kernel] = R;
            MaximumNumOfStrides[kernel] = R;
        }
    }

#if 0
    if (LLVM_UNLIKELY(P[0].ExpectedStridesPerSegment != Rational{1})) {
        auto checkIO = [](const Bindings & bindings) -> bool {
            for (const Binding & binding : bindings) {
                if (isCountable(binding) && !binding.isDeferred()) {
                    return true;
                }
            }
            return false;
        };
        if (checkIO(mPipelineKernel->getInputStreamSetBindings()) || checkIO(mPipelineKernel->getOutputStreamSetBindings())) {
            errs() << "WARNING! Pipeline "
                   << mPipelineKernel->getName() <<
                      " requires more than one stride of input but has at least one "
                      "non-deferred Countable I/O rate. This may cause I/O errors.\n\n"
                      "Check -PrintPipelineGraph for details.\n";
        }
    }
#endif

}

} // end of kernel namespace
