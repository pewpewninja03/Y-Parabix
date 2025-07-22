#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <z3.h>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief simpleEstimateInterPartitionDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::simpleEstimateInterPartitionDataflow(PartitionGraph & P, pipeline_random_engine & rng) {

    using NonFixedTaintGraph = adjacency_list<hash_setS, vecS, bidirectionalS, bool>;

    const auto cfg = Z3_mk_config();
    Z3_set_param_value(cfg, "model", "true");
    Z3_set_param_value(cfg, "proof", "false");
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

    auto multiply_vars =[&](Z3_ast X, Z3_ast Y) {
        Z3_ast args[2] = { X, Y };
        return Z3_mk_mul(ctx, 2, args);
    };

    auto multiply =[&](Z3_ast X, const Rational & value) {
        if ((value.numerator() == 1) && (value.denominator() == 1)) {
            return X;
        }
        Z3_ast args[2] = { X, constant_real(value) };
        return Z3_mk_mul(ctx, 2, args);
    };



    const auto numOfPartitions = num_vertices(P);

    std::vector<Z3_ast> VarList(num_vertices(Relationships));

    assert (P[0].Kernels.size() == 1);
    assert (P[0].Kernels[0] == PipelineInput);

    SmallVector<Z3_ast, 2> fakeIOVars;

    const auto z3_ZERO = constant_real(0);

    const auto z3_ONE = constant_real(1);

    for (size_t partId = 0; partId < numOfPartitions; ++partId) {
        const PartitionData & N = P[partId];
        const auto & K = N.Kernels;
        assert (K.size() > 0);
        auto rootVar = Z3_mk_fresh_const(ctx, nullptr, varType);
        hard_assert(Z3_mk_ge(ctx, rootVar, z3_ONE));
        const auto m = K.size(); assert (m > 0);
        for (unsigned i = 0; i < m; ++i) {
            const auto k = K[i];
            VarList[k] = multiply(rootVar, N.Repetitions[i]);
            assert (Relationships[k].Type == RelationshipNode::IsKernel);
        }
    }

    if (out_degree(0, P) == 0) {
        Z3_ast args[2];
        args[0] = VarList[PipelineInput];
        SmallVector<Z3_ast, 2> fakeIOConstraint;
        for (auto partId = 1U; partId < numOfPartitions; ++partId) {
            if (in_degree(partId, P) == 0) {
                const PartitionData & N = P[partId];
                const auto & K = N.Kernels;
                auto fakeIOVar = Z3_mk_fresh_const(ctx, nullptr, varType);
                fakeIOVars.push_back(fakeIOVar);
                hard_assert(Z3_mk_gt(ctx, fakeIOVar, z3_ZERO));
                args[1] = fakeIOVar;
                auto val = Z3_mk_mul(ctx, 2, args);
                hard_assert(Z3_mk_eq(ctx, val, VarList[K[0]]));
                fakeIOConstraint.push_back(Z3_mk_eq(ctx, fakeIOVar, z3_ONE));
            }
        }

        const auto m = fakeIOConstraint.size(); assert (m > 0);
        Z3_ast constraint;
        if (m == 1) {
            constraint = fakeIOConstraint[0];
        } else {
            constraint = Z3_mk_or(ctx, m, fakeIOConstraint.data());
        }
        hard_assert(constraint);
    }

    NonFixedTaintGraph T(numOfPartitions);
    for (unsigned prodId = 0; prodId < numOfPartitions; ++prodId) {
        T[prodId] = false;
    }

    for (unsigned prodId = 0; prodId < numOfPartitions; ++prodId) {
        const PartitionData & N = P[prodId];
        const auto & K = N.Kernels;
        const auto m = K.size();
        for (unsigned i = 0; i < m; ++i) {
            const auto producer = K[i];
            assert (VarList[producer]);
            assert (Relationships[producer].Type == RelationshipNode::IsKernel);

            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto prodBinding = target(e, Relationships);
                if (Relationships[prodBinding].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(prodBinding, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsStreamSet);
                    const RelationshipNode & output = Relationships[prodBinding];
                    assert (output.Type == RelationshipNode::IsBinding);
                    const Binding & outputBinding = output.Binding;
                    const ProcessingRate & oRate = outputBinding.getRate();

                    const RelationshipNode & producerNode = Relationships[producer];
                    assert (producerNode.Type == RelationshipNode::IsKernel);

                    Z3_ast expOutRate = nullptr;

                    if (LLVM_UNLIKELY(producer == PipelineInput || oRate.isUnknown())) {
                        // we cannot predict how much data will be passed to a pipeline
                        expOutRate = Z3_mk_fresh_const(ctx, nullptr, varType);
                    } else {
                        const auto s = producerNode.Kernel->getStride();
                        assert (oRate.getUpperBound() >= oRate.getLowerBound() || oRate.isUnknown());
                        const auto expectedOutput = (oRate.getLowerBound() + oRate.getUpperBound()) * Rational{s, 2};
                        assert (expectedOutput.numerator() > 0);
                        expOutRate = multiply(VarList[producer], expectedOutput);
                    }

                    VarList[streamSet] = expOutRate; assert (expOutRate);
                    hard_assert(Z3_mk_gt(ctx, expOutRate, z3_ZERO));

                    for (const auto e : make_iterator_range(out_edges(streamSet, Relationships))) {
                        const auto binding = target(e, Relationships);
                        const RelationshipNode & input = Relationships[binding];
                        if (input.Type == RelationshipNode::IsBinding) {
                            const Binding & inputBinding = input.Binding;
                            const ProcessingRate & iRate = inputBinding.getRate();

                            const auto f = first_out_edge(binding, Relationships);
                            assert (Relationships[f].Reason != ReasonType::Reference);
                            const unsigned consumer = target(f, Relationships);

                            const auto c = PartitionIds.find(consumer);
                            assert (c != PartitionIds.end());
                            const auto consumerPartitionId = c->second;
                            assert (prodId <= consumerPartitionId);

                            Z3_ast expInRate = nullptr;

                            if (LLVM_UNLIKELY(iRate.isGreedy())) {
                                expInRate = expOutRate; assert (expOutRate);
                            } else {
                                const RelationshipNode & consumerNode = Relationships[consumer];
                                assert (consumerNode.Type == RelationshipNode::IsKernel);
                                const auto s = consumerNode.Kernel->getStride();
                                assert (s > 0);
                                const auto expectedInput = (iRate.getLowerBound() + iRate.getUpperBound()) * Rational{s, 2};
                                assert (expectedInput.numerator() > 0);
                                expInRate = multiply(VarList[consumer], expectedInput);
                            }

                            Z3_ast constraint = Z3_mk_eq(ctx, expOutRate, expInRate);
                            if (prodId != consumerPartitionId) {
                                soft_assert(constraint);

                                 // mark that there is non-fixed dataflow between these
                                if (prodId != PipelineInput) {
                                    add_edge(prodId, consumerPartitionId, T);
                                    if (!oRate.isFixed() || !iRate.isFixed()) {
                                        T[consumerPartitionId] = true;
                                    }
                                }

                            } else {
                                hard_assert(constraint);
                            }
                        }
                    }

                }
            }
        }
    }

    #ifndef NDEBUG
    const reverse_traversal ordering(numOfPartitions);
    assert (is_valid_topological_sorting(ordering, T));
    #endif


    // iterate through the graph in topological order to determine what portions of
    // the program are not strictly fixed rate
    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        if (T[partId]) {
            for (const auto output : make_iterator_range(out_edges(partId, T))) {
                T[target(output, T)] = true;
            }
        }
    }

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        report_fatal_error("Z3 failed to find a solution to the maximum permitted dataflow problem");
    }

    const auto model = Z3_optimize_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);


    Z3_ast value;
    if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, VarList[PipelineInput], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
        report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
    }

    Z3_int64 pipelineInputNum, pipelineInputDenom;
    if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &pipelineInputNum, &pipelineInputDenom) != Z3_L_TRUE)) {
        report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
    }
    assert (pipelineInputDenom > 0);
    assert (pipelineInputNum > 0);

    size_t lcmOfDenom = 1UL;

    for (size_t partId = 0; partId < numOfPartitions; ++partId) {
        PartitionData & N = P[partId];

#if 0

        Z3_ast const stridesPerSegmentVar = VarList[N.Kernels[0]];
        Z3_ast value;
        if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, stridesPerSegmentVar, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
        }

        Z3_int64 num, denom;
        if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
        }

        assert (denom > 0);
        assert (num > 0);
        assert (N.Repetitions[0] > 0);

        N.ExpectedStridesPerSegment = Rational{pipelineInputDenom * num, pipelineInputNum * denom} / N.Repetitions[0];

        const auto m = N.ExpectedStridesPerSegment.denominator();
        if (m > 1) {
            lcmOfDenom = boost::lcm(lcmOfDenom, m);
        }
#endif
        N.ExpectedStridesPerSegment = Rational{1};

        N.StridesPerSegmentCoV = Rational{T[partId] ? 1U : 0U, 3U};
        N.LinkedGroupId = partId;
    }

    assert (lcmOfDenom > 0);

    pipelineInputDenom *= lcmOfDenom;

    for (unsigned prodId = 0; prodId < numOfPartitions; ++prodId) {
        PartitionData & N = P[prodId];
        const auto & K = N.Kernels;
        const auto m = K.size();
        for (unsigned i = 0; i < m; ++i) {
            const auto producer = K[i];
            assert (Relationships[producer].Type == RelationshipNode::IsKernel);
            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto prodBinding = target(e, Relationships);

                if (Relationships[prodBinding].Type == RelationshipNode::IsBinding) {

                    const auto f = first_out_edge(prodBinding, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsStreamSet);
                    assert (VarList[streamSet]);

                    Z3_ast value;
                    if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, VarList[streamSet], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
                        report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
                    }

                    Z3_int64 num, denom;
                    if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
                        report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
                    }

                    assert (denom > 0);

                    // StreamSetIORateMap.emplace(streamSet, Rational{pipelineInputDenom * num, pipelineInputNum * denom});
                }
            }
        }
    }

    Z3_model_dec_ref(ctx, model);
    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    Z3_reset_memory();

    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        PartitionData & N = P[partId];
        N.ExpectedStridesPerSegment *= lcmOfDenom;
    }

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

}


}
