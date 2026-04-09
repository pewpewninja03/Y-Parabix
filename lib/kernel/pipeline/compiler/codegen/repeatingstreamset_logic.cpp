#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMetaDataForRepeatingStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateMetaDataForRepeatingStreamSets(KernelBuilder & b) {

    if (mTarget->hasInternallyGeneratedStreamSets()) {

        const PipelineKernel * const pk = cast<PipelineKernel>(mTarget);
        const auto & kernels = pk->getKernels();
        const auto m = kernels.size();

        flat_set<const RepeatingStreamSet *> touched;

        std::vector<Constant *> maxStrides;

        // the ordering of the kernels may differ between the input ordering of the
        // pipeline kernel and what was actually compiled by the program.

        for (unsigned i = 0; i < m; ++i) {
            const Kernel * const kernel = kernels[i].Object;
            if (LLVM_UNLIKELY(kernel->hasInternallyGeneratedStreamSets())) {
                maxStrides.push_back(b.getSize(MaximumNumOfStrides[i]));
            }
        }

        // TODO: use graph for this?

        Constant * sz_ZERO = b.getSize(0);

        const auto & S = mTarget->getInternallyGeneratedStreamSets();
        for (auto s : S) {
            Constant * ms = sz_ZERO;
            for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
                const RelationshipNode & rn = mStreamGraph[streamSet];
                assert (rn.Type == RelationshipNode::IsStreamSet);
                if (rn.Relationship == s) {
                    ms = b.getSize(getGuaranteedRepeatingStreamSetLength(b, streamSet, true));
                    break;
                }
            }
            maxStrides.push_back(ms);
        }

        Module * const module = mTarget->getModule();
        NamedMDNode * const md = module->getOrInsertNamedMetadata("rsl");
        assert (md->getNumOperands() == 0);
        Constant * ar = ConstantArray::get(ArrayType::get(b.getSizeTy(), maxStrides.size()), maxStrides);
        md->addOperand(MDNode::get(module->getContext(), {ConstantAsMetadata::get(ar)}));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getMaximumNumOfStridesForRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
size_t PipelineCompiler::getGuaranteedRepeatingStreamSetLength(KernelBuilder & b, const unsigned streamSet, const bool addOverflow) const {
    Rational ub{0U};
    const auto bw = b.getBitBlockWidth();
    for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
        const auto consumer = target(e, mBufferGraph);
        assert (consumer >= FirstKernel && consumer <= PipelineOutput);
        const auto m = MaximumNumOfStrides[consumer];
        const BufferPort & bp = mBufferGraph[e];
        auto v = bp.Maximum * m;
        if (addOverflow) {
            v += std::max(std::max(bp.Add, bp.LookAhead), bp.EmptyOverflow);
        }
        ub = std::max(ub, v);
    }
    assert (ub.denominator() == 1);
    return round_up_to<size_t>(ub.numerator(), bw);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindRepeatingStreamSetInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::bindRepeatingStreamSetInitializationArguments(KernelBuilder & b, ArgIterator & arg, const ArgIterator & arg_end) const {

    if (mTarget->hasInternallyGeneratedStreamSets()) {

        const auto & S = mTarget->getInternallyGeneratedStreamSets();
        const auto n = S.size(); assert (n > 0);
        assert (out_degree(PipelineInput, mInternallyGeneratedStreamSetGraph) == n);

        InternallyGeneratedStreamSetGraph::out_edge_iterator ei, ei_end;
        std::tie(ei, ei_end) = out_edges(PipelineInput, mInternallyGeneratedStreamSetGraph);

        for (unsigned i = 0; i < n; ++i) {

            assert (arg != arg_end);
            Value * const addr = &*arg++;
            assert (arg != arg_end);
            Value * const runLength = &*arg++;

            assert (ei != ei_end);

            assert (mInternallyGeneratedStreamSetGraph[*ei] == i);
            const auto streamSet = target(*ei++, mInternallyGeneratedStreamSetGraph);
            auto & N = mInternallyGeneratedStreamSetGraph[streamSet];

            N.StreamSet = addr;
            N.RunLength = runLength;

            assert (streamSet >= FirstStreamSet);
            // an internally generated streamset might only be used by one of the nested kernels
            if (streamSet <= LastStreamSet) {
                const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
                Value * const handle = b.getScalarFieldPtr(handleName).first;
                const BufferNode & bn = mBufferGraph[streamSet];
                #ifndef NDEBUG
                assert (bn.isConstant());
                assert (!bn.isTruncated());
                const RelationshipNode & rn = mStreamGraph[streamSet];
                assert (rn.Type == RelationshipNode::IsStreamSet);
                assert (isa<RepeatingStreamSet>(rn.Relationship));
                assert (cast<RepeatingStreamSet>(rn.Relationship)->isDynamic());
                #endif
                // external buffers already have a buffer handle
                RepeatingBuffer * const buffer = cast<RepeatingBuffer>(bn.OutputBuffer);
                buffer->setHandle(handle);
                Value * const ba = b.CreatePointerCast(addr, buffer->getPointerType());
                buffer->setBaseAddress(b, ba);
                buffer->setModulus(runLength);
                const auto lengthName = REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet);
                b.setScalarField(lengthName, runLength);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addRepeatingStreamSetInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addRepeatingStreamSetInitializationArguments(const unsigned kernelId, ArgVec & args) const {

    #ifndef NDEBUG
    const auto kernel = getKernel(kernelId);
    unsigned m = 0;
    if (kernel->hasInternallyGeneratedStreamSets()) {
        m = kernel->getInternallyGeneratedStreamSets().size(); assert (m > 0);
    }
    assert (in_degree(kernelId, mInternallyGeneratedStreamSetGraph) == m);
    #endif

    if (in_degree(kernelId, mInternallyGeneratedStreamSetGraph) > 0) {
        #ifndef NDEBUG
        unsigned expected = 0;
        #endif
        for (auto e : make_iterator_range(in_edges(kernelId, mInternallyGeneratedStreamSetGraph))) {
            assert (mInternallyGeneratedStreamSetGraph[e] == expected++);
            const auto streamSet = source(e, mInternallyGeneratedStreamSetGraph);
            const auto & N = mInternallyGeneratedStreamSetGraph[streamSet];
            args.push_back(N.StreamSet);
            args.push_back(N.RunLength);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateGlobalDataForRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateGlobalDataForRepeatingStreamSet(KernelBuilder & b, const unsigned streamSet) {
    const BufferNode & bn = mBufferGraph[streamSet];
    RepeatingBuffer * const buffer = cast<RepeatingBuffer>(bn.OutputBuffer);

    const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
    Value * const handle = b.getScalarFieldPtr(handleName).first;
    buffer->setHandle(handle);

    const RelationshipNode & rn = mStreamGraph[streamSet];
    assert (rn.Type == RelationshipNode::IsStreamSet);
    const RepeatingStreamSet * const ss = cast<RepeatingStreamSet>(rn.Relationship);

    if (ss->isUnaligned()) {
        bool unaligned = true;
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & bp = mBufferGraph[e];
            const auto & attrs = bp.getAttributes();
            unaligned &= attrs.hasAttribute(AttrId::AllowsUnalignedAccess);
        }
        if (LLVM_UNLIKELY(!unaligned)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Repeating streamset is marked as unaligned but ";
            bool notFirst = false;
            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const BufferPort & bp = mBufferGraph[e];
                const auto & attrs = bp.getAttributes();
                if (!attrs.hasAttribute(AttrId::AllowsUnalignedAccess)) {
                    const auto consumer = target(e, mBufferGraph);
                    const Binding & input = bp.Binding;
                    if (notFirst) {
                        out << ", ";
                    }
                    out << getKernel(consumer)->getName() << "." << input.getName();
                    notFirst = true;
                }
            }
            out << " is not explicitly marked as allowing unaligned access";
            report_fatal_error(StringRef(out.str()));
        }
    }

    if (ss->isDynamic()) {
        assert (isFromCurrentFunction(b, buffer->getBaseAddress(b)));
    } else {
        Rational ub{1U};
        const auto maxStrideLength = getGuaranteedRepeatingStreamSetLength(b, streamSet, true);
        auto info = cast<PipelineKernel>(mTarget)->createRepeatingStreamSet(b, ss, maxStrideLength);
        Value * const ba = b.CreatePointerCast(info.first, buffer->getPointerType());
        buffer->setBaseAddress(b, ba);
        buffer->setModulus(info.second);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addRepeatingStreamSetBufferProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addRepeatingStreamSetBufferProperties(KernelBuilder & b) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isConstant() && !bn.isTruncated())) {
            auto & S = mStreamGraph[streamSet];
            assert (S.Type == RelationshipNode::IsStreamSet);
            assert (isa<RepeatingStreamSet>(S.Relationship));

            Type * const handleTy = bn.OutputBuffer->getHandleType(b);
            mTarget->addInternalScalar(handleTy,
                REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet),
                                       getCacheLineGroupId(PipelineOutput));
            if (cast<RepeatingStreamSet>(S.Relationship)->isDynamic()) {
                mTarget->addInternalScalar(b.getSizeTy(),
                    REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet),
                                           getCacheLineGroupId(PipelineOutput));
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief deallocateRepeatingBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::deallocateRepeatingBuffers(KernelBuilder & b) {
//    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
//        const BufferNode & bn = mBufferGraph[streamSet];
//        if (LLVM_UNLIKELY(bn.isConstant())) {
//            const auto bufferName = REPEATING_STREAMSET_MALLOCED_DATA_PREFIX + std::to_string(streamSet);
//            b.CreateFree(b.getScalarField(bufferName));
//        }
//    }
}

}
