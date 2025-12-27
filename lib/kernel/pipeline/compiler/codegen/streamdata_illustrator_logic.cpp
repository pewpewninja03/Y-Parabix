#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerStreamSetIllustrator
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::registerStreamSetIllustrator(KernelBuilder & b, const size_t streamSet) const {

    if ((mBufferGraph[streamSet].Type & HasIllustratedStreamset) == 0) {
        return;
    }

    for (const auto & bind : mIllustratedStreamSetBindings) {
        if (bind.StreamSet == streamSet) {

            const auto & entry = mStreamGraph[streamSet];
            assert (entry.Type == RelationshipNode::IsStreamSet);
            const StreamSet * const ss = cast <StreamSet>(entry.Relationship);

            Value * handle = mKernelSharedHandle;
            if (LLVM_UNLIKELY(handle == nullptr)) {
                handle = b.CreateIntToPtr(b.getInt64(mKernelId), b.getVoidPtrTy());
            }

            assert (mKernel);

            // TODO: should buffers have row major streamsets?

            FixedArray<size_t, 0> emptyLoopVec;

            registerIllustrator(b,
                                b.getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT),
                                b.GetString(mKernel->getName()),
                                b.GetString(bind.Name),
                                handle,
                                ss->getNumElements(), 1, ss->getFieldWidth(), MemoryOrdering::RowMajor,
                                bind.IllustratorType, bind.ReplacementCharacter[0], bind.ReplacementCharacter[1],
                                emptyLoopVec);
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief illustrateStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::illustrateStreamSet(KernelBuilder & b, const size_t streamSet, Value * const initial, Value * const current) const {

    if ((mBufferGraph[streamSet].Type & HasIllustratedStreamset) == 0) {
        return;
    }

    assert (mInternallySynchronizedSubsegmentNumber);
    for (const auto & bind : mIllustratedStreamSetBindings) {
        if (bind.StreamSet == streamSet) {

            const auto & bn = mBufferGraph[streamSet];

            StreamSetBuffer * const buffer = bn.OutputBuffer;

            const auto & rt = mBufferGraph[in_edge(streamSet, mBufferGraph)];

            Value * produced = mCurrentProducedItemCountPhi[rt.Port];

            Value * const vba = getVirtualBaseAddress(b, rt, bn, produced, bn.isNonThreadLocal(), true);

            // TODO: if this kernel is state-free, we need to pass in some other value for the handle.
            // We can easily use kernel # for it here but what if we capture a value in the kernel itself?

            assert (mKernel);

            Value * handle = mKernelSharedHandle;
            if (LLVM_UNLIKELY(handle == nullptr)) {
                handle = b.CreateIntToPtr(b.getInt64(mKernelId), b.getVoidPtrTy());
            }

            // TODO: should we pass the values of the min repetition vector to better group the output?

            // TODO: should buffers have row major streamsets?
            captureStreamData(b,
                              b.GetString(mKernel->getName()),
                              b.GetString(bind.Name),
                              handle,
                              mInternallySynchronizedSubsegmentNumber,
                              buffer->getType(), MemoryOrdering::RowMajor,
                              vba, initial, current);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief kernelHasAnyPipelineIllustratedStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::kernelHasAnyPipelineIllustratedStreamSet(const size_t kernel) const {
    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (LLVM_UNLIKELY(br.isIllustrated())) {
            return true;
        }
    }
    return false;
}


}
