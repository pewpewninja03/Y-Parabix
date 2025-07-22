#include "pipeline_analysis.hpp"

#include <queue>

namespace kernel {

void PipelineAnalysis::addFlowControlAnnotations() {

    size_t firstPartitionId = KernelPartitionId[FirstKernel];
    size_t lastPartitionId = KernelPartitionId[LastKernel];

    AllowIOProcessThread = false;

    if (codegen::UseProcessThreadForIO && !IsNestedPipeline && FirstKernel < LastKernel) {

        assert (FirstKernel != PipelineInput && LastKernel != PipelineOutput);

        for (; firstPartitionId < lastPartitionId; ++firstPartitionId) {
            const auto kernelId = FirstKernelInPartition[firstPartitionId];
            if (in_degree(kernelId, mBufferGraph) > 0) {
                break;
            }
        }
        assert (firstPartitionId > 0);
        for (; lastPartitionId >= firstPartitionId; --lastPartitionId) {
            const auto kernelId = FirstKernelInPartition[lastPartitionId];
            if (out_degree(kernelId, mBufferGraph) != 0) {
                break;
            }
        }

        if (LLVM_UNLIKELY(lastPartitionId < firstPartitionId)) {
            // No kernels that can be isolated? outside of a nested pipeline, the only
            // way for this to occur is if all input and output are transferred through
            // pipeline I/O streamsets.
            firstPartitionId = KernelPartitionId[FirstKernel];
            lastPartitionId = KernelPartitionId[LastKernel];
        } else {
            #ifndef NDEBUG
            for (auto kernel = FirstKernelInPartition[firstPartitionId]; kernel < FirstKernelInPartition[lastPartitionId + 1]; ++kernel) {
                assert (in_degree(kernel, mBufferGraph) > 0);
                assert (out_degree(kernel, mBufferGraph) > 0);
            }
            #endif
            const auto firstComputeKernel = FirstKernelInPartition[firstPartitionId];
            const auto afterLastComputeKernel = FirstKernelInPartition[lastPartitionId + 1];
            for (auto kernel = PipelineInput; kernel < firstComputeKernel; ++kernel) {
                assert (in_degree(kernel, mBufferGraph) == 0);
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                        const auto consumer = target(input, mBufferGraph);
                        if (LLVM_LIKELY(consumer < afterLastComputeKernel)) {
                            mBufferGraph[input].Flags |= BufferPortType::IsCrossThreaded;
                            mBufferGraph[output].Flags |= BufferPortType::IsCrossThreaded;
                            auto & bn = mBufferGraph[streamSet];
                            bn.Type |= BufferType::CrossThreaded;
                            if (LLVM_UNLIKELY(bn.isThreadLocal())) {
                                bn.Locality = BufferLocality::GloballyShared;
                            }
                        }
                    }
                }
            }

            for (auto kernel = afterLastComputeKernel; kernel <= PipelineOutput; ++kernel) {
                assert (out_degree(kernel, mBufferGraph) == 0);
                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    const auto output = in_edge(streamSet, mBufferGraph);
                    const auto producer = source(output, mBufferGraph);
                    if (LLVM_LIKELY(producer >= firstComputeKernel)) {
                        mBufferGraph[input].Flags |= BufferPortType::IsCrossThreaded;
                        mBufferGraph[output].Flags |= BufferPortType::IsCrossThreaded;
                        auto & bn = mBufferGraph[streamSet];
                        bn.Type |= BufferType::CrossThreaded;
                        if (LLVM_UNLIKELY(bn.isThreadLocal())) {
                            bn.Locality = BufferLocality::GloballyShared;
                        }
                    }
                }
            }
            AllowIOProcessThread = true;
        }

    }

    FirstComputePartitionId = firstPartitionId;
    LastComputePartitionId = lastPartitionId;

//    for (size_t partitionId = firstPartitionId; partitionId <= lastPartitionId; ++partitionId) {
//        const auto kernelId = FirstKernelInPartition[partitionId];
//        if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || IsNestedPipeline) {
//            mBufferGraph[kernelId].Type |= PermitSegmentSizeSlidingWindowing;
//        }
//    }

}


}
