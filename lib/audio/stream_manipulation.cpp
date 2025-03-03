#include "audio/stream_manipulation.h"

namespace audio
{

    MergeKernel::MergeKernel(LLVMTypeSystemInterface &b, const unsigned int bitsPerSample, StreamSet *const firstInputStream, StreamSet *const secondInputStream, StreamSet *const outputStream)
        : MultiBlockKernel(b, "MergeKernel_" + std::to_string(bitsPerSample),
                           {Binding{"firstInputStream", firstInputStream}, Binding{"secondInputStream", secondInputStream}},
                           {Binding{"outputStream", outputStream}}, {}, {}, {}),
          bitsPerSample(bitsPerSample) {}

    void MergeKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = bitsPerSample;
        const unsigned inputPacksPerStride = fw * 1;
        const unsigned outputPacksPerStride = fw * 2;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *packLoop = b.CreateBasicBlock("packLoop");
        BasicBlock *packFinalize = b.CreateBasicBlock("packFinalize");
        Constant *const ZERO = b.getSize(0);
        Value *numOfBlocks = numOfStrides;
        b.CreateBr(packLoop);
        b.SetInsertPoint(packLoop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);

        std::vector<Value *> bytepack_1(inputPacksPerStride);
        std::vector<Value *> bytepack_2(inputPacksPerStride);
        for (unsigned i = 0; i < inputPacksPerStride; i++)
        {
            bytepack_1[i] = b.loadInputStreamPack("firstInputStream", ZERO, b.getInt32(i), blockOffsetPhi);
            bytepack_2[i] = b.loadInputStreamPack("secondInputStream", ZERO, b.getInt32(i), blockOffsetPhi);
        }

        std::vector<Value *> output(outputPacksPerStride);
        for (unsigned i = 0; i < inputPacksPerStride; i++)
        {
            output[2*i] = b.esimd_mergel(bitsPerSample, bytepack_1[i], bytepack_2[i]);
            output[2*i + 1] = b.esimd_mergeh(bitsPerSample, bytepack_1[i], bytepack_2[i]);
        }

        for (int i = 0; i < outputPacksPerStride; ++i)
        {
            b.storeOutputStreamPack("outputStream", ZERO, b.getInt32(i), blockOffsetPhi, output[i]);
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, packLoop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);
        b.SetInsertPoint(packFinalize);
    }
    
    Split2Kernel::Split2Kernel(LLVMTypeSystemInterface &b, const unsigned int bitsPerSample, StreamSet *const inputStream, StreamSet *const outputStream_1, StreamSet *const outputStream_2)
        : MultiBlockKernel(b, "Split2Kernel_" + std::to_string(bitsPerSample),
                           {Binding{"inputStream", inputStream}},
                           {Binding{"outputStream_1", outputStream_1}, Binding{"outputStream_2", outputStream_2}}, {}, {}, {}), bitsPerSample(bitsPerSample) 
    {
        if (inputStream->getNumElements() != 1)
        {
            throw std::invalid_argument("Error: Input has " + std::to_string(inputStream->getNumElements()) + " streams. Input must be single stream.");
        }
    }

    void Split2Kernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = bitsPerSample;
        const unsigned inputPacksPerStride = fw * 2;
        const unsigned outputPacksPerStride = fw * 1;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *packLoop = b.CreateBasicBlock("packLoop");
        BasicBlock *packFinalize = b.CreateBasicBlock("packFinalize");
        Constant *const ZERO = b.getSize(0);
        Value *numOfBlocks = numOfStrides;
        b.CreateBr(packLoop);
        b.SetInsertPoint(packLoop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);



        std::vector<Value *> bytepack(inputPacksPerStride);
        for (unsigned i = 0; i < inputPacksPerStride; i++)
        {
            bytepack[i] = b.loadInputStreamPack("inputStream", ZERO, b.getInt32(i), blockOffsetPhi);
        }

        
        for (unsigned i = 0; i < outputPacksPerStride; i++)
        {
            Value *lo = b.hsimd_packl(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
            Value *hi = b.hsimd_packh(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
            b.storeOutputStreamPack("outputStream_1", ZERO, b.getInt32(i), blockOffsetPhi, lo);
            b.storeOutputStreamPack("outputStream_2", ZERO, b.getInt32(i), blockOffsetPhi, hi);
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, packLoop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);
        b.SetInsertPoint(packFinalize);
    }

    SplitKernel::SplitKernel(LLVMTypeSystemInterface &b, const unsigned int bitsPerSample, StreamSet *const inputStreams, StreamSet *const outputStreams)
        : MultiBlockKernel(b, "SplitKernel_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                           {Binding{"inputStreams", inputStreams}},
                           {Binding{"outputStreams", outputStreams}}, {}, {}, {}),
        bitsPerSample(bitsPerSample), numInputStreams(inputStreams->getNumElements()) {}

    void SplitKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned fw = bitsPerSample;
        const unsigned inputPacksPerStride = fw * 2;
        const unsigned outputPacksPerStride = fw * 1;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *packLoop = b.CreateBasicBlock("packLoop");
        BasicBlock *packFinalize = b.CreateBasicBlock("packFinalize");
        Constant *const ZERO = b.getSize(0);
        Value *numOfBlocks = numOfStrides;
        b.CreateBr(packLoop);
        b.SetInsertPoint(packLoop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        std::vector<Value *> bytepack(inputPacksPerStride);
        std::vector<Value *> lo(outputPacksPerStride);
        std::vector<Value *> hi(outputPacksPerStride);
        for (int streamIndex = 0; streamIndex < numInputStreams; ++streamIndex)
        {
            Constant *const STREAMINDEX = b.getSize(streamIndex);
            Constant *const LOWSTREAMINDEX = b.getSize(2 * streamIndex);
            Constant *const HIGHSTREAMINDEX = b.getSize(2 * streamIndex + 1);
            for (unsigned i = 0; i < inputPacksPerStride; i++)
            {
                bytepack[i] = b.loadInputStreamPack("inputStreams", STREAMINDEX, b.getInt32(i), blockOffsetPhi);
            }

            for (unsigned i = 0; i < outputPacksPerStride; i++)
            {
                lo[i] = b.hsimd_packl(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                hi[i] = b.hsimd_packh(2 * bitsPerSample, bytepack[2 * i], bytepack[2 * i + 1]);
                b.storeOutputStreamPack("outputStreams", LOWSTREAMINDEX, b.getInt32(i), blockOffsetPhi, lo[i]);
                b.storeOutputStreamPack("outputStreams", HIGHSTREAMINDEX, b.getInt32(i), blockOffsetPhi, hi[i]);
            }
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, packLoop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, packLoop, packFinalize);
        b.SetInsertPoint(packFinalize);
    }
}
