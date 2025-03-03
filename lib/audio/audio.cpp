#include "audio/audio.h"
#include <iostream>
#include <kernel/io/source_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Value.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/core/relationship.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/stream_select.h>
#include "audio/stream_manipulation.h"
#include <llvm/IR/Intrinsics.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <kernel/pipeline/program_builder.h>

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P.captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P.captureByteData(#name, name)

#define NUM_HEADER_BYTES 44

namespace audio
{
    void ParseAudioBuffer(
        ProgramBuilder & P,
        Scalar *const fileDescriptor,
        unsigned int numChannels,
        unsigned int bitsPerSample,
        std::vector<StreamSet *> &outputDataStreams,
        const bool& splitChannels)
    {
        if (numChannels != 1 && numChannels != 2)
        {
            throw std::invalid_argument("Error: numChannels " + std::to_string(numChannels) + " is not valid");
        }

        if (splitChannels && outputDataStreams.size() != numChannels)
        {
            throw std::invalid_argument("Error: Splitting channel is on but numChannels " + std::to_string(numChannels) + " is not equal to number output streams");
        }

        if (numChannels == 2 && splitChannels)
        {
            unsigned SampleStreamFW = (splitChannels) ? bitsPerSample * numChannels : bitsPerSample;
            StreamSet *SampleStream = P.CreateStreamSet(1, SampleStreamFW);
            P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, SampleStream);
            P.CreateKernelCall<Split2Kernel>(bitsPerSample, SampleStream, outputDataStreams[0], outputDataStreams[1]);
        }
        else
        {
            P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, outputDataStreams[0]);
        }
    }

    void ParseAudioBuffer(
        ProgramBuilder & P,
        Scalar *const buffer,
        Scalar *const length,
        unsigned int numChannels,
        unsigned int bitsPerSample,
        std::vector<StreamSet *> &outputDataStreams,
        const bool& splitChannels)
    {
        if (numChannels != 1 && numChannels != 2)
        {
            throw std::invalid_argument("Error: numChannels " + std::to_string(numChannels) + " is not valid");
        }

        if (splitChannels && outputDataStreams.size() != numChannels)
        {
            throw std::invalid_argument("Error: Splitting channel is on but numChannels " + std::to_string(numChannels) + " is not equal to number output streams");
        }

        if (numChannels == 2 && splitChannels)
        {
            unsigned SampleStreamFW = (splitChannels) ? bitsPerSample * numChannels : bitsPerSample;
            StreamSet *SampleStream = P.CreateStreamSet(1, SampleStreamFW);
            P.CreateKernelCall<MemorySourceKernel>(buffer, length, SampleStream);
            P.CreateKernelCall<Split2Kernel>(bitsPerSample, SampleStream, outputDataStreams[0], outputDataStreams[1]);
        }
        else
        {
            P.CreateKernelCall<MemorySourceKernel>(buffer, length, outputDataStreams[0]);
        }
    }

    // adapted from chatgpt with some modifications :)
    struct WAVHeader
    {
        char RIFF[4] = {'R', 'I', 'F', 'F'};
        uint32_t chunkSize;
        char WAVE[4] = {'W', 'A', 'V', 'E'};
        char FMT[4] = {'f', 'm', 't', ' '};
        uint32_t subchunk1Size = 16; // For PCM
        uint16_t audioFormat = 1;    // PCM = 1
        uint16_t numChannels;
        uint32_t sampleRate;
        uint32_t byteRate;
        uint16_t blockAlign;
        uint16_t bitsPerSample;
        char DATA[4] = {'d', 'a', 't', 'a'};
        uint32_t subchunk2Size;
    };

    std::string createWAVHeader(
        const unsigned int &numChannels,
        const unsigned int &sampleRate,
        const unsigned int &bitsPerSample,
        const unsigned int &numSamples)
    {
        WAVHeader header;

        header.numChannels = numChannels;
        header.sampleRate = sampleRate;
        header.bitsPerSample = bitsPerSample;
        header.blockAlign = numChannels * (bitsPerSample / 8);
        header.byteRate = sampleRate * header.blockAlign;
        header.subchunk2Size = numSamples * header.blockAlign;
        header.chunkSize = 36 + header.subchunk2Size;

        std::ostringstream oss;
        oss.write(reinterpret_cast<char *>(&header), sizeof(header));
        return oss.str();
    }

    void readTextFile(const int &fd, std::vector<int8_t, AlignedAllocator<int8_t, 64>>& buffer)
    {
        buffer.clear();
        std::vector<int8_t> temp_buffer(4096);
        ssize_t bytesRead;

        while ((bytesRead = read(fd, temp_buffer.data(), temp_buffer.size())) > 0) {
            buffer.insert(buffer.end(), temp_buffer.begin(), temp_buffer.begin() + bytesRead);
        }
    }

    void readWAVHeader(const int &fd,
                       unsigned int &numChannels,
                       unsigned int &sampleRate,
                       unsigned int &bitsPerSample,
                       unsigned int &numSamples)
    {
        char temp_buffer[11];

        // validate file format
        ssize_t bytesRead = read(fd, &temp_buffer, 8);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "RIFF")
        {
            throw std::runtime_error("Error parsing file format: Chunk ID does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "WAVE")
        {
            throw std::runtime_error("Error parsing file format: Header does not match Wav format.");
        }

        bytesRead = read(fd, &temp_buffer, 10);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "fmt\x20")
        {
            throw std::runtime_error("Error parsing file format: Subchunk ID does not match Wav format.");
        }

        // read 2 bytes for num channels
        uint16_t num_channels;
        bytesRead = read(fd, reinterpret_cast<char *>(&num_channels), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret number channels.");
        }
        numChannels = num_channels;

        // read 4 bytes for sample rate
        bytesRead = read(fd, reinterpret_cast<char *>(&sampleRate), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret sample rate.");
        }

        // skip the next 6 bytes
        bytesRead = read(fd, temp_buffer, 6);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }

        // read 4 bytes for num bits per sample
        uint16_t bits_per_sample;
        bytesRead = read(fd, reinterpret_cast<char *>(&bits_per_sample), 2);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format.");
        }
        bitsPerSample = bits_per_sample;

        bytesRead = read(fd, temp_buffer, 4);
        temp_buffer[4] = '\0';
        if (bytesRead <= 0 || std::string(temp_buffer) != "data")
        {
            throw std::runtime_error("Error parsing file format: Subchunk 2 ID does not match Wav format.");
        }

        unsigned subchunk2_size;
        bytesRead = read(fd, reinterpret_cast<char *>(&subchunk2_size), 4);
        if (bytesRead <= 0)
        {
            throw std::runtime_error("Error parsing file format: Cannot interpret subchunk 2 size.");
        }

        numSamples = subchunk2_size / (numChannels * bitsPerSample / 8);
    }

    void S2P(
        ProgramBuilder &P,
        unsigned int bitsPerSample,
        StreamSet *const inputStream,
        StreamSet *&outputStreams)
    {
        if (bitsPerSample == 16)
        {
            StreamSet *LowStream = P.CreateStreamSet(1, 8);
            StreamSet *HighStream = P.CreateStreamSet(1, 8);
            P.CreateKernelCall<Split2Kernel>(8, inputStream, LowStream, HighStream);
            StreamSet *LowBitBasisStream = P.CreateStreamSet(8);
            StreamSet *HighBitBasisStream = P.CreateStreamSet(8);
            P.CreateKernelCall<S2PKernel>(LowStream, LowBitBasisStream);
            P.CreateKernelCall<S2PKernel>(HighStream, HighBitBasisStream);
            P.CreateKernelCall<ConcatenateKernel>(LowBitBasisStream, HighBitBasisStream, outputStreams);
        }
        else if (bitsPerSample == 8)
        {
            P.CreateKernelCall<S2PKernel>(inputStream, outputStreams);
        }
        else
        {
            throw std::invalid_argument("Only 8 and 16 bit depths are supported");
        }
    }

    void P2S(ProgramBuilder &P,
        StreamSet *const inputStreams,
        StreamSet *&outputStream)
    {
        if (inputStreams->getNumElements() == 16)
        {
            P.CreateKernelCall<P2S16Kernel>(inputStreams, outputStream);
        }
        else if (inputStreams->getNumElements() == 8)
        {
            P.CreateKernelCall<P2SKernel>(inputStreams, outputStream);
        }
        else
        {
            throw std::invalid_argument("Only 8 and 16 bit depths are supported");
        }
    }

    FlexS2PKernel::FlexS2PKernel(LLVMTypeSystemInterface &b, const unsigned int bitsPerSample, StreamSet *const inputStream, StreamSet *const outputStreams)
        :
         MultiBlockKernel(b, "FlexS2PKernel_" + std::to_string(bitsPerSample),
                           {Binding{"inputStream", inputStream, FixedRate(1)}},
                           {Binding{"outputStreams", outputStreams, FixedRate(1)}}, {}, {}, {}),
         bitsPerSample(bitsPerSample)
    {
        if (bitsPerSample != 4 && bitsPerSample % 8 != 0)
        {
            throw std::invalid_argument("bitsPerSample: " + std::to_string(bitsPerSample) + ". bitsPerSample must be 4 or multiple of 8");
        }
        if (inputStream->getNumElements() != 1)
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(inputStream->getNumElements()) + ". Input must be a mono stream");
        }
    }

    void FlexS2PKernel::generateMultiBlockLogic(KernelBuilder &b, Value *const numOfStrides)
    {
        const unsigned inputPacksPerStride = bitsPerSample;
        const unsigned packSize = b.getBitBlockWidth();
        const unsigned numElementsPerPack = packSize / bitsPerSample;

        BasicBlock *entry = b.GetInsertBlock();
        BasicBlock *loop = b.CreateBasicBlock("loop");
        BasicBlock *exit = b.CreateBasicBlock("exit");
        Constant *const ZERO = b.getSize(0);

        Type *vecType = FixedVectorType::get(b.getIntNTy(bitsPerSample), static_cast<unsigned>(numElementsPerPack));

        Value *numOfBlocks = numOfStrides;
        b.CreateBr(loop);
        b.SetInsertPoint(loop);
        PHINode *blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        Value *bytepack[inputPacksPerStride];
        for (unsigned i = 0; i < inputPacksPerStride; ++i)
        {
            bytepack[i] = b.loadInputStreamPack("inputStream", ZERO, b.getInt32(i), blockOffsetPhi);
        }

        for (unsigned j = 0;j<bitsPerSample;++j)
        {   
            Value* output = UndefValue::get(vecType);
            for (unsigned i = 0;i<inputPacksPerStride;++i)
            {
                Value* shifted = b.simd_slli(bitsPerSample, bytepack[i], bitsPerSample-1-j);
                Value *extractedBit = b.hsimd_signmask(bitsPerSample, shifted);
                output = b.CreateInsertElement(output, extractedBit,b.getInt32(i));
            }
            b.storeOutputStreamBlock("outputStreams", b.getSize(j), blockOffsetPhi, output);
        }

        Value *nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, loop);
        Value *moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

        b.CreateCondBr(moreToDo, loop, exit);
        b.SetInsertPoint(exit);
    }

    DiscontinuityKernel::DiscontinuityKernel(LLVMTypeSystemInterface &b, StreamSet *const inputStreams, const unsigned int &threshold, StreamSet *const markStream)
        : PabloKernel(b, "DiscontinuityKernel_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(threshold),
                      {Binding{"inputStreams", inputStreams, FixedRate(1), LookAhead(1)}},
                      {Binding{"markStream", markStream}}),
          threshold(threshold)
    {
    }

    void DiscontinuityKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> inputStreams = getInputStreamSet("inputStreams");
        const unsigned bitsPerSample = inputStreams.size();

        std::vector<PabloAST *> extendedStreams = bnc.SignExtend(inputStreams, bitsPerSample + 1);
        std::vector<PabloAST *> srExtendedStreams(extendedStreams.size());

        for (unsigned i = 0; i < extendedStreams.size(); ++i)
        {
            srExtendedStreams[i] = pb.createAdvance(extendedStreams[i], 1);
        } 

        std::vector<PabloAST *> srDifference = bnc.SubModular(extendedStreams, srExtendedStreams);

        std::vector<PabloAST *> flipBits(srDifference.size());
        for (unsigned i = 0; i < srDifference.size(); ++i)
        {
            flipBits[i] = pb.createNot(srDifference[i]);
        }

        std::vector<PabloAST *> negatives = bnc.AddModular(flipBits, 1);
        std::vector<PabloAST *> srAbsDiff = bnc.Select(srDifference[srDifference.size() - 1] /*sign*/, negatives, srDifference);

        PabloAST * exceedThreshold = pb.createAnd(pb.createAdvance(pb.createOnes(), 1), bnc.UGE(srAbsDiff, threshold));

        Var *result = getOutputStreamVar("markStream");
        pb.createAssign(pb.createExtract(result, pb.getInteger(0)), exceedThreshold);
    }

    AmplifyPabloKernel::AmplifyPabloKernel(LLVMTypeSystemInterface &b, const unsigned int bitsPerSample, StreamSet *const inputStreams, const unsigned int &factor, StreamSet *const outputStreams)
        : PabloKernel(b, "AmplifyPabloKernel_" + std::to_string(factor) + "_" + std::to_string(inputStreams->getNumElements()) + "_" + std::to_string(bitsPerSample),
                      {Binding{"inputStreams", inputStreams}},
                      {Binding{"outputStreams", outputStreams}}),
          bitsPerSample(bitsPerSample), numInputStreams(inputStreams->getNumElements()), factor(factor)
    {
        if (inputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("numInputStreams: " + std::to_string(inputStreams->getNumElements()) + " != numOutputStreams: " + std::to_string(outputStreams->getNumElements()));
        }
    }

    void AmplifyPabloKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> inputStreams = getInputStreamSet("inputStreams");
        const unsigned bitsPerSample = inputStreams.size();

        std::vector<PabloAST *> ExtendedStreams = bnc.SignExtend(inputStreams, bitsPerSample + std::log2(factor) + 1);
        std::vector<PabloAST *> AmplifiedStreams = bnc.MulModular(ExtendedStreams, factor);
        std::vector<PabloAST *> flipStreams(AmplifiedStreams.size());
        for (unsigned i=0;i<AmplifiedStreams.size();++i)
        {
            flipStreams[i] = pb.createNot(AmplifiedStreams[i]);
        }
        std::vector<PabloAST *> NegativeStreams = bnc.AddModular(flipStreams, 1);
        std::vector<PabloAST *> UnsignedStreams = bnc.Select(inputStreams[bitsPerSample-1] /*sign*/, NegativeStreams, AmplifiedStreams);

        PabloAST *overflow = pb.createZeroes();
        for (int i = (int) bitsPerSample - 1;i < (int)UnsignedStreams.size() - 1;++i)
        {
            overflow = pb.createOr(overflow, UnsignedStreams[i]);
        }

        PabloAST *is_negative_overflow = pb.createAnd(inputStreams[bitsPerSample-1], overflow);
        PabloAST *is_positive_overflow = pb.createAnd(pb.createNot(inputStreams[bitsPerSample-1]), overflow);
        
        for (int i = 0; i < (int) bitsPerSample - 1;++i)
        {
            AmplifiedStreams[i] = pb.createSel(is_negative_overflow, pb.createZeroes(), AmplifiedStreams[i]);
            AmplifiedStreams[i] = pb.createSel(is_positive_overflow, pb.createOnes(), AmplifiedStreams[i]);
        }
        
        AmplifiedStreams[bitsPerSample-1] = inputStreams[bitsPerSample-1];

        writeOutputStreamSet("outputStreams", AmplifiedStreams);
    }

    Stereo2MonoPabloKernel::Stereo2MonoPabloKernel(LLVMTypeSystemInterface &b, StreamSet *const firstInputStreams, StreamSet *const secondInputStreams, StreamSet *const outputStreams)
        : PabloKernel(b, "Stereo2MonoPabloKernel_" + std::to_string(firstInputStreams->getNumElements()),
                      {Binding{"firstInputStreams", firstInputStreams}, Binding{"secondInputStreams", secondInputStreams}},
                      {Binding{"outputStreams", outputStreams}})
    {
        if (firstInputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("firstInputStreams: " + std::to_string(firstInputStreams->getNumElements()) + " != outputStreams: " + std::to_string(outputStreams->getNumElements()));
        }

        if (secondInputStreams->getNumElements() != firstInputStreams->getNumElements())
        {
            throw std::invalid_argument("firstInputStreams: " + std::to_string(firstInputStreams->getNumElements()) + " != secondInputStreams: " + std::to_string(secondInputStreams->getNumElements()));
        }
    }

    void Stereo2MonoPabloKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> firstInputStreams = getInputStreamSet("firstInputStreams");
        std::vector<PabloAST *> secondInputStreams = getInputStreamSet("secondInputStreams");

        const unsigned bitsPerSample = firstInputStreams.size() + 1;
        std::vector<PabloAST *> extendedFirstInputStreams = bnc.SignExtend(firstInputStreams, bitsPerSample);
        std::vector<PabloAST *> extendedSecondInputStreams = bnc.SignExtend(secondInputStreams, bitsPerSample);

        std::vector<PabloAST *> resultStreams = bnc.AddModular(extendedFirstInputStreams, extendedSecondInputStreams);
        Var *result = getOutputStreamVar("outputStreams");
        for (unsigned i = 1; i < resultStreams.size(); i++)
        {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i - 1)), resultStreams[i]);
        }
    }

    ConcatenateKernel::ConcatenateKernel(LLVMTypeSystemInterface & b, StreamSet *const firstInputStreams, StreamSet *const secondInputStreams, StreamSet *const outputStreams)
        : PabloKernel(b, "ConcatenateKernel_" + std::to_string(firstInputStreams->getNumElements()) + "_" + std::to_string(secondInputStreams->getNumElements()),
                      {Binding{"firstInputStreams", firstInputStreams}, Binding{"secondInputStreams", secondInputStreams}},
                      {Binding{"outputStreams", outputStreams}}),
          numFirstInputStreams(firstInputStreams->getNumElements()), numSecondInputStreams(secondInputStreams->getNumElements())
    {
        if (firstInputStreams->getNumElements() + secondInputStreams->getNumElements() != outputStreams->getNumElements())
        {
            throw std::invalid_argument("numOutputStreams(" + std::to_string(firstInputStreams->getNumElements()) + ") != numFirstInputStreams(" + std::to_string(secondInputStreams->getNumElements()) + ") + numSecondInputStreams(" + std::to_string(outputStreams->getNumElements()) + ")");
        }
    }

    void ConcatenateKernel::generatePabloMethod()
    {
        pablo::PabloBuilder pb(getEntryScope());
        BixNumCompiler bnc(pb);
        std::vector<PabloAST *> firstInputStreams = getInputStreamSet("firstInputStreams");
        std::vector<PabloAST *> secondInputStreams = getInputStreamSet("secondInputStreams");
        Var *result = getOutputStreamVar("outputStreams");
        for (unsigned i = 0; i < numFirstInputStreams; ++i)
        {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i)), firstInputStreams[i]);
        }

        for (unsigned i = 0; i < numSecondInputStreams; ++i)
        {
            pb.createAssign(pb.createExtract(result, pb.getInteger(i + numFirstInputStreams)), secondInputStreams[i]);
        }
    }
}

#undef NUM_HEADER_BYTES
