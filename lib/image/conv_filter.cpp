#include <image/conv_filter.h>

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace kernel;
using namespace llvm;

namespace kernel::image {
namespace {

constexpr unsigned ColorChannelCount = 3;

uint64_t appendBytesToHash(uint64_t hash, const void *data, std::size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t hashConfiguration(const ConvFilterConfig &config) {
    uint64_t hash = 1469598103934665603ULL;
    hash = appendBytesToHash(hash, &config.mode, sizeof(config.mode));
    hash = appendBytesToHash(hash, &config.width, sizeof(config.width));
    hash = appendBytesToHash(hash, &config.height, sizeof(config.height));
    hash = appendBytesToHash(hash, &config.kernelHeight, sizeof(config.kernelHeight));
    hash = appendBytesToHash(hash, &config.kernelWidth, sizeof(config.kernelWidth));
    hash = appendBytesToHash(hash, &config.weightCount, sizeof(config.weightCount));
    hash = appendBytesToHash(hash, config.weights,
                             static_cast<std::size_t>(config.weightCount) * sizeof(float));
    return hash;
}

class ConvolutionKernel final : public SegmentOrientedKernel {
  public:
    ConvolutionKernel(LLVMTypeSystemInterface &typeSystem, StreamSet *triggerStream,
                      Scalar *inputPixels, Scalar *outputPixels, const unsigned width,
                      const unsigned height, const unsigned kernelHeight,
                      const unsigned kernelWidth, std::vector<float> weights)
        : SegmentOrientedKernel(typeSystem,
                                "convolution_" + std::to_string(width) + "x" +
                                    std::to_string(height) + "_k" + std::to_string(kernelHeight) +
                                    "x" + std::to_string(kernelWidth) + "_h" +
                                    std::to_string(hashWeightValues(weights)),
                                {Binding{"triggerStream", triggerStream}}, {},
                                {Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels},
                                 Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels}},
                                {}, {}),
          imageWidth(width), imageHeight(height), kernelRowCount(kernelHeight),
          kernelColumnCount(kernelWidth), convolutionWeights(std::move(weights)) {
        addAttribute(SideEffecting());
    }

  private:
    static uint64_t hashWeightValues(const std::vector<float> &weights) {
        uint64_t hash = 1469598103934665603ULL;
        hash = appendBytesToHash(hash, weights.data(), weights.size() * sizeof(float));
        return hash;
    }

    Value *pixelByteOffset(KernelBuilder &builder, Value *row, Value *column,
                           const unsigned channelIndex) const {
        Value *offset = builder.CreateMul(row, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channelIndex));
    }

    Value *loadBorderSamples(KernelBuilder &builder, Value *inputPixels, Value *row,
                             Value *columnGroupStart, const unsigned channelIndex,
                             const unsigned kernelRow, const unsigned kernelColumn,
                             const unsigned laneCount) const {
        Value *samples =
            Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        const unsigned verticalRadius = kernelRowCount / 2U;
        const unsigned horizontalRadius = kernelColumnCount / 2U;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            Value *column = builder.CreateAdd(columnGroupStart, builder.getSize(laneIndex));
            Value *paddedRow = builder.CreateAdd(row, builder.getSize(kernelRow));
            Value *paddedColumn = builder.CreateAdd(column, builder.getSize(kernelColumn));
            Value *isValidLane = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            isValidLane = builder.CreateAnd(
                isValidLane, builder.CreateICmpUGE(paddedRow, builder.getSize(verticalRadius)));
            isValidLane = builder.CreateAnd(
                isValidLane,
                builder.CreateICmpUGE(paddedColumn, builder.getSize(horizontalRadius)));
            isValidLane = builder.CreateAnd(
                isValidLane,
                builder.CreateICmpULT(paddedRow, builder.getSize(imageHeight + verticalRadius)));
            isValidLane = builder.CreateAnd(
                isValidLane, builder.CreateICmpULT(paddedColumn,
                                                   builder.getSize(imageWidth + horizontalRadius)));

            BasicBlock *zeroBlock = builder.GetInsertBlock();
            BasicBlock *loadBlock = builder.CreateBasicBlock("border_load");
            BasicBlock *joinBlock = builder.CreateBasicBlock("border_join");
            builder.CreateCondBr(isValidLane, loadBlock, joinBlock);

            builder.SetInsertPoint(loadBlock);
            Value *sourceRow = builder.CreateSub(paddedRow, builder.getSize(verticalRadius));
            Value *sourceColumn =
                builder.CreateSub(paddedColumn, builder.getSize(horizontalRadius));
            Value *byteValue = builder.CreateLoad(
                builder.getInt8Ty(),
                builder.CreateGEP(builder.getInt8Ty(), inputPixels,
                                  pixelByteOffset(builder, sourceRow, sourceColumn, channelIndex)));
            Value *sample = builder.CreateUIToFP(
                builder.CreateZExt(byteValue, builder.getInt32Ty()), builder.getFloatTy());
            builder.CreateBr(joinBlock);

            builder.SetInsertPoint(joinBlock);
            PHINode *selectedSample = builder.CreatePHI(builder.getFloatTy(), 2);
            selectedSample->addIncoming(ConstantFP::get(builder.getFloatTy(), 0.0F), zeroBlock);
            selectedSample->addIncoming(sample, loadBlock);
            samples =
                builder.CreateInsertElement(samples, selectedSample, builder.getInt32(laneIndex));
        }
        return samples;
    }

    Value *loadInteriorSamples(KernelBuilder &builder, Value *inputPixels, Value *row,
                               Value *columnGroupStart, const unsigned channelIndex,
                               const unsigned kernelRow, const unsigned kernelColumn,
                               const unsigned laneCount) const {
        const unsigned verticalRadius = kernelRowCount / 2U;
        const unsigned horizontalRadius = kernelColumnCount / 2U;
        Value *sourceRow = builder.CreateSub(builder.CreateAdd(row, builder.getSize(kernelRow)),
                                             builder.getSize(verticalRadius));
        Value *sourceColumn =
            builder.CreateSub(builder.CreateAdd(columnGroupStart, builder.getSize(kernelColumn)),
                              builder.getSize(horizontalRadius));
        Value *firstChannelOffset = pixelByteOffset(builder, sourceRow, sourceColumn, channelIndex);
        auto *interleavedByteType =
            FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        Value *sourcePointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), inputPixels, firstChannelOffset),
            interleavedByteType->getPointerTo());
        LoadInst *interleavedBytes = builder.CreateLoad(interleavedByteType, sourcePointer);
        interleavedBytes->setAlignment(Align(1));

        SmallVector<int, 16> channelIndexes;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            channelIndexes.push_back(static_cast<int>(laneIndex * ColorChannelCount));
        }
        Value *channelBytes = builder.CreateShuffleVector(
            interleavedBytes, UndefValue::get(interleavedByteType), channelIndexes);
        auto *integerVectorType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        return builder.CreateUIToFP(builder.CreateZExt(channelBytes, integerVectorType),
                                    FixedVectorType::get(builder.getFloatTy(), laneCount));
    }

    Value *accumulateConvolution(KernelBuilder &builder, Value *inputPixels, Value *row,
                                 Value *columnGroupStart, const unsigned channelIndex,
                                 const bool usesBorderLoads, const unsigned laneCount) const {
        Value *accumulator =
            Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        Function *multiplyAddIntrinsic = Intrinsic::getDeclaration(
            builder.getModule(), Intrinsic::fmuladd, {accumulator->getType()});
        for (unsigned kernelRow = 0; kernelRow < kernelRowCount; ++kernelRow) {
            for (unsigned kernelColumn = 0; kernelColumn < kernelColumnCount; ++kernelColumn) {
                Value *samples =
                    usesBorderLoads
                        ? loadBorderSamples(builder, inputPixels, row, columnGroupStart,
                                            channelIndex, kernelRow, kernelColumn, laneCount)
                        : loadInteriorSamples(builder, inputPixels, row, columnGroupStart,
                                              channelIndex, kernelRow, kernelColumn, laneCount);
                const std::size_t weightIndex =
                    static_cast<std::size_t>(kernelRow) * kernelColumnCount + kernelColumn;
                Value *weight =
                    builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(),
                                                                convolutionWeights[weightIndex]));
                accumulator =
                    builder.CreateCall(multiplyAddIntrinsic->getFunctionType(),
                                       multiplyAddIntrinsic, {samples, weight, accumulator});
            }
        }
        Value *zeroValue = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.0F));
        Value *maxByteValue =
            builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 255.0F));
        Value *roundingOffset =
            builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.5F));
        Value *lowerClamped = builder.CreateSelect(builder.CreateFCmpOLT(accumulator, zeroValue),
                                                   zeroValue, accumulator);
        Value *bounded = builder.CreateSelect(builder.CreateFCmpOGT(lowerClamped, maxByteValue),
                                              maxByteValue, lowerClamped);
        auto *integerVectorType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        return builder.CreateFPToUI(builder.CreateFAdd(bounded, roundingOffset), integerVectorType);
    }

    void storeOutputSamples(KernelBuilder &builder, Value *outputPixels, Value *row,
                            Value *columnGroupStart, const unsigned channelIndex,
                            Value *roundedValues, const bool isRightEdgeGroup,
                            const unsigned laneCount) const {
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            Value *column = builder.CreateAdd(columnGroupStart, builder.getSize(laneIndex));
            Value *outputByte = builder.CreateTrunc(
                builder.CreateExtractElement(roundedValues, builder.getInt32(laneIndex)),
                builder.getInt8Ty());
            if (!isRightEdgeGroup) {
                builder.CreateStore(
                    outputByte,
                    builder.CreateGEP(builder.getInt8Ty(), outputPixels,
                                      pixelByteOffset(builder, row, column, channelIndex)));
            } else {
                Value *isValidLane = builder.CreateICmpULT(column, builder.getSize(imageWidth));
                BasicBlock *storeBlock = builder.CreateBasicBlock("store_checked");
                BasicBlock *joinBlock = builder.CreateBasicBlock("store_join");
                builder.CreateCondBr(isValidLane, storeBlock, joinBlock);
                builder.SetInsertPoint(storeBlock);
                builder.CreateStore(
                    outputByte,
                    builder.CreateGEP(builder.getInt8Ty(), outputPixels,
                                      pixelByteOffset(builder, row, column, channelIndex)));
                builder.CreateBr(joinBlock);
                builder.SetInsertPoint(joinBlock);
            }
        }
    }

    void generateDoSegmentMethod(KernelBuilder &builder) final {
        Value *inputPixels = builder.getScalarField("inputPixels");
        Value *outputPixels = builder.getScalarField("outputPixels");
        const unsigned laneCount = builder.getBitBlockWidth() / 32U;
        const unsigned verticalRadius = kernelRowCount / 2U;
        const unsigned horizontalRadius = kernelColumnCount / 2U;

        BasicBlock *entry = builder.GetInsertBlock();
        BasicBlock *rowLoop = builder.CreateBasicBlock("row_loop");
        BasicBlock *done = builder.CreateBasicBlock("done");
        builder.CreateBr(rowLoop);

        builder.SetInsertPoint(rowLoop);
        PHINode *row = builder.CreatePHI(builder.getSizeTy(), 2);
        row->addIncoming(builder.getSize(0), entry);

        BasicBlock *columnLoop = builder.CreateBasicBlock("column_loop");
        BasicBlock *doneColumns = builder.CreateBasicBlock("done_columns");
        builder.CreateBr(columnLoop);

        builder.SetInsertPoint(columnLoop);
        PHINode *columnGroupStart = builder.CreatePHI(builder.getSizeTy(), 2);
        columnGroupStart->addIncoming(builder.getSize(0), rowLoop);

        Value *isInteriorGroup = builder.CreateICmpUGE(row, builder.getSize(verticalRadius));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup,
            builder.CreateICmpULT(builder.CreateAdd(row, builder.getSize(verticalRadius)),
                                  builder.getSize(imageHeight)));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup,
            builder.CreateICmpUGE(columnGroupStart, builder.getSize(horizontalRadius)));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup,
            builder.CreateICmpULT(
                builder.CreateAdd(columnGroupStart, builder.getSize(laneCount - 1U)),
                builder.getSize(imageWidth)));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup,
            builder.CreateICmpULT(
                builder.CreateAdd(
                    builder.CreateAdd(columnGroupStart, builder.getSize(laneCount - 1U)),
                    builder.getSize(horizontalRadius)),
                builder.getSize(imageWidth)));

        for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
            BasicBlock *interiorBlock = builder.CreateBasicBlock("interior");
            BasicBlock *borderBlock = builder.CreateBasicBlock("border");
            BasicBlock *nextChannel = builder.CreateBasicBlock("next_channel");
            builder.CreateCondBr(isInteriorGroup, interiorBlock, borderBlock);

            builder.SetInsertPoint(interiorBlock);
            Value *interiorValues = accumulateConvolution(
                builder, inputPixels, row, columnGroupStart, channelIndex, false, laneCount);
            storeOutputSamples(builder, outputPixels, row, columnGroupStart, channelIndex,
                               interiorValues, false, laneCount);
            builder.CreateBr(nextChannel);

            builder.SetInsertPoint(borderBlock);
            Value *borderValues = accumulateConvolution(builder, inputPixels, row, columnGroupStart,
                                                        channelIndex, true, laneCount);
            storeOutputSamples(builder, outputPixels, row, columnGroupStart, channelIndex,
                               borderValues, true, laneCount);
            builder.CreateBr(nextChannel);

            builder.SetInsertPoint(nextChannel);
        }

        Value *nextColumn = builder.CreateAdd(columnGroupStart, builder.getSize(laneCount));
        columnGroupStart->addIncoming(nextColumn, builder.GetInsertBlock());
        builder.CreateCondBr(builder.CreateICmpULT(nextColumn, builder.getSize(imageWidth)),
                             columnLoop, doneColumns);

        builder.SetInsertPoint(doneColumns);
        Value *nextRow = builder.CreateAdd(row, builder.getSize(1));
        row->addIncoming(nextRow, doneColumns);
        builder.CreateCondBr(builder.CreateICmpULT(nextRow, builder.getSize(imageHeight)), rowLoop,
                             done);
        builder.SetInsertPoint(done);
    }

    const unsigned imageWidth;
    const unsigned imageHeight;
    const unsigned kernelRowCount;
    const unsigned kernelColumnCount;
    const std::vector<float> convolutionWeights;
};

struct ConvolutionPipeline {
    explicit ConvolutionPipeline(const ConvFilterConfig &config) : driver("convolution") {
        auto pipeline = CreatePipeline(
            driver, Input<const uint8_t *>("inputPixels"), Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("triggerBuffer"), Input<std::size_t>("triggerLength"));
        StreamSet *triggerStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<MemorySourceKernel>(pipeline.getInputScalar("triggerBuffer"),
                                                      pipeline.getInputScalar("triggerLength"),
                                                      triggerStream);
        std::vector<float> weights(config.weights, config.weights + config.weightCount);
        pipeline.CreateKernelCall<ConvolutionKernel>(
            triggerStream, pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"), config.width, config.height,
            config.kernelHeight, config.kernelWidth, weights);
        function = pipeline.compile();
    }

    void run(const uint8_t *inputPixels, uint8_t *outputPixels) const {
        uint8_t triggerByte = 0;
        function(inputPixels, outputPixels, &triggerByte, 1);
    }

    CPUDriver driver;
    void (*function)(const uint8_t *, uint8_t *, uint8_t *, std::size_t) = nullptr;
};

ConvolutionPipeline &compiledConvolutionPipeline(const ConvFilterConfig &config) {
    static std::mutex mutex;
    static std::unordered_map<uint64_t, std::unique_ptr<ConvolutionPipeline>> cache;
    const uint64_t key = hashConfiguration(config);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = cache.find(key);
    if (found != cache.end()) {
        return *found->second;
    }
    auto pipeline = std::make_unique<ConvolutionPipeline>(config);
    ConvolutionPipeline &result = *pipeline;
    cache.emplace(key, std::move(pipeline));
    return result;
}

} // namespace

bool applyConvFilter(const uint8_t *inputPixels, uint8_t *outputPixels,
                     const ConvFilterConfig &config) {
    const auto weightCount = static_cast<std::uint64_t>(config.kernelHeight) * config.kernelWidth;
    if (config.mode != ConvFilterMode::Default || inputPixels == nullptr ||
        outputPixels == nullptr || config.width == 0 || config.height == 0 ||
        (config.kernelHeight & 1U) == 0 || (config.kernelWidth & 1U) == 0 ||
        config.weights == nullptr || config.weightCount != weightCount) {
        return false;
    }
    compiledConvolutionPipeline(config).run(inputPixels, outputPixels);
    return true;
}

} // namespace kernel::image
