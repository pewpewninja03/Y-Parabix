#include "conv_filter_common.h"

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

using namespace kernel;
using namespace llvm;

namespace kernel::image::internal {
namespace {

uint64_t appendBytesToHash(uint64_t hash, const void * byteData, const std::size_t byteCount) {
    const auto * bytes = static_cast<const uint8_t *>(byteData);
    for (std::size_t index = 0; index < byteCount; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

Bindings convolutionScalarBindings(
    LLVMTypeSystemInterface & typeSystem, Scalar * inputPixels, Scalar * outputPixels, Scalar * borderLoopWeights, Scalar * borderLoopWeightCount
) {
    Bindings bindings{
        Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels}, Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels}
    };
    if (borderLoopWeights != nullptr) {
        bindings.emplace_back(typeSystem.getInt8PtrTy(), "borderLoopWeights", borderLoopWeights);
        bindings.emplace_back(typeSystem.getSizeTy(), "borderLoopWeightCount", borderLoopWeightCount);
    }
    return bindings;
}

class ConvolutionKernel final : public SegmentOrientedKernel {
   public:
    ConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * borderLoopWeights,
        Scalar * borderLoopWeightCount,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelHeight,
        const unsigned kernelWidth,
        std::vector<float> weightValues,
        const std::string & persistentIdentity
    )
        : SegmentOrientedKernel(
              typeSystem,
              "convolution_" + std::to_string(imageWidth) + "x" + std::to_string(imageHeight) + "_k" + std::to_string(kernelHeight) + "x"
                  + std::to_string(kernelWidth) + "_h" + std::to_string(computeWeightHash(weightValues)) + "_c" + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              convolutionScalarBindings(typeSystem, inputPixels, outputPixels, borderLoopWeights, borderLoopWeightCount),
              {},
              {}
          ),
          imageWidth(imageWidth),
          imageHeight(imageHeight),
          kernelHeight(kernelHeight),
          kernelWidth(kernelWidth),
          weights(std::move(weightValues)),
          useBorderTapLoop(borderLoopWeights != nullptr) {
        addAttribute(SideEffecting());
    }

   private:
    static uint64_t computeWeightHash(const std::vector<float> & weights) {
        uint64_t hash = 1469598103934665603ULL;
        hash =
            appendBytesToHash(hash, weights.data(), checkedMultiply(weights.size(), sizeof(float), "default kernel-name weight byte count overflow"));
        return hash;
    }

    Value * pixelByteOffset(KernelBuilder & builder, Value * row, Value * column, const unsigned channelIndex) const {
        Value * offset = builder.CreateMul(row, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channelIndex));
    }

    Value * loadBorderSamples(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned channelIndex,
        Value * kernelRow,
        Value * kernelColumn,
        const unsigned laneCount
    ) const {
        Value * samples = Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        const unsigned verticalRadius = kernelHeight / 2U;
        const unsigned horizontalRadius = kernelWidth / 2U;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(laneIndex));
            Value * paddedRow = builder.CreateAdd(row, kernelRow);
            Value * paddedColumn = builder.CreateAdd(column, kernelColumn);
            Value * isValidLane = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            isValidLane = builder.CreateAnd(isValidLane, builder.CreateICmpUGE(paddedRow, builder.getSize(verticalRadius)));
            isValidLane = builder.CreateAnd(isValidLane, builder.CreateICmpUGE(paddedColumn, builder.getSize(horizontalRadius)));
            isValidLane = builder.CreateAnd(
                isValidLane,
                builder.CreateICmpULT(paddedRow, builder.getSize(checkedAdd(imageHeight, verticalRadius, "default padded row extent overflow")))
            );
            isValidLane = builder.CreateAnd(
                isValidLane,
                builder.CreateICmpULT(
                    paddedColumn, builder.getSize(checkedAdd(imageWidth, horizontalRadius, "default padded column extent overflow"))
                )
            );

            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("border_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("border_join");
            builder.CreateCondBr(isValidLane, loadBlock, joinBlock);

            builder.SetInsertPoint(loadBlock);
            Value * sourceRow = builder.CreateSub(paddedRow, builder.getSize(verticalRadius));
            Value * sourceColumn = builder.CreateSub(paddedColumn, builder.getSize(horizontalRadius));
            Value * byteValue = builder.CreateLoad(
                builder.getInt8Ty(),
                builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, sourceRow, sourceColumn, channelIndex))
            );
            Value * sample = builder.CreateUIToFP(builder.CreateZExt(byteValue, builder.getInt32Ty()), builder.getFloatTy());
            builder.CreateBr(joinBlock);

            builder.SetInsertPoint(joinBlock);
            PHINode * selectedSample = builder.CreatePHI(builder.getFloatTy(), 2);
            selectedSample->addIncoming(ConstantFP::get(builder.getFloatTy(), 0.0F), zeroBlock);
            selectedSample->addIncoming(sample, loadBlock);
            samples = builder.CreateInsertElement(samples, selectedSample, builder.getInt32(laneIndex));
        }
        return samples;
    }

    std::array<Value *, ColorChannelCount> loadInteriorSamples(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned kernelRow,
        const unsigned kernelColumn,
        const unsigned laneCount
    ) const {
        const unsigned verticalRadius = kernelHeight / 2U;
        const unsigned horizontalRadius = kernelWidth / 2U;
        Value * sourceRow = builder.CreateSub(builder.CreateAdd(row, builder.getSize(kernelRow)), builder.getSize(verticalRadius));
        Value * sourceColumn =
            builder.CreateSub(builder.CreateAdd(columnGroupStart, builder.getSize(kernelColumn)), builder.getSize(horizontalRadius));
        Value * firstChannelOffset = pixelByteOffset(builder, sourceRow, sourceColumn, 0);
        auto * interleavedByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        Value * sourcePointer =
            builder.CreatePointerCast(builder.CreateGEP(builder.getInt8Ty(), inputPixels, firstChannelOffset), interleavedByteType->getPointerTo());
        LoadInst * interleavedBytes = builder.CreateLoad(interleavedByteType, sourcePointer);
        interleavedBytes->setAlignment(Align(1));

        std::array<Value *, ColorChannelCount> channelSamples;
        auto * integerVectorType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        auto * floatVectorType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
            SmallVector<int, 16> channelIndexes;
            for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
                channelIndexes.push_back(static_cast<int>(laneIndex * ColorChannelCount + channelIndex));
            }
            Value * channelBytes = builder.CreateShuffleVector(interleavedBytes, UndefValue::get(interleavedByteType), channelIndexes);
            channelSamples[channelIndex] = builder.CreateUIToFP(builder.CreateZExt(channelBytes, integerVectorType), floatVectorType);
        }
        return channelSamples;
    }

    Value * clampAndRound(KernelBuilder & builder, Value * accumulator, const unsigned laneCount) const {
        Value * zeroValue = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.0F));
        Value * maxByteValue = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 255.0F));
        Value * roundingOffset = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.5F));
        Value * lowerClamped = builder.CreateSelect(builder.CreateFCmpOLT(accumulator, zeroValue), zeroValue, accumulator);
        Value * bounded = builder.CreateSelect(builder.CreateFCmpOGT(lowerClamped, maxByteValue), maxByteValue, lowerClamped);
        auto * integerVectorType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        return builder.CreateFPToUI(builder.CreateFAdd(bounded, roundingOffset), integerVectorType);
    }

    Value * computeBorderChannel(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned channelIndex,
        Value * borderLoopWeights,
        Value * borderLoopWeightCount,
        const unsigned laneCount
    ) const {
        Value * accumulator = Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        Function * multiplyAddIntrinsic = Intrinsic::getDeclaration(builder.getModule(), Intrinsic::fmuladd, {accumulator->getType()});
        if (borderLoopWeights != nullptr) {
            BasicBlock * entryBlock = builder.GetInsertBlock();
            BasicBlock * tapLoop = builder.CreateBasicBlock("border_tap_loop");
            BasicBlock * tapBody = builder.CreateBasicBlock("border_tap_body");
            BasicBlock * done = builder.CreateBasicBlock("border_taps_done");
            builder.CreateBr(tapLoop);

            builder.SetInsertPoint(tapLoop);
            PHINode * tapIndex = builder.CreatePHI(builder.getSizeTy(), 2);
            tapIndex->addIncoming(builder.getSize(0), entryBlock);
            PHINode * runningAccumulator = builder.CreatePHI(accumulator->getType(), 2);
            runningAccumulator->addIncoming(accumulator, entryBlock);
            builder.CreateCondBr(builder.CreateICmpULT(tapIndex, borderLoopWeightCount), tapBody, done);

            builder.SetInsertPoint(tapBody);
            Value * kernelRow = builder.CreateUDiv(tapIndex, builder.getSize(kernelWidth));
            Value * kernelColumn = builder.CreateURem(tapIndex, builder.getSize(kernelWidth));
            Value * samples = loadBorderSamples(builder, inputPixels, row, columnGroupStart, channelIndex, kernelRow, kernelColumn, laneCount);
            Value * weightPointer = builder.CreateGEP(builder.getFloatTy(), borderLoopWeights, tapIndex);
            LoadInst * weightValue = builder.CreateLoad(builder.getFloatTy(), weightPointer);
            weightValue->setAlignment(Align(sizeof(float)));
            Value * weight =
                builder.CreateBitCast(builder.simd_fill(32, builder.CreateBitCast(weightValue, builder.getInt32Ty())), accumulator->getType());
            Value * nextAccumulator =
                builder.CreateCall(multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples, weight, runningAccumulator});
            Value * nextTap = builder.CreateAdd(tapIndex, builder.getSize(1));
            BasicBlock * loopBack = builder.GetInsertBlock();
            builder.CreateBr(tapLoop);
            tapIndex->addIncoming(nextTap, loopBack);
            runningAccumulator->addIncoming(nextAccumulator, loopBack);

            builder.SetInsertPoint(done);
            return clampAndRound(builder, runningAccumulator, laneCount);
        }
        for (unsigned kernelRow = 0; kernelRow < kernelHeight; ++kernelRow) {
            for (unsigned kernelColumn = 0; kernelColumn < kernelWidth; ++kernelColumn) {
                const std::size_t weightIndex = static_cast<std::size_t>(kernelRow) * kernelWidth + kernelColumn;
                if (weights[weightIndex] == 0.0F)
                    continue;
                Value * samples = loadBorderSamples(
                    builder, inputPixels, row, columnGroupStart, channelIndex, builder.getSize(kernelRow), builder.getSize(kernelColumn), laneCount
                );
                Value * weight = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), weights[weightIndex]));
                accumulator = builder.CreateCall(multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples, weight, accumulator});
            }
        }
        return clampAndRound(builder, accumulator, laneCount);
    }

    std::array<Value *, ColorChannelCount> computeInteriorChannels(
        KernelBuilder & builder, Value * inputPixels, Value * row, Value * columnGroupStart, const unsigned laneCount
    ) const {
        auto * floatVectorType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        std::array<Value *, ColorChannelCount> accumulators;
        accumulators.fill(Constant::getNullValue(floatVectorType));
        Function * multiplyAddIntrinsic = Intrinsic::getDeclaration(builder.getModule(), Intrinsic::fmuladd, {floatVectorType});
        for (unsigned kernelRow = 0; kernelRow < kernelHeight; ++kernelRow) {
            for (unsigned kernelColumn = 0; kernelColumn < kernelWidth; ++kernelColumn) {
                const std::size_t weightIndex = static_cast<std::size_t>(kernelRow) * kernelWidth + kernelColumn;
                if (weights[weightIndex] == 0.0F)
                    continue;
                const auto samples = loadInteriorSamples(builder, inputPixels, row, columnGroupStart, kernelRow, kernelColumn, laneCount);
                Value * weight = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), weights[weightIndex]));
                for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
                    accumulators[channelIndex] = builder.CreateCall(
                        multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples[channelIndex], weight, accumulators[channelIndex]}
                    );
                }
            }
        }
        for (Value *& accumulator : accumulators) {
            accumulator = clampAndRound(builder, accumulator, laneCount);
        }
        return accumulators;
    }

    void storeCheckedChannel(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned channelIndex,
        Value * roundedValues,
        const unsigned laneCount
    ) const {
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(laneIndex));
            Value * outputByte = builder.CreateTrunc(builder.CreateExtractElement(roundedValues, builder.getInt32(laneIndex)), builder.getInt8Ty());
            Value * isValidLane = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            BasicBlock * storeBlock = builder.CreateBasicBlock("store_checked");
            BasicBlock * joinBlock = builder.CreateBasicBlock("store_join");
            builder.CreateCondBr(isValidLane, storeBlock, joinBlock);
            builder.SetInsertPoint(storeBlock);
            builder.CreateStore(
                outputByte, builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, column, channelIndex))
            );
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
        }
    }

    void storeInteriorPixels(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const std::array<Value *, ColorChannelCount> & roundedChannels,
        const unsigned laneCount
    ) const {
        auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        auto * interleavedByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channelBytes;
        for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
            channelBytes[channelIndex] = builder.CreateTrunc(roundedChannels[channelIndex], channelByteType);
        }
        SmallVector<int, 16> redGreenIndexes;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            redGreenIndexes.push_back(static_cast<int>(laneIndex));
            redGreenIndexes.push_back(static_cast<int>(laneCount + laneIndex));
        }
        Value * redGreenBytes = builder.CreateShuffleVector(channelBytes[0], channelBytes[1], redGreenIndexes);
        SmallVector<int, 16> blueIndexes;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            blueIndexes.push_back(static_cast<int>(laneIndex));
        }
        blueIndexes.resize(laneCount * 2U, -1);
        Value * blueBytes = builder.CreateShuffleVector(channelBytes[2], UndefValue::get(channelByteType), blueIndexes);
        SmallVector<int, 32> interleavedIndexes;
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            interleavedIndexes.push_back(static_cast<int>(laneIndex * 2U));
            interleavedIndexes.push_back(static_cast<int>(laneIndex * 2U + 1U));
            interleavedIndexes.push_back(static_cast<int>(laneCount * 2U + laneIndex));
        }
        Value * interleavedBytes = builder.CreateShuffleVector(redGreenBytes, blueBytes, interleavedIndexes);
        Value * outputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)),
            interleavedByteType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(interleavedBytes, outputPointer);
        store->setAlignment(Align(1));
    }

    void generateDoSegmentMethod(KernelBuilder & builder) final {
        Value * inputPixels = builder.getScalarField("inputPixels");
        Value * outputPixels = builder.getScalarField("outputPixels");
        Value * borderLoopWeights = useBorderTapLoop ? builder.getScalarField("borderLoopWeights") : nullptr;
        Value * borderLoopWeightCount = useBorderTapLoop ? builder.getScalarField("borderLoopWeightCount") : nullptr;
        const unsigned laneCount = builder.getBitBlockWidth() / 32U;
        const unsigned verticalRadius = kernelHeight / 2U;
        const unsigned horizontalRadius = kernelWidth / 2U;

        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * rowLoop = builder.CreateBasicBlock("row_loop");
        BasicBlock * done = builder.CreateBasicBlock("done");
        builder.CreateBr(rowLoop);

        builder.SetInsertPoint(rowLoop);
        PHINode * row = builder.CreatePHI(builder.getSizeTy(), 2);
        row->addIncoming(builder.getSize(0), entry);

        BasicBlock * columnLoop = builder.CreateBasicBlock("column_loop");
        BasicBlock * doneColumns = builder.CreateBasicBlock("done_columns");
        builder.CreateBr(columnLoop);

        builder.SetInsertPoint(columnLoop);
        PHINode * columnGroupStart = builder.CreatePHI(builder.getSizeTy(), 2);
        columnGroupStart->addIncoming(builder.getSize(0), rowLoop);

        Value * isInteriorGroup = builder.CreateICmpUGE(row, builder.getSize(verticalRadius));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup, builder.CreateICmpULT(builder.CreateAdd(row, builder.getSize(verticalRadius)), builder.getSize(imageHeight))
        );
        isInteriorGroup = builder.CreateAnd(isInteriorGroup, builder.CreateICmpUGE(columnGroupStart, builder.getSize(horizontalRadius)));
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup, builder.CreateICmpULT(builder.CreateAdd(columnGroupStart, builder.getSize(laneCount - 1U)), builder.getSize(imageWidth))
        );
        isInteriorGroup = builder.CreateAnd(
            isInteriorGroup,
            builder.CreateICmpULT(
                builder.CreateAdd(builder.CreateAdd(columnGroupStart, builder.getSize(laneCount - 1U)), builder.getSize(horizontalRadius)),
                builder.getSize(imageWidth)
            )
        );

        BasicBlock * interiorBlock = builder.CreateBasicBlock("interior");
        BasicBlock * borderBlock = builder.CreateBasicBlock("border");
        BasicBlock * nextColumnGroup = builder.CreateBasicBlock("next_column_group");
        builder.CreateCondBr(isInteriorGroup, interiorBlock, borderBlock);

        builder.SetInsertPoint(interiorBlock);
        if (kernelHeight == 1U && kernelWidth == 1U && weights[0] == 1.0F) {
            auto * interleavedByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
            Value * inputPointer = builder.CreatePointerCast(
                builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)),
                interleavedByteType->getPointerTo()
            );
            LoadInst * interleavedBytes = builder.CreateLoad(interleavedByteType, inputPointer);
            interleavedBytes->setAlignment(Align(1));
            Value * outputPointer = builder.CreatePointerCast(
                builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)),
                interleavedByteType->getPointerTo()
            );
            StoreInst * store = builder.CreateStore(interleavedBytes, outputPointer);
            store->setAlignment(Align(1));
        } else {
            const auto roundedChannels = computeInteriorChannels(builder, inputPixels, row, columnGroupStart, laneCount);
            storeInteriorPixels(builder, outputPixels, row, columnGroupStart, roundedChannels, laneCount);
        }
        builder.CreateBr(nextColumnGroup);

        builder.SetInsertPoint(borderBlock);
        for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
            Value * roundedChannel = computeBorderChannel(
                builder,
                inputPixels,
                row,
                columnGroupStart,
                channelIndex,
                useBorderTapLoop ? borderLoopWeights : nullptr,
                borderLoopWeightCount,
                laneCount
            );
            storeCheckedChannel(builder, outputPixels, row, columnGroupStart, channelIndex, roundedChannel, laneCount);
        }
        builder.CreateBr(nextColumnGroup);

        builder.SetInsertPoint(nextColumnGroup);

        Value * nextColumn = builder.CreateAdd(columnGroupStart, builder.getSize(laneCount));
        columnGroupStart->addIncoming(nextColumn, builder.GetInsertBlock());
        builder.CreateCondBr(builder.CreateICmpULT(nextColumn, builder.getSize(imageWidth)), columnLoop, doneColumns);

        builder.SetInsertPoint(doneColumns);
        Value * nextRow = builder.CreateAdd(row, builder.getSize(1));
        row->addIncoming(nextRow, doneColumns);
        builder.CreateCondBr(builder.CreateICmpULT(nextRow, builder.getSize(imageHeight)), rowLoop, done);
        builder.SetInsertPoint(done);
    }

    const unsigned imageWidth;
    const unsigned imageHeight;
    const unsigned kernelHeight;
    const unsigned kernelWidth;
    const std::vector<float> weights;
    const bool useBorderTapLoop;
};

class DefaultFilterImplementation final : public CompiledFilterImplementation {
   public:
    DefaultFilterImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        std::vector<float> weights,
        const std::string & persistentIdentity
    )
        : CompiledFilterImplementation(ConvFilterMode::Default, imageWidth, imageHeight, checkedImageByteCount(imageWidth, imageHeight), 0, 1),
          driver(std::move(cpuDriver)),
          kernelWeights(std::move(weights)),
          useBorderTapLoop(kernelWeights.size() >= 25U && std::all_of(kernelWeights.begin(), kernelWeights.end(), [](const float weight) {
                               return weight != 0.0F;
                           })) {
        if (useBorderTapLoop) {
            auto pipeline = CreatePipeline(
                *driver,
                Input<const uint8_t *>("inputPixels"),
                Input<uint8_t *>("outputPixels"),
                Input<const float *>("borderLoopWeights"),
                Input<std::size_t>("borderLoopWeightCount"),
                Input<uint8_t *>("triggerBuffer"),
                Input<std::size_t>("triggerLength")
            );
            StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
            pipeline.CreateKernelCall<MemorySourceKernel>(
                pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
            );
            pipeline.CreateKernelCall<ConvolutionKernel>(
                triggerStream,
                pipeline.getInputScalar("inputPixels"),
                pipeline.getInputScalar("outputPixels"),
                pipeline.getInputScalar("borderLoopWeights"),
                pipeline.getInputScalar("borderLoopWeightCount"),
                imageWidth,
                imageHeight,
                kernelHeight,
                kernelWidth,
                kernelWeights,
                persistentIdentity
            );
            borderLoopFunction = pipeline.compile();
        } else {
            auto pipeline = CreatePipeline(
                *driver,
                Input<const uint8_t *>("inputPixels"),
                Input<uint8_t *>("outputPixels"),
                Input<uint8_t *>("triggerBuffer"),
                Input<std::size_t>("triggerLength")
            );
            StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
            pipeline.CreateKernelCall<MemorySourceKernel>(
                pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
            );
            pipeline.CreateKernelCall<ConvolutionKernel>(
                triggerStream,
                pipeline.getInputScalar("inputPixels"),
                pipeline.getInputScalar("outputPixels"),
                nullptr,
                nullptr,
                imageWidth,
                imageHeight,
                kernelHeight,
                kernelWidth,
                kernelWeights,
                persistentIdentity
            );
            fixedWeightFunction = pipeline.compile();
        }
    }

   private:
    void invoke(const uint8_t * inputPixels, uint8_t * outputPixels, void *) const noexcept final {
        uint8_t triggerByte = 0;
        if (useBorderTapLoop) {
            borderLoopFunction(inputPixels, outputPixels, kernelWeights.data(), kernelWeights.size(), &triggerByte, 1U);
        } else {
            fixedWeightFunction(inputPixels, outputPixels, &triggerByte, 1U);
        }
    }

    std::unique_ptr<CPUDriver> driver;
    const std::vector<float> kernelWeights;
    const bool useBorderTapLoop;
    void (*fixedWeightFunction)(const uint8_t *, uint8_t *, uint8_t *, std::size_t) = nullptr;
    void (*borderLoopFunction)(const uint8_t *, uint8_t *, const float *, std::size_t, uint8_t *, std::size_t) = nullptr;
};

}  // namespace

std::shared_ptr<const CompiledFilterImplementation> compileDefaultFilter(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    std::vector<float> weights,
    std::string persistentIdentity
) {
    return std::make_shared<DefaultFilterImplementation>(
        std::move(driver), imageWidth, imageHeight, kernelWidth, kernelHeight, std::move(weights), persistentIdentity
    );
}

}  // namespace kernel::image::internal
