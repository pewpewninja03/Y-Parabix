#include "conv_filter_common.h"
#include "conv_filter_default_illustration.h"

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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kernel;
using namespace llvm;

namespace kernel::image::internal {
namespace {

Bindings convolutionScalarBindings(
    LLVMTypeSystemInterface & typeSystem,
    Scalar * inputPixels,
    Scalar * outputPixels,
    Scalar * borderLoopWeights,
    Scalar * borderLoopWeightCount,
    const IllustrationScalars illustration
) {
    Bindings bindings{
        Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels}, Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels}
    };
    if (borderLoopWeights != nullptr) {
        bindings.emplace_back(typeSystem.getInt8PtrTy(), "borderLoopWeights", borderLoopWeights);
        bindings.emplace_back(typeSystem.getSizeTy(), "borderLoopWeightCount", borderLoopWeightCount);
    }
    if (illustration) {
        bindings.emplace_back(typeSystem.getInt8PtrTy(), "captureContext", illustration.captureContext);
        bindings.emplace_back(typeSystem.getInt32Ty(), "selectionKind", illustration.selectionKind);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionRow", illustration.selectionRow);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionColumn", illustration.selectionColumn);
    }
    return bindings;
}

bool usesBorderTapLoop(const std::vector<float> & weights) {
    return weights.size() >= 25U && std::all_of(weights.begin(), weights.end(), [](const float weight) { return weight != 0.0F; });
}

class DefaultIllustrationEmitter {
   public:
    DefaultIllustrationEmitter(
        KernelBuilder & builder,
        Value * captureContext,
        Value * selectionKind,
        Value * selectionRow,
        Value * selectionColumn,
        const unsigned imageWidth,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const std::vector<float> & weights,
        const bool useBorderTapLoop,
        const unsigned laneCount
    )
        : builder(builder),
          captureContext(captureContext),
          selectionKind(selectionKind),
          selectionRow(selectionRow),
          selectionColumn(selectionColumn),
          imageWidth(imageWidth),
          kernelWidth(kernelWidth),
          kernelHeight(kernelHeight),
          weights(weights),
          useBorderTapLoop(useBorderTapLoop),
          laneCount(laneCount) {
        captureFunction = builder.getModule()->getFunction("captureDefaultConvFilterValue");
        assert(captureFunction != nullptr);
    }

    Value * selectedGroup(Value * row, Value * columnGroupStart) const {
        Value * selected = builder.CreateAnd(isOutputSelection(), outputSelectionMatchesGroup(row, columnGroupStart));
        Value * inputGroup = builder.getInt1(false);
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(position));
            Value * positionSelected =
                builder.CreateAnd(builder.CreateICmpULT(column, builder.getSize(imageWidth)), inputSelectionAffectsPosition(row, column));
            inputGroup = builder.CreateOr(inputGroup, positionSelected);
        }
        return builder.CreateOr(selected, builder.CreateAnd(isInputSelection(), inputGroup));
    }

    Value * selectedTap(Value * row, Value * columnGroupStart, Value * kernelRow, Value * kernelColumn) const {
        Value * selected = builder.CreateAnd(isOutputSelection(), outputSelectionMatchesGroup(row, columnGroupStart));
        Value * inputTap = builder.getInt1(false);
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(position));
            Value * relationship = inputSelectionMatchesTap(row, column, kernelRow, kernelColumn);
            relationship = builder.CreateAnd(relationship, builder.CreateICmpULT(column, builder.getSize(imageWidth)));
            inputTap = builder.CreateOr(inputTap, relationship);
        }
        return builder.CreateOr(selected, builder.CreateAnd(isInputSelection(), inputTap));
    }

    void captureGroup(Value * condition, Value * row, Value * columnGroupStart) const {
        capture(
            condition,
            DefaultIllustrationEvent::OutputColumns,
            outputColumns(columnGroupStart),
            builder.getSize(0),
            builder.getSize(0),
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        capture(
            condition,
            DefaultIllustrationEvent::SelectedPositions,
            selectedPositionMask(row, columnGroupStart),
            builder.getSize(0),
            builder.getSize(0),
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        capture(
            condition,
            DefaultIllustrationEvent::OutputValid,
            outputMask(columnGroupStart),
            builder.getSize(0),
            builder.getSize(0),
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
    }

    void captureInteriorTap(
        Value * condition,
        Value * row,
        Value * columnGroupStart,
        const unsigned kernelRow,
        const unsigned kernelColumn,
        Value * packedInput,
        const std::array<Value *, ColorChannelCount> & samples,
        Value * weight,
        const std::array<Value *, ColorChannelCount> & accumulators
    ) const {
        Value * kernelRowValue = builder.getSize(kernelRow);
        Value * kernelColumnValue = builder.getSize(kernelColumn);
        captureTapCoordinates(
            condition,
            row,
            columnGroupStart,
            kernelRowValue,
            kernelColumnValue,
            Constant::getAllOnesValue(FixedVectorType::get(builder.getInt8Ty(), laneCount))
        );
        capture(
            condition,
            DefaultIllustrationEvent::PackedInput,
            packedInput,
            kernelRowValue,
            kernelColumnValue,
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            capture(condition, DefaultIllustrationEvent::Sample, samples[channel], kernelRowValue, kernelColumnValue, channel, row, columnGroupStart);
        }
        capture(
            condition,
            DefaultIllustrationEvent::Weight,
            weight,
            kernelRowValue,
            kernelColumnValue,
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            capture(
                condition,
                DefaultIllustrationEvent::Accumulator,
                accumulators[channel],
                kernelRowValue,
                kernelColumnValue,
                channel,
                row,
                columnGroupStart
            );
        }
    }

    void captureBorderTap(
        Value * condition,
        Value * row,
        Value * columnGroupStart,
        Value * kernelRow,
        Value * kernelColumn,
        const unsigned channel,
        Value * sourceValid,
        Value * sample,
        Value * weight,
        Value * accumulator
    ) const {
        if (channel == 0U)
            captureTapCoordinates(condition, row, columnGroupStart, kernelRow, kernelColumn, sourceValid);
        capture(condition, DefaultIllustrationEvent::Sample, sample, kernelRow, kernelColumn, channel, row, columnGroupStart);
        if (channel == 0U) {
            capture(
                condition, DefaultIllustrationEvent::Weight, weight, kernelRow, kernelColumn, NoDefaultIllustrationChannel, row, columnGroupStart
            );
        }
        capture(condition, DefaultIllustrationEvent::Accumulator, accumulator, kernelRow, kernelColumn, channel, row, columnGroupStart);
    }

    void captureValue(
        Value * condition, const DefaultIllustrationEvent event, Value * value, const std::uint32_t channel, Value * row, Value * columnGroupStart
    ) const {
        capture(condition, event, value, builder.getSize(0), builder.getSize(0), channel, row, columnGroupStart);
    }

   private:
    Value * isInputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Input)));
    }

    Value * isOutputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Output)));
    }

    Value * outputSelectionMatchesGroup(Value * row, Value * columnGroupStart) const {
        Value * sameRow = builder.CreateICmpEQ(row, selectionRow);
        Value * startsBefore = builder.CreateICmpULE(columnGroupStart, selectionColumn);
        Value * endsAfter = builder.CreateICmpULT(selectionColumn, builder.CreateAdd(columnGroupStart, builder.getSize(laneCount)));
        return builder.CreateAnd(sameRow, builder.CreateAnd(startsBefore, endsAfter));
    }

    Value * inputSelectionMatchesTap(Value * row, Value * column, Value * kernelRow, Value * kernelColumn) const {
        Value * paddedInputRow = builder.CreateAdd(selectionRow, builder.getSize(kernelHeight / 2U));
        Value * paddedInputColumn = builder.CreateAdd(selectionColumn, builder.getSize(kernelWidth / 2U));
        Value * sameRow = builder.CreateICmpEQ(builder.CreateAdd(row, kernelRow), paddedInputRow);
        Value * sameColumn = builder.CreateICmpEQ(builder.CreateAdd(column, kernelColumn), paddedInputColumn);
        return builder.CreateAnd(sameRow, sameColumn);
    }

    Value * denseInputSelectionAffectsPosition(Value * row, Value * column) const {
        Value * paddedInputRow = builder.CreateAdd(selectionRow, builder.getSize(kernelHeight / 2U));
        Value * paddedInputColumn = builder.CreateAdd(selectionColumn, builder.getSize(kernelWidth / 2U));
        Value * rowStartsBefore = builder.CreateICmpULE(row, paddedInputRow);
        Value * rowEndsAfter = builder.CreateICmpULT(paddedInputRow, builder.CreateAdd(row, builder.getSize(kernelHeight)));
        Value * columnStartsBefore = builder.CreateICmpULE(column, paddedInputColumn);
        Value * columnEndsAfter = builder.CreateICmpULT(paddedInputColumn, builder.CreateAdd(column, builder.getSize(kernelWidth)));
        return builder.CreateAnd(builder.CreateAnd(rowStartsBefore, rowEndsAfter), builder.CreateAnd(columnStartsBefore, columnEndsAfter));
    }

    Value * inputSelectionAffectsPosition(Value * row, Value * column) const {
        if (useBorderTapLoop)
            return denseInputSelectionAffectsPosition(row, column);
        Value * selected = builder.getInt1(false);
        for (unsigned kernelRow = 0; kernelRow < kernelHeight; ++kernelRow) {
            for (unsigned kernelColumn = 0; kernelColumn < kernelWidth; ++kernelColumn) {
                if (weights[static_cast<std::size_t>(kernelRow) * kernelWidth + kernelColumn] == 0.0F)
                    continue;
                selected =
                    builder.CreateOr(selected, inputSelectionMatchesTap(row, column, builder.getSize(kernelRow), builder.getSize(kernelColumn)));
            }
        }
        return selected;
    }

    Value * selectedPositionMask(Value * row, Value * columnGroupStart) const {
        auto * type = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        Value * mask = Constant::getNullValue(type);
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(position));
            Value * outputSelected = builder.CreateAnd(builder.CreateICmpEQ(row, selectionRow), builder.CreateICmpEQ(column, selectionColumn));
            Value * inputSelected =
                builder.CreateAnd(builder.CreateICmpULT(column, builder.getSize(imageWidth)), inputSelectionAffectsPosition(row, column));
            Value * selected =
                builder.CreateOr(builder.CreateAnd(isOutputSelection(), outputSelected), builder.CreateAnd(isInputSelection(), inputSelected));
            mask = builder.CreateInsertElement(mask, builder.CreateZExt(selected, builder.getInt8Ty()), builder.getInt32(position));
        }
        return mask;
    }

    Value * outputMask(Value * columnGroupStart) const {
        auto * type = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        Value * mask = Constant::getNullValue(type);
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(position));
            Value * valid = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            mask = builder.CreateInsertElement(mask, builder.CreateZExt(valid, builder.getInt8Ty()), builder.getInt32(position));
        }
        return mask;
    }

    Value * outputColumns(Value * columnGroupStart) const {
        auto * type = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        Value * columns = Constant::getNullValue(type);
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * column = builder.CreateTrunc(builder.CreateAdd(columnGroupStart, builder.getSize(position)), builder.getInt32Ty());
            columns = builder.CreateInsertElement(columns, column, builder.getInt32(position));
        }
        return columns;
    }

    Value * sourceRows(Value * row, Value * kernelRow) const {
        auto * type = FixedVectorType::get(builder.getInt64Ty(), laneCount);
        Value * coordinates = Constant::getNullValue(type);
        Value * coordinate = builder.CreateSub(
            builder.CreateAdd(builder.CreateZExtOrTrunc(row, builder.getInt64Ty()), builder.CreateZExtOrTrunc(kernelRow, builder.getInt64Ty())),
            builder.getInt64(kernelHeight / 2U)
        );
        for (unsigned position = 0; position < laneCount; ++position)
            coordinates = builder.CreateInsertElement(coordinates, coordinate, builder.getInt32(position));
        return coordinates;
    }

    Value * sourceColumns(Value * columnGroupStart, Value * kernelColumn) const {
        auto * type = FixedVectorType::get(builder.getInt64Ty(), laneCount);
        Value * coordinates = Constant::getNullValue(type);
        Value * firstCoordinate = builder.CreateSub(
            builder.CreateAdd(
                builder.CreateZExtOrTrunc(columnGroupStart, builder.getInt64Ty()), builder.CreateZExtOrTrunc(kernelColumn, builder.getInt64Ty())
            ),
            builder.getInt64(kernelWidth / 2U)
        );
        for (unsigned position = 0; position < laneCount; ++position) {
            Value * coordinate = builder.CreateAdd(firstCoordinate, builder.getInt64(position));
            coordinates = builder.CreateInsertElement(coordinates, coordinate, builder.getInt32(position));
        }
        return coordinates;
    }

    void captureTapCoordinates(
        Value * condition, Value * row, Value * columnGroupStart, Value * kernelRow, Value * kernelColumn, Value * sourceValid
    ) const {
        capture(
            condition,
            DefaultIllustrationEvent::SourceRows,
            sourceRows(row, kernelRow),
            kernelRow,
            kernelColumn,
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        capture(
            condition,
            DefaultIllustrationEvent::SourceColumns,
            sourceColumns(columnGroupStart, kernelColumn),
            kernelRow,
            kernelColumn,
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
        capture(
            condition,
            DefaultIllustrationEvent::SourceValid,
            sourceValid,
            kernelRow,
            kernelColumn,
            NoDefaultIllustrationChannel,
            row,
            columnGroupStart
        );
    }

    void capture(
        Value * condition,
        const DefaultIllustrationEvent event,
        Value * value,
        Value * kernelRow,
        Value * kernelColumn,
        const std::uint32_t channel,
        Value * row,
        Value * columnGroupStart
    ) const {
        auto * vectorType = dyn_cast<FixedVectorType>(value->getType());
        if (vectorType == nullptr)
            throw std::logic_error("Default illustration capture requires a fixed vector");
        Type * elementType = vectorType->getElementType();
        std::size_t elementByteCount = 0;
        if (elementType->isFloatTy()) {
            elementByteCount = sizeof(float);
        } else if (auto * integerType = dyn_cast<IntegerType>(elementType)) {
            if (integerType->getBitWidth() % 8U != 0U)
                throw std::logic_error("Default illustration capture requires byte-sized elements");
            elementByteCount = integerType->getBitWidth() / 8U;
        } else {
            throw std::logic_error("Default illustration capture element type is unsupported");
        }

        if (elementByteCount != defaultIllustrationElementByteCount(event))
            throw std::logic_error("Default illustration capture event type mismatch");

        BasicBlock * captureBlock = builder.CreateBasicBlock("capture_default_value");
        BasicBlock * continueBlock = builder.CreateBasicBlock("after_default_capture");
        builder.CreateCondBr(condition, captureBlock, continueBlock);
        builder.SetInsertPoint(captureBlock);
        Value * storage = builder.CreateAllocaAtEntryPoint(value->getType());
        builder.CreateStore(value, storage);
        builder.CreateCall(
            captureFunction->getFunctionType(),
            captureFunction,
            {captureContext,
             builder.getInt32(static_cast<std::uint32_t>(event)),
             kernelRow,
             kernelColumn,
             builder.getInt32(channel),
             builder.getSize(vectorType->getNumElements()),
             builder.getSize(elementByteCount),
             builder.CreatePointerCast(storage, builder.getInt8PtrTy()),
             row,
             columnGroupStart}
        );
        builder.CreateBr(continueBlock);
        builder.SetInsertPoint(continueBlock);
    }

    KernelBuilder & builder;
    Value * const captureContext;
    Value * const selectionKind;
    Value * const selectionRow;
    Value * const selectionColumn;
    const unsigned imageWidth;
    const unsigned kernelWidth;
    const unsigned kernelHeight;
    const std::vector<float> & weights;
    const bool useBorderTapLoop;
    const unsigned laneCount;
    Function * captureFunction = nullptr;
};

class ConvolutionKernel final : public SegmentOrientedKernel {
   public:
    ConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * borderLoopWeights,
        Scalar * borderLoopWeightCount,
        const IllustrationScalars illustration,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelHeight,
        const unsigned kernelWidth,
        std::vector<float> weightValues,
        const std::string & persistentIdentity
    )
        : SegmentOrientedKernel(
              typeSystem,
              std::string(illustration ? "illustrated_convolution_" : "convolution_") + std::to_string(imageWidth) + "x" + std::to_string(imageHeight)
                  + "_k" + std::to_string(kernelHeight) + "x" + std::to_string(kernelWidth) + "_h"
                  + std::to_string(appendKernelNameHashBytes(
                      KernelNameHashInitialValue,
                      weightValues.data(),
                      checkedMultiply(weightValues.size(), sizeof(float), "default kernel-name weight byte count overflow")
                  ))
                  + "_c" + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              convolutionScalarBindings(typeSystem, inputPixels, outputPixels, borderLoopWeights, borderLoopWeightCount, illustration),
              {},
              {}
          ),
          imageWidth(imageWidth),
          imageHeight(imageHeight),
          kernelHeight(kernelHeight),
          kernelWidth(kernelWidth),
          weights(std::move(weightValues)),
          useBorderTapLoop(borderLoopWeights != nullptr),
          illustrated(static_cast<bool>(illustration)) {
        addAttribute(SideEffecting());
    }

   private:
    void linkExternalMethods(KernelBuilder & builder) final {
        SegmentOrientedKernel::linkExternalMethods(builder);
        if (!illustrated)
            return;
        auto * captureType = FunctionType::get(
            builder.getVoidTy(),
            {builder.getInt8PtrTy(),
             builder.getInt32Ty(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getInt32Ty(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getInt8PtrTy(),
             builder.getSizeTy(),
             builder.getSizeTy()},
            false
        );
        builder.LinkFunction("captureDefaultConvFilterValue", captureType, reinterpret_cast<void *>(&captureDefaultConvFilterValue));
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
        const unsigned laneCount,
        Value ** sourceValidity
    ) const {
        Value * samples = Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        Value * sourceValid =
            sourceValidity == nullptr ? nullptr : static_cast<Value *>(Constant::getNullValue(FixedVectorType::get(builder.getInt8Ty(), laneCount)));
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
            if (sourceValid != nullptr) {
                sourceValid =
                    builder.CreateInsertElement(sourceValid, builder.CreateZExt(isValidLane, builder.getInt8Ty()), builder.getInt32(laneIndex));
            }

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
        if (sourceValidity != nullptr)
            *sourceValidity = sourceValid;
        return samples;
    }

    std::array<Value *, ColorChannelCount> loadInteriorSamples(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned kernelRow,
        const unsigned kernelColumn,
        const unsigned laneCount,
        Value ** packedInput
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
        if (packedInput != nullptr)
            *packedInput = interleavedBytes;

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
        const unsigned laneCount,
        DefaultIllustrationEmitter * illustration
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
            Value * sourceValid = nullptr;
            Value * samples = loadBorderSamples(
                builder,
                inputPixels,
                row,
                columnGroupStart,
                channelIndex,
                kernelRow,
                kernelColumn,
                laneCount,
                illustration == nullptr ? nullptr : &sourceValid
            );
            Value * weightPointer = builder.CreateGEP(builder.getFloatTy(), borderLoopWeights, tapIndex);
            LoadInst * weightValue = builder.CreateLoad(builder.getFloatTy(), weightPointer);
            weightValue->setAlignment(Align(sizeof(float)));
            Value * weight =
                builder.CreateBitCast(builder.simd_fill(32, builder.CreateBitCast(weightValue, builder.getInt32Ty())), accumulator->getType());
            Value * nextAccumulator =
                builder.CreateCall(multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples, weight, runningAccumulator});
            if (illustration != nullptr) {
                illustration->captureBorderTap(
                    illustration->selectedTap(row, columnGroupStart, kernelRow, kernelColumn),
                    row,
                    columnGroupStart,
                    kernelRow,
                    kernelColumn,
                    channelIndex,
                    sourceValid,
                    samples,
                    weight,
                    nextAccumulator
                );
            }
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
                Value * sourceValid = nullptr;
                Value * samples = loadBorderSamples(
                    builder,
                    inputPixels,
                    row,
                    columnGroupStart,
                    channelIndex,
                    builder.getSize(kernelRow),
                    builder.getSize(kernelColumn),
                    laneCount,
                    illustration == nullptr ? nullptr : &sourceValid
                );
                Value * weight = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), weights[weightIndex]));
                accumulator = builder.CreateCall(multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples, weight, accumulator});
                if (illustration != nullptr) {
                    illustration->captureBorderTap(
                        illustration->selectedTap(row, columnGroupStart, builder.getSize(kernelRow), builder.getSize(kernelColumn)),
                        row,
                        columnGroupStart,
                        builder.getSize(kernelRow),
                        builder.getSize(kernelColumn),
                        channelIndex,
                        sourceValid,
                        samples,
                        weight,
                        accumulator
                    );
                }
            }
        }
        return clampAndRound(builder, accumulator, laneCount);
    }

    std::array<Value *, ColorChannelCount> computeInteriorChannels(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned laneCount,
        DefaultIllustrationEmitter * illustration
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
                Value * packedInput = nullptr;
                const auto samples = loadInteriorSamples(
                    builder, inputPixels, row, columnGroupStart, kernelRow, kernelColumn, laneCount, illustration == nullptr ? nullptr : &packedInput
                );
                Value * weight = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), weights[weightIndex]));
                for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
                    accumulators[channelIndex] = builder.CreateCall(
                        multiplyAddIntrinsic->getFunctionType(), multiplyAddIntrinsic, {samples[channelIndex], weight, accumulators[channelIndex]}
                    );
                }
                if (illustration != nullptr) {
                    illustration->captureInteriorTap(
                        illustration->selectedTap(row, columnGroupStart, builder.getSize(kernelRow), builder.getSize(kernelColumn)),
                        row,
                        columnGroupStart,
                        kernelRow,
                        kernelColumn,
                        packedInput,
                        samples,
                        weight,
                        accumulators
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
        const unsigned laneCount,
        DefaultIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        Value * storeMask = nullptr;
        if (illustration != nullptr) {
            auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
            Value * channelBytes = builder.CreateTrunc(roundedValues, channelByteType);
            illustration->captureValue(
                illustrationCondition, DefaultIllustrationEvent::OutputBytes, channelBytes, channelIndex, row, columnGroupStart
            );
            storeMask = Constant::getNullValue(channelByteType);
        }
        for (unsigned laneIndex = 0; laneIndex < laneCount; ++laneIndex) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(laneIndex));
            Value * outputByte = builder.CreateTrunc(builder.CreateExtractElement(roundedValues, builder.getInt32(laneIndex)), builder.getInt8Ty());
            Value * isValidLane = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            if (storeMask != nullptr) {
                storeMask = builder.CreateInsertElement(storeMask, builder.CreateZExt(isValidLane, builder.getInt8Ty()), builder.getInt32(laneIndex));
            }
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
        if (illustration != nullptr && channelIndex == 0U) {
            illustration->captureValue(
                illustrationCondition, DefaultIllustrationEvent::StoreMask, storeMask, NoDefaultIllustrationChannel, row, columnGroupStart
            );
        }
    }

    void storeInteriorPixels(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const std::array<Value *, ColorChannelCount> & roundedChannels,
        const unsigned laneCount,
        DefaultIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        auto * interleavedByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channelBytes;
        for (unsigned channelIndex = 0; channelIndex < ColorChannelCount; ++channelIndex) {
            channelBytes[channelIndex] = builder.CreateTrunc(roundedChannels[channelIndex], channelByteType);
            if (illustration != nullptr) {
                illustration->captureValue(
                    illustrationCondition, DefaultIllustrationEvent::OutputBytes, channelBytes[channelIndex], channelIndex, row, columnGroupStart
                );
            }
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
        if (illustration != nullptr) {
            illustration->captureValue(
                illustrationCondition, DefaultIllustrationEvent::PackedOutput, interleavedBytes, NoDefaultIllustrationChannel, row, columnGroupStart
            );
        }
        Value * outputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)),
            interleavedByteType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(interleavedBytes, outputPointer);
        store->setAlignment(Align(1));
        if (illustration != nullptr) {
            Value * storeMask = Constant::getAllOnesValue(FixedVectorType::get(builder.getInt8Ty(), laneCount));
            illustration->captureValue(
                illustrationCondition, DefaultIllustrationEvent::StoreMask, storeMask, NoDefaultIllustrationChannel, row, columnGroupStart
            );
        }
    }

    void generateDoSegmentMethod(KernelBuilder & builder) final {
        Value * inputPixels = builder.getScalarField("inputPixels");
        Value * outputPixels = builder.getScalarField("outputPixels");
        Value * borderLoopWeights = useBorderTapLoop ? builder.getScalarField("borderLoopWeights") : nullptr;
        Value * borderLoopWeightCount = useBorderTapLoop ? builder.getScalarField("borderLoopWeightCount") : nullptr;
        const unsigned laneCount = builder.getBitBlockWidth() / 32U;
        const unsigned verticalRadius = kernelHeight / 2U;
        const unsigned horizontalRadius = kernelWidth / 2U;
        std::optional<DefaultIllustrationEmitter> illustration;
        if (illustrated) {
            illustration.emplace(
                builder,
                builder.getScalarField("captureContext"),
                builder.getScalarField("selectionKind"),
                builder.getScalarField("selectionRow"),
                builder.getScalarField("selectionColumn"),
                imageWidth,
                kernelWidth,
                kernelHeight,
                weights,
                useBorderTapLoop,
                laneCount
            );
        }
        DefaultIllustrationEmitter * const illustrationEmitter = illustration ? &*illustration : nullptr;

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
        Value * illustrationCondition = nullptr;
        if (illustrationEmitter != nullptr) {
            illustrationCondition = illustrationEmitter->selectedGroup(row, columnGroupStart);
            illustrationEmitter->captureGroup(illustrationCondition, row, columnGroupStart);
        }

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
            if (illustrationEmitter != nullptr) {
                illustrationEmitter->captureValue(
                    illustrationCondition,
                    DefaultIllustrationEvent::IdentityPackedInput,
                    interleavedBytes,
                    NoDefaultIllustrationChannel,
                    row,
                    columnGroupStart
                );
            }
            Value * outputPointer = builder.CreatePointerCast(
                builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)),
                interleavedByteType->getPointerTo()
            );
            StoreInst * store = builder.CreateStore(interleavedBytes, outputPointer);
            store->setAlignment(Align(1));
            if (illustrationEmitter != nullptr) {
                illustrationEmitter->captureValue(
                    illustrationCondition,
                    DefaultIllustrationEvent::IdentityPackedOutput,
                    interleavedBytes,
                    NoDefaultIllustrationChannel,
                    row,
                    columnGroupStart
                );
                illustrationEmitter->captureValue(
                    illustrationCondition,
                    DefaultIllustrationEvent::StoreMask,
                    Constant::getAllOnesValue(FixedVectorType::get(builder.getInt8Ty(), laneCount)),
                    NoDefaultIllustrationChannel,
                    row,
                    columnGroupStart
                );
            }
        } else {
            const auto roundedChannels = computeInteriorChannels(builder, inputPixels, row, columnGroupStart, laneCount, illustrationEmitter);
            storeInteriorPixels(builder, outputPixels, row, columnGroupStart, roundedChannels, laneCount, illustrationEmitter, illustrationCondition);
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
                laneCount,
                illustrationEmitter
            );
            storeCheckedChannel(
                builder, outputPixels, row, columnGroupStart, channelIndex, roundedChannel, laneCount, illustrationEmitter, illustrationCondition
            );
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
    const bool illustrated;
};

void populateConvolutionPipeline(
    ProgramBuilder & pipeline,
    const bool hasRuntimeWeights,
    const bool illustrated,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelHeight,
    const unsigned kernelWidth,
    const std::vector<float> & weights,
    const std::string & persistentIdentity
) {
    Scalar * borderLoopWeights = nullptr;
    Scalar * borderLoopWeightCount = nullptr;
    if (hasRuntimeWeights) {
        borderLoopWeights = pipeline.getInputScalar("borderLoopWeights");
        borderLoopWeightCount = pipeline.getInputScalar("borderLoopWeightCount");
    }
    IllustrationScalars illustration;
    if (illustrated) {
        illustration = {
            pipeline.getInputScalar("captureContext"),
            pipeline.getInputScalar("selectionKind"),
            pipeline.getInputScalar("selectionRow"),
            pipeline.getInputScalar("selectionColumn"),
        };
    }
    StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<MemorySourceKernel>(pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream);
    pipeline.CreateKernelCall<ConvolutionKernel>(
        triggerStream,
        pipeline.getInputScalar("inputPixels"),
        pipeline.getInputScalar("outputPixels"),
        borderLoopWeights,
        borderLoopWeightCount,
        illustration,
        imageWidth,
        imageHeight,
        kernelHeight,
        kernelWidth,
        weights,
        persistentIdentity
    );
}

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
          useBorderTapLoop(usesBorderTapLoop(kernelWeights)) {
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
            populateConvolutionPipeline(pipeline, true, false, imageWidth, imageHeight, kernelHeight, kernelWidth, kernelWeights, persistentIdentity);
            borderLoopFunction = pipeline.compile();
        } else {
            auto pipeline = CreatePipeline(
                *driver,
                Input<const uint8_t *>("inputPixels"),
                Input<uint8_t *>("outputPixels"),
                Input<uint8_t *>("triggerBuffer"),
                Input<std::size_t>("triggerLength")
            );
            populateConvolutionPipeline(
                pipeline, false, false, imageWidth, imageHeight, kernelHeight, kernelWidth, kernelWeights, persistentIdentity
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

class DefaultFilterIllustrationImplementation final : public CompiledFilterIllustrationImplementation {
   public:
    DefaultFilterIllustrationImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        std::vector<float> weights,
        const std::string & persistentIdentity
    )
        : driver(std::move(cpuDriver)),
          imageWidthInPixels(imageWidth),
          imageHeightInPixels(imageHeight),
          imageByteCount(checkedImageByteCount(imageWidth, imageHeight)),
          kernelWeights(std::move(weights)),
          useBorderTapLoop(usesBorderTapLoop(kernelWeights)) {
        if (useBorderTapLoop) {
            auto pipeline = CreatePipeline(
                *driver,
                Input<const uint8_t *>("inputPixels"),
                Input<uint8_t *>("outputPixels"),
                Input<const float *>("borderLoopWeights"),
                Input<std::size_t>("borderLoopWeightCount"),
                Input<uint8_t *>("captureContext"),
                Input<std::uint32_t>("selectionKind"),
                Input<std::size_t>("selectionRow"),
                Input<std::size_t>("selectionColumn"),
                Input<uint8_t *>("triggerBuffer"),
                Input<std::size_t>("triggerLength")
            );
            populateConvolutionPipeline(pipeline, true, true, imageWidth, imageHeight, kernelHeight, kernelWidth, kernelWeights, persistentIdentity);
            borderLoopFunction = pipeline.compile();
        } else {
            auto pipeline = CreatePipeline(
                *driver,
                Input<const uint8_t *>("inputPixels"),
                Input<uint8_t *>("outputPixels"),
                Input<uint8_t *>("captureContext"),
                Input<std::uint32_t>("selectionKind"),
                Input<std::size_t>("selectionRow"),
                Input<std::size_t>("selectionColumn"),
                Input<uint8_t *>("triggerBuffer"),
                Input<std::size_t>("triggerLength")
            );
            populateConvolutionPipeline(pipeline, false, true, imageWidth, imageHeight, kernelHeight, kernelWidth, kernelWeights, persistentIdentity);
            fixedWeightFunction = pipeline.compile();
        }
    }

    unsigned imageWidth() const noexcept final {
        return imageWidthInPixels;
    }

    unsigned imageHeight() const noexcept final {
        return imageHeightInPixels;
    }

    std::size_t workspaceSize() const noexcept final {
        return 0;
    }

    std::size_t workspaceAlignment() const noexcept final {
        return 1;
    }

    bool apply(
        const std::uint8_t * input, std::uint8_t * output, void *, const ConvFilterIllustrationSelection selection, std::string & trace
    ) const final {
        assert(input != nullptr);
        assert(output != nullptr);
        switch (selection.kind) {
        case ConvFilterIllustrationSelectionKind::Input:
        case ConvFilterIllustrationSelectionKind::Output:
            break;
        default:
            return false;
        }
        if (selection.row >= imageHeightInPixels || selection.column >= imageWidthInPixels)
            return false;
        if (!hasDisjointImageRanges(input, output, imageByteCount))
            return false;

        DefaultIllustrationCaptureLog capture;
        std::uint8_t triggerByte = 0;
        const auto selectionKind = static_cast<std::uint32_t>(selection.kind);
        if (useBorderTapLoop) {
            borderLoopFunction(
                input,
                output,
                kernelWeights.data(),
                kernelWeights.size(),
                reinterpret_cast<std::uint8_t *>(&capture),
                selectionKind,
                selection.row,
                selection.column,
                &triggerByte,
                1U
            );
        } else {
            fixedWeightFunction(
                input, output, reinterpret_cast<std::uint8_t *>(&capture), selectionKind, selection.row, selection.column, &triggerByte, 1U
            );
        }
        if (capture.failure)
            std::rethrow_exception(capture.failure);
        trace = formatDefaultConvFilterIllustration(std::move(capture), selection.kind == ConvFilterIllustrationSelectionKind::Input);
        return true;
    }

   private:
    using FixedWeightFunction = void (*)(const uint8_t *, uint8_t *, uint8_t *, std::uint32_t, std::size_t, std::size_t, uint8_t *, std::size_t);
    using BorderLoopFunction =
        void (*)(const uint8_t *, uint8_t *, const float *, std::size_t, uint8_t *, std::uint32_t, std::size_t, std::size_t, uint8_t *, std::size_t);

    std::unique_ptr<CPUDriver> driver;
    const unsigned imageWidthInPixels;
    const unsigned imageHeightInPixels;
    const std::size_t imageByteCount;
    const std::vector<float> kernelWeights;
    const bool useBorderTapLoop;
    FixedWeightFunction fixedWeightFunction = nullptr;
    BorderLoopFunction borderLoopFunction = nullptr;
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

std::shared_ptr<const CompiledFilterIllustrationImplementation> compileDefaultFilterIllustration(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    std::vector<float> weights,
    std::string persistentIdentity
) {
    return std::make_shared<DefaultFilterIllustrationImplementation>(
        std::move(driver), imageWidth, imageHeight, kernelWidth, kernelHeight, std::move(weights), persistentIdentity
    );
}

}  // namespace kernel::image::internal
