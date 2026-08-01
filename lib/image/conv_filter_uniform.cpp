#include "conv_filter_common.h"
#include "conv_filter_uniform_illustration.h"

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

using namespace kernel;
using namespace llvm;

namespace kernel::image::internal {
namespace {

struct UniformKernelConfiguration {
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelHeight;
    unsigned kernelWidth;
    float weight;
};

Bindings uniformScalarBindings(
    LLVMTypeSystemInterface & typeSystem, Scalar * inputPixels, Scalar * outputPixels, Scalar * workspace, const IllustrationScalars illustration
) {
    Bindings bindings{
        Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels},
        Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels},
        Binding{typeSystem.getInt8PtrTy(), "workspace", workspace},
    };
    if (illustration) {
        bindings.emplace_back(typeSystem.getInt8PtrTy(), "captureContext", illustration.captureContext);
        bindings.emplace_back(typeSystem.getInt32Ty(), "selectionKind", illustration.selectionKind);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionRow", illustration.selectionRow);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionColumn", illustration.selectionColumn);
    }
    return bindings;
}

uint32_t floatBits(const float floatingValue) {
    uint32_t bits;
    std::memcpy(&bits, &floatingValue, sizeof(bits));
    return bits;
}

struct UniformIllustrationMetadata {
    UniformIllustrationPath path;
    std::uint32_t channel;
    Value * outputRow;
    Value * outputColumn;
    Value * groupStart;
    Value * workspaceColumn;
    Value * sourceInputRow;
    Value * recurrenceSource;
    Value * recurrenceDestination;
};

class UniformIllustrationEmitter {
   public:
    UniformIllustrationEmitter(
        KernelBuilder & builder,
        Value * captureContext,
        Value * selectionKind,
        Value * selectionRow,
        Value * selectionColumn,
        const unsigned imageWidth,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const unsigned logicalOutputs
    )
        : builder(builder),
          captureContext(captureContext),
          selectionKind(selectionKind),
          selectionRow(selectionRow),
          selectionColumn(selectionColumn),
          imageWidth(imageWidth),
          kernelWidth(kernelWidth),
          kernelHeight(kernelHeight),
          logicalOutputs(logicalOutputs) {
        captureFunction = builder.getModule()->getFunction("captureUniformConvFilterValue");
        if (captureFunction == nullptr)
            throw std::logic_error("Uniform illustration callback is not declared");
    }

    Value * selectedOperation(Value * row, Value * firstColumn, const unsigned outputCount) const {
        Value * selected = builder.getInt1(false);
        for (unsigned position = 0; position < outputCount; ++position)
            selected = builder.CreateOr(selected, selectedOutputState(row, builder.CreateAdd(firstColumn, builder.getSize(position))));
        return selected;
    }

    Value * selectedOutputOperation(Value * row, Value * firstColumn, const unsigned outputCount) const {
        Value * selected = builder.getInt1(false);
        for (unsigned position = 0; position < outputCount; ++position)
            selected = builder.CreateOr(selected, selectedOutputCoordinate(row, builder.CreateAdd(firstColumn, builder.getSize(position))));
        return selected;
    }

    Value * selectedOutputState(Value * row, Value * column) const {
        Value * outputSelected = selectedOutputCoordinate(row, column);
        Value * inputSelected =
            builder.CreateAnd(isInputSelection(), builder.CreateICmpULE(row, builder.CreateAdd(selectionRow, builder.getSize(kernelHeight / 2U))));
        inputSelected =
            builder.CreateAnd(inputSelected, builder.CreateICmpULE(selectionRow, builder.CreateAdd(row, builder.getSize(kernelHeight / 2U))));
        inputSelected =
            builder.CreateAnd(inputSelected, builder.CreateICmpULE(column, builder.CreateAdd(selectionColumn, builder.getSize(kernelWidth / 2U))));
        inputSelected =
            builder.CreateAnd(inputSelected, builder.CreateICmpULE(selectionColumn, builder.CreateAdd(column, builder.getSize(kernelWidth / 2U))));
        return builder.CreateOr(outputSelected, inputSelected);
    }

    Value * horizontalInitializationSelected(Value * row) const {
        Value * selected = builder.CreateAnd(isOutputSelection(), builder.CreateICmpEQ(row, selectionRow));
        Value * operationStartsAtZero = builder.CreateICmpEQ(selectionColumn, builder.getSize(0));
        if (kernelWidth == 1U && logicalOutputs < imageWidth)
            operationStartsAtZero = builder.CreateICmpULT(selectionColumn, builder.getSize(logicalOutputs));
        return builder.CreateAnd(selected, operationStartsAtZero);
    }

    Value * selectedInput(Value * workspaceColumn, Value * sourceInputRow) const {
        Value * selected = builder.CreateAnd(isInputSelection(), builder.CreateICmpEQ(workspaceColumn, selectionColumn));
        return builder.CreateAnd(selected, builder.CreateICmpEQ(sourceInputRow, selectionRow));
    }

    Value * workspaceSummarySelected(Value * row, Value * workspaceColumn) const {
        Value * selected = builder.CreateAnd(isInputSelection(), builder.CreateICmpEQ(workspaceColumn, selectionColumn));
        selected = builder.CreateAnd(selected, builder.CreateICmpULE(row, builder.CreateAdd(selectionRow, builder.getSize(kernelHeight / 2U))));
        return builder.CreateAnd(selected, builder.CreateICmpULE(selectionRow, builder.CreateAdd(row, builder.getSize(kernelHeight / 2U))));
    }

    Value * selectedOutputCoordinate(Value * row, Value * column) const {
        Value * selected = builder.CreateAnd(isOutputSelection(), builder.CreateICmpEQ(row, selectionRow));
        return builder.CreateAnd(selected, builder.CreateICmpEQ(column, selectionColumn));
    }

    Value * groupRecurrenceSelected(Value * row, Value * groupStart, Value * destinationColumn, const bool destinationInsideGroup) const {
        Value * selected = selectedOutputCoordinate(row, destinationColumn);
        if (destinationInsideGroup)
            selected = builder.CreateOr(selected, selectedOutputOperation(row, groupStart, logicalOutputs));
        return selected;
    }

    void captureRgb(Value * condition, const UniformIllustrationEvent event, Value * value, const UniformIllustrationMetadata & metadata) const {
        auto * vectorType = dyn_cast<FixedVectorType>(value->getType());
        if (vectorType == nullptr || vectorType->getNumElements() != 4U)
            throw std::logic_error("Uniform illustration RGB projection requires four elements");
        Value * rgb = builder.CreateShuffleVector(value, UndefValue::get(vectorType), ArrayRef<int>({0, 1, 2}));
        capture(condition, event, rgb, metadata);
    }

    void captureDirect(Value * condition, const UniformIllustrationEvent event, Value * value, const UniformIllustrationMetadata & metadata) const {
        capture(condition, event, value, metadata);
    }

    UniformIllustrationMetadata metadata(
        const UniformIllustrationPath path,
        Value * outputRow,
        Value * outputColumn = nullptr,
        Value * groupStart = nullptr,
        Value * workspaceColumn = nullptr,
        Value * sourceInputRow = nullptr,
        Value * recurrenceSource = nullptr,
        Value * recurrenceDestination = nullptr,
        const std::uint32_t channel = NoUniformIllustrationChannel
    ) const {
        Value * absent = builder.getSize(NoUniformIllustrationCoordinate);
        return {
            path,
            channel,
            outputRow == nullptr ? absent : outputRow,
            outputColumn == nullptr ? absent : outputColumn,
            groupStart == nullptr ? absent : groupStart,
            workspaceColumn == nullptr ? absent : workspaceColumn,
            sourceInputRow == nullptr ? absent : sourceInputRow,
            recurrenceSource == nullptr ? absent : recurrenceSource,
            recurrenceDestination == nullptr ? absent : recurrenceDestination,
        };
    }

   private:
    Value * isInputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Input)));
    }

    Value * isOutputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Output)));
    }

    void capture(Value * condition, const UniformIllustrationEvent event, Value * value, const UniformIllustrationMetadata & metadata) const {
        auto * vectorType = dyn_cast<FixedVectorType>(value->getType());
        if (vectorType == nullptr)
            throw std::logic_error("Uniform illustration capture requires a fixed vector");
        Type * elementType = vectorType->getElementType();
        std::size_t elementByteCount = 0;
        UniformIllustrationValueType valueType;
        if (elementType->isFloatTy()) {
            valueType = UniformIllustrationValueType::Float32;
            elementByteCount = sizeof(float);
        } else if (elementType->isIntegerTy(32)) {
            valueType = UniformIllustrationValueType::Int32;
            elementByteCount = sizeof(std::uint32_t);
        } else if (elementType->isIntegerTy(8)) {
            valueType = UniformIllustrationValueType::UInt8;
            elementByteCount = sizeof(std::uint8_t);
        } else {
            throw std::logic_error("Uniform illustration capture element type is unsupported");
        }
        if (valueType != uniformIllustrationValueType(event) || elementByteCount != uniformIllustrationElementByteCount(event))
            throw std::logic_error("Uniform illustration capture event type mismatch");

        BasicBlock * captureBlock = builder.CreateBasicBlock("capture_uniform_value");
        BasicBlock * continueBlock = builder.CreateBasicBlock("after_uniform_capture");
        builder.CreateCondBr(condition, captureBlock, continueBlock);
        builder.SetInsertPoint(captureBlock);
        Value * storage = builder.CreateAllocaAtEntryPoint(value->getType());
        builder.CreateStore(value, storage);
        builder.CreateCall(
            captureFunction->getFunctionType(),
            captureFunction,
            {captureContext,
             builder.getInt32(static_cast<std::uint32_t>(event)),
             builder.getInt32(static_cast<std::uint32_t>(metadata.path)),
             builder.getInt32(metadata.channel),
             builder.getSize(vectorType->getNumElements()),
             builder.getSize(elementByteCount),
             builder.CreatePointerCast(storage, builder.getInt8PtrTy()),
             metadata.outputRow,
             metadata.outputColumn,
             metadata.groupStart,
             metadata.workspaceColumn,
             metadata.sourceInputRow,
             metadata.recurrenceSource,
             metadata.recurrenceDestination}
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
    const unsigned logicalOutputs;
    Function * captureFunction = nullptr;
};

class UniformConvolutionKernel final : public SegmentOrientedKernel {
   public:
    UniformConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * workspace,
        const UniformKernelConfiguration & configuration,
        const std::string & persistentIdentity,
        const IllustrationScalars illustration = {}
    )
        : SegmentOrientedKernel(
              typeSystem,
              std::string(illustration ? "illustrated_uniform_convolution_" : "uniform_convolution_") + std::to_string(configuration.imageWidth) + "x"
                  + std::to_string(configuration.imageHeight) + "_k" + std::to_string(configuration.kernelHeight) + "x"
                  + std::to_string(configuration.kernelWidth) + "_w" + std::to_string(floatBits(configuration.weight)) + "_c" + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              uniformScalarBindings(typeSystem, inputPixels, outputPixels, workspace, illustration),
              {},
              {}
          ),
          imageWidth(configuration.imageWidth),
          imageHeight(configuration.imageHeight),
          kernelHeight(configuration.kernelHeight),
          kernelWidth(configuration.kernelWidth),
          uniformWeight(configuration.weight),
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
             builder.getInt32Ty(),
             builder.getInt32Ty(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getInt8PtrTy(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getSizeTy(),
             builder.getSizeTy()},
            false
        );
        builder.LinkFunction("captureUniformConvFilterValue", captureType, reinterpret_cast<void *>(&captureUniformConvFilterValue));
    }

    Value * pixelByteOffset(KernelBuilder & builder, Value * row, Value * column, const unsigned channel) const {
        Value * offset = builder.CreateMul(row, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channel));
    }

    Value * convertWindowSum(
        KernelBuilder & builder,
        Value * sum,
        const unsigned laneCount,
        UniformIllustrationEmitter * illustration,
        Value * illustrationCondition,
        Value * weightedIllustrationCondition,
        const UniformIllustrationMetadata & metadata,
        const bool grouped
    ) const {
        auto * integerType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        if (illustration != nullptr) {
            if (grouped)
                illustration->captureDirect(illustrationCondition, UniformIllustrationEvent::GroupSum, sum, metadata);
            else
                illustration->captureRgb(illustrationCondition, UniformIllustrationEvent::SingleSum, sum, metadata);
        }
        if (uniformWeight <= 0.0F)
            return Constant::getNullValue(integerType);
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Value * weighted = builder.CreateFMul(
            builder.CreateSIToFP(sum, floatType), builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), uniformWeight))
        );
        if (illustration != nullptr) {
            if (grouped)
                illustration->captureDirect(weightedIllustrationCondition, UniformIllustrationEvent::GroupWeighted, weighted, metadata);
            else
                illustration->captureRgb(weightedIllustrationCondition, UniformIllustrationEvent::SingleWeighted, weighted, metadata);
        }
        Value * maximum = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 255.0F));
        Value * rounding = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.5F));
        Value * bounded = builder.CreateSelect(builder.CreateFCmpOGT(weighted, maximum), maximum, weighted);
        Value * rounded = builder.CreateFAdd(bounded, rounding);
        return builder.CreateFPToSI(rounded, integerType);
    }

    void storePackedGroup(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const std::array<Value *, ColorChannelCount> & roundedChannels,
        const unsigned laneCount,
        UniformIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        auto * interleavedType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channelBytes;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            channelBytes[channel] = builder.CreateTrunc(roundedChannels[channel], channelByteType);
            if (illustration != nullptr) {
                illustration->captureDirect(
                    illustrationCondition,
                    UniformIllustrationEvent::GroupOutputBytes,
                    channelBytes[channel],
                    illustration->metadata(
                        UniformIllustrationPath::AdjacentOutputs, row, columnGroupStart, columnGroupStart, nullptr, nullptr, nullptr, nullptr, channel
                    )
                );
            }
        }
        SmallVector<int, 16> redGreenIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            redGreenIndexes.push_back(static_cast<int>(lane));
            redGreenIndexes.push_back(static_cast<int>(laneCount + lane));
        }
        Value * redGreenBytes = builder.CreateShuffleVector(channelBytes[0], channelBytes[1], redGreenIndexes);
        SmallVector<int, 16> blueIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            blueIndexes.push_back(static_cast<int>(lane));
        }
        blueIndexes.resize(laneCount * 2U, -1);
        Value * blueBytes = builder.CreateShuffleVector(channelBytes[2], UndefValue::get(channelByteType), blueIndexes);
        SmallVector<int, 32> interleavedIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            interleavedIndexes.push_back(static_cast<int>(lane * 2U));
            interleavedIndexes.push_back(static_cast<int>(lane * 2U + 1U));
            interleavedIndexes.push_back(static_cast<int>(laneCount * 2U + lane));
        }
        Value * interleavedBytes = builder.CreateShuffleVector(redGreenBytes, blueBytes, interleavedIndexes);
        if (illustration != nullptr) {
            illustration->captureDirect(
                illustrationCondition,
                UniformIllustrationEvent::GroupPackedOutput,
                interleavedBytes,
                illustration->metadata(UniformIllustrationPath::AdjacentOutputs, row, columnGroupStart, columnGroupStart)
            );
        }
        Value * outputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart, 0)), interleavedType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(interleavedBytes, outputPointer);
        store->setAlignment(Align(1));
    }

    Value * workspaceOffset(KernelBuilder & builder, Value * column) const {
        // Each column sum stores R, G, B, and one padding value.
        return builder.CreateMul(column, builder.getSize(4));
    }

    Value * loadPixel(KernelBuilder & builder, Value * inputPixels, Value * row, Value * column, Value ** packedValue = nullptr) const {
        auto * vectorType = FixedVectorType::get(builder.getInt32Ty(), 4);
        auto * packedType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
        Value * inputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, row, column, 0)), packedType->getPointerTo()
        );
        LoadInst * packedBytes = builder.CreateLoad(packedType, inputPointer);
        packedBytes->setAlignment(Align(1));
        if (packedValue != nullptr)
            *packedValue = packedBytes;
        Value * withPadding = builder.CreateShuffleVector(packedBytes, UndefValue::get(packedType), ArrayRef<int>({0, 1, 2, -1}));
        return builder.CreateZExt(withPadding, vectorType);
    }

    Value * loadWorkspace(KernelBuilder & builder, Value * workspace, Value * column) const {
        auto * vectorType = FixedVectorType::get(builder.getInt32Ty(), 4);
        Value * workspacePointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt32Ty(), workspace, workspaceOffset(builder, column)), vectorType->getPointerTo()
        );
        LoadInst * load = builder.CreateLoad(vectorType, workspacePointer);
        load->setAlignment(Align(1));
        return load;
    }

    void storeWorkspace(KernelBuilder & builder, Value * workspace, Value * column, Value * columnSum) const {
        auto * vectorType = FixedVectorType::get(builder.getInt32Ty(), 4);
        Value * workspacePointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt32Ty(), workspace, workspaceOffset(builder, column)), vectorType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(columnSum, workspacePointer);
        store->setAlignment(Align(1));
    }

    Value * adjustHorizontalSum(
        KernelBuilder & builder,
        Value * sum,
        Value * workspace,
        Value * column,
        Value * valid,
        const bool addition,
        UniformIllustrationEmitter * illustration,
        Value * row,
        Value * recurrenceSource,
        Value * illustrationCondition
    ) const {
        BasicBlock * unchangedBlock = builder.GetInsertBlock();
        BasicBlock * adjustBlock = builder.CreateBasicBlock(addition ? "add_entering_column" : "subtract_leaving_column");
        BasicBlock * joinBlock = builder.CreateBasicBlock("column_adjustment_join");
        builder.CreateCondBr(valid, adjustBlock, joinBlock);
        builder.SetInsertPoint(adjustBlock);
        Value * columnSum = loadWorkspace(builder, workspace, column);
        if (illustration != nullptr) {
            const UniformIllustrationEvent event =
                addition ? UniformIllustrationEvent::HorizontalEnteringOperand : UniformIllustrationEvent::HorizontalLeavingOperand;
            Value * operandCondition = illustrationCondition;
            if (addition)
                operandCondition = builder.CreateOr(operandCondition, illustration->workspaceSummarySelected(row, column));
            illustration->captureRgb(
                operandCondition,
                event,
                columnSum,
                illustration->metadata(
                    UniformIllustrationPath::SingleOutput,
                    row,
                    recurrenceSource,
                    nullptr,
                    column,
                    nullptr,
                    recurrenceSource,
                    builder.CreateAdd(recurrenceSource, builder.getSize(1))
                )
            );
        }
        Value * adjusted = addition ? builder.CreateAdd(sum, columnSum) : builder.CreateSub(sum, columnSum);
        BasicBlock * adjustedBlock = builder.GetInsertBlock();
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(joinBlock);
        PHINode * selectedSum = builder.CreatePHI(sum->getType(), 2);
        selectedSum->addIncoming(sum, unchangedBlock);
        selectedSum->addIncoming(adjusted, adjustedBlock);
        return selectedSum;
    }

    void storeSinglePixel(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * column,
        Value * sum,
        UniformIllustrationEmitter * illustration,
        Value * illustrationCondition,
        Value * weightedIllustrationCondition
    ) const {
        const UniformIllustrationMetadata metadata =
            illustration == nullptr ? UniformIllustrationMetadata{} : illustration->metadata(UniformIllustrationPath::SingleOutput, row, column);
        Value * rounded = convertWindowSum(builder, sum, 4, illustration, illustrationCondition, weightedIllustrationCondition, metadata, false);
        auto * channelType = FixedVectorType::get(builder.getInt8Ty(), 4);
        auto * packedType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
        Value * channelBytes = builder.CreateTrunc(rounded, channelType);
        Value * packedBytes = builder.CreateShuffleVector(channelBytes, UndefValue::get(channelType), ArrayRef<int>({0, 1, 2}));
        if (illustration != nullptr) {
            illustration->captureRgb(illustrationCondition, UniformIllustrationEvent::SingleOutputBytes, channelBytes, metadata);
            illustration->captureDirect(illustrationCondition, UniformIllustrationEvent::SinglePackedOutput, packedBytes, metadata);
        }
        Value * outputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, column, 0)), packedType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(packedBytes, outputPointer);
        store->setAlignment(Align(1));
    }

    void generateDoSegmentMethod(KernelBuilder & builder) final {
        Value * inputPixels = builder.getScalarField("inputPixels");
        Value * outputPixels = builder.getScalarField("outputPixels");
        Value * workspace = builder.CreatePointerCast(builder.getScalarField("workspace"), builder.getInt32Ty()->getPointerTo());
        auto * vectorType = FixedVectorType::get(builder.getInt32Ty(), 4);
        const unsigned outputLaneCount = builder.getBitBlockWidth() / 32U;
        const unsigned initialRowCount = std::min(imageHeight, kernelHeight / 2U + 1U);
        const unsigned initialColumnCount = std::min(imageWidth, kernelWidth / 2U + 1U);
        std::optional<UniformIllustrationEmitter> illustration;
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
                outputLaneCount
            );
        }
        UniformIllustrationEmitter * const illustrationEmitter = illustration ? &*illustration : nullptr;

        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * initializeColumns = builder.CreateBasicBlock("initialize_columns");
        BasicBlock * initializationDone = builder.CreateBasicBlock("initialization_done");
        builder.CreateBr(initializeColumns);
        builder.SetInsertPoint(initializeColumns);
        PHINode * initialColumn = builder.CreatePHI(builder.getSizeTy(), 2);
        initialColumn->addIncoming(builder.getSize(0), entry);
        BasicBlock * initializeRows = builder.CreateBasicBlock("initialize_column_rows");
        BasicBlock * columnInitialized = builder.CreateBasicBlock("column_initialized");
        builder.CreateBr(initializeRows);
        builder.SetInsertPoint(initializeRows);
        PHINode * initialRow = builder.CreatePHI(builder.getSizeTy(), 2);
        initialRow->addIncoming(builder.getSize(0), initializeColumns);
        PHINode * initialSum = builder.CreatePHI(vectorType, 2);
        initialSum->addIncoming(Constant::getNullValue(vectorType), initializeColumns);
        Value * packedInitialPixel = nullptr;
        Value * initialPixel =
            loadPixel(builder, inputPixels, initialRow, initialColumn, illustrationEmitter == nullptr ? nullptr : &packedInitialPixel);
        if (illustrationEmitter != nullptr) {
            Value * condition = illustrationEmitter->selectedInput(initialColumn, initialRow);
            const auto metadata = illustrationEmitter->metadata(
                UniformIllustrationPath::ColumnInitialization, builder.getSize(0), nullptr, nullptr, initialColumn, initialRow
            );
            illustrationEmitter->captureDirect(condition, UniformIllustrationEvent::PackedInput, packedInitialPixel, metadata);
            illustrationEmitter->captureRgb(condition, UniformIllustrationEvent::InputRgb, initialPixel, metadata);
        }
        Value * nextInitialSum = builder.CreateAdd(initialSum, initialPixel);
        Value * nextInitialRow = builder.CreateAdd(initialRow, builder.getSize(1));
        BasicBlock * initialRowBack = builder.GetInsertBlock();
        builder.CreateCondBr(builder.CreateICmpULT(nextInitialRow, builder.getSize(initialRowCount)), initializeRows, columnInitialized);
        initialRow->addIncoming(nextInitialRow, initialRowBack);
        initialSum->addIncoming(nextInitialSum, initialRowBack);
        builder.SetInsertPoint(columnInitialized);
        storeWorkspace(builder, workspace, initialColumn, nextInitialSum);
        Value * nextInitialColumn = builder.CreateAdd(initialColumn, builder.getSize(1));
        initialColumn->addIncoming(nextInitialColumn, columnInitialized);
        builder.CreateCondBr(builder.CreateICmpULT(nextInitialColumn, builder.getSize(imageWidth)), initializeColumns, initializationDone);

        builder.SetInsertPoint(initializationDone);
        BasicBlock * rowLoop = builder.CreateBasicBlock("rolling_rows");
        BasicBlock * done = builder.CreateBasicBlock("rolling_done");
        builder.CreateBr(rowLoop);
        builder.SetInsertPoint(rowLoop);
        PHINode * row = builder.CreatePHI(builder.getSizeTy(), 2);
        row->addIncoming(builder.getSize(0), initializationDone);

        BasicBlock * initializeHorizontal = builder.CreateBasicBlock("initialize_horizontal_sum");
        BasicBlock * horizontalReady = builder.CreateBasicBlock("horizontal_sum_ready");
        builder.CreateBr(initializeHorizontal);
        builder.SetInsertPoint(initializeHorizontal);
        PHINode * windowColumn = builder.CreatePHI(builder.getSizeTy(), 2);
        windowColumn->addIncoming(builder.getSize(0), rowLoop);
        PHINode * windowSum = builder.CreatePHI(vectorType, 2);
        windowSum->addIncoming(Constant::getNullValue(vectorType), rowLoop);
        Value * workspaceOperand = loadWorkspace(builder, workspace, windowColumn);
        Value * horizontalInitializationCondition = nullptr;
        Value * horizontalOperandCondition = nullptr;
        UniformIllustrationMetadata horizontalInitializationMetadata{};
        if (illustrationEmitter != nullptr) {
            horizontalInitializationCondition = illustrationEmitter->horizontalInitializationSelected(row);
            horizontalOperandCondition =
                builder.CreateOr(horizontalInitializationCondition, illustrationEmitter->workspaceSummarySelected(row, windowColumn));
            horizontalInitializationMetadata =
                illustrationEmitter->metadata(UniformIllustrationPath::WindowInitialization, row, builder.getSize(0), nullptr, windowColumn);
            illustrationEmitter->captureRgb(
                horizontalOperandCondition, UniformIllustrationEvent::HorizontalInitialOperand, workspaceOperand, horizontalInitializationMetadata
            );
        }
        Value * nextWindowSum = builder.CreateAdd(windowSum, workspaceOperand);
        if (illustrationEmitter != nullptr) {
            illustrationEmitter->captureRgb(
                horizontalInitializationCondition,
                UniformIllustrationEvent::HorizontalInitialAfterAdd,
                nextWindowSum,
                horizontalInitializationMetadata
            );
        }
        Value * nextWindowColumn = builder.CreateAdd(windowColumn, builder.getSize(1));
        BasicBlock * windowBack = builder.GetInsertBlock();
        builder.CreateCondBr(builder.CreateICmpULT(nextWindowColumn, builder.getSize(initialColumnCount)), initializeHorizontal, horizontalReady);
        windowColumn->addIncoming(nextWindowColumn, windowBack);
        windowSum->addIncoming(nextWindowSum, windowBack);

        builder.SetInsertPoint(horizontalReady);
        BasicBlock * outputColumns = builder.CreateBasicBlock("rolling_output_columns");
        BasicBlock * outputRowDone = builder.CreateBasicBlock("rolling_output_row_done");
        builder.CreateBr(outputColumns);
        builder.SetInsertPoint(outputColumns);
        PHINode * column = builder.CreatePHI(builder.getSizeTy(), 2);
        column->addIncoming(builder.getSize(0), horizontalReady);
        PHINode * rollingSum = builder.CreatePHI(vectorType, 2);
        rollingSum->addIncoming(nextWindowSum, horizontalReady);
        const unsigned horizontalRadius = kernelWidth / 2U;
        Value * groupEligible = builder.CreateICmpUGE(column, builder.getSize(horizontalRadius));
        groupEligible = builder.CreateAnd(
            groupEligible,
            builder.CreateICmpULT(builder.CreateAdd(column, builder.getSize(outputLaneCount + horizontalRadius)), builder.getSize(imageWidth))
        );
        BasicBlock * groupBlock = builder.CreateBasicBlock("rolling_grouped_output");
        BasicBlock * scalarBlock = builder.CreateBasicBlock("rolling_scalar_output");
        BasicBlock * outputJoin = builder.CreateBasicBlock("rolling_output_join");
        builder.CreateCondBr(groupEligible, groupBlock, scalarBlock);

        builder.SetInsertPoint(groupBlock);
        std::array<Value *, ColorChannelCount> groupedChannels;
        auto * outputVectorType = FixedVectorType::get(builder.getInt32Ty(), outputLaneCount);
        Value * groupIllustrationCondition =
            illustrationEmitter == nullptr ? nullptr : illustrationEmitter->selectedOperation(row, column, outputLaneCount);
        Value * groupWeightedIllustrationCondition =
            illustrationEmitter == nullptr ? nullptr : illustrationEmitter->selectedOutputOperation(row, column, outputLaneCount);
        Value * nextGroupSum = rollingSum;
        SmallVector<Value *, 16> pixelSums(outputLaneCount);
        for (unsigned lane = 0; lane < outputLaneCount; ++lane) {
            pixelSums[lane] = nextGroupSum;
            Value * removeColumn = builder.CreateAdd(builder.CreateSub(column, builder.getSize(horizontalRadius)), builder.getSize(lane));
            Value * addColumn = builder.CreateAdd(column, builder.getSize(horizontalRadius + lane + 1U));
            Value * recurrenceSource = builder.CreateAdd(column, builder.getSize(lane));
            Value * recurrenceDestination = builder.CreateAdd(recurrenceSource, builder.getSize(1));
            Value * recurrenceCondition =
                illustrationEmitter == nullptr
                    ? nullptr
                    : illustrationEmitter->groupRecurrenceSelected(row, column, recurrenceDestination, lane + 1U < outputLaneCount);
            Value * leavingColumnSum = loadWorkspace(builder, workspace, removeColumn);
            if (illustrationEmitter != nullptr) {
                illustrationEmitter->captureRgb(
                    recurrenceCondition,
                    UniformIllustrationEvent::HorizontalLeavingOperand,
                    leavingColumnSum,
                    illustrationEmitter->metadata(
                        UniformIllustrationPath::AdjacentOutputs, row, nullptr, column, removeColumn, nullptr, recurrenceSource, recurrenceDestination
                    )
                );
            }
            nextGroupSum = builder.CreateSub(nextGroupSum, leavingColumnSum);
            Value * enteringColumnSum = loadWorkspace(builder, workspace, addColumn);
            if (illustrationEmitter != nullptr) {
                Value * operandCondition = builder.CreateOr(recurrenceCondition, illustrationEmitter->workspaceSummarySelected(row, addColumn));
                illustrationEmitter->captureRgb(
                    operandCondition,
                    UniformIllustrationEvent::HorizontalEnteringOperand,
                    enteringColumnSum,
                    illustrationEmitter->metadata(
                        UniformIllustrationPath::AdjacentOutputs, row, nullptr, column, addColumn, nullptr, recurrenceSource, recurrenceDestination
                    )
                );
            }
            nextGroupSum = builder.CreateAdd(nextGroupSum, enteringColumnSum);
        }
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            Value * channelSums = Constant::getNullValue(outputVectorType);
            for (unsigned lane = 0; lane < outputLaneCount; ++lane) {
                channelSums = builder.CreateInsertElement(
                    channelSums, builder.CreateExtractElement(pixelSums[lane], builder.getInt32(channel)), builder.getInt32(lane)
                );
            }
            const UniformIllustrationMetadata metadata =
                illustrationEmitter == nullptr
                    ? UniformIllustrationMetadata{}
                    : illustrationEmitter->metadata(
                          UniformIllustrationPath::AdjacentOutputs, row, nullptr, column, nullptr, nullptr, nullptr, nullptr, channel
                      );
            groupedChannels[channel] = convertWindowSum(
                builder,
                channelSums,
                outputLaneCount,
                illustrationEmitter,
                groupIllustrationCondition,
                groupWeightedIllustrationCondition,
                metadata,
                true
            );
        }
        storePackedGroup(builder, outputPixels, row, column, groupedChannels, outputLaneCount, illustrationEmitter, groupIllustrationCondition);
        Value * nextGroupedColumn = builder.CreateAdd(column, builder.getSize(outputLaneCount));
        BasicBlock * groupBack = builder.GetInsertBlock();
        builder.CreateBr(outputJoin);

        builder.SetInsertPoint(scalarBlock);
        Value * scalarIllustrationCondition = illustrationEmitter == nullptr ? nullptr : illustrationEmitter->selectedOperation(row, column, 1U);
        Value * scalarWeightedIllustrationCondition =
            illustrationEmitter == nullptr ? nullptr : illustrationEmitter->selectedOutputOperation(row, column, 1U);
        storeSinglePixel(
            builder, outputPixels, row, column, rollingSum, illustrationEmitter, scalarIllustrationCondition, scalarWeightedIllustrationCondition
        );
        Value * removeValid = builder.CreateICmpUGE(column, builder.getSize(horizontalRadius));
        Value * removeColumn = builder.CreateSub(column, builder.getSize(horizontalRadius));
        Value * recurrenceDestination = builder.CreateAdd(column, builder.getSize(1));
        Value * scalarRecurrenceCondition =
            illustrationEmitter == nullptr ? nullptr : illustrationEmitter->selectedOutputCoordinate(row, recurrenceDestination);
        Value * nextScalarSum = adjustHorizontalSum(
            builder, rollingSum, workspace, removeColumn, removeValid, false, illustrationEmitter, row, column, scalarRecurrenceCondition
        );
        Value * addColumn = builder.CreateAdd(column, builder.getSize(horizontalRadius + 1U));
        Value * addValid = builder.CreateICmpULT(addColumn, builder.getSize(imageWidth));
        nextScalarSum = adjustHorizontalSum(
            builder, nextScalarSum, workspace, addColumn, addValid, true, illustrationEmitter, row, column, scalarRecurrenceCondition
        );
        Value * nextScalarColumn = builder.CreateAdd(column, builder.getSize(1));
        BasicBlock * scalarBack = builder.GetInsertBlock();
        builder.CreateBr(outputJoin);

        builder.SetInsertPoint(outputJoin);
        PHINode * nextColumn = builder.CreatePHI(builder.getSizeTy(), 2);
        nextColumn->addIncoming(nextGroupedColumn, groupBack);
        nextColumn->addIncoming(nextScalarColumn, scalarBack);
        PHINode * nextSum = builder.CreatePHI(vectorType, 2);
        nextSum->addIncoming(nextGroupSum, groupBack);
        nextSum->addIncoming(nextScalarSum, scalarBack);
        BasicBlock * outputBack = builder.GetInsertBlock();
        builder.CreateCondBr(builder.CreateICmpULT(nextColumn, builder.getSize(imageWidth)), outputColumns, outputRowDone);
        column->addIncoming(nextColumn, outputBack);
        rollingSum->addIncoming(nextSum, outputBack);

        builder.SetInsertPoint(outputRowDone);
        Value * nextRow = builder.CreateAdd(row, builder.getSize(1));
        Value * hasNextRow = builder.CreateICmpULT(nextRow, builder.getSize(imageHeight));
        BasicBlock * updateDispatch = builder.CreateBasicBlock("update_row_region");
        builder.CreateCondBr(hasNextRow, updateDispatch, done);
        builder.SetInsertPoint(updateDispatch);
        Value * removeRowValid = builder.CreateICmpUGE(row, builder.getSize(kernelHeight / 2U));
        Value * removeRow = builder.CreateSub(row, builder.getSize(kernelHeight / 2U));
        Value * addRow = builder.CreateAdd(row, builder.getSize(kernelHeight / 2U + 1U));
        Value * addRowValid = builder.CreateICmpULT(addRow, builder.getSize(imageHeight));
        BasicBlock * removeDecision = builder.CreateBasicBlock("update_with_removal");
        BasicBlock * noRemoveDecision = builder.CreateBasicBlock("update_without_removal");
        BasicBlock * addAndRemove = builder.CreateBasicBlock("update_add_and_remove");
        BasicBlock * removeOnly = builder.CreateBasicBlock("update_remove_only");
        BasicBlock * addOnly = builder.CreateBasicBlock("update_add_only");
        BasicBlock * unchanged = builder.CreateBasicBlock("update_unchanged");
        builder.CreateCondBr(removeRowValid, removeDecision, noRemoveDecision);
        builder.SetInsertPoint(removeDecision);
        builder.CreateCondBr(addRowValid, addAndRemove, removeOnly);
        builder.SetInsertPoint(noRemoveDecision);
        builder.CreateCondBr(addRowValid, addOnly, unchanged);

        const auto generateUpdateLoop = [&](BasicBlock * preheader, const bool removeSample, const bool addSample, const StringRef loopName) {
            builder.SetInsertPoint(preheader);
            BasicBlock * loop = builder.CreateBasicBlock(loopName);
            builder.CreateBr(loop);
            builder.SetInsertPoint(loop);
            PHINode * updateColumn = builder.CreatePHI(builder.getSizeTy(), 2);
            updateColumn->addIncoming(builder.getSize(0), preheader);
            Value * updatedColumnSum = loadWorkspace(builder, workspace, updateColumn);
            if (removeSample)
                updatedColumnSum = builder.CreateSub(updatedColumnSum, loadPixel(builder, inputPixels, removeRow, updateColumn));
            if (addSample) {
                Value * packedAddedPixel = nullptr;
                Value * addedPixel =
                    loadPixel(builder, inputPixels, addRow, updateColumn, illustrationEmitter == nullptr ? nullptr : &packedAddedPixel);
                if (illustrationEmitter != nullptr) {
                    Value * condition = illustrationEmitter->selectedInput(updateColumn, addRow);
                    const auto metadata = illustrationEmitter->metadata(
                        UniformIllustrationPath::ColumnUpdate, nextRow, nullptr, nullptr, updateColumn, addRow, row, nextRow
                    );
                    illustrationEmitter->captureDirect(condition, UniformIllustrationEvent::PackedInput, packedAddedPixel, metadata);
                    illustrationEmitter->captureRgb(condition, UniformIllustrationEvent::InputRgb, addedPixel, metadata);
                }
                updatedColumnSum = builder.CreateAdd(updatedColumnSum, addedPixel);
            }
            storeWorkspace(builder, workspace, updateColumn, updatedColumnSum);
            Value * nextUpdateColumn = builder.CreateAdd(updateColumn, builder.getSize(1));
            BasicBlock * updateBack = builder.GetInsertBlock();
            builder.CreateCondBr(builder.CreateICmpULT(nextUpdateColumn, builder.getSize(imageWidth)), loop, rowLoop);
            updateColumn->addIncoming(nextUpdateColumn, updateBack);
            row->addIncoming(nextRow, updateBack);
        };
        generateUpdateLoop(addAndRemove, true, true, "update_add_remove_columns");
        generateUpdateLoop(removeOnly, true, false, "update_remove_columns");
        generateUpdateLoop(addOnly, false, true, "update_add_columns");
        builder.SetInsertPoint(unchanged);
        builder.CreateBr(rowLoop);
        row->addIncoming(nextRow, unchanged);
        builder.SetInsertPoint(done);
    }

    const unsigned imageWidth;
    const unsigned imageHeight;
    const unsigned kernelHeight;
    const unsigned kernelWidth;
    const float uniformWeight;
    const bool illustrated;
};

using UniformFunction = void (*)(const uint8_t *, uint8_t *, uint8_t *, uint8_t *, std::size_t);
using IllustratedUniformFunction =
    void (*)(const uint8_t *, uint8_t *, uint8_t *, uint8_t *, std::uint32_t, std::size_t, std::size_t, uint8_t *, std::size_t);

class UniformFilterImplementation final : public CompiledFilterImplementation {
   public:
    UniformFilterImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const float weight,
        const std::string & persistentIdentity
    )
        : CompiledFilterImplementation(
              ConvFilterMode::Uniform,
              imageWidth,
              imageHeight,
              checkedImageByteCount(imageWidth, imageHeight),
              checkedUniformWorkspaceByteCount(imageWidth),
              alignof(std::uint32_t)
          ),
          driver(std::move(cpuDriver)) {
        const UniformKernelConfiguration configuration{imageWidth, imageHeight, kernelHeight, kernelWidth, weight};
        auto pipeline = CreatePipeline(
            *driver,
            Input<const uint8_t *>("inputPixels"),
            Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("workspace"),
            Input<uint8_t *>("triggerBuffer"),
            Input<std::size_t>("triggerLength")
        );
        StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<MemorySourceKernel>(
            pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
        );
        pipeline.CreateKernelCall<UniformConvolutionKernel>(
            triggerStream,
            pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"),
            pipeline.getInputScalar("workspace"),
            configuration,
            persistentIdentity
        );
        compiledFunction = pipeline.compile();
    }

   private:
    void invoke(const uint8_t * inputPixels, uint8_t * outputPixels, void * workspace) const noexcept final {
        uint8_t triggerByte = 0;
        compiledFunction(inputPixels, outputPixels, static_cast<uint8_t *>(workspace), &triggerByte, 1U);
    }

    std::unique_ptr<CPUDriver> driver;
    UniformFunction compiledFunction = nullptr;
};

class UniformFilterIllustrationImplementation final : public CompiledFilterIllustrationImplementation {
   public:
    UniformFilterIllustrationImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const float weight,
        const std::string & persistentIdentity
    )
        : driver(std::move(cpuDriver)),
          imageWidthInPixels(imageWidth),
          imageHeightInPixels(imageHeight),
          kernelWidthInPixels(kernelWidth),
          kernelHeightInPixels(kernelHeight),
          uniformWeight(weight),
          imageByteCount(checkedImageByteCount(imageWidth, imageHeight)),
          workspaceByteCount(checkedUniformWorkspaceByteCount(imageWidth)),
          logicalOutputs(driver->getBitBlockWidth() / 32U) {
        const UniformKernelConfiguration configuration{imageWidth, imageHeight, kernelHeight, kernelWidth, weight};
        auto pipeline = CreatePipeline(
            *driver,
            Input<const uint8_t *>("inputPixels"),
            Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("workspace"),
            Input<uint8_t *>("captureContext"),
            Input<std::uint32_t>("selectionKind"),
            Input<std::size_t>("selectionRow"),
            Input<std::size_t>("selectionColumn"),
            Input<uint8_t *>("triggerBuffer"),
            Input<std::size_t>("triggerLength")
        );
        StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<MemorySourceKernel>(
            pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
        );
        const IllustrationScalars illustration{
            pipeline.getInputScalar("captureContext"),
            pipeline.getInputScalar("selectionKind"),
            pipeline.getInputScalar("selectionRow"),
            pipeline.getInputScalar("selectionColumn"),
        };
        pipeline.CreateKernelCall<UniformConvolutionKernel>(
            triggerStream,
            pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"),
            pipeline.getInputScalar("workspace"),
            configuration,
            persistentIdentity,
            illustration
        );
        compiledFunction = pipeline.compile();
    }

    unsigned imageWidth() const noexcept final {
        return imageWidthInPixels;
    }

    unsigned imageHeight() const noexcept final {
        return imageHeightInPixels;
    }

    std::size_t workspaceSize() const noexcept final {
        return workspaceByteCount;
    }

    std::size_t workspaceAlignment() const noexcept final {
        return alignof(std::uint32_t);
    }

    bool apply(
        const std::uint8_t * input, std::uint8_t * output, void * workspace, const ConvFilterIllustrationSelection selection, std::string & trace
    ) const final {
        assert(input != nullptr);
        assert(output != nullptr);
        assert(workspace != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(workspace) % alignof(std::uint32_t) == 0U);
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

        UniformIllustrationCaptureLog capture;
        capture.logicalOutputs = logicalOutputs;
        std::uint8_t triggerByte = 0;
        compiledFunction(
            input,
            output,
            static_cast<std::uint8_t *>(workspace),
            reinterpret_cast<std::uint8_t *>(&capture),
            static_cast<std::uint32_t>(selection.kind),
            selection.row,
            selection.column,
            &triggerByte,
            1U
        );
        if (capture.failure)
            std::rethrow_exception(capture.failure);
        const UniformIllustrationConfiguration configuration{
            imageWidthInPixels,
            imageHeightInPixels,
            kernelWidthInPixels,
            kernelHeightInPixels,
            uniformWeight,
            logicalOutputs,
        };
        std::string completeTrace = formatUniformConvFilterIllustration(configuration, selection, capture);
        trace = std::move(completeTrace);
        return true;
    }

   private:
    std::unique_ptr<CPUDriver> driver;
    const unsigned imageWidthInPixels;
    const unsigned imageHeightInPixels;
    const unsigned kernelWidthInPixels;
    const unsigned kernelHeightInPixels;
    const float uniformWeight;
    const std::size_t imageByteCount;
    const std::size_t workspaceByteCount;
    const unsigned logicalOutputs;
    IllustratedUniformFunction compiledFunction = nullptr;
};

}  // namespace

std::shared_ptr<const CompiledFilterImplementation> compileUniformFilter(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const float weight,
    std::string persistentIdentity
) {
    return std::make_shared<UniformFilterImplementation>(
        std::move(driver), imageWidth, imageHeight, kernelWidth, kernelHeight, weight, persistentIdentity
    );
}

std::shared_ptr<const CompiledFilterIllustrationImplementation> compileUniformFilterIllustration(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const float weight,
    std::string persistentIdentity
) {
    return std::make_shared<UniformFilterIllustrationImplementation>(
        std::move(driver), imageWidth, imageHeight, kernelWidth, kernelHeight, weight, persistentIdentity
    );
}

}  // namespace kernel::image::internal
