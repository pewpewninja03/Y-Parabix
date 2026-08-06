#include "conv_filter_common.h"
#include "conv_filter_low_rank_illustration.h"

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>

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

struct LowRankKernelConfiguration {
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelHeight;
    unsigned kernelWidth;
    const std::vector<float> & horizontalFactors;
    const std::vector<float> & verticalFactors;
    unsigned rankCount;
};

Bindings lowRankScalarBindings(
    LLVMTypeSystemInterface & typeSystem,
    Scalar * inputPixels,
    Scalar * outputPixels,
    Scalar * workspace,
    Scalar * horizontalFactors,
    Scalar * verticalFactors,
    const IllustrationScalars illustration
) {
    Bindings bindings{
        Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels},
        Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels},
        Binding{typeSystem.getInt8PtrTy(), "workspace", workspace},
        Binding{typeSystem.getFloatTy()->getPointerTo(), "horizontalFactors", horizontalFactors},
        Binding{typeSystem.getFloatTy()->getPointerTo(), "verticalFactors", verticalFactors},
    };
    if (illustration) {
        bindings.emplace_back(typeSystem.getInt8PtrTy(), "captureContext", illustration.captureContext);
        bindings.emplace_back(typeSystem.getInt32Ty(), "selectionKind", illustration.selectionKind);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionRow", illustration.selectionRow);
        bindings.emplace_back(typeSystem.getSizeTy(), "selectionColumn", illustration.selectionColumn);
    }
    return bindings;
}

struct LowRankIllustrationMetadata {
    LowRankIllustrationPath path;
    std::uint32_t channel;
    Value * row;
    Value * groupStart;
    Value * rank;
    Value * horizontalTap;
    Value * verticalTap;
    Value * paddedSourceRow;
    Value * sourceColumn;
    Value * lane;
};

class LowRankIllustrationEmitter {
   public:
    LowRankIllustrationEmitter(
        KernelBuilder & builder,
        Value * captureContext,
        Value * selectionKind,
        Value * selectionRow,
        Value * selectionColumn,
        const unsigned imageWidth,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
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
          laneCount(laneCount) {
        captureFunction = builder.getModule()->getFunction("captureLowRankConvFilterValue");
        if (captureFunction == nullptr)
            throw std::logic_error("LowRank illustration callback is not declared");
    }

    Value * horizontalGroupSelected(Value * row, Value * groupStart) const {
        Value * outputSelected = builder.CreateAnd(isOutputSelection(), builder.CreateICmpEQ(row, selectionRow));
        outputSelected = builder.CreateAnd(outputSelected, groupContains(groupStart, selectionColumn));

        Value * inputSelected = builder.CreateAnd(isInputSelection(), builder.CreateICmpEQ(row, selectionRow));
        inputSelected = builder.CreateAnd(inputSelected, groupContainsInputWindow(groupStart));
        return builder.CreateOr(outputSelected, inputSelected);
    }

    Value * verticalGroupSelected(Value * row, Value * groupStart) const {
        Value * outputSelected = builder.CreateAnd(isOutputSelection(), builder.CreateICmpEQ(row, selectionRow));
        outputSelected = builder.CreateAnd(outputSelected, groupContains(groupStart, selectionColumn));

        const unsigned radius = kernelHeight / 2U;
        Value * selectedRow = builder.CreateICmpULE(row, builder.CreateAdd(selectionRow, builder.getSize(radius)));
        selectedRow = builder.CreateAnd(selectedRow, builder.CreateICmpULE(selectionRow, builder.CreateAdd(row, builder.getSize(radius))));
        Value * inputSelected = builder.CreateAnd(isInputSelection(), selectedRow);
        inputSelected = builder.CreateAnd(inputSelected, groupContainsInputWindow(groupStart));
        return builder.CreateOr(outputSelected, inputSelected);
    }

    LowRankIllustrationMetadata metadata(
        const LowRankIllustrationPath path,
        Value * row,
        Value * groupStart,
        Value * rank = nullptr,
        Value * horizontalTap = nullptr,
        Value * verticalTap = nullptr,
        Value * paddedSourceRow = nullptr,
        Value * sourceColumn = nullptr,
        Value * lane = nullptr,
        const std::uint32_t channel = NoLowRankIllustrationChannel
    ) const {
        Value * absent = builder.getSize(NoLowRankIllustrationCoordinate);
        return {
            path,
            channel,
            row == nullptr ? absent : row,
            groupStart == nullptr ? absent : groupStart,
            rank == nullptr ? absent : rank,
            horizontalTap == nullptr ? absent : horizontalTap,
            verticalTap == nullptr ? absent : verticalTap,
            paddedSourceRow == nullptr ? absent : paddedSourceRow,
            sourceColumn == nullptr ? absent : sourceColumn,
            lane == nullptr ? absent : lane,
        };
    }

    void capture(Value * condition, const LowRankIllustrationEvent event, Value * value, const LowRankIllustrationMetadata & metadata) const {
        std::size_t elementCount = 1U;
        Type * elementType = value->getType();
        if (auto * vectorType = dyn_cast<FixedVectorType>(value->getType())) {
            elementCount = vectorType->getNumElements();
            elementType = vectorType->getElementType();
        }

        LowRankIllustrationElementType captureType;
        std::size_t elementByteWidth;
        if (elementType->isFloatTy()) {
            captureType = LowRankIllustrationElementType::Float32;
            elementByteWidth = sizeof(float);
        } else if (elementType->isIntegerTy(8)) {
            captureType = LowRankIllustrationElementType::UInt8;
            elementByteWidth = sizeof(std::uint8_t);
        } else {
            throw std::logic_error("LowRank illustration capture element type is unsupported");
        }
        if (captureType != lowRankIllustrationElementType(event) || elementByteWidth != lowRankIllustrationElementByteWidth(captureType))
            throw std::logic_error("LowRank illustration capture event type mismatch");

        BasicBlock * captureBlock = builder.CreateBasicBlock("capture_low_rank_value");
        BasicBlock * continueBlock = builder.CreateBasicBlock("after_low_rank_capture");
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
             builder.getInt32(static_cast<std::uint32_t>(captureType)),
             builder.getInt32(metadata.channel),
             builder.getSize(elementCount),
             builder.getSize(elementByteWidth),
             builder.CreatePointerCast(storage, builder.getInt8PtrTy()),
             metadata.row,
             metadata.groupStart,
             metadata.rank,
             metadata.horizontalTap,
             metadata.verticalTap,
             metadata.paddedSourceRow,
             metadata.sourceColumn,
             metadata.lane}
        );
        builder.CreateBr(continueBlock);
        builder.SetInsertPoint(continueBlock);
    }

   private:
    Value * isInputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Input)));
    }

    Value * isOutputSelection() const {
        return builder.CreateICmpEQ(selectionKind, builder.getInt32(static_cast<std::uint32_t>(ConvFilterIllustrationSelectionKind::Output)));
    }

    Value * groupContains(Value * groupStart, Value * column) const {
        Value * selected = builder.CreateICmpUGE(column, groupStart);
        return builder.CreateAnd(selected, builder.CreateICmpULT(column, builder.CreateAdd(groupStart, builder.getSize(laneCount))));
    }

    Value * groupContainsInputWindow(Value * groupStart) const {
        Value * selected = builder.getInt1(false);
        const unsigned radius = kernelWidth / 2U;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            Value * outputColumn = builder.CreateAdd(groupStart, builder.getSize(lane));
            Value * laneSelected = builder.CreateICmpULT(outputColumn, builder.getSize(imageWidth));
            laneSelected =
                builder.CreateAnd(laneSelected, builder.CreateICmpULE(outputColumn, builder.CreateAdd(selectionColumn, builder.getSize(radius))));
            laneSelected =
                builder.CreateAnd(laneSelected, builder.CreateICmpULE(selectionColumn, builder.CreateAdd(outputColumn, builder.getSize(radius))));
            selected = builder.CreateOr(selected, laneSelected);
        }
        return selected;
    }

    KernelBuilder & builder;
    Value * const captureContext;
    Value * const selectionKind;
    Value * const selectionRow;
    Value * const selectionColumn;
    const unsigned imageWidth;
    const unsigned kernelWidth;
    const unsigned kernelHeight;
    const unsigned laneCount;
    Function * captureFunction = nullptr;
};

std::uint64_t computeConfigurationHash(const LowRankKernelConfiguration & configuration) {
    std::uint64_t hash = KernelNameHashInitialValue;
    hash = appendKernelNameHashBytes(
        hash,
        configuration.horizontalFactors.data(),
        checkedMultiply(configuration.horizontalFactors.size(), sizeof(float), "low-rank kernel-name horizontal byte count overflow")
    );
    hash = appendKernelNameHashBytes(
        hash,
        configuration.verticalFactors.data(),
        checkedMultiply(configuration.verticalFactors.size(), sizeof(float), "low-rank kernel-name vertical byte count overflow")
    );
    return appendKernelNameHashBytes(hash, &configuration.rankCount, sizeof(configuration.rankCount));
}

class LowRankConvolutionKernel final : public SegmentOrientedKernel {
   public:
    LowRankConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * workspace,
        Scalar * horizontalFactors,
        Scalar * verticalFactors,
        const LowRankKernelConfiguration & configuration,
        const std::string & persistentIdentity,
        const IllustrationScalars illustration = {}
    )
        : SegmentOrientedKernel(
              typeSystem,
              std::string(illustration ? "illustrated_low_rank_runtime_" : "low_rank_runtime_") + std::to_string(configuration.imageWidth) + "x"
                  + std::to_string(configuration.imageHeight) + "_k" + std::to_string(configuration.kernelHeight) + "x"
                  + std::to_string(configuration.kernelWidth) + "_h" + std::to_string(computeConfigurationHash(configuration)) + "_c"
                  + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              lowRankScalarBindings(typeSystem, inputPixels, outputPixels, workspace, horizontalFactors, verticalFactors, illustration),
              {},
              {}
          ),
          imageWidth(configuration.imageWidth),
          imageHeight(configuration.imageHeight),
          kernelHeight(configuration.kernelHeight),
          kernelWidth(configuration.kernelWidth),
          rankCount(configuration.rankCount),
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
             builder.getSizeTy(),
             builder.getSizeTy()},
            false
        );
        builder.LinkFunction("captureLowRankConvFilterValue", captureType, reinterpret_cast<void *>(&captureLowRankConvFilterValue));
    }

    Value * pixelByteOffset(KernelBuilder & builder, Value * row, Value * column, const unsigned channel = 0) const {
        Value * offset = builder.CreateMul(row, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channel));
    }

    std::array<Value *, ColorChannelCount> loadPackedFloatGroup(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * sourceRow,
        Value * sourceColumn,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition,
        const LowRankIllustrationMetadata & metadata
    ) const {
        auto * interleavedType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        Value * bytePointer = builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, sourceRow, sourceColumn));
        Value * interleaved = nullptr;
        if (laneCount == 8U) {
            auto * firstType = FixedVectorType::get(builder.getInt8Ty(), 16);
            auto * secondType = FixedVectorType::get(builder.getInt8Ty(), 8);
            LoadInst * first = builder.CreateLoad(firstType, builder.CreatePointerCast(bytePointer, firstType->getPointerTo()));
            LoadInst * second = builder.CreateLoad(
                secondType,
                builder.CreatePointerCast(builder.CreateGEP(builder.getInt8Ty(), bytePointer, builder.getSize(16)), secondType->getPointerTo())
            );
            first->setAlignment(Align(1));
            second->setAlignment(Align(1));
            SmallVector<int, 16> extendIndexes;
            for (unsigned index = 0; index < 8; ++index) {
                extendIndexes.push_back(static_cast<int>(index));
            }
            extendIndexes.resize(16, -1);
            Value * extendedSecond = builder.CreateShuffleVector(second, UndefValue::get(secondType), extendIndexes);
            SmallVector<int, 32> combineIndexes;
            for (unsigned index = 0; index < 24; ++index) {
                combineIndexes.push_back(static_cast<int>(index));
            }
            interleaved = builder.CreateShuffleVector(first, extendedSecond, combineIndexes);
        } else {
            Value * inputPointer = builder.CreatePointerCast(bytePointer, interleavedType->getPointerTo());
            LoadInst * load = builder.CreateLoad(interleavedType, inputPointer);
            load->setAlignment(Align(1));
            interleaved = load;
        }
        if (illustration != nullptr)
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::HorizontalPackedInput, interleaved, metadata);
        auto * integerType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        std::array<Value *, ColorChannelCount> channels;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            SmallVector<int, 16> indexes;
            for (unsigned lane = 0; lane < laneCount; ++lane) {
                indexes.push_back(static_cast<int>(lane * ColorChannelCount + channel));
            }
            Value * channelBytes = builder.CreateShuffleVector(interleaved, UndefValue::get(interleavedType), indexes);
            channels[channel] = builder.CreateUIToFP(builder.CreateZExt(channelBytes, integerType), floatType);
        }
        return channels;
    }

    Value * interleaveWorkspaceChannels(
        KernelBuilder & builder, const std::array<Value *, ColorChannelCount> & channels, const unsigned laneCount
    ) const {
        auto * channelType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        SmallVector<int, 16> redGreenIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            redGreenIndexes.push_back(static_cast<int>(lane));
            redGreenIndexes.push_back(static_cast<int>(laneCount + lane));
        }
        Value * redGreen = builder.CreateShuffleVector(channels[0], channels[1], redGreenIndexes);
        SmallVector<int, 16> blueIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            blueIndexes.push_back(static_cast<int>(lane));
        }
        blueIndexes.resize(laneCount * 2U, -1);
        Value * blue = builder.CreateShuffleVector(channels[2], UndefValue::get(channelType), blueIndexes);
        SmallVector<int, 32> interleavedIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            interleavedIndexes.push_back(static_cast<int>(lane * 2U));
            interleavedIndexes.push_back(static_cast<int>(lane * 2U + 1U));
            interleavedIndexes.push_back(static_cast<int>(laneCount * 2U + lane));
        }
        return builder.CreateShuffleVector(redGreen, blue, interleavedIndexes);
    }

    std::array<Value *, ColorChannelCount> extractWorkspaceChannels(KernelBuilder & builder, Value * interleaved, const unsigned laneCount) const {
        auto * interleavedType = FixedVectorType::get(builder.getFloatTy(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channels;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            SmallVector<int, 16> indexes;
            for (unsigned lane = 0; lane < laneCount; ++lane) {
                indexes.push_back(static_cast<int>(lane * ColorChannelCount + channel));
            }
            channels[channel] = builder.CreateShuffleVector(interleaved, UndefValue::get(interleavedType), indexes);
        }
        return channels;
    }

    Value * loadBorderFloatGroup(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * row,
        Value * columnGroupStart,
        const unsigned channel,
        Value * kernelRow,
        Value * kernelColumn,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition,
        const LowRankIllustrationMetadata & metadata
    ) const {
        Value * samples = Constant::getNullValue(FixedVectorType::get(builder.getFloatTy(), laneCount));
        const unsigned verticalRadius = kernelHeight / 2U;
        const unsigned horizontalRadius = kernelWidth / 2U;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(lane));
            Value * paddedRow = builder.CreateAdd(row, kernelRow);
            Value * paddedColumn = builder.CreateAdd(column, kernelColumn);
            Value * valid = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            valid = builder.CreateAnd(valid, builder.CreateICmpUGE(paddedRow, builder.getSize(verticalRadius)));
            valid = builder.CreateAnd(valid, builder.CreateICmpUGE(paddedColumn, builder.getSize(horizontalRadius)));
            valid = builder.CreateAnd(
                valid,
                builder.CreateICmpULT(paddedRow, builder.getSize(checkedAdd(imageHeight, verticalRadius, "low-rank padded row extent overflow")))
            );
            valid = builder.CreateAnd(
                valid,
                builder.CreateICmpULT(
                    paddedColumn, builder.getSize(checkedAdd(imageWidth, horizontalRadius, "low-rank padded column extent overflow"))
                )
            );
            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("border_sample_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("border_sample_join");
            builder.CreateCondBr(valid, loadBlock, joinBlock);
            builder.SetInsertPoint(loadBlock);
            Value * sourceRow = builder.CreateSub(paddedRow, builder.getSize(verticalRadius));
            Value * sourceColumn = builder.CreateSub(paddedColumn, builder.getSize(horizontalRadius));
            Value * byteValue = builder.CreateLoad(
                builder.getInt8Ty(), builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, sourceRow, sourceColumn, channel))
            );
            if (illustration != nullptr) {
                LowRankIllustrationMetadata byteMetadata = metadata;
                byteMetadata.sourceColumn = sourceColumn;
                byteMetadata.paddedSourceRow = sourceRow;
                byteMetadata.lane = builder.getSize(lane);
                illustration->capture(illustrationCondition, LowRankIllustrationEvent::HorizontalInputByte, byteValue, byteMetadata);
            }
            Value * sample = builder.CreateUIToFP(builder.CreateZExt(byteValue, builder.getInt32Ty()), builder.getFloatTy());
            BasicBlock * loadedBlock = builder.GetInsertBlock();
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
            PHINode * selected = builder.CreatePHI(builder.getFloatTy(), 2);
            selected->addIncoming(ConstantFP::get(builder.getFloatTy(), 0.0F), zeroBlock);
            selected->addIncoming(sample, loadedBlock);
            samples = builder.CreateInsertElement(samples, selected, builder.getInt32(lane));
        }
        return samples;
    }

    Value * clampAndRound(KernelBuilder & builder, Value * accumulator, const unsigned laneCount) const {
        Value * zero = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.0F));
        Value * maximum = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 255.0F));
        Value * rounding = builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), 0.5F));
        Value * lower = builder.CreateSelect(builder.CreateFCmpOLT(accumulator, zero), zero, accumulator);
        Value * bounded = builder.CreateSelect(builder.CreateFCmpOGT(lower, maximum), maximum, lower);
        auto * integerType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        return builder.CreateFPToUI(builder.CreateFAdd(bounded, rounding), integerType);
    }

    void storePackedGroup(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const std::array<Value *, ColorChannelCount> & roundedChannels,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        auto * interleavedType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channelBytes;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            channelBytes[channel] = builder.CreateTrunc(roundedChannels[channel], channelByteType);
            if (illustration != nullptr) {
                const auto metadata = illustration->metadata(
                    LowRankIllustrationPath::OutputFull, row, columnGroupStart, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, channel
                );
                illustration->capture(illustrationCondition, LowRankIllustrationEvent::OutputByteVector, channelBytes[channel], metadata);
            }
        }
        SmallVector<int, 16> redGreenIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            redGreenIndexes.push_back(static_cast<int>(lane));
            redGreenIndexes.push_back(static_cast<int>(laneCount + lane));
        }
        Value * redGreen = builder.CreateShuffleVector(channelBytes[0], channelBytes[1], redGreenIndexes);
        SmallVector<int, 16> blueIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            blueIndexes.push_back(static_cast<int>(lane));
        }
        blueIndexes.resize(laneCount * 2U, -1);
        Value * blue = builder.CreateShuffleVector(channelBytes[2], UndefValue::get(channelByteType), blueIndexes);
        SmallVector<int, 32> interleavedIndexes;
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            interleavedIndexes.push_back(static_cast<int>(lane * 2U));
            interleavedIndexes.push_back(static_cast<int>(lane * 2U + 1U));
            interleavedIndexes.push_back(static_cast<int>(laneCount * 2U + lane));
        }
        Value * interleaved = builder.CreateShuffleVector(redGreen, blue, interleavedIndexes);
        Value * outputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, columnGroupStart)), interleavedType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(interleaved, outputPointer);
        store->setAlignment(Align(1));
        if (illustration != nullptr) {
            const auto metadata = illustration->metadata(LowRankIllustrationPath::OutputFull, row, columnGroupStart);
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::PackedOutput, interleaved, metadata);
            Value * storeMask = Constant::getAllOnesValue(FixedVectorType::get(builder.getInt8Ty(), laneCount));
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::StoreMask, storeMask, metadata);
        }
    }

    void storeCheckedGroup(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * row,
        Value * columnGroupStart,
        const std::array<Value *, ColorChannelCount> & roundedChannels,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        Value * storeMask =
            illustration == nullptr ? nullptr : static_cast<Value *>(Constant::getNullValue(FixedVectorType::get(builder.getInt8Ty(), laneCount)));
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            Value * column = builder.CreateAdd(columnGroupStart, builder.getSize(lane));
            Value * valid = builder.CreateICmpULT(column, builder.getSize(imageWidth));
            if (storeMask != nullptr)
                storeMask = builder.CreateInsertElement(storeMask, builder.CreateZExt(valid, builder.getInt8Ty()), builder.getInt32(lane));
            BasicBlock * storeBlock = builder.CreateBasicBlock("checked_output_store");
            BasicBlock * joinBlock = builder.CreateBasicBlock("output_store_join");
            builder.CreateCondBr(valid, storeBlock, joinBlock);
            builder.SetInsertPoint(storeBlock);
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                Value * byteValue =
                    builder.CreateTrunc(builder.CreateExtractElement(roundedChannels[channel], builder.getInt32(lane)), builder.getInt8Ty());
                if (illustration != nullptr) {
                    const auto metadata = illustration->metadata(
                        LowRankIllustrationPath::OutputChecked,
                        row,
                        columnGroupStart,
                        nullptr,
                        nullptr,
                        nullptr,
                        nullptr,
                        column,
                        builder.getSize(lane),
                        channel
                    );
                    illustration->capture(illustrationCondition, LowRankIllustrationEvent::StoredOutputByte, byteValue, metadata);
                }
                builder.CreateStore(byteValue, builder.CreateGEP(builder.getInt8Ty(), outputPixels, pixelByteOffset(builder, row, column, channel)));
            }
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
        }
        if (illustration != nullptr) {
            const auto metadata = illustration->metadata(LowRankIllustrationPath::OutputChecked, row, columnGroupStart);
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::StoreMask, storeMask, metadata);
        }
    }

    const unsigned imageWidth;
    const unsigned imageHeight;
    const unsigned kernelHeight;
    const unsigned kernelWidth;
    Value * workspaceOffset(KernelBuilder & builder, Value * rank, Value * row, Value * column, const unsigned channel = 0) const {
        // The full-image workspace is ordered by rank, row, column, then channel.
        Value * offset = builder.CreateMul(rank, builder.getSize(imageHeight));
        offset = builder.CreateAdd(offset, row);
        offset = builder.CreateMul(offset, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channel));
    }

    Value * broadcastFloat(KernelBuilder & builder, Value * scalarValue, const unsigned laneCount) const {
        auto * vectorType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Value * seed = builder.CreateInsertElement(UndefValue::get(vectorType), scalarValue, builder.getInt32(0));
        SmallVector<int, 16> indexes(laneCount, 0);
        return builder.CreateShuffleVector(seed, UndefValue::get(vectorType), indexes);
    }

    void storeWorkspacePacked(
        KernelBuilder & builder,
        Value * workspace,
        Value * rank,
        Value * row,
        Value * column,
        const std::array<Value *, ColorChannelCount> & channels,
        const unsigned laneCount
    ) const {
        auto * interleavedType = FixedVectorType::get(builder.getFloatTy(), laneCount * ColorChannelCount);
        Value * interleaved = interleaveWorkspaceChannels(builder, channels, laneCount);
        Value * floatWorkspace = builder.CreatePointerCast(workspace, builder.getFloatTy()->getPointerTo());
        Value * workspacePointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getFloatTy(), floatWorkspace, workspaceOffset(builder, rank, row, column)), interleavedType->getPointerTo()
        );
        StoreInst * store = builder.CreateStore(interleaved, workspacePointer);
        store->setAlignment(Align(4));
    }

    void storeWorkspaceChecked(
        KernelBuilder & builder,
        Value * workspace,
        Value * rank,
        Value * row,
        Value * column,
        const std::array<Value *, ColorChannelCount> & channels,
        const unsigned laneCount
    ) const {
        Value * floatWorkspace = builder.CreatePointerCast(workspace, builder.getFloatTy()->getPointerTo());
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            Value * laneColumn = builder.CreateAdd(column, builder.getSize(lane));
            Value * valid = builder.CreateICmpULT(laneColumn, builder.getSize(imageWidth));
            BasicBlock * storeBlock = builder.CreateBasicBlock("low_rank_runtime_workspace_store");
            BasicBlock * joinBlock = builder.CreateBasicBlock("low_rank_runtime_workspace_store_done");
            builder.CreateCondBr(valid, storeBlock, joinBlock);
            builder.SetInsertPoint(storeBlock);
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                builder.CreateStore(
                    builder.CreateExtractElement(channels[channel], builder.getInt32(lane)),
                    builder.CreateGEP(builder.getFloatTy(), floatWorkspace, workspaceOffset(builder, rank, row, laneColumn, channel))
                );
            }
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
        }
    }

    void storeWorkspace(
        KernelBuilder & builder,
        Value * workspace,
        Value * rank,
        Value * row,
        Value * column,
        const std::array<Value *, ColorChannelCount> & channels,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        if (illustration != nullptr) {
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                const auto metadata = illustration->metadata(
                    LowRankIllustrationPath::HorizontalResult, row, column, rank, nullptr, nullptr, nullptr, nullptr, nullptr, channel
                );
                illustration->capture(illustrationCondition, LowRankIllustrationEvent::HorizontalWorkspaceStore, channels[channel], metadata);
            }
        }
        Value * full = builder.CreateICmpULE(builder.CreateAdd(column, builder.getSize(laneCount)), builder.getSize(imageWidth));
        BasicBlock * packedBlock = builder.CreateBasicBlock("low_rank_runtime_workspace_packed");
        BasicBlock * checkedBlock = builder.CreateBasicBlock("low_rank_runtime_workspace_checked");
        BasicBlock * joinBlock = builder.CreateBasicBlock("low_rank_runtime_workspace_done");
        builder.CreateCondBr(full, packedBlock, checkedBlock);
        builder.SetInsertPoint(packedBlock);
        storeWorkspacePacked(builder, workspace, rank, row, column, channels, laneCount);
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(checkedBlock);
        storeWorkspaceChecked(builder, workspace, rank, row, column, channels, laneCount);
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(joinBlock);
    }

    std::array<Value *, ColorChannelCount> loadWorkspacePacked(
        KernelBuilder & builder, Value * workspace, Value * rank, Value * row, Value * column, const unsigned laneCount
    ) const {
        auto * interleavedType = FixedVectorType::get(builder.getFloatTy(), laneCount * ColorChannelCount);
        Value * floatWorkspace = builder.CreatePointerCast(workspace, builder.getFloatTy()->getPointerTo());
        Value * workspacePointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getFloatTy(), floatWorkspace, workspaceOffset(builder, rank, row, column)), interleavedType->getPointerTo()
        );
        LoadInst * interleaved = builder.CreateLoad(interleavedType, workspacePointer);
        interleaved->setAlignment(Align(4));
        return extractWorkspaceChannels(builder, interleaved, laneCount);
    }

    std::array<Value *, ColorChannelCount> loadWorkspace(
        KernelBuilder & builder,
        Value * workspace,
        Value * rank,
        Value * paddedRow,
        Value * column,
        const bool checkedColumns,
        const unsigned laneCount
    ) const {
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        const unsigned verticalRadius = kernelHeight / 2U;
        Value * validRow = builder.CreateICmpUGE(paddedRow, builder.getSize(verticalRadius));
        validRow = builder.CreateAnd(
            validRow,
            builder.CreateICmpULT(paddedRow, builder.getSize(checkedAdd(imageHeight, verticalRadius, "low-rank vertical padded extent overflow")))
        );
        if (!checkedColumns) {
            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_load_done");
            builder.CreateCondBr(validRow, loadBlock, joinBlock);
            builder.SetInsertPoint(loadBlock);
            auto loaded =
                loadWorkspacePacked(builder, workspace, rank, builder.CreateSub(paddedRow, builder.getSize(verticalRadius)), column, laneCount);
            BasicBlock * loadedBlock = builder.GetInsertBlock();
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
            std::array<Value *, ColorChannelCount> selected;
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                PHINode * selectedValue = builder.CreatePHI(floatType, 2);
                selectedValue->addIncoming(Constant::getNullValue(floatType), zeroBlock);
                selectedValue->addIncoming(loaded[channel], loadedBlock);
                selected[channel] = selectedValue;
            }
            return selected;
        }

        Value * floatWorkspace = builder.CreatePointerCast(workspace, builder.getFloatTy()->getPointerTo());
        std::array<Value *, ColorChannelCount> channels;
        channels.fill(Constant::getNullValue(floatType));
        for (unsigned lane = 0; lane < laneCount; ++lane) {
            Value * laneColumn = builder.CreateAdd(column, builder.getSize(lane));
            Value * valid = builder.CreateAnd(validRow, builder.CreateICmpULT(laneColumn, builder.getSize(imageWidth)));
            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_edge_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_edge_done");
            builder.CreateCondBr(valid, loadBlock, joinBlock);
            builder.SetInsertPoint(loadBlock);
            std::array<Value *, ColorChannelCount> loaded;
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                loaded[channel] = builder.CreateLoad(
                    builder.getFloatTy(),
                    builder.CreateGEP(
                        builder.getFloatTy(),
                        floatWorkspace,
                        workspaceOffset(builder, rank, builder.CreateSub(paddedRow, builder.getSize(verticalRadius)), laneColumn, channel)
                    )
                );
            }
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
            std::array<PHINode *, ColorChannelCount> selected;
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                selected[channel] = builder.CreatePHI(builder.getFloatTy(), 2);
                selected[channel]->addIncoming(ConstantFP::get(builder.getFloatTy(), 0.0F), zeroBlock);
                selected[channel]->addIncoming(loaded[channel], loadBlock);
            }
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                channels[channel] = builder.CreateInsertElement(channels[channel], selected[channel], builder.getInt32(lane));
            }
        }
        return channels;
    }

    std::array<Value *, ColorChannelCount> accumulateHorizontalPath(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * horizontalFactors,
        Value * rank,
        Value * row,
        Value * column,
        const bool border,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Function * multiplyAdd = Intrinsic::getDeclaration(builder.getModule(), Intrinsic::fmuladd, {floatType});
        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * tapLoop = builder.CreateBasicBlock("low_rank_runtime_horizontal_taps");
        BasicBlock * done = builder.CreateBasicBlock("low_rank_runtime_horizontal_taps_done");
        builder.CreateBr(tapLoop);
        builder.SetInsertPoint(tapLoop);
        PHINode * tap = builder.CreatePHI(builder.getSizeTy(), 2);
        tap->addIncoming(builder.getSize(0), entry);
        std::array<PHINode *, ColorChannelCount> accumulators;
        for (PHINode *& accumulator : accumulators) {
            accumulator = builder.CreatePHI(floatType, 2);
            accumulator->addIncoming(Constant::getNullValue(floatType), entry);
        }
        std::array<Value *, ColorChannelCount> samples;
        const LowRankIllustrationPath illustrationPath =
            border ? LowRankIllustrationPath::HorizontalBorder : LowRankIllustrationPath::HorizontalInterior;
        const auto tapMetadata =
            illustration == nullptr ? LowRankIllustrationMetadata{} : illustration->metadata(illustrationPath, row, column, rank, tap);
        if (border) {
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                LowRankIllustrationMetadata channelMetadata = tapMetadata;
                channelMetadata.channel = channel;
                samples[channel] = loadBorderFloatGroup(
                    builder,
                    inputPixels,
                    row,
                    column,
                    channel,
                    builder.getSize(kernelHeight / 2U),
                    tap,
                    laneCount,
                    illustration,
                    illustrationCondition,
                    channelMetadata
                );
            }
        } else {
            Value * sourceColumn = builder.CreateSub(builder.CreateAdd(column, tap), builder.getSize(kernelWidth / 2U));
            LowRankIllustrationMetadata packedMetadata = tapMetadata;
            packedMetadata.sourceColumn = sourceColumn;
            samples = loadPackedFloatGroup(builder, inputPixels, row, sourceColumn, laneCount, illustration, illustrationCondition, packedMetadata);
        }
        if (illustration != nullptr) {
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                LowRankIllustrationMetadata channelMetadata = tapMetadata;
                channelMetadata.channel = channel;
                illustration->capture(illustrationCondition, LowRankIllustrationEvent::HorizontalSample, samples[channel], channelMetadata);
            }
        }
        Value * factorIndex = builder.CreateAdd(builder.CreateMul(rank, builder.getSize(kernelWidth)), tap);
        LoadInst * factor = builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), horizontalFactors, factorIndex));
        factor->setAlignment(Align(4));
        if (illustration != nullptr)
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::HorizontalFactor, factor, tapMetadata);
        Value * weight = broadcastFloat(builder, factor, laneCount);
        std::array<Value *, ColorChannelCount> nextAccumulators;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            nextAccumulators[channel] =
                builder.CreateCall(multiplyAdd->getFunctionType(), multiplyAdd, {samples[channel], weight, accumulators[channel]});
            if (illustration != nullptr) {
                LowRankIllustrationMetadata channelMetadata = tapMetadata;
                channelMetadata.channel = channel;
                illustration->capture(
                    illustrationCondition, LowRankIllustrationEvent::HorizontalAccumulator, nextAccumulators[channel], channelMetadata
                );
            }
        }
        Value * nextTap = builder.CreateAdd(tap, builder.getSize(1));
        BasicBlock * loopBack = builder.GetInsertBlock();
        builder.CreateCondBr(builder.CreateICmpULT(nextTap, builder.getSize(kernelWidth)), tapLoop, done);
        tap->addIncoming(nextTap, loopBack);
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            accumulators[channel]->addIncoming(nextAccumulators[channel], loopBack);
        }
        builder.SetInsertPoint(done);
        return nextAccumulators;
    }

    std::array<Value *, ColorChannelCount> accumulateHorizontal(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * horizontalFactors,
        Value * rank,
        Value * row,
        Value * column,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        const unsigned radius = kernelWidth / 2U;
        Value * interior = builder.CreateICmpUGE(column, builder.getSize(radius));
        interior = builder.CreateAnd(
            interior,
            builder.CreateICmpULT(
                builder.CreateAdd(builder.CreateAdd(column, builder.getSize(laneCount - 1U)), builder.getSize(radius)), builder.getSize(imageWidth)
            )
        );
        BasicBlock * interiorBlock = builder.CreateBasicBlock("low_rank_runtime_horizontal_interior");
        BasicBlock * borderBlock = builder.CreateBasicBlock("low_rank_runtime_horizontal_border");
        BasicBlock * joinBlock = builder.CreateBasicBlock("low_rank_runtime_horizontal_path_done");
        builder.CreateCondBr(interior, interiorBlock, borderBlock);
        builder.SetInsertPoint(interiorBlock);
        auto interiorAccumulators = accumulateHorizontalPath(
            builder, inputPixels, horizontalFactors, rank, row, column, false, laneCount, illustration, illustrationCondition
        );
        BasicBlock * interiorDone = builder.GetInsertBlock();
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(borderBlock);
        auto borderAccumulators = accumulateHorizontalPath(
            builder, inputPixels, horizontalFactors, rank, row, column, true, laneCount, illustration, illustrationCondition
        );
        BasicBlock * borderDone = builder.GetInsertBlock();
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(joinBlock);
        std::array<Value *, ColorChannelCount> selected;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            PHINode * selectedValue = builder.CreatePHI(FixedVectorType::get(builder.getFloatTy(), laneCount), 2);
            selectedValue->addIncoming(interiorAccumulators[channel], interiorDone);
            selectedValue->addIncoming(borderAccumulators[channel], borderDone);
            selected[channel] = selectedValue;
        }
        return selected;
    }

    void generateHorizontalPass(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * workspace,
        const unsigned laneCount,
        BasicBlock * entry,
        BasicBlock * verticalEntry,
        LowRankIllustrationEmitter * illustration
    ) const {
        builder.SetInsertPoint(entry);
        Value * horizontalFactors = builder.getScalarField("horizontalFactors");
        BasicBlock * rowLoop = builder.CreateBasicBlock("low_rank_runtime_horizontal_rows");
        BasicBlock * doneRows = builder.CreateBasicBlock("low_rank_runtime_horizontal_done");
        builder.CreateBr(rowLoop);
        builder.SetInsertPoint(rowLoop);
        PHINode * row = builder.CreatePHI(builder.getSizeTy(), 2);
        row->addIncoming(builder.getSize(0), entry);
        BasicBlock * columnLoop = builder.CreateBasicBlock("low_rank_runtime_horizontal_columns");
        BasicBlock * doneColumns = builder.CreateBasicBlock("low_rank_runtime_horizontal_columns_done");
        builder.CreateBr(columnLoop);
        builder.SetInsertPoint(columnLoop);
        PHINode * column = builder.CreatePHI(builder.getSizeTy(), 2);
        column->addIncoming(builder.getSize(0), rowLoop);
        BasicBlock * rankLoop = builder.CreateBasicBlock("low_rank_runtime_horizontal_ranks");
        BasicBlock * groupDone = builder.CreateBasicBlock("low_rank_runtime_horizontal_group_done");
        builder.CreateBr(rankLoop);
        builder.SetInsertPoint(rankLoop);
        PHINode * rank = builder.CreatePHI(builder.getSizeTy(), 2);
        rank->addIncoming(builder.getSize(0), columnLoop);
        Value * illustrationCondition = illustration == nullptr ? nullptr : illustration->horizontalGroupSelected(row, column);
        auto accumulators =
            accumulateHorizontal(builder, inputPixels, horizontalFactors, rank, row, column, laneCount, illustration, illustrationCondition);
        storeWorkspace(builder, workspace, rank, row, column, accumulators, laneCount, illustration, illustrationCondition);
        Value * nextRank = builder.CreateAdd(rank, builder.getSize(1));
        BasicBlock * rankBack = builder.GetInsertBlock();
        builder.CreateCondBr(builder.CreateICmpULT(nextRank, builder.getSize(rankCount)), rankLoop, groupDone);
        rank->addIncoming(nextRank, rankBack);
        builder.SetInsertPoint(groupDone);
        Value * nextColumn = builder.CreateAdd(column, builder.getSize(laneCount));
        column->addIncoming(nextColumn, groupDone);
        builder.CreateCondBr(builder.CreateICmpULT(nextColumn, builder.getSize(imageWidth)), columnLoop, doneColumns);
        builder.SetInsertPoint(doneColumns);
        Value * nextRow = builder.CreateAdd(row, builder.getSize(1));
        row->addIncoming(nextRow, doneColumns);
        builder.CreateCondBr(builder.CreateICmpULT(nextRow, builder.getSize(imageHeight)), rowLoop, doneRows);
        builder.SetInsertPoint(doneRows);
        builder.CreateBr(verticalEntry);
    }

    std::array<Value *, ColorChannelCount> accumulateVertical(
        KernelBuilder & builder,
        Value * workspace,
        Value * verticalFactors,
        Value * row,
        Value * column,
        const bool checkedColumns,
        const unsigned laneCount,
        LowRankIllustrationEmitter * illustration,
        Value * illustrationCondition
    ) const {
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Function * multiplyAdd = Intrinsic::getDeclaration(builder.getModule(), Intrinsic::fmuladd, {floatType});
        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * tapLoop = builder.CreateBasicBlock("low_rank_runtime_vertical_taps");
        BasicBlock * done = builder.CreateBasicBlock("low_rank_runtime_vertical_taps_done");
        builder.CreateBr(tapLoop);
        builder.SetInsertPoint(tapLoop);
        PHINode * tap = builder.CreatePHI(builder.getSizeTy(), 2);
        tap->addIncoming(builder.getSize(0), entry);
        std::array<PHINode *, ColorChannelCount> accumulators;
        for (PHINode *& accumulator : accumulators) {
            accumulator = builder.CreatePHI(floatType, 2);
            accumulator->addIncoming(Constant::getNullValue(floatType), entry);
        }
        Value * rank = builder.CreateUDiv(tap, builder.getSize(kernelHeight));
        Value * kernelRow = builder.CreateURem(tap, builder.getSize(kernelHeight));
        Value * paddedSourceRow = builder.CreateAdd(row, kernelRow);
        auto samples = loadWorkspace(builder, workspace, rank, paddedSourceRow, column, checkedColumns, laneCount);
        const LowRankIllustrationPath illustrationPath =
            checkedColumns ? LowRankIllustrationPath::VerticalEdge : LowRankIllustrationPath::VerticalFull;
        const auto tapMetadata = illustration == nullptr
                                     ? LowRankIllustrationMetadata{}
                                     : illustration->metadata(illustrationPath, row, column, rank, nullptr, kernelRow, paddedSourceRow);
        if (illustration != nullptr) {
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                LowRankIllustrationMetadata channelMetadata = tapMetadata;
                channelMetadata.channel = channel;
                illustration->capture(illustrationCondition, LowRankIllustrationEvent::VerticalWorkspaceLoad, samples[channel], channelMetadata);
            }
        }
        LoadInst * factor = builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), verticalFactors, tap));
        factor->setAlignment(Align(4));
        if (illustration != nullptr)
            illustration->capture(illustrationCondition, LowRankIllustrationEvent::VerticalFactor, factor, tapMetadata);
        Value * weight = broadcastFloat(builder, factor, laneCount);
        std::array<Value *, ColorChannelCount> nextAccumulators;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            nextAccumulators[channel] =
                builder.CreateCall(multiplyAdd->getFunctionType(), multiplyAdd, {samples[channel], weight, accumulators[channel]});
            if (illustration != nullptr) {
                LowRankIllustrationMetadata channelMetadata = tapMetadata;
                channelMetadata.channel = channel;
                illustration->capture(
                    illustrationCondition, LowRankIllustrationEvent::VerticalAccumulator, nextAccumulators[channel], channelMetadata
                );
            }
        }
        Value * nextTap = builder.CreateAdd(tap, builder.getSize(1));
        BasicBlock * loopBack = builder.GetInsertBlock();
        builder.CreateCondBr(
            builder.CreateICmpULT(nextTap, builder.getSize(checkedMultiply(rankCount, kernelHeight, "low-rank vertical tap count overflow"))),
            tapLoop,
            done
        );
        tap->addIncoming(nextTap, loopBack);
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            accumulators[channel]->addIncoming(nextAccumulators[channel], loopBack);
        }
        builder.SetInsertPoint(done);
        return nextAccumulators;
    }

    void generateVerticalPass(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * workspace,
        const unsigned laneCount,
        BasicBlock * entry,
        BasicBlock * done,
        LowRankIllustrationEmitter * illustration
    ) const {
        builder.SetInsertPoint(entry);
        Value * verticalFactors = builder.getScalarField("verticalFactors");
        BasicBlock * rowLoop = builder.CreateBasicBlock("low_rank_runtime_vertical_rows");
        builder.CreateBr(rowLoop);
        builder.SetInsertPoint(rowLoop);
        PHINode * row = builder.CreatePHI(builder.getSizeTy(), 2);
        row->addIncoming(builder.getSize(0), entry);
        BasicBlock * columnLoop = builder.CreateBasicBlock("low_rank_runtime_vertical_columns");
        BasicBlock * doneColumns = builder.CreateBasicBlock("low_rank_runtime_vertical_columns_done");
        builder.CreateBr(columnLoop);
        builder.SetInsertPoint(columnLoop);
        PHINode * column = builder.CreatePHI(builder.getSizeTy(), 2);
        column->addIncoming(builder.getSize(0), rowLoop);
        Value * full = builder.CreateICmpULE(builder.CreateAdd(column, builder.getSize(laneCount)), builder.getSize(imageWidth));
        BasicBlock * fullBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_full");
        BasicBlock * edgeBlock = builder.CreateBasicBlock("low_rank_runtime_vertical_edge");
        BasicBlock * groupDone = builder.CreateBasicBlock("low_rank_runtime_vertical_group_done");
        builder.CreateCondBr(full, fullBlock, edgeBlock);
        builder.SetInsertPoint(fullBlock);
        Value * fullIllustrationCondition = illustration == nullptr ? nullptr : illustration->verticalGroupSelected(row, column);
        auto fullAccumulators =
            accumulateVertical(builder, workspace, verticalFactors, row, column, false, laneCount, illustration, fullIllustrationCondition);
        for (Value *& roundedChannel : fullAccumulators) {
            roundedChannel = clampAndRound(builder, roundedChannel, laneCount);
        }
        storePackedGroup(builder, outputPixels, row, column, fullAccumulators, laneCount, illustration, fullIllustrationCondition);
        builder.CreateBr(groupDone);
        builder.SetInsertPoint(edgeBlock);
        Value * edgeIllustrationCondition = illustration == nullptr ? nullptr : illustration->verticalGroupSelected(row, column);
        auto edgeAccumulators =
            accumulateVertical(builder, workspace, verticalFactors, row, column, true, laneCount, illustration, edgeIllustrationCondition);
        for (Value *& roundedChannel : edgeAccumulators) {
            roundedChannel = clampAndRound(builder, roundedChannel, laneCount);
        }
        storeCheckedGroup(builder, outputPixels, row, column, edgeAccumulators, laneCount, illustration, edgeIllustrationCondition);
        builder.CreateBr(groupDone);
        builder.SetInsertPoint(groupDone);
        Value * nextColumn = builder.CreateAdd(column, builder.getSize(laneCount));
        column->addIncoming(nextColumn, groupDone);
        builder.CreateCondBr(builder.CreateICmpULT(nextColumn, builder.getSize(imageWidth)), columnLoop, doneColumns);
        builder.SetInsertPoint(doneColumns);
        Value * nextRow = builder.CreateAdd(row, builder.getSize(1));
        row->addIncoming(nextRow, doneColumns);
        builder.CreateCondBr(builder.CreateICmpULT(nextRow, builder.getSize(imageHeight)), rowLoop, done);
        builder.SetInsertPoint(done);
    }

    void generateDoSegmentMethod(KernelBuilder & builder) final {
        Value * inputPixels = builder.getScalarField("inputPixels");
        Value * outputPixels = builder.getScalarField("outputPixels");
        Value * workspace = builder.getScalarField("workspace");
        const unsigned laneCount = builder.getBitBlockWidth() / 32U;
        std::optional<LowRankIllustrationEmitter> illustration;
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
                laneCount
            );
        }
        LowRankIllustrationEmitter * const illustrationEmitter = illustration ? &*illustration : nullptr;
        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * verticalEntry = builder.CreateBasicBlock("low_rank_runtime_vertical_entry");
        BasicBlock * done = builder.CreateBasicBlock("low_rank_runtime_done");
        generateHorizontalPass(builder, inputPixels, workspace, laneCount, entry, verticalEntry, illustrationEmitter);
        generateVerticalPass(builder, outputPixels, workspace, laneCount, verticalEntry, done, illustrationEmitter);
    }

    const unsigned rankCount;
    const bool illustrated;
};

using LowRankFunction = void (*)(const uint8_t *, uint8_t *, uint8_t *, const float *, const float *, uint8_t *, std::size_t);
using IllustratedLowRankFunction = void (*)(
    const uint8_t *, uint8_t *, uint8_t *, const float *, const float *, uint8_t *, std::uint32_t, std::size_t, std::size_t, uint8_t *, std::size_t
);

class LowRankFilterImplementation final : public CompiledFilterImplementation {
   public:
    LowRankFilterImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const unsigned rank,
        std::vector<float> horizontalFactorValues,
        std::vector<float> verticalFactorValues,
        const std::string & persistentIdentity
    )
        : CompiledFilterImplementation(
              ConvFilterMode::LowRank,
              imageWidth,
              imageHeight,
              checkedImageByteCount(imageWidth, imageHeight),
              checkedLowRankWorkspaceByteCount(imageWidth, imageHeight, rank),
              alignof(float)
          ),
          driver(std::move(cpuDriver)),
          horizontalFactors(std::move(horizontalFactorValues)),
          verticalFactors(std::move(verticalFactorValues)) {
        const LowRankKernelConfiguration configuration{imageWidth, imageHeight, kernelHeight, kernelWidth, horizontalFactors, verticalFactors, rank};
        auto pipeline = CreatePipeline(
            *driver,
            Input<const uint8_t *>("inputPixels"),
            Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("workspace"),
            Input<const float *>("horizontalFactors"),
            Input<const float *>("verticalFactors"),
            Input<uint8_t *>("triggerBuffer"),
            Input<std::size_t>("triggerLength")
        );
        StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<MemorySourceKernel>(
            pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
        );
        pipeline.CreateKernelCall<LowRankConvolutionKernel>(
            triggerStream,
            pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"),
            pipeline.getInputScalar("workspace"),
            pipeline.getInputScalar("horizontalFactors"),
            pipeline.getInputScalar("verticalFactors"),
            configuration,
            persistentIdentity
        );
        compiledFunction = pipeline.compile();
    }

   private:
    void invoke(const uint8_t * inputPixels, uint8_t * outputPixels, void * workspace) const noexcept final {
        uint8_t triggerByte = 0;
        compiledFunction(
            inputPixels, outputPixels, static_cast<uint8_t *>(workspace), horizontalFactors.data(), verticalFactors.data(), &triggerByte, 1U
        );
    }

    std::unique_ptr<CPUDriver> driver;
    const std::vector<float> horizontalFactors;
    const std::vector<float> verticalFactors;
    LowRankFunction compiledFunction = nullptr;
};

class LowRankFilterIllustrationImplementation final : public CompiledFilterIllustrationImplementation {
   public:
    LowRankFilterIllustrationImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const unsigned rank,
        std::vector<float> horizontalFactorValues,
        std::vector<float> verticalFactorValues,
        const std::string & persistentIdentity
    )
        : driver(std::move(cpuDriver)),
          imageWidthInPixels(imageWidth),
          imageHeightInPixels(imageHeight),
          kernelWidthInPixels(kernelWidth),
          kernelHeightInPixels(kernelHeight),
          rankCount(rank),
          imageByteCount(checkedImageByteCount(imageWidth, imageHeight)),
          workspaceByteCount(checkedLowRankWorkspaceByteCount(imageWidth, imageHeight, rank)),
          horizontalFactors(std::move(horizontalFactorValues)),
          verticalFactors(std::move(verticalFactorValues)),
          logicalOutputCount(driver->getBitBlockWidth() / 32U) {
        const LowRankKernelConfiguration configuration{
            imageWidth,
            imageHeight,
            kernelHeight,
            kernelWidth,
            horizontalFactors,
            verticalFactors,
            rank,
        };
        auto pipeline = CreatePipeline(
            *driver,
            Input<const uint8_t *>("inputPixels"),
            Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("workspace"),
            Input<const float *>("horizontalFactors"),
            Input<const float *>("verticalFactors"),
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
        pipeline.CreateKernelCall<LowRankConvolutionKernel>(
            triggerStream,
            pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"),
            pipeline.getInputScalar("workspace"),
            pipeline.getInputScalar("horizontalFactors"),
            pipeline.getInputScalar("verticalFactors"),
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
        return alignof(float);
    }

    bool apply(
        const std::uint8_t * input, std::uint8_t * output, void * workspace, const ConvFilterIllustrationSelection selection, std::string & trace
    ) const final {
        assert(input != nullptr);
        assert(output != nullptr);
        assert(workspace != nullptr);
        assert(reinterpret_cast<std::uintptr_t>(workspace) % alignof(float) == 0U);
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

        LowRankIllustrationCaptureLog capture;
        capture.logicalOutputs = logicalOutputCount;
        std::uint8_t triggerByte = 0;
        compiledFunction(
            input,
            output,
            static_cast<std::uint8_t *>(workspace),
            horizontalFactors.data(),
            verticalFactors.data(),
            reinterpret_cast<std::uint8_t *>(&capture),
            static_cast<std::uint32_t>(selection.kind),
            selection.row,
            selection.column,
            &triggerByte,
            1U
        );
        if (capture.failure)
            std::rethrow_exception(capture.failure);

        sortLowRankIllustrationEvents(capture);
        const LowRankIllustrationConfiguration configuration{
            imageWidthInPixels,
            imageHeightInPixels,
            kernelWidthInPixels,
            kernelHeightInPixels,
            rankCount,
            logicalOutputCount,
        };
        std::string completeTrace = formatLowRankConvFilterIllustration(configuration, selection, capture);
        trace = std::move(completeTrace);
        return true;
    }

   private:
    std::unique_ptr<CPUDriver> driver;
    const unsigned imageWidthInPixels;
    const unsigned imageHeightInPixels;
    const unsigned kernelWidthInPixels;
    const unsigned kernelHeightInPixels;
    const unsigned rankCount;
    const std::size_t imageByteCount;
    const std::size_t workspaceByteCount;
    const std::vector<float> horizontalFactors;
    const std::vector<float> verticalFactors;
    const unsigned logicalOutputCount;
    IllustratedLowRankFunction compiledFunction = nullptr;
};

}  // namespace

std::shared_ptr<const CompiledFilterImplementation> compileLowRankFilter(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const unsigned rank,
    std::vector<float> horizontalFactors,
    std::vector<float> verticalFactors,
    std::string persistentIdentity
) {
    return std::make_shared<LowRankFilterImplementation>(
        std::move(driver),
        imageWidth,
        imageHeight,
        kernelWidth,
        kernelHeight,
        rank,
        std::move(horizontalFactors),
        std::move(verticalFactors),
        persistentIdentity
    );
}

std::shared_ptr<const CompiledFilterIllustrationImplementation> compileLowRankFilterIllustration(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const unsigned rank,
    std::vector<float> horizontalFactors,
    std::vector<float> verticalFactors,
    std::string persistentIdentity
) {
    return std::make_shared<LowRankFilterIllustrationImplementation>(
        std::move(driver),
        imageWidth,
        imageHeight,
        kernelWidth,
        kernelHeight,
        rank,
        std::move(horizontalFactors),
        std::move(verticalFactors),
        persistentIdentity
    );
}

}  // namespace kernel::image::internal
