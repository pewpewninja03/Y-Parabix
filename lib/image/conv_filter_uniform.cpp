#include "conv_filter_common.h"

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>

#include <algorithm>
#include <array>
#include <cstring>
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

uint32_t floatBits(const float floatingValue) {
    uint32_t bits;
    std::memcpy(&bits, &floatingValue, sizeof(bits));
    return bits;
}

class UniformConvolutionKernel final : public SegmentOrientedKernel {
   public:
    UniformConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * workspace,
        const UniformKernelConfiguration & configuration,
        const std::string & persistentIdentity
    )
        : SegmentOrientedKernel(
              typeSystem,
              "uniform_convolution_" + std::to_string(configuration.imageWidth) + "x" + std::to_string(configuration.imageHeight) + "_k"
                  + std::to_string(configuration.kernelHeight) + "x" + std::to_string(configuration.kernelWidth) + "_w"
                  + std::to_string(floatBits(configuration.weight)) + "_c" + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              {Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels},
               Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels},
               Binding{typeSystem.getInt8PtrTy(), "workspace", workspace}},
              {},
              {}
          ),
          imageWidth(configuration.imageWidth),
          imageHeight(configuration.imageHeight),
          kernelHeight(configuration.kernelHeight),
          kernelWidth(configuration.kernelWidth),
          uniformWeight(configuration.weight) {
        addAttribute(SideEffecting());
    }

   private:
    Value * pixelByteOffset(KernelBuilder & builder, Value * row, Value * column, const unsigned channel) const {
        Value * offset = builder.CreateMul(row, builder.getSize(imageWidth));
        offset = builder.CreateAdd(offset, column);
        offset = builder.CreateMul(offset, builder.getSize(ColorChannelCount));
        return builder.CreateAdd(offset, builder.getSize(channel));
    }

    Value * convertWindowSum(KernelBuilder & builder, Value * sum, const unsigned laneCount) const {
        auto * integerType = FixedVectorType::get(builder.getInt32Ty(), laneCount);
        if (uniformWeight <= 0.0F)
            return Constant::getNullValue(integerType);
        auto * floatType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Value * weighted = builder.CreateFMul(
            builder.CreateSIToFP(sum, floatType), builder.getSplat(laneCount, ConstantFP::get(builder.getFloatTy(), uniformWeight))
        );
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
        const unsigned laneCount
    ) const {
        auto * channelByteType = FixedVectorType::get(builder.getInt8Ty(), laneCount);
        auto * interleavedType = FixedVectorType::get(builder.getInt8Ty(), laneCount * ColorChannelCount);
        std::array<Value *, ColorChannelCount> channelBytes;
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            channelBytes[channel] = builder.CreateTrunc(roundedChannels[channel], channelByteType);
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

    Value * loadPixel(KernelBuilder & builder, Value * inputPixels, Value * row, Value * column) const {
        auto * vectorType = FixedVectorType::get(builder.getInt32Ty(), 4);
        auto * packedType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
        Value * inputPointer = builder.CreatePointerCast(
            builder.CreateGEP(builder.getInt8Ty(), inputPixels, pixelByteOffset(builder, row, column, 0)), packedType->getPointerTo()
        );
        LoadInst * packedBytes = builder.CreateLoad(packedType, inputPointer);
        packedBytes->setAlignment(Align(1));
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

    Value * adjustHorizontalSum(KernelBuilder & builder, Value * sum, Value * workspace, Value * column, Value * valid, const bool addition) const {
        BasicBlock * unchangedBlock = builder.GetInsertBlock();
        BasicBlock * adjustBlock = builder.CreateBasicBlock(addition ? "add_entering_column" : "subtract_leaving_column");
        BasicBlock * joinBlock = builder.CreateBasicBlock("column_adjustment_join");
        builder.CreateCondBr(valid, adjustBlock, joinBlock);
        builder.SetInsertPoint(adjustBlock);
        Value * columnSum = loadWorkspace(builder, workspace, column);
        Value * adjusted = addition ? builder.CreateAdd(sum, columnSum) : builder.CreateSub(sum, columnSum);
        builder.CreateBr(joinBlock);
        builder.SetInsertPoint(joinBlock);
        PHINode * selectedSum = builder.CreatePHI(sum->getType(), 2);
        selectedSum->addIncoming(sum, unchangedBlock);
        selectedSum->addIncoming(adjusted, adjustBlock);
        return selectedSum;
    }

    void storeSinglePixel(KernelBuilder & builder, Value * outputPixels, Value * row, Value * column, Value * sum) const {
        Value * rounded = convertWindowSum(builder, sum, 4);
        auto * channelType = FixedVectorType::get(builder.getInt8Ty(), 4);
        auto * packedType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
        Value * channelBytes = builder.CreateTrunc(rounded, channelType);
        Value * packedBytes = builder.CreateShuffleVector(channelBytes, UndefValue::get(channelType), ArrayRef<int>({0, 1, 2}));
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
        const unsigned initialRowCount = std::min(imageHeight, kernelHeight / 2U + 1U);
        const unsigned initialColumnCount = std::min(imageWidth, kernelWidth / 2U + 1U);

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
        Value * nextInitialSum = builder.CreateAdd(initialSum, loadPixel(builder, inputPixels, initialRow, initialColumn));
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
        Value * nextWindowSum = builder.CreateAdd(windowSum, loadWorkspace(builder, workspace, windowColumn));
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
        const unsigned outputLaneCount = builder.getBitBlockWidth() / 32U;
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
        Value * nextGroupSum = rollingSum;
        SmallVector<Value *, 16> pixelSums(outputLaneCount);
        for (unsigned lane = 0; lane < outputLaneCount; ++lane) {
            pixelSums[lane] = nextGroupSum;
            Value * removeColumn = builder.CreateAdd(builder.CreateSub(column, builder.getSize(horizontalRadius)), builder.getSize(lane));
            Value * addColumn = builder.CreateAdd(column, builder.getSize(horizontalRadius + lane + 1U));
            nextGroupSum = builder.CreateSub(nextGroupSum, loadWorkspace(builder, workspace, removeColumn));
            nextGroupSum = builder.CreateAdd(nextGroupSum, loadWorkspace(builder, workspace, addColumn));
        }
        for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
            Value * channelSums = Constant::getNullValue(outputVectorType);
            for (unsigned lane = 0; lane < outputLaneCount; ++lane) {
                channelSums = builder.CreateInsertElement(
                    channelSums, builder.CreateExtractElement(pixelSums[lane], builder.getInt32(channel)), builder.getInt32(lane)
                );
            }
            groupedChannels[channel] = convertWindowSum(builder, channelSums, outputLaneCount);
        }
        storePackedGroup(builder, outputPixels, row, column, groupedChannels, outputLaneCount);
        Value * nextGroupedColumn = builder.CreateAdd(column, builder.getSize(outputLaneCount));
        BasicBlock * groupBack = builder.GetInsertBlock();
        builder.CreateBr(outputJoin);

        builder.SetInsertPoint(scalarBlock);
        storeSinglePixel(builder, outputPixels, row, column, rollingSum);
        Value * removeValid = builder.CreateICmpUGE(column, builder.getSize(horizontalRadius));
        Value * removeColumn = builder.CreateSub(column, builder.getSize(horizontalRadius));
        Value * nextScalarSum = adjustHorizontalSum(builder, rollingSum, workspace, removeColumn, removeValid, false);
        Value * addColumn = builder.CreateAdd(column, builder.getSize(horizontalRadius + 1U));
        Value * addValid = builder.CreateICmpULT(addColumn, builder.getSize(imageWidth));
        nextScalarSum = adjustHorizontalSum(builder, nextScalarSum, workspace, addColumn, addValid, true);
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
            if (addSample)
                updatedColumnSum = builder.CreateAdd(updatedColumnSum, loadPixel(builder, inputPixels, addRow, updateColumn));
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
};

using UniformFunction = void (*)(const uint8_t *, uint8_t *, uint8_t *, uint8_t *, std::size_t);

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

}  // namespace kernel::image::internal
