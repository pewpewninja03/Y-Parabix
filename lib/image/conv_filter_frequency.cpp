#include "conv_filter_common.h"

#include <kernel/core/kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Instructions.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

using namespace kernel;
using namespace llvm;

namespace kernel::image::internal {
namespace {

constexpr std::size_t FrequencyWorkspaceAlignment = 64U;

struct FrequencyKernelConfiguration {
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelHeight;
    unsigned kernelWidth;
    unsigned transformExtent;
    const std::vector<float> & weights;
};

uint64_t computeConfigurationHash(const FrequencyKernelConfiguration & configuration, const FrequencyLayout & layout, const unsigned bitBlockWidth) {
    uint64_t hash = 1469598103934665603ULL;
    const auto append = [&](const void * byteData, const std::size_t byteCount) {
        const auto * bytes = static_cast<const uint8_t *>(byteData);
        for (std::size_t index = 0; index < byteCount; ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    };
    append(&ArithmeticContractVersion, sizeof(ArithmeticContractVersion));
    append(&bitBlockWidth, sizeof(bitBlockWidth));
    append(&layout.logicalFloatLanes, sizeof(layout.logicalFloatLanes));
    append(&layout.independentVectorCount, sizeof(layout.independentVectorCount));
    append(&configuration.imageWidth, sizeof(configuration.imageWidth));
    append(&configuration.imageHeight, sizeof(configuration.imageHeight));
    append(&configuration.kernelHeight, sizeof(configuration.kernelHeight));
    append(&configuration.kernelWidth, sizeof(configuration.kernelWidth));
    append(&configuration.transformExtent, sizeof(configuration.transformExtent));
    append(
        configuration.weights.data(), checkedMultiply(configuration.weights.size(), sizeof(float), "frequency kernel-name weight byte count overflow")
    );
    return hash;
}

std::vector<Binding> scalarBindings(
    LLVMTypeSystemInterface & typeSystem,
    Scalar * inputPixels,
    Scalar * outputPixels,
    Scalar * workspace,
    Scalar * bitReverse,
    Scalar * weights,
    Scalar * control,
    Scalar * frequencyData
) {
    return {
        Binding{typeSystem.getInt8PtrTy(), "inputPixels", inputPixels},
        Binding{typeSystem.getInt8PtrTy(), "outputPixels", outputPixels},
        Binding{typeSystem.getInt8PtrTy(), "workspace", workspace},
        Binding{typeSystem.getInt32PtrTy(), "bitReverse", bitReverse},
        Binding{typeSystem.getFloatTy()->getPointerTo(), "weights", weights},
        Binding{typeSystem.getInt32PtrTy(), "control", control},
        Binding{typeSystem.getFloatTy()->getPointerTo(), "frequencyData", frequencyData}
    };
}

class FrequencyConvolutionKernel final : public SegmentOrientedKernel {
   public:
    FrequencyConvolutionKernel(
        LLVMTypeSystemInterface & typeSystem,
        StreamSet * triggerStream,
        Scalar * inputPixels,
        Scalar * outputPixels,
        Scalar * workspace,
        Scalar * bitReverse,
        Scalar * weights,
        Scalar * control,
        Scalar * frequencyData,
        const FrequencyKernelConfiguration & configuration,
        const FrequencyLayout & layout,
        const unsigned bitBlockWidth,
        const std::string & persistentIdentity
    )
        : SegmentOrientedKernel(
              typeSystem,
              "frequency_overlap_save_target_adaptive_b" + std::to_string(bitBlockWidth) + "_l" + std::to_string(layout.logicalFloatLanes) + "_v"
                  + std::to_string(layout.independentVectorCount) + "_" + std::to_string(configuration.imageWidth) + "x"
                  + std::to_string(configuration.imageHeight) + "_k" + std::to_string(configuration.kernelHeight) + "_n"
                  + std::to_string(configuration.transformExtent) + "_h"
                  + std::to_string(computeConfigurationHash(configuration, layout, bitBlockWidth)) + "_c" + persistentIdentity,
              {Binding{"triggerStream", triggerStream}},
              {},
              scalarBindings(typeSystem, inputPixels, outputPixels, workspace, bitReverse, weights, control, frequencyData),
              {},
              {}
          ),
          imageWidth(configuration.imageWidth),
          imageHeight(configuration.imageHeight),
          kernelExtent(configuration.kernelHeight),
          transformExtent(configuration.transformExtent),
          logicalFloatLanes(layout.logicalFloatLanes),
          independentVectorCount(layout.independentVectorCount),
          tileCountPerVector(layout.tilesPerVector),
          tileCountPerGroup(layout.tilesPerGroup) {
        addAttribute(SideEffecting());
    }

   private:
    FixedVectorType * floatVectorType(KernelBuilder & builder, const unsigned laneCount) const {
        return FixedVectorType::get(builder.getFloatTy(), laneCount);
    }

    template <typename Body>
    void emitLoop(KernelBuilder & builder, const std::string & loopName, const uint64_t iterationCount, const Body & emitBody) const {
        BasicBlock * entry = builder.GetInsertBlock();
        BasicBlock * loop = builder.CreateBasicBlock(loopName);
        BasicBlock * done = builder.CreateBasicBlock(loopName + "_done");
        builder.CreateBr(loop);
        builder.SetInsertPoint(loop);
        PHINode * index = builder.CreatePHI(builder.getSizeTy(), 2);
        index->addIncoming(builder.getSize(0), entry);
        emitBody(index);
        BasicBlock * latch = builder.GetInsertBlock();
        Value * next = builder.CreateAdd(index, builder.getSize(1));
        index->addIncoming(next, latch);
        builder.CreateCondBr(builder.CreateICmpULT(next, builder.getSize(iterationCount)), loop, done);
        builder.SetInsertPoint(done);
    }

    Value * splat(KernelBuilder & builder, Value * scalarValue, const unsigned laneCount) const {
        FixedVectorType * vectorType = floatVectorType(builder, laneCount);
        Value * seed = builder.CreateInsertElement(UndefValue::get(vectorType), scalarValue, builder.getInt32(0));
        SmallVector<int, 16> indexes(laneCount, 0);
        return builder.CreateShuffleVector(seed, UndefValue::get(vectorType), indexes);
    }

    Value * complexPartPointer(
        KernelBuilder & builder, Value * base, Value * index, const unsigned laneCount, const unsigned vectorIndex, const bool imaginary
    ) const {
        const uint64_t pointCount = static_cast<uint64_t>(transformExtent) * transformExtent;
        const uint64_t vectorStride = pointCount * 2U * laneCount;
        const uint64_t planeOffset = imaginary ? pointCount * laneCount : 0U;
        Value * floatBase = builder.CreatePointerCast(base, builder.getFloatTy()->getPointerTo());
        Value * offset = builder.CreateAdd(
            builder.getSize(static_cast<uint64_t>(vectorIndex) * vectorStride + planeOffset), builder.CreateMul(index, builder.getSize(laneCount))
        );
        return builder.CreatePointerCast(
            builder.CreateGEP(builder.getFloatTy(), floatBase, offset), floatVectorType(builder, laneCount)->getPointerTo()
        );
    }

    Value * loadComplexPart(
        KernelBuilder & builder, Value * base, Value * index, const unsigned laneCount, const unsigned vectorIndex, const bool imaginary
    ) const {
        FixedVectorType * vectorType = floatVectorType(builder, laneCount);
        LoadInst * load = builder.CreateLoad(vectorType, complexPartPointer(builder, base, index, laneCount, vectorIndex, imaginary));
        load->setAlignment(Align(4));
        return load;
    }

    void storeComplexPart(
        KernelBuilder & builder,
        Value * base,
        Value * index,
        Value * complexPart,
        const unsigned laneCount,
        const unsigned vectorIndex,
        const bool imaginary
    ) const {
        StoreInst * store = builder.CreateStore(complexPart, complexPartPointer(builder, base, index, laneCount, vectorIndex, imaginary));
        store->setAlignment(Align(4));
    }

    Value * multiplySubtract(KernelBuilder & builder, Value * left, Value * right, Value * subtractLeft, Value * subtractRight) const {
        return builder.CreateFSub(builder.CreateFMul(left, right), builder.CreateFMul(subtractLeft, subtractRight));
    }

    Value * multiplyAdd(KernelBuilder & builder, Value * left, Value * right, Value * addLeft, Value * addRight) const {
        return builder.CreateFAdd(builder.CreateFMul(left, right), builder.CreateFMul(addLeft, addRight));
    }

    void bitReverseWorkspace(
        KernelBuilder & builder, Value * workspace, Value * bitReverse, const unsigned laneCount, const unsigned vectorCount, const std::string & name
    ) const {
        const uint64_t pointCount = static_cast<uint64_t>(transformExtent) * transformExtent;
        emitLoop(builder, name, pointCount, [&](Value * index) {
            Value * row = builder.CreateUDiv(index, builder.getSize(transformExtent));
            Value * column = builder.CreateURem(index, builder.getSize(transformExtent));
            Value * reversedRow = builder.CreateZExt(
                builder.CreateLoad(builder.getInt32Ty(), builder.CreateGEP(builder.getInt32Ty(), bitReverse, row)), builder.getSizeTy()
            );
            Value * reversedColumn = builder.CreateZExt(
                builder.CreateLoad(builder.getInt32Ty(), builder.CreateGEP(builder.getInt32Ty(), bitReverse, column)), builder.getSizeTy()
            );
            Value * target = builder.CreateAdd(builder.CreateMul(reversedRow, builder.getSize(transformExtent)), reversedColumn);
            BasicBlock * swap = builder.CreateBasicBlock(name + "_swap");
            BasicBlock * join = builder.CreateBasicBlock(name + "_join");
            builder.CreateCondBr(builder.CreateICmpUGT(target, index), swap, join);
            builder.SetInsertPoint(swap);
            for (unsigned vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex) {
                Value * real = loadComplexPart(builder, workspace, index, laneCount, vectorIndex, false);
                Value * imaginary = loadComplexPart(builder, workspace, index, laneCount, vectorIndex, true);
                Value * targetReal = loadComplexPart(builder, workspace, target, laneCount, vectorIndex, false);
                Value * targetImaginary = loadComplexPart(builder, workspace, target, laneCount, vectorIndex, true);
                storeComplexPart(builder, workspace, index, targetReal, laneCount, vectorIndex, false);
                storeComplexPart(builder, workspace, index, targetImaginary, laneCount, vectorIndex, true);
                storeComplexPart(builder, workspace, target, real, laneCount, vectorIndex, false);
                storeComplexPart(builder, workspace, target, imaginary, laneCount, vectorIndex, true);
            }
            builder.CreateBr(join);
            builder.SetInsertPoint(join);
        });
    }

    void emitTransform(
        KernelBuilder & builder,
        Value * workspace,
        Value * frequencyData,
        const unsigned laneCount,
        const unsigned vectorCount,
        const bool inverse,
        const std::string & name
    ) const {
        const uint64_t pointCount = static_cast<uint64_t>(transformExtent) * transformExtent;
        Value * twiddleReal = builder.CreateGEP(builder.getFloatTy(), frequencyData, builder.getSize(pointCount * 2U));
        Value * twiddleImaginary = builder.CreateGEP(builder.getFloatTy(), frequencyData, builder.getSize(pointCount * 2U + transformExtent / 2U));
        for (unsigned dimension = 0; dimension < 2U; ++dimension) {
            for (unsigned stageExtent = 2U; stageExtent <= transformExtent; stageExtent *= 2U) {
                const unsigned half = stageExtent / 2U;
                const unsigned groupCount = transformExtent / stageExtent;
                const uint64_t butterflyCount = pointCount / 2U;
                emitLoop(
                    builder, name + "_d" + std::to_string(dimension) + "_s" + std::to_string(stageExtent), butterflyCount, [&](Value * butterfly) {
                        Value * within = builder.CreateURem(butterfly, builder.getSize(half));
                        Value * groupAndLine = builder.CreateUDiv(butterfly, builder.getSize(half));
                        Value * group = builder.CreateURem(groupAndLine, builder.getSize(groupCount));
                        Value * line = builder.CreateUDiv(groupAndLine, builder.getSize(groupCount));
                        Value * position = builder.CreateAdd(builder.CreateMul(group, builder.getSize(stageExtent)), within);
                        Value * first = dimension == 0U ? builder.CreateAdd(builder.CreateMul(line, builder.getSize(transformExtent)), position)
                                                        : builder.CreateAdd(builder.CreateMul(position, builder.getSize(transformExtent)), line);
                        Value * second = dimension == 0U ? builder.CreateAdd(first, builder.getSize(half))
                                                         : builder.CreateAdd(first, builder.getSize(static_cast<uint64_t>(half) * transformExtent));
                        Value * twiddleIndex = builder.CreateMul(within, builder.getSize(transformExtent / stageExtent));
                        Value * realScalar =
                            builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), twiddleReal, twiddleIndex));
                        Value * imaginaryScalar =
                            builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), twiddleImaginary, twiddleIndex));
                        if (inverse)
                            imaginaryScalar = builder.CreateFNeg(imaginaryScalar);
                        Value * realTwiddle = splat(builder, realScalar, laneCount);
                        Value * imaginaryTwiddle = splat(builder, imaginaryScalar, laneCount);
                        for (unsigned vectorIndex = 0; vectorIndex < vectorCount; ++vectorIndex) {
                            Value * firstReal = loadComplexPart(builder, workspace, first, laneCount, vectorIndex, false);
                            Value * firstImaginary = loadComplexPart(builder, workspace, first, laneCount, vectorIndex, true);
                            Value * secondReal = loadComplexPart(builder, workspace, second, laneCount, vectorIndex, false);
                            Value * secondImaginary = loadComplexPart(builder, workspace, second, laneCount, vectorIndex, true);
                            Value * productReal = multiplySubtract(builder, secondReal, realTwiddle, secondImaginary, imaginaryTwiddle);
                            Value * productImaginary = multiplyAdd(builder, secondReal, imaginaryTwiddle, secondImaginary, realTwiddle);
                            storeComplexPart(builder, workspace, first, builder.CreateFAdd(firstReal, productReal), laneCount, vectorIndex, false);
                            storeComplexPart(
                                builder, workspace, first, builder.CreateFAdd(firstImaginary, productImaginary), laneCount, vectorIndex, true
                            );
                            storeComplexPart(builder, workspace, second, builder.CreateFSub(firstReal, productReal), laneCount, vectorIndex, false);
                            storeComplexPart(
                                builder, workspace, second, builder.CreateFSub(firstImaginary, productImaginary), laneCount, vectorIndex, true
                            );
                        }
                    }
                );
            }
        }
    }

    void initializeKernelSpectrum(KernelBuilder & builder, Value * weights, Value * bitReverse, Value * frequencyData) const {
        const uint64_t pointCount = static_cast<uint64_t>(transformExtent) * transformExtent;
        FixedVectorType * singleLaneType = FixedVectorType::get(builder.getFloatTy(), 1);
        Value * spectrumWorkspace = builder.CreatePointerCast(frequencyData, builder.getInt8PtrTy());
        emitLoop(builder, "frequency_kernel_initialize", pointCount, [&](Value * index) {
            Value * row = builder.CreateUDiv(index, builder.getSize(transformExtent));
            Value * column = builder.CreateURem(index, builder.getSize(transformExtent));
            Value * valid = builder.CreateAnd(
                builder.CreateICmpULT(row, builder.getSize(kernelExtent)), builder.CreateICmpULT(column, builder.getSize(kernelExtent))
            );
            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("frequency_kernel_weight_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("frequency_kernel_weight_join");
            builder.CreateCondBr(valid, loadBlock, joinBlock);
            builder.SetInsertPoint(loadBlock);
            Value * flippedRow = builder.CreateSub(builder.getSize(kernelExtent - 1U), row);
            Value * flippedColumn = builder.CreateSub(builder.getSize(kernelExtent - 1U), column);
            Value * weightIndex = builder.CreateAdd(builder.CreateMul(flippedRow, builder.getSize(kernelExtent)), flippedColumn);
            Value * weight = builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), weights, weightIndex));
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
            PHINode * selected = builder.CreatePHI(builder.getFloatTy(), 2);
            selected->addIncoming(ConstantFP::get(builder.getFloatTy(), 0.0F), zeroBlock);
            selected->addIncoming(weight, loadBlock);
            Value * real = builder.CreateInsertElement(UndefValue::get(singleLaneType), selected, builder.getInt32(0));
            storeComplexPart(builder, spectrumWorkspace, index, real, 1U, 0U, false);
            storeComplexPart(builder, spectrumWorkspace, index, Constant::getNullValue(singleLaneType), 1U, 0U, true);
        });
        bitReverseWorkspace(builder, spectrumWorkspace, bitReverse, 1U, 1U, "frequency_kernel_bit_reverse");
        emitTransform(builder, spectrumWorkspace, frequencyData, 1U, 1U, false, "frequency_kernel_forward");
    }

    struct PackedSamples {
        Value * real;
        Value * imaginary;
    };

    PackedSamples loadPackedSamples(
        KernelBuilder & builder,
        Value * inputPixels,
        Value * tileRow,
        Value * tileColumn,
        Value * pointRow,
        Value * pointColumn,
        const unsigned vectorIndex
    ) const {
        FixedVectorType * vectorType = floatVectorType(builder, logicalFloatLanes);
        Value * realSamples = Constant::getNullValue(vectorType);
        Value * imaginarySamples = Constant::getNullValue(vectorType);
        const unsigned validExtent = transformExtent - kernelExtent + 1U;
        const unsigned radius = kernelExtent / 2U;
        const std::size_t tileColumnCount = checkedCeilDivide(imageWidth, validExtent, "frequency tile-column count overflow");
        const std::size_t paddedHeight = checkedAdd(imageHeight, radius, "frequency padded row extent overflow");
        const std::size_t paddedWidth = checkedAdd(imageWidth, radius, "frequency padded column extent overflow");
        // Each tile uses two complex lanes: R + iG and B + i0.
        for (unsigned tile = 0; tile < tileCountPerVector; ++tile) {
            const unsigned tileOffset = vectorIndex * tileCountPerVector + tile;
            Value * currentTileColumn = builder.CreateAdd(tileColumn, builder.getSize(tileOffset));
            Value * paddedRow = builder.CreateAdd(builder.CreateMul(tileRow, builder.getSize(validExtent)), pointRow);
            Value * paddedColumn = builder.CreateAdd(builder.CreateMul(currentTileColumn, builder.getSize(validExtent)), pointColumn);
            Value * valid = builder.CreateICmpULT(currentTileColumn, builder.getSize(tileColumnCount));
            valid = builder.CreateAnd(valid, builder.CreateICmpUGE(paddedRow, builder.getSize(radius)));
            valid = builder.CreateAnd(valid, builder.CreateICmpUGE(paddedColumn, builder.getSize(radius)));
            valid = builder.CreateAnd(valid, builder.CreateICmpULT(paddedRow, builder.getSize(paddedHeight)));
            valid = builder.CreateAnd(valid, builder.CreateICmpULT(paddedColumn, builder.getSize(paddedWidth)));
            BasicBlock * zeroBlock = builder.GetInsertBlock();
            BasicBlock * loadBlock = builder.CreateBasicBlock("frequency_input_load");
            BasicBlock * joinBlock = builder.CreateBasicBlock("frequency_input_join");
            builder.CreateCondBr(valid, loadBlock, joinBlock);
            builder.SetInsertPoint(loadBlock);
            Value * sourceRow = builder.CreateSub(paddedRow, builder.getSize(radius));
            Value * sourceColumn = builder.CreateSub(paddedColumn, builder.getSize(radius));
            Value * pixel = builder.CreateAdd(builder.CreateMul(sourceRow, builder.getSize(imageWidth)), sourceColumn);
            Value * byteOffset = builder.CreateMul(pixel, builder.getSize(ColorChannelCount));
            FixedVectorType * rgbType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
            Value * inputPointer =
                builder.CreatePointerCast(builder.CreateGEP(builder.getInt8Ty(), inputPixels, byteOffset), rgbType->getPointerTo());
            LoadInst * rgb = builder.CreateLoad(rgbType, inputPointer);
            rgb->setAlignment(Align(1));
            Value * loadedReal = realSamples;
            Value * loadedImaginary = imaginarySamples;
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                Value * sample = builder.CreateUIToFP(
                    builder.CreateZExt(builder.CreateExtractElement(rgb, builder.getInt32(channel)), builder.getInt32Ty()), builder.getFloatTy()
                );
                if (channel == 1U) {
                    loadedImaginary = builder.CreateInsertElement(loadedImaginary, sample, builder.getInt32(tile * 2U));
                } else {
                    loadedReal = builder.CreateInsertElement(loadedReal, sample, builder.getInt32(tile * 2U + (channel == 2U ? 1U : 0U)));
                }
            }
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
            PHINode * selectedReal = builder.CreatePHI(vectorType, 2);
            selectedReal->addIncoming(realSamples, zeroBlock);
            selectedReal->addIncoming(loadedReal, loadBlock);
            PHINode * selectedImaginary = builder.CreatePHI(vectorType, 2);
            selectedImaginary->addIncoming(imaginarySamples, zeroBlock);
            selectedImaginary->addIncoming(loadedImaginary, loadBlock);
            realSamples = selectedReal;
            imaginarySamples = selectedImaginary;
        }
        return {realSamples, imaginarySamples};
    }

    Value * clampAndRound(KernelBuilder & builder, Value * values, const unsigned laneCount) const {
        FixedVectorType * vectorType = FixedVectorType::get(builder.getFloatTy(), laneCount);
        Value * zero = Constant::getNullValue(vectorType);
        Value * maximum = splat(builder, ConstantFP::get(builder.getFloatTy(), 255.0F), laneCount);
        Value * rounding = splat(builder, ConstantFP::get(builder.getFloatTy(), 0.5F), laneCount);
        Value * ordered = builder.CreateSelect(builder.CreateFCmpORD(values, values), values, zero);
        Value * lower = builder.CreateSelect(builder.CreateFCmpOLT(ordered, zero), zero, ordered);
        Value * bounded = builder.CreateSelect(builder.CreateFCmpOGT(lower, maximum), maximum, lower);
        return builder.CreateFPToUI(builder.CreateFAdd(bounded, rounding), FixedVectorType::get(builder.getInt32Ty(), laneCount));
    }

    void storeOutput(
        KernelBuilder & builder,
        Value * outputPixels,
        Value * tileRow,
        Value * tileColumn,
        Value * outputRow,
        Value * outputColumn,
        Value * realValues,
        Value * imaginaryValues,
        const unsigned vectorIndex
    ) const {
        const unsigned validExtent = transformExtent - kernelExtent + 1U;
        FixedVectorType * rgbFloatType = FixedVectorType::get(builder.getFloatTy(), ColorChannelCount);
        FixedVectorType * rgbByteType = FixedVectorType::get(builder.getInt8Ty(), ColorChannelCount);
        for (unsigned tile = 0; tile < tileCountPerVector; ++tile) {
            const unsigned tileOffset = vectorIndex * tileCountPerVector + tile;
            Value * currentTileColumn = builder.CreateAdd(tileColumn, builder.getSize(tileOffset));
            Value * row = builder.CreateAdd(builder.CreateMul(tileRow, builder.getSize(validExtent)), outputRow);
            Value * column = builder.CreateAdd(builder.CreateMul(currentTileColumn, builder.getSize(validExtent)), outputColumn);
            Value * valid = builder.CreateAnd(
                builder.CreateICmpULT(row, builder.getSize(imageHeight)), builder.CreateICmpULT(column, builder.getSize(imageWidth))
            );
            BasicBlock * storeBlock = builder.CreateBasicBlock("frequency_output_store");
            BasicBlock * joinBlock = builder.CreateBasicBlock("frequency_output_join");
            builder.CreateCondBr(valid, storeBlock, joinBlock);
            builder.SetInsertPoint(storeBlock);
            Value * channels = Constant::getNullValue(rgbFloatType);
            channels =
                builder.CreateInsertElement(channels, builder.CreateExtractElement(realValues, builder.getInt32(tile * 2U)), builder.getInt32(0));
            channels = builder.CreateInsertElement(
                channels, builder.CreateExtractElement(imaginaryValues, builder.getInt32(tile * 2U)), builder.getInt32(1)
            );
            channels = builder.CreateInsertElement(
                channels, builder.CreateExtractElement(realValues, builder.getInt32(tile * 2U + 1U)), builder.getInt32(2)
            );
            Value * rounded = clampAndRound(builder, channels, ColorChannelCount);
            Value * rgb = UndefValue::get(rgbByteType);
            for (unsigned channel = 0; channel < ColorChannelCount; ++channel) {
                Value * byte = builder.CreateTrunc(builder.CreateExtractElement(rounded, builder.getInt32(channel)), builder.getInt8Ty());
                rgb = builder.CreateInsertElement(rgb, byte, builder.getInt32(channel));
            }
            Value * pixel = builder.CreateAdd(builder.CreateMul(row, builder.getSize(imageWidth)), column);
            Value * byteOffset = builder.CreateMul(pixel, builder.getSize(ColorChannelCount));
            Value * outputPointer =
                builder.CreatePointerCast(builder.CreateGEP(builder.getInt8Ty(), outputPixels, byteOffset), rgbByteType->getPointerTo());
            StoreInst * store = builder.CreateStore(rgb, outputPointer);
            store->setAlignment(Align(1));
            builder.CreateBr(joinBlock);
            builder.SetInsertPoint(joinBlock);
        }
    }

    void processImage(
        KernelBuilder & builder, Value * inputPixels, Value * outputPixels, Value * workspace, Value * bitReverse, Value * frequencyData
    ) const {
        const uint64_t pointCount = static_cast<uint64_t>(transformExtent) * transformExtent;
        const unsigned validExtent = transformExtent - kernelExtent + 1U;
        const std::size_t tileRows = checkedCeilDivide(imageHeight, validExtent, "frequency tile-row count overflow");
        const std::size_t tileColumns = checkedCeilDivide(imageWidth, validExtent, "frequency tile-column count overflow");
        const std::size_t tileColumnGroups = checkedCeilDivide(tileColumns, tileCountPerGroup, "frequency tile-column group count overflow");
        const std::size_t tileGroupCount = checkedMultiply(tileRows, tileColumnGroups, "frequency tile-group count overflow");
        emitLoop(builder, "frequency_tile_groups", tileGroupCount, [&](Value * tileGroup) {
            Value * tileRow = builder.CreateUDiv(tileGroup, builder.getSize(tileColumnGroups));
            Value * tileColumn =
                builder.CreateMul(builder.CreateURem(tileGroup, builder.getSize(tileColumnGroups)), builder.getSize(tileCountPerGroup));
            emitLoop(builder, "frequency_input_initialize", pointCount, [&](Value * index) {
                Value * pointRow = builder.CreateUDiv(index, builder.getSize(transformExtent));
                Value * pointColumn = builder.CreateURem(index, builder.getSize(transformExtent));
                for (unsigned vectorIndex = 0; vectorIndex < independentVectorCount; ++vectorIndex) {
                    PackedSamples samples = loadPackedSamples(builder, inputPixels, tileRow, tileColumn, pointRow, pointColumn, vectorIndex);
                    storeComplexPart(builder, workspace, index, samples.real, logicalFloatLanes, vectorIndex, false);
                    storeComplexPart(builder, workspace, index, samples.imaginary, logicalFloatLanes, vectorIndex, true);
                }
            });
            bitReverseWorkspace(builder, workspace, bitReverse, logicalFloatLanes, independentVectorCount, "frequency_forward_bit_reverse");
            emitTransform(builder, workspace, frequencyData, logicalFloatLanes, independentVectorCount, false, "frequency_forward");
            emitLoop(builder, "frequency_products", pointCount, [&](Value * index) {
                Value * kernelReal = splat(
                    builder,
                    builder.CreateLoad(builder.getFloatTy(), builder.CreateGEP(builder.getFloatTy(), frequencyData, index)),
                    logicalFloatLanes
                );
                Value * kernelImaginary = splat(
                    builder,
                    builder.CreateLoad(
                        builder.getFloatTy(),
                        builder.CreateGEP(builder.getFloatTy(), frequencyData, builder.CreateAdd(index, builder.getSize(pointCount)))
                    ),
                    logicalFloatLanes
                );
                for (unsigned vectorIndex = 0; vectorIndex < independentVectorCount; ++vectorIndex) {
                    Value * inputReal = loadComplexPart(builder, workspace, index, logicalFloatLanes, vectorIndex, false);
                    Value * inputImaginary = loadComplexPart(builder, workspace, index, logicalFloatLanes, vectorIndex, true);
                    storeComplexPart(
                        builder,
                        workspace,
                        index,
                        multiplySubtract(builder, inputReal, kernelReal, inputImaginary, kernelImaginary),
                        logicalFloatLanes,
                        vectorIndex,
                        false
                    );
                    storeComplexPart(
                        builder,
                        workspace,
                        index,
                        multiplyAdd(builder, inputReal, kernelImaginary, inputImaginary, kernelReal),
                        logicalFloatLanes,
                        vectorIndex,
                        true
                    );
                }
            });
            bitReverseWorkspace(builder, workspace, bitReverse, logicalFloatLanes, independentVectorCount, "frequency_inverse_bit_reverse");
            emitTransform(builder, workspace, frequencyData, logicalFloatLanes, independentVectorCount, true, "frequency_inverse");
            emitLoop(builder, "frequency_output", static_cast<uint64_t>(validExtent) * validExtent, [&](Value * outputIndex) {
                Value * outputRow = builder.CreateUDiv(outputIndex, builder.getSize(validExtent));
                Value * outputColumn = builder.CreateURem(outputIndex, builder.getSize(validExtent));
                Value * workspaceRow = builder.CreateAdd(outputRow, builder.getSize(kernelExtent - 1U));
                Value * workspaceColumn = builder.CreateAdd(outputColumn, builder.getSize(kernelExtent - 1U));
                Value * workspaceIndex = builder.CreateAdd(builder.CreateMul(workspaceRow, builder.getSize(transformExtent)), workspaceColumn);
                Value * normalization =
                    splat(builder, ConstantFP::get(builder.getFloatTy(), 1.0F / static_cast<float>(pointCount)), logicalFloatLanes);
                for (unsigned vectorIndex = 0; vectorIndex < independentVectorCount; ++vectorIndex) {
                    Value * realValues =
                        builder.CreateFMul(loadComplexPart(builder, workspace, workspaceIndex, logicalFloatLanes, vectorIndex, false), normalization);
                    Value * imaginaryValues =
                        builder.CreateFMul(loadComplexPart(builder, workspace, workspaceIndex, logicalFloatLanes, vectorIndex, true), normalization);
                    storeOutput(builder, outputPixels, tileRow, tileColumn, outputRow, outputColumn, realValues, imaginaryValues, vectorIndex);
                }
            });
        });
    }

    void generateDoSegmentMethod(KernelBuilder & builder) override {
        Value * inputPixels = builder.getScalarField("inputPixels");
        Value * outputPixels = builder.getScalarField("outputPixels");
        Value * workspace = builder.getScalarField("workspace");
        Value * bitReverse = builder.getScalarField("bitReverse");
        Value * weights = builder.getScalarField("weights");
        Value * control = builder.getScalarField("control");
        Value * frequencyData = builder.getScalarField("frequencyData");
        Value * transformMode = builder.CreateICmpNE(builder.CreateLoad(builder.getInt32Ty(), control), builder.getInt32(0));
        BasicBlock * transformBlock = builder.CreateBasicBlock("frequency_kernel_transform");
        BasicBlock * imageBlock = builder.CreateBasicBlock("frequency_image_convolution");
        BasicBlock * done = builder.CreateBasicBlock("frequency_done");
        builder.CreateCondBr(transformMode, transformBlock, imageBlock);
        builder.SetInsertPoint(transformBlock);
        initializeKernelSpectrum(builder, weights, bitReverse, frequencyData);
        builder.CreateBr(done);
        builder.SetInsertPoint(imageBlock);
        processImage(builder, inputPixels, outputPixels, workspace, bitReverse, frequencyData);
        builder.CreateBr(done);
        builder.SetInsertPoint(done);
    }

    const unsigned imageWidth;
    const unsigned imageHeight;
    const unsigned kernelExtent;
    const unsigned transformExtent;
    const unsigned logicalFloatLanes;
    const unsigned independentVectorCount;
    const unsigned tileCountPerVector;
    const unsigned tileCountPerGroup;
};

using FrequencyFunction =
    void (*)(const uint8_t *, uint8_t *, uint8_t *, const uint32_t *, const float *, const uint32_t *, float *, uint8_t *, std::size_t);

unsigned reverseBits(unsigned remainingValue, const unsigned bitCount) {
    unsigned reversed = 0;
    for (unsigned bit = 0; bit < bitCount; ++bit) {
        reversed = (reversed << 1U) | (remainingValue & 1U);
        remainingValue >>= 1U;
    }
    return reversed;
}

class FrequencyFilterImplementation final : public CompiledFilterImplementation {
   public:
    FrequencyFilterImplementation(
        std::unique_ptr<CPUDriver> cpuDriver,
        const unsigned imageWidth,
        const unsigned imageHeight,
        const unsigned kernelWidth,
        const unsigned kernelHeight,
        const FrequencyLayout selectedLayout,
        std::vector<float> convolutionWeights,
        const std::string & persistentIdentity
    )
        : CompiledFilterImplementation(
              ConvFilterMode::Frequency,
              imageWidth,
              imageHeight,
              checkedImageByteCount(imageWidth, imageHeight),
              workspaceByteCount(selectedLayout),
              FrequencyWorkspaceAlignment
          ),
          driver(std::move(cpuDriver)),
          weights(std::move(convolutionWeights)) {
        const FrequencyKernelConfiguration configuration{imageWidth, imageHeight, kernelHeight, kernelWidth, selectedLayout.transformExtent, weights};
        const std::size_t kernelArea = checkedKernelArea(kernelWidth, kernelHeight);
        const unsigned transform = configuration.transformExtent;
        if (kernelHeight != kernelWidth)
            throw std::invalid_argument("frequency kernel must be square");
        if (weights.size() != kernelArea)
            throw std::invalid_argument("frequency weight count does not match kernel area");
        if (transform == 0U || (transform & (transform - 1U)) != 0U)
            throw std::invalid_argument("frequency transform extent must be a nonzero power of two");
        if (transform < kernelHeight)
            throw std::invalid_argument("frequency transform extent is smaller than the kernel");
        long double absoluteWeightSum = 0.0L;
        for (const float weight : weights) {
            const long double magnitude = std::fabs(static_cast<long double>(weight));
            if (!std::isfinite(magnitude) || magnitude > std::numeric_limits<long double>::max() - absoluteWeightSum)
                throw std::invalid_argument("frequency absolute-weight sum exceeds long-double range");
            absoluteWeightSum += magnitude;
        }
        const long double transformPower = static_cast<long double>(transform) * transform * transform * transform;
        const long double intermediateBound = 2.0L * 255.0L * transformPower * absoluteWeightSum;
        if (!std::isfinite(intermediateBound) || intermediateBound > static_cast<long double>(std::numeric_limits<float>::max()) / 8.0L)
            throw std::invalid_argument("frequency intermediate-value bound exceeds float range");
        const unsigned bitBlockWidth = driver->getBitBlockWidth();
        const FrequencyLayout verifiedLayout = resolveFrequencyLayout(bitBlockWidth, transform);
        if (verifiedLayout.logicalFloatLanes != selectedLayout.logicalFloatLanes
            || verifiedLayout.independentVectorCount != selectedLayout.independentVectorCount
            || verifiedLayout.tilesPerVector != selectedLayout.tilesPerVector || verifiedLayout.tilesPerGroup != selectedLayout.tilesPerGroup)
        {
            throw std::invalid_argument("frequency layout is incompatible with the active builder");
        }

        unsigned bitCount = 0;
        for (unsigned remainingExtent = transform; remainingExtent > 1U; remainingExtent >>= 1U) {
            ++bitCount;
        }
        checkedMultiply(transform, sizeof(std::uint32_t), "frequency bit-reversal byte count overflow");
        bitReverse.resize(transform);
        for (unsigned index = 0; index < transform; ++index) {
            bitReverse[index] = reverseBits(index, bitCount);
        }
        const std::size_t points = checkedMultiply(transform, transform, "frequency metadata point count overflow");
        const std::size_t spectrumValues = checkedMultiply(points, 2U, "frequency spectrum size overflow");
        const std::size_t frequencyValueCount = checkedAdd(spectrumValues, transform, "frequency twiddle metadata size overflow");
        checkedMultiply(frequencyValueCount, sizeof(float), "frequency metadata byte count overflow");
        frequencyData.assign(frequencyValueCount, 0.0F);
        constexpr double Pi = 3.14159265358979323846264338327950288;
        const std::size_t twiddleRealOffset = spectrumValues;
        const std::size_t twiddleImaginaryOffset = checkedAdd(twiddleRealOffset, transform / 2U, "frequency twiddle offset overflow");
        for (unsigned index = 0; index < transform / 2U; ++index) {
            const double angle = -2.0 * Pi * static_cast<double>(index) / transform;
            frequencyData[twiddleRealOffset + index] = static_cast<float>(std::cos(angle));
            frequencyData[twiddleImaginaryOffset + index] = static_cast<float>(std::sin(angle));
        }
        auto pipeline = CreatePipeline(
            *driver,
            Input<const uint8_t *>("inputPixels"),
            Input<uint8_t *>("outputPixels"),
            Input<uint8_t *>("workspace"),
            Input<const uint32_t *>("bitReverse"),
            Input<const float *>("weights"),
            Input<const uint32_t *>("control"),
            Input<float *>("frequencyData"),
            Input<uint8_t *>("triggerBuffer"),
            Input<std::size_t>("triggerLength")
        );
        StreamSet * triggerStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<MemorySourceKernel>(
            pipeline.getInputScalar("triggerBuffer"), pipeline.getInputScalar("triggerLength"), triggerStream
        );
        pipeline.CreateKernelCall<FrequencyConvolutionKernel>(
            triggerStream,
            pipeline.getInputScalar("inputPixels"),
            pipeline.getInputScalar("outputPixels"),
            pipeline.getInputScalar("workspace"),
            pipeline.getInputScalar("bitReverse"),
            pipeline.getInputScalar("weights"),
            pipeline.getInputScalar("control"),
            pipeline.getInputScalar("frequencyData"),
            configuration,
            selectedLayout,
            bitBlockWidth,
            persistentIdentity
        );
        compiledFunction = pipeline.compile();

        const auto releaseAlignedWorkspace = [](void * allocatedWorkspace) noexcept {
            ::operator delete(allocatedWorkspace, std::align_val_t(FrequencyWorkspaceAlignment));
        };
        std::unique_ptr<void, decltype(releaseAlignedWorkspace)> transformWorkspace(
            workspaceSize() == 0U ? nullptr : ::operator new(workspaceSize(), std::align_val_t(FrequencyWorkspaceAlignment)), releaseAlignedWorkspace
        );
        if (transformWorkspace != nullptr && reinterpret_cast<std::uintptr_t>(transformWorkspace.get()) % FrequencyWorkspaceAlignment != 0U)
            throw std::runtime_error("frequency construction workspace alignment failed");
        uint8_t triggerByte = 0;
        compiledFunction(
            nullptr,
            nullptr,
            static_cast<uint8_t *>(transformWorkspace.get()),
            bitReverse.data(),
            weights.data(),
            &transformKernel,
            frequencyData.data(),
            &triggerByte,
            1U
        );
        transformKernel = 0U;
    }

   private:
    static std::size_t workspaceByteCount(const FrequencyLayout & layout) {
        std::size_t floatValueCount = checkedMultiply(layout.transformExtent, layout.transformExtent, "frequency transform point count overflow");
        floatValueCount = checkedMultiply(floatValueCount, 2U, "frequency complex workspace overflow");
        floatValueCount = checkedMultiply(floatValueCount, layout.logicalFloatLanes, "frequency logical-lane workspace overflow");
        floatValueCount = checkedMultiply(floatValueCount, layout.independentVectorCount, "frequency vector-group workspace overflow");
        return checkedMultiply(floatValueCount, sizeof(float), "frequency workspace byte count overflow");
    }

    void invoke(const uint8_t * inputPixels, uint8_t * outputPixels, void * workspace) const noexcept final {
        uint8_t triggerByte = 0;
        // Construction uses this signature to write the spectrum; apply only reads it.
        compiledFunction(
            inputPixels,
            outputPixels,
            static_cast<uint8_t *>(workspace),
            bitReverse.data(),
            weights.data(),
            &transformKernel,
            const_cast<float *>(frequencyData.data()),
            &triggerByte,
            1U
        );
    }

    std::unique_ptr<CPUDriver> driver;
    FrequencyFunction compiledFunction = nullptr;
    std::vector<uint32_t> bitReverse;
    const std::vector<float> weights;
    std::uint32_t transformKernel = 1U;
    std::vector<float> frequencyData;
};

}  // namespace

std::shared_ptr<const CompiledFilterImplementation> compileFrequencyFilter(
    std::unique_ptr<CPUDriver> driver,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const FrequencyLayout layout,
    std::vector<float> weights,
    std::string persistentIdentity
) {
    return std::make_shared<FrequencyFilterImplementation>(
        std::move(driver), imageWidth, imageHeight, kernelWidth, kernelHeight, layout, std::move(weights), persistentIdentity
    );
}

}  // namespace kernel::image::internal
