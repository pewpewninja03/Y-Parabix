#include <image/bmp_io.h>
#include <image/conv_filter.h>

#include "bmp_pipeline_internal.h"

#include <kernel/bitwise/bixlogic.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>

#include <toolchain/toolchain.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr unsigned ChannelCount = 3;
constexpr unsigned Repetitions = 20;
constexpr float ConvolutionWeight = 1.0F / 1024.0F;

volatile std::uint8_t observedScalarByte = 0;

class AlignedWorkspace {
   public:
    AlignedWorkspace(const std::size_t size, const std::size_t alignment) {
        if (size == 0U)
            return;
        storage.resize(size + alignment - 1U);
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
        pointer = reinterpret_cast<void *>((address + alignment - 1U) & ~(alignment - 1U));
    }

    void * data() noexcept {
        return pointer;
    }

   private:
    std::vector<std::uint8_t> storage;
    void * pointer = nullptr;
};

std::size_t byteIndex(const std::uint32_t width, const std::uint32_t row, const std::uint32_t column, const unsigned channel) {
    return (static_cast<std::size_t>(row) * width + column) * ChannelCount + channel;
}

image::BGRImage makeInput(const std::uint32_t side) {
    image::BGRImage input(side, side);
    for (std::uint32_t row = 0; row < side; ++row) {
        for (std::uint32_t column = 0; column < side; ++column) {
            input.pixels[byteIndex(side, row, column, 0)] = static_cast<std::uint8_t>(17U + 13U * row + 7U * column);
            input.pixels[byteIndex(side, row, column, 1)] = static_cast<std::uint8_t>(53U + 5U * row + 11U * column);
            input.pixels[byteIndex(side, row, column, 2)] = static_cast<std::uint8_t>(101U + 3U * row + 17U * column);
        }
    }
    return input;
}

std::uint8_t zeroExtendedInput(const image::BGRImage & input, const int row, const int column, const unsigned channel) {
    if (row < 0 || column < 0 || row >= static_cast<int>(input.height) || column >= static_cast<int>(input.width))
        return 0U;
    return input.pixels[byteIndex(input.width, static_cast<std::uint32_t>(row), static_cast<std::uint32_t>(column), channel)];
}

std::uint8_t outputByte(const float value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0F, 255.0F) + 0.5F);
}

std::uint64_t elapsedNanoseconds(const Clock::time_point start, const Clock::time_point end) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

void printResult(
    const char * method,
    const std::uint32_t sourceSide,
    const std::uint32_t regionSide,
    const std::uint32_t kernelSide,
    const unsigned rank,
    const std::size_t pixelCount,
    const std::uint64_t setup,
    const std::uint64_t warmup,
    const std::uint64_t warmElapsed,
    const std::uint64_t scalarElapsed
) {
    const double averageWarmNanoseconds = static_cast<double>(warmElapsed) / Repetitions;
    const double averageScalarNanoseconds = static_cast<double>(scalarElapsed) / Repetitions;
    std::cout << method << ',' << sourceSide << ',' << regionSide << ',' << kernelSide << ',' << rank << ',' << setup << ',' << warmup << ','
              << averageWarmNanoseconds << ',' << averageWarmNanoseconds / pixelCount << ',' << averageScalarNanoseconds << ','
              << averageScalarNanoseconds / pixelCount << '\n'
              << std::flush;
}

image::BGRImage scalarCrop(const image::BGRImage & source, const std::uint32_t regionSide) {
    image::BGRImage output(regionSide, regionSide);
    for (std::uint32_t row = 0; row < regionSide; ++row) {
        for (std::uint32_t column = 0; column < regionSide; ++column) {
            const std::size_t sourceByte = byteIndex(source.width, row, column, 0);
            const std::size_t destinationByte = byteIndex(regionSide, row, column, 0);
            output.pixels[destinationByte] = source.pixels[sourceByte];
            output.pixels[destinationByte + 1U] = source.pixels[sourceByte + 1U];
            output.pixels[destinationByte + 2U] = source.pixels[sourceByte + 2U];
        }
    }
    return output;
}

void benchmarkCrop(const image::BGRImage & source, const std::uint32_t regionSide) {
    const std::size_t pixelCount = static_cast<std::size_t>(source.width) * source.height;
    const std::size_t sourcePixelCount = static_cast<std::size_t>(regionSide - 1U) * source.width + regionSide;
    const std::size_t outputPixelCount = static_cast<std::size_t>(regionSide) * regionSide;
    std::vector<std::uint8_t> keepPattern((sourcePixelCount + 7U) / 8U);
    for (std::uint32_t row = 0; row < regionSide; ++row) {
        const std::size_t firstPixel = static_cast<std::size_t>(row) * source.width;
        const std::size_t endPixel = firstPixel + regionSide;
        for (std::size_t pixel = firstPixel; pixel < endPixel; ++pixel)
            keepPattern[pixel / 8U] |= static_cast<std::uint8_t>(1U << (pixel % 8U));
    }

    const auto setupStart = Clock::now();
    CPUDriver driver("image_benchmark_crop");
    const std::size_t alignment = driver.getBitBlockWidth() / 8U;
    const image::internal::AlignedByteBuffer packedSource(source.pixels, alignment);
    const image::internal::AlignedByteBuffer packedKeepPattern(keepPattern, alignment);
    kernel::StreamSetPtr croppedBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"croppedBytes", 1, 24, kernel::ReturnedBuffer(outputPixelCount == sourcePixelCount ? 2U : 1U)},
        kernel::Input<const std::uint8_t *>{"packedPixels"},
        kernel::Input<std::size_t>{"byteCount"},
        kernel::Input<const std::uint8_t *>{"keepPattern"},
        kernel::Input<std::size_t>{"keepPatternBitCount"}
    );

    kernel::StreamSet * const sourceBytes = pipeline.CreateStreamSet(1, 24);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(pipeline.getInputScalar("packedPixels"), pipeline.getInputScalar("byteCount"), sourceBytes);
    kernel::StreamSet * const cropMask = pipeline.CreateStreamSet(1);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("keepPattern"), pipeline.getInputScalar("keepPatternBitCount"), cropMask
    );
    kernel::FilterByMask(pipeline, cropMask, sourceBytes, pipeline.getOutputStreamSet("croppedBytes"));

    const auto run = pipeline.compile();
    const auto setupEnd = Clock::now();

    const auto warmupStart = Clock::now();
    run(croppedBytes, packedSource.data(), sourcePixelCount, packedKeepPattern.data(), sourcePixelCount);
    image::BGRImage generated = image::internal::materializePackedColor(croppedBytes, regionSide, regionSide);
    const auto warmupEnd = Clock::now();

    const auto warmStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        run(croppedBytes, packedSource.data(), sourcePixelCount, packedKeepPattern.data(), sourcePixelCount);
        generated = image::internal::materializePackedColor(croppedBytes, regionSide, regionSide);
    }
    const auto warmEnd = Clock::now();

    image::BGRImage scalar(0U, 0U);
    const auto scalarStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        scalar = scalarCrop(source, regionSide);
        observedScalarByte = scalar.pixels[repetition % scalar.pixels.size()];
    }
    const auto scalarEnd = Clock::now();

    printResult(
        "crop",
        source.width,
        regionSide,
        0U,
        0U,
        pixelCount,
        elapsedNanoseconds(setupStart, setupEnd),
        elapsedNanoseconds(warmupStart, warmupEnd),
        elapsedNanoseconds(warmStart, warmEnd),
        elapsedNanoseconds(scalarStart, scalarEnd)
    );
}

image::BGRImage scalarMask(const image::BGRImage & source, const std::vector<std::uint8_t> & blackMask) {
    image::BGRImage output(source.width, source.height);
    for (std::size_t pixel = 0; pixel < blackMask.size(); ++pixel) {
        const std::size_t byte = pixel * ChannelCount;
        if (blackMask[pixel] != 0U) {
            output.pixels[byte] = 0U;
            output.pixels[byte + 1U] = 0U;
            output.pixels[byte + 2U] = 0U;
        } else {
            output.pixels[byte] = source.pixels[byte];
            output.pixels[byte + 1U] = source.pixels[byte + 1U];
            output.pixels[byte + 2U] = source.pixels[byte + 2U];
        }
    }
    return output;
}

void benchmarkMask(const image::BGRImage & source, const std::uint32_t regionSide) {
    const std::size_t pixelCount = static_cast<std::size_t>(source.width) * source.height;
    std::vector<std::uint8_t> blackMask(pixelCount);
    for (std::uint32_t row = 0; row < regionSide; ++row)
        std::fill_n(blackMask.begin() + static_cast<std::size_t>(row) * source.width, regionSide, 1U);
    std::vector<std::uint8_t> keepBytes(source.pixels.size(), 0xFFU);
    for (std::uint32_t row = 0; row < regionSide; ++row) {
        const std::size_t firstByte = static_cast<std::size_t>(row) * source.width * ChannelCount;
        std::fill_n(keepBytes.begin() + firstByte, static_cast<std::size_t>(regionSide) * ChannelCount, 0U);
    }

    const auto setupStart = Clock::now();
    CPUDriver driver("image_benchmark_mask");
    const std::size_t alignment = driver.getBitBlockWidth() / 8U;
    const image::internal::AlignedByteBuffer packedSource(source.pixels, alignment);
    const image::internal::AlignedByteBuffer packedKeepBytes(keepBytes, alignment);
    kernel::StreamSetPtr maskedBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"maskedBytes", 1, 8, kernel::ReturnedBuffer(2)},
        kernel::Input<const std::uint8_t *>{"sourcePixels"},
        kernel::Input<std::size_t>{"sourceByteCount"},
        kernel::Input<const std::uint8_t *>{"keepBytes"},
        kernel::Input<std::size_t>{"keepByteCount"}
    );

    kernel::StreamSet * const sourceBytes = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("sourcePixels"), pipeline.getInputScalar("sourceByteCount"), sourceBytes
    );
    kernel::StreamSet * const keepByteStream = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("keepBytes"), pipeline.getInputScalar("keepByteCount"), keepByteStream
    );
    kernel::AndCombine(pipeline, sourceBytes, keepByteStream, pipeline.getOutputStreamSet("maskedBytes"));

    const auto run = pipeline.compile();
    const auto setupEnd = Clock::now();

    const auto warmupStart = Clock::now();
    run(maskedBytes, packedSource.data(), source.pixels.size(), packedKeepBytes.data(), keepBytes.size());
    image::BGRImage generated = image::internal::materializePackedColor(maskedBytes, source.width, source.height);
    const auto warmupEnd = Clock::now();

    const auto warmStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        run(maskedBytes, packedSource.data(), source.pixels.size(), packedKeepBytes.data(), keepBytes.size());
        generated = image::internal::materializePackedColor(maskedBytes, source.width, source.height);
    }
    const auto warmEnd = Clock::now();

    image::BGRImage scalar(0U, 0U);
    const auto scalarStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        scalar = scalarMask(source, blackMask);
        observedScalarByte = scalar.pixels[repetition % scalar.pixels.size()];
    }
    const auto scalarEnd = Clock::now();

    printResult(
        "mask",
        source.width,
        regionSide,
        0U,
        0U,
        pixelCount,
        elapsedNanoseconds(setupStart, setupEnd),
        elapsedNanoseconds(warmupStart, warmupEnd),
        elapsedNanoseconds(warmStart, warmEnd),
        elapsedNanoseconds(scalarStart, scalarEnd)
    );
}

void scalarDefault(const image::BGRImage & input, image::BGRImage & output, const std::uint32_t kernelSide, const std::vector<float> & weights) {
    const int radius = static_cast<int>(kernelSide / 2U);
    for (std::uint32_t row = 0; row < input.height; ++row) {
        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                float sum = 0.0F;
                for (std::uint32_t kernelRow = 0; kernelRow < kernelSide; ++kernelRow) {
                    for (std::uint32_t kernelColumn = 0; kernelColumn < kernelSide; ++kernelColumn) {
                        const float sample = static_cast<float>(zeroExtendedInput(
                            input,
                            static_cast<int>(row) + static_cast<int>(kernelRow) - radius,
                            static_cast<int>(column) + static_cast<int>(kernelColumn) - radius,
                            channel
                        ));
                        sum = std::fma(sample, weights[static_cast<std::size_t>(kernelRow) * kernelSide + kernelColumn], sum);
                    }
                }
                output.pixels[byteIndex(input.width, row, column, channel)] = outputByte(sum);
            }
        }
    }
}

void benchmarkDefault(const image::BGRImage & input, const std::uint32_t kernelSide) {
    const std::size_t pixelCount = static_cast<std::size_t>(input.width) * input.height;
    const std::vector<float> weights(static_cast<std::size_t>(kernelSide) * kernelSide, ConvolutionWeight);
    const kernel::image::DefaultConvFilter configuration{kernelSide, kernelSide, {weights.data(), weights.size()}};

    const auto setupStart = Clock::now();
    const auto filter = kernel::image::compileConvFilter(input.width, input.height, configuration);
    image::BGRImage generated(input.width, input.height);
    AlignedWorkspace workspace(filter->workspaceSize(), filter->workspaceAlignment());
    const auto setupEnd = Clock::now();

    const auto warmupStart = Clock::now();
    (void)filter->apply(input.data(), generated.data(), workspace.data());
    const auto warmupEnd = Clock::now();

    const auto warmStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        (void)filter->apply(input.data(), generated.data(), workspace.data());
    }
    const auto warmEnd = Clock::now();

    image::BGRImage scalar(input.width, input.height);
    const auto scalarStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        scalarDefault(input, scalar, kernelSide, weights);
        observedScalarByte = scalar.pixels[repetition % scalar.pixels.size()];
    }
    const auto scalarEnd = Clock::now();

    printResult(
        "default",
        input.width,
        0U,
        kernelSide,
        0U,
        pixelCount,
        elapsedNanoseconds(setupStart, setupEnd),
        elapsedNanoseconds(warmupStart, warmupEnd),
        elapsedNanoseconds(warmStart, warmEnd),
        elapsedNanoseconds(scalarStart, scalarEnd)
    );
}

using BGRSum = std::array<std::int32_t, ChannelCount>;

void scalarUniform(const image::BGRImage & input, image::BGRImage & output, const std::uint32_t kernelSide, std::vector<BGRSum> & columnSums) {
    std::fill(columnSums.begin(), columnSums.end(), BGRSum{});
    const std::uint32_t radius = kernelSide / 2U;
    const std::uint32_t initialRowCount = std::min(input.height, radius + 1U);
    for (std::uint32_t sourceRow = 0; sourceRow < initialRowCount; ++sourceRow) {
        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel)
                columnSums[column][channel] += input.pixels[byteIndex(input.width, sourceRow, column, channel)];
        }
    }

    for (std::uint32_t row = 0; row < input.height; ++row) {
        BGRSum windowSum{};
        const std::uint32_t initialColumnCount = std::min(input.width, radius + 1U);
        for (std::uint32_t column = 0; column < initialColumnCount; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel)
                windowSum[channel] += columnSums[column][channel];
        }

        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                output.pixels[byteIndex(input.width, row, column, channel)] = outputByte(static_cast<float>(windowSum[channel]) * ConvolutionWeight);
            }
            const int leavingColumn = static_cast<int>(column) - static_cast<int>(radius);
            if (leavingColumn >= 0) {
                for (unsigned channel = 0; channel < ChannelCount; ++channel)
                    windowSum[channel] -= columnSums[static_cast<std::size_t>(leavingColumn)][channel];
            }
            const std::uint32_t enteringColumn = column + radius + 1U;
            if (enteringColumn < input.width) {
                for (unsigned channel = 0; channel < ChannelCount; ++channel)
                    windowSum[channel] += columnSums[enteringColumn][channel];
            }
        }

        if (row + 1U == input.height)
            continue;
        const int leavingRow = static_cast<int>(row) - static_cast<int>(radius);
        const std::uint32_t enteringRow = row + radius + 1U;
        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                if (leavingRow >= 0)
                    columnSums[column][channel] -= input.pixels[byteIndex(input.width, static_cast<std::uint32_t>(leavingRow), column, channel)];
                if (enteringRow < input.height)
                    columnSums[column][channel] += input.pixels[byteIndex(input.width, enteringRow, column, channel)];
            }
        }
    }
}

void benchmarkUniform(const image::BGRImage & input, const std::uint32_t kernelSide) {
    const std::size_t pixelCount = static_cast<std::size_t>(input.width) * input.height;
    const kernel::image::UniformConvFilter configuration{kernelSide, kernelSide, ConvolutionWeight};

    const auto setupStart = Clock::now();
    const auto filter = kernel::image::compileConvFilter(input.width, input.height, configuration);
    image::BGRImage generated(input.width, input.height);
    AlignedWorkspace workspace(filter->workspaceSize(), filter->workspaceAlignment());
    const auto setupEnd = Clock::now();

    const auto warmupStart = Clock::now();
    (void)filter->apply(input.data(), generated.data(), workspace.data());
    const auto warmupEnd = Clock::now();

    const auto warmStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        (void)filter->apply(input.data(), generated.data(), workspace.data());
    }
    const auto warmEnd = Clock::now();

    image::BGRImage scalar(input.width, input.height);
    std::vector<BGRSum> columnSums(input.width);
    const auto scalarStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        scalarUniform(input, scalar, kernelSide, columnSums);
        observedScalarByte = scalar.pixels[repetition % scalar.pixels.size()];
    }
    const auto scalarEnd = Clock::now();

    printResult(
        "uniform",
        input.width,
        0U,
        kernelSide,
        0U,
        pixelCount,
        elapsedNanoseconds(setupStart, setupEnd),
        elapsedNanoseconds(warmupStart, warmupEnd),
        elapsedNanoseconds(warmStart, warmEnd),
        elapsedNanoseconds(scalarStart, scalarEnd)
    );
}

void scalarLowRank(
    const image::BGRImage & input,
    image::BGRImage & output,
    const std::uint32_t kernelSide,
    const std::vector<float> & horizontalFactors,
    const std::vector<float> & verticalFactors,
    std::vector<float> & workspace
) {
    const int radius = static_cast<int>(kernelSide / 2U);
    for (std::uint32_t row = 0; row < input.height; ++row) {
        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                float sum = 0.0F;
                for (std::uint32_t tap = 0; tap < kernelSide; ++tap) {
                    const float sample = static_cast<float>(
                        zeroExtendedInput(input, static_cast<int>(row), static_cast<int>(column) + static_cast<int>(tap) - radius, channel)
                    );
                    sum = std::fma(sample, horizontalFactors[tap], sum);
                }
                workspace[byteIndex(input.width, row, column, channel)] = sum;
            }
        }
    }

    for (std::uint32_t row = 0; row < input.height; ++row) {
        for (std::uint32_t column = 0; column < input.width; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                float sum = 0.0F;
                for (std::uint32_t tap = 0; tap < kernelSide; ++tap) {
                    const int sourceRow = static_cast<int>(row) + static_cast<int>(tap) - radius;
                    const float sample = sourceRow < 0 || sourceRow >= static_cast<int>(input.height)
                                             ? 0.0F
                                             : workspace[byteIndex(input.width, static_cast<std::uint32_t>(sourceRow), column, channel)];
                    sum = std::fma(sample, verticalFactors[tap], sum);
                }
                output.pixels[byteIndex(input.width, row, column, channel)] = outputByte(sum);
            }
        }
    }
}

void benchmarkLowRank(const image::BGRImage & input, const std::uint32_t kernelSide) {
    const std::size_t pixelCount = static_cast<std::size_t>(input.width) * input.height;
    const std::vector<float> horizontalFactors(kernelSide, 1.0F);
    const std::vector<float> verticalFactors(kernelSide, ConvolutionWeight);
    const kernel::image::LowRankConvFilter configuration{
        kernelSide,
        kernelSide,
        1U,
        {horizontalFactors.data(), horizontalFactors.size()},
        {verticalFactors.data(), verticalFactors.size()},
    };

    const auto setupStart = Clock::now();
    const auto filter = kernel::image::compileConvFilter(input.width, input.height, configuration);
    image::BGRImage generated(input.width, input.height);
    AlignedWorkspace workspace(filter->workspaceSize(), filter->workspaceAlignment());
    const auto setupEnd = Clock::now();

    const auto warmupStart = Clock::now();
    (void)filter->apply(input.data(), generated.data(), workspace.data());
    const auto warmupEnd = Clock::now();

    const auto warmStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        (void)filter->apply(input.data(), generated.data(), workspace.data());
    }
    const auto warmEnd = Clock::now();

    image::BGRImage scalar(input.width, input.height);
    std::vector<float> scalarWorkspace(pixelCount * ChannelCount);
    const auto scalarStart = Clock::now();
    for (unsigned repetition = 0; repetition < Repetitions; ++repetition) {
        scalarLowRank(input, scalar, kernelSide, horizontalFactors, verticalFactors, scalarWorkspace);
        observedScalarByte = scalar.pixels[repetition % scalar.pixels.size()];
    }
    const auto scalarEnd = Clock::now();

    printResult(
        "lowrank",
        input.width,
        0U,
        kernelSide,
        1U,
        pixelCount,
        elapsedNanoseconds(setupStart, setupEnd),
        elapsedNanoseconds(warmupStart, warmupEnd),
        elapsedNanoseconds(warmStart, warmEnd),
        elapsedNanoseconds(scalarStart, scalarEnd)
    );
}

}  // namespace

int main() {
    codegen::ProgramName = "image_benchmark";
    codegen::EnableObjectCache = false;
    codegen::EnablePipelineObjectCache = false;

    std::cout << std::fixed << std::setprecision(6)
              << "method,source_side,region_side,kernel_side,rank,setup_ns,warmup_ns,warm_average_ns,warm_ns_per_pixel,scalar_average_ns,scalar_ns_"
                 "per_pixel\n"
              << std::flush;
    try {
        for (std::uint32_t sourceSide = 256U; sourceSide <= 8192U; sourceSide += 256U) {
            const image::BGRImage input = makeInput(sourceSide);
            for (std::uint32_t regionSide = 256U; regionSide <= sourceSide; regionSide += 256U) {
                benchmarkCrop(input, regionSide);
                benchmarkMask(input, regionSide);
            }
            for (std::uint32_t kernelSide = 3U; kernelSide <= 21U; kernelSide += 2U) {
                benchmarkDefault(input, kernelSide);
                benchmarkUniform(input, kernelSide);
                benchmarkLowRank(input, kernelSide);
            }
        }
    } catch (const std::exception & error) {
        std::cerr << "image_benchmark: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
