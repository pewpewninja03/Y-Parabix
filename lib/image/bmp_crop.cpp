#include <image/bmp_crop.h>

#include "bmp_pipeline_internal.h"

#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace image {

BGRImage cropImage(const BGRImage & source, const std::uint32_t width, const std::uint32_t height) {
    return cropImage(source, 0U, 0U, width, height);
}

BGRImage cropImage(
    const BGRImage & source, const std::uint32_t originX, const std::uint32_t originY, const std::uint32_t width, const std::uint32_t height
) {
    if (width == 0U || height == 0U)
        throw std::runtime_error("BMP crop dimensions must be nonzero");
    if (originX >= source.width || originY >= source.height)
        throw std::runtime_error("BMP crop origin is outside source");
    if (width > source.width - originX || height > source.height - originY)
        throw std::runtime_error("BMP crop dimensions exceed source");

    CPUDriver driver("image_crop");
    const std::size_t alignment = driver.getBitBlockWidth() / 8U;
    const internal::AlignedByteBuffer packedSource(source.pixels, alignment);
    const std::size_t firstSelectedPixel = static_cast<std::size_t>(originY) * source.width + originX;
    const std::size_t alignedFirstPixel = firstSelectedPixel - firstSelectedPixel % driver.getBitBlockWidth();
    const std::size_t lastSelectedPixel = (static_cast<std::size_t>(originY) + height - 1U) * source.width + originX + width;
    const std::size_t sourcePixelCount = lastSelectedPixel - alignedFirstPixel;
    const std::size_t outputPixelCount = static_cast<std::size_t>(width) * height;
    std::vector<std::uint8_t> cropPattern((sourcePixelCount + 7U) / 8U);
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::size_t firstPixel = firstSelectedPixel + static_cast<std::size_t>(row) * source.width - alignedFirstPixel;
        const std::size_t endPixel = firstPixel + width;
        for (std::size_t pixel = firstPixel; pixel < endPixel; ++pixel)
            cropPattern[pixel / 8U] |= static_cast<std::uint8_t>(1U << (pixel % 8U));
    }
    const internal::AlignedByteBuffer alignedCropPattern(cropPattern, alignment);

    kernel::StreamSetPtr croppedBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"croppedBytes", 1, 24, kernel::ReturnedBuffer(outputPixelCount == sourcePixelCount ? 2U : 1U)},
        kernel::Input<const std::uint8_t *>{"packedPixels"},
        kernel::Input<std::size_t>{"byteCount"},
        kernel::Input<const std::uint8_t *>{"cropPattern"},
        kernel::Input<std::size_t>{"cropPatternBitCount"}
    );

    kernel::StreamSet * const sourceBytes = pipeline.CreateStreamSet(1, 24);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(pipeline.getInputScalar("packedPixels"), pipeline.getInputScalar("byteCount"), sourceBytes);
    kernel::StreamSet * const cropMask = pipeline.CreateStreamSet(1);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("cropPattern"), pipeline.getInputScalar("cropPatternBitCount"), cropMask
    );
    kernel::FilterByMask(pipeline, cropMask, sourceBytes, pipeline.getOutputStreamSet("croppedBytes"));

    const auto run = pipeline.compile();
    run(croppedBytes, packedSource.data() + alignedFirstPixel * 3U, sourcePixelCount, alignedCropPattern.data(), sourcePixelCount);
    return internal::materializePackedColor(croppedBytes, width, height);
}

}  // namespace image
