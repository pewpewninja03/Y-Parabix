#include <image/bmp_mask.h>

#include "bmp_pipeline_internal.h"

#include <kernel/bitwise/bixlogic.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>

#include <fcntl.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace image {
namespace {

struct BlackMask {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::uint8_t> pixels;
};

BlackMask readBlackMask(const std::string & path) {
    const int fileDescriptor = ::open(path.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
        throw std::runtime_error("BMP open failed");

    try {
        const internal::BitmapMetadata info = internal::readBMPMetadata(fileDescriptor, 1U);
        std::array<bool, 2> isBlack{};
        for (std::size_t index = 0; index < info.palette.size(); ++index) {
            isBlack[index] = info.palette[index][0] == 0U && info.palette[index][1] == 0U && info.palette[index][2] == 0U;
        }

        BlackMask mask{info.width, info.height, std::vector<std::uint8_t>(static_cast<std::size_t>(info.width) * info.height)};
        std::vector<std::uint8_t> row(info.rowStride);
        for (std::uint32_t storedRow = 0; storedRow < info.height; ++storedRow) {
            internal::readExact(fileDescriptor, row.data(), row.size());
            const std::uint32_t outputRow = info.rowsBottomUp ? info.height - storedRow - 1U : storedRow;
            for (std::uint32_t column = 0; column < info.width; ++column) {
                const unsigned index = (row[column / 8U] >> (7U - column % 8U)) & 1U;
                mask.pixels[static_cast<std::size_t>(outputRow) * info.width + column] = isBlack[index] ? 1U : 0U;
            }
        }
        ::close(fileDescriptor);
        return mask;
    } catch (...) {
        ::close(fileDescriptor);
        throw;
    }
}

}  // namespace

BGRImage maskImage(const BGRImage & source, const std::string & maskPath, const BGRColor color) {
    const BlackMask mask = readBlackMask(maskPath);
    if (mask.width != source.width || mask.height != source.height)
        throw std::runtime_error("BMP mask dimensions do not match source");

    CPUDriver driver("image_mask");
    const std::size_t alignment = driver.getBitBlockWidth() / 8U;
    const internal::AlignedByteBuffer packedSource(source.pixels, alignment);
    const bool replaceWithBlack = color.blue == 0U && color.green == 0U && color.red == 0U;
    std::vector<std::uint8_t> selectionBytes(source.pixels.size());
    for (std::size_t pixel = 0; pixel < mask.pixels.size(); ++pixel) {
        const std::uint8_t selection = mask.pixels[pixel] != static_cast<std::uint8_t>(replaceWithBlack) ? 0xFFU : 0U;
        selectionBytes[pixel * 3U] = selection;
        selectionBytes[pixel * 3U + 1U] = selection;
        selectionBytes[pixel * 3U + 2U] = selection;
    }
    const internal::AlignedByteBuffer packedSelection(selectionBytes, alignment);

    kernel::StreamSetPtr maskedBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"maskedBytes", 1, 8, kernel::ReturnedBuffer(2)},
        kernel::Input<const std::uint8_t *>{"sourcePixels"},
        kernel::Input<std::size_t>{"sourceByteCount"},
        kernel::Input<const std::uint8_t *>{"selectionBytes"},
        kernel::Input<std::size_t>{"selectionByteCount"}
    );

    kernel::StreamSet * const sourceBytes = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("sourcePixels"), pipeline.getInputScalar("sourceByteCount"), sourceBytes
    );
    kernel::StreamSet * const selection = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(
        pipeline.getInputScalar("selectionBytes"), pipeline.getInputScalar("selectionByteCount"), selection
    );

    if (replaceWithBlack) {
        kernel::AndCombine(pipeline, sourceBytes, selection, pipeline.getOutputStreamSet("maskedBytes"));
    } else {
        kernel::StreamSet * const replacement = pipeline.CreateRepeatingStreamSet(8, {color.blue, color.green, color.red});
        kernel::StreamSet * const difference = pipeline.CreateStreamSet(1, 8);
        kernel::XorCombine(pipeline, sourceBytes, replacement, difference);
        kernel::StreamSet * const selectedDifference = pipeline.CreateStreamSet(1, 8);
        kernel::AndCombine(pipeline, difference, selection, selectedDifference);
        kernel::XorCombine(pipeline, sourceBytes, selectedDifference, pipeline.getOutputStreamSet("maskedBytes"));
    }

    const auto run = pipeline.compile();
    run(maskedBytes, packedSource.data(), source.pixels.size(), packedSelection.data(), selectionBytes.size());
    return internal::materializePackedColor(maskedBytes, source.width, source.height);
}

}  // namespace image
