#include <image/bmp_mask.h>

#include "bmp_pipeline_internal.h"

#include <kernel/basis/s2p_kernel.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/stream_select.h>

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

std::vector<std::uint8_t> readBlackMask(const std::string & path) {
    const int fileDescriptor = ::open(path.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
        throw std::runtime_error("BMP open failed");

    try {
        const internal::BitmapMetadata info = internal::readBMPMetadata(fileDescriptor, 1U);
        std::array<bool, 2> isBlack{};
        for (std::size_t index = 0; index < info.palette.size(); ++index) {
            isBlack[index] = info.palette[index][0] == 0U && info.palette[index][1] == 0U && info.palette[index][2] == 0U;
        }

        std::vector<std::uint8_t> mask(static_cast<std::size_t>(info.width) * info.height);
        std::vector<std::uint8_t> row(info.rowStride);
        for (std::uint32_t storedRow = 0; storedRow < info.height; ++storedRow) {
            internal::readExact(fileDescriptor, row.data(), row.size());
            const std::uint32_t outputRow = info.rowsBottomUp ? info.height - storedRow - 1U : storedRow;
            for (std::uint32_t column = 0; column < info.width; ++column) {
                const unsigned index = (row[column / 8U] >> (7U - column % 8U)) & 1U;
                mask[static_cast<std::size_t>(outputRow) * info.width + column] = isBlack[index] ? 1U : 0U;
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
    const std::vector<std::uint8_t> mask = readBlackMask(maskPath);
    CPUDriver driver("image_mask");
    const std::size_t alignment = driver.getBitBlockWidth() / 8U;
    const internal::AlignedByteBuffer packedSource(source.pixels, alignment);
    const internal::AlignedByteBuffer packedMask(mask, alignment);

    kernel::StreamSetPtr blueBytes;
    kernel::StreamSetPtr greenBytes;
    kernel::StreamSetPtr redBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"blueBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Output<kernel::streamset_t>{"greenBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Output<kernel::streamset_t>{"redBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Input<const std::uint8_t *>{"sourcePixels"},
        kernel::Input<std::size_t>{"sourceByteCount"},
        kernel::Input<const std::uint8_t *>{"maskPixels"},
        kernel::Input<std::size_t>{"maskByteCount"}
    );

    kernel::StreamSet * const sourceColor =
        internal::createColorStream(pipeline, pipeline.getInputScalar("sourcePixels"), pipeline.getInputScalar("sourceByteCount"));
    kernel::StreamSet * const maskBytes = pipeline.CreateStreamSet(1, 8);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(pipeline.getInputScalar("maskPixels"), pipeline.getInputScalar("maskByteCount"), maskBytes);
    kernel::StreamSet * const maskBasis = pipeline.CreateStreamSet(8);
    pipeline.CreateKernelCall<kernel::S2PKernel>(maskBytes, maskBasis);
    kernel::StreamSet * const blackMask = kernel::streamutils::Select(pipeline, maskBasis, kernel::streamutils::Range(0, 1));

    kernel::StreamSet * const keepMask = pipeline.CreateStreamSet(1);
    kernel::Invert(pipeline, blackMask, keepMask);
    kernel::StreamSet * const keptSource = pipeline.CreateStreamSet(internal::ColorStreamCount);
    kernel::ZeroByMask(pipeline, keepMask, sourceColor, keptSource);

    kernel::StreamSet * maskedColor = keptSource;
    if (color.blue != 0U || color.green != 0U || color.red != 0U) {
        const std::uint64_t packedColor =
            color.blue | (static_cast<std::uint64_t>(color.green) << 8U) | (static_cast<std::uint64_t>(color.red) << 16U);
        kernel::StreamSet * const replacementColor = pipeline.CreateRepeatingBixNum(internal::ColorStreamCount, {packedColor});
        kernel::StreamSet * const selectedReplacement = pipeline.CreateStreamSet(internal::ColorStreamCount);
        kernel::ZeroByMask(pipeline, blackMask, replacementColor, selectedReplacement);
        maskedColor = pipeline.CreateStreamSet(internal::ColorStreamCount);
        kernel::OrCombine(pipeline, keptSource, selectedReplacement, maskedColor);
    }

    internal::createColorByteStreams(
        pipeline,
        maskedColor,
        pipeline.getOutputStreamSet("blueBytes"),
        pipeline.getOutputStreamSet("greenBytes"),
        pipeline.getOutputStreamSet("redBytes")
    );

    const auto run = pipeline.compile();
    run(blueBytes, greenBytes, redBytes, packedSource.data(), source.pixels.size(), packedMask.data(), mask.size());
    return internal::materializeColor(blueBytes, greenBytes, redBytes, source.width, source.height, false);
}

}  // namespace image
