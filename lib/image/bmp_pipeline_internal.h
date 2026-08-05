#pragma once

#include <cassert>

#include <image/bmp_io.h>

#include <kernel/core/relationship.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/pipeline/pipeline_builder.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace image::internal {

constexpr unsigned ChannelBits = 8;
constexpr unsigned ColorStreamCount = 24;
constexpr unsigned BlueStreamBase = 0;
constexpr unsigned GreenStreamBase = 8;
constexpr unsigned RedStreamBase = 16;

struct BitmapMetadata {
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t rowStride;
    std::uint32_t pixelOffset;
    bool rowsBottomUp;
    std::vector<std::array<std::uint8_t, 4>> palette;
};

BitmapMetadata readBMPMetadata(int fileDescriptor, std::uint16_t bitsPerPixel);
void readExact(int fileDescriptor, void * data, std::size_t byteCount);

class AlignedByteBuffer final {
   public:
    AlignedByteBuffer(const std::vector<std::uint8_t> & bytes, std::size_t alignment);

    const std::uint8_t * data() const noexcept;

   private:
    std::vector<std::uint8_t> mStorage;
    std::uint8_t * mData;
};

kernel::StreamSet * createColorStream(kernel::PipelineBuilder & pipeline, kernel::Scalar * packedPixels, kernel::Scalar * byteCount);

void createColorByteStreams(
    kernel::PipelineBuilder & pipeline,
    kernel::StreamSet * colorStream,
    kernel::StreamSet * blueBytes,
    kernel::StreamSet * greenBytes,
    kernel::StreamSet * redBytes
);

BGRImage materializeColor(
    const kernel::StreamSetPtr & blueBytes,
    const kernel::StreamSetPtr & greenBytes,
    const kernel::StreamSetPtr & redBytes,
    std::uint32_t width,
    std::uint32_t height,
    bool rowsBottomUp
);

BGRImage materializePackedColor(const kernel::StreamSetPtr & packedBytes, std::uint32_t width, std::uint32_t height);

}  // namespace image::internal
