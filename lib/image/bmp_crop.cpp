#include <image/bmp_crop.h>

#include "bmp_pipeline_internal.h"

#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/stream_select.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace image {

BGRImage cropImage(const BGRImage & source, const std::uint32_t width, const std::uint32_t height) {
    CPUDriver driver("image_crop");
    const internal::AlignedByteBuffer packedSource(source.pixels, driver.getBitBlockWidth() / 8U);
    std::vector<std::uint64_t> cropPattern(static_cast<std::size_t>(source.width) * source.height);
    for (std::uint32_t row = 0; row < height; ++row) {
        const std::size_t rowStart = static_cast<std::size_t>(row) * source.width;
        std::fill_n(cropPattern.begin() + rowStart, width, 1U);
    }

    kernel::StreamSetPtr blueBytes;
    kernel::StreamSetPtr greenBytes;
    kernel::StreamSetPtr redBytes;
    auto pipeline = kernel::CreatePipeline(
        driver,
        kernel::Output<kernel::streamset_t>{"blueBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Output<kernel::streamset_t>{"greenBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Output<kernel::streamset_t>{"redBytes", 1, 8, kernel::ReturnedBuffer(1)},
        kernel::Input<const std::uint8_t *>{"packedPixels"},
        kernel::Input<std::size_t>{"byteCount"}
    );

    kernel::StreamSet * const sourceColor =
        internal::createColorStream(pipeline, pipeline.getInputScalar("packedPixels"), pipeline.getInputScalar("byteCount"));
    kernel::StreamSet * const cropMask = pipeline.CreateRepeatingStreamSet(1, cropPattern);
    const std::array<kernel::StreamSet *, 3> sourceBasis = {
        kernel::streamutils::Select(pipeline, sourceColor, kernel::streamutils::Range(internal::BlueStreamBase, internal::GreenStreamBase)),
        kernel::streamutils::Select(pipeline, sourceColor, kernel::streamutils::Range(internal::GreenStreamBase, internal::RedStreamBase)),
        kernel::streamutils::Select(pipeline, sourceColor, kernel::streamutils::Range(internal::RedStreamBase, internal::ColorStreamCount)),
    };
    std::array<kernel::StreamSet *, 3> croppedBasis{};
    for (unsigned channel = 0; channel < croppedBasis.size(); ++channel) {
        kernel::StreamSet * const sourceBytes = pipeline.CreateStreamSet(1, internal::ChannelBits);
        pipeline.CreateKernelCall<kernel::P2SKernel>(sourceBasis[channel], sourceBytes);
        kernel::StreamSet * const croppedBytes = pipeline.CreateStreamSet(1, internal::ChannelBits);
        kernel::FilterByMask(pipeline, cropMask, sourceBytes, croppedBytes);
        croppedBasis[channel] = pipeline.CreateStreamSet(internal::ChannelBits);
        pipeline.CreateKernelCall<kernel::S2PKernel>(croppedBytes, croppedBasis[channel]);
    }
    kernel::StreamSet * const croppedColor = kernel::streamutils::Select(pipeline, {croppedBasis[0], croppedBasis[1], croppedBasis[2]});
    internal::createColorByteStreams(
        pipeline,
        croppedColor,
        pipeline.getOutputStreamSet("blueBytes"),
        pipeline.getOutputStreamSet("greenBytes"),
        pipeline.getOutputStreamSet("redBytes")
    );

    const auto run = pipeline.compile();
    run(blueBytes, greenBytes, redBytes, packedSource.data(), source.pixels.size());
    return internal::materializeColor(blueBytes, greenBytes, redBytes, width, height, false);
}

}  // namespace image
