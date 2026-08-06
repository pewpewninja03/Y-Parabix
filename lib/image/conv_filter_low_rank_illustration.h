#pragma once

#include <image/conv_filter.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace kernel::image::internal {

constexpr std::size_t NoLowRankIllustrationCoordinate = std::numeric_limits<std::size_t>::max();
constexpr std::uint32_t NoLowRankIllustrationChannel = 3U;

enum class LowRankIllustrationEvent : std::uint32_t {
    HorizontalPackedInput,
    HorizontalInputByte,
    HorizontalSample,
    HorizontalFactor,
    HorizontalAccumulator,
    HorizontalWorkspaceStore,
    VerticalWorkspaceLoad,
    VerticalFactor,
    VerticalAccumulator,
    OutputByteVector,
    PackedOutput,
    StoredOutputByte,
    StoreMask,
    Count
};

enum class LowRankIllustrationPath : std::uint32_t {
    HorizontalInterior,
    HorizontalBorder,
    HorizontalResult,
    VerticalFull,
    VerticalEdge,
    OutputFull,
    OutputChecked
};

enum class LowRankIllustrationElementType : std::uint32_t {
    UInt8,
    Float32
};

struct LowRankIllustrationCaptureEvent {
    LowRankIllustrationEvent event;
    LowRankIllustrationPath path;
    std::uint32_t channel;
    std::size_t elementCount;
    std::size_t byteCount;
    std::size_t byteOffset;
    std::size_t row;
    std::size_t groupStart;
    std::size_t rank;
    std::size_t horizontalTap;
    std::size_t verticalTap;
    std::size_t paddedSourceRow;
    std::size_t sourceColumn;
    std::size_t lane;
};

struct LowRankIllustrationCaptureLog {
    std::vector<LowRankIllustrationCaptureEvent> events;
    std::vector<std::uint8_t> bytes;
    std::exception_ptr failure;
    std::size_t logicalOutputs = 0;
};

struct LowRankIllustrationConfiguration {
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelWidth;
    unsigned kernelHeight;
    unsigned rank;
    unsigned logicalOutputs;
};

LowRankIllustrationElementType lowRankIllustrationElementType(LowRankIllustrationEvent event);
std::size_t lowRankIllustrationElementByteWidth(LowRankIllustrationElementType type);
void sortLowRankIllustrationEvents(LowRankIllustrationCaptureLog & capture);

std::string formatLowRankConvFilterIllustration(
    const LowRankIllustrationConfiguration & configuration, ConvFilterIllustrationSelection selection, const LowRankIllustrationCaptureLog & capture
);

extern "C" void captureLowRankConvFilterValue(
    std::uint8_t * context,
    std::uint32_t event,
    std::uint32_t path,
    std::uint32_t elementType,
    std::uint32_t channel,
    std::size_t elementCount,
    std::size_t elementByteWidth,
    const std::uint8_t * bytes,
    std::size_t row,
    std::size_t groupStart,
    std::size_t rank,
    std::size_t horizontalTap,
    std::size_t verticalTap,
    std::size_t paddedSourceRow,
    std::size_t sourceColumn,
    std::size_t lane
) noexcept;

}  // namespace kernel::image::internal
