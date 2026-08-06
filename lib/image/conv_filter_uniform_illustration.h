#pragma once

#include <image/conv_filter.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace kernel::image::internal {

constexpr std::size_t NoUniformIllustrationCoordinate = std::numeric_limits<std::size_t>::max();
constexpr std::uint32_t NoUniformIllustrationChannel = 3U;

enum class UniformIllustrationEvent : std::uint32_t {
    PackedInput,
    InputRgb,
    HorizontalInitialOperand,
    HorizontalInitialAfterAdd,
    HorizontalLeavingOperand,
    HorizontalEnteringOperand,
    SingleSum,
    SingleWeighted,
    SingleOutputBytes,
    SinglePackedOutput,
    GroupSum,
    GroupWeighted,
    GroupOutputBytes,
    GroupPackedOutput,
    Count
};

enum class UniformIllustrationValueType {
    Float32,
    Int32,
    UInt8
};

enum class UniformIllustrationPath : std::uint32_t {
    ColumnInitialization,
    ColumnUpdate,
    WindowInitialization,
    AdjacentOutputs,
    SingleOutput
};

struct UniformIllustrationCaptureEvent {
    UniformIllustrationEvent event;
    UniformIllustrationPath path;
    std::uint32_t channel;
    std::size_t elementCount;
    std::size_t byteOffset;
    std::size_t outputRow;
    std::size_t outputColumn;
    std::size_t groupStart;
    std::size_t workspaceColumn;
    std::size_t sourceInputRow;
    std::size_t recurrenceSource;
    std::size_t recurrenceDestination;
};

struct UniformIllustrationCaptureLog {
    std::vector<UniformIllustrationCaptureEvent> events;
    std::vector<std::uint8_t> bytes;
    std::exception_ptr failure;
    std::size_t logicalOutputs = 0;
};

struct UniformIllustrationConfiguration {
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelWidth;
    unsigned kernelHeight;
    float weight;
    unsigned logicalOutputs;
};

UniformIllustrationValueType uniformIllustrationValueType(UniformIllustrationEvent event);
std::size_t uniformIllustrationElementByteCount(UniformIllustrationEvent event);

extern "C" void captureUniformConvFilterValue(
    std::uint8_t * context,
    std::uint32_t event,
    std::uint32_t path,
    std::uint32_t channel,
    std::size_t elementCount,
    std::size_t elementByteCount,
    const std::uint8_t * bytes,
    std::size_t outputRow,
    std::size_t outputColumn,
    std::size_t groupStart,
    std::size_t workspaceColumn,
    std::size_t sourceInputRow,
    std::size_t recurrenceSource,
    std::size_t recurrenceDestination
) noexcept;

std::string formatUniformConvFilterIllustration(
    const UniformIllustrationConfiguration & configuration, ConvFilterIllustrationSelection selection, const UniformIllustrationCaptureLog & capture
);

}  // namespace kernel::image::internal
