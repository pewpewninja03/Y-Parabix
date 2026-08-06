#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

namespace kernel::image::internal {

enum class DefaultIllustrationEvent : std::uint32_t {
    OutputColumns,
    SelectedPositions,
    OutputValid,
    SourceRows,
    SourceColumns,
    SourceValid,
    PackedInput,
    Sample,
    Weight,
    Accumulator,
    OutputBytes,
    PackedOutput,
    StoreMask,
    IdentityPackedInput,
    IdentityPackedOutput
};

enum class DefaultIllustrationValueType {
    Float32,
    UInt8,
    UInt32,
    Int64,
    Mask
};

constexpr std::uint32_t NoDefaultIllustrationChannel = 3U;

struct DefaultIllustrationCaptureEvent {
    DefaultIllustrationEvent event;
    std::size_t kernelRow;
    std::size_t kernelColumn;
    std::uint32_t channel;
    std::size_t elementCount;
    std::size_t outputRow;
    std::size_t groupStart;
    std::size_t byteOffset;
};

struct DefaultIllustrationCaptureLog {
    std::vector<DefaultIllustrationCaptureEvent> events;
    std::vector<std::uint8_t> bytes;
    std::exception_ptr failure;
};

DefaultIllustrationValueType defaultIllustrationValueType(DefaultIllustrationEvent event);
std::size_t defaultIllustrationElementByteCount(DefaultIllustrationEvent event);

extern "C" void captureDefaultConvFilterValue(
    std::uint8_t * context,
    std::uint32_t event,
    std::size_t kernelRow,
    std::size_t kernelColumn,
    std::uint32_t channel,
    std::size_t elementCount,
    std::size_t elementByteCount,
    const std::uint8_t * bytes,
    std::size_t outputRow,
    std::size_t groupStart
) noexcept;

std::string formatDefaultConvFilterIllustration(DefaultIllustrationCaptureLog capture, bool inputFocus);

}  // namespace kernel::image::internal
