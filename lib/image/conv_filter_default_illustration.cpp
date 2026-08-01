#include "conv_filter_default_illustration.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace kernel::image::internal {
namespace {

template <typename T>
void appendInteger(std::string & output, const T value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc{})
        throw std::logic_error("Default illustration integer formatting failed");
    output.append(buffer, result.ptr);
}

void appendHex(std::string & output, std::uint32_t value, const unsigned digitCount) {
    static constexpr char HexDigits[] = "0123456789ABCDEF";
    const std::size_t begin = output.size();
    output.resize(begin + digitCount);
    for (unsigned digit = 0; digit < digitCount; ++digit) {
        output[begin + digitCount - digit - 1U] = HexDigits[value & 0xfU];
        value >>= 4U;
    }
}

template <typename T>
T readElement(const std::vector<std::uint8_t> & capturedBytes, const DefaultIllustrationCaptureEvent & event, const std::size_t index) {
    T value{};
    std::memcpy(&value, capturedBytes.data() + event.byteOffset + index * sizeof(T), sizeof(T));
    return value;
}

std::string tapPrefix(const DefaultIllustrationCaptureEvent & event) {
    if (event.kernelRow <= 9U && event.kernelColumn <= 9U)
        return "tap" + std::to_string(event.kernelRow) + std::to_string(event.kernelColumn);
    return "tap[" + std::to_string(event.kernelRow) + "," + std::to_string(event.kernelColumn) + "]";
}

char channelName(const std::uint32_t channel) {
    switch (channel) {
    case 0:
        return 'R';
    case 1:
        return 'G';
    case 2:
        return 'B';
    default:
        throw std::logic_error("invalid Default illustration channel");
    }
}

std::string eventLabel(const DefaultIllustrationCaptureEvent & event, const bool inputFocus) {
    const std::string tap = tapPrefix(event);
    switch (event.event) {
    case DefaultIllustrationEvent::OutputColumns:
        return "outputX";
    case DefaultIllustrationEvent::SelectedPositions:
        return inputFocus ? "inSupport" : "target";
    case DefaultIllustrationEvent::OutputValid:
        return "inOutputDomain";
    case DefaultIllustrationEvent::SourceRows:
        return tap + ".sourceY";
    case DefaultIllustrationEvent::SourceColumns:
        return tap + ".sourceX";
    case DefaultIllustrationEvent::SourceValid:
        return tap + ".inInputDomain";
    case DefaultIllustrationEvent::PackedInput:
        return tap + ".rgb.hex";
    case DefaultIllustrationEvent::Sample:
        return tap + "." + channelName(event.channel);
    case DefaultIllustrationEvent::Weight:
        return tap + ".w";
    case DefaultIllustrationEvent::Accumulator:
        return tap + ".partialSum" + channelName(event.channel);
    case DefaultIllustrationEvent::OutputBytes:
        return "output" + std::string(1U, channelName(event.channel));
    case DefaultIllustrationEvent::PackedOutput:
    case DefaultIllustrationEvent::IdentityPackedOutput:
        return "outputRGB.hex";
    case DefaultIllustrationEvent::StoreMask:
        return "storeMask";
    case DefaultIllustrationEvent::IdentityPackedInput:
        return "rgb.hex";
    }
    throw std::logic_error("unknown Default illustration event");
}

unsigned tapEventOrder(const DefaultIllustrationEvent event) {
    switch (event) {
    case DefaultIllustrationEvent::SourceRows:
        return 0;
    case DefaultIllustrationEvent::SourceColumns:
        return 1;
    case DefaultIllustrationEvent::SourceValid:
        return 2;
    case DefaultIllustrationEvent::PackedInput:
        return 3;
    case DefaultIllustrationEvent::Sample:
        return 4;
    case DefaultIllustrationEvent::Weight:
        return 7;
    case DefaultIllustrationEvent::Accumulator:
        return 8;
    default:
        return 20;
    }
}

unsigned eventSection(const DefaultIllustrationEvent event) {
    switch (event) {
    case DefaultIllustrationEvent::OutputColumns:
    case DefaultIllustrationEvent::SelectedPositions:
    case DefaultIllustrationEvent::OutputValid:
        return 0;
    case DefaultIllustrationEvent::SourceRows:
    case DefaultIllustrationEvent::SourceColumns:
    case DefaultIllustrationEvent::SourceValid:
    case DefaultIllustrationEvent::PackedInput:
    case DefaultIllustrationEvent::Sample:
    case DefaultIllustrationEvent::Weight:
    case DefaultIllustrationEvent::Accumulator:
        return 1;
    case DefaultIllustrationEvent::IdentityPackedInput:
        return 2;
    case DefaultIllustrationEvent::OutputBytes:
    case DefaultIllustrationEvent::PackedOutput:
    case DefaultIllustrationEvent::IdentityPackedOutput:
    case DefaultIllustrationEvent::StoreMask:
        return 3;
    }
    return 4;
}

unsigned headerEventOrder(const DefaultIllustrationEvent event) {
    switch (event) {
    case DefaultIllustrationEvent::OutputColumns:
        return 0;
    case DefaultIllustrationEvent::SelectedPositions:
        return 1;
    case DefaultIllustrationEvent::OutputValid:
        return 2;
    default:
        return 3;
    }
}

unsigned outputEventOrder(const DefaultIllustrationCaptureEvent & event) {
    switch (event.event) {
    case DefaultIllustrationEvent::IdentityPackedInput:
        return 0;
    case DefaultIllustrationEvent::OutputBytes:
        return event.channel + 1U;
    case DefaultIllustrationEvent::PackedOutput:
    case DefaultIllustrationEvent::IdentityPackedOutput:
        return 4;
    case DefaultIllustrationEvent::StoreMask:
        return 5;
    default:
        return 6;
    }
}

bool eventLess(const DefaultIllustrationCaptureEvent & left, const DefaultIllustrationCaptureEvent & right) {
    const unsigned leftSection = eventSection(left.event);
    const unsigned rightSection = eventSection(right.event);
    if (leftSection != rightSection)
        return leftSection < rightSection;
    if (leftSection == 0)
        return headerEventOrder(left.event) < headerEventOrder(right.event);
    if (leftSection == 1) {
        return std::tuple{left.kernelRow, left.kernelColumn, tapEventOrder(left.event), left.channel}
               < std::tuple{right.kernelRow, right.kernelColumn, tapEventOrder(right.event), right.channel};
    }
    return outputEventOrder(left) < outputEventOrder(right);
}

void printEventValue(
    std::string & output, const std::vector<std::uint8_t> & capturedBytes, const DefaultIllustrationCaptureEvent & event, const std::size_t index
) {
    switch (defaultIllustrationValueType(event.event)) {
    case DefaultIllustrationValueType::Float32:
        {
            char buffer[64];
            const auto result =
                std::to_chars(buffer, buffer + sizeof(buffer), readElement<float>(capturedBytes, event, index), std::chars_format::general, 7);
            if (result.ec != std::errc{})
                throw std::logic_error("Default illustration float formatting failed");
            output.append(buffer, result.ptr);
            return;
        }
    case DefaultIllustrationValueType::UInt8:
        appendHex(output, capturedBytes[event.byteOffset + index], 2);
        return;
    case DefaultIllustrationValueType::UInt32:
        appendInteger(output, readElement<std::uint32_t>(capturedBytes, event, index));
        return;
    case DefaultIllustrationValueType::Int64:
        appendInteger(output, readElement<std::int64_t>(capturedBytes, event, index));
        return;
    case DefaultIllustrationValueType::Mask:
        output.push_back(capturedBytes[event.byteOffset + index] == 0U ? '.' : '1');
        return;
    }
    throw std::logic_error("unknown Default illustration value type");
}

void printEvent(
    std::string & output,
    const std::vector<std::uint8_t> & capturedBytes,
    const DefaultIllustrationCaptureEvent & event,
    const std::size_t labelWidth,
    const bool inputFocus
) {
    const std::string label = eventLabel(event, inputFocus);
    auto printValues = [&](const std::string & rowLabel, const auto & printValue) {
        output.append(labelWidth - rowLabel.size(), ' ');
        output.append(rowLabel).append(" |");
        for (std::size_t index = 0; index < event.elementCount; ++index) {
            output.push_back(' ');
            printValue(index);
        }
        output.push_back('\n');
    };

    printValues(label, [&](const std::size_t index) { printEventValue(output, capturedBytes, event, index); });
    if (event.event == DefaultIllustrationEvent::PackedInput || event.event == DefaultIllustrationEvent::IdentityPackedInput) {
        const std::string decimalLabel = event.event == DefaultIllustrationEvent::PackedInput ? tapPrefix(event) + ".rgb.dec" : "rgb.dec";
        printValues(decimalLabel, [&](const std::size_t index) {
            appendInteger(output, static_cast<unsigned>(capturedBytes[event.byteOffset + index]));
        });
    }
    if (defaultIllustrationValueType(event.event) == DefaultIllustrationValueType::Float32) {
        printValues(label + ".bits", [&](const std::size_t index) { appendHex(output, readElement<std::uint32_t>(capturedBytes, event, index), 8); });
    }
}

}  // namespace

DefaultIllustrationValueType defaultIllustrationValueType(const DefaultIllustrationEvent event) {
    switch (event) {
    case DefaultIllustrationEvent::Sample:
    case DefaultIllustrationEvent::Weight:
    case DefaultIllustrationEvent::Accumulator:
        return DefaultIllustrationValueType::Float32;
    case DefaultIllustrationEvent::PackedInput:
    case DefaultIllustrationEvent::OutputBytes:
    case DefaultIllustrationEvent::PackedOutput:
    case DefaultIllustrationEvent::IdentityPackedInput:
    case DefaultIllustrationEvent::IdentityPackedOutput:
        return DefaultIllustrationValueType::UInt8;
    case DefaultIllustrationEvent::OutputColumns:
        return DefaultIllustrationValueType::UInt32;
    case DefaultIllustrationEvent::SourceRows:
    case DefaultIllustrationEvent::SourceColumns:
        return DefaultIllustrationValueType::Int64;
    case DefaultIllustrationEvent::SelectedPositions:
    case DefaultIllustrationEvent::OutputValid:
    case DefaultIllustrationEvent::SourceValid:
    case DefaultIllustrationEvent::StoreMask:
        return DefaultIllustrationValueType::Mask;
    }
    throw std::logic_error("unknown Default illustration event");
}

std::size_t defaultIllustrationElementByteCount(const DefaultIllustrationEvent event) {
    switch (defaultIllustrationValueType(event)) {
    case DefaultIllustrationValueType::Float32:
    case DefaultIllustrationValueType::UInt32:
        return sizeof(std::uint32_t);
    case DefaultIllustrationValueType::Int64:
        return sizeof(std::int64_t);
    case DefaultIllustrationValueType::UInt8:
    case DefaultIllustrationValueType::Mask:
        return sizeof(std::uint8_t);
    }
    throw std::logic_error("unknown Default illustration value type");
}

extern "C" void captureDefaultConvFilterValue(
    std::uint8_t * context,
    const std::uint32_t eventValue,
    const std::size_t kernelRow,
    const std::size_t kernelColumn,
    const std::uint32_t channel,
    const std::size_t elementCount,
    const std::size_t elementByteCount,
    const std::uint8_t * bytes,
    const std::size_t outputRow,
    const std::size_t groupStart
) noexcept {
    auto & capture = *reinterpret_cast<DefaultIllustrationCaptureLog *>(context);
    if (capture.failure)
        return;
    try {
        const auto event = static_cast<DefaultIllustrationEvent>(eventValue);
        if (elementByteCount != defaultIllustrationElementByteCount(event))
            throw std::logic_error("Default illustration element type mismatch");
        if (elementCount != 0U && elementByteCount > std::numeric_limits<std::size_t>::max() / elementCount)
            throw std::overflow_error("Default illustration capture size overflow");
        const std::size_t byteCount = elementCount * elementByteCount;
        const std::size_t byteOffset = capture.bytes.size();
        if (byteCount > std::numeric_limits<std::size_t>::max() - byteOffset)
            throw std::overflow_error("Default illustration capture storage overflow");
        capture.bytes.resize(byteOffset + byteCount);
        std::memcpy(capture.bytes.data() + byteOffset, bytes, byteCount);
        capture.events.emplace_back(
            DefaultIllustrationCaptureEvent{event, kernelRow, kernelColumn, channel, elementCount, outputRow, groupStart, byteOffset}
        );
    } catch (...) {
        capture.failure = std::current_exception();
    }
}

std::string formatDefaultConvFilterIllustration(DefaultIllustrationCaptureLog capture, const bool inputFocus) {
    auto & events = capture.events;
    std::stable_sort(events.begin(), events.end(), [](const DefaultIllustrationCaptureEvent & left, const DefaultIllustrationCaptureEvent & right) {
        return std::tie(left.outputRow, left.groupStart) < std::tie(right.outputRow, right.groupStart);
    });

    std::string output;
    std::size_t begin = 0;
    bool firstGroup = true;
    while (begin < events.size()) {
        std::size_t end = begin + 1U;
        while (end < events.size() && events[end].outputRow == events[begin].outputRow && events[end].groupStart == events[begin].groupStart)
            ++end;
        std::stable_sort(events.begin() + static_cast<std::ptrdiff_t>(begin), events.begin() + static_cast<std::ptrdiff_t>(end), eventLess);

        std::size_t labelWidth = 0;
        for (std::size_t index = begin; index < end; ++index) {
            const std::string label = eventLabel(events[index], inputFocus);
            labelWidth = std::max(labelWidth, label.size());
            if (defaultIllustrationValueType(events[index].event) == DefaultIllustrationValueType::Float32)
                labelWidth = std::max(labelWidth, label.size() + 5U);
        }
        labelWidth = std::max<std::size_t>(labelWidth, 8U);

        if (!firstGroup)
            output.push_back('\n');
        output.append("group y=");
        appendInteger(output, events[begin].outputRow);
        output.append(" x=");
        appendInteger(output, events[begin].groupStart);
        output.push_back('\n');
        for (std::size_t index = begin; index < end; ++index)
            printEvent(output, capture.bytes, events[index], labelWidth, inputFocus);
        firstGroup = false;
        begin = end;
    }
    return output;
}

}  // namespace kernel::image::internal
