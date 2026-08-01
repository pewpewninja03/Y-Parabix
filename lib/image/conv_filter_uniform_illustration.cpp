#include "conv_filter_uniform_illustration.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace kernel::image::internal {
namespace {

constexpr std::array<char, 3> ChannelNames{'R', 'G', 'B'};

struct EventType {
    UniformIllustrationValueType valueType;
    std::size_t fixedElementCount;
    unsigned logicalOutputMultiplier;
};

constexpr EventType FixedUInt8{UniformIllustrationValueType::UInt8, 3U, 0U};
constexpr EventType FixedInt32{UniformIllustrationValueType::Int32, 3U, 0U};
constexpr EventType FixedFloat32{UniformIllustrationValueType::Float32, 3U, 0U};
constexpr EventType GroupUInt8{UniformIllustrationValueType::UInt8, 0U, 1U};
constexpr EventType GroupInt32{UniformIllustrationValueType::Int32, 0U, 1U};
constexpr EventType GroupFloat32{UniformIllustrationValueType::Float32, 0U, 1U};
constexpr EventType PackedGroupUInt8{UniformIllustrationValueType::UInt8, 0U, 3U};

constexpr EventType EventTypes[] = {
    FixedUInt8,
    FixedInt32,
    FixedInt32,
    FixedInt32,
    FixedInt32,
    FixedInt32,
    FixedInt32,
    FixedFloat32,
    FixedUInt8,
    FixedUInt8,
    GroupInt32,
    GroupFloat32,
    GroupUInt8,
    PackedGroupUInt8,
};

static_assert(std::size(EventTypes) == static_cast<std::size_t>(UniformIllustrationEvent::Count), "Uniform illustration event table is incomplete");

const EventType & eventType(const UniformIllustrationEvent event) {
    const auto index = static_cast<std::size_t>(event);
    if (index >= static_cast<std::size_t>(UniformIllustrationEvent::Count))
        throw std::invalid_argument("Uniform illustration event is invalid");
    return EventTypes[index];
}

template <typename T>
T readElement(const UniformIllustrationCaptureLog & capture, const UniformIllustrationCaptureEvent & event, const std::size_t index) {
    if (sizeof(T) != uniformIllustrationElementByteCount(event.event) || index >= event.elementCount)
        throw std::logic_error("Uniform illustration event type mismatch");
    const std::size_t byteCount = event.elementCount * sizeof(T);
    if (event.byteOffset > capture.bytes.size() || byteCount > capture.bytes.size() - event.byteOffset)
        throw std::logic_error("Uniform illustration event storage is invalid");
    T value{};
    std::memcpy(&value, capture.bytes.data() + event.byteOffset + index * sizeof(T), sizeof(T));
    return value;
}

template <typename T>
std::string integerText(const T value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc{})
        throw std::logic_error("Uniform illustration integer formatting failed");
    return {buffer, result.ptr};
}

std::string floatText(const float value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general, 7);
    if (result.ec != std::errc{})
        throw std::logic_error("Uniform illustration float formatting failed");
    return {buffer, result.ptr};
}

std::string byteHex(const std::uint8_t value) {
    static constexpr char HexDigits[] = "0123456789ABCDEF";
    std::string text(2, '0');
    text[0] = HexDigits[value >> 4U];
    text[1] = HexDigits[value & 0x0fU];
    return text;
}

std::vector<std::string> coordinateValues(const std::vector<std::size_t> & coordinates) {
    std::vector<std::string> values;
    values.reserve(coordinates.size());
    for (const std::size_t coordinate : coordinates)
        values.push_back(integerText(coordinate));
    return values;
}

std::vector<std::string> maskValues(const std::vector<bool> & mask) {
    std::vector<std::string> values;
    values.reserve(mask.size());
    for (const bool value : mask)
        values.emplace_back(value ? "1" : ".");
    return values;
}

std::vector<std::string> packedHex(const UniformIllustrationCaptureLog & capture, const UniformIllustrationCaptureEvent & event) {
    if (uniformIllustrationValueType(event.event) != UniformIllustrationValueType::UInt8)
        throw std::logic_error("Uniform illustration packed event is not byte-valued");
    std::vector<std::string> values;
    values.reserve(event.elementCount);
    for (std::size_t index = 0; index < event.elementCount; ++index)
        values.push_back(byteHex(readElement<std::uint8_t>(capture, event, index)));
    return values;
}

struct TraceRow {
    std::string label;
    std::vector<std::string> values;
};

class TraceBuilder {
   public:
    void beginInitialization(const std::size_t row) {
        if (initialization)
            throw std::logic_error("Uniform illustration initialization is duplicated");
        initialization = Block{"window initialization y=" + integerText(row) + " x=0", {}};
    }

    void initializationRow(std::string label, std::vector<std::string> values) {
        if (!initialization)
            throw std::logic_error("Uniform illustration initialization row has no block");
        initialization->rows.push_back({std::move(label), std::move(values)});
    }

    void beginGroup(const std::size_t row, const std::size_t column) {
        groups.push_back({"group y=" + integerText(row) + " x=" + integerText(column), {}});
    }

    void row(std::string label, std::vector<std::string> values) {
        if (groups.empty())
            throw std::logic_error("Uniform illustration row has no group");
        groups.back().rows.push_back({std::move(label), std::move(values)});
    }

    std::string text() const {
        std::string output;
        if (initialization)
            appendBlock(output, *initialization);
        for (std::size_t index = 0; index < groups.size(); ++index) {
            if (index != 0U)
                output.push_back('\n');
            appendBlock(output, groups[index]);
        }
        return output;
    }

   private:
    struct Block {
        std::string heading;
        std::vector<TraceRow> rows;
    };

    static void appendBlock(std::string & output, const Block & block) {
        output.append(block.heading).push_back('\n');
        std::size_t labelWidth = 0;
        for (const TraceRow & row : block.rows)
            labelWidth = std::max(labelWidth, row.label.size());
        for (const TraceRow & row : block.rows) {
            output.append(labelWidth - row.label.size(), ' ');
            output.append(row.label).append(" |");
            for (const std::string & value : row.values) {
                output.push_back(' ');
                output.append(value);
            }
            output.push_back('\n');
        }
    }

    std::optional<Block> initialization;
    std::vector<Block> groups;
};

struct HorizontalRecurrence {
    const UniformIllustrationCaptureEvent * leaving = nullptr;
    const UniformIllustrationCaptureEvent * entering = nullptr;
};

struct HorizontalInitialization {
    const UniformIllustrationCaptureEvent * operand = nullptr;
    const UniformIllustrationCaptureEvent * afterAdd = nullptr;
};

struct ChannelFinalization {
    const UniformIllustrationCaptureEvent * sum = nullptr;
    const UniformIllustrationCaptureEvent * weighted = nullptr;
    const UniformIllustrationCaptureEvent * output = nullptr;
};

struct OutputOperation {
    UniformIllustrationPath path = UniformIllustrationPath::SingleOutput;
    std::size_t row = NoUniformIllustrationCoordinate;
    std::size_t start = NoUniformIllustrationCoordinate;
    std::vector<std::size_t> columns;
    std::array<ChannelFinalization, 3> channels;
    ChannelFinalization rgb;
    const UniformIllustrationCaptureEvent * packedOutput = nullptr;
};

struct SelectedInputEvents {
    const UniformIllustrationCaptureEvent * packed = nullptr;
    const UniformIllustrationCaptureEvent * rgb = nullptr;
};

using RecurrenceKey = std::pair<std::size_t, std::size_t>;
using InitializationKey = std::pair<std::size_t, std::size_t>;
using OperationKey = std::tuple<std::size_t, std::size_t, UniformIllustrationPath>;

void assignUnique(const UniformIllustrationCaptureEvent *& destination, const UniformIllustrationCaptureEvent & event, const char * description) {
    if (destination != nullptr)
        throw std::logic_error(std::string("Uniform illustration duplicate ") + description);
    destination = &event;
}

class EventIndex {
   public:
    EventIndex(const UniformIllustrationConfiguration & configuration, const UniformIllustrationCaptureLog & capture)
        : configuration(configuration), capture(capture) {
        for (const UniformIllustrationCaptureEvent & event : capture.events)
            index(event);
        finishOperations();
    }

    const std::map<OperationKey, OutputOperation> & operations() const noexcept {
        return outputOperations;
    }

    const HorizontalRecurrence * recurrence(const std::size_t row, const std::size_t sourceColumn) const {
        const auto found = recurrences.find({row, sourceColumn});
        return found == recurrences.end() ? nullptr : &found->second;
    }

    const HorizontalInitialization & initialization(const std::size_t row, const std::size_t workspaceColumn) const {
        const auto found = initializations.find({row, workspaceColumn});
        if (found == initializations.end())
            throw std::logic_error("Uniform illustration lacks a horizontal initialization event");
        return found->second;
    }

    const UniformIllustrationCaptureEvent & workspaceColumn(const std::size_t row, const std::size_t column) const {
        const UniformIllustrationCaptureEvent * result = nullptr;
        for (const UniformIllustrationCaptureEvent & event : capture.events) {
            const bool generatedLoad = event.event == UniformIllustrationEvent::HorizontalInitialOperand
                                       || event.event == UniformIllustrationEvent::HorizontalEnteringOperand;
            if (generatedLoad && event.outputRow == row && event.workspaceColumn == column)
                assignUnique(result, event, "workspace-column load");
        }
        if (result == nullptr)
            throw std::logic_error("Uniform illustration lacks a workspace-column load");
        return *result;
    }

    SelectedInputEvents selectedInput(const ConvFilterIllustrationSelection selection) const {
        SelectedInputEvents result;
        for (const UniformIllustrationCaptureEvent & event : capture.events) {
            if (event.workspaceColumn == selection.column && event.sourceInputRow == selection.row
                && event.event == UniformIllustrationEvent::InputRgb)
            {
                assignUnique(result.rgb, event, "selected input RGB");
            }
        }
        if (result.rgb == nullptr)
            throw std::logic_error("Uniform illustration lacks the selected input RGB value");
        for (const UniformIllustrationCaptureEvent & event : capture.events) {
            if (event.event == UniformIllustrationEvent::PackedInput && event.path == result.rgb->path
                && event.workspaceColumn == result.rgb->workspaceColumn && event.sourceInputRow == result.rgb->sourceInputRow
                && event.recurrenceSource == result.rgb->recurrenceSource && event.recurrenceDestination == result.rgb->recurrenceDestination)
            {
                assignUnique(result.packed, event, "selected packed input");
            }
        }
        if (result.packed == nullptr)
            throw std::logic_error("Uniform illustration lacks the selected packed input");
        return result;
    }

   private:
    static ChannelFinalization & finalization(OutputOperation & operation, const UniformIllustrationCaptureEvent & event) {
        if (event.path == UniformIllustrationPath::SingleOutput)
            return operation.rgb;
        if (event.channel >= operation.channels.size())
            throw std::logic_error("Uniform illustration grouped channel is invalid");
        return operation.channels[event.channel];
    }

    OutputOperation & operation(const UniformIllustrationCaptureEvent & event) {
        const std::size_t start = event.path == UniformIllustrationPath::AdjacentOutputs ? event.groupStart : event.outputColumn;
        auto [position, inserted] = outputOperations.try_emplace({event.outputRow, start, event.path});
        OutputOperation & output = position->second;
        if (inserted) {
            output.path = event.path;
            output.row = event.outputRow;
            output.start = start;
        }
        return output;
    }

    void assignFinalization(OutputOperation & output, const UniformIllustrationCaptureEvent & event) {
        ChannelFinalization & channel = finalization(output, event);
        switch (event.event) {
        case UniformIllustrationEvent::SingleSum:
        case UniformIllustrationEvent::GroupSum:
            assignUnique(channel.sum, event, "integer sum");
            return;
        case UniformIllustrationEvent::SingleWeighted:
        case UniformIllustrationEvent::GroupWeighted:
            assignUnique(channel.weighted, event, "weighted sum");
            return;
        case UniformIllustrationEvent::SingleOutputBytes:
        case UniformIllustrationEvent::GroupOutputBytes:
            assignUnique(channel.output, event, "output bytes");
            return;
        default:
            return;
        }
    }

    void index(const UniformIllustrationCaptureEvent & event) {
        switch (event.event) {
        case UniformIllustrationEvent::HorizontalInitialOperand:
            assignUnique(initializations[{event.outputRow, event.workspaceColumn}].operand, event, "horizontal initialization operand");
            return;
        case UniformIllustrationEvent::HorizontalInitialAfterAdd:
            assignUnique(initializations[{event.outputRow, event.workspaceColumn}].afterAdd, event, "horizontal initialization result");
            return;
        case UniformIllustrationEvent::HorizontalLeavingOperand:
            assignUnique(recurrences[{event.outputRow, event.recurrenceSource}].leaving, event, "leaving workspace value");
            return;
        case UniformIllustrationEvent::HorizontalEnteringOperand:
            assignUnique(recurrences[{event.outputRow, event.recurrenceSource}].entering, event, "entering workspace value");
            return;
        case UniformIllustrationEvent::SinglePackedOutput:
        case UniformIllustrationEvent::GroupPackedOutput:
            assignUnique(operation(event).packedOutput, event, "packed output");
            return;
        case UniformIllustrationEvent::SingleSum:
        case UniformIllustrationEvent::SingleWeighted:
        case UniformIllustrationEvent::SingleOutputBytes:
        case UniformIllustrationEvent::GroupSum:
        case UniformIllustrationEvent::GroupWeighted:
        case UniformIllustrationEvent::GroupOutputBytes:
            assignFinalization(operation(event), event);
            return;
        default:
            return;
        }
    }

    void finishOperations() {
        for (auto & [key, output] : outputOperations) {
            if (output.packedOutput == nullptr)
                throw std::logic_error("Uniform illustration output operation lacks packed bytes");
            const std::size_t count = output.path == UniformIllustrationPath::AdjacentOutputs ? configuration.logicalOutputs : 1U;
            output.columns.reserve(count);
            for (std::size_t position = 0; position < count; ++position)
                output.columns.push_back(output.start + position);
        }
    }

    const UniformIllustrationConfiguration & configuration;
    const UniformIllustrationCaptureLog & capture;
    std::map<InitializationKey, HorizontalInitialization> initializations;
    std::map<RecurrenceKey, HorizontalRecurrence> recurrences;
    std::map<OperationKey, OutputOperation> outputOperations;
};

const UniformIllustrationCaptureEvent & required(const UniformIllustrationCaptureEvent * event, const char * description) {
    if (event == nullptr)
        throw std::logic_error(std::string("Uniform illustration lacks ") + description);
    return *event;
}

std::vector<std::string> repeatedCoordinate(const std::vector<bool> & selected, const std::size_t coordinate) {
    std::vector<std::string> values;
    values.reserve(selected.size());
    for (const bool isSelected : selected)
        values.push_back(isSelected ? integerText(coordinate) : ".");
    return values;
}

std::vector<std::string> repeatedInteger(
    const UniformIllustrationCaptureLog & capture,
    const UniformIllustrationCaptureEvent & event,
    const std::size_t element,
    const std::vector<bool> & selected
) {
    const std::string value = integerText(readElement<std::int32_t>(capture, event, element));
    std::vector<std::string> values;
    values.reserve(selected.size());
    for (const bool isSelected : selected)
        values.push_back(isSelected ? value : ".");
    return values;
}

std::vector<std::string> repeatedPacked(
    const UniformIllustrationCaptureLog & capture,
    const UniformIllustrationCaptureEvent & event,
    const std::vector<bool> & selected,
    const bool hexadecimal
) {
    std::array<std::string, 3> packed;
    for (std::size_t channel = 0; channel < packed.size(); ++channel) {
        const std::uint8_t value = readElement<std::uint8_t>(capture, event, channel);
        packed[channel] = hexadecimal ? byteHex(value) : integerText(static_cast<unsigned>(value));
    }
    std::vector<std::string> values;
    values.reserve(selected.size() * packed.size());
    for (const bool isSelected : selected) {
        if (!isSelected) {
            values.emplace_back(".");
            continue;
        }
        values.insert(values.end(), packed.begin(), packed.end());
    }
    return values;
}

std::vector<std::string> channelIntegers(
    const UniformIllustrationCaptureLog & capture, const OutputOperation & operation, const std::size_t channel
) {
    const ChannelFinalization & finalization =
        operation.path == UniformIllustrationPath::AdjacentOutputs ? operation.channels[channel] : operation.rgb;
    const UniformIllustrationCaptureEvent & event = required(finalization.sum, "integer window sum");
    if (operation.path == UniformIllustrationPath::AdjacentOutputs) {
        std::vector<std::string> values;
        values.reserve(event.elementCount);
        for (std::size_t index = 0; index < event.elementCount; ++index)
            values.push_back(integerText(readElement<std::int32_t>(capture, event, index)));
        return values;
    }
    return {integerText(readElement<std::int32_t>(capture, event, channel))};
}

std::vector<std::string> channelOutput(const UniformIllustrationCaptureLog & capture, const OutputOperation & operation, const std::size_t channel) {
    const ChannelFinalization & finalization =
        operation.path == UniformIllustrationPath::AdjacentOutputs ? operation.channels[channel] : operation.rgb;
    const UniformIllustrationCaptureEvent & event = required(finalization.output, "output channel bytes");
    if (operation.path == UniformIllustrationPath::AdjacentOutputs) {
        std::vector<std::string> values;
        values.reserve(event.elementCount);
        for (std::size_t index = 0; index < event.elementCount; ++index)
            values.push_back(byteHex(readElement<std::uint8_t>(capture, event, index)));
        return values;
    }
    return {byteHex(readElement<std::uint8_t>(capture, event, channel))};
}

struct RecurrenceRows {
    std::vector<std::string> leavingColumn;
    std::array<std::vector<std::string>, 3> leavingChannels;
    std::vector<std::string> enteringColumn;
    std::array<std::vector<std::string>, 3> enteringChannels;
};

RecurrenceRows recurrenceRows(
    const UniformIllustrationConfiguration & configuration,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index,
    const OutputOperation & operation
) {
    RecurrenceRows rows;
    const std::size_t count = operation.columns.size();
    const std::size_t radius = configuration.kernelWidth / 2U;
    for (std::size_t position = 0; position < count; ++position) {
        const bool hasProducer = operation.path == UniformIllustrationPath::SingleOutput ? operation.columns[position] != 0U : position != 0U;
        const std::size_t destination = operation.columns[position];
        const std::size_t source = hasProducer ? destination - 1U : NoUniformIllustrationCoordinate;
        const bool leavingValid = hasProducer && source >= radius;
        const bool enteringValid = hasProducer && destination + radius < configuration.imageWidth;
        const HorizontalRecurrence * recurrence = hasProducer ? index.recurrence(operation.row, source) : nullptr;

        rows.leavingColumn.push_back(leavingValid ? integerText(source - radius) : ".");
        rows.enteringColumn.push_back(enteringValid ? integerText(destination + radius) : ".");
        for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel) {
            if (leavingValid) {
                const auto & event = required(recurrence == nullptr ? nullptr : recurrence->leaving, "leaving workspace value");
                rows.leavingChannels[channel].push_back(integerText(readElement<std::int32_t>(capture, event, channel)));
            } else {
                rows.leavingChannels[channel].emplace_back(".");
            }
            if (enteringValid) {
                const auto & event = required(recurrence == nullptr ? nullptr : recurrence->entering, "entering workspace value");
                rows.enteringChannels[channel].push_back(integerText(readElement<std::int32_t>(capture, event, channel)));
            } else {
                rows.enteringChannels[channel].emplace_back(".");
            }
        }
    }
    return rows;
}

void commonRows(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const OutputOperation & operation,
    const std::vector<bool> & selected,
    const char * selectionLabel
) {
    trace.row("outputX", coordinateValues(operation.columns));
    trace.row(selectionLabel, maskValues(selected));
    std::vector<bool> valid;
    valid.reserve(operation.columns.size());
    for (const std::size_t column : operation.columns)
        valid.push_back(column < configuration.imageWidth);
    trace.row("inOutputDomain", maskValues(valid));
}

void initializationRows(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index,
    const std::size_t outputRow
) {
    const std::size_t columnCount = std::min<std::size_t>(configuration.imageWidth, configuration.kernelWidth / 2U + 1U);
    std::vector<std::size_t> columns;
    std::array<std::vector<std::string>, 3> operands;
    std::array<std::vector<std::string>, 3> cumulative;
    columns.reserve(columnCount);
    for (std::size_t column = 0; column < columnCount; ++column) {
        const HorizontalInitialization & initialization = index.initialization(outputRow, column);
        const auto & operand = required(initialization.operand, "horizontal initialization operand");
        const auto & afterAdd = required(initialization.afterAdd, "horizontal initialization result");
        columns.push_back(column);
        for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel) {
            operands[channel].push_back(integerText(readElement<std::int32_t>(capture, operand, channel)));
            cumulative[channel].push_back(integerText(readElement<std::int32_t>(capture, afterAdd, channel)));
        }
    }

    trace.beginInitialization(outputRow);
    trace.initializationRow("columnX", coordinateValues(columns));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.initializationRow(std::string("columnSum") + ChannelNames[channel], std::move(operands[channel]));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.initializationRow(std::string("partialSum") + ChannelNames[channel], std::move(cumulative[channel]));
}

void outputRows(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index,
    const OutputOperation & operation,
    const std::vector<bool> & selected
) {
    trace.beginGroup(operation.row, operation.start);
    commonRows(trace, configuration, operation, selected, "target");

    const std::size_t radius = configuration.kernelWidth / 2U;
    std::vector<std::string> left;
    std::vector<std::string> right;
    for (const std::size_t column : operation.columns) {
        left.push_back(integerText(column > radius ? column - radius : 0U));
        right.push_back(integerText(std::min<std::size_t>(configuration.imageWidth - 1U, column + radius)));
    }
    trace.row("windowL", std::move(left));
    trace.row("windowR", std::move(right));

    RecurrenceRows recurrence = recurrenceRows(configuration, capture, index, operation);
    trace.row("subColumnX", std::move(recurrence.leavingColumn));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("subColumnSum") + ChannelNames[channel], std::move(recurrence.leavingChannels[channel]));
    trace.row("addColumnX", std::move(recurrence.enteringColumn));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("addColumnSum") + ChannelNames[channel], std::move(recurrence.enteringChannels[channel]));

    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("windowSum") + ChannelNames[channel], channelIntegers(capture, operation, channel));
    trace.row("w", std::vector<std::string>(operation.columns.size(), floatText(configuration.weight)));
    if (configuration.weight > 0.0F) {
        for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel) {
            const ChannelFinalization & finalization =
                operation.path == UniformIllustrationPath::AdjacentOutputs ? operation.channels[channel] : operation.rgb;
            const UniformIllustrationCaptureEvent & event = required(finalization.weighted, "weighted window sum");
            std::vector<std::string> values;
            if (operation.path == UniformIllustrationPath::AdjacentOutputs) {
                values.reserve(event.elementCount);
                for (std::size_t index = 0; index < event.elementCount; ++index)
                    values.push_back(floatText(readElement<float>(capture, event, index)));
            } else {
                values.push_back(floatText(readElement<float>(capture, event, channel)));
            }
            trace.row(std::string("weightedSum") + ChannelNames[channel], std::move(values));
        }
    }
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("output") + ChannelNames[channel], channelOutput(capture, operation, channel));
    trace.row("outputRGB.hex", packedHex(capture, required(operation.packedOutput, "packed output")));
}

void inputRows(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index,
    const OutputOperation & operation,
    const std::vector<bool> & selected,
    const SelectedInputEvents input
) {
    trace.beginGroup(operation.row, operation.start);
    commonRows(trace, configuration, operation, selected, "inWindow");
    trace.row("inputY", repeatedCoordinate(selected, selection.row));
    trace.row("inputX", repeatedCoordinate(selected, selection.column));
    trace.row("inputRGB.hex", repeatedPacked(capture, required(input.packed, "selected packed input"), selected, true));
    trace.row("inputRGB.dec", repeatedPacked(capture, required(input.packed, "selected packed input"), selected, false));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel) {
        trace.row(
            std::string("input") + ChannelNames[channel], repeatedInteger(capture, required(input.rgb, "selected input RGB"), channel, selected)
        );
    }
    trace.row("columnX", repeatedCoordinate(selected, selection.column));
    const UniformIllustrationCaptureEvent & columnSum = index.workspaceColumn(operation.row, selection.column);
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("columnSum") + ChannelNames[channel], repeatedInteger(capture, columnSum, channel, selected));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("windowSum") + ChannelNames[channel], channelIntegers(capture, operation, channel));
    for (std::size_t channel = 0; channel < ChannelNames.size(); ++channel)
        trace.row(std::string("output") + ChannelNames[channel], channelOutput(capture, operation, channel));
    trace.row("outputRGB.hex", packedHex(capture, required(operation.packedOutput, "packed output")));
}

const OutputOperation & selectedOutputOperation(const EventIndex & index, const ConvFilterIllustrationSelection selection) {
    for (const auto & [key, operation] : index.operations()) {
        if (operation.row == selection.row
            && std::find(operation.columns.begin(), operation.columns.end(), selection.column) != operation.columns.end())
        {
            return operation;
        }
    }
    throw std::logic_error("Uniform illustration lacks the selected output operation");
}

void renderOutput(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index
) {
    const OutputOperation & operation = selectedOutputOperation(index, selection);
    if (operation.start == 0U)
        initializationRows(trace, configuration, capture, index, operation.row);
    std::vector<bool> selected;
    selected.reserve(operation.columns.size());
    for (const std::size_t column : operation.columns)
        selected.push_back(operation.row == selection.row && column == selection.column);
    outputRows(trace, configuration, capture, index, operation, selected);
}

void renderInput(
    TraceBuilder & trace,
    const UniformIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const UniformIllustrationCaptureLog & capture,
    const EventIndex & index
) {
    const std::size_t radiusY = configuration.kernelHeight / 2U;
    const std::size_t radiusX = configuration.kernelWidth / 2U;
    const std::size_t firstRow = selection.row > radiusY ? selection.row - radiusY : 0U;
    const std::size_t lastRow = std::min<std::size_t>(configuration.imageHeight - 1U, selection.row + radiusY);
    const std::size_t firstColumn = selection.column > radiusX ? selection.column - radiusX : 0U;
    const std::size_t lastColumn = std::min<std::size_t>(configuration.imageWidth - 1U, selection.column + radiusX);
    const SelectedInputEvents input = index.selectedInput(selection);
    for (const auto & [key, operation] : index.operations()) {
        std::vector<bool> selected;
        selected.reserve(operation.columns.size());
        for (const std::size_t column : operation.columns)
            selected.push_back(operation.row >= firstRow && operation.row <= lastRow && column >= firstColumn && column <= lastColumn);
        if (std::none_of(selected.begin(), selected.end(), [](const bool value) { return value; }))
            continue;
        inputRows(trace, configuration, selection, capture, index, operation, selected, input);
    }
}

}  // namespace

UniformIllustrationValueType uniformIllustrationValueType(const UniformIllustrationEvent event) {
    return eventType(event).valueType;
}

std::size_t uniformIllustrationElementByteCount(const UniformIllustrationEvent event) {
    switch (uniformIllustrationValueType(event)) {
    case UniformIllustrationValueType::Float32:
    case UniformIllustrationValueType::Int32:
        return sizeof(std::uint32_t);
    case UniformIllustrationValueType::UInt8:
        return sizeof(std::uint8_t);
    }
    throw std::logic_error("Uniform illustration value type is invalid");
}

extern "C" void captureUniformConvFilterValue(
    std::uint8_t * context,
    const std::uint32_t eventValue,
    const std::uint32_t pathValue,
    const std::uint32_t channel,
    const std::size_t elementCount,
    const std::size_t elementByteCount,
    const std::uint8_t * bytes,
    const std::size_t outputRow,
    const std::size_t outputColumn,
    const std::size_t groupStart,
    const std::size_t workspaceColumn,
    const std::size_t sourceInputRow,
    const std::size_t recurrenceSource,
    const std::size_t recurrenceDestination
) noexcept {
    if (context == nullptr)
        return;
    auto & capture = *reinterpret_cast<UniformIllustrationCaptureLog *>(context);
    if (capture.failure)
        return;
    try {
        const auto event = static_cast<UniformIllustrationEvent>(eventValue);
        const auto path = static_cast<UniformIllustrationPath>(pathValue);
        switch (path) {
        case UniformIllustrationPath::ColumnInitialization:
        case UniformIllustrationPath::ColumnUpdate:
        case UniformIllustrationPath::WindowInitialization:
        case UniformIllustrationPath::AdjacentOutputs:
        case UniformIllustrationPath::SingleOutput:
            break;
        default:
            throw std::invalid_argument("Uniform illustration path is invalid");
        }
        const EventType & type = eventType(event);
        std::size_t expectedCount = type.fixedElementCount;
        if (expectedCount == 0U) {
            if (capture.logicalOutputs != 0U && type.logicalOutputMultiplier > std::numeric_limits<std::size_t>::max() / capture.logicalOutputs) {
                throw std::overflow_error("Uniform illustration element count overflow");
            }
            expectedCount = capture.logicalOutputs * type.logicalOutputMultiplier;
        }
        if (elementCount != expectedCount)
            throw std::logic_error("Uniform illustration element count mismatch");
        const std::size_t expectedByteCount = uniformIllustrationElementByteCount(event);
        if (elementByteCount != expectedByteCount)
            throw std::logic_error("Uniform illustration element type mismatch");
        if (elementCount != 0U && expectedByteCount > std::numeric_limits<std::size_t>::max() / elementCount)
            throw std::overflow_error("Uniform illustration capture size overflow");
        const std::size_t byteCount = elementCount * expectedByteCount;
        const std::size_t byteOffset = capture.bytes.size();
        if (byteCount > std::numeric_limits<std::size_t>::max() - byteOffset)
            throw std::overflow_error("Uniform illustration capture storage overflow");
        if (byteCount != 0U && bytes == nullptr)
            throw std::invalid_argument("Uniform illustration capture bytes are null");
        capture.bytes.resize(byteOffset + byteCount);
        std::memcpy(capture.bytes.data() + byteOffset, bytes, byteCount);
        capture.events.push_back(
            {event,
             path,
             channel,
             elementCount,
             byteOffset,
             outputRow,
             outputColumn,
             groupStart,
             workspaceColumn,
             sourceInputRow,
             recurrenceSource,
             recurrenceDestination}
        );
    } catch (...) {
        capture.failure = std::current_exception();
    }
}

std::string formatUniformConvFilterIllustration(
    const UniformIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const UniformIllustrationCaptureLog & capture
) {
    if (capture.failure)
        std::rethrow_exception(capture.failure);
    if (capture.logicalOutputs != configuration.logicalOutputs)
        throw std::logic_error("Uniform illustration logical output count mismatch");

    const EventIndex index(configuration, capture);
    TraceBuilder trace;
    switch (selection.kind) {
    case ConvFilterIllustrationSelectionKind::Output:
        renderOutput(trace, configuration, selection, capture, index);
        break;
    case ConvFilterIllustrationSelectionKind::Input:
        renderInput(trace, configuration, selection, capture, index);
        break;
    default:
        throw std::invalid_argument("Uniform illustration selection kind is invalid");
    }
    return trace.text();
}

}  // namespace kernel::image::internal
