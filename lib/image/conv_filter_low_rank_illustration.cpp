#include "conv_filter_low_rank_illustration.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace kernel::image::internal {
namespace {

constexpr std::array<char, 3> ChannelNames{'R', 'G', 'B'};

struct EventAbi {
    LowRankIllustrationElementType elementType;
    std::size_t fixedElementCount;
    unsigned logicalOutputMultiplier;
};

constexpr EventAbi ScalarFloat{LowRankIllustrationElementType::Float32, 1U, 0U};
constexpr EventAbi ScalarByte{LowRankIllustrationElementType::UInt8, 1U, 0U};
constexpr EventAbi LaneFloat{LowRankIllustrationElementType::Float32, 0U, 1U};
constexpr EventAbi LaneByte{LowRankIllustrationElementType::UInt8, 0U, 1U};
constexpr EventAbi PackedByte{LowRankIllustrationElementType::UInt8, 0U, 3U};

constexpr EventAbi EventAbis[] = {
    PackedByte,
    ScalarByte,
    LaneFloat,
    ScalarFloat,
    LaneFloat,
    LaneFloat,
    LaneFloat,
    ScalarFloat,
    LaneFloat,
    LaneByte,
    PackedByte,
    ScalarByte,
    LaneByte,
};

static_assert(
    sizeof(EventAbis) / sizeof(EventAbis[0]) == static_cast<std::size_t>(LowRankIllustrationEvent::Count),
    "LowRank illustration event ABI is incomplete"
);

const EventAbi & eventAbi(const LowRankIllustrationEvent event) {
    const std::size_t index = static_cast<std::size_t>(event);
    if (index >= static_cast<std::size_t>(LowRankIllustrationEvent::Count))
        throw std::invalid_argument("LowRank illustration event is invalid");
    return EventAbis[index];
}

unsigned passOrder(const LowRankIllustrationEvent event) {
    switch (event) {
    case LowRankIllustrationEvent::HorizontalPackedInput:
    case LowRankIllustrationEvent::HorizontalInputByte:
    case LowRankIllustrationEvent::HorizontalSample:
    case LowRankIllustrationEvent::HorizontalFactor:
    case LowRankIllustrationEvent::HorizontalAccumulator:
    case LowRankIllustrationEvent::HorizontalWorkspaceStore:
        return 0;
    case LowRankIllustrationEvent::VerticalWorkspaceLoad:
    case LowRankIllustrationEvent::VerticalFactor:
    case LowRankIllustrationEvent::VerticalAccumulator:
        return 1;
    case LowRankIllustrationEvent::OutputByteVector:
    case LowRankIllustrationEvent::PackedOutput:
    case LowRankIllustrationEvent::StoredOutputByte:
    case LowRankIllustrationEvent::StoreMask:
        return 2;
    case LowRankIllustrationEvent::Count:
        break;
    }
    throw std::invalid_argument("LowRank illustration event is invalid");
}

auto semanticKey(const LowRankIllustrationCaptureEvent & event) {
    return std::tuple{
        passOrder(event.event),
        event.row,
        event.groupStart,
        event.rank,
        event.horizontalTap,
        event.verticalTap,
        static_cast<unsigned>(event.event),
        event.lane,
        event.channel,
        event.paddedSourceRow,
        event.sourceColumn,
        static_cast<std::uint32_t>(event.path),
    };
}

void validateEventPath(const LowRankIllustrationEvent event, const LowRankIllustrationPath path) {
    bool valid = false;
    switch (event) {
    case LowRankIllustrationEvent::HorizontalPackedInput:
        valid = path == LowRankIllustrationPath::HorizontalInterior;
        break;
    case LowRankIllustrationEvent::HorizontalInputByte:
        valid = path == LowRankIllustrationPath::HorizontalBorder;
        break;
    case LowRankIllustrationEvent::HorizontalSample:
    case LowRankIllustrationEvent::HorizontalFactor:
    case LowRankIllustrationEvent::HorizontalAccumulator:
        valid = path == LowRankIllustrationPath::HorizontalInterior || path == LowRankIllustrationPath::HorizontalBorder;
        break;
    case LowRankIllustrationEvent::HorizontalWorkspaceStore:
        valid = path == LowRankIllustrationPath::HorizontalResult;
        break;
    case LowRankIllustrationEvent::VerticalWorkspaceLoad:
    case LowRankIllustrationEvent::VerticalFactor:
    case LowRankIllustrationEvent::VerticalAccumulator:
        valid = path == LowRankIllustrationPath::VerticalFull || path == LowRankIllustrationPath::VerticalEdge;
        break;
    case LowRankIllustrationEvent::OutputByteVector:
    case LowRankIllustrationEvent::PackedOutput:
        valid = path == LowRankIllustrationPath::OutputFull;
        break;
    case LowRankIllustrationEvent::StoredOutputByte:
        valid = path == LowRankIllustrationPath::OutputChecked;
        break;
    case LowRankIllustrationEvent::StoreMask:
        valid = path == LowRankIllustrationPath::OutputFull || path == LowRankIllustrationPath::OutputChecked;
        break;
    case LowRankIllustrationEvent::Count:
        break;
    }
    if (!valid)
        throw std::logic_error("LowRank illustration event path mismatch");
}

struct Row {
    std::string label;
    std::vector<std::string> values;
};

struct Block {
    std::string heading;
    std::vector<Row> rows;
};

template <typename T>
T readElement(const LowRankIllustrationCaptureLog & capture, const LowRankIllustrationCaptureEvent & event, const std::size_t index) {
    if (index >= event.elementCount || event.byteOffset > capture.bytes.size() || event.byteCount > capture.bytes.size() - event.byteOffset)
        throw std::logic_error("LowRank formatter event storage is invalid");
    T value{};
    std::memcpy(&value, capture.bytes.data() + event.byteOffset + index * sizeof(T), sizeof(T));
    return value;
}

std::string decimal(const std::size_t value) {
    char buffer[32];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value);
    if (result.ec != std::errc{})
        throw std::logic_error("LowRank formatter integer formatting failed");
    return std::string(buffer, result.ptr);
}

std::string decimal(const unsigned value) {
    return decimal(static_cast<std::size_t>(value));
}

std::string floating(const float value) {
    char buffer[64];
    const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::general, 7);
    if (result.ec != std::errc{})
        throw std::logic_error("LowRank formatter float formatting failed");
    return std::string(buffer, result.ptr);
}

std::string hexadecimal(const std::uint8_t value) {
    static constexpr char Digits[] = "0123456789ABCDEF";
    std::string result(2U, '0');
    result[0] = Digits[value >> 4U];
    result[1] = Digits[value & 0xfU];
    return result;
}

void appendBlock(std::string & output, const Block & block, const bool blankBefore = false) {
    if (blankBefore)
        output.push_back('\n');
    if (!block.heading.empty())
        output.append(block.heading).push_back('\n');
    std::size_t labelWidth = 0;
    for (const Row & row : block.rows)
        labelWidth = std::max(labelWidth, row.label.size());
    for (const Row & row : block.rows) {
        output.append(labelWidth - row.label.size(), ' ');
        output.append(row.label).append(" |");
        for (const std::string & value : row.values) {
            output.push_back(' ');
            output.append(value);
        }
        output.push_back('\n');
    }
}

using GroupKey = std::pair<std::size_t, std::size_t>;
using HorizontalKey = std::tuple<LowRankIllustrationEvent, std::size_t, std::size_t, std::size_t, std::size_t, std::uint32_t>;
using InputByteKey = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::uint32_t>;
using VerticalKey = std::tuple<LowRankIllustrationEvent, std::size_t, std::size_t, std::size_t, std::size_t, std::uint32_t>;
using OutputKey = std::tuple<LowRankIllustrationEvent, std::size_t, std::size_t, std::uint32_t, std::size_t>;

class EventIndex {
   public:
    EventIndex(const LowRankIllustrationConfiguration & configuration, const LowRankIllustrationCaptureLog & capture) {
        for (const LowRankIllustrationCaptureEvent & event : capture.events) {
            validateMetadata(configuration, event);
            switch (event.event) {
            case LowRankIllustrationEvent::HorizontalPackedInput:
            case LowRankIllustrationEvent::HorizontalSample:
            case LowRankIllustrationEvent::HorizontalFactor:
            case LowRankIllustrationEvent::HorizontalAccumulator:
            case LowRankIllustrationEvent::HorizontalWorkspaceStore:
                insert(
                    horizontalEvents,
                    HorizontalKey{event.event, event.row, event.groupStart, event.rank, event.horizontalTap, event.channel},
                    event,
                    "horizontal event"
                );
                break;
            case LowRankIllustrationEvent::HorizontalInputByte:
                insert(
                    inputBytes,
                    InputByteKey{event.row, event.groupStart, event.rank, event.horizontalTap, event.lane, event.sourceColumn, event.channel},
                    event,
                    "border input byte"
                );
                break;
            case LowRankIllustrationEvent::VerticalWorkspaceLoad:
            case LowRankIllustrationEvent::VerticalFactor:
            case LowRankIllustrationEvent::VerticalAccumulator:
                insert(
                    verticalEvents,
                    VerticalKey{event.event, event.row, event.groupStart, event.rank, event.verticalTap, event.channel},
                    event,
                    "vertical event"
                );
                break;
            case LowRankIllustrationEvent::OutputByteVector:
            case LowRankIllustrationEvent::PackedOutput:
            case LowRankIllustrationEvent::StoredOutputByte:
            case LowRankIllustrationEvent::StoreMask:
                groups.emplace(event.row, event.groupStart);
                insert(outputEvents, OutputKey{event.event, event.row, event.groupStart, event.channel, event.lane}, event, "output event");
                break;
            case LowRankIllustrationEvent::Count:
                break;
            }
        }
        for (const LowRankIllustrationCaptureEvent & event : capture.events) {
            if (groups.count({event.row, event.groupStart}) == 0U)
                throw std::logic_error("LowRank formatter event has no output operation");
        }
    }

    const std::set<GroupKey> & outputGroups() const noexcept {
        return groups;
    }

    const LowRankIllustrationCaptureEvent & horizontal(
        const LowRankIllustrationEvent event,
        const std::size_t row,
        const std::size_t group,
        const std::size_t rank,
        const std::size_t tap,
        const std::uint32_t channel = NoLowRankIllustrationChannel
    ) const {
        return find(horizontalEvents, HorizontalKey{event, row, group, rank, tap, channel}, "horizontal event");
    }

    const LowRankIllustrationCaptureEvent & horizontalWorkspace(
        const std::size_t row, const std::size_t group, const std::size_t rank, const std::uint32_t channel
    ) const {
        return horizontal(LowRankIllustrationEvent::HorizontalWorkspaceStore, row, group, rank, NoLowRankIllustrationCoordinate, channel);
    }

    const LowRankIllustrationCaptureEvent & vertical(
        const LowRankIllustrationEvent event,
        const std::size_t row,
        const std::size_t group,
        const std::size_t rank,
        const std::size_t tap,
        const std::uint32_t channel = NoLowRankIllustrationChannel
    ) const {
        return find(verticalEvents, VerticalKey{event, row, group, rank, tap, channel}, "vertical event");
    }

    const LowRankIllustrationCaptureEvent & output(
        const LowRankIllustrationEvent event,
        const std::size_t row,
        const std::size_t group,
        const std::uint32_t channel = NoLowRankIllustrationChannel
    ) const {
        return find(outputEvents, OutputKey{event, row, group, channel, NoLowRankIllustrationCoordinate}, "output event");
    }

    const LowRankIllustrationCaptureEvent * optionalOutput(
        const LowRankIllustrationEvent event,
        const std::size_t row,
        const std::size_t group,
        const std::uint32_t channel = NoLowRankIllustrationChannel
    ) const {
        return findOptional(outputEvents, OutputKey{event, row, group, channel, NoLowRankIllustrationCoordinate});
    }

    const LowRankIllustrationCaptureEvent & storedOutput(
        const std::size_t row, const std::size_t group, const std::size_t lane, const std::uint32_t channel
    ) const {
        return find(outputEvents, OutputKey{LowRankIllustrationEvent::StoredOutputByte, row, group, channel, lane}, "stored output byte");
    }

    const LowRankIllustrationCaptureEvent * packedInput(
        const std::size_t row, const std::size_t group, const std::size_t rank, const std::size_t tap
    ) const {
        return findOptional(
            horizontalEvents, HorizontalKey{LowRankIllustrationEvent::HorizontalPackedInput, row, group, rank, tap, NoLowRankIllustrationChannel}
        );
    }

    const LowRankIllustrationCaptureEvent & inputByte(
        const std::size_t row,
        const std::size_t group,
        const std::size_t rank,
        const std::size_t tap,
        const std::size_t lane,
        const std::size_t sourceColumn,
        const std::uint32_t channel
    ) const {
        return find(inputBytes, InputByteKey{row, group, rank, tap, lane, sourceColumn, channel}, "border input byte");
    }

   private:
    template <typename Map, typename Key>
    static void insert(Map & events, Key key, const LowRankIllustrationCaptureEvent & event, const char * description) {
        if (!events.emplace(std::move(key), &event).second)
            throw std::logic_error(std::string("LowRank formatter duplicate ") + description);
    }

    template <typename Map, typename Key>
    static const LowRankIllustrationCaptureEvent * findOptional(const Map & events, const Key & key) {
        const auto entry = events.find(key);
        return entry == events.end() ? nullptr : entry->second;
    }

    template <typename Map, typename Key>
    static const LowRankIllustrationCaptureEvent & find(const Map & events, const Key & key, const char * description) {
        const LowRankIllustrationCaptureEvent * result = findOptional(events, key);
        if (result == nullptr)
            throw std::logic_error(std::string("LowRank formatter missing ") + description);
        return *result;
    }

    static void validateMetadata(const LowRankIllustrationConfiguration & configuration, const LowRankIllustrationCaptureEvent & event) {
        if (event.row >= configuration.imageHeight || event.groupStart >= configuration.imageWidth
            || event.groupStart % configuration.logicalOutputs != 0U)
            throw std::logic_error("LowRank formatter output group metadata is invalid");
        if (event.channel != NoLowRankIllustrationChannel && event.channel >= 3U)
            throw std::logic_error("LowRank formatter channel metadata is invalid");
        if (event.rank != NoLowRankIllustrationCoordinate && event.rank >= configuration.rank)
            throw std::logic_error("LowRank formatter rank metadata is invalid");
        if (event.horizontalTap != NoLowRankIllustrationCoordinate && event.horizontalTap >= configuration.kernelWidth)
            throw std::logic_error("LowRank formatter horizontal tap metadata is invalid");
        if (event.verticalTap != NoLowRankIllustrationCoordinate && event.verticalTap >= configuration.kernelHeight)
            throw std::logic_error("LowRank formatter vertical tap metadata is invalid");
        if (event.lane != NoLowRankIllustrationCoordinate && event.lane >= configuration.logicalOutputs)
            throw std::logic_error("LowRank formatter lane metadata is invalid");
        const bool hasRank = event.rank != NoLowRankIllustrationCoordinate;
        const bool hasHorizontalTap = event.horizontalTap != NoLowRankIllustrationCoordinate;
        const bool hasVerticalTap = event.verticalTap != NoLowRankIllustrationCoordinate;
        const bool hasChannel = event.channel != NoLowRankIllustrationChannel;
        const bool hasLane = event.lane != NoLowRankIllustrationCoordinate;
        bool validShape = false;
        switch (event.event) {
        case LowRankIllustrationEvent::HorizontalPackedInput:
            validShape = hasRank && hasHorizontalTap && !hasVerticalTap && !hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::HorizontalInputByte:
            validShape = hasRank && hasHorizontalTap && !hasVerticalTap && hasChannel && hasLane;
            break;
        case LowRankIllustrationEvent::HorizontalSample:
        case LowRankIllustrationEvent::HorizontalAccumulator:
            validShape = hasRank && hasHorizontalTap && !hasVerticalTap && hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::HorizontalFactor:
            validShape = hasRank && hasHorizontalTap && !hasVerticalTap && !hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::HorizontalWorkspaceStore:
            validShape = hasRank && !hasHorizontalTap && !hasVerticalTap && hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::VerticalWorkspaceLoad:
        case LowRankIllustrationEvent::VerticalAccumulator:
            validShape = hasRank && !hasHorizontalTap && hasVerticalTap && hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::VerticalFactor:
            validShape = hasRank && !hasHorizontalTap && hasVerticalTap && !hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::OutputByteVector:
            validShape = !hasRank && !hasHorizontalTap && !hasVerticalTap && hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::PackedOutput:
        case LowRankIllustrationEvent::StoreMask:
            validShape = !hasRank && !hasHorizontalTap && !hasVerticalTap && !hasChannel && !hasLane;
            break;
        case LowRankIllustrationEvent::StoredOutputByte:
            validShape = !hasRank && !hasHorizontalTap && !hasVerticalTap && hasChannel && hasLane;
            break;
        case LowRankIllustrationEvent::Count:
            break;
        }
        if (!validShape)
            throw std::logic_error("LowRank formatter event metadata shape is invalid");
    }

    std::map<HorizontalKey, const LowRankIllustrationCaptureEvent *> horizontalEvents;
    std::map<InputByteKey, const LowRankIllustrationCaptureEvent *> inputBytes;
    std::map<VerticalKey, const LowRankIllustrationCaptureEvent *> verticalEvents;
    std::map<OutputKey, const LowRankIllustrationCaptureEvent *> outputEvents;
    std::set<GroupKey> groups;
};

std::vector<std::string> outputColumns(const LowRankIllustrationConfiguration & configuration, const std::size_t group) {
    std::vector<std::string> values(configuration.logicalOutputs);
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane)
        values[lane] = decimal(group + lane);
    return values;
}

std::vector<std::string> selectedOutput(
    const LowRankIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const std::size_t row,
    const std::size_t group
) {
    std::vector<std::string> values(configuration.logicalOutputs);
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
        const std::size_t column = group + lane;
        values[lane] =
            selection.kind == ConvFilterIllustrationSelectionKind::Output && row == selection.row && column == selection.column ? "1" : ".";
    }
    return values;
}

std::vector<std::string> maskValues(const LowRankIllustrationCaptureLog & capture, const LowRankIllustrationCaptureEvent & event) {
    std::vector<std::string> values(event.elementCount);
    for (std::size_t lane = 0; lane < event.elementCount; ++lane)
        values[lane] = readElement<std::uint8_t>(capture, event, lane) == 0U ? "." : "1";
    return values;
}

std::vector<std::string> floatVector(
    const LowRankIllustrationCaptureLog & capture, const LowRankIllustrationCaptureEvent & event, const std::vector<std::string> * mask = nullptr
) {
    std::vector<std::string> values(event.elementCount);
    for (std::size_t lane = 0; lane < event.elementCount; ++lane) {
        if (mask != nullptr && (*mask)[lane] == ".")
            values[lane] = ".";
        else
            values[lane] = floating(readElement<float>(capture, event, lane));
    }
    return values;
}

void validateWorkspaceValues(
    const LowRankIllustrationCaptureLog & capture,
    const LowRankIllustrationCaptureEvent & stored,
    const LowRankIllustrationCaptureEvent & loaded,
    const std::vector<std::string> & valid
) {
    if (stored.elementCount != loaded.elementCount || stored.elementCount != valid.size())
        throw std::logic_error("LowRank formatter workspace vector size mismatch");
    for (std::size_t lane = 0; lane < stored.elementCount; ++lane) {
        if (valid[lane] == ".")
            continue;
        const float storedValue = readElement<float>(capture, stored, lane);
        const float loadedValue = readElement<float>(capture, loaded, lane);
        if (std::memcmp(&storedValue, &loadedValue, sizeof(float)) != 0)
            throw std::logic_error("LowRank formatter workspace store/load mismatch");
    }
}

std::vector<std::string> outputByteValues(
    const LowRankIllustrationCaptureLog & capture,
    const EventIndex & index,
    const LowRankIllustrationConfiguration & configuration,
    const std::size_t row,
    const std::size_t group,
    const std::uint32_t channel,
    const std::vector<std::string> & storeMask
) {
    std::vector<std::string> values(configuration.logicalOutputs, ".");
    if (const auto * vector = index.optionalOutput(LowRankIllustrationEvent::OutputByteVector, row, group, channel)) {
        for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane)
            values[lane] = hexadecimal(readElement<std::uint8_t>(capture, *vector, lane));
        return values;
    }
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
        if (storeMask[lane] == ".")
            continue;
        const auto & stored = index.storedOutput(row, group, lane, channel);
        values[lane] = hexadecimal(readElement<std::uint8_t>(capture, stored, 0));
    }
    return values;
}

std::vector<std::string> packedOutputValues(
    const LowRankIllustrationCaptureLog & capture,
    const EventIndex & index,
    const LowRankIllustrationConfiguration & configuration,
    const std::size_t row,
    const std::size_t group,
    const std::vector<std::string> & storeMask
) {
    std::vector<std::string> values;
    values.reserve(static_cast<std::size_t>(configuration.logicalOutputs) * 3U);
    if (const auto * packed = index.optionalOutput(LowRankIllustrationEvent::PackedOutput, row, group)) {
        for (std::size_t byte = 0; byte < packed->elementCount; ++byte)
            values.push_back(hexadecimal(readElement<std::uint8_t>(capture, *packed, byte)));
        return values;
    }
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
        for (unsigned channel = 0; channel < 3U; ++channel) {
            if (storeMask[lane] == ".") {
                values.emplace_back(".");
                continue;
            }
            const auto & stored = index.storedOutput(row, group, lane, channel);
            values.push_back(hexadecimal(readElement<std::uint8_t>(capture, stored, 0)));
        }
    }
    return values;
}

std::vector<std::string> tapLabels(const unsigned count) {
    std::vector<std::string> values(count);
    for (unsigned tap = 0; tap < count; ++tap)
        values[tap] = decimal(tap);
    return values;
}

void renderOutput(
    std::string & trace,
    const LowRankIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const LowRankIllustrationCaptureLog & capture,
    const EventIndex & index,
    const GroupKey groupKey
) {
    const std::size_t row = groupKey.first;
    const std::size_t group = groupKey.second;
    const std::size_t selectedLane = selection.column - group;
    const LowRankIllustrationCaptureEvent & store = index.output(LowRankIllustrationEvent::StoreMask, row, group);
    const std::vector<std::string> valid = maskValues(capture, store);

    Block header;
    header.heading = "group y=" + decimal(row) + " x=" + decimal(group) + " rank=" + decimal(configuration.rank);
    header.rows = {
        {"outputX", outputColumns(configuration, group)},
        {"target", selectedOutput(configuration, selection, row, group)},
        {"inOutputDomain", valid},
    };
    appendBlock(trace, header);

    for (unsigned rank = 0; rank < configuration.rank; ++rank) {
        Block factors;
        factors.heading = "factors rank=" + decimal(rank);
        factors.rows.push_back({"hIndex", tapLabels(configuration.kernelWidth)});
        Row horizontal{"H", {}};
        for (unsigned tap = 0; tap < configuration.kernelWidth; ++tap) {
            const auto & factor = index.horizontal(LowRankIllustrationEvent::HorizontalFactor, selection.row, group, rank, tap);
            horizontal.values.push_back(floating(readElement<float>(capture, factor, 0)));
        }
        factors.rows.push_back(std::move(horizontal));
        factors.rows.push_back({"vIndex", tapLabels(configuration.kernelHeight)});
        Row vertical{"V", {}};
        for (unsigned tap = 0; tap < configuration.kernelHeight; ++tap) {
            const auto & factor = index.vertical(LowRankIllustrationEvent::VerticalFactor, row, group, rank, tap);
            vertical.values.push_back(floating(readElement<float>(capture, factor, 0)));
        }
        factors.rows.push_back(std::move(vertical));
        appendBlock(trace, factors, true);
    }

    Block horizontal;
    horizontal.heading = "horizontal row=" + decimal(selection.row);
    horizontal.rows.push_back({"outputX", outputColumns(configuration, group)});
    for (unsigned rank = 0; rank < configuration.rank; ++rank) {
        for (unsigned channel = 0; channel < 3U; ++channel) {
            const auto & event = index.horizontalWorkspace(selection.row, group, rank, channel);
            horizontal.rows.push_back({"Q" + decimal(rank) + "." + std::string(1U, ChannelNames[channel]), floatVector(capture, event, &valid)});
        }
    }
    appendBlock(trace, horizontal, true);

    const unsigned horizontalRadius = configuration.kernelWidth / 2U;
    for (unsigned rank = 0; rank < configuration.rank; ++rank) {
        Block detail;
        detail.heading = "horizontal rank=" + decimal(rank) + " row=" + decimal(selection.row) + " x=" + decimal(selection.column);
        detail.rows.push_back({"hIndex", tapLabels(configuration.kernelWidth)});
        Row source{"sourceX", {}};
        Row sourceValid{"inInputDomain", {}};
        Row factor{"H", {}};
        std::array<Row, 3> input{{{"inputR", {}}, {"inputG", {}}, {"inputB", {}}}};
        std::array<Row, 3> accumulator{{{"partialSumR", {}}, {"partialSumG", {}}, {"partialSumB", {}}}};
        for (unsigned tap = 0; tap < configuration.kernelWidth; ++tap) {
            const std::int64_t coordinate =
                static_cast<std::int64_t>(selection.column) + static_cast<std::int64_t>(tap) - static_cast<std::int64_t>(horizontalRadius);
            const bool isValid = coordinate >= 0 && coordinate < static_cast<std::int64_t>(configuration.imageWidth);
            source.values.push_back(isValid ? decimal(static_cast<std::size_t>(coordinate)) : ".");
            sourceValid.values.emplace_back(isValid ? "1" : ".");
            const auto & factorEvent = index.horizontal(LowRankIllustrationEvent::HorizontalFactor, selection.row, group, rank, tap);
            factor.values.push_back(floating(readElement<float>(capture, factorEvent, 0)));
            for (unsigned channel = 0; channel < 3U; ++channel) {
                const auto & sample = index.horizontal(LowRankIllustrationEvent::HorizontalSample, selection.row, group, rank, tap, channel);
                const auto & sum = index.horizontal(LowRankIllustrationEvent::HorizontalAccumulator, selection.row, group, rank, tap, channel);
                input[channel].values.push_back(floating(readElement<float>(capture, sample, selectedLane)));
                accumulator[channel].values.push_back(floating(readElement<float>(capture, sum, selectedLane)));
            }
        }
        detail.rows.push_back(std::move(source));
        detail.rows.push_back(std::move(sourceValid));
        detail.rows.push_back(std::move(factor));
        for (Row & value : input)
            detail.rows.push_back(std::move(value));
        for (Row & value : accumulator)
            detail.rows.push_back(std::move(value));
        appendBlock(trace, detail, true);
    }

    Block vertical;
    vertical.heading = "vertical y=" + decimal(row) + " x=" + decimal(selection.column);
    Row ranks{"rank", {}};
    Row verticalIndices{"vIndex", {}};
    Row workspaceY{"Q.y", {}};
    Row workspaceValid{"inQDomain", {}};
    Row factor{"V", {}};
    std::array<Row, 3> workspaceValues{{{"Q.R", {}}, {"Q.G", {}}, {"Q.B", {}}}};
    std::array<Row, 3> accumulators{{{"partialSumR", {}}, {"partialSumG", {}}, {"partialSumB", {}}}};
    const unsigned verticalRadius = configuration.kernelHeight / 2U;
    for (unsigned rank = 0; rank < configuration.rank; ++rank) {
        for (unsigned tap = 0; tap < configuration.kernelHeight; ++tap) {
            ranks.values.push_back(decimal(rank));
            verticalIndices.values.push_back(decimal(tap));
            const std::int64_t coordinate =
                static_cast<std::int64_t>(row) + static_cast<std::int64_t>(tap) - static_cast<std::int64_t>(verticalRadius);
            const bool isValid = coordinate >= 0 && coordinate < static_cast<std::int64_t>(configuration.imageHeight);
            if (isValid)
                workspaceY.values.push_back(decimal(static_cast<std::size_t>(coordinate)));
            else
                workspaceY.values.emplace_back(".");
            workspaceValid.values.emplace_back(isValid ? "1" : ".");
            const auto & factorEvent = index.vertical(LowRankIllustrationEvent::VerticalFactor, row, group, rank, tap);
            factor.values.push_back(floating(readElement<float>(capture, factorEvent, 0)));
            for (unsigned channel = 0; channel < 3U; ++channel) {
                const auto & sample = index.vertical(LowRankIllustrationEvent::VerticalWorkspaceLoad, row, group, rank, tap, channel);
                const auto & sum = index.vertical(LowRankIllustrationEvent::VerticalAccumulator, row, group, rank, tap, channel);
                if (isValid && static_cast<std::size_t>(coordinate) == selection.row) {
                    const auto & stored = index.horizontalWorkspace(selection.row, group, rank, channel);
                    validateWorkspaceValues(capture, stored, sample, valid);
                }
                workspaceValues[channel].values.push_back(floating(readElement<float>(capture, sample, selectedLane)));
                accumulators[channel].values.push_back(floating(readElement<float>(capture, sum, selectedLane)));
            }
        }
    }
    vertical.rows.push_back(std::move(ranks));
    vertical.rows.push_back(std::move(verticalIndices));
    vertical.rows.push_back(std::move(workspaceY));
    vertical.rows.push_back(std::move(workspaceValid));
    vertical.rows.push_back(std::move(factor));
    for (Row & value : workspaceValues)
        vertical.rows.push_back(std::move(value));
    for (Row & value : accumulators)
        vertical.rows.push_back(std::move(value));
    appendBlock(trace, vertical, true);

    Block output;
    output.heading = "output group y=" + decimal(row) + " x=" + decimal(group);
    output.rows.push_back({"outputX", outputColumns(configuration, group)});
    output.rows.push_back({"target", selectedOutput(configuration, selection, row, group)});
    output.rows.push_back({"inOutputDomain", valid});
    for (unsigned channel = 0; channel < 3U; ++channel) {
        const auto & finalAccumulator = index.vertical(
            LowRankIllustrationEvent::VerticalAccumulator, row, group, configuration.rank - 1U, configuration.kernelHeight - 1U, channel
        );
        output.rows.push_back({"result" + std::string(1U, ChannelNames[channel]), floatVector(capture, finalAccumulator)});
    }
    for (unsigned channel = 0; channel < 3U; ++channel) {
        output.rows.push_back(
            {"output" + std::string(1U, ChannelNames[channel]), outputByteValues(capture, index, configuration, row, group, channel, valid)}
        );
    }
    output.rows.push_back({"outputRGB.hex", packedOutputValues(capture, index, configuration, row, group, valid)});
    output.rows.push_back({"storeMask", valid});
    appendBlock(trace, output, true);
}

void renderInputGroup(
    std::string & trace,
    const LowRankIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const LowRankIllustrationCaptureLog & capture,
    const EventIndex & index,
    const GroupKey groupKey,
    const bool blankBefore
) {
    const std::size_t outputRow = groupKey.first;
    const std::size_t group = groupKey.second;
    std::vector<bool> selected(configuration.logicalOutputs, false);
    const unsigned horizontalRadius = configuration.kernelWidth / 2U;
    const unsigned verticalRadius = configuration.kernelHeight / 2U;
    const bool rowSelected = outputRow <= static_cast<std::size_t>(selection.row) + verticalRadius && selection.row <= outputRow + verticalRadius;
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
        const std::size_t outputColumn = group + lane;
        selected[lane] = rowSelected && outputColumn < configuration.imageWidth
                         && outputColumn <= static_cast<std::size_t>(selection.column) + horizontalRadius
                         && selection.column <= outputColumn + horizontalRadius;
    }
    if (std::none_of(selected.begin(), selected.end(), [](const bool value) { return value; }))
        throw std::logic_error("LowRank formatter captured an unaffected input group");
    const auto & store = index.output(LowRankIllustrationEvent::StoreMask, outputRow, group);
    const std::vector<std::string> valid = maskValues(capture, store);

    Block block;
    block.heading = "group y=" + decimal(outputRow) + " x=" + decimal(group);
    block.rows.push_back({"outputX", outputColumns(configuration, group)});
    Row selectedRow{"inWindow", {}};
    Row inputY{"inputY", {}};
    Row inputX{"inputX", {}};
    Row packedHex{"inputRGB.hex", {}};
    Row packedDec{"inputRGB.dec", {}};
    std::array<Row, 3> inputValues{{{"inputR", {}}, {"inputG", {}}, {"inputB", {}}}};
    for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
        if (!selected[lane]) {
            selectedRow.values.emplace_back(".");
            inputY.values.emplace_back(".");
            inputX.values.emplace_back(".");
            packedHex.values.emplace_back(".");
            packedDec.values.emplace_back(".");
            for (Row & row : inputValues)
                row.values.emplace_back(".");
            continue;
        }
        selectedRow.values.emplace_back("1");
        inputY.values.push_back(decimal(selection.row));
        inputX.values.push_back(decimal(selection.column));
        const std::size_t outputColumn = group + lane;
        const std::size_t tap = static_cast<std::size_t>(selection.column) + horizontalRadius - outputColumn;
        std::array<std::uint8_t, 3> bytes{};
        if (const auto * packed = index.packedInput(selection.row, group, 0U, tap)) {
            for (unsigned channel = 0; channel < 3U; ++channel)
                bytes[channel] = readElement<std::uint8_t>(capture, *packed, lane * 3U + channel);
        } else {
            for (unsigned channel = 0; channel < 3U; ++channel) {
                const auto & byte = index.inputByte(selection.row, group, 0U, tap, lane, selection.column, channel);
                bytes[channel] = readElement<std::uint8_t>(capture, byte, 0);
            }
        }
        std::string packedHexCell;
        std::string packedDecCell;
        for (unsigned channel = 0; channel < 3U; ++channel) {
            if (channel != 0U) {
                packedHexCell.push_back(' ');
                packedDecCell.push_back(' ');
            }
            packedHexCell.append(hexadecimal(bytes[channel]));
            packedDecCell.append(decimal(static_cast<unsigned>(bytes[channel])));
            const auto & sample = index.horizontal(LowRankIllustrationEvent::HorizontalSample, selection.row, group, 0U, tap, channel);
            inputValues[channel].values.push_back(floating(readElement<float>(capture, sample, lane)));
        }
        packedHex.values.push_back(std::move(packedHexCell));
        packedDec.values.push_back(std::move(packedDecCell));
    }
    block.rows.push_back(std::move(selectedRow));
    block.rows.push_back({"inOutputDomain", valid});
    block.rows.push_back(std::move(inputY));
    block.rows.push_back(std::move(inputX));
    block.rows.push_back(std::move(packedHex));
    block.rows.push_back(std::move(packedDec));
    for (Row & row : inputValues)
        block.rows.push_back(std::move(row));

    const std::size_t verticalTap = static_cast<std::size_t>(selection.row) + verticalRadius - outputRow;
    for (unsigned rank = 0; rank < configuration.rank; ++rank) {
        Row horizontalTap{"r" + decimal(rank) + ".hIndex", {}};
        Row horizontalFactor{"r" + decimal(rank) + ".H", {}};
        Row workspaceY{"Q" + decimal(rank) + ".y", {}};
        std::array<Row, 3> workspaceValues{{
            {"Q" + decimal(rank) + ".R", {}},
            {"Q" + decimal(rank) + ".G", {}},
            {"Q" + decimal(rank) + ".B", {}},
        }};
        Row verticalTapRow{"r" + decimal(rank) + ".vIndex", {}};
        Row verticalFactor{"r" + decimal(rank) + ".V", {}};
        for (unsigned channel = 0; channel < 3U; ++channel) {
            validateWorkspaceValues(
                capture,
                index.horizontalWorkspace(selection.row, group, rank, channel),
                index.vertical(LowRankIllustrationEvent::VerticalWorkspaceLoad, outputRow, group, rank, verticalTap, channel),
                valid
            );
        }
        for (unsigned lane = 0; lane < configuration.logicalOutputs; ++lane) {
            if (!selected[lane]) {
                horizontalTap.values.emplace_back(".");
                horizontalFactor.values.emplace_back(".");
                workspaceY.values.emplace_back(".");
                for (Row & row : workspaceValues)
                    row.values.emplace_back(".");
                verticalTapRow.values.emplace_back(".");
                verticalFactor.values.emplace_back(".");
                continue;
            }
            const std::size_t outputColumn = group + lane;
            const std::size_t tap = static_cast<std::size_t>(selection.column) + horizontalRadius - outputColumn;
            horizontalTap.values.push_back(decimal(tap));
            const auto & hFactor = index.horizontal(LowRankIllustrationEvent::HorizontalFactor, selection.row, group, rank, tap);
            horizontalFactor.values.push_back(floating(readElement<float>(capture, hFactor, 0)));
            workspaceY.values.push_back(decimal(selection.row));
            for (unsigned channel = 0; channel < 3U; ++channel) {
                const auto & workspace = index.horizontalWorkspace(selection.row, group, rank, channel);
                const float storedValue = readElement<float>(capture, workspace, lane);
                workspaceValues[channel].values.push_back(floating(storedValue));
            }
            verticalTapRow.values.push_back(decimal(verticalTap));
            const auto & vFactor = index.vertical(LowRankIllustrationEvent::VerticalFactor, outputRow, group, rank, verticalTap);
            verticalFactor.values.push_back(floating(readElement<float>(capture, vFactor, 0)));
        }
        block.rows.push_back(std::move(horizontalTap));
        block.rows.push_back(std::move(horizontalFactor));
        block.rows.push_back(std::move(workspaceY));
        for (Row & row : workspaceValues)
            block.rows.push_back(std::move(row));
        block.rows.push_back(std::move(verticalTapRow));
        block.rows.push_back(std::move(verticalFactor));
    }

    for (unsigned channel = 0; channel < 3U; ++channel) {
        block.rows.push_back(
            {"output" + std::string(1U, ChannelNames[channel]), outputByteValues(capture, index, configuration, outputRow, group, channel, valid)}
        );
    }
    block.rows.push_back({"outputRGB.hex", packedOutputValues(capture, index, configuration, outputRow, group, valid)});
    appendBlock(trace, block, blankBefore);
}

}  // namespace

LowRankIllustrationElementType lowRankIllustrationElementType(const LowRankIllustrationEvent event) {
    return eventAbi(event).elementType;
}

std::size_t lowRankIllustrationElementByteWidth(const LowRankIllustrationElementType type) {
    switch (type) {
    case LowRankIllustrationElementType::UInt8:
        return sizeof(std::uint8_t);
    case LowRankIllustrationElementType::Float32:
        return sizeof(float);
    }
    throw std::invalid_argument("LowRank illustration element type is invalid");
}

extern "C" void captureLowRankConvFilterValue(
    std::uint8_t * context,
    const std::uint32_t eventValue,
    const std::uint32_t pathValue,
    const std::uint32_t elementTypeValue,
    const std::uint32_t channel,
    const std::size_t elementCount,
    const std::size_t elementByteWidth,
    const std::uint8_t * bytes,
    const std::size_t row,
    const std::size_t groupStart,
    const std::size_t rank,
    const std::size_t horizontalTap,
    const std::size_t verticalTap,
    const std::size_t paddedSourceRow,
    const std::size_t sourceColumn,
    const std::size_t lane
) noexcept {
    if (context == nullptr)
        return;
    auto & capture = *reinterpret_cast<LowRankIllustrationCaptureLog *>(context);
    if (capture.failure)
        return;
    try {
        const auto event = static_cast<LowRankIllustrationEvent>(eventValue);
        const auto path = static_cast<LowRankIllustrationPath>(pathValue);
        const auto elementType = static_cast<LowRankIllustrationElementType>(elementTypeValue);
        if (static_cast<std::uint32_t>(path) > static_cast<std::uint32_t>(LowRankIllustrationPath::OutputChecked))
            throw std::invalid_argument("LowRank illustration path is invalid");
        const EventAbi & abi = eventAbi(event);
        validateEventPath(event, path);
        if (elementType != abi.elementType || elementByteWidth != lowRankIllustrationElementByteWidth(elementType))
            throw std::logic_error("LowRank illustration element type mismatch");
        const std::size_t expectedCount = abi.fixedElementCount == 0U ? abi.logicalOutputMultiplier * capture.logicalOutputs : abi.fixedElementCount;
        if (elementCount != expectedCount)
            throw std::logic_error("LowRank illustration element count mismatch");
        if (elementCount != 0U && elementByteWidth > std::numeric_limits<std::size_t>::max() / elementCount)
            throw std::overflow_error("LowRank illustration capture size overflow");
        const std::size_t byteCount = elementCount * elementByteWidth;
        const std::size_t byteOffset = capture.bytes.size();
        if (byteCount > std::numeric_limits<std::size_t>::max() - byteOffset)
            throw std::overflow_error("LowRank illustration capture storage overflow");
        if (byteCount != 0U && bytes == nullptr)
            throw std::invalid_argument("LowRank illustration capture bytes are null");

        capture.bytes.resize(byteOffset + byteCount);
        std::memcpy(capture.bytes.data() + byteOffset, bytes, byteCount);
        capture.events.push_back(
            {event,
             path,
             channel,
             elementCount,
             byteCount,
             byteOffset,
             row,
             groupStart,
             rank,
             horizontalTap,
             verticalTap,
             paddedSourceRow,
             sourceColumn,
             lane}
        );
    } catch (...) {
        capture.failure = std::current_exception();
    }
}

void sortLowRankIllustrationEvents(LowRankIllustrationCaptureLog & capture) {
    std::sort(capture.events.begin(), capture.events.end(), [](const auto & left, const auto & right) {
        return semanticKey(left) < semanticKey(right);
    });
    for (std::size_t index = 1; index < capture.events.size(); ++index) {
        if (semanticKey(capture.events[index - 1U]) == semanticKey(capture.events[index]))
            throw std::logic_error("LowRank illustration semantic event is duplicated");
    }
}

std::string formatLowRankConvFilterIllustration(
    const LowRankIllustrationConfiguration & configuration,
    const ConvFilterIllustrationSelection selection,
    const LowRankIllustrationCaptureLog & capture
) {
    if (capture.failure)
        std::rethrow_exception(capture.failure);
    if (capture.logicalOutputs != configuration.logicalOutputs)
        throw std::logic_error("LowRank formatter logical output count mismatch");
    EventIndex index(configuration, capture);
    if (index.outputGroups().empty())
        throw std::logic_error("LowRank formatter has no output operation");

    std::string trace;
    switch (selection.kind) {
    case ConvFilterIllustrationSelectionKind::Output:
        {
            if (index.outputGroups().size() != 1U)
                throw std::logic_error("LowRank output selection captured multiple output operations");
            renderOutput(trace, configuration, selection, capture, index, *index.outputGroups().begin());
            break;
        }
    case ConvFilterIllustrationSelectionKind::Input:
        {
            bool blankBefore = false;
            for (const GroupKey group : index.outputGroups()) {
                renderInputGroup(trace, configuration, selection, capture, index, group, blankBefore);
                blankBefore = true;
            }
            break;
        }
    default:
        throw std::invalid_argument("LowRank formatter selection kind is invalid");
    }
    return trace;
}

}  // namespace kernel::image::internal
