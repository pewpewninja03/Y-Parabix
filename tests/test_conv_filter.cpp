#include <image/conv_filter.h>

#include <testing/testing.h>
#include <toolchain/toolchain.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace testing;

namespace {

using kernel::image::CompiledConvFilter;
using kernel::image::CompiledConvFilterIllustration;
using kernel::image::ConvFilterIllustrationSelection;
using kernel::image::ConvFilterIllustrationSelectionKind;

constexpr unsigned ImageWidth = 7;
constexpr unsigned ImageHeight = 5;
constexpr unsigned ChannelCount = 3;
constexpr std::size_t ImageByteCount = static_cast<std::size_t>(ImageWidth) * ImageHeight * ChannelCount;
using Image = std::array<std::uint8_t, ImageByteCount>;

constexpr std::size_t byteIndex(const unsigned row, const unsigned column, const unsigned channel) {
    return (static_cast<std::size_t>(row) * ImageWidth + column) * ChannelCount + channel;
}

std::uint8_t zeroExtendedInput(const Image & input, const int row, const int column, const unsigned channel) {
    if (row < 0 || row >= static_cast<int>(ImageHeight) || column < 0 || column >= static_cast<int>(ImageWidth))
        return 0;
    return input[byteIndex(static_cast<unsigned>(row), static_cast<unsigned>(column), channel)];
}

constexpr std::size_t workspaceIndex(const unsigned rank, const unsigned row, const unsigned column, const unsigned channel) {
    return ((static_cast<std::size_t>(rank) * ImageHeight + row) * ImageWidth + column) * ChannelCount + channel;
}

class AlignedWorkspace {
   public:
    AlignedWorkspace(const std::size_t size, const std::size_t alignment) {
        if (size == 0)
            return;
        storage.resize(size + alignment - 1U);
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(storage.data());
        alignedPointer = reinterpret_cast<void *>((address + alignment - 1U) & ~(alignment - 1U));
    }

    void * data() noexcept {
        return alignedPointer;
    }

   private:
    std::vector<std::uint8_t> storage;
    void * alignedPointer = nullptr;
};

Image rampInput() {
    Image input{};
    for (unsigned row = 0; row < ImageHeight; ++row) {
        for (unsigned column = 0; column < ImageWidth; ++column) {
            input[byteIndex(row, column, 0)] = static_cast<std::uint8_t>(10U + 7U * row + 5U * column);
            input[byteIndex(row, column, 1)] = static_cast<std::uint8_t>(80U + 11U * row + 3U * column);
            input[byteIndex(row, column, 2)] = static_cast<std::uint8_t>(150U + 5U * row + 9U * column);
        }
    }
    return input;
}

Image mixedInput() {
    Image input{};
    for (std::size_t index = 0; index < ImageByteCount; ++index)
        input[index] = static_cast<std::uint8_t>((index * 73U + index / ChannelCount * 19U + 17U) & 255U);
    return input;
}

std::uint8_t outputByte(const float value) {
    return static_cast<std::uint8_t>(std::clamp(value, 0.0F, 255.0F) + 0.5F);
}

Image defaultReference(const Image & input, const std::array<float, 9> & weights) {
    Image output{};
    for (unsigned row = 0; row < ImageHeight; ++row) {
        for (unsigned column = 0; column < ImageWidth; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                float sum = 0.0F;
                for (unsigned kernelRow = 0; kernelRow < 3U; ++kernelRow) {
                    for (unsigned kernelColumn = 0; kernelColumn < 3U; ++kernelColumn) {
                        const float weight = weights[kernelRow * 3U + kernelColumn];
                        if (weight == 0.0F)
                            continue;
                        const int sourceRow = static_cast<int>(row) + static_cast<int>(kernelRow) - 1;
                        const int sourceColumn = static_cast<int>(column) + static_cast<int>(kernelColumn) - 1;
                        const float sample = static_cast<float>(zeroExtendedInput(input, sourceRow, sourceColumn, channel));
                        sum = std::fma(sample, weight, sum);
                    }
                }
                output[byteIndex(row, column, channel)] = outputByte(sum);
            }
        }
    }
    return output;
}

Image uniformReference(const Image & input, const float weight) {
    Image output{};
    for (unsigned row = 0; row < ImageHeight; ++row) {
        for (unsigned column = 0; column < ImageWidth; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                std::int32_t sum = 0;
                for (int rowOffset = -1; rowOffset <= 1; ++rowOffset) {
                    for (int columnOffset = -1; columnOffset <= 1; ++columnOffset) {
                        const int sourceRow = static_cast<int>(row) + rowOffset;
                        const int sourceColumn = static_cast<int>(column) + columnOffset;
                        sum += zeroExtendedInput(input, sourceRow, sourceColumn, channel);
                    }
                }
                output[byteIndex(row, column, channel)] = outputByte(static_cast<float>(sum) * weight);
            }
        }
    }
    return output;
}

Image lowRankReference(const Image & input, const std::array<float, 6> & horizontalFactors, const std::array<float, 6> & verticalFactors) {
    constexpr unsigned Rank = 2;
    std::array<float, Rank * ImageWidth * ImageHeight * ChannelCount> workspace{};
    for (unsigned row = 0; row < ImageHeight; ++row) {
        for (unsigned column = 0; column < ImageWidth; ++column) {
            for (unsigned rank = 0; rank < Rank; ++rank) {
                for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                    float sum = 0.0F;
                    for (unsigned tap = 0; tap < 3U; ++tap) {
                        const int sourceColumn = static_cast<int>(column) + static_cast<int>(tap) - 1;
                        const float sample = static_cast<float>(zeroExtendedInput(input, static_cast<int>(row), sourceColumn, channel));
                        sum = std::fma(sample, horizontalFactors[rank * 3U + tap], sum);
                    }
                    workspace[workspaceIndex(rank, row, column, channel)] = sum;
                }
            }
        }
    }

    Image output{};
    for (unsigned row = 0; row < ImageHeight; ++row) {
        for (unsigned column = 0; column < ImageWidth; ++column) {
            for (unsigned channel = 0; channel < ChannelCount; ++channel) {
                float sum = 0.0F;
                for (unsigned rank = 0; rank < Rank; ++rank) {
                    for (unsigned tap = 0; tap < 3U; ++tap) {
                        const int sourceRow = static_cast<int>(row) + static_cast<int>(tap) - 1;
                        const float sample = sourceRow < 0 || sourceRow >= static_cast<int>(ImageHeight)
                                                 ? 0.0F
                                                 : workspace[workspaceIndex(rank, static_cast<unsigned>(sourceRow), column, channel)];
                        sum = std::fma(sample, verticalFactors[rank * 3U + tap], sum);
                    }
                }
                output[byteIndex(row, column, channel)] = outputByte(sum);
            }
        }
    }
    return output;
}

bool compareOutput(const char * caseName, const Image & actual, const Image & expected) {
    static constexpr char ChannelNames[] = {'R', 'G', 'B'};
    for (std::size_t index = 0; index < ImageByteCount; ++index) {
        if (actual[index] == expected[index])
            continue;
        const std::size_t pixel = index / ChannelCount;
        std::cerr << caseName << ": y=" << pixel / ImageWidth << " x=" << pixel % ImageWidth << " channel=" << ChannelNames[index % ChannelCount]
                  << " expected=" << static_cast<unsigned>(expected[index]) << " actual=" << static_cast<unsigned>(actual[index]) << '\n';
        return false;
    }
    return true;
}

bool checkNormal(const char * caseName, const CompiledConvFilter & filter, const Image & input, const Image & expected) {
    AlignedWorkspace workspace(filter.workspaceSize(), filter.workspaceAlignment());
    Image output{};
    if (!filter.apply(input.data(), output.data(), workspace.data())) {
        std::cerr << caseName << ": apply returned false\n";
        return false;
    }
    return compareOutput(caseName, output, expected);
}

bool checkIllustrated(const char * caseName, const CompiledConvFilterIllustration & filter, const Image & input, const Image & expected) {
    AlignedWorkspace workspace(filter.workspaceSize(), filter.workspaceAlignment());
    Image output{};
    std::string ignoredTrace;
    const ConvFilterIllustrationSelection selection{ConvFilterIllustrationSelectionKind::Output, 2U, 3U};
    if (!filter.apply(input.data(), output.data(), workspace.data(), selection, ignoredTrace)) {
        std::cerr << caseName << ": apply returned false\n";
        return false;
    }
    return compareOutput(caseName, output, expected);
}

bool testDefault() {
    const std::array<float, 9> weights = {0.125F, -0.0625F, 0.0F, 0.25F, 0.5F, -0.125F, 0.0F, 0.0625F, -0.03125F};
    const kernel::image::DefaultConvFilter configuration{3U, 3U, {weights.data(), weights.size()}};
    const auto normal = kernel::image::compileConvFilter(ImageWidth, ImageHeight, configuration);
    const Image zero{};
    const auto ramp = rampInput();
    const auto mixed = mixedInput();
    const auto rampExpected = defaultReference(ramp, weights);
    const auto mixedExpected = defaultReference(mixed, weights);
    if (!checkNormal("Default zero", *normal, zero, zero) || !checkNormal("Default ramp", *normal, ramp, rampExpected)
        || !checkNormal("Default mixed", *normal, mixed, mixedExpected))
    {
        return false;
    }
    const auto illustrated = kernel::image::compileConvFilterIllustration(ImageWidth, ImageHeight, configuration);
    return checkIllustrated("Default illustrated", *illustrated, mixed, mixedExpected);
}

bool testUniform() {
    constexpr float Weight = 0.125F;
    const kernel::image::UniformConvFilter configuration{3U, 3U, Weight};
    const auto normal = kernel::image::compileConvFilter(ImageWidth, ImageHeight, configuration);
    const Image zero{};
    const auto ramp = rampInput();
    const auto mixed = mixedInput();
    const auto rampExpected = uniformReference(ramp, Weight);
    const auto mixedExpected = uniformReference(mixed, Weight);
    if (!checkNormal("Uniform zero", *normal, zero, zero) || !checkNormal("Uniform ramp", *normal, ramp, rampExpected)
        || !checkNormal("Uniform mixed", *normal, mixed, mixedExpected))
    {
        return false;
    }
    const auto illustrated = kernel::image::compileConvFilterIllustration(ImageWidth, ImageHeight, configuration);
    return checkIllustrated("Uniform illustrated", *illustrated, mixed, mixedExpected);
}

bool testLowRank() {
    constexpr unsigned Rank = 2;
    const std::array<float, 6> horizontalFactors = {0.25F, 0.0F, 0.5F, -0.125F, 0.25F, 0.0625F};
    const std::array<float, 6> verticalFactors = {0.25F, 0.0F, 0.125F, 0.0F, -0.25F, 0.25F};
    const kernel::image::LowRankConvFilter configuration{
        3U,
        3U,
        Rank,
        {horizontalFactors.data(), horizontalFactors.size()},
        {verticalFactors.data(), verticalFactors.size()},
    };
    const auto normal = kernel::image::compileConvFilter(ImageWidth, ImageHeight, configuration);
    const Image zero{};
    const auto ramp = rampInput();
    const auto mixed = mixedInput();
    const auto rampExpected = lowRankReference(ramp, horizontalFactors, verticalFactors);
    const auto mixedExpected = lowRankReference(mixed, horizontalFactors, verticalFactors);
    if (!checkNormal("LowRank zero", *normal, zero, zero) || !checkNormal("LowRank ramp", *normal, ramp, rampExpected)
        || !checkNormal("LowRank mixed", *normal, mixed, mixedExpected))
    {
        return false;
    }
    const auto illustrated = kernel::image::compileConvFilterIllustration(ImageWidth, ImageHeight, configuration);
    return checkIllustrated("LowRank illustrated", *illustrated, mixed, mixedExpected);
}

bool testFrequency() {
    const std::array<float, 9> weights = {-0.0625F, -0.125F, 0.125F, 0.03125F, 0.1875F, 0.09375F, 0.0625F, -0.15625F, -0.0625F};
    const kernel::image::FrequencyConvFilter configuration{3U, 3U, {weights.data(), weights.size()}};
    const auto filter = kernel::image::compileConvFilter(ImageWidth, ImageHeight, configuration);
    const Image zero{};
    if (!checkNormal("Frequency zero", *filter, zero, zero))
        return false;

    Image input{};
    std::uint32_t state = 0x4b1d2397U;
    for (std::size_t index = 0; index < ImageByteCount; ++index) {
        state = state * 1664525U + 1013904223U;
        input[index] = static_cast<std::uint8_t>((state >> 24U) ^ (index * 29U));
    }
    static constexpr Image Expected = {
        26, 8,  0,  46, 21, 0,  22, 52, 25, 17, 44, 21, 32, 21, 21, 20, 0,  3,  26, 0,  17, 0,  0,  8,  0,  0,  28, 0,  12, 23, 23, 0,  14, 6,  0,
        0,  10, 13, 0,  0,  26, 0,  13, 6,  28, 43, 32, 38, 52, 28, 10, 44, 40, 0,  0,  9,  16, 0,  29, 6,  0,  0,  25, 0,  19, 28, 3,  0,  15, 7,
        0,  0,  0,  3,  0,  19, 23, 26, 22, 42, 7,  16, 0,  0,  16, 41, 20, 35, 36, 44, 24, 23, 62, 65, 64, 12, 22, 50, 55, 52, 0,  48, 0,  0,  23,
    };
    return checkNormal("Frequency mixed", *filter, input, Expected);
}

}  // namespace

namespace {

int32_t runNamedTest(const char * name, bool (*test)()) {
    try {
        return test() ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << name << ": " << error.what() << '\n';
        return 1;
    }
}

}  // namespace

int32_t invoke_Default() {
    return runNamedTest("Default", testDefault);
}

int32_t invoke_Uniform() {
    return runNamedTest("Uniform", testUniform);
}

int32_t invoke_LowRank() {
    return runNamedTest("LowRank", testLowRank);
}

int32_t invoke_Frequency() {
    return runNamedTest("Frequency", testFrequency);
}

int main(int argc, char ** argv) {
    codegen::ParseCommandLineOptions(argc, argv, {&codegen::JIT_InfoOptions, testing::cli::testFlags()});
    return testing::RunTestSuite({
        CASE(Default),
        CASE(Uniform),
        CASE(LowRank),
        CASE(Frequency),
    });
}
