#include <image/conv_filter.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using kernel::image::CompiledConvFilter;

struct KernelCase {
    const char * caseName;
    unsigned imageWidth;
    unsigned imageHeight;
    unsigned kernelExtent;
    std::uint64_t expectedHash;
};

class AlignedWorkspace {
   public:
    explicit AlignedWorkspace(const CompiledConvFilter & filter) {
        if (filter.workspaceSize() == 0)
            return;
        storage.resize(filter.workspaceSize() + filter.workspaceAlignment() - 1U);
        const auto baseAddress = reinterpret_cast<std::uintptr_t>(storage.data());
        alignedPointer = reinterpret_cast<void *>((baseAddress + filter.workspaceAlignment() - 1U) & ~(filter.workspaceAlignment() - 1U));
    }

    void * data() noexcept {
        return alignedPointer;
    }

   private:
    std::vector<std::uint8_t> storage;
    void * alignedPointer = nullptr;
};

std::uint64_t hashBytes(const std::vector<std::uint8_t> & values) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::uint8_t value : values) {
        hash = (hash ^ value) * 1099511628211ULL;
    }
    return hash;
}

std::vector<std::uint8_t> makeDefaultInput(const unsigned imageWidth, const unsigned imageHeight) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(imageWidth) * imageHeight * 3U);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index * 37U + index / 7U * 19U + 13U) & 255U);
    }
    return input;
}

std::vector<float> makeDefaultWeights(const unsigned kernelExtent) {
    std::vector<float> weights(static_cast<std::size_t>(kernelExtent) * kernelExtent);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        weights[index] = static_cast<float>(static_cast<int>((index * 7U) % 13U) - 6) / 17.0F;
    }
    return weights;
}

std::vector<std::uint8_t> makeUniformInput(const unsigned imageWidth, const unsigned imageHeight) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(imageWidth) * imageHeight * 3U);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index * 29U + index / 11U * 17U + 31U) & 255U);
    }
    return input;
}

std::vector<std::uint8_t> makeLowRankInput(const unsigned imageWidth, const unsigned imageHeight) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(imageWidth) * imageHeight * 3U);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index * 41U + index / 5U * 23U + 7U) & 255U);
    }
    return input;
}

void makeFactors(const unsigned kernelExtent, const unsigned rank, std::vector<float> & horizontalFactors, std::vector<float> & verticalFactors) {
    horizontalFactors.resize(static_cast<std::size_t>(rank) * kernelExtent);
    verticalFactors.resize(static_cast<std::size_t>(rank) * kernelExtent);
    for (unsigned rankIndex = 0; rankIndex < rank; ++rankIndex) {
        for (unsigned index = 0; index < kernelExtent; ++index) {
            horizontalFactors[rankIndex * kernelExtent + index] =
                static_cast<float>(static_cast<int>((rankIndex * 5U + index * 3U) % 9U) - 4) / static_cast<float>(kernelExtent * 3U);
            verticalFactors[rankIndex * kernelExtent + index] =
                static_cast<float>(static_cast<int>((rankIndex * 7U + index * 5U) % 11U) - 5) / static_cast<float>(kernelExtent * 4U);
        }
    }
}

std::vector<std::uint8_t> makeFrequencyInput(const unsigned imageWidth, const unsigned imageHeight) {
    std::vector<std::uint8_t> input(static_cast<std::size_t>(imageWidth) * imageHeight * 3U);
    std::uint32_t state = 0x4b1d2397U;
    for (std::size_t index = 0; index < input.size(); ++index) {
        state = state * 1664525U + 1013904223U;
        input[index] = static_cast<std::uint8_t>((state >> 24U) ^ (index * 29U));
    }
    return input;
}

std::vector<float> makeFrequencyWeights(const unsigned kernelExtent) {
    static constexpr float weightPattern[] = {0.03125F, -0.0625F, 0.09375F, 0.125F, -0.15625F, 0.1875F, -0.125F, 0.0625F};
    const unsigned kernelArea = kernelExtent * kernelExtent;
    std::vector<float> weights(kernelArea);
    for (unsigned index = 0; index < kernelArea; ++index) {
        weights[index] = weightPattern[(index * 5U + kernelArea) % 8U] / static_cast<float>(std::max(1U, kernelArea / 11U));
    }
    return weights;
}

bool applyAndCheckHash(
    const std::string & caseName,
    const std::shared_ptr<const CompiledConvFilter> & filter,
    const std::vector<std::uint8_t> & input,
    const std::uint64_t expectedHash
) {
    AlignedWorkspace workspace(*filter);
    std::vector<std::uint8_t> output(input.size());
    if (!filter->apply(input.data(), output.data(), workspace.data())) {
        std::cerr << caseName << ": non-overlapping invocation rejected\n";
        return false;
    }
    const std::uint64_t actualHash = hashBytes(output);
    if (actualHash != expectedHash) {
        std::cerr << caseName << ": output hash " << actualHash << " != " << expectedHash << '\n';
        return false;
    }
    return true;
}

bool testDefault() {
    const KernelCase cases[] = {
        {"default_dense", 7, 5, 3, 11177650904061215901ULL},
        {"default_odd_shape", 65, 64, 5, 16886679512232514536ULL},
        {"default_smaller_than_kernel", 2, 2, 5, 11518649368490289918ULL},
    };
    for (const KernelCase & testCase : cases) {
        const auto weights = makeDefaultWeights(testCase.kernelExtent);
        const kernel::image::DefaultConvFilter configuration{testCase.kernelExtent, testCase.kernelExtent, {weights.data(), weights.size()}};
        const auto filter = kernel::image::compileConvFilter(testCase.imageWidth, testCase.imageHeight, configuration);
        if (filter->mode() != kernel::image::ConvFilterMode::Default || filter->workspaceSize() != 0U
            || !applyAndCheckHash(testCase.caseName, filter, makeDefaultInput(testCase.imageWidth, testCase.imageHeight), testCase.expectedHash))
        {
            return false;
        }
    }
    return true;
}

bool testUniform() {
    const KernelCase cases[] = {
        {"uniform_standard", 7, 5, 3, 10971302695152207229ULL},
        {"uniform_large_kernel", 65, 64, 11, 928285097014713005ULL},
    };
    for (const KernelCase & testCase : cases) {
        const kernel::image::UniformConvFilter configuration{
            testCase.kernelExtent, testCase.kernelExtent, 1.0F / static_cast<float>(testCase.kernelExtent * testCase.kernelExtent)
        };
        const auto filter = kernel::image::compileConvFilter(testCase.imageWidth, testCase.imageHeight, configuration);
        if (filter->mode() != kernel::image::ConvFilterMode::Uniform || filter->workspaceSize() != static_cast<std::size_t>(testCase.imageWidth) * 16U
            || !applyAndCheckHash(testCase.caseName, filter, makeUniformInput(testCase.imageWidth, testCase.imageHeight), testCase.expectedHash))
        {
            return false;
        }
    }
    return true;
}

bool testLowRank() {
    struct LowRankCase {
        const char * caseName;
        unsigned imageWidth;
        unsigned imageHeight;
        unsigned kernelExtent;
        unsigned rank;
        std::uint64_t expectedHash;
    };
    const LowRankCase cases[] = {
        {"low_rank_one", 31, 17, 5, 1, 11148156061091218269ULL},
        {"low_rank_three", 65, 64, 7, 3, 15132234023381778503ULL},
    };
    for (const LowRankCase & testCase : cases) {
        std::vector<float> horizontalFactors;
        std::vector<float> verticalFactors;
        makeFactors(testCase.kernelExtent, testCase.rank, horizontalFactors, verticalFactors);
        const kernel::image::LowRankConvFilter configuration{
            testCase.kernelExtent,
            testCase.kernelExtent,
            testCase.rank,
            {horizontalFactors.data(), horizontalFactors.size()},
            {verticalFactors.data(), verticalFactors.size()}
        };
        const auto filter = kernel::image::compileConvFilter(testCase.imageWidth, testCase.imageHeight, configuration);
        if (filter->mode() != kernel::image::ConvFilterMode::LowRank || filter->workspaceSize() == 0U
            || !applyAndCheckHash(testCase.caseName, filter, makeLowRankInput(testCase.imageWidth, testCase.imageHeight), testCase.expectedHash))
        {
            return false;
        }
    }

    constexpr unsigned runtimeImageWidth = 256;
    constexpr unsigned runtimeImageHeight = 257;
    constexpr unsigned runtimeKernelExtent = 7;
    constexpr unsigned runtimeRank = 3;
    std::vector<float> horizontalFactors;
    std::vector<float> verticalFactors;
    makeFactors(runtimeKernelExtent, runtimeRank, horizontalFactors, verticalFactors);
    const kernel::image::LowRankConvFilter configuration{
        runtimeKernelExtent,
        runtimeKernelExtent,
        runtimeRank,
        {horizontalFactors.data(), horizontalFactors.size()},
        {verticalFactors.data(), verticalFactors.size()}
    };
    const auto filter = kernel::image::compileConvFilter(runtimeImageWidth, runtimeImageHeight, configuration);
    constexpr std::size_t expectedWorkspace = static_cast<std::size_t>(runtimeImageWidth) * runtimeImageHeight * runtimeRank * 3U * sizeof(float);
    if (filter->workspaceSize() != expectedWorkspace || filter->workspaceAlignment() != alignof(float)) {
        std::cerr << "low-rank workspace contract mismatch\n";
        return false;
    }
    return true;
}

bool testFrequency() {
    const KernelCase cases[] = {
        {"frequency_complete_tile", 6, 6, 3, 10939659442403922399ULL},
        {"frequency_incomplete_tile", 7, 5, 3, 4120616837981645535ULL},
        {"frequency_smaller_than_kernel", 2, 2, 5, 9598195261456431560ULL},
    };
    for (const KernelCase & testCase : cases) {
        const auto weights = makeFrequencyWeights(testCase.kernelExtent);
        const kernel::image::FrequencyConvFilter configuration{testCase.kernelExtent, testCase.kernelExtent, {weights.data(), weights.size()}};
        const auto filter = kernel::image::compileConvFilter(testCase.imageWidth, testCase.imageHeight, configuration);
        if (filter->mode() != kernel::image::ConvFilterMode::Frequency || filter->workspaceSize() == 0U
            || !applyAndCheckHash(testCase.caseName, filter, makeFrequencyInput(testCase.imageWidth, testCase.imageHeight), testCase.expectedHash))
        {
            return false;
        }
    }
    return true;
}

bool testCacheAndRanges() {
    constexpr unsigned imageWidth = 7;
    constexpr unsigned imageHeight = 5;
    constexpr unsigned kernelExtent = 3;
    auto firstWeights = makeDefaultWeights(kernelExtent);
    auto secondWeights = firstWeights;
    secondWeights[0] += 0.25F;
    const kernel::image::DefaultConvFilter firstConfiguration{kernelExtent, kernelExtent, {firstWeights.data(), firstWeights.size()}};
    const kernel::image::DefaultConvFilter secondConfiguration{kernelExtent, kernelExtent, {secondWeights.data(), secondWeights.size()}};
    const auto firstFilter = kernel::image::compileConvFilter(imageWidth, imageHeight, firstConfiguration);
    const auto repeatedFilter = kernel::image::compileConvFilter(imageWidth, imageHeight, firstConfiguration);
    const auto secondFilter = kernel::image::compileConvFilter(imageWidth, imageHeight, secondConfiguration);
    const auto input = makeDefaultInput(imageWidth, imageHeight);
    std::vector<std::uint8_t> firstOutput(input.size());
    std::vector<std::uint8_t> repeatedOutput(input.size());
    std::vector<std::uint8_t> secondOutput(input.size());
    if (!firstFilter->apply(input.data(), firstOutput.data(), nullptr) || !repeatedFilter->apply(input.data(), repeatedOutput.data(), nullptr)
        || !secondFilter->apply(input.data(), secondOutput.data(), nullptr) || firstOutput != repeatedOutput || firstOutput == secondOutput)
    {
        std::cerr << "cache identity did not preserve weight identity\n";
        return false;
    }

    std::vector<std::uint8_t> overlapping(input.size() + 1U);
    std::copy(input.begin(), input.end(), overlapping.begin());
    if (firstFilter->apply(overlapping.data(), overlapping.data(), nullptr)
        || firstFilter->apply(overlapping.data(), overlapping.data() + 1U, nullptr))
    {
        std::cerr << "overlapping packed RGB ranges were accepted\n";
        return false;
    }

    std::vector<std::uint8_t> adjacent(input.size() * 2U);
    std::copy(input.begin(), input.end(), adjacent.begin());
    if (!firstFilter->apply(adjacent.data(), adjacent.data() + input.size(), nullptr)) {
        std::cerr << "adjacent non-overlapping packed RGB ranges were rejected\n";
        return false;
    }
    const std::vector<std::uint8_t> adjacentOutput(adjacent.begin() + input.size(), adjacent.end());
    return adjacentOutput == firstOutput;
}

bool testConcurrentFrequency() {
    constexpr unsigned imageWidth = 65;
    constexpr unsigned imageHeight = 64;
    constexpr unsigned kernelExtent = 11;
    const auto weights = makeFrequencyWeights(kernelExtent);
    const kernel::image::FrequencyConvFilter configuration{kernelExtent, kernelExtent, {weights.data(), weights.size()}};
    const auto filter = kernel::image::compileConvFilter(imageWidth, imageHeight, configuration);
    const auto input = makeFrequencyInput(imageWidth, imageHeight);
    constexpr std::uint64_t expectedHash = 12980758057963005113ULL;
    std::vector<std::uint8_t> results(4U);
    std::vector<std::thread> threads;
    for (unsigned index = 0; index < results.size(); ++index) {
        threads.emplace_back([&, index] {
            AlignedWorkspace workspace(*filter);
            std::vector<std::uint8_t> output(input.size());
            results[index] = filter->apply(input.data(), output.data(), workspace.data()) && hashBytes(output) == expectedHash;
        });
    }
    for (std::thread & thread : threads) {
        thread.join();
    }
    if (!std::all_of(results.begin(), results.end(), [](const std::uint8_t result) { return result != 0U; })) {
        std::cerr << "concurrent frequency calls with separate workspaces diverged\n";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    return testDefault() && testUniform() && testLowRank() && testFrequency() && testCacheAndRanges() && testConcurrentFrequency() ? 0 : 1;
}
