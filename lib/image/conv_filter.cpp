#include "conv_filter_common.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#if LLVM_VERSION_MAJOR >= 17
#include <llvm/TargetParser/Host.h>
#else
#include <llvm/Support/Host.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <future>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace kernel::image {
namespace internal {
namespace {

constexpr std::uint32_t DefaultImplementationVersion = 2;
constexpr std::uint32_t UniformImplementationVersion = 2;
constexpr std::uint32_t LowRankImplementationVersion = 3;
constexpr std::uint32_t FrequencyImplementationVersion = 2;

std::mutex compiledFilterCacheMutex;
std::mutex jitCompilationMutex;
std::unordered_map<std::string, std::weak_ptr<const CompiledFilterImplementation>> compiledFilterCache;
using CompilationFuture = std::shared_future<std::shared_ptr<const CompiledFilterImplementation>>;
std::unordered_map<std::string, CompilationFuture> pendingCompilations;

template <typename Factory>
std::shared_ptr<const CompiledFilterImplementation> getOrCompile(const std::string & cacheKey, Factory compileFilter) {
    std::shared_ptr<std::promise<std::shared_ptr<const CompiledFilterImplementation>>> compilationPromise;
    CompilationFuture pendingCompilation;
    {
        const std::lock_guard<std::mutex> lock(compiledFilterCacheMutex);
        const auto cacheEntry = compiledFilterCache.find(cacheKey);
        if (cacheEntry != compiledFilterCache.end()) {
            if (auto compiledFilter = cacheEntry->second.lock())
                return compiledFilter;
        }
        const auto pendingEntry = pendingCompilations.find(cacheKey);
        if (pendingEntry != pendingCompilations.end()) {
            pendingCompilation = pendingEntry->second;
        } else {
            compilationPromise = std::make_shared<std::promise<std::shared_ptr<const CompiledFilterImplementation>>>();
            pendingCompilation = compilationPromise->get_future().share();
            pendingCompilations.emplace(cacheKey, pendingCompilation);
        }
    }
    if (!compilationPromise)
        return pendingCompilation.get();

    try {
        std::shared_ptr<const CompiledFilterImplementation> compiledFilter;
        {
            const std::lock_guard<std::mutex> lock(jitCompilationMutex);
            compiledFilter = std::move(compileFilter());
        }
        {
            const std::lock_guard<std::mutex> lock(compiledFilterCacheMutex);
            compiledFilterCache[cacheKey] = compiledFilter;
            pendingCompilations.erase(cacheKey);
        }
        compilationPromise->set_value(compiledFilter);
        return compiledFilter;
    } catch (...) {
        {
            const std::lock_guard<std::mutex> lock(compiledFilterCacheMutex);
            pendingCompilations.erase(cacheKey);
        }
        compilationPromise->set_exception(std::current_exception());
        throw;
    }
}

std::size_t validateDimensions(const unsigned imageWidth, const unsigned imageHeight, const unsigned kernelWidth, const unsigned kernelHeight) {
    if (imageWidth == 0U || imageHeight == 0U || kernelWidth == 0U || kernelHeight == 0U)
        throw std::invalid_argument("image and kernel dimensions must be nonzero");
    assert(imageWidth != 0 && imageHeight != 0);
    assert(kernelWidth != 0 && kernelHeight != 0);
    assert((kernelWidth & 1U) != 0 && (kernelHeight & 1U) != 0);
    checkedImageByteCount(imageWidth, imageHeight);
    const std::size_t kernelArea = checkedKernelArea(kernelWidth, kernelHeight);
    checkedAdd(imageWidth, kernelWidth - 1U, "horizontal convolution coordinate overflow");
    checkedAdd(imageHeight, kernelHeight - 1U, "vertical convolution coordinate overflow");
    return kernelArea;
}

std::vector<float> copyFiniteValues(const FloatArrayView values, const std::size_t expectedCount, const char * errorMessage) {
    assert(values.data != nullptr);
    assert(values.size == expectedCount);
    std::vector<float> copiedValues(values.data, values.data + values.size);
    if (!std::all_of(copiedValues.begin(), copiedValues.end(), [](const float value) { return std::isfinite(value); }))
        throw std::invalid_argument(errorMessage);
    return copiedValues;
}

void validateDefaultNumericalBound(const std::vector<float> & weights) {
    constexpr long double maximumFloat = static_cast<long double>(std::numeric_limits<float>::max());
    long double accumulatedBound = 0.0L;
    for (const float weight : weights) {
        const long double termBound = 255.0L * std::fabs(static_cast<long double>(weight));
        if (!std::isfinite(termBound) || termBound > maximumFloat - accumulatedBound)
            throw std::invalid_argument("default accumulated-value bound exceeds float range");
        accumulatedBound += termBound;
    }
}

void validateLowRankNumericalBound(
    const std::vector<float> & horizontalFactors,
    const std::vector<float> & verticalFactors,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const unsigned rank
) {
    constexpr long double maximumFloat = static_cast<long double>(std::numeric_limits<float>::max());
    long double outputBound = 0.0L;
    for (unsigned rankIndex = 0; rankIndex < rank; ++rankIndex) {
        long double horizontalBound = 0.0L;
        const std::size_t horizontalOffset = static_cast<std::size_t>(rankIndex) * kernelWidth;
        for (unsigned column = 0; column < kernelWidth; ++column) {
            const long double termBound = 255.0L * std::fabs(static_cast<long double>(horizontalFactors[horizontalOffset + column]));
            if (!std::isfinite(termBound) || termBound > maximumFloat - horizontalBound)
                throw std::invalid_argument("low-rank horizontal-value bound exceeds float range");
            horizontalBound += termBound;
        }

        const std::size_t verticalOffset = static_cast<std::size_t>(rankIndex) * kernelHeight;
        for (unsigned row = 0; row < kernelHeight; ++row) {
            const long double termBound = horizontalBound * std::fabs(static_cast<long double>(verticalFactors[verticalOffset + row]));
            if (!std::isfinite(termBound) || termBound > maximumFloat - outputBound)
                throw std::invalid_argument("low-rank output-value bound exceeds float range");
            outputBound += termBound;
        }
    }
}

}  // namespace

CacheKeyBuilder::CacheKeyBuilder(const ConvFilterMode mode) {
    appendString("parabix.image.convolution");
    append(ArithmeticContractVersion);
    append(mode);
}

void CacheKeyBuilder::appendBytes(const void * byteData, const std::size_t byteCount) {
    const auto length = static_cast<std::uint64_t>(byteCount);
    keyBytes.append(reinterpret_cast<const char *>(&length), sizeof(length));
    keyBytes.append(static_cast<const char *>(byteData), byteCount);
}

void CacheKeyBuilder::appendString(const std::string & value) {
    appendBytes(value.data(), value.size());
}

const std::string & CacheKeyBuilder::cacheKey() const noexcept {
    return keyBytes;
}

std::string CacheKeyBuilder::persistentIdentity() const {
    const auto digest = llvm::SHA256::hash(llvm::ArrayRef<std::uint8_t>(reinterpret_cast<const std::uint8_t *>(keyBytes.data()), keyBytes.size()));
    static constexpr char HexDigits[] = "0123456789abcdef";
    std::string identity;
    identity.reserve(digest.size() * 2U);
    for (const std::uint8_t byte : digest) {
        identity.push_back(HexDigits[byte >> 4U]);
        identity.push_back(HexDigits[byte & 0x0fU]);
    }
    return identity;
}

CompiledFilterImplementation::CompiledFilterImplementation(
    const ConvFilterMode mode,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const std::size_t imageByteCount,
    const std::size_t workspaceByteCount,
    const std::size_t workspaceAlignment
)
    : filterMode(mode),
      imageWidthInPixels(imageWidth),
      imageHeightInPixels(imageHeight),
      imageByteCount(imageByteCount),
      workspaceByteCount(workspaceByteCount),
      workspaceByteAlignment(workspaceAlignment) {}

ConvFilterMode CompiledFilterImplementation::mode() const noexcept {
    return filterMode;
}

unsigned CompiledFilterImplementation::imageWidth() const noexcept {
    return imageWidthInPixels;
}

unsigned CompiledFilterImplementation::imageHeight() const noexcept {
    return imageHeightInPixels;
}

std::size_t CompiledFilterImplementation::workspaceSize() const noexcept {
    return workspaceByteCount;
}

std::size_t CompiledFilterImplementation::workspaceAlignment() const noexcept {
    return workspaceByteAlignment;
}

bool CompiledFilterImplementation::apply(const std::uint8_t * input, std::uint8_t * output, void * workspace) const noexcept {
    assert(input != nullptr);
    assert(output != nullptr);
    assert(workspaceByteCount == 0 || workspace != nullptr);
    assert(workspaceByteCount == 0 || reinterpret_cast<std::uintptr_t>(workspace) % workspaceByteAlignment == 0);

    const auto inputAddress = reinterpret_cast<std::uintptr_t>(input);
    const auto outputAddress = reinterpret_cast<std::uintptr_t>(output);
    if (inputAddress > std::numeric_limits<std::uintptr_t>::max() - imageByteCount
        || outputAddress > std::numeric_limits<std::uintptr_t>::max() - imageByteCount)
    {
        return false;
    }
    const auto inputEnd = inputAddress + imageByteCount;
    const auto outputEnd = outputAddress + imageByteCount;
    if (inputAddress < outputEnd && outputAddress < inputEnd)
        return false;
    invoke(input, output, workspace);
    return true;
}

std::size_t checkedAdd(const std::size_t left, const std::size_t right, const char * description) {
    if (right > std::numeric_limits<std::size_t>::max() - left)
        throw std::overflow_error(description);
    return left + right;
}

std::size_t checkedMultiply(const std::size_t left, const std::size_t right, const char * description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        throw std::overflow_error(description);
    return left * right;
}

std::size_t checkedCeilDivide(const std::size_t value, const std::size_t divisor, const char * description) {
    if (divisor == 0U)
        throw std::invalid_argument(description);
    const std::size_t quotient = value / divisor;
    return value % divisor == 0U ? quotient : checkedAdd(quotient, 1U, description);
}

std::size_t checkedImageByteCount(const unsigned imageWidth, const unsigned imageHeight) {
    return checkedMultiply(
        checkedMultiply(imageWidth, imageHeight, "image pixel count overflow"), ColorChannelCount, "packed RGB image size overflow"
    );
}

std::size_t checkedKernelArea(const unsigned kernelWidth, const unsigned kernelHeight) {
    return checkedMultiply(kernelWidth, kernelHeight, "kernel area overflow");
}

std::size_t checkedUniformWorkspaceByteCount(const unsigned imageWidth) {
    return checkedMultiply(
        checkedMultiply(imageWidth, 4U, "uniform workspace value count overflow"), sizeof(std::uint32_t), "uniform workspace byte count overflow"
    );
}

std::size_t checkedLowRankWorkspaceByteCount(const unsigned imageWidth, const unsigned imageHeight, const unsigned rank) {
    std::size_t values = checkedMultiply(imageWidth, imageHeight, "low-rank workspace pixel count overflow");
    values = checkedMultiply(values, ColorChannelCount, "low-rank workspace channel count overflow");
    values = checkedMultiply(values, rank, "low-rank workspace rank count overflow");
    return checkedMultiply(values, sizeof(float), "low-rank workspace byte count overflow");
}

std::string targetIdentity() {
    static const std::string targetDescription = [] {
        llvm::InitializeNativeTarget();
        const std::string triple = llvm::sys::getDefaultTargetTriple();
        const std::string cpu = llvm::sys::getHostCPUName().str();
#if LLVM_VERSION_MAJOR >= 19
        const llvm::StringMap<bool> hostFeatures = llvm::sys::getHostCPUFeatures();
#else
        llvm::StringMap<bool> hostFeatures;
        llvm::sys::getHostCPUFeatures(hostFeatures);
#endif
        std::vector<std::string> enabledFeatures;
        for (const auto & feature : hostFeatures) {
            if (feature.second)
                enabledFeatures.push_back(feature.first().str());
        }
        std::sort(enabledFeatures.begin(), enabledFeatures.end());
        std::string featureString;
        for (const std::string & feature : enabledFeatures) {
            featureString.append("+").append(feature).append(",");
        }

        std::string targetError;
        const llvm::Target * target = llvm::TargetRegistry::lookupTarget(triple, targetError);
        if (target == nullptr)
            throw std::runtime_error("unable to resolve host LLVM target: " + targetError);
        llvm::TargetOptions targetOptions;
        std::unique_ptr<llvm::TargetMachine> targetMachine(target->createTargetMachine(triple, cpu, featureString, targetOptions, llvm::Reloc::PIC_));
        if (targetMachine == nullptr)
            throw std::runtime_error("unable to create host LLVM target machine");
        return triple + '\n' + cpu + '\n' + featureString + '\n' + targetMachine->createDataLayout().getStringRepresentation() + "\nCPUDriver/IDISA";
    }();
    return targetDescription;
}

void appendCommonCacheIdentity(
    CacheKeyBuilder & key,
    const unsigned imageWidth,
    const unsigned imageHeight,
    const unsigned kernelWidth,
    const unsigned kernelHeight,
    const unsigned bitBlockWidth
) {
    key.append(imageWidth);
    key.append(imageHeight);
    key.append(kernelWidth);
    key.append(kernelHeight);
    key.appendString(targetIdentity());
    key.append(bitBlockWidth);
}

unsigned frequencyTransformExtent(const unsigned kernelWidth, const unsigned kernelHeight) {
    const unsigned kernelExtent = std::max(kernelWidth, kernelHeight);
    if (kernelExtent == 1U)
        return 1U;
    const std::size_t minimumExtent = checkedMultiply(kernelExtent - 1U, 2U, "frequency transform extent overflow");
    unsigned transformExtent = 1;
    while (transformExtent < std::max<std::size_t>(minimumExtent, 8U)) {
        if (transformExtent > 2048U)
            throw std::invalid_argument("unsupported frequency transform extent");
        transformExtent = static_cast<unsigned>(checkedMultiply(transformExtent, 2U, "frequency transform extent overflow"));
    }
    return transformExtent;
}

FrequencyLayout resolveFrequencyLayout(const unsigned bitBlockWidth, const unsigned transformExtent) {
    const unsigned availableFloatLanes = bitBlockWidth / 32U;
    if (availableFloatLanes < 2U)
        throw std::invalid_argument("IDISA width cannot form one packed-real frequency record");
    const unsigned maximumLanes = transformExtent <= 16U ? 8U : 4U;
    unsigned logicalFloatLanes = std::min(availableFloatLanes, maximumLanes);
    logicalFloatLanes -= logicalFloatLanes % 2U;
    logicalFloatLanes = std::max(logicalFloatLanes, 2U);
    const unsigned independentVectorCount = transformExtent <= 16U ? 2U : 1U;
    const unsigned tilesPerVector = logicalFloatLanes / 2U;
    return {transformExtent, logicalFloatLanes, independentVectorCount, tilesPerVector, tilesPerVector * independentVectorCount};
}

}  // namespace internal

CompiledConvFilter::CompiledConvFilter(std::shared_ptr<const internal::CompiledFilterImplementation> compiledImplementation)
    : implementation(std::move(compiledImplementation)) {}

ConvFilterMode CompiledConvFilter::mode() const noexcept {
    return implementation->mode();
}

unsigned CompiledConvFilter::imageWidth() const noexcept {
    return implementation->imageWidth();
}

unsigned CompiledConvFilter::imageHeight() const noexcept {
    return implementation->imageHeight();
}

std::size_t CompiledConvFilter::workspaceSize() const noexcept {
    return implementation->workspaceSize();
}

std::size_t CompiledConvFilter::workspaceAlignment() const noexcept {
    return implementation->workspaceAlignment();
}

bool CompiledConvFilter::apply(const std::uint8_t * input, std::uint8_t * output, void * workspace) const noexcept {
    return implementation->apply(input, output, workspace);
}

std::shared_ptr<const CompiledConvFilter> compileConvFilter(
    const unsigned imageWidth, const unsigned imageHeight, const DefaultConvFilter & configuration
) {
    const std::size_t kernelArea = internal::validateDimensions(imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight);
    auto weights = internal::copyFiniteValues(configuration.weights, kernelArea, "default weights must be finite");
    internal::validateDefaultNumericalBound(weights);
    auto driver = std::make_unique<CPUDriver>("image_default_convolution");
    const unsigned bitBlockWidth = driver->getBitBlockWidth();
    internal::CacheKeyBuilder cacheKey(ConvFilterMode::Default);
    internal::appendCommonCacheIdentity(cacheKey, imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight, bitBlockWidth);
    cacheKey.append(internal::DefaultImplementationVersion);
    cacheKey.appendBytes(weights.data(), internal::checkedMultiply(weights.size(), sizeof(float), "default weight byte count overflow"));
    auto persistentIdentity = cacheKey.persistentIdentity();
    auto compiledImplementation = internal::getOrCompile(
        cacheKey.cacheKey(),
        [driver = std::move(driver),
         imageWidth,
         imageHeight,
         configuration,
         weights = std::move(weights),
         persistentIdentity = std::move(persistentIdentity)]() mutable {
            return internal::compileDefaultFilter(
                std::move(driver),
                imageWidth,
                imageHeight,
                configuration.kernelWidth,
                configuration.kernelHeight,
                std::move(weights),
                std::move(persistentIdentity)
            );
        }
    );
    return std::shared_ptr<const CompiledConvFilter>(new CompiledConvFilter(std::move(compiledImplementation)));
}

std::shared_ptr<const CompiledConvFilter> compileConvFilter(
    const unsigned imageWidth, const unsigned imageHeight, const UniformConvFilter & configuration
) {
    const std::size_t kernelArea = internal::validateDimensions(imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight);
    if (!std::isfinite(configuration.weight))
        throw std::invalid_argument("uniform weight must be finite");
    if (kernelArea > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) / 255U)
        throw std::invalid_argument("uniform window sum exceeds int32 range");
    internal::checkedUniformWorkspaceByteCount(imageWidth);
    auto driver = std::make_unique<CPUDriver>("image_uniform_convolution");
    const unsigned bitBlockWidth = driver->getBitBlockWidth();
    internal::CacheKeyBuilder cacheKey(ConvFilterMode::Uniform);
    internal::appendCommonCacheIdentity(cacheKey, imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight, bitBlockWidth);
    cacheKey.append(internal::UniformImplementationVersion);
    cacheKey.append(configuration.weight);
    auto persistentIdentity = cacheKey.persistentIdentity();
    auto compiledImplementation = internal::getOrCompile(
        cacheKey.cacheKey(),
        [driver = std::move(driver), imageWidth, imageHeight, configuration, persistentIdentity = std::move(persistentIdentity)]() mutable {
            return internal::compileUniformFilter(
                std::move(driver),
                imageWidth,
                imageHeight,
                configuration.kernelWidth,
                configuration.kernelHeight,
                configuration.weight,
                std::move(persistentIdentity)
            );
        }
    );
    return std::shared_ptr<const CompiledConvFilter>(new CompiledConvFilter(std::move(compiledImplementation)));
}

std::shared_ptr<const CompiledConvFilter> compileConvFilter(
    const unsigned imageWidth, const unsigned imageHeight, const LowRankConvFilter & configuration
) {
    internal::validateDimensions(imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight);
    assert(configuration.rank >= 1U && configuration.rank <= 3U);
    internal::checkedLowRankWorkspaceByteCount(imageWidth, imageHeight, configuration.rank);
    const std::size_t horizontalFactorCount =
        internal::checkedMultiply(configuration.rank, configuration.kernelWidth, "horizontal factor count overflow");
    const std::size_t verticalFactorCount =
        internal::checkedMultiply(configuration.rank, configuration.kernelHeight, "vertical factor count overflow");
    auto horizontalFactors = internal::copyFiniteValues(configuration.horizontalFactors, horizontalFactorCount, "horizontal factors must be finite");
    auto verticalFactors = internal::copyFiniteValues(configuration.verticalFactors, verticalFactorCount, "vertical factors must be finite");
    internal::validateLowRankNumericalBound(
        horizontalFactors, verticalFactors, configuration.kernelWidth, configuration.kernelHeight, configuration.rank
    );
    auto driver = std::make_unique<CPUDriver>("image_low_rank_convolution");
    const unsigned bitBlockWidth = driver->getBitBlockWidth();
    internal::CacheKeyBuilder cacheKey(ConvFilterMode::LowRank);
    internal::appendCommonCacheIdentity(cacheKey, imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight, bitBlockWidth);
    cacheKey.append(internal::LowRankImplementationVersion);
    cacheKey.append(configuration.rank);
    cacheKey.appendBytes(
        horizontalFactors.data(), internal::checkedMultiply(horizontalFactors.size(), sizeof(float), "horizontal factor byte count overflow")
    );
    cacheKey.appendBytes(
        verticalFactors.data(), internal::checkedMultiply(verticalFactors.size(), sizeof(float), "vertical factor byte count overflow")
    );
    auto persistentIdentity = cacheKey.persistentIdentity();
    auto compiledImplementation = internal::getOrCompile(
        cacheKey.cacheKey(),
        [driver = std::move(driver),
         imageWidth,
         imageHeight,
         configuration,
         horizontalFactors = std::move(horizontalFactors),
         verticalFactors = std::move(verticalFactors),
         persistentIdentity = std::move(persistentIdentity)]() mutable {
            return internal::compileLowRankFilter(
                std::move(driver),
                imageWidth,
                imageHeight,
                configuration.kernelWidth,
                configuration.kernelHeight,
                configuration.rank,
                std::move(horizontalFactors),
                std::move(verticalFactors),
                std::move(persistentIdentity)
            );
        }
    );
    return std::shared_ptr<const CompiledConvFilter>(new CompiledConvFilter(std::move(compiledImplementation)));
}

std::shared_ptr<const CompiledConvFilter> compileConvFilter(
    const unsigned imageWidth, const unsigned imageHeight, const FrequencyConvFilter & configuration
) {
    const std::size_t kernelArea = internal::validateDimensions(imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight);
    const unsigned transformExtent = internal::frequencyTransformExtent(configuration.kernelWidth, configuration.kernelHeight);
    if (configuration.kernelWidth != configuration.kernelHeight)
        throw std::invalid_argument("frequency kernel must be square");
    auto weights = internal::copyFiniteValues(configuration.weights, kernelArea, "frequency weights must be finite");
    auto driver = std::make_unique<CPUDriver>("image_frequency_convolution");
    const unsigned bitBlockWidth = driver->getBitBlockWidth();
    const internal::FrequencyLayout frequencyLayout = internal::resolveFrequencyLayout(bitBlockWidth, transformExtent);
    const std::size_t validTileExtent = transformExtent - configuration.kernelWidth + 1U;
    const std::size_t tileRows = internal::checkedCeilDivide(imageHeight, validTileExtent, "frequency tile-row count overflow");
    const std::size_t tileColumns = internal::checkedCeilDivide(imageWidth, validTileExtent, "frequency tile-column count overflow");
    const std::size_t tileColumnGroups =
        internal::checkedCeilDivide(tileColumns, frequencyLayout.tilesPerGroup, "frequency tile-column group count overflow");
    internal::checkedMultiply(tileRows, tileColumnGroups, "frequency tile-group count overflow");
    internal::CacheKeyBuilder cacheKey(ConvFilterMode::Frequency);
    internal::appendCommonCacheIdentity(cacheKey, imageWidth, imageHeight, configuration.kernelWidth, configuration.kernelHeight, bitBlockWidth);
    cacheKey.append(internal::FrequencyImplementationVersion);
    cacheKey.append(frequencyLayout.transformExtent);
    cacheKey.append(frequencyLayout.logicalFloatLanes);
    cacheKey.append(frequencyLayout.independentVectorCount);
    cacheKey.append(frequencyLayout.tilesPerVector);
    cacheKey.append(frequencyLayout.tilesPerGroup);
    cacheKey.appendBytes(weights.data(), internal::checkedMultiply(weights.size(), sizeof(float), "frequency weight byte count overflow"));
    auto persistentIdentity = cacheKey.persistentIdentity();
    auto compiledImplementation = internal::getOrCompile(
        cacheKey.cacheKey(),
        [driver = std::move(driver),
         imageWidth,
         imageHeight,
         configuration,
         frequencyLayout,
         weights = std::move(weights),
         persistentIdentity = std::move(persistentIdentity)]() mutable {
            return internal::compileFrequencyFilter(
                std::move(driver),
                imageWidth,
                imageHeight,
                configuration.kernelWidth,
                configuration.kernelHeight,
                frequencyLayout,
                std::move(weights),
                std::move(persistentIdentity)
            );
        }
    );
    return std::shared_ptr<const CompiledConvFilter>(new CompiledConvFilter(std::move(compiledImplementation)));
}

}  // namespace kernel::image
