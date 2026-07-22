#pragma once

#include <image/conv_filter.h>

#include <kernel/pipeline/driver/cpudriver.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace kernel::image::internal {

constexpr unsigned ColorChannelCount = 3;
constexpr std::uint32_t ArithmeticContractVersion = 1;

struct FrequencyLayout {
    unsigned transformExtent;
    unsigned logicalFloatLanes;
    unsigned independentVectorCount;
    unsigned tilesPerVector;
    unsigned tilesPerGroup;
};

class CacheKeyBuilder {
   public:
    explicit CacheKeyBuilder(ConvFilterMode mode);

    template <typename T>
    void append(const T & value) {
        static_assert(std::is_trivially_copyable<T>::value, "cache values must be byte-copyable");
        appendBytes(&value, sizeof(value));
    }

    void appendBytes(const void * byteData, std::size_t byteCount);
    void appendString(const std::string & value);
    const std::string & cacheKey() const noexcept;
    std::string persistentIdentity() const;

   private:
    std::string keyBytes;
};

class CompiledFilterImplementation {
   public:
    virtual ~CompiledFilterImplementation() = default;

    ConvFilterMode mode() const noexcept;
    unsigned imageWidth() const noexcept;
    unsigned imageHeight() const noexcept;
    std::size_t workspaceSize() const noexcept;
    std::size_t workspaceAlignment() const noexcept;

    bool apply(const std::uint8_t * input, std::uint8_t * output, void * workspace) const noexcept;

   protected:
    CompiledFilterImplementation(
        ConvFilterMode mode,
        unsigned imageWidth,
        unsigned imageHeight,
        std::size_t imageByteCount,
        std::size_t workspaceByteCount,
        std::size_t workspaceAlignment
    );

   private:
    virtual void invoke(const std::uint8_t * input, std::uint8_t * output, void * workspace) const noexcept = 0;

    const ConvFilterMode filterMode;
    const unsigned imageWidthInPixels;
    const unsigned imageHeightInPixels;
    const std::size_t imageByteCount;
    const std::size_t workspaceByteCount;
    const std::size_t workspaceByteAlignment;
};

std::size_t checkedAdd(std::size_t left, std::size_t right, const char * description);
std::size_t checkedMultiply(std::size_t left, std::size_t right, const char * description);
std::size_t checkedCeilDivide(std::size_t value, std::size_t divisor, const char * description);
std::size_t checkedImageByteCount(unsigned imageWidth, unsigned imageHeight);
std::size_t checkedKernelArea(unsigned kernelWidth, unsigned kernelHeight);
std::size_t checkedUniformWorkspaceByteCount(unsigned imageWidth);
std::size_t checkedLowRankWorkspaceByteCount(unsigned imageWidth, unsigned imageHeight, unsigned rank);

std::string targetIdentity();
void appendCommonCacheIdentity(
    CacheKeyBuilder & key, unsigned imageWidth, unsigned imageHeight, unsigned kernelWidth, unsigned kernelHeight, unsigned bitBlockWidth
);

unsigned frequencyTransformExtent(unsigned kernelWidth, unsigned kernelHeight);
FrequencyLayout resolveFrequencyLayout(unsigned bitBlockWidth, unsigned transformExtent);

std::shared_ptr<const CompiledFilterImplementation> compileDefaultFilter(
    std::unique_ptr<CPUDriver> driver,
    unsigned imageWidth,
    unsigned imageHeight,
    unsigned kernelWidth,
    unsigned kernelHeight,
    std::vector<float> weights,
    std::string persistentIdentity
);

std::shared_ptr<const CompiledFilterImplementation> compileUniformFilter(
    std::unique_ptr<CPUDriver> driver,
    unsigned imageWidth,
    unsigned imageHeight,
    unsigned kernelWidth,
    unsigned kernelHeight,
    float weight,
    std::string persistentIdentity
);

std::shared_ptr<const CompiledFilterImplementation> compileLowRankFilter(
    std::unique_ptr<CPUDriver> driver,
    unsigned imageWidth,
    unsigned imageHeight,
    unsigned kernelWidth,
    unsigned kernelHeight,
    unsigned rank,
    std::vector<float> horizontalFactors,
    std::vector<float> verticalFactors,
    std::string persistentIdentity
);

std::shared_ptr<const CompiledFilterImplementation> compileFrequencyFilter(
    std::unique_ptr<CPUDriver> driver,
    unsigned imageWidth,
    unsigned imageHeight,
    unsigned kernelWidth,
    unsigned kernelHeight,
    FrequencyLayout layout,
    std::vector<float> weights,
    std::string persistentIdentity
);

}  // namespace kernel::image::internal
