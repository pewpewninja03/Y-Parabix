#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace kernel::image {

enum class ConvFilterMode {
    Default,
    Uniform,
    LowRank,
    Frequency
};

struct FloatArrayView {
    const float * data;
    std::size_t size;
};

struct DefaultConvFilter {
    unsigned kernelWidth;
    unsigned kernelHeight;
    FloatArrayView weights;
};

struct UniformConvFilter {
    unsigned kernelWidth;
    unsigned kernelHeight;
    float weight;
};

struct LowRankConvFilter {
    unsigned kernelWidth;
    unsigned kernelHeight;
    unsigned rank;
    FloatArrayView horizontalFactors;
    FloatArrayView verticalFactors;
};

struct FrequencyConvFilter {
    unsigned kernelWidth;
    unsigned kernelHeight;
    FloatArrayView weights;
};

namespace internal {
class CompiledFilterImplementation;
}

class CompiledConvFilter {
   public:
    ConvFilterMode mode() const noexcept;
    unsigned imageWidth() const noexcept;
    unsigned imageHeight() const noexcept;
    std::size_t workspaceSize() const noexcept;
    std::size_t workspaceAlignment() const noexcept;

    bool apply(const std::uint8_t * input, std::uint8_t * output, void * workspace) const noexcept;

   private:
    explicit CompiledConvFilter(std::shared_ptr<const internal::CompiledFilterImplementation> implementation);

    std::shared_ptr<const internal::CompiledFilterImplementation> implementation;

    friend std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned, unsigned, const DefaultConvFilter &);
    friend std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned, unsigned, const UniformConvFilter &);
    friend std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned, unsigned, const LowRankConvFilter &);
    friend std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned, unsigned, const FrequencyConvFilter &);
};

std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned imageWidth, unsigned imageHeight, const DefaultConvFilter & configuration);

std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned imageWidth, unsigned imageHeight, const UniformConvFilter & configuration);

std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned imageWidth, unsigned imageHeight, const LowRankConvFilter & configuration);

std::shared_ptr<const CompiledConvFilter> compileConvFilter(unsigned imageWidth, unsigned imageHeight, const FrequencyConvFilter & configuration);

}  // namespace kernel::image
