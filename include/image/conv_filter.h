#pragma once

#include <cstdint>

namespace kernel::image {

enum class ConvFilterMode { Default };

struct ConvFilterConfig {
    ConvFilterMode mode;
    unsigned width;
    unsigned height;
    unsigned kernelHeight;
    unsigned kernelWidth;
    const float *weights;
    unsigned weightCount;
};

bool applyConvFilter(const uint8_t *inputPixels, uint8_t *outputPixels,
                     const ConvFilterConfig &config);

} // namespace kernel::image
