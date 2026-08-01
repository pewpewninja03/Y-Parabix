#pragma once

#include <image/bmp_io.h>

#include <cstdint>

namespace image {

BGRImage cropImage(const BGRImage & source, std::uint32_t width, std::uint32_t height);

}  // namespace image
