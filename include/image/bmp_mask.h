#pragma once

#include <image/bmp_io.h>

#include <cstdint>
#include <string>

namespace image {

struct BGRColor {
    std::uint8_t blue;
    std::uint8_t green;
    std::uint8_t red;
};

BGRImage maskImage(const BGRImage & source, const std::string & maskPath, BGRColor color);

}  // namespace image
