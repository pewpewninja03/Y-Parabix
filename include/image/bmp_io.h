#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace image {

struct BGRImage {
    std::uint32_t width;
    std::uint32_t height;
    std::vector<std::uint8_t> pixels;

    BGRImage(std::uint32_t width, std::uint32_t height);

    std::uint8_t * data() noexcept;
    const std::uint8_t * data() const noexcept;
};

BGRImage loadBMP(const std::string & path);

void saveBMP(const std::string & path, const BGRImage & image);

}  // namespace image
