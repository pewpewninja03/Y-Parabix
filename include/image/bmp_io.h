/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cstdint>
#include <kernel/core/streamsetptr.h>
#include <string>
#include <vector>

namespace image {

struct BMPInfo {
  uint32_t width;
  uint32_t height;
  uint32_t rowStride;     // padded row size in bytes: ((width + 3) / 4) * 4
  uint32_t pixelOffset;   // byte offset from file start to first pixel byte
  uint32_t paletteOffset; // byte offset from file start to first palette entry
  uint32_t numColors;     // number of palette entries (1..256)
  bool rowsBottomUp;      // true when the original BMP height was positive

  // The BGR0 palette is exposed as three 256-entry lookup tables.
  std::vector<unsigned> bTable; // Blue  channel (palette[i*4+0])
  std::vector<unsigned> gTable; // Green channel (palette[i*4+1])
  std::vector<unsigned> rTable; // Red   channel (palette[i*4+2])
};

void readBMPHeader(int fd, BMPInfo &info);

uint32_t getBMP8RowStride(uint32_t width);

// (3 red bits, 3 green bits, 2 blue bits: RRRGGGBB).
uint8_t quantizeRGB332(uint8_t red, uint8_t green, uint8_t blue);

struct BMP24Image {
  uint32_t width = 0;
  uint32_t height = 0;
  bool rowsBottomUp = true;
  std::vector<uint8_t> rgb;

  BMP24Image() = default;
  BMP24Image(uint32_t imageWidth, uint32_t imageHeight, bool bottomUp)
      : width(imageWidth), height(imageHeight), rowsBottomUp(bottomUp),
        rgb(static_cast<std::size_t>(imageWidth) * imageHeight * 3u, 0u) {}

  std::size_t pixelCount() const {
    return static_cast<std::size_t>(width) * height;
  }
  uint8_t *data() { return rgb.data(); }
  const uint8_t *data() const { return rgb.data(); }
};

BMP24Image createBMP24Image(const kernel::StreamSetPtr &redBytes,
                            const kernel::StreamSetPtr &greenBytes,
                            const kernel::StreamSetPtr &blueBytes,
                            uint32_t width, uint32_t height,
                            bool rowsBottomUp);

void writeBMP8(const std::string &outputPath, const BMP24Image &image);

} // namespace image
