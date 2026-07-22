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


uint32_t getBMP24RowStride(uint32_t width);

std::vector<uint8_t>
createBMP24PixelData(const kernel::StreamSetPtr &redBytes,
                     const kernel::StreamSetPtr &greenBytes,
                     const kernel::StreamSetPtr &blueBytes,
                     const BMPInfo &outputInfo);


void writeBMP24(const std::string &outputPath,
                const std::vector<uint8_t> &bmpPixelData,
                const BMPInfo &outputInfo);

} // namespace image
