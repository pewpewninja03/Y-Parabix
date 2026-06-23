/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_loader.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>

namespace image {

struct __attribute__((packed)) BMPFileHeader {
  uint8_t signature[2]; // 'B', 'M'
  uint32_t fileSize;
  uint32_t reserved;
  uint32_t dataOffset;
};

struct __attribute__((packed)) BMPInfoHeader {
  uint32_t size; // size of this header (=40)
  int32_t width;
  int32_t height;
  uint16_t planes; // must be 1
  uint16_t bitsPerPixel;
  uint32_t compression; // 0 = BI_RGB no compression
  uint32_t imageSize;
  int32_t xPixelsPerM;
  int32_t yPixelsPerM;
  uint32_t colorsUsed;
  uint32_t importantColors;
};

static_assert(sizeof(BMPFileHeader) == 14, "BMPFileHeader must be 14 bytes");
static_assert(sizeof(BMPInfoHeader) == 40, "BMPInfoHeader must be 40 bytes");

void readBMPHeader(int fd, BMPInfo &info) {
  BMPFileHeader fh;
  ssize_t bytesRead = read(fd, &fh, sizeof(fh));
  if (bytesRead != static_cast<ssize_t>(sizeof(fh))) {
    throw std::runtime_error("BMP: failed to read file header");
  }
  if (fh.signature[0] != 'B' || fh.signature[1] != 'M') {
    throw std::runtime_error("BMP: file signature does not match BMP format");
  }

  BMPInfoHeader ih;
  bytesRead = read(fd, &ih, sizeof(ih));
  if (bytesRead != static_cast<ssize_t>(sizeof(ih))) {
    throw std::runtime_error("BMP: failed to read info header");
  }
  if (ih.size != 40) {
    throw std::runtime_error("BMP: unsupported header (size != 40)");
  }
  if (ih.bitsPerPixel != 8) {
    throw std::runtime_error(
        "BMP: only 8-bit BMPs are supported, got bitsPerPixel=" +
        std::to_string(ih.bitsPerPixel));
  }
  if (ih.compression != 0) {
    throw std::runtime_error(
        "BMP: only uncompressed BMPs (BI_RGB) are supported, got compression=" +
        std::to_string(ih.compression));
  }
  if (ih.width <= 0) {
    throw std::runtime_error("BMP: zero or negative width is not supported");
  }

  info.width = static_cast<uint32_t>(ih.width);
  info.height = static_cast<uint32_t>(ih.height < 0 ? -ih.height : ih.height);
  info.rowStride =
      ((info.width + 3u) / 4u) * 4u; // round up to nearest multiple of 4
  info.pixelOffset = fh.dataOffset;
  info.rowsBottomUp = (ih.height > 0);

  if (info.height == 0) {
    throw std::runtime_error("BMP: zero height is not supported");
  }

  // Seek to the first pixel byte.
  if (lseek(fd, static_cast<off_t>(info.pixelOffset), SEEK_SET) ==
      static_cast<off_t>(-1)) {
    throw std::runtime_error("BMP: failed to seek to pixel data at offset " +
                             std::to_string(info.pixelOffset));
  }
}

} // namespace image
