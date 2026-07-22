/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_io.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>
#include <unistd.h>

namespace image {
namespace {

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
  uint32_t compression; // 0 = BI_RGB (no compression)
  uint32_t imageSize;
  int32_t xPixelsPerM;
  int32_t yPixelsPerM;
  uint32_t colorsUsed;
  uint32_t importantColors;
};

static_assert(sizeof(BMPFileHeader) == 14, "BMPFileHeader must be 14 bytes");
static_assert(sizeof(BMPInfoHeader) == 40, "BMPInfoHeader must be 40 bytes");

constexpr uint16_t RequiredBMPPlanes = 1u;
constexpr uint16_t SupportedBitsPerPixel = 8u;
constexpr uint16_t OutputBitsPerPixel = 24u;
constexpr uint32_t BMPInfoHeaderSize = 40u;
constexpr uint32_t BMPRowAlignment = 4u;
constexpr uint32_t PaletteSize = 256u;
constexpr uint32_t PaletteEntryBytes = 4u;
constexpr uint32_t OutputColorChannelCount = 3u;
constexpr uint32_t BMPHeaderBytes =
    static_cast<uint32_t>(sizeof(BMPFileHeader) + sizeof(BMPInfoHeader));

uint64_t alignBMPRow(uint64_t rowBytes) {
  return ((rowBytes + BMPRowAlignment - 1u) / BMPRowAlignment) *
         BMPRowAlignment;
}

uint32_t checkedBMP24ImageSize(const BMPInfo &info) {
  if (info.height == 0 ||
      info.height >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("BMP output: height is outside the BMP range");
  }
  const uint64_t imageSize =
      static_cast<uint64_t>(getBMP24RowStride(info.width)) * info.height;
  if (imageSize >
      std::numeric_limits<uint32_t>::max() - BMPHeaderBytes) {
    throw std::runtime_error("BMP output: image is too large");
  }
  return static_cast<uint32_t>(imageSize);
}

void readExact(int fd, void *data, std::size_t size,
               const std::string &description) {
  uint8_t *bytes = static_cast<uint8_t *>(data);
  while (size != 0) {
    const ssize_t bytesRead = ::read(fd, bytes, size);
    if (bytesRead < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int error = errno;
      throw std::runtime_error("BMP: failed to read " + description + ": " +
                               std::strerror(error));
    }
    if (bytesRead == 0) {
      throw std::runtime_error("BMP: unexpected end of file while reading " +
                               description);
    }
    bytes += bytesRead;
    size -= static_cast<std::size_t>(bytesRead);
  }
}

void writeAll(int fd, const void *data, std::size_t size,
              const std::string &outputPath) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  while (size != 0) {
    const ssize_t written = ::write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      const int error = errno;
      throw std::runtime_error("BMP output: failed to write " + outputPath +
                               ": " + std::strerror(error));
    }
    if (written == 0) {
      throw std::runtime_error("BMP output: short write to " + outputPath);
    }
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
}

class ScopedFileDescriptor final {
public:
  explicit ScopedFileDescriptor(int fd) : mFD(fd) {}

  ScopedFileDescriptor(const ScopedFileDescriptor &) = delete;
  ScopedFileDescriptor &operator=(const ScopedFileDescriptor &) = delete;

  ~ScopedFileDescriptor() {
    if (mFD >= 0) {
      ::close(mFD);
    }
  }

  int get() const { return mFD; }

  void closeOrThrow(const std::string &path) {
    const int fd = mFD;
    mFD = -1;
    if (::close(fd) != 0) {
      const int error = errno;
      throw std::runtime_error("BMP output: failed to close " + path + ": " +
                               std::strerror(error));
    }
  }

private:
  int mFD;
};

void readPalette(int fd, const BMPInfoHeader &infoHeader, BMPInfo &info) {
  const uint32_t declaredColors =
      infoHeader.colorsUsed != 0 ? infoHeader.colorsUsed : PaletteSize;
  info.numColors = std::min(declaredColors, PaletteSize);
  info.paletteOffset =
      static_cast<uint32_t>(sizeof(BMPFileHeader)) + infoHeader.size;

  const uint32_t paletteBytes = PaletteEntryBytes * info.numColors;
  if (info.pixelOffset < info.paletteOffset ||
      paletteBytes > info.pixelOffset - info.paletteOffset) {
    throw std::runtime_error("BMP: palette region overflows pixel data offset");
  }

  std::vector<uint8_t> paletteData(paletteBytes);
  readExact(fd, paletteData.data(), paletteData.size(), "color table");

  info.bTable.assign(PaletteSize, 0u);
  info.gTable.assign(PaletteSize, 0u);
  info.rTable.assign(PaletteSize, 0u);
  for (uint32_t index = 0; index < info.numColors; ++index) {
    const uint32_t entry = PaletteEntryBytes * index;
    info.bTable[index] = paletteData[entry];
    info.gTable[index] = paletteData[entry + 1u];
    info.rTable[index] = paletteData[entry + 2u];
  }
}

} // namespace

uint32_t getBMP24RowStride(uint32_t width) {
  if (width == 0 ||
      width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("BMP output: width is outside the BMP range");
  }
  const uint64_t rowBytes =
      static_cast<uint64_t>(width) * OutputColorChannelCount;
  const uint64_t rowStride = alignBMPRow(rowBytes);
  if (rowStride > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("BMP output: row stride is too large");
  }
  return static_cast<uint32_t>(rowStride);
}

std::vector<uint8_t>
createBMP24PixelData(const kernel::StreamSetPtr &redBytes,
                     const kernel::StreamSetPtr &greenBytes,
                     const kernel::StreamSetPtr &blueBytes,
                     const BMPInfo &outputInfo) {
  const uint32_t imageSize = checkedBMP24ImageSize(outputInfo);
  const uint64_t pixelCount =
      static_cast<uint64_t>(outputInfo.width) * outputInfo.height;
  if (redBytes.length() != pixelCount ||
      greenBytes.length() != pixelCount ||
      blueBytes.length() != pixelCount) {
    throw std::runtime_error(
        "BMP output: channel buffer lengths do not match the image dimensions");
  }
  if (pixelCount != 0 &&
      (redBytes.data() == nullptr || greenBytes.data() == nullptr ||
       blueBytes.data() == nullptr)) {
    throw std::runtime_error("BMP output: channel buffer is null");
  }

  const uint32_t rowStride = getBMP24RowStride(outputInfo.width);
  std::vector<uint8_t> bmpPixelData(imageSize, 0u);
  const uint8_t *red = redBytes.data();
  const uint8_t *green = greenBytes.data();
  const uint8_t *blue = blueBytes.data();
  for (uint32_t row = 0; row < outputInfo.height; ++row) {
    const std::size_t inputRow =
        static_cast<std::size_t>(row) * outputInfo.width;
    const std::size_t outputRow =
        static_cast<std::size_t>(row) * rowStride;
    for (uint32_t column = 0; column < outputInfo.width; ++column) {
      const std::size_t inputPixel = inputRow + column;
      const std::size_t outputPixel =
          outputRow + column * OutputColorChannelCount;
      bmpPixelData[outputPixel] = blue[inputPixel];
      bmpPixelData[outputPixel + 1u] = green[inputPixel];
      bmpPixelData[outputPixel + 2u] = red[inputPixel];
    }
  }
  return bmpPixelData;
}

void writeBMP24(const std::string &outputPath,
                const std::vector<uint8_t> &bmpPixelData,
                const BMPInfo &outputInfo) {
  if (outputPath.empty()) {
    throw std::runtime_error("BMP output: output path is empty");
  }

  const uint32_t imageSize = checkedBMP24ImageSize(outputInfo);
  if (bmpPixelData.size() != imageSize) {
    throw std::runtime_error(
        "BMP output: pixel buffer length is " +
        std::to_string(bmpPixelData.size()) + ", expected " +
        std::to_string(imageSize));
  }

  BMPFileHeader fileHeader{};
  fileHeader.signature[0] = 'B';
  fileHeader.signature[1] = 'M';
  fileHeader.fileSize = BMPHeaderBytes + imageSize;
  fileHeader.dataOffset = BMPHeaderBytes;

  BMPInfoHeader infoHeader{};
  infoHeader.size = sizeof(BMPInfoHeader);
  infoHeader.width = static_cast<int32_t>(outputInfo.width);
  const int32_t height = static_cast<int32_t>(outputInfo.height);
  infoHeader.height = outputInfo.rowsBottomUp ? height : -height;
  infoHeader.planes = RequiredBMPPlanes;
  infoHeader.bitsPerPixel = OutputBitsPerPixel;
  infoHeader.imageSize = imageSize;

  const int fd =
      ::open(outputPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    const int error = errno;
    throw std::runtime_error("BMP output: failed to open " + outputPath +
                             ": " + std::strerror(error));
  }
  ScopedFileDescriptor output(fd);

  writeAll(output.get(), &fileHeader, sizeof(fileHeader), outputPath);
  writeAll(output.get(), &infoHeader, sizeof(infoHeader), outputPath);
  writeAll(output.get(), bmpPixelData.data(), imageSize, outputPath);
  output.closeOrThrow(outputPath);
}

void readBMPHeader(int fd, BMPInfo &info) {
  BMPFileHeader fileHeader{};
  readExact(fd, &fileHeader, sizeof(fileHeader), "file header");
  if (fileHeader.signature[0] != 'B' || fileHeader.signature[1] != 'M') {
    throw std::runtime_error("BMP: file signature does not match BMP format");
  }

  BMPInfoHeader infoHeader{};
  readExact(fd, &infoHeader, sizeof(infoHeader), "info header");
  if (infoHeader.size != BMPInfoHeaderSize) {
    throw std::runtime_error("BMP: unsupported header (size != 40)");
  }
  if (infoHeader.planes != RequiredBMPPlanes) {
    throw std::runtime_error("BMP: invalid plane count, got planes=" +
                             std::to_string(infoHeader.planes));
  }
  if (infoHeader.bitsPerPixel != SupportedBitsPerPixel) {
    throw std::runtime_error(
        "BMP: only 8-bit BMPs are supported, got bitsPerPixel=" +
        std::to_string(infoHeader.bitsPerPixel));
  }
  if (infoHeader.compression != 0) {
    throw std::runtime_error(
        "BMP: only uncompressed BMPs (BI_RGB) are supported, got compression=" +
        std::to_string(infoHeader.compression));
  }
  if (infoHeader.width <= 0) {
    throw std::runtime_error("BMP: zero or negative width is not supported");
  }
  if (infoHeader.height == 0) {
    throw std::runtime_error("BMP: zero height is not supported");
  }

  info.width = static_cast<uint32_t>(infoHeader.width);
  const int64_t signedHeight = infoHeader.height;
  info.height = static_cast<uint32_t>(
      signedHeight < 0 ? -signedHeight : signedHeight);
  info.rowStride = static_cast<uint32_t>(alignBMPRow(info.width));
  info.pixelOffset = fileHeader.dataOffset;
  info.rowsBottomUp = infoHeader.height > 0;

  readPalette(fd, infoHeader, info);

  if (::lseek(fd, static_cast<off_t>(info.pixelOffset), SEEK_SET) ==
      static_cast<off_t>(-1)) {
    throw std::runtime_error("BMP: failed to seek to pixel data at offset " +
                             std::to_string(info.pixelOffset));
  }
}

} // namespace image
