/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_loader.h>

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

#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/stream_select.h>
#include <toolchain/toolchain.h>

#include <pablo/bixnum/bixnum.h>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>

#define SHOW_BIXNUM(name)                                                      \
  if (codegen::EnableIllustrator)                                              \
  P.captureBixNum(#name, name)
#define SHOW_BYTES(name)                                                       \
  if (codegen::EnableIllustrator)                                              \
  P.captureByteData(#name, name)

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

namespace {

constexpr uint16_t RequiredBMPPlanes = 1u;
constexpr uint16_t SupportedBitsPerPixel = 8u;
constexpr uint32_t BMPInfoHeaderSize = 40u;
constexpr uint32_t BMPRowAlignment = 4u;
constexpr uint32_t PaletteSize = 256u;
constexpr uint32_t PaletteEntryBytes = 4u;
constexpr unsigned ChannelBits = 8u;
constexpr unsigned ColorChannelCount = 3u;
constexpr unsigned ColorStreamCount = ColorChannelCount * ChannelBits;
constexpr uint16_t OutputBitsPerPixel = 24u;
constexpr uint32_t BMPHeaderBytes =
    sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

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

void writeAll(int fd, const void *data, std::size_t size,
              const std::string &outputPath) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  while (size != 0) {
    const ssize_t written = write(fd, bytes, size);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("BMP output: failed to write " + outputPath +
                               ": " + std::strerror(errno));
    }
    if (written == 0) {
      throw std::runtime_error("BMP output: short write to " + outputPath);
    }
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
}

// Parse each pixel BGR value
void readPalette(int fd, const BMPInfoHeader &ih, BMPInfo &info) {
  info.numColors = (ih.colorsUsed != 0) ? ih.colorsUsed : PaletteSize;
  if (info.numColors > PaletteSize) {
    info.numColors = PaletteSize;
  }
  info.paletteOffset = static_cast<uint32_t>(sizeof(BMPFileHeader)) + ih.size;

  const uint32_t paletteBytes = PaletteEntryBytes * info.numColors;
  if (info.paletteOffset + paletteBytes > info.pixelOffset) {
    throw std::runtime_error("BMP: palette region overflows pixel data offset");
  }

  std::vector<uint8_t> palBuf(paletteBytes);
  ssize_t palRead = read(fd, palBuf.data(), paletteBytes);
  if (palRead != static_cast<ssize_t>(paletteBytes)) {
    throw std::runtime_error("BMP: failed to read color table");
  }

  info.bTable.assign(PaletteSize, 0u);
  info.gTable.assign(PaletteSize, 0u);
  info.rTable.assign(PaletteSize, 0u);
  for (uint32_t p = 0; p < info.numColors; ++p) {
    const uint32_t paletteEntry = PaletteEntryBytes * p;
    info.bTable[p] = palBuf[paletteEntry + 0u]; // B
    info.gTable[p] = palBuf[paletteEntry + 1u]; // G
    info.rTable[p] = palBuf[paletteEntry + 2u]; // R
    // Reserve byte ignored
  }
}

} // anonymous namespace

uint32_t getBMP24RowStride(uint32_t width) {
  if (width == 0 ||
      width > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    throw std::runtime_error("BMP output: width is outside the BMP range");
  }
  const uint64_t rowBytes =
      static_cast<uint64_t>(width) * ColorChannelCount;
  const uint64_t rowStride =
      ((rowBytes + BMPRowAlignment - 1u) / BMPRowAlignment) *
      BMPRowAlignment;
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
      const std::size_t outputPixel = outputRow + column * ColorChannelCount;
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
      open(outputPath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    throw std::runtime_error("BMP output: failed to open " + outputPath +
                             ": " + std::strerror(errno));
  }

  try {
    writeAll(fd, &fileHeader, sizeof(fileHeader), outputPath);
    writeAll(fd, &infoHeader, sizeof(infoHeader), outputPath);
    writeAll(fd, bmpPixelData.data(), imageSize, outputPath);
  } catch (...) {
    close(fd);
    throw;
  }
  if (close(fd) != 0) {
    throw std::runtime_error("BMP output: failed to close " + outputPath +
                             ": " + std::strerror(errno));
  }
}

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
  if (ih.size != BMPInfoHeaderSize) {
    throw std::runtime_error("BMP: unsupported header (size != 40)");
  }
  if (ih.planes != RequiredBMPPlanes) {
    throw std::runtime_error("BMP: invalid plane count, got planes=" +
                             std::to_string(ih.planes));
  }
  if (ih.bitsPerPixel != SupportedBitsPerPixel) {
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
      ((info.width + BMPRowAlignment - 1u) / BMPRowAlignment) * BMPRowAlignment;
  info.pixelOffset = fh.dataOffset;
  info.rowsBottomUp = (ih.height > 0);

  if (info.height == 0) {
    throw std::runtime_error("BMP: zero height is not supported");
  }

  readPalette(fd, ih, info);

  // Seek to the first pixel byte.
  if (lseek(fd, static_cast<off_t>(info.pixelOffset), SEEK_SET) ==
      static_cast<off_t>(-1)) {
    throw std::runtime_error("BMP: failed to seek to pixel data at offset " +
                             std::to_string(info.pixelOffset));
  }
}

void ParseBMPBuffer(kernel::ProgramBuilder &P, kernel::Scalar *fileDescriptor,
                    const BMPInfo &info, kernel::StreamSet *&pixelStream,
                    kernel::StreamSet *&basisBits) {

  kernel::StreamSet *byteStream = P.CreateStreamSet(1, SupportedBitsPerPixel);
  P.CreateKernelCall<kernel::ReadSourceKernel>(fileDescriptor, byteStream);
  SHOW_BYTES(byteStream);

  if (info.rowStride == info.width) {
    // No padding - use the raw byte stream directly.
    pixelStream = byteStream;
  } else {
    std::vector<uint64_t> rowPat(info.rowStride, 0u);
    // mask location of pixel data
    std::fill_n(rowPat.begin(), info.width, 1u);
    kernel::StreamSet *rowMask = P.CreateRepeatingStreamSet(1, rowPat);
    pixelStream = P.CreateStreamSet(1, SupportedBitsPerPixel);
    FilterByMask(P, rowMask, byteStream, pixelStream); // get rid of padding
  }
  SHOW_BYTES(pixelStream);

  basisBits = P.CreateStreamSet(SupportedBitsPerPixel);
  P.CreateKernelCall<kernel::S2PKernel>(pixelStream, basisBits);
  SHOW_BIXNUM(basisBits);
}

// ---------------------------------------------------------------------------
// PaletteLUTKernel
//
// Maps an 8-bit pixel-index BixNum (8x1 StreamSet) to a 24-stream BixNum
// (24x1 StreamSet) using three compile-time 256-entry palette tables:
//   streams  0.. 7  = Blue  channel
//   streams  8..15  = Green channel
//   streams 16..23  = Red   channel
//
// Each channel is compiled separately via BixNumTableCompiler, which
// generates bit-parallel boolean logic over the 8 index bit-streams.
// ---------------------------------------------------------------------------
namespace {

class PaletteLUTKernel final : public pablo::PabloKernel {
public:
  PaletteLUTKernel(LLVMTypeSystemInterface &ts, kernel::StreamSet *index,
                   kernel::StreamSet *color, std::vector<unsigned> bTable,
                   std::vector<unsigned> gTable, std::vector<unsigned> rTable)
      : pablo::PabloKernel(ts, "palette_lut", {kernel::Binding{"index", index}},
                           {kernel::Binding{"color", color}}),
        mB(std::move(bTable)), mG(std::move(gTable)), mR(std::move(rTable)) {}

protected:
  void generatePabloMethod() override;

private:
  std::vector<unsigned> mB;
  std::vector<unsigned> mG;
  std::vector<unsigned> mR;
};

void PaletteLUTKernel::generatePabloMethod() {
  pablo::PabloBuilder pb(getEntryScope());
  pablo::BixNum idx = getInputStreamSet("index"); // 8-bit index BixNum
  pablo::Var *out = getOutputStreamVar("color");  // 24-stream output

  // Compile one channel's 256-entry table into 8 contiguous output streams
  // starting at stream index baseStream.
  auto emitChannel = [&](unsigned baseStream, std::vector<unsigned> &tbl) {
    pablo::BixVar chan(ChannelBits);
    for (unsigned i = 0; i < ChannelBits; ++i) {
      chan[i] = pb.createVar("ch" + std::to_string(baseStream + i),
                             pb.createZeroes());
    }
    pablo::BixNumTableCompiler c(tbl, idx, chan);
    std::vector<unsigned> partitionLevels{ChannelBits};
    c.setRecursivePartitionLevels(partitionLevels);
    c.compileSubTable(pb, 0, pb.createOnes()); // full 0..255 range
    for (unsigned i = 0; i < ChannelBits; ++i) {
      pb.createAssign(pb.createExtract(out, pb.getInteger(baseStream + i)),
                      chan[i]);
    }
  };

  emitChannel(0, mB);               // streams  0.. 7 = Blue
  emitChannel(ChannelBits, mG);     // streams  8..15 = Green
  emitChannel(2 * ChannelBits, mR); // streams 16..23 = Red
}

} // anonymous namespace

void ParseBMPColorStreams(kernel::ProgramBuilder &P,
                          kernel::Scalar *fileDescriptor, const BMPInfo &info,
                          kernel::StreamSet *&colorStream) {
  kernel::StreamSet *pixelStream = nullptr;
  kernel::StreamSet *basisBits = nullptr;
  ParseBMPBuffer(P, fileDescriptor, info, pixelStream, basisBits);

  colorStream = P.CreateStreamSet(ColorStreamCount);
  P.CreateKernelCall<PaletteLUTKernel>(basisBits, colorStream, info.bTable,
                                       info.gTable, info.rTable);
  SHOW_BIXNUM(colorStream);
}

void CropImage(kernel::ProgramBuilder &P, kernel::StreamSet *sourceImageData,
               const BMPInfo &sourceInfo, uint32_t cropWidth,
               uint32_t cropHeight, uint32_t cropX, uint32_t cropY,
               kernel::StreamSet *&croppedImageData) {
  if (sourceImageData->getNumElements() != ColorStreamCount ||
      sourceImageData->getFieldWidth() != 1) {
    throw std::runtime_error("BMP crop: source image data must be a 24x1 "
                             "B/G/R color stream");
  }
  if (cropWidth == 0 || cropHeight == 0) {
    throw std::runtime_error("BMP crop: zero-sized crop is not supported");
  }
  if (cropX > sourceInfo.width || cropWidth > sourceInfo.width - cropX) {
    throw std::runtime_error("BMP crop: crop rectangle exceeds source width");
  }
  if (cropY > sourceInfo.height || cropHeight > sourceInfo.height - cropY) {
    throw std::runtime_error("BMP crop: crop rectangle exceeds source height");
  }

  const uint32_t cropRight = cropX + cropWidth;
  const uint32_t cropBottom = cropY + cropHeight;
  const uint64_t sourcePixels =
      static_cast<uint64_t>(sourceInfo.width) * sourceInfo.height;
  std::vector<uint64_t> cropPattern(sourcePixels, 0u);

  for (uint32_t storedRow = 0; storedRow < sourceInfo.height; ++storedRow) {
    const uint32_t logicalRow =
        sourceInfo.rowsBottomUp ? sourceInfo.height - storedRow - 1u
                                : storedRow;
    if (logicalRow < cropY || logicalRow >= cropBottom) {
      continue;
    }
    const uint64_t rowBase =
        static_cast<uint64_t>(storedRow) * sourceInfo.width;
    std::fill_n(cropPattern.begin() + rowBase + cropX, cropRight - cropX, 1u);
  }

  kernel::StreamSet *cropMask =
      P.CreateRepeatingStreamSet(1, cropPattern, false);
  kernel::StreamSet *blue = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(0, 8));
  kernel::StreamSet *green = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(8, 16));
  kernel::StreamSet *red = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(16, 24));
  kernel::StreamSet *croppedBlue = P.CreateStreamSet(ChannelBits);
  kernel::StreamSet *croppedGreen = P.CreateStreamSet(ChannelBits);
  kernel::StreamSet *croppedRed = P.CreateStreamSet(ChannelBits);
  FilterByMask(P, cropMask, blue, croppedBlue);
  FilterByMask(P, cropMask, green, croppedGreen);
  FilterByMask(P, cropMask, red, croppedRed);
  croppedImageData =
      kernel::streamutils::Select(P, {croppedBlue, croppedGreen, croppedRed});
  SHOW_BIXNUM(croppedImageData);
}

void CreateBMPColorByteStreams(kernel::ProgramBuilder &P,
                               kernel::StreamSet *sourceImageData,
                               kernel::StreamSet *redBytes,
                               kernel::StreamSet *greenBytes,
                               kernel::StreamSet *blueBytes) {
  if (sourceImageData->getNumElements() != ColorStreamCount ||
      sourceImageData->getFieldWidth() != 1) {
    throw std::runtime_error("BMP output: source image data must be a 24x1 "
                             "B/G/R color stream");
  }
  const auto isByteStream = [](const kernel::StreamSet *stream) {
    return stream->getNumElements() == 1 &&
           stream->getFieldWidth() == ChannelBits;
  };
  if (!isByteStream(redBytes) || !isByteStream(greenBytes) ||
      !isByteStream(blueBytes)) {
    throw std::runtime_error("BMP output: channel outputs must be 1x8 byte "
                             "streams");
  }

  kernel::StreamSet *blueBasis = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(0, 8));
  kernel::StreamSet *greenBasis = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(8, 16));
  kernel::StreamSet *redBasis = kernel::streamutils::Select(
      P, sourceImageData, kernel::streamutils::Range(16, 24));
  P.CreateKernelCall<kernel::P2SKernel>(blueBasis, blueBytes);
  P.CreateKernelCall<kernel::P2SKernel>(greenBasis, greenBytes);
  P.CreateKernelCall<kernel::P2SKernel>(redBasis, redBytes);
  SHOW_BYTES(redBytes);
  SHOW_BYTES(greenBytes);
  SHOW_BYTES(blueBytes);
}

} // namespace image
