/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_pipeline.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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
namespace {

constexpr unsigned ChannelBits = 8u;
constexpr unsigned ColorChannelCount = 3u;
constexpr unsigned ColorStreamCount = ColorChannelCount * ChannelBits;
constexpr unsigned BlueStreamBase = 0u;
constexpr unsigned GreenStreamBase = BlueStreamBase + ChannelBits;
constexpr unsigned RedStreamBase = GreenStreamBase + ChannelBits;

struct ColorChannels {
  kernel::StreamSet *blue;
  kernel::StreamSet *green;
  kernel::StreamSet *red;
};

void requireColorStream(const kernel::StreamSet *stream,
                        const std::string &operation) {
  if (stream == nullptr || stream->getNumElements() != ColorStreamCount ||
      stream->getFieldWidth() != 1) {
    throw std::runtime_error(operation +
                             ": source image data must be a 24x1 B/G/R "
                             "color stream");
  }
}

bool isByteStream(const kernel::StreamSet *stream) {
  return stream != nullptr && stream->getNumElements() == 1 &&
         stream->getFieldWidth() == ChannelBits;
}

ColorChannels selectColorChannels(kernel::ProgramBuilder &P,
                                  kernel::StreamSet *colorStream,
                                  const std::string &operation) {
  requireColorStream(colorStream, operation);
  return {
      kernel::streamutils::Select(
          P, colorStream,
          kernel::streamutils::Range(BlueStreamBase,
                                     BlueStreamBase + ChannelBits)),
      kernel::streamutils::Select(
          P, colorStream,
          kernel::streamutils::Range(GreenStreamBase,
                                     GreenStreamBase + ChannelBits)),
      kernel::streamutils::Select(
          P, colorStream,
          kernel::streamutils::Range(RedStreamBase,
                                     RedStreamBase + ChannelBits)),
  };
}

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
  pablo::BixNum index = getInputStreamSet("index");
  pablo::Var *output = getOutputStreamVar("color");

  auto emitChannel = [&](unsigned baseStream, std::vector<unsigned> &table) {
    pablo::BixVar channel(ChannelBits);
    for (unsigned bit = 0; bit < ChannelBits; ++bit) {
      channel[bit] =
          pb.createVar("ch" + std::to_string(baseStream + bit),
                       pb.createZeroes());
    }
    pablo::BixNumTableCompiler compiler(table, index, channel);
    std::vector<unsigned> partitionLevels{ChannelBits};
    compiler.setRecursivePartitionLevels(partitionLevels);
    compiler.compileSubTable(pb, 0, pb.createOnes());
    for (unsigned bit = 0; bit < ChannelBits; ++bit) {
      pb.createAssign(
          pb.createExtract(output, pb.getInteger(baseStream + bit)),
          channel[bit]);
    }
  };

  emitChannel(BlueStreamBase, mB);
  emitChannel(GreenStreamBase, mG);
  emitChannel(RedStreamBase, mR);
}

} // namespace

void ParseBMPBuffer(kernel::ProgramBuilder &P, kernel::Scalar *fileDescriptor,
                    const BMPInfo &info, kernel::StreamSet *&pixelStream,
                    kernel::StreamSet *&basisBits) {
  kernel::StreamSet *byteStream = P.CreateStreamSet(1, ChannelBits);
  P.CreateKernelCall<kernel::ReadSourceKernel>(fileDescriptor, byteStream);
  SHOW_BYTES(byteStream);

  if (info.rowStride == info.width) {
    pixelStream = byteStream;
  } else {
    std::vector<uint64_t> rowPattern(info.rowStride, 0u);
    std::fill_n(rowPattern.begin(), info.width, 1u);
    kernel::StreamSet *rowMask =
        P.CreateRepeatingStreamSet(1, rowPattern);
    pixelStream = P.CreateStreamSet(1, ChannelBits);
    FilterByMask(P, rowMask, byteStream, pixelStream);
  }
  SHOW_BYTES(pixelStream);

  basisBits = P.CreateStreamSet(ChannelBits);
  P.CreateKernelCall<kernel::S2PKernel>(pixelStream, basisBits);
  SHOW_BIXNUM(basisBits);
}

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
  const ColorChannels source =
      selectColorChannels(P, sourceImageData, "BMP crop");
  if (cropWidth == 0 || cropHeight == 0) {
    throw std::runtime_error("BMP crop: zero-sized crop is not supported");
  }
  if (cropX > sourceInfo.width || cropWidth > sourceInfo.width - cropX) {
    throw std::runtime_error("BMP crop: crop rectangle exceeds source width");
  }
  if (cropY > sourceInfo.height || cropHeight > sourceInfo.height - cropY) {
    throw std::runtime_error("BMP crop: crop rectangle exceeds source height");
  }

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
    const std::size_t rowStart =
        static_cast<std::size_t>(storedRow) * sourceInfo.width + cropX;
    std::fill_n(cropPattern.begin() + rowStart, cropWidth, 1u);
  }

  kernel::StreamSet *cropMask =
      P.CreateRepeatingStreamSet(1, cropPattern);

  ColorChannels sourceBytes{
      P.CreateStreamSet(1, ChannelBits),
      P.CreateStreamSet(1, ChannelBits),
      P.CreateStreamSet(1, ChannelBits),
  };
  P.CreateKernelCall<kernel::P2SKernel>(source.blue, sourceBytes.blue);
  P.CreateKernelCall<kernel::P2SKernel>(source.green, sourceBytes.green);
  P.CreateKernelCall<kernel::P2SKernel>(source.red, sourceBytes.red);

  ColorChannels croppedBytes{
      P.CreateStreamSet(1, ChannelBits),
      P.CreateStreamSet(1, ChannelBits),
      P.CreateStreamSet(1, ChannelBits),
  };
  FilterByMask(P, cropMask, sourceBytes.blue, croppedBytes.blue);
  FilterByMask(P, cropMask, sourceBytes.green, croppedBytes.green);
  FilterByMask(P, cropMask, sourceBytes.red, croppedBytes.red);

  ColorChannels croppedBasis{
      P.CreateStreamSet(ChannelBits),
      P.CreateStreamSet(ChannelBits),
      P.CreateStreamSet(ChannelBits),
  };
  P.CreateKernelCall<kernel::S2PKernel>(croppedBytes.blue, croppedBasis.blue);
  P.CreateKernelCall<kernel::S2PKernel>(croppedBytes.green, croppedBasis.green);
  P.CreateKernelCall<kernel::S2PKernel>(croppedBytes.red, croppedBasis.red);
  croppedImageData = kernel::streamutils::Select(
      P, {croppedBasis.blue, croppedBasis.green, croppedBasis.red});
  SHOW_BIXNUM(croppedImageData);
}

void CreateBMPColorByteStreams(kernel::ProgramBuilder &P,
                               kernel::StreamSet *sourceImageData,
                               kernel::StreamSet *redBytes,
                               kernel::StreamSet *greenBytes,
                               kernel::StreamSet *blueBytes) {
  const ColorChannels source =
      selectColorChannels(P, sourceImageData, "BMP output");
  if (!isByteStream(redBytes) || !isByteStream(greenBytes) ||
      !isByteStream(blueBytes)) {
    throw std::runtime_error("BMP output: channel outputs must be 1x8 byte "
                             "streams");
  }

  P.CreateKernelCall<kernel::P2SKernel>(source.blue, blueBytes);
  P.CreateKernelCall<kernel::P2SKernel>(source.green, greenBytes);
  P.CreateKernelCall<kernel::P2SKernel>(source.red, redBytes);
  SHOW_BYTES(redBytes);
  SHOW_BYTES(greenBytes);
  SHOW_BYTES(blueBytes);
}

} // namespace image
