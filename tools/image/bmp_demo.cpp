/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_io.h>
#include <image/bmp_pipeline.h>
#include <image/conv_filter.h>

#include <kernel/core/attributes.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input.bmp>"),
                                            llvm::cl::Required);
static llvm::cl::opt<unsigned> cropWidth("crop-width",
                                         llvm::cl::desc("crop width"),
                                         llvm::cl::Required);
static llvm::cl::opt<unsigned> cropHeight("crop-height",
                                          llvm::cl::desc("crop height"),
                                          llvm::cl::Required);
static llvm::cl::opt<unsigned> cropX("crop-x",
                                     llvm::cl::desc("crop top-left x"),
                                     llvm::cl::Required);
static llvm::cl::opt<unsigned> cropY("crop-y",
                                     llvm::cl::desc("crop top-left y"),
                                     llvm::cl::Required);
static llvm::cl::opt<std::string>
    outputFile("o",
               llvm::cl::desc("write the blurred crop as a 24-bit BMP"),
               llvm::cl::value_desc("blurred.bmp"), llvm::cl::Required);

typedef void (*BmpPipelineFn)(kernel::StreamSetPtr &redBytes,
                              kernel::StreamSetPtr &greenBytes,
                              kernel::StreamSetPtr &blueBytes,
                              uint32_t fd);

static image::BMPInfo makeCroppedInfo(const image::BMPInfo &info,
                                      const uint32_t cropW,
                                      const uint32_t cropH) {
  image::BMPInfo croppedInfo = info;
  croppedInfo.width = cropW;
  croppedInfo.height = cropH;
  croppedInfo.rowStride = ((cropW + 3u) / 4u) * 4u;
  return croppedInfo;
}

BmpPipelineFn buildPipeline(CPUDriver &driver, const image::BMPInfo &info,
                            const uint32_t cropW, const uint32_t cropH,
                            const uint32_t cropLeft, const uint32_t cropTop) {
  auto P = kernel::CreatePipeline(
      driver,
      kernel::Output<kernel::streamset_t>{"redBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"greenBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"blueBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Input<uint32_t>{"fd"});

  kernel::Scalar *fdScalar = P.getInputScalar("fd");

  kernel::StreamSet *colorStream = nullptr;
  image::ParseBMPColorStreams(P, fdScalar, info, colorStream);

  kernel::StreamSet *croppedColorStream = nullptr;
  image::CropImage(P, colorStream, info, cropW, cropH, cropLeft, cropTop,
                   croppedColorStream);
  image::CreateBMPColorByteStreams(
      P, croppedColorStream, P.getOutputStreamSet("redBytes"),
      P.getOutputStreamSet("greenBytes"), P.getOutputStreamSet("blueBytes"));

  return P.compile();
}

// Combine separate R/G/B byte streams (in stored row order) into interleaved
// RGB bytes in logical top-down row order, which is what applyConvFilter
// expects.
static void interleaveRGB(const kernel::StreamSetPtr &redBytes,
                          const kernel::StreamSetPtr &greenBytes,
                          const kernel::StreamSetPtr &blueBytes,
                          const image::BMPInfo &info,
                          std::vector<uint8_t> &rgb) {
  const uint8_t *r = redBytes.data();
  const uint8_t *g = greenBytes.data();
  const uint8_t *b = blueBytes.data();
  for (std::size_t outputRow = 0; outputRow < info.height; ++outputRow) {
    const std::size_t inputRow =
        info.rowsBottomUp ? info.height - outputRow - 1 : outputRow;
    for (std::size_t column = 0; column < info.width; ++column) {
      const std::size_t inputPixel = inputRow * info.width + column;
      const std::size_t outputPixel = outputRow * info.width + column;
      rgb[outputPixel * 3 + 0] = r[inputPixel];
      rgb[outputPixel * 3 + 1] = g[inputPixel];
      rgb[outputPixel * 3 + 2] = b[inputPixel];
    }
  }
}

// Split interleaved top-down RGB back into separate R/G/B byte streams in
// stored row order, which is what createBMP24PixelData expects.
static void splitChannels(const std::vector<uint8_t> &rgb,
                          const image::BMPInfo &info,
                          std::vector<uint8_t> &red,
                          std::vector<uint8_t> &green,
                          std::vector<uint8_t> &blue) {
  const std::size_t pixelCount =
      static_cast<std::size_t>(info.width) * info.height;
  red.resize(pixelCount);
  green.resize(pixelCount);
  blue.resize(pixelCount);

  for (std::size_t storedRow = 0; storedRow < info.height; ++storedRow) {
    const std::size_t inputRow =
        info.rowsBottomUp ? info.height - storedRow - 1u : storedRow;
    for (std::size_t column = 0; column < info.width; ++column) {
      const std::size_t inputPixel = inputRow * info.width + column;
      const std::size_t outputPixel = storedRow * info.width + column;
      red[outputPixel] = rgb[inputPixel * 3u];
      green[outputPixel] = rgb[inputPixel * 3u + 1u];
      blue[outputPixel] = rgb[inputPixel * 3u + 2u];
    }
  }
}

int main(int argc, char **argv) {
  codegen::ParseCommandLineOptions(
      argc, argv,
      {&codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});

  int fd = open(inputFile.c_str(), O_RDONLY);
  if (fd < 0) {
    std::perror("open");
    return 1;
  }

  image::BMPInfo info;
  try {
    image::readBMPHeader(fd, info);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    close(fd);
    return 2;
  }

  std::cout << "=== BMP Header ===\n";
  std::cout << "width        = " << info.width << "\n";
  std::cout << "height       = " << info.height << "\n";
  std::cout << "rowStride    = " << info.rowStride << "\n";
  std::cout << "pixelOffset  = " << info.pixelOffset << "\n";
  std::cout << "rowsBottomUp = " << info.rowsBottomUp << "\n";
  std::cout << "total pixels = " << (info.width * info.height) << "\n";

  CPUDriver driver("bmp_demo");
  BmpPipelineFn pipelineFn;
  try {
    pipelineFn =
        buildPipeline(driver, info, cropWidth, cropHeight, cropX, cropY);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    close(fd);
    return 3;
  }

  kernel::StreamSetPtr redBytes;
  kernel::StreamSetPtr greenBytes;
  kernel::StreamSetPtr blueBytes;
  pipelineFn(redBytes, greenBytes, blueBytes, static_cast<uint32_t>(fd));
  close(fd);

  const image::BMPInfo croppedInfo =
      makeCroppedInfo(info, cropWidth, cropHeight);
  const uint64_t croppedPixels =
      static_cast<uint64_t>(cropWidth) * cropHeight;

  uint64_t croppedBrightPixels = 0;
  const uint8_t *croppedRed = redBytes.data();
  for (uint64_t i = 0; i < redBytes.length(); ++i) {
    if (croppedRed[i] >= 128) {
      ++croppedBrightPixels;
    }
  }
  std::cout << "\n=== ParseBMPColorStreams output ===\n";
  std::cout << "crop rectangle = " << cropWidth << "x" << cropHeight
            << " at (" << cropX << ", " << cropY << ")\n";
  std::cout << "bright cropped red pixels (value >= 128) = "
            << croppedBrightPixels << " / " << croppedPixels << "\n";
  std::cout << "dark  cropped red pixels (value <  128) = "
            << (croppedPixels - croppedBrightPixels) << " / "
            << croppedPixels << "\n";

  const std::size_t croppedRGBBytes =
      static_cast<std::size_t>(cropWidth) * cropHeight * 3;
  std::vector<uint8_t> croppedRGB(croppedRGBBytes);
  interleaveRGB(redBytes, greenBytes, blueBytes, croppedInfo, croppedRGB);

  std::vector<uint8_t> outputRGB(croppedRGBBytes);
  const float weights[] = {1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                           1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f};
  kernel::image::ConvFilterConfig config{
      kernel::image::ConvFilterMode::Default,
      croppedInfo.width,
      croppedInfo.height,
      3,
      3,
      weights,
      9,
  };
  if (!kernel::image::applyConvFilter(croppedRGB.data(), outputRGB.data(),
                                      config)) {
    std::cerr << "applyConvFilter failed\n";
    return 4;
  }

  std::cout << "\n=== ConvFilter output ===\n";
  if (!outputRGB.empty()) {
    std::cout << "first cropped pixel RGB = "
              << static_cast<unsigned>(outputRGB[0]) << ", "
              << static_cast<unsigned>(outputRGB[1]) << ", "
              << static_cast<unsigned>(outputRGB[2]) << "\n";
  }

  std::vector<uint8_t> blurredRed;
  std::vector<uint8_t> blurredGreen;
  std::vector<uint8_t> blurredBlue;
  splitChannels(outputRGB, croppedInfo, blurredRed, blurredGreen, blurredBlue);

  kernel::StreamSetPtr blurredRedBytes(blurredRed.data(), blurredRed.size());
  kernel::StreamSetPtr blurredGreenBytes(blurredGreen.data(),
                                         blurredGreen.size());
  kernel::StreamSetPtr blurredBlueBytes(blurredBlue.data(), blurredBlue.size());

  try {
    std::vector<uint8_t> bmpPixelData = image::createBMP24PixelData(
        blurredRedBytes, blurredGreenBytes, blurredBlueBytes, croppedInfo);
    image::writeBMP24(outputFile, bmpPixelData, croppedInfo);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 5;
  }
  std::cout << "wrote blurred BMP = " << outputFile << "\n";

  return 0;
}
