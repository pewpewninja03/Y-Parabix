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
                              kernel::StreamSetPtr &croppedRedBytes,
                              kernel::StreamSetPtr &croppedGreenBytes,
                              kernel::StreamSetPtr &croppedBlueBytes,
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
      driver, kernel::Output<kernel::streamset_t>{"redBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"greenBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"blueBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"croppedRedBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"croppedGreenBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Output<kernel::streamset_t>{"croppedBlueBytes", 1, 8,
                                           kernel::ReturnedBuffer(1)},
      kernel::Input<uint32_t>{"fd"});

  kernel::Scalar *fdScalar = P.getInputScalar("fd");

  kernel::StreamSet *colorStream = nullptr;
  image::ParseBMPColorStreams(P, fdScalar, info, colorStream);

  image::CreateBMPColorByteStreams(
      P, colorStream, P.getOutputStreamSet("redBytes"),
      P.getOutputStreamSet("greenBytes"), P.getOutputStreamSet("blueBytes"));

  kernel::StreamSet *croppedColorStream = nullptr;
  image::CropImage(P, colorStream, info, cropW, cropH, cropLeft, cropTop,
                   croppedColorStream);
  image::CreateBMPColorByteStreams(
      P, croppedColorStream, P.getOutputStreamSet("croppedRedBytes"),
      P.getOutputStreamSet("croppedGreenBytes"),
      P.getOutputStreamSet("croppedBlueBytes"));

  return P.compile();
}

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

static void splitRGBToStoredChannels(const std::vector<uint8_t> &rgb,
                                     const image::BMPInfo &info,
                                     std::vector<uint8_t> &red,
                                     std::vector<uint8_t> &green,
                                     std::vector<uint8_t> &blue) {
  const std::size_t pixelCount =
      static_cast<std::size_t>(info.width) * info.height;
  if (rgb.size() != pixelCount * 3u) {
    throw std::runtime_error(
        "BMP output: RGB buffer length does not match the image dimensions");
  }
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

static bool compareCroppedRGB(const std::vector<uint8_t> &fullRGB,
                              const std::vector<uint8_t> &croppedRGB,
                              const image::BMPInfo &info,
                              const uint32_t cropW, const uint32_t cropH,
                              const uint32_t cropLeft,
                              const uint32_t cropTop) {
  for (uint32_t row = 0; row < cropH; ++row) {
    for (uint32_t col = 0; col < cropW; ++col) {
      const std::size_t fullPixel =
          (static_cast<std::size_t>(cropTop + row) * info.width + cropLeft + col) * 3;
      const std::size_t cropPixel =
          (static_cast<std::size_t>(row) * cropW + col) * 3;
      for (std::size_t channel = 0; channel < 3; ++channel) {
        if (croppedRGB[cropPixel + channel] != fullRGB[fullPixel + channel]) {
          std::cerr << "crop mismatch at row=" << row << " col=" << col
                    << " channel=" << channel << ": expected "
                    << static_cast<unsigned>(fullRGB[fullPixel + channel])
                    << ", got "
                    << static_cast<unsigned>(croppedRGB[cropPixel + channel])
                    << "\n";
          return false;
        }
      }
    }
  }
  return true;
}

static bool verifyBMP24Pixels(const std::vector<uint8_t> &bmpPixelData,
                              const kernel::StreamSetPtr &redBytes,
                              const kernel::StreamSetPtr &greenBytes,
                              const kernel::StreamSetPtr &blueBytes,
                              const image::BMPInfo &info) {
  const uint32_t rowStride = image::getBMP24RowStride(info.width);
  const uint32_t rowBytes = info.width * 3u;
  const uint64_t expectedBytes =
      static_cast<uint64_t>(rowStride) * info.height;
  if (bmpPixelData.size() != expectedBytes) {
    std::cerr << "unexpected BMP pixel buffer length: "
              << bmpPixelData.size() << ", expected " << expectedBytes
              << "\n";
    return false;
  }

  const uint8_t *bmp = bmpPixelData.data();
  const uint8_t *red = redBytes.data();
  const uint8_t *green = greenBytes.data();
  const uint8_t *blue = blueBytes.data();
  for (uint32_t row = 0; row < info.height; ++row) {
    const std::size_t pixelRow =
        static_cast<std::size_t>(row) * info.width;
    const std::size_t bmpRow = static_cast<std::size_t>(row) * rowStride;
    for (uint32_t column = 0; column < info.width; ++column) {
      const std::size_t pixel = pixelRow + column;
      const std::size_t output = bmpRow + column * 3u;
      if (bmp[output] != blue[pixel] ||
          bmp[output + 1u] != green[pixel] ||
          bmp[output + 2u] != red[pixel]) {
        std::cerr << "BMP pixel mismatch at stored row=" << row
                  << " col=" << column << "\n";
        return false;
      }
    }
    for (uint32_t byte = rowBytes; byte < rowStride; ++byte) {
      if (bmp[bmpRow + byte] != 0) {
        std::cerr << "nonzero BMP padding at stored row=" << row
                  << " byte=" << byte << "\n";
        return false;
      }
    }
  }
  return true;
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

  if (cropWidth == 0 || cropHeight == 0 || cropX > info.width ||
      cropWidth > info.width - cropX || cropY > info.height ||
      cropHeight > info.height - cropY) {
    std::cerr << "invalid crop rectangle\n";
    close(fd);
    return 3;
  }

  CPUDriver driver("bmp_header_test");
  BmpPipelineFn pipelineFn =
      buildPipeline(driver, info, cropWidth, cropHeight, cropX, cropY);

  kernel::StreamSetPtr redBytes;
  kernel::StreamSetPtr greenBytes;
  kernel::StreamSetPtr blueBytes;
  kernel::StreamSetPtr croppedRedBytes;
  kernel::StreamSetPtr croppedGreenBytes;
  kernel::StreamSetPtr croppedBlueBytes;
  pipelineFn(redBytes, greenBytes, blueBytes, croppedRedBytes,
             croppedGreenBytes, croppedBlueBytes, static_cast<uint32_t>(fd));
  close(fd);

  const uint64_t totalPixels = static_cast<uint64_t>(info.width) * info.height;
  const uint64_t croppedPixels =
      static_cast<uint64_t>(cropWidth) * cropHeight;
  const image::BMPInfo croppedInfo =
      makeCroppedInfo(info, cropWidth, cropHeight);
  uint64_t croppedBrightPixels = 0;
  const uint8_t *croppedRed = croppedRedBytes.data();
  for (uint64_t i = 0; i < croppedRedBytes.length(); ++i) {
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

  if (redBytes.length() < totalPixels || greenBytes.length() < totalPixels ||
      blueBytes.length() < totalPixels ||
      croppedRedBytes.length() < croppedPixels ||
      croppedGreenBytes.length() < croppedPixels ||
      croppedBlueBytes.length() < croppedPixels) {
    std::cerr << "unexpected returned buffer length: full R/G/B="
              << redBytes.length() << "/" << greenBytes.length() << "/"
              << blueBytes.length() << ", cropped R/G/B="
              << croppedRedBytes.length() << "/" << croppedGreenBytes.length()
              << "/" << croppedBlueBytes.length() << ", expected at least full="
              << totalPixels << ", expected at least cropped=" << croppedPixels
              << "\n";
    return 4;
  }

  const std::size_t rgbBytes =
      static_cast<std::size_t>(info.width) * info.height * 3;
  std::vector<uint8_t> inputRGB(rgbBytes);
  interleaveRGB(redBytes, greenBytes, blueBytes, info, inputRGB);

  const std::size_t croppedRGBBytes =
      static_cast<std::size_t>(cropWidth) * cropHeight * 3;
  std::vector<uint8_t> croppedRGB(croppedRGBBytes);
  std::vector<uint8_t> outputRGB(croppedRGBBytes);
  interleaveRGB(croppedRedBytes, croppedGreenBytes, croppedBlueBytes,
                croppedInfo, croppedRGB);

  if (!compareCroppedRGB(inputRGB, croppedRGB, info, cropWidth, cropHeight,
                         cropX, cropY)) {
    return 5;
  }
  std::cout << "crop verification = ok\n";

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
    return 6;
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
  try {
    splitRGBToStoredChannels(outputRGB, croppedInfo, blurredRed, blurredGreen,
                             blurredBlue);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 7;
  }

  kernel::StreamSetPtr blurredRedBytes(blurredRed.data(), blurredRed.size());
  kernel::StreamSetPtr blurredGreenBytes(blurredGreen.data(),
                                         blurredGreen.size());
  kernel::StreamSetPtr blurredBlueBytes(blurredBlue.data(), blurredBlue.size());

  std::vector<uint8_t> bmpPixelData;
  try {
    bmpPixelData =
        image::createBMP24PixelData(blurredRedBytes, blurredGreenBytes,
                                    blurredBlueBytes, croppedInfo);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 8;
  }
  if (!verifyBMP24Pixels(bmpPixelData, blurredRedBytes, blurredGreenBytes,
                         blurredBlueBytes, croppedInfo)) {
    return 9;
  }
  std::cout << "BMP serialization verification = ok\n";

  try {
    image::writeBMP24(outputFile, bmpPixelData, croppedInfo);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 10;
  }
  std::cout << "wrote blurred BMP = " << outputFile << "\n";

  return 0;
}
