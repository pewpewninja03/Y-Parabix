/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_io.h>
#include <image/bmp_pipeline.h>
#include <image/conv_filter.h>

#include <kernel/pipeline/driver/cpudriver.h>
#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

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

int main(int argc, char **argv) {
  codegen::ParseCommandLineOptions(
      argc, argv,
      {&codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});

  const image::BMPCrop crop{cropWidth, cropHeight, cropX, cropY};

  CPUDriver driver("bmp_demo");
  image::BMPCropResult result;
  try {
    result = image::LoadBMPCrop(driver, inputFile, crop);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }

  const image::BMPInfo &info = result.sourceInfo;
  std::cout << "=== BMP Header ===\n";
  std::cout << "width        = " << info.width << "\n";
  std::cout << "height       = " << info.height << "\n";
  std::cout << "rowStride    = " << info.rowStride << "\n";
  std::cout << "pixelOffset  = " << info.pixelOffset << "\n";
  std::cout << "rowsBottomUp = " << info.rowsBottomUp << "\n";
  std::cout << "total pixels = " << (info.width * info.height) << "\n";

  const image::BMP24Image &cropped = result.image;
  const uint64_t croppedPixels =
      static_cast<uint64_t>(cropped.pixelCount());

  uint64_t croppedBrightPixels = 0;
  const uint8_t *croppedRGB = cropped.data();
  for (uint64_t i = 0; i < croppedPixels; ++i) {
    if (croppedRGB[i * 3u + 0u] >= 128) {
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

  image::BMP24Image blurred(cropped.width, cropped.height,
                            cropped.rowsBottomUp);
  const float weights[] = {1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                           1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f};
  const kernel::image::DefaultConvFilter config{
      3,
      3,
      {weights, 9},
  };
  const auto filter = kernel::image::compileConvFilter(
      cropped.width, cropped.height, config);
  if (!filter->apply(cropped.data(), blurred.data(), nullptr)) {
    std::cerr << "applyConvFilter failed\n";
    return 2;
  }

  std::cout << "\n=== ConvFilter output ===\n";
  if (croppedPixels != 0) {
    std::cout << "first cropped pixel RGB = "
              << static_cast<unsigned>(blurred.data()[0]) << ", "
              << static_cast<unsigned>(blurred.data()[1]) << ", "
              << static_cast<unsigned>(blurred.data()[2]) << "\n";
  }

  try {
    image::writeBMP24(outputFile, blurred);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 3;
  }
  std::cout << "wrote blurred BMP = " << outputFile << "\n";

  return 0;
}
