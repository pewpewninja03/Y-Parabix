/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <image/bmp_io.h>
#include <image/bmp_pipeline.h>
#include <image/conv_filter.h>

#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/streamutils/stream_select.h>
#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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
static llvm::cl::opt<bool>
    maskBrightRed("mask-bright-red",
                  llvm::cl::desc("black out cropped pixels whose red value "
                                 "is >= 128 before blurring"),
                  llvm::cl::init(false));
static llvm::cl::opt<bool>
    maskBrightGreen("mask-bright-green",
                    llvm::cl::desc("black out cropped pixels whose green value "
                                   "is >= 128 before blurring"),
                    llvm::cl::init(false));
static llvm::cl::opt<bool>
    maskBrightBlue("mask-bright-blue",
                   llvm::cl::desc("black out cropped pixels whose blue value "
                                  "is >= 128 before blurring"),
                   llvm::cl::init(false));
static llvm::cl::opt<bool>
    noBlur("no-blur",
           llvm::cl::desc("skip the 3x3 box blur and write the masked crop"),
           llvm::cl::init(false));
static llvm::cl::opt<std::string>
    outputFile("o",
               llvm::cl::desc("write the (optionally blurred) crop as a 24-bit BMP"),
               llvm::cl::value_desc("output.bmp"), llvm::cl::Required);

int main(int argc, char **argv) {
  codegen::ParseCommandLineOptions(
      argc, argv,
      {&codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});

  const image::BMPCrop crop{cropWidth, cropHeight, cropX, cropY};

  constexpr unsigned kBlueMSB = 7u;
  constexpr unsigned kGreenMSB = 15u;
  constexpr unsigned kRedMSB = 23u;

  std::vector<uint32_t> maskStreamIndices;
  if (maskBrightBlue) {
    maskStreamIndices.push_back(kBlueMSB);
  }
  if (maskBrightGreen) {
    maskStreamIndices.push_back(kGreenMSB);
  }
  if (maskBrightRed) {
    maskStreamIndices.push_back(kRedMSB);
  }

  image::ColorStreamTransform transform;
  if (!maskStreamIndices.empty()) {
    transform = [indices = std::move(maskStreamIndices)](
                    kernel::ProgramBuilder &P,
                    kernel::StreamSet *sourceImageData) -> kernel::StreamSet * {
      kernel::StreamSet *maskStream =
          kernel::streamutils::Merge(P, sourceImageData, indices);
      kernel::StreamSet *maskedImageData = nullptr;
      image::MaskImage(P, sourceImageData, maskStream, maskedImageData);
      return maskedImageData;
    };
  }

  CPUDriver driver("bmp_demo");
  image::BMPCropResult result;
  try {
    result = image::LoadBMPCrop(driver, inputFile, crop, transform);
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

  uint64_t brightRed = 0;
  uint64_t brightGreen = 0;
  uint64_t brightBlue = 0;
  uint64_t croppedBlackPixels = 0;
  const uint8_t *croppedRGB = cropped.data();
  for (uint64_t i = 0; i < croppedPixels; ++i) {
    const uint8_t r = croppedRGB[i * 3u + 0u];
    const uint8_t g = croppedRGB[i * 3u + 1u];
    const uint8_t b = croppedRGB[i * 3u + 2u];
    if (r >= 128u) ++brightRed;
    if (g >= 128u) ++brightGreen;
    if (b >= 128u) ++brightBlue;
    if (r == 0u && g == 0u && b == 0u) ++croppedBlackPixels;
  }
  std::cout << "\n=== ParseBMPColorStreams output ===\n";
  std::cout << "crop rectangle = " << cropWidth << "x" << cropHeight
            << " at (" << cropX << ", " << cropY << ")\n";
  std::cout << "bright cropped red pixels (value >= 128) = "
            << brightRed << " / " << croppedPixels << "\n";
  std::cout << "bright cropped green pixels (value >= 128) = "
            << brightGreen << " / " << croppedPixels << "\n";
  std::cout << "bright cropped blue pixels (value >= 128) = "
            << brightBlue << " / " << croppedPixels << "\n";
  if (maskBrightBlue || maskBrightGreen || maskBrightRed) {
    std::cout << "MaskImage: blacked out pixels with"
              << (maskBrightBlue ? " blue>=128" : "")
              << (maskBrightGreen ? " green>=128" : "")
              << (maskBrightRed ? " red>=128" : "")
              << " (" << croppedBlackPixels << " / " << croppedPixels << ")\n";
  }

  if (!noBlur) {
    image::BMP24Image blurred(cropped.width, cropped.height,
                              cropped.rowsBottomUp);
    const float weights[] = {
        1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
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
  } else {
    std::cout << "\n=== ConvFilter skipped (--no-blur) ===\n";
    try {
      image::writeBMP24(outputFile, cropped);
    } catch (const std::exception &e) {
      std::cerr << e.what() << "\n";
      return 3;
    }
  }
  std::cout << "wrote " << (noBlur ? "masked" : "blurred")
            << " BMP = " << outputFile << "\n";

  return 0;
}
