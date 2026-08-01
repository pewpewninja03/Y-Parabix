#include <image/bmp_crop.h>
#include <image/bmp_io.h>
#include <image/bmp_mask.h>
#include <image/conv_filter.h>

#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

llvm::cl::opt<std::string> inputPath(llvm::cl::Positional, llvm::cl::desc("[input BMP]"), llvm::cl::init("./demo.bmp"));
llvm::cl::opt<std::string> maskPath(llvm::cl::Positional, llvm::cl::desc("[mask BMP]"), llvm::cl::init("./demo_mask.bmp"));
llvm::cl::opt<std::string> outputPath(llvm::cl::Positional, llvm::cl::desc("[output BMP]"), llvm::cl::init("./demo_output.bmp"));

// BITMAPINFOHEADER width/height live at file offsets 18 and 22.
void readMaskSize(const std::string & path, std::uint32_t & width, std::uint32_t & height) {
    std::ifstream in(path, std::ios::binary);
    std::int32_t w = 0, h = 0;
    in.seekg(18);
    in.read(reinterpret_cast<char *>(&w), 4);
    in.read(reinterpret_cast<char *>(&h), 4);
    width = static_cast<std::uint32_t>(w);
    height = static_cast<std::uint32_t>(h < 0 ? -h : h);
}

}  // namespace

int main(int argc, char ** argv) {
    codegen::ParseCommandLineOptions(argc, argv, {&codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});

    const image::BGRImage source = image::loadBMP(inputPath);
    std::uint32_t width = 0, height = 0;
    readMaskSize(maskPath, width, height);
    if (width == 0 || height == 0 || width > source.width || height > source.height)
        throw std::runtime_error("mask dimension exceeds input");

    image::BGRImage processed = image::cropImage(source, width, height);
    image::BGRImage alternate(processed.width, processed.height);

    constexpr std::array<float, 9> SharpenWeights = {0.0F, -1.0F, 0.0F, -1.0F, 5.0F, -1.0F, 0.0F, -1.0F, 0.0F};
    constexpr std::array<float, 3> SobelHorizontalFactors = {-1.0F, 0.0F, 1.0F};
    constexpr std::array<float, 3> SobelVerticalFactors = {1.0F, 2.0F, 1.0F};

    const kernel::image::DefaultConvFilter sharpenConfiguration{3, 3, {SharpenWeights.data(), SharpenWeights.size()}};
    const kernel::image::UniformConvFilter blurConfiguration{3, 3, 1.0F / 9.0F};
    const kernel::image::LowRankConvFilter sobelConfiguration{
        3, 3, 1, {SobelHorizontalFactors.data(), SobelHorizontalFactors.size()}, {SobelVerticalFactors.data(), SobelVerticalFactors.size()}
    };

    const auto sharpenFilter = kernel::image::compileConvFilter(processed.width, processed.height, sharpenConfiguration);
    const auto blurFilter = kernel::image::compileConvFilter(processed.width, processed.height, blurConfiguration);
    const auto sobelFilter = kernel::image::compileConvFilter(processed.width, processed.height, sobelConfiguration);

    const std::size_t workspaceSize = std::max(blurFilter->workspaceSize(), sobelFilter->workspaceSize());
    const std::size_t workspaceAlignment = std::max(blurFilter->workspaceAlignment(), sobelFilter->workspaceAlignment());
    std::vector<std::uint8_t> workspaceStorage(workspaceSize + workspaceAlignment - 1U);
    void * workspace = workspaceStorage.data();
    std::size_t workspaceSpace = workspaceStorage.size();
    workspace = std::align(workspaceAlignment, workspaceSize, workspace, workspaceSpace);

    (void)sharpenFilter->apply(processed.data(), alternate.data(), nullptr);
    (void)blurFilter->apply(alternate.data(), processed.data(), workspace);
    processed = image::maskImage(processed, maskPath, image::BGRColor{0, 0, 0});
    (void)sobelFilter->apply(processed.data(), alternate.data(), workspace);
    image::saveBMP(outputPath, alternate);
    return 0;
}
