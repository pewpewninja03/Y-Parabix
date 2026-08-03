#include <image/bmp_crop.h>
#include <image/bmp_io.h>
#include <image/bmp_mask.h>

#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

llvm::cl::OptionCategory BenchOptions("Bench Options", "Isolated crop/mask timing options.");

llvm::cl::opt<std::string> Operation("op", llvm::cl::desc("Operation: crop | mask"), llvm::cl::init("crop"), llvm::cl::cat(BenchOptions));
llvm::cl::opt<std::string> InputPath("input", llvm::cl::desc("Source BMP"), llvm::cl::Required, llvm::cl::cat(BenchOptions));
llvm::cl::opt<std::uint32_t> CropWidth("crop-width", llvm::cl::desc("Crop width (op=crop)"), llvm::cl::init(0U), llvm::cl::cat(BenchOptions));
llvm::cl::opt<std::uint32_t> CropHeight("crop-height", llvm::cl::desc("Crop height (op=crop)"), llvm::cl::init(0U), llvm::cl::cat(BenchOptions));
llvm::cl::opt<std::string> MaskPath("mask", llvm::cl::desc("Mask BMP (op=mask)"), llvm::cl::init(""), llvm::cl::cat(BenchOptions));
llvm::cl::opt<unsigned> Trials("trials", llvm::cl::desc("Timed trials"), llvm::cl::init(10U), llvm::cl::cat(BenchOptions));
llvm::cl::opt<unsigned> Warmup("warmup", llvm::cl::desc("Warm-up calls (not timed)"), llvm::cl::init(1U), llvm::cl::cat(BenchOptions));

void runCrop(const image::BGRImage & source, const std::uint32_t width, const std::uint32_t height, const unsigned trials, const unsigned warmup) {
    if (width == 0U || height == 0U)
        throw std::runtime_error("crop requires --crop-width and --crop-height");
    if (width > source.width || height > source.height)
        throw std::runtime_error("crop dimension exceeds input");

    std::size_t sink = 0U;
    for (unsigned i = 0U; i < warmup; ++i) {
        const image::BGRImage out = image::cropImage(source, width, height);
        sink += out.pixels.size();
    }

    double sum = 0.0, mn = 1e18, mx = 0.0;
    for (unsigned i = 0U; i < trials; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const image::BGRImage out = image::cropImage(source, width, height);
        const auto t1 = std::chrono::high_resolution_clock::now();
        sink += out.pixels.size();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum += ms;
        mn = std::min(mn, ms);
        mx = std::max(mx, ms);
    }
    if (sink == 0) std::fprintf(stderr, "empty");  // never true; keeps results live
    std::printf("RESULT op=crop src=%u crop=%u trials=%u avg_ms=%.4f min_ms=%.4f max_ms=%.4f\n",
                source.width, width, trials, sum / trials, mn, mx);
}

void runMask(const image::BGRImage & source, const std::string & maskPath, const unsigned trials, const unsigned warmup) {
    if (maskPath.empty())
        throw std::runtime_error("mask requires --mask");
    const image::BGRColor black{0U, 0U, 0U};

    std::size_t sink = 0U;
    for (unsigned i = 0U; i < warmup; ++i) {
        const image::BGRImage out = image::maskImage(source, maskPath, black);
        sink += out.pixels.size();
    }

    double sum = 0.0, mn = 1e18, mx = 0.0;
    for (unsigned i = 0U; i < trials; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const image::BGRImage out = image::maskImage(source, maskPath, black);
        const auto t1 = std::chrono::high_resolution_clock::now();
        sink += out.pixels.size();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum += ms;
        mn = std::min(mn, ms);
        mx = std::max(mx, ms);
    }
    if (sink == 0) std::fprintf(stderr, "empty");  // never true; keeps results live
    std::printf("RESULT op=mask src=%u mask=%u trials=%u avg_ms=%.4f min_ms=%.4f max_ms=%.4f\n",
                source.width, source.width, trials, sum / trials, mn, mx);
}

}  // namespace

int main(int argc, char ** argv) {
    codegen::ParseCommandLineOptions(argc, argv, {&codegen::JIT_InfoOptions, &codegen::InstrumentationOptions, &BenchOptions});

    const image::BGRImage source = image::loadBMP(InputPath);

    if (Operation == "crop") {
        runCrop(source, CropWidth, CropHeight, Trials, Warmup);
    } else if (Operation == "mask") {
        runMask(source, MaskPath, Trials, Warmup);
    } else {
        throw std::runtime_error("unknown --op: " + Operation);
    }
    return 0;
}
