#include <image/bmp_io.h>

#include "bmp_pipeline_internal.h"

#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/stream_select.h>

#include <pablo/bixnum/bixnum.h>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/builder.hpp>

#include <fcntl.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace image {
namespace internal {
namespace {

struct __attribute__((packed)) BitmapFileHeader {
    std::uint8_t signature[2];
    std::uint32_t fileSize;
    std::uint32_t reserved;
    std::uint32_t dataOffset;
};

struct __attribute__((packed)) BitmapInfoHeader {
    std::uint32_t size;
    std::int32_t width;
    std::int32_t height;
    std::uint16_t planes;
    std::uint16_t bitsPerPixel;
    std::uint32_t compression;
    std::uint32_t imageSize;
    std::int32_t xPixelsPerM;
    std::int32_t yPixelsPerM;
    std::uint32_t colorsUsed;
    std::uint32_t importantColors;
};

static_assert(sizeof(BitmapFileHeader) == 14);
static_assert(sizeof(BitmapInfoHeader) == 40);

constexpr std::uint32_t PaletteSize = 256;

std::size_t bmpRowStride(const std::uint32_t width, const std::uint16_t bitsPerPixel) {
    return ((static_cast<std::size_t>(width) * bitsPerPixel + 31U) / 32U) * 4U;
}

std::string makePaletteSignature(const std::vector<unsigned> & blue, const std::vector<unsigned> & green, const std::vector<unsigned> & red) {
    std::string signature = "palette_lut";
    const auto appendTable = [&signature](const std::vector<unsigned> & table) {
        signature += '[';
        signature += std::to_string(table.size());
        for (const unsigned value : table) {
            signature += ',';
            signature += std::to_string(value);
        }
        signature += ']';
    };
    appendTable(blue);
    appendTable(green);
    appendTable(red);
    return signature;
}

class PaletteLUTKernel final : public pablo::PabloKernel {
   public:
    PaletteLUTKernel(
        LLVMTypeSystemInterface & typeSystem,
        kernel::StreamSet * index,
        kernel::StreamSet * color,
        std::vector<unsigned> blue,
        std::vector<unsigned> green,
        std::vector<unsigned> red
    )
        : pablo::PabloKernel(
              typeSystem,
              "palette_lut_" + kernel::Kernel::getStringHash(makePaletteSignature(blue, green, red)),
              {kernel::Binding{"index", index}},
              {kernel::Binding{"color", color}}
          ),
          mBlue(std::move(blue)),
          mGreen(std::move(green)),
          mRed(std::move(red)),
          mSignature(makePaletteSignature(mBlue, mGreen, mRed)) {}

    llvm::StringRef getSignature() const override {
        return mSignature;
    }

    bool hasSignature() const override {
        return true;
    }

   protected:
    void generatePabloMethod() override;

   private:
    std::vector<unsigned> mBlue;
    std::vector<unsigned> mGreen;
    std::vector<unsigned> mRed;
    std::string mSignature;
};

void PaletteLUTKernel::generatePabloMethod() {
    pablo::PabloBuilder builder(getEntryScope());
    pablo::BixNum index = getInputStreamSet("index");
    pablo::Var * const output = getOutputStreamVar("color");

    const auto emitChannel = [&](const unsigned baseStream, std::vector<unsigned> & table) {
        pablo::BixVar channel(ChannelBits);
        for (unsigned bit = 0; bit < ChannelBits; ++bit) {
            channel[bit] = builder.createVar("ch" + std::to_string(baseStream + bit), builder.createZeroes());
        }
        pablo::BixNumTableCompiler compiler(table, index, channel);
        std::vector<unsigned> partitionLevels{ChannelBits};
        compiler.setRecursivePartitionLevels(partitionLevels);
        compiler.compileSubTable(builder, 0, builder.createOnes());
        for (unsigned bit = 0; bit < ChannelBits; ++bit) {
            builder.createAssign(builder.createExtract(output, builder.getInteger(baseStream + bit)), channel[bit]);
        }
    };

    emitChannel(BlueStreamBase, mBlue);
    emitChannel(GreenStreamBase, mGreen);
    emitChannel(RedStreamBase, mRed);
}

}  // namespace

void readExact(const int fileDescriptor, void * data, std::size_t byteCount) {
    auto * bytes = static_cast<std::uint8_t *>(data);
    while (byteCount != 0) {
        const ssize_t result = ::read(fileDescriptor, bytes, byteCount);
        if (result < 0 && errno == EINTR)
            continue;
        if (result <= 0)
            throw std::runtime_error("BMP read failed");
        bytes += result;
        byteCount -= static_cast<std::size_t>(result);
    }
}

BitmapMetadata readBMPMetadata(const int fileDescriptor, const std::uint16_t bitsPerPixel) {
    BitmapFileHeader fileHeader{};
    BitmapInfoHeader infoHeader{};
    readExact(fileDescriptor, &fileHeader, sizeof(fileHeader));
    readExact(fileDescriptor, &infoHeader, sizeof(infoHeader));

    const std::uint32_t maximumColors = bitsPerPixel == 1U ? 2U : bitsPerPixel == 8U ? PaletteSize : 0U;
    const std::uint32_t colorCount = infoHeader.colorsUsed == 0U ? maximumColors : infoHeader.colorsUsed;
    if (fileHeader.signature[0] != 'B' || fileHeader.signature[1] != 'M' || infoHeader.size != sizeof(BitmapInfoHeader) || infoHeader.width <= 0
        || infoHeader.height == 0 || infoHeader.planes != 1U || infoHeader.bitsPerPixel != bitsPerPixel || maximumColors == 0U || colorCount == 0U
        || colorCount > maximumColors || infoHeader.compression != 0U)
    {
        throw std::runtime_error("unsupported BMP");
    }

    BitmapMetadata info{};
    info.width = static_cast<std::uint32_t>(infoHeader.width);
    const std::int64_t signedHeight = infoHeader.height;
    info.height = static_cast<std::uint32_t>(signedHeight < 0 ? -signedHeight : signedHeight);
    info.rowStride = static_cast<std::uint32_t>(bmpRowStride(info.width, bitsPerPixel));
    info.pixelOffset = fileHeader.dataOffset;
    info.rowsBottomUp = infoHeader.height > 0;
    info.palette.resize(colorCount);
    readExact(fileDescriptor, info.palette.data(), info.palette.size() * sizeof(info.palette.front()));

    if (::lseek(fileDescriptor, static_cast<off_t>(info.pixelOffset), SEEK_SET) < 0)
        throw std::runtime_error("BMP read failed");
    return info;
}

AlignedByteBuffer::AlignedByteBuffer(const std::vector<std::uint8_t> & bytes, const std::size_t alignment)
    : mStorage(bytes.size() + alignment - 1U), mData(nullptr) {
    void * data = mStorage.data();
    std::size_t space = mStorage.size();
    mData = static_cast<std::uint8_t *>(std::align(alignment, bytes.size(), data, space));
    std::copy(bytes.begin(), bytes.end(), mData);
}

const std::uint8_t * AlignedByteBuffer::data() const noexcept {
    return mData;
}

kernel::StreamSet * createColorStream(kernel::PipelineBuilder & pipeline, kernel::Scalar * packedPixels, kernel::Scalar * byteCount) {
    kernel::StreamSet * const packedBytes = pipeline.CreateStreamSet(1, ChannelBits);
    pipeline.CreateKernelCall<kernel::MemorySourceKernel>(packedPixels, byteCount, packedBytes);

    const std::array<kernel::StreamSet *, 3> channelMasks = {
        pipeline.CreateRepeatingStreamSet(1, {1U, 0U, 0U}),
        pipeline.CreateRepeatingStreamSet(1, {0U, 1U, 0U}),
        pipeline.CreateRepeatingStreamSet(1, {0U, 0U, 1U}),
    };
    std::array<kernel::StreamSet *, 3> channelBasis{};
    for (unsigned channel = 0; channel < channelBasis.size(); ++channel) {
        kernel::StreamSet * const channelBytes = pipeline.CreateStreamSet(1, ChannelBits);
        kernel::FilterByMask(pipeline, channelMasks[channel], packedBytes, channelBytes);
        channelBasis[channel] = pipeline.CreateStreamSet(ChannelBits);
        pipeline.CreateKernelCall<kernel::S2PKernel>(channelBytes, channelBasis[channel]);
    }
    return kernel::streamutils::Select(pipeline, {channelBasis[0], channelBasis[1], channelBasis[2]});
}

void createColorByteStreams(
    kernel::PipelineBuilder & pipeline,
    kernel::StreamSet * colorStream,
    kernel::StreamSet * blueBytes,
    kernel::StreamSet * greenBytes,
    kernel::StreamSet * redBytes
) {
    kernel::StreamSet * const blue = kernel::streamutils::Select(pipeline, colorStream, kernel::streamutils::Range(BlueStreamBase, GreenStreamBase));
    kernel::StreamSet * const green = kernel::streamutils::Select(pipeline, colorStream, kernel::streamutils::Range(GreenStreamBase, RedStreamBase));
    kernel::StreamSet * const red = kernel::streamutils::Select(pipeline, colorStream, kernel::streamutils::Range(RedStreamBase, ColorStreamCount));
    pipeline.CreateKernelCall<kernel::P2SKernel>(blue, blueBytes);
    pipeline.CreateKernelCall<kernel::P2SKernel>(green, greenBytes);
    pipeline.CreateKernelCall<kernel::P2SKernel>(red, redBytes);
}

BGRImage materializeColor(
    const kernel::StreamSetPtr & blueBytes,
    const kernel::StreamSetPtr & greenBytes,
    const kernel::StreamSetPtr & redBytes,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool rowsBottomUp
) {
    const std::unique_ptr<std::uint8_t, decltype(&std::free)> blue(blueBytes.data(), &std::free);
    const std::unique_ptr<std::uint8_t, decltype(&std::free)> green(greenBytes.data(), &std::free);
    const std::unique_ptr<std::uint8_t, decltype(&std::free)> red(redBytes.data(), &std::free);
    BGRImage image(width, height);
    for (std::uint32_t outputRow = 0; outputRow < height; ++outputRow) {
        const std::uint32_t inputRow = rowsBottomUp ? height - outputRow - 1U : outputRow;
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::size_t inputPixel = static_cast<std::size_t>(inputRow) * width + column;
            const std::size_t outputPixel = static_cast<std::size_t>(outputRow) * width + column;
            image.pixels[outputPixel * 3U] = blue.get()[inputPixel];
            image.pixels[outputPixel * 3U + 1U] = green.get()[inputPixel];
            image.pixels[outputPixel * 3U + 2U] = red.get()[inputPixel];
        }
    }
    return image;
}

BGRImage materializePackedColor(const kernel::StreamSetPtr & packedBytes, const std::uint32_t width, const std::uint32_t height) {
    const std::unique_ptr<std::uint8_t, decltype(&std::free)> bytes(packedBytes.data(), &std::free);
    BGRImage image(width, height);
    std::copy_n(bytes.get(), image.pixels.size(), image.pixels.begin());
    return image;
}

}  // namespace internal

BGRImage::BGRImage(const std::uint32_t width, const std::uint32_t height)
    : width(width), height(height), pixels(static_cast<std::size_t>(width) * height * 3U) {}

std::uint8_t * BGRImage::data() noexcept {
    return pixels.data();
}

const std::uint8_t * BGRImage::data() const noexcept {
    return pixels.data();
}

BGRImage loadBMP(const std::string & path) {
    const int fileDescriptor = ::open(path.c_str(), O_RDONLY);
    if (fileDescriptor < 0)
        throw std::runtime_error("BMP open failed");

    internal::BitmapMetadata info{};
    kernel::StreamSetPtr blueBytes;
    kernel::StreamSetPtr greenBytes;
    kernel::StreamSetPtr redBytes;
    try {
        info = internal::readBMPMetadata(fileDescriptor, 8U);
        std::vector<unsigned> blue(internal::PaletteSize);
        std::vector<unsigned> green(internal::PaletteSize);
        std::vector<unsigned> red(internal::PaletteSize);
        for (std::size_t index = 0; index < info.palette.size(); ++index) {
            blue[index] = info.palette[index][0];
            green[index] = info.palette[index][1];
            red[index] = info.palette[index][2];
        }

        CPUDriver driver("image_load_bmp");
        auto pipeline = kernel::CreatePipeline(
            driver,
            kernel::Output<kernel::streamset_t>{"blueBytes", 1, 8, kernel::ReturnedBuffer(1)},
            kernel::Output<kernel::streamset_t>{"greenBytes", 1, 8, kernel::ReturnedBuffer(1)},
            kernel::Output<kernel::streamset_t>{"redBytes", 1, 8, kernel::ReturnedBuffer(1)},
            kernel::Input<std::uint32_t>{"fileDescriptor"}
        );

        kernel::StreamSet * byteStream = pipeline.CreateStreamSet(1, 8);
        pipeline.CreateKernelCall<kernel::ReadSourceKernel>(pipeline.getInputScalar("fileDescriptor"), byteStream);
        kernel::StreamSet * pixelStream = byteStream;
        if (info.rowStride != info.width) {
            std::vector<std::uint64_t> rowPattern(info.rowStride);
            std::fill_n(rowPattern.begin(), info.width, 1U);
            kernel::StreamSet * const rowMask = pipeline.CreateRepeatingStreamSet(1, rowPattern);
            pixelStream = pipeline.CreateStreamSet(1, 8);
            kernel::FilterByMask(pipeline, rowMask, byteStream, pixelStream);
        }

        kernel::StreamSet * const indexBasis = pipeline.CreateStreamSet(8);
        pipeline.CreateKernelCall<kernel::S2PKernel>(pixelStream, indexBasis);
        kernel::StreamSet * const colorStream = pipeline.CreateStreamSet(internal::ColorStreamCount);
        pipeline.CreateKernelCall<internal::PaletteLUTKernel>(indexBasis, colorStream, blue, green, red);
        internal::createColorByteStreams(
            pipeline,
            colorStream,
            pipeline.getOutputStreamSet("blueBytes"),
            pipeline.getOutputStreamSet("greenBytes"),
            pipeline.getOutputStreamSet("redBytes")
        );

        const auto run = pipeline.compile();
        run(blueBytes, greenBytes, redBytes, static_cast<std::uint32_t>(fileDescriptor));
    } catch (...) {
        ::close(fileDescriptor);
        throw;
    }
    ::close(fileDescriptor);
    return internal::materializeColor(blueBytes, greenBytes, redBytes, info.width, info.height, info.rowsBottomUp);
}

void saveBMP(const std::string & path, const BGRImage & image) {
    constexpr std::uint32_t HeaderSize = sizeof(internal::BitmapFileHeader) + sizeof(internal::BitmapInfoHeader);
    constexpr std::uint32_t PaletteByteCount = internal::PaletteSize * 4U;
    constexpr std::uint32_t DataOffset = HeaderSize + PaletteByteCount;

    const std::size_t rowStride = internal::bmpRowStride(image.width, 8U);
    const std::uint32_t imageSize = static_cast<std::uint32_t>(rowStride * image.height);

    internal::BitmapFileHeader fileHeader{};
    fileHeader.signature[0] = 'B';
    fileHeader.signature[1] = 'M';
    fileHeader.fileSize = DataOffset + imageSize;
    fileHeader.dataOffset = DataOffset;

    internal::BitmapInfoHeader infoHeader{};
    infoHeader.size = sizeof(internal::BitmapInfoHeader);
    infoHeader.width = static_cast<std::int32_t>(image.width);
    infoHeader.height = static_cast<std::int32_t>(image.height);
    infoHeader.planes = 1U;
    infoHeader.bitsPerPixel = 8U;
    infoHeader.imageSize = imageSize;
    infoHeader.colorsUsed = internal::PaletteSize;

    std::array<std::uint8_t, PaletteByteCount> palette{};
    for (unsigned index = 0; index < internal::PaletteSize; ++index) {
        const std::size_t entry = static_cast<std::size_t>(index) * 4U;
        palette[entry] = static_cast<std::uint8_t>((index & 0x3U) * 255U / 3U);
        palette[entry + 1U] = static_cast<std::uint8_t>(((index >> 2U) & 0x7U) * 255U / 7U);
        palette[entry + 2U] = static_cast<std::uint8_t>(((index >> 5U) & 0x7U) * 255U / 7U);
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("BMP open failed");
    output.write(reinterpret_cast<const char *>(&fileHeader), sizeof(fileHeader));
    output.write(reinterpret_cast<const char *>(&infoHeader), sizeof(infoHeader));
    output.write(reinterpret_cast<const char *>(palette.data()), palette.size());

    std::vector<std::uint8_t> row(rowStride);
    for (std::uint32_t storedRow = 0; storedRow < image.height; ++storedRow) {
        const std::uint32_t sourceRow = image.height - storedRow - 1U;
        for (std::uint32_t column = 0; column < image.width; ++column) {
            const std::size_t source = (static_cast<std::size_t>(sourceRow) * image.width + column) * 3U;
            const unsigned blue = image.pixels[source];
            const unsigned green = image.pixels[source + 1U];
            const unsigned red = image.pixels[source + 2U];
            row[column] = static_cast<std::uint8_t>(((red >> 5U) << 5U) | ((green >> 5U) << 2U) | (blue >> 6U));
        }
        output.write(reinterpret_cast<const char *>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    if (!output)
        throw std::runtime_error("BMP write failed");
}

}  // namespace image
