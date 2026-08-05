#include <image/bmp_crop.h>
#include <image/bmp_io.h>
#include <image/bmp_mask.h>

#include <toolchain/toolchain.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace {

using Color = std::array<std::uint8_t, 3>;

void append16(std::vector<std::uint8_t> & bytes, const std::uint16_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
}

void append32(std::vector<std::uint8_t> & bytes, const std::uint32_t value) {
    bytes.push_back(static_cast<std::uint8_t>(value));
    bytes.push_back(static_cast<std::uint8_t>(value >> 8U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 16U));
    bytes.push_back(static_cast<std::uint8_t>(value >> 24U));
}

void writeIndexedBMP(
    const std::filesystem::path & path,
    const std::uint32_t width,
    const std::uint32_t height,
    const bool topDown,
    const unsigned bitsPerPixel,
    const std::vector<Color> & palette,
    const std::vector<std::uint8_t> & indices
) {
    const std::size_t rowStride = ((static_cast<std::size_t>(width) * bitsPerPixel + 31U) / 32U) * 4U;
    const std::uint32_t dataOffset = static_cast<std::uint32_t>(14U + 40U + palette.size() * 4U);
    const std::uint32_t imageSize = static_cast<std::uint32_t>(rowStride * height);
    std::vector<std::uint8_t> bytes;
    bytes.reserve(dataOffset + imageSize);

    bytes.push_back('B');
    bytes.push_back('M');
    append32(bytes, dataOffset + imageSize);
    append32(bytes, 0U);
    append32(bytes, dataOffset);
    append32(bytes, 40U);
    append32(bytes, width);
    const std::int32_t signedHeight = topDown ? -static_cast<std::int32_t>(height) : static_cast<std::int32_t>(height);
    append32(bytes, static_cast<std::uint32_t>(signedHeight));
    append16(bytes, 1U);
    append16(bytes, static_cast<std::uint16_t>(bitsPerPixel));
    append32(bytes, 0U);
    append32(bytes, imageSize);
    append32(bytes, 0U);
    append32(bytes, 0U);
    append32(bytes, static_cast<std::uint32_t>(palette.size()));
    append32(bytes, 0U);
    for (const Color & color : palette) {
        bytes.insert(bytes.end(), color.begin(), color.end());
        bytes.push_back(0U);
    }

    for (std::uint32_t storedRow = 0; storedRow < height; ++storedRow) {
        const std::uint32_t sourceRow = topDown ? storedRow : height - storedRow - 1U;
        std::vector<std::uint8_t> row(rowStride);
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::uint8_t index = indices[static_cast<std::size_t>(sourceRow) * width + column];
            if (bitsPerPixel == 8U)
                row[column] = index;
            else
                row[column / 8U] |= static_cast<std::uint8_t>(index << (7U - column % 8U));
        }
        bytes.insert(bytes.end(), row.begin(), row.end());
    }

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("fixture write failed");
}

bool compareImage(std::string_view name, const image::BGRImage & actual, const image::BGRImage & expected) {
    if (actual.width != expected.width || actual.height != expected.height) {
        std::cerr << name << ": dimension mismatch\n";
        return false;
    }
    static constexpr char ChannelNames[] = {'B', 'G', 'R'};
    for (std::size_t byte = 0; byte < expected.pixels.size(); ++byte) {
        if (actual.pixels[byte] == expected.pixels[byte])
            continue;
        const std::size_t pixel = byte / 3U;
        std::cerr << name << ": y=" << pixel / expected.width << " x=" << pixel % expected.width << " channel=" << ChannelNames[byte % 3U]
                  << " expected=" << static_cast<unsigned>(expected.pixels[byte]) << " actual=" << static_cast<unsigned>(actual.pixels[byte]) << '\n';
        return false;
    }
    return true;
}

bool rejectsCrop(
    const image::BGRImage & source,
    const std::uint32_t originX,
    const std::uint32_t originY,
    const std::uint32_t width,
    const std::uint32_t height,
    std::string_view name
) {
    try {
        (void)image::cropImage(source, originX, originY, width, height);
    } catch (const std::runtime_error &) {
        return true;
    }
    std::cerr << name << ": crop was accepted\n";
    return false;
}

bool testEightBitLoadAndCrop(const std::filesystem::path & directory) {
    const std::vector<Color> palette = {
        {1, 11, 101},
        {2, 12, 102},
        {3, 13, 103},
        {4, 14, 104},
        {5, 15, 105},
        {6, 16, 106},
        {7, 17, 107},
        {8, 18, 108},
        {9, 19, 109},
        {10, 20, 110},
        {21, 31, 121},
        {22, 32, 122},
        {23, 33, 123},
        {24, 34, 124},
        {25, 35, 125},
    };
    const std::vector<std::uint8_t> indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    image::BGRImage expected(5, 3);
    expected.pixels = {
        1,   11, 101, 2,   12, 102, 3,   13, 103, 4,   14, 104, 5,   15, 105, 6,   16, 106, 7,   17, 107, 8,   18,
        108, 9,  19,  109, 10, 20,  110, 21, 31,  121, 22, 32,  122, 23, 33,  123, 24, 34,  124, 25, 35,  125,
    };
    const auto bottomUpPath = directory / "bottom_up.bmp";
    const auto topDownPath = directory / "top_down.bmp";
    writeIndexedBMP(bottomUpPath, 5, 3, false, 8, palette, indices);
    writeIndexedBMP(topDownPath, 5, 3, true, 8, palette, indices);

    if (!compareImage("8-bit bottom-up", image::loadBMP(bottomUpPath.string()), expected)
        || !compareImage("8-bit top-down", image::loadBMP(topDownPath.string()), expected))
    {
        return false;
    }

    image::BGRImage topLeft(2, 2);
    topLeft.pixels = {1, 11, 101, 2, 12, 102, 6, 16, 106, 7, 17, 107};
    image::BGRImage interior(3, 2);
    interior.pixels = {7, 17, 107, 8, 18, 108, 9, 19, 109, 22, 32, 122, 23, 33, 123, 24, 34, 124};
    image::BGRImage rightEdge(2, 2);
    rightEdge.pixels = {4, 14, 104, 5, 15, 105, 9, 19, 109, 10, 20, 110};
    image::BGRImage bottomEdge(3, 1);
    bottomEdge.pixels = {21, 31, 121, 22, 32, 122, 23, 33, 123};
    image::BGRImage singlePixel(1, 1);
    singlePixel.pixels = {25, 35, 125};

    return compareImage("three-argument top-left crop", image::cropImage(expected, 2, 2), topLeft)
           && compareImage("origin top-left crop", image::cropImage(expected, 0, 0, 2, 2), topLeft)
           && compareImage("interior crop", image::cropImage(expected, 1, 1, 3, 2), interior)
           && compareImage("right-edge crop", image::cropImage(expected, 3, 0, 2, 2), rightEdge)
           && compareImage("bottom-edge crop", image::cropImage(expected, 0, 2, 3, 1), bottomEdge)
           && compareImage("single-pixel crop", image::cropImage(expected, 4, 2, 1, 1), singlePixel)
           && compareImage("full-image crop", image::cropImage(expected, 0, 0, 5, 3), expected) && rejectsCrop(expected, 0, 0, 0, 2, "zero-width")
           && rejectsCrop(expected, 0, 0, 2, 0, "zero-height") && rejectsCrop(expected, expected.width, 0, 1, 1, "origin-x-at-width")
           && rejectsCrop(expected, 0, expected.height, 1, 1, "origin-y-at-height") && rejectsCrop(expected, 4, 0, 2, 1, "width-beyond-right-edge")
           && rejectsCrop(expected, 0, 2, 1, 2, "height-beyond-bottom-edge");
}

bool testOneBitMask(const std::filesystem::path & directory) {
    image::BGRImage source(5, 2);
    for (std::size_t byte = 0; byte < source.pixels.size(); ++byte)
        source.pixels[byte] = static_cast<std::uint8_t>(byte * 7U + 3U);

    const std::vector<std::uint8_t> blackPositions = {1, 0, 1, 0, 0, 0, 1, 0, 1, 0};
    std::vector<std::uint8_t> blackThenWhiteIndices(blackPositions.size());
    std::vector<std::uint8_t> whiteThenBlackIndices(blackPositions.size());
    for (std::size_t pixel = 0; pixel < blackPositions.size(); ++pixel) {
        blackThenWhiteIndices[pixel] = blackPositions[pixel] ? 0U : 1U;
        whiteThenBlackIndices[pixel] = blackPositions[pixel] ? 1U : 0U;
    }

    const auto normalPath = directory / "mask_black_white.bmp";
    const auto reversedPath = directory / "mask_white_black.bmp";
    writeIndexedBMP(normalPath, 5, 2, true, 1, {{0, 0, 0}, {255, 255, 255}}, blackThenWhiteIndices);
    writeIndexedBMP(reversedPath, 5, 2, false, 1, {{255, 255, 255}, {0, 0, 0}}, whiteThenBlackIndices);

    image::BGRImage expectedBlack = source;
    image::BGRImage expectedColor = source;
    for (std::size_t pixel = 0; pixel < blackPositions.size(); ++pixel) {
        if (!blackPositions[pixel])
            continue;
        expectedBlack.pixels[pixel * 3U] = 0U;
        expectedBlack.pixels[pixel * 3U + 1U] = 0U;
        expectedBlack.pixels[pixel * 3U + 2U] = 0U;
        expectedColor.pixels[pixel * 3U] = 5U;
        expectedColor.pixels[pixel * 3U + 1U] = 77U;
        expectedColor.pixels[pixel * 3U + 2U] = 201U;
    }

    const auto mismatchPath = directory / "mask_shape_mismatch.bmp";
    writeIndexedBMP(mismatchPath, 2, 5, true, 1, {{0, 0, 0}, {255, 255, 255}}, blackThenWhiteIndices);
    try {
        (void)image::maskImage(source, mismatchPath.string(), {0, 0, 0});
        std::cerr << "mask dimension mismatch: mask was accepted\n";
        return false;
    } catch (const std::runtime_error &) {
    }

    return compareImage("top-down black mask", image::maskImage(source, normalPath.string(), {0, 0, 0}), expectedBlack)
           && compareImage("bottom-up reversed-palette color mask", image::maskImage(source, reversedPath.string(), {5, 77, 201}), expectedColor);
}

bool testSave(const std::filesystem::path & directory) {
    image::BGRImage source(3, 2);
    source.pixels = {0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255, 63, 31, 95, 130, 70, 200};
    const auto path = directory / "saved.bmp";
    image::saveBMP(path.string(), source);

    image::BGRImage expected(3, 2);
    for (std::size_t pixel = 0; pixel < source.pixels.size() / 3U; ++pixel) {
        const unsigned blue = source.pixels[pixel * 3U];
        const unsigned green = source.pixels[pixel * 3U + 1U];
        const unsigned red = source.pixels[pixel * 3U + 2U];
        expected.pixels[pixel * 3U] = static_cast<std::uint8_t>((blue >> 6U) * 255U / 3U);
        expected.pixels[pixel * 3U + 1U] = static_cast<std::uint8_t>((green >> 5U) * 255U / 7U);
        expected.pixels[pixel * 3U + 2U] = static_cast<std::uint8_t>((red >> 5U) * 255U / 7U);
    }

    std::array<std::uint8_t, 30> header{};
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char *>(header.data()), static_cast<std::streamsize>(header.size()));
    const std::int32_t storedHeight = static_cast<std::int32_t>(header[22] | (header[23] << 8U) | (header[24] << 16U) | (header[25] << 24U));
    const unsigned bitsPerPixel = header[28] | (header[29] << 8U);
    if (!input || storedHeight != 2 || bitsPerPixel != 8U) {
        std::cerr << "RGB332 save: header mismatch\n";
        return false;
    }
    return compareImage("RGB332 save", image::loadBMP(path.string()), expected);
}

}  // namespace

int main(int argc, char ** argv) {
    codegen::ParseCommandLineOptions(argc, argv, {&codegen::JIT_InfoOptions});
    const std::filesystem::path directory = std::filesystem::temp_directory_path() / ("parabix_test_bmp_" + std::to_string(::getpid()));
    try {
        std::filesystem::create_directory(directory);
        const bool passed = testEightBitLoadAndCrop(directory) && testOneBitMask(directory) && testSave(directory);
        std::filesystem::remove_all(directory);
        return passed ? 0 : 1;
    } catch (const std::exception & error) {
        std::filesystem::remove_all(directory);
        std::cerr << "test_bmp: " << error.what() << '\n';
        return 1;
    }
}
