/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 *  Tests for the BMP image I/O and pipeline APIs
 *  (lib/image/bmp_io.cpp, lib/image/bmp_pipeline.cpp).
 *
 *  The BMP pipeline functions need a file descriptor and a CPUDriver, so this
 *  file follows the custom-main pattern of tests/test_emptyprogram.cpp rather
 *  than the TEST_CASE/RUN_TESTS macros (which only handle in-memory streams).
 *  Fixtures are small synthetic 8-bit BMPs written to temp files (deterministic,
 *  with known palettes) plus a real-data smoke test against
 *  tools/image/lena_gray.bmp.
 */

#include <image/bmp_io.h>
#include <image/bmp_pipeline.h>

#include <kernel/core/attributes.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/stream_select.h>
#include <toolchain/toolchain.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace kernel;

namespace {

// ---------------------------------------------------------------------------
// Test scaffolding (modeled on tests/test_emptyprogram.cpp)
// ---------------------------------------------------------------------------

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

static int g_failureCount = 0;

template <typename Function, typename... Params>
void run_test(const char *testName, Function func, Params &&...params) {
  try {
    func(std::forward<Params>(params)...);
  } catch (const std::exception &e) {
    llvm::errs() << "[FAIL] " << testName << " threw: " << e.what() << "\n";
    ++g_failureCount;
  } catch (...) {
    llvm::errs() << "[FAIL] " << testName << " threw a non-std exception\n";
    ++g_failureCount;
  }
}

// Expect `func(args...)` to throw std::runtime_error; failure == no throw.
template <typename Function, typename... Params>
void expect_throw(const char *testName, Function func, Params &&...params) {
  bool threw = false;
  try {
    func(std::forward<Params>(params)...);
  } catch (const std::exception &) {
    threw = true;
  } catch (...) {
    threw = true;
  }
  if (!threw) {
    llvm::errs() << "[FAIL] " << testName
                 << " expected an exception but none was thrown\n";
    ++g_failureCount;
  }
}

#define CHECK(cond, msg)                                                      \
  do {                                                                       \
    if (!(cond)) {                                                            \
      llvm::errs() << "[FAIL] " << __func__ << ": " << (msg) << "\n";         \
      ++g_failureCount;                                                       \
      return;                                                                \
    }                                                                        \
  } while (0)

// ---------------------------------------------------------------------------
// TempFile - RAII temp file that unlinks itself on destruction.
// ---------------------------------------------------------------------------

class TempFile {
public:
  TempFile() = default;
  explicit TempFile(std::string path) : m_path(std::move(path)) {}
  ~TempFile() {
    if (!m_path.empty()) {
      ::unlink(m_path.c_str());
    }
  }
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
  TempFile(TempFile &&other) noexcept : m_path(std::move(other.m_path)) {
    other.m_path.clear();
  }
  TempFile &operator=(TempFile &&other) noexcept {
    if (this != &other) {
      if (!m_path.empty()) {
        ::unlink(m_path.c_str());
      }
      m_path = std::move(other.m_path);
      other.m_path.clear();
    }
    return *this;
  }
  const std::string &path() const { return m_path; }

  static TempFile create(const std::string &tag) {
    std::string tmpl = "/tmp/parabix_bmp_" + tag + "_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = ::mkstemp(buf.data());
    if (fd < 0) {
      throw std::runtime_error("TempFile: mkstemp failed");
    }
    ::close(fd);
    return TempFile(std::string(buf.data()));
  }

private:
  std::string m_path;
};

// ---------------------------------------------------------------------------
// Little-endian byte writers + synthetic BMP builder.
// ---------------------------------------------------------------------------

void appendLE16(std::vector<uint8_t> &buf, uint16_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFFu));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
}

void appendLE32(std::vector<uint8_t> &buf, uint32_t v) {
  buf.push_back(static_cast<uint8_t>(v & 0xFFu));
  buf.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
  buf.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
  buf.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
}

void storeLE32(std::vector<uint8_t> &buf, std::size_t offset, uint32_t v) {
  if (offset + sizeof(v) > buf.size()) {
    throw std::runtime_error("storeLE32: offset is outside the buffer");
  }
  buf[offset] = static_cast<uint8_t>(v & 0xFFu);
  buf[offset + 1u] = static_cast<uint8_t>((v >> 8) & 0xFFu);
  buf[offset + 2u] = static_cast<uint8_t>((v >> 16) & 0xFFu);
  buf[offset + 3u] = static_cast<uint8_t>((v >> 24) & 0xFFu);
}

// Build a complete 8-bit uncompressed BMP (40-byte info header) in memory.
// `palette` must hold `colorsUsed` entries of (B, G, R).
// `pixelIndices` must hold width*height bytes, in stored (file) row order.
std::vector<uint8_t>
buildSyntheticBMP(uint32_t width, uint32_t height, bool rowsBottomUp,
                  uint32_t colorsUsed,
                  const std::vector<std::array<uint8_t, 3>> &palette,
                  const std::vector<uint8_t> &pixelIndices) {
  if (palette.size() < colorsUsed) {
    throw std::runtime_error("buildSyntheticBMP: palette too small");
  }
  if (pixelIndices.size() < static_cast<std::size_t>(width) * height) {
    throw std::runtime_error("buildSyntheticBMP: pixelIndices too small");
  }
  const uint32_t rowStride = ((width + 3u) / 4u) * 4u;
  const uint32_t paletteBytes = colorsUsed * 4u;
  const uint32_t pixelOffset = 14u + 40u + paletteBytes;
  const uint32_t imageSize = rowStride * height;
  const uint32_t fileSize = pixelOffset + imageSize;

  std::vector<uint8_t> buf;
  buf.reserve(fileSize);

  // BMPFileHeader (14 bytes)
  buf.push_back('B');
  buf.push_back('M');
  appendLE32(buf, fileSize);
  appendLE32(buf, 0u); // reserved
  appendLE32(buf, pixelOffset);

  // BMPInfoHeader (40 bytes)
  appendLE32(buf, 40u);                       // size
  appendLE32(buf, width);                     // width (int32, positive)
  const int32_t signedHeight =
      rowsBottomUp ? static_cast<int32_t>(height) : -static_cast<int32_t>(height);
  appendLE32(buf, static_cast<uint32_t>(signedHeight));
  appendLE16(buf, 1u);                        // planes
  appendLE16(buf, 8u);                        // bitsPerPixel
  appendLE32(buf, 0u);                        // compression (BI_RGB)
  appendLE32(buf, imageSize);                 // imageSize
  appendLE32(buf, 0u);                        // xPixelsPerM
  appendLE32(buf, 0u);                        // yPixelsPerM
  appendLE32(buf, colorsUsed);                // colorsUsed
  appendLE32(buf, 0u);                        // importantColors

  // Palette (BGR0 per entry)
  for (uint32_t i = 0; i < colorsUsed; ++i) {
    buf.push_back(palette[i][0]); // B
    buf.push_back(palette[i][1]); // G
    buf.push_back(palette[i][2]); // R
    buf.push_back(0u);             // reserved
  }

  // Pixel rows with 4-byte padding
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      buf.push_back(pixelIndices[static_cast<std::size_t>(row) * width + col]);
    }
    for (uint32_t p = width; p < rowStride; ++p) {
      buf.push_back(0u);
    }
  }
  return buf;
}

void writeAllBytes(const std::string &path, const std::vector<uint8_t> &bytes) {
  FILE *fp = std::fopen(path.c_str(), "wb");
  if (!fp) {
    throw std::runtime_error("writeAllBytes: failed to open " + path);
  }
  if (!bytes.empty() &&
      std::fwrite(bytes.data(), 1, bytes.size(), fp) != bytes.size()) {
    std::fclose(fp);
    throw std::runtime_error("writeAllBytes: short write to " + path);
  }
  std::fclose(fp);
}

// Open a BMP file (path) read-only; throws on failure.
int openReadOnly(const std::string &path) {
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("openReadOnly: failed to open " + path);
  }
  return fd;
}

int openFirstCandidate(const std::vector<std::string> &paths) {
  for (const std::string &p : paths) {
    int fd = ::open(p.c_str(), O_RDONLY);
    if (fd >= 0) {
      return fd;
    }
  }
  return -1;
}

// Build a 256-entry palette where entry i maps to BGR = (i, 2i, 3i) mod 256.
std::vector<std::array<uint8_t, 3>> makeIdentityPalette() {
  std::vector<std::array<uint8_t, 3>> pal(256);
  for (unsigned i = 0; i < 256; ++i) {
    pal[i][0] = static_cast<uint8_t>(i);          // B
    pal[i][1] = static_cast<uint8_t>((i * 2u) & 0xFFu); // G
    pal[i][2] = static_cast<uint8_t>((i * 3u) & 0xFFu); // R
  }
  return pal;
}

// ---------------------------------------------------------------------------
// CLI: path to the real lena fixture (skipped if absent).
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string>
    LenaPath("lena",
             llvm::cl::desc("Path to a real 8-bit BMP for the smoke test"),
             llvm::cl::init("../../../tools/image/lena_gray.bmp"));

// ---------------------------------------------------------------------------
// Test 1: getBMP24RowStride (pure function)
// ---------------------------------------------------------------------------

void testGetBMP24RowStride() {
  struct Case {
    uint32_t width;
    uint32_t expected;
  };
  const Case cases[] = {
      {1u, 4u},   // 3  -> 4
      {2u, 8u},   // 6  -> 8
      {3u, 12u},  // 9  -> 12
      {4u, 12u},  // 12 -> 12
      {5u, 16u},  // 15 -> 16
      {6u, 20u},  // 18 -> 20
      {512u, 1536u}, // 1536 -> 1536
  };
  for (const Case &c : cases) {
    const uint32_t got = image::getBMP24RowStride(c.width);
    CHECK(got == c.expected,
          "getBMP24RowStride(" + std::to_string(c.width) +
              ") = " + std::to_string(got) + ", expected " +
              std::to_string(c.expected));
  }
  expect_throw("getBMP24RowStride(0) throws", [](uint32_t w) {
    image::getBMP24RowStride(w);
  }, 0u);
}

// ---------------------------------------------------------------------------
// Test 2: readBMPHeader on a valid synthetic BMP (bottom-up and top-down)
// ---------------------------------------------------------------------------

void testReadBMPHeaderValid() {
  const uint32_t width = 4;
  const uint32_t height = 3;
  const uint32_t colorsUsed = 4;
  std::vector<std::array<uint8_t, 3>> palette(256, {0u, 0u, 0u});
  palette[0] = {10u, 20u, 30u};  // B, G, R
  palette[1] = {40u, 50u, 60u};
  palette[2] = {70u, 80u, 90u};
  palette[3] = {100u, 110u, 120u};
  std::vector<uint8_t> indices(width * height, 0u);
  for (uint8_t i = 0; i < width * height; ++i) {
    indices[i] = i;
  }

  for (bool bottomUp : {true, false}) {
    std::vector<uint8_t> bmp = buildSyntheticBMP(width, height, bottomUp,
                                                colorsUsed, palette, indices);
    TempFile tmp = TempFile::create("hdrvalid");
    writeAllBytes(tmp.path(), bmp);

    int fd = openReadOnly(tmp.path());
    image::BMPInfo info;
    run_test("readBMPHeader(valid)", [&]() {
      image::readBMPHeader(fd, info);
    });
    ::close(fd);
    if (g_failureCount) return;

    CHECK(info.width == width, "width mismatch");
    CHECK(info.height == height, "height mismatch");
    CHECK(info.rowStride == width, "rowStride mismatch"); // 4 -> 4
    CHECK(info.pixelOffset == 14u + 40u + colorsUsed * 4u,
          "pixelOffset mismatch");
    CHECK(info.numColors == colorsUsed, "numColors mismatch");
    CHECK(info.rowsBottomUp == bottomUp, "rowsBottomUp mismatch");
    CHECK(info.bTable.size() == 256u, "bTable size");
    CHECK(info.gTable.size() == 256u, "gTable size");
    CHECK(info.rTable.size() == 256u, "rTable size");
    for (uint32_t i = 0; i < colorsUsed; ++i) {
      CHECK(info.bTable[i] == palette[i][0], "bTable entry");
      CHECK(info.gTable[i] == palette[i][1], "gTable entry");
      CHECK(info.rTable[i] == palette[i][2], "rTable entry");
    }
    for (uint32_t i = colorsUsed; i < 256u; ++i) {
      CHECK(info.bTable[i] == 0u && info.gTable[i] == 0u &&
                info.rTable[i] == 0u,
            "unused palette entry not zero");
    }
  }
}

// ---------------------------------------------------------------------------
// Test 3: readBMPHeader on the real lena fixture (skipped if absent)
// ---------------------------------------------------------------------------

void testReadBMPHeaderLena() {
  std::vector<std::string> candidates = {
      LenaPath,                              // explicit --lena override
      "../../../tools/image/lena_gray.bmp",  // CWD = build/tests/bin (ctest)
      "../../tools/image/lena_gray.bmp",     // CWD = build/tests
      "../tools/image/lena_gray.bmp",        // CWD = build
      "tools/image/lena_gray.bmp",           // CWD = repo root
  };
  int fd = openFirstCandidate(candidates);
  if (fd < 0) {
    llvm::errs() << "[SKIP] testReadBMPHeaderLena: lena fixture not found (tried "
                 << LenaPath << " and fallbacks)\n";
    return;
  }
  image::BMPInfo info;
  run_test("readBMPHeader(lena)", [&]() {
    image::readBMPHeader(fd, info);
  });
  ::close(fd);
  if (g_failureCount) return;

  CHECK(info.width == 512u, "lena width");
  CHECK(info.height == 512u, "lena height");
  CHECK(info.rowStride == 512u, "lena rowStride");
  CHECK(info.pixelOffset == 1078u, "lena pixelOffset");
  CHECK(info.numColors == 256u, "lena numColors");
  CHECK(info.rowsBottomUp == true, "lena rowsBottomUp");
  CHECK(info.bTable.size() == 256u, "lena bTable size");
  // Grayscale palette: R == G == B for every entry.
  for (uint32_t i = 0; i < 256u; ++i) {
    CHECK(info.bTable[i] == info.gTable[i] && info.gTable[i] == info.rTable[i],
          "lena palette not grayscale at entry " + std::to_string(i));
  }
}

// ---------------------------------------------------------------------------
// Test 4: readBMPHeader error paths (malformed synthetic BMPs)
// ---------------------------------------------------------------------------

void writeAndOpen(const std::vector<uint8_t> &bytes, int &fd, TempFile &tmp) {
  tmp = TempFile::create("malformed");
  writeAllBytes(tmp.path(), bytes);
  fd = openReadOnly(tmp.path());
}

void testReadBMPHeaderErrors() {
  auto palette256 = makeIdentityPalette();
  std::vector<uint8_t> indicesGood(8, 0u);

  auto goodBMP = [&]() {
    return buildSyntheticBMP(4, 2, true, 256u, palette256, indicesGood);
  };

  // Bad signature
  {
    std::vector<uint8_t> b = goodBMP();
    b[0] = 'P';
    b[1] = 'X';
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(bad signature)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // Unsupported bitsPerPixel (24)
  {
    std::vector<uint8_t> b = goodBMP();
    const std::size_t bppOff = 14u + 14u; // info header: size(4)+width(4)+height(4)+planes(2) = 14
    b[bppOff] = 24u;
    b[bppOff + 1u] = 0u;
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(bad bpp)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // Non-zero compression
  {
    std::vector<uint8_t> b = goodBMP();
    const std::size_t compOff = 14u + 16u; // size(4)+width(4)+height(4)+planes(2)+bpp(2) = 16
    b[compOff] = 1u;
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(compressed)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // planes != 1
  {
    std::vector<uint8_t> b = goodBMP();
    const std::size_t planesOff = 14u + 12u; // size(4)+width(4)+height(4) = 12
    b[planesOff] = 2u;
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(bad planes)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // info header size != 40
  {
    std::vector<uint8_t> b = goodBMP();
    b[14u] = 100u; // size field low byte
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(bad info size)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // width <= 0
  {
    std::vector<uint8_t> b = goodBMP();
    const std::size_t widthOff = 14u + 4u; // after size(4)
    b[widthOff] = 0u; b[widthOff + 1u] = 0u; b[widthOff + 2u] = 0u; b[widthOff + 3u] = 0u;
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(zero width)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // Truncated file header (only 10 bytes)
  {
    std::vector<uint8_t> b(10, 0u);
    b[0] = 'B'; b[1] = 'M';
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(truncated)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // Truncated info header.
  {
    std::vector<uint8_t> b = goodBMP();
    b.resize(14u + 20u);
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(truncated info header)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }

  // Truncated color table.
  {
    std::vector<uint8_t> b = goodBMP();
    b.resize(14u + 40u + 100u);
    int fd = -1;
    TempFile tmp;
    writeAndOpen(b, fd, tmp);
    image::BMPInfo info;
    expect_throw("readBMPHeader(truncated color table)", [&](int f) {
      image::readBMPHeader(f, info);
    }, fd);
    ::close(fd);
  }
}

void testReadBMPPaletteSemantics() {
  constexpr std::size_t ColorsUsedOffset = 14u + 32u;
  const uint32_t width = 4u;
  const uint32_t height = 2u;
  std::vector<uint8_t> indices(width * height, 0u);

  // A zero colorsUsed field means that all 256 palette entries are present.
  {
    const auto palette = makeIdentityPalette();
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, true, 256u, palette, indices);
    storeLE32(bmp, ColorsUsedOffset, 0u);

    TempFile tmp = TempFile::create("palette_default");
    writeAllBytes(tmp.path(), bmp);
    const int fd = openReadOnly(tmp.path());
    image::BMPInfo info;
    image::readBMPHeader(fd, info);
    ::close(fd);

    CHECK(info.numColors == 256u, "colorsUsed=0 should load 256 entries");
    CHECK(info.bTable[255] == palette[255][0],
          "colorsUsed=0 blue palette entry");
    CHECK(info.gTable[255] == palette[255][1],
          "colorsUsed=0 green palette entry");
    CHECK(info.rTable[255] == palette[255][2],
          "colorsUsed=0 red palette entry");
  }

  // Preserve compatibility with oversized declarations by loading the first
  // 256 entries and seeking past the complete declared table to pixel data.
  {
    constexpr uint32_t DeclaredColors = 300u;
    auto palette = makeIdentityPalette();
    palette.resize(DeclaredColors, {17u, 34u, 51u});
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, true, DeclaredColors, palette, indices);

    TempFile tmp = TempFile::create("palette_oversized");
    writeAllBytes(tmp.path(), bmp);
    const int fd = openReadOnly(tmp.path());
    image::BMPInfo info;
    image::readBMPHeader(fd, info);
    const off_t currentOffset = ::lseek(fd, 0, SEEK_CUR);
    ::close(fd);

    CHECK(info.numColors == 256u,
          "oversized colorsUsed should be capped at 256");
    CHECK(info.bTable[255] == palette[255][0],
          "oversized palette blue entry");
    CHECK(info.gTable[255] == palette[255][1],
          "oversized palette green entry");
    CHECK(info.rTable[255] == palette[255][2],
          "oversized palette red entry");
    CHECK(currentOffset == static_cast<off_t>(info.pixelOffset),
          "readBMPHeader should seek to pixel data after oversized palette");
  }
}

// ---------------------------------------------------------------------------
// Pipeline helper: open a synthetic BMP, read its header, run
// ParseBMPColorStreams + CreateBMPColorByteStreams, and return the R/G/B
// StreamSetPtrs and the parsed BMPInfo.
// ---------------------------------------------------------------------------

struct ColorPipelineResult {
  kernel::StreamSetPtr redBytes;
  kernel::StreamSetPtr greenBytes;
  kernel::StreamSetPtr blueBytes;
  image::BMPInfo info;
};

ColorPipelineResult runColorPipeline(CPUDriver &driver, const std::string &path) {
  ColorPipelineResult result{};
  int fd = openReadOnly(path);
  image::readBMPHeader(fd, result.info);

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
  image::ParseBMPColorStreams(P, fdScalar, result.info, colorStream);

  image::CreateBMPColorByteStreams(P, colorStream,
                                   P.getOutputStreamSet("redBytes"),
                                   P.getOutputStreamSet("greenBytes"),
                                   P.getOutputStreamSet("blueBytes"));

  auto pipelineFn = P.compile();
  pipelineFn(result.redBytes, result.greenBytes, result.blueBytes,
             static_cast<uint32_t>(fd));
  ::close(fd);
  return result;
}

// ---------------------------------------------------------------------------
// Test 5: ParseBMPColorStreams + CreateBMPColorByteStreams (no padding)
//
// Top-down 4x2 BMP, palette[i] = BGR(i, 2i, 3i), pixel index at stored
// position i == i. So color-stream pixel i must be BGR(i, 2i, 3i).
// ---------------------------------------------------------------------------

void testParseBMPColorStreamsAndByteStreams(CPUDriver &driver) {
  const uint32_t width = 4;
  const uint32_t height = 2;
  const uint32_t pixels = width * height;
  auto palette = makeIdentityPalette();
  std::vector<uint8_t> indices(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    indices[i] = static_cast<uint8_t>(i);
  }
  std::vector<uint8_t> bmp =
      buildSyntheticBMP(width, height, /*rowsBottomUp=*/false, 256u, palette, indices);
  TempFile tmp = TempFile::create("color");
  writeAllBytes(tmp.path(), bmp);

  ColorPipelineResult r = runColorPipeline(driver, tmp.path());
  if (g_failureCount) return;

  CHECK(r.info.width == width, "width");
  CHECK(r.info.height == height, "height");
  CHECK(r.redBytes.length() == pixels, "red length");
  CHECK(r.greenBytes.length() == pixels, "green length");
  CHECK(r.blueBytes.length() == pixels, "blue length");

  const uint8_t *red = r.redBytes.data();
  const uint8_t *green = r.greenBytes.data();
  const uint8_t *blue = r.blueBytes.data();
  for (uint32_t i = 0; i < pixels; ++i) {
    const uint8_t expB = static_cast<uint8_t>(i);
    const uint8_t expG = static_cast<uint8_t>((i * 2u) & 0xFFu);
    const uint8_t expR = static_cast<uint8_t>((i * 3u) & 0xFFu);
    if (blue[i] != expB || green[i] != expG || red[i] != expR) {
      llvm::errs() << "[FAIL] testParseBMPColorStreamsAndByteStreams: pixel "
                   << i << " got BGR(" << static_cast<unsigned>(blue[i]) << ","
                   << static_cast<unsigned>(green[i]) << ","
                   << static_cast<unsigned>(red[i]) << ") expected BGR("
                   << static_cast<unsigned>(expB) << ","
                   << static_cast<unsigned>(expG) << ","
                   << static_cast<unsigned>(expR) << ")\n";
      ++g_failureCount;
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Test 6: ParseBMPBuffer padding removal
//
// Top-down 5x2 BMP -> rowStride 8 (3 pad bytes/row). Color stream must have
// 10 pixels (padding removed) and pixel i must be BGR(i, 2i, 3i).
// ---------------------------------------------------------------------------

void testParseBMPBufferPaddingRemoval(CPUDriver &driver) {
  const uint32_t width = 5;
  const uint32_t height = 2;
  const uint32_t pixels = width * height;
  auto palette = makeIdentityPalette();
  std::vector<uint8_t> indices(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    indices[i] = static_cast<uint8_t>(i);
  }
  std::vector<uint8_t> bmp =
      buildSyntheticBMP(width, height, /*rowsBottomUp=*/false, 256u, palette, indices);
  TempFile tmp = TempFile::create("padding");
  writeAllBytes(tmp.path(), bmp);

  ColorPipelineResult r = runColorPipeline(driver, tmp.path());
  if (g_failureCount) return;

  CHECK(r.info.width == width, "width");
  CHECK(r.info.height == height, "height");
  CHECK(r.info.rowStride == 8u, "rowStride should be 8 (padded)");
  // The padded-row path through ParseBMPBuffer filters out the 3 pad bytes per
  // row, so the logical pixel count is width*height.  The returned buffer may
  // be over-allocated, so we only require it to hold at least that many items
  // and then verify the leading pixel values exactly.
  CHECK(r.redBytes.length() >= pixels, "red length (padding not removed)");
  CHECK(r.greenBytes.length() >= pixels, "green length");
  CHECK(r.blueBytes.length() >= pixels, "blue length");

  const uint8_t *red = r.redBytes.data();
  const uint8_t *green = r.greenBytes.data();
  const uint8_t *blue = r.blueBytes.data();
  for (uint32_t i = 0; i < pixels; ++i) {
    const uint8_t expB = static_cast<uint8_t>(i);
    const uint8_t expG = static_cast<uint8_t>((i * 2u) & 0xFFu);
    const uint8_t expR = static_cast<uint8_t>((i * 3u) & 0xFFu);
    if (blue[i] != expB || green[i] != expG || red[i] != expR) {
      llvm::errs() << "[FAIL] testParseBMPBufferPaddingRemoval: pixel " << i
                   << " BGR mismatch\n";
      ++g_failureCount;
      return;
    }
  }
}

void testCreateBMPColorByteStreamsValidation(CPUDriver &driver) {
  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *invalidColorStream = P.CreateStreamSet(23);
    kernel::StreamSet *redBytes = P.CreateStreamSet(1, 8);
    kernel::StreamSet *greenBytes = P.CreateStreamSet(1, 8);
    kernel::StreamSet *blueBytes = P.CreateStreamSet(1, 8);
    expect_throw("CreateBMPColorByteStreams(invalid color stream)", [&]() {
      image::CreateBMPColorByteStreams(P, invalidColorStream, redBytes,
                                       greenBytes, blueBytes);
    });
  }

  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *colorStream = P.CreateStreamSet(24);
    kernel::StreamSet *invalidRedBytes = P.CreateStreamSet(1, 7);
    kernel::StreamSet *greenBytes = P.CreateStreamSet(1, 8);
    kernel::StreamSet *blueBytes = P.CreateStreamSet(1, 8);
    expect_throw("CreateBMPColorByteStreams(invalid byte stream)", [&]() {
      image::CreateBMPColorByteStreams(P, colorStream, invalidRedBytes,
                                       greenBytes, blueBytes);
    });
  }
}

// ---------------------------------------------------------------------------
// Test 7: CropImage input validation
//
// CropImage validates its crop rectangle (and source stream shape) and throws
// std::runtime_error synchronously during pipeline construction, before any
// JIT compilation.  We exercise those validation paths here.
// ---------------------------------------------------------------------------

void expectCropThrow(CPUDriver &driver, const image::BMPInfo &info,
                     uint32_t cropW, uint32_t cropH, uint32_t cropX,
                     uint32_t cropY, const char *label) {
  auto P = kernel::CreatePipeline(driver,
                                  kernel::Input<uint32_t>{"fd"});
  kernel::Scalar *fdScalar = P.getInputScalar("fd");
  kernel::StreamSet *colorStream = nullptr;
  image::ParseBMPColorStreams(P, fdScalar, info, colorStream);
  kernel::StreamSet *croppedStream = nullptr;
  expect_throw(label, [&]() {
    image::CropImage(P, colorStream, info, cropW, cropH, cropX, cropY,
                     croppedStream);
  });
}

void testCropImage(CPUDriver &driver) {
  // Use a width that is a multiple of 4 so ParseBMPBuffer takes its no-padding
  // fast path; we never compile/run, so the fd/pixels are irrelevant.
  const uint32_t width = 8;
  const uint32_t height = 4;
  auto palette = makeIdentityPalette();
  std::vector<uint8_t> indices(width * height, 0u);
  std::vector<uint8_t> bmp =
      buildSyntheticBMP(width, height, /*rowsBottomUp=*/false, 256u, palette, indices);
  TempFile tmp = TempFile::create("crop");
  writeAllBytes(tmp.path(), bmp);

  int fd = openReadOnly(tmp.path());
  image::BMPInfo info;
  image::readBMPHeader(fd, info);
  ::close(fd);
  if (g_failureCount) return;

  CHECK(info.width == width && info.height == height, "header for crop");

  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *invalidColorStream = P.CreateStreamSet(23);
    kernel::StreamSet *croppedStream = nullptr;
    expect_throw("CropImage(invalid color stream)", [&]() {
      image::CropImage(P, invalidColorStream, info, 2u, 2u, 0u, 0u,
                       croppedStream);
    });
  }

  // Zero-sized crop must be rejected.
  expectCropThrow(driver, info, 0u, 2u, 0u, 0u, "CropImage(zero width)");
  expectCropThrow(driver, info, 3u, 0u, 0u, 0u, "CropImage(zero height)");

  // Crop rectangle exceeding the source bounds must be rejected.
  expectCropThrow(driver, info, 1u, 1u, width + 1u, 0u,
                  "CropImage(cropX > width)");
  expectCropThrow(driver, info, 1u, 1u, 0u, height + 1u,
                  "CropImage(cropY > height)");
  expectCropThrow(driver, info, 2u, 2u, width - 1u, 0u,
                  "CropImage(cropX+cropW > width)");
  expectCropThrow(driver, info, 2u, 2u, 0u, height - 1u,
                  "CropImage(cropY+cropH > height)");
}

// ---------------------------------------------------------------------------
// Helpers for the output (24-bit) side
// ---------------------------------------------------------------------------

image::BMPInfo makeOutputInfo(uint32_t width, uint32_t height, bool bottomUp) {
  image::BMPInfo info{};
  info.width = width;
  info.height = height;
  info.rowStride = ((width + 3u) / 4u) * 4u;
  info.pixelOffset = 14u + 40u;
  info.paletteOffset = 0u;
  info.numColors = 0u;
  info.rowsBottomUp = bottomUp;
  return info;
}

// ---------------------------------------------------------------------------
// Test 8: createBMP24PixelData (host)
//
// 3x2 image, distinct R/G/B per pixel. Verify size, BGR ordering, zero
// padding, and that mismatched channel lengths throw.
// ---------------------------------------------------------------------------

void testCreateBMP24PixelData() {
  const uint32_t width = 3;   // rowBytes = 9 -> rowStride 12 (3 pad bytes)
  const uint32_t height = 2;
  const uint32_t pixels = width * height;
  std::vector<uint8_t> red(pixels), green(pixels), blue(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    red[i] = static_cast<uint8_t>(0x10 + i);
    green[i] = static_cast<uint8_t>(0x20 + i);
    blue[i] = static_cast<uint8_t>(0x30 + i);
  }
  kernel::StreamSetPtr redPtr(red.data(), pixels);
  kernel::StreamSetPtr greenPtr(green.data(), pixels);
  kernel::StreamSetPtr bluePtr(blue.data(), pixels);

  image::BMPInfo info = makeOutputInfo(width, height, true);
  const uint32_t rowStride = image::getBMP24RowStride(width);
  CHECK(rowStride == 12u, "rowStride for width 3");

  std::vector<uint8_t> pixelData;
  run_test("createBMP24PixelData", [&]() {
    pixelData = image::createBMP24PixelData(redPtr, greenPtr, bluePtr, info);
  });
  if (g_failureCount) return;

  const uint32_t expectedSize = rowStride * height;
  CHECK(pixelData.size() == expectedSize, "pixel data size");

  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      const uint32_t inIdx = row * width + col;
      const uint32_t outIdx = row * rowStride + col * 3u;
      CHECK(pixelData[outIdx + 0u] == blue[inIdx], "B byte");
      CHECK(pixelData[outIdx + 1u] == green[inIdx], "G byte");
      CHECK(pixelData[outIdx + 2u] == red[inIdx], "R byte");
    }
    for (uint32_t p = width * 3u; p < rowStride; ++p) {
      CHECK(pixelData[row * rowStride + p] == 0u, "padding not zero");
    }
  }

  // Mismatched channel lengths must throw.
  kernel::StreamSetPtr shortPtr(red.data(), pixels - 1u);
  expect_throw("createBMP24PixelData(length mismatch)", [&]() {
    image::createBMP24PixelData(shortPtr, greenPtr, bluePtr, info);
  });
}

// ---------------------------------------------------------------------------
// Test 9: writeBMP24 round-trip (host)
//
// Build 24-bit pixel data, write it, re-open the file, and verify the 14+40
// byte headers and the pixel bytes round-trip exactly. Also check that an
// empty output path throws.
// ---------------------------------------------------------------------------

void testWriteBMP24RoundTrip() {
  const uint32_t width = 2;   // rowBytes = 6 -> rowStride 8 (2 pad bytes)
  const uint32_t height = 2;
  const uint32_t pixels = width * height;
  std::vector<uint8_t> red(pixels, 0xAA);
  std::vector<uint8_t> green(pixels, 0xBB);
  std::vector<uint8_t> blue(pixels, 0xCC);
  kernel::StreamSetPtr redPtr(red.data(), pixels);
  kernel::StreamSetPtr greenPtr(green.data(), pixels);
  kernel::StreamSetPtr bluePtr(blue.data(), pixels);

  image::BMPInfo info = makeOutputInfo(width, height, true);
  std::vector<uint8_t> pixelData =
      image::createBMP24PixelData(redPtr, greenPtr, bluePtr, info);
  const uint32_t rowStride = image::getBMP24RowStride(width);

  TempFile tmp = TempFile::create("writert");
  run_test("writeBMP24", [&]() {
    image::writeBMP24(tmp.path(), pixelData, info);
  });
  if (g_failureCount) return;

  // Re-read and verify headers + pixel bytes.
  int fd = openReadOnly(tmp.path());
  std::vector<uint8_t> fileBuf(pixelData.size() + 54u, 0u);
  ssize_t total = 0;
  while (static_cast<std::size_t>(total) < fileBuf.size()) {
    ssize_t n = ::read(fd, fileBuf.data() + total, fileBuf.size() - total);
    if (n <= 0) break;
    total += n;
  }
  ::close(fd);
  CHECK(static_cast<std::size_t>(total) == fileBuf.size(), "file size");

  CHECK(fileBuf[0] == 'B' && fileBuf[1] == 'M', "signature");
  const uint32_t fileSize = fileBuf[2] | (fileBuf[3] << 8) | (fileBuf[4] << 16) |
                            (static_cast<uint32_t>(fileBuf[5]) << 24);
  const uint32_t dataOffset = fileBuf[10] | (fileBuf[11] << 8) |
                              (fileBuf[12] << 16) |
                              (static_cast<uint32_t>(fileBuf[13]) << 24);
  CHECK(fileSize == 54u + pixelData.size(), "fileSize");
  CHECK(dataOffset == 54u, "dataOffset");

  const uint32_t ihSize = fileBuf[14] | (fileBuf[15] << 8) | (fileBuf[16] << 16) |
                          (static_cast<uint32_t>(fileBuf[17]) << 24);
  const int32_t ihWidth = static_cast<int32_t>(
      fileBuf[18] | (fileBuf[19] << 8) | (fileBuf[20] << 16) |
      (static_cast<uint32_t>(fileBuf[21]) << 24));
  const int32_t ihHeight = static_cast<int32_t>(
      fileBuf[22] | (fileBuf[23] << 8) | (fileBuf[24] << 16) |
      (static_cast<uint32_t>(fileBuf[25]) << 24));
  const uint16_t ihPlanes = fileBuf[26] | (fileBuf[27] << 8);
  const uint16_t ihBPP = fileBuf[28] | (fileBuf[29] << 8);
  const uint32_t ihComp = fileBuf[30] | (fileBuf[31] << 8) | (fileBuf[32] << 16) |
                          (static_cast<uint32_t>(fileBuf[33]) << 24);
  CHECK(ihSize == 40u, "info header size");
  CHECK(ihWidth == static_cast<int32_t>(width), "info width");
  CHECK(ihHeight == static_cast<int32_t>(height), "info height (bottom-up)");
  CHECK(ihPlanes == 1u, "info planes");
  CHECK(ihBPP == 24u, "info bitsPerPixel");
  CHECK(ihComp == 0u, "info compression");

  for (uint32_t i = 0; i < pixelData.size(); ++i) {
    CHECK(fileBuf[54u + i] == pixelData[i], "pixel byte round-trip");
  }
  (void)rowStride;

  // Empty output path must throw.
  std::vector<uint8_t> dummy = image::createBMP24PixelData(redPtr, greenPtr, bluePtr, info);
  expect_throw("writeBMP24(empty path)", [&]() {
    image::writeBMP24("", dummy, info);
  });
}

// ---------------------------------------------------------------------------
// Test 10: createBMP24Image (host)
//
// Convert planar R/G/B byte streams in BMP stored row order into a top-down
// interleaved RGB BMP24Image, for both bottom-up and top-down sources.
// ---------------------------------------------------------------------------

void testCreateBMP24Image() {
  const uint32_t width = 3;
  const uint32_t height = 2;
  const uint32_t pixels = width * height;
  std::vector<uint8_t> red(pixels), green(pixels), blue(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    red[i] = static_cast<uint8_t>(0x10 + i);
    green[i] = static_cast<uint8_t>(0x20 + i);
    blue[i] = static_cast<uint8_t>(0x30 + i);
  }

  for (bool bottomUp : {true, false}) {
    kernel::StreamSetPtr redPtr(red.data(), pixels);
    kernel::StreamSetPtr greenPtr(green.data(), pixels);
    kernel::StreamSetPtr bluePtr(blue.data(), pixels);

    image::BMP24Image img;
    run_test("createBMP24Image", [&]() {
      img = image::createBMP24Image(redPtr, greenPtr, bluePtr, width, height,
                                    bottomUp);
    });
    if (g_failureCount) return;

    CHECK(img.width == width, "image width");
    CHECK(img.height == height, "image height");
    CHECK(img.rowsBottomUp == bottomUp, "image rowsBottomUp");
    CHECK(img.rgb.size() == pixels * 3u, "rgb buffer size");

    for (uint32_t row = 0; row < height; ++row) {
      const uint32_t storedRow =
          bottomUp ? height - row - 1u : row;
      for (uint32_t col = 0; col < width; ++col) {
        const uint32_t storedIdx = storedRow * width + col;
        const uint32_t outIdx = (row * width + col) * 3u;
        CHECK(img.rgb[outIdx + 0u] == red[storedIdx], "R byte top-down");
        CHECK(img.rgb[outIdx + 1u] == green[storedIdx], "G byte top-down");
        CHECK(img.rgb[outIdx + 2u] == blue[storedIdx], "B byte top-down");
      }
    }
  }

  // Over-allocated channel buffers (pipeline may produce these) must work;
  // only the first width*height bytes are read.
  {
    std::vector<uint8_t> bigRed(pixels + 5u, 0u), bigGreen(pixels + 5u, 0u),
        bigBlue(pixels + 5u, 0u);
    for (uint32_t i = 0; i < pixels; ++i) {
      bigRed[i] = red[i];
      bigGreen[i] = green[i];
      bigBlue[i] = blue[i];
    }
    kernel::StreamSetPtr redPtr(bigRed.data(), bigRed.size());
    kernel::StreamSetPtr greenPtr(bigGreen.data(), bigGreen.size());
    kernel::StreamSetPtr bluePtr(bigBlue.data(), bigBlue.size());
    image::BMP24Image img;
    run_test("createBMP24Image(over-allocated)", [&]() {
      img = image::createBMP24Image(redPtr, greenPtr, bluePtr, width, height,
                                    true);
    });
    if (g_failureCount) return;
    CHECK(img.rgb.size() == pixels * 3u, "over-allocated rgb size");
    // Bottom-up: top-down row 0 is stored row (height-1) = row 1, col 0 ->
    // stored index `width`.
    CHECK(img.rgb[0] == red[width] && img.rgb[1] == green[width] &&
              img.rgb[2] == blue[width],
          "over-allocated first pixel (bottom-up row 0 = stored row 1)");
  }

  // Short channel buffers must throw.
  {
    kernel::StreamSetPtr shortPtr(red.data(), pixels - 1u);
    kernel::StreamSetPtr greenPtr(green.data(), pixels);
    kernel::StreamSetPtr bluePtr(blue.data(), pixels);
    expect_throw("createBMP24Image(length mismatch)", [&]() {
      image::createBMP24Image(shortPtr, greenPtr, bluePtr, width, height, true);
    });
  }
}

// ---------------------------------------------------------------------------
// Test 11: writeBMP24(path, BMP24Image) round-trip (host)
//
// Build a top-down RGB image, write it, re-open the file, and verify the
// headers (including the height sign for both orientations) and the pixel
// bytes round-trip with BGR ordering and zero padding.
// ---------------------------------------------------------------------------

void testWriteBMP24ImageRoundTrip() {
  const uint32_t width = 2;   // rowBytes = 6 -> rowStride 8 (2 pad bytes)
  const uint32_t height = 2;
  const uint32_t pixels = width * height;

  for (bool bottomUp : {true, false}) {
    image::BMP24Image img(width, height, bottomUp);
    for (uint32_t i = 0; i < pixels; ++i) {
      img.rgb[i * 3u + 0u] = static_cast<uint8_t>(0xA0 + i); // R
      img.rgb[i * 3u + 1u] = static_cast<uint8_t>(0xB0 + i); // G
      img.rgb[i * 3u + 2u] = static_cast<uint8_t>(0xC0 + i); // B
    }

    TempFile tmp = TempFile::create("imgwr");
    run_test("writeBMP24(BMP24Image)", [&]() {
      image::writeBMP24(tmp.path(), img);
    });
    if (g_failureCount) return;

    int fd = openReadOnly(tmp.path());
    const uint32_t rowStride = image::getBMP24RowStride(width);
    const uint32_t imageSize = rowStride * height;
    std::vector<uint8_t> fileBuf(imageSize + 54u, 0u);
    ssize_t total = 0;
    while (static_cast<std::size_t>(total) < fileBuf.size()) {
      ssize_t n = ::read(fd, fileBuf.data() + total, fileBuf.size() - total);
      if (n <= 0) break;
      total += n;
    }
    ::close(fd);
    CHECK(static_cast<std::size_t>(total) == fileBuf.size(), "file size");

    CHECK(fileBuf[0] == 'B' && fileBuf[1] == 'M', "signature");
    const int32_t ihHeight = static_cast<int32_t>(
        fileBuf[22] | (fileBuf[23] << 8) | (fileBuf[24] << 16) |
        (static_cast<uint32_t>(fileBuf[25]) << 24));
    const int32_t expectedSignedHeight =
        bottomUp ? static_cast<int32_t>(height) : -static_cast<int32_t>(height);
    CHECK(ihHeight == expectedSignedHeight, "info height sign/orientation");
    const uint16_t ihBPP = fileBuf[28] | (fileBuf[29] << 8);
    CHECK(ihBPP == 24u, "info bitsPerPixel");

    // Stored row order: bottom-up files store the bottom row first.
    for (uint32_t row = 0; row < height; ++row) {
      const uint32_t sourceRow =
          bottomUp ? height - row - 1u : row;
      for (uint32_t col = 0; col < width; ++col) {
        const uint32_t inIdx = (sourceRow * width + col) * 3u;
        const uint32_t outIdx = row * rowStride + col * 3u;
        CHECK(fileBuf[54u + outIdx + 0u] == img.rgb[inIdx + 2u], "B byte");
        CHECK(fileBuf[54u + outIdx + 1u] == img.rgb[inIdx + 1u], "G byte");
        CHECK(fileBuf[54u + outIdx + 2u] == img.rgb[inIdx + 0u], "R byte");
      }
      for (uint32_t p = width * 3u; p < rowStride; ++p) {
        CHECK(fileBuf[54u + row * rowStride + p] == 0u, "padding not zero");
      }
    }
  }

  // Empty output path must throw.
  image::BMP24Image img(2, 2, true);
  expect_throw("writeBMP24(BMP24Image empty path)", [&]() {
    image::writeBMP24("", img);
  });
}

// ---------------------------------------------------------------------------
// Test 12: LoadBMPCrop end-to-end (pipeline)
//
// Build a synthetic 8-bit BMP with the identity palette, crop a region, and
// verify the returned top-down RGB image matches the expected palette
// expansion for the cropped pixels. Exercises both orientations and the
// invalid-crop error path.
// ---------------------------------------------------------------------------

void testLoadBMPCrop(CPUDriver &driver) {
  const uint32_t width = 4;
  const uint32_t height = 4;
  const uint32_t pixels = width * height;
  auto palette = makeIdentityPalette();
  std::vector<uint8_t> indices(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    indices[i] = static_cast<uint8_t>(i);
  }

  const image::BMPCrop crop{2u, 2u, 1u, 1u};

  for (bool bottomUp : {true, false}) {
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, bottomUp, 256u, palette, indices);
    TempFile tmp = TempFile::create("loadcrop");
    writeAllBytes(tmp.path(), bmp);

    image::BMPCropResult result = image::LoadBMPCrop(driver, tmp.path(), crop);
    if (g_failureCount) return;

    CHECK(result.sourceInfo.width == width, "source width");
    CHECK(result.sourceInfo.height == height, "source height");
    CHECK(result.sourceInfo.rowsBottomUp == bottomUp, "source rowsBottomUp");
    CHECK(result.image.width == crop.width, "crop width");
    CHECK(result.image.height == crop.height, "crop height");
    CHECK(result.image.rowsBottomUp == bottomUp, "crop rowsBottomUp");
    CHECK(result.image.rgb.size() == crop.width * crop.height * 3u,
          "crop rgb size");

    for (uint32_t cr = 0; cr < crop.height; ++cr) {
      for (uint32_t cc = 0; cc < crop.width; ++cc) {
        const uint32_t logicalRow = crop.y + cr;
        const uint32_t logicalCol = crop.x + cc;
        const uint32_t storedRow =
            bottomUp ? height - 1u - logicalRow : logicalRow;
        const uint32_t idx = storedRow * width + logicalCol;
        const uint8_t expB = static_cast<uint8_t>(idx & 0xFFu);
        const uint8_t expG = static_cast<uint8_t>((idx * 2u) & 0xFFu);
        const uint8_t expR = static_cast<uint8_t>((idx * 3u) & 0xFFu);
        const uint32_t outIdx = (cr * crop.width + cc) * 3u;
        CHECK(result.image.rgb[outIdx + 0u] == expR, "crop R byte");
        CHECK(result.image.rgb[outIdx + 1u] == expG, "crop G byte");
        CHECK(result.image.rgb[outIdx + 2u] == expB, "crop B byte");
      }
    }
  }

  // Invalid crop (exceeds source bounds) must throw.
  {
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, true, 256u, palette, indices);
    TempFile tmp = TempFile::create("loadcropbad");
    writeAllBytes(tmp.path(), bmp);
    const image::BMPCrop badCrop{2u, 2u, width, 0u};
    expect_throw("LoadBMPCrop(cropX out of bounds)", [&]() {
      image::LoadBMPCrop(driver, tmp.path(), badCrop);
    });
  }

  // Missing input file must throw.
  expect_throw("LoadBMPCrop(missing file)", [&]() {
    image::LoadBMPCrop(driver, "/tmp/parabix_bmp_does_not_exist_XXXXXX",
                       crop);
  });
}

// ---------------------------------------------------------------------------
// Test 13: MaskImage pipeline behavior
//
// Build a synthetic 8-bit BMP with the identity palette (pixel index i ->
// BGR(i, 2i, 3i)), parse it into a 24x1 color stream, apply MaskImage with a
// repeating 1x1 mask, and convert the result back to channel bytes. Verify
// that mask-1 pixels are black, mask-0 pixels are unchanged, and the output
// position count is preserved. Uses non-black source pixels so polarity
// errors are observable.
// ---------------------------------------------------------------------------

struct MaskPipelineResult {
  kernel::StreamSetPtr redBytes;
  kernel::StreamSetPtr greenBytes;
  kernel::StreamSetPtr blueBytes;
  image::BMPInfo info;
};

MaskPipelineResult runMaskPipeline(CPUDriver &driver, const std::string &path,
                                   const std::vector<uint64_t> &maskPattern) {
  MaskPipelineResult result{};
  int fd = openReadOnly(path);
  image::readBMPHeader(fd, result.info);

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
  image::ParseBMPColorStreams(P, fdScalar, result.info, colorStream);

  kernel::StreamSet *maskStream = P.CreateRepeatingStreamSet(1, maskPattern);
  kernel::StreamSet *maskedStream = nullptr;
  image::MaskImage(P, colorStream, maskStream, maskedStream);

  image::CreateBMPColorByteStreams(P, maskedStream,
                                   P.getOutputStreamSet("redBytes"),
                                   P.getOutputStreamSet("greenBytes"),
                                   P.getOutputStreamSet("blueBytes"));

  auto pipelineFn = P.compile();
  pipelineFn(result.redBytes, result.greenBytes, result.blueBytes,
             static_cast<uint32_t>(fd));
  ::close(fd);
  return result;
}

void testMaskImage(CPUDriver &driver) {
  const uint32_t width = 4;   // multiple of 4 -> no row padding
  const uint32_t height = 4;
  const uint32_t pixels = width * height;
  auto palette = makeIdentityPalette();
  std::vector<uint8_t> indices(pixels);
  for (uint32_t i = 0; i < pixels; ++i) {
    indices[i] = static_cast<uint8_t>(i);
  }
  std::vector<uint8_t> bmp =
      buildSyntheticBMP(width, height, /*rowsBottomUp=*/false, 256u, palette,
                        indices);
  TempFile tmp = TempFile::create("mask");
  writeAllBytes(tmp.path(), bmp);

  auto expectedColor = [&](uint32_t i, uint8_t &expB, uint8_t &expG,
                           uint8_t &expR) {
    expB = static_cast<uint8_t>(i);
    expG = static_cast<uint8_t>((i * 2u) & 0xFFu);
    expR = static_cast<uint8_t>((i * 3u) & 0xFFu);
  };

  auto verifyAgainst = [&](const MaskPipelineResult &r,
                           const std::vector<uint64_t> &mask,
                           const char *label) {
    CHECK(r.info.width == width, std::string(label) + ": width");
    CHECK(r.info.height == height, std::string(label) + ": height");
    CHECK(r.redBytes.length() >= pixels, std::string(label) + ": red length");
    CHECK(r.greenBytes.length() >= pixels, std::string(label) + ": green length");
    CHECK(r.blueBytes.length() >= pixels, std::string(label) + ": blue length");
    if (g_failureCount) return;
    const uint8_t *red = r.redBytes.data();
    const uint8_t *green = r.greenBytes.data();
    const uint8_t *blue = r.blueBytes.data();
    for (uint32_t i = 0; i < pixels; ++i) {
      if (mask[i] == 1u) {
        if (red[i] != 0u || green[i] != 0u || blue[i] != 0u) {
          llvm::errs() << "[FAIL] " << label << ": pixel " << i
                       << " expected black, got BGR("
                       << static_cast<unsigned>(blue[i]) << ","
                       << static_cast<unsigned>(green[i]) << ","
                       << static_cast<unsigned>(red[i]) << ")\n";
          ++g_failureCount;
          return;
        }
      } else {
        uint8_t expB, expG, expR;
        expectedColor(i, expB, expG, expR);
        if (blue[i] != expB || green[i] != expG || red[i] != expR) {
          llvm::errs() << "[FAIL] " << label << ": pixel " << i
                       << " expected BGR(" << static_cast<unsigned>(expB) << ","
                       << static_cast<unsigned>(expG) << ","
                       << static_cast<unsigned>(expR) << ") got BGR("
                       << static_cast<unsigned>(blue[i]) << ","
                       << static_cast<unsigned>(green[i]) << ","
                       << static_cast<unsigned>(red[i]) << ")\n";
          ++g_failureCount;
          return;
        }
      }
    }
  };

  // Mixed mask: black out even-indexed pixels, keep odd-indexed pixels.
  {
    std::vector<uint64_t> maskPattern(pixels, 0u);
    for (uint32_t i = 0; i < pixels; ++i) {
      maskPattern[i] = (i % 2u == 0u) ? 1u : 0u;
    }
    MaskPipelineResult r = runMaskPipeline(driver, tmp.path(), maskPattern);
    if (g_failureCount) return;
    verifyAgainst(r, maskPattern, "MaskImage(mixed)");
  }

  // All-zero mask: nothing is blacked out, every pixel is unchanged.
  {
    std::vector<uint64_t> maskPattern(pixels, 0u);
    MaskPipelineResult r = runMaskPipeline(driver, tmp.path(), maskPattern);
    if (g_failureCount) return;
    verifyAgainst(r, maskPattern, "MaskImage(all-zero)");
  }

  // All-one mask: every pixel is blacked out.
  {
    std::vector<uint64_t> maskPattern(pixels, 1u);
    MaskPipelineResult r = runMaskPipeline(driver, tmp.path(), maskPattern);
    if (g_failureCount) return;
    verifyAgainst(r, maskPattern, "MaskImage(all-one)");
  }
}

// ---------------------------------------------------------------------------
// Test 14: MaskImage input validation
//
// MaskImage validates the source color stream and mask stream shapes and
// throws std::runtime_error synchronously during pipeline construction.
// ---------------------------------------------------------------------------

void testMaskImageValidation(CPUDriver &driver) {
  // Invalid source color stream (must be 24x1).
  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *invalidColorStream = P.CreateStreamSet(23);
    kernel::StreamSet *maskStream = P.CreateStreamSet(1);
    kernel::StreamSet *maskedStream = nullptr;
    expect_throw("MaskImage(invalid color stream)", [&]() {
      image::MaskImage(P, invalidColorStream, maskStream, maskedStream);
    });
  }

  // Invalid mask stream (wrong element count).
  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *colorStream = P.CreateStreamSet(24);
    kernel::StreamSet *invalidMask = P.CreateStreamSet(2);
    kernel::StreamSet *maskedStream = nullptr;
    expect_throw("MaskImage(invalid mask element count)", [&]() {
      image::MaskImage(P, colorStream, invalidMask, maskedStream);
    });
  }

  // Invalid mask stream (wrong field width).
  {
    auto P = kernel::CreatePipeline(driver);
    kernel::StreamSet *colorStream = P.CreateStreamSet(24);
    kernel::StreamSet *invalidMask = P.CreateStreamSet(1, 8);
    kernel::StreamSet *maskedStream = nullptr;
    expect_throw("MaskImage(invalid mask field width)", [&]() {
      image::MaskImage(P, colorStream, invalidMask, maskedStream);
    });
  }
}


image::ColorStreamTransform buildChannelMSBMaskTransform(
    std::vector<uint32_t> msbIndices) {
  return [indices = std::move(msbIndices)](
             kernel::ProgramBuilder &P,
             kernel::StreamSet *sourceImageData) -> kernel::StreamSet * {
    kernel::StreamSet *maskStream =
        kernel::streamutils::Merge(P, sourceImageData, indices);
    kernel::StreamSet *maskedImageData = nullptr;
    image::MaskImage(P, sourceImageData, maskStream, maskedImageData);
    return maskedImageData;
  };
}

void testLoadBMPCropWithTransform(CPUDriver &driver) {
  const uint32_t width = 4;
  const uint32_t height = 4;
  auto palette = makeIdentityPalette();

  // Palette indices chosen so each channel MSB (value >= 128) is exercised:
  //   i=10  -> BGR(10, 20, 30)     none
  //   i=43  -> BGR(43, 86, 129)    red
  //   i=86  -> BGR(86, 172, 2)     green
  //   i=128 -> BGR(128, 0, 128)    blue + red
  //   i=20  -> BGR(20, 40, 60)     none
  //   i=64  -> BGR(64, 128, 192)   green + red
  //   i=192 -> BGR(192, 128, 64)   blue + green
  //   i=170 -> BGR(170, 84, 254)   blue + red
  //   i=30  -> BGR(30, 60, 90)     none
  //   i=224 -> BGR(224, 192, 160)  blue + green + red
  //   i=65  -> BGR(65, 130, 195)   green + red
  //   i=200 -> BGR(200, 144, 88)   blue + green
  //   i=5   -> BGR(5, 10, 15)      none
  //   i=160 -> BGR(160, 64, 224)   blue + red
  //   i=225 -> BGR(225, 194, 163)  blue + green + red
  //   i=180 -> BGR(180, 104, 28)   blue only
  // This guarantees some pixels are blacked out and some are kept for every
  // single-channel mask, and that the blue-only pixel distinguishes the
  // combined mask from the red/green-only masks.
  const std::vector<uint8_t> indices = {
      10u, 43u, 86u, 128u, 20u, 64u, 192u, 170u,
      30u, 224u, 65u, 200u, 5u, 160u, 225u, 180u};

  const image::BMPCrop crop{width, height, 0u, 0u};

  auto expectedColor = [&](uint32_t i, uint8_t &expB, uint8_t &expG,
                           uint8_t &expR) {
    expB = static_cast<uint8_t>(i);
    expG = static_cast<uint8_t>((i * 2u) & 0xFFu);
    expR = static_cast<uint8_t>((i * 3u) & 0xFFu);
  };

  constexpr uint32_t kBlueMSB = 7u;
  constexpr uint32_t kGreenMSB = 15u;
  constexpr uint32_t kRedMSB = 23u;

  struct MaskCase {
    const char *label;
    std::vector<uint32_t> msbIndices;
    bool selBlue;
    bool selGreen;
    bool selRed;
  };
  const MaskCase cases[] = {
      {"red", {kRedMSB}, false, false, true},
      {"green", {kGreenMSB}, false, true, false},
      {"blue", {kBlueMSB}, true, false, false},
      {"red+green+blue", {kRedMSB, kGreenMSB, kBlueMSB}, true, true, true},
  };

  for (const MaskCase &mc : cases) {
    image::ColorStreamTransform mask =
        buildChannelMSBMaskTransform(mc.msbIndices);

    for (bool bottomUp : {true, false}) {
      std::vector<uint8_t> bmp =
          buildSyntheticBMP(width, height, bottomUp, 256u, palette, indices);
      TempFile tmp = TempFile::create("loadcropxform");
      writeAllBytes(tmp.path(), bmp);

      image::BMPCropResult masked =
          image::LoadBMPCrop(driver, tmp.path(), crop, mask);
      if (g_failureCount) return;

      const std::string labelBase = std::string(mc.label) +
                                    (bottomUp ? "/bottomUp" : "/topDown");
      CHECK(masked.sourceInfo.width == width, labelBase + ": source width");
      CHECK(masked.sourceInfo.height == height, labelBase + ": source height");
      CHECK(masked.image.width == crop.width, labelBase + ": crop width");
      CHECK(masked.image.height == crop.height, labelBase + ": crop height");
      CHECK(masked.image.rgb.size() == crop.width * crop.height * 3u,
            labelBase + ": rgb size");

      // The plain overload (no transform) must reproduce the unmasked crop, so
      // the transform overload is a strict superset of the original behavior.
      image::BMPCropResult plain =
          image::LoadBMPCrop(driver, tmp.path(), crop);
      if (g_failureCount) return;

      for (uint32_t r = 0; r < height; ++r) {
        for (uint32_t c = 0; c < width; ++c) {
          const uint32_t storedRow = bottomUp ? height - 1u - r : r;
          const uint32_t idx = storedRow * width + c;
          uint8_t expB, expG, expR;
          expectedColor(indices[idx], expB, expG, expR);
          const uint32_t outIdx = (r * width + c) * 3u;

          CHECK(plain.image.rgb[outIdx + 0u] == expR,
                labelBase + ": plain R byte");
          CHECK(plain.image.rgb[outIdx + 1u] == expG,
                labelBase + ": plain G byte");
          CHECK(plain.image.rgb[outIdx + 2u] == expB,
                labelBase + ": plain B byte");

          const bool shouldMask =
              (mc.selRed && expR >= 128u) ||
              (mc.selGreen && expG >= 128u) ||
              (mc.selBlue && expB >= 128u);
          if (shouldMask) {
            if (masked.image.rgb[outIdx + 0u] != 0u ||
                masked.image.rgb[outIdx + 1u] != 0u ||
                masked.image.rgb[outIdx + 2u] != 0u) {
              llvm::errs() << "[FAIL] " << labelBase << ": pixel (" << r
                           << "," << c << ") expected black, got RGB("
                           << static_cast<unsigned>(masked.image.rgb[outIdx + 0u])
                           << ","
                           << static_cast<unsigned>(masked.image.rgb[outIdx + 1u])
                           << ","
                           << static_cast<unsigned>(masked.image.rgb[outIdx + 2u])
                           << ")\n";
              ++g_failureCount;
              return;
            }
          } else {
            CHECK(masked.image.rgb[outIdx + 0u] == expR,
                  labelBase + ": masked R byte");
            CHECK(masked.image.rgb[outIdx + 1u] == expG,
                  labelBase + ": masked G byte");
            CHECK(masked.image.rgb[outIdx + 2u] == expB,
                  labelBase + ": masked B byte");
          }
        }
      }
    }
  }

  // A transform that returns the source unchanged must equal the plain crop.
  {
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, true, 256u, palette, indices);
    TempFile tmp = TempFile::create("loadcroppassthrough");
    writeAllBytes(tmp.path(), bmp);

    image::ColorStreamTransform passthrough =
        []([[maybe_unused]] kernel::ProgramBuilder &P,
           kernel::StreamSet *sourceImageData) -> kernel::StreamSet * {
      return sourceImageData;
    };
    image::BMPCropResult passthroughResult =
        image::LoadBMPCrop(driver, tmp.path(), crop, passthrough);
    if (g_failureCount) return;
    image::BMPCropResult plain = image::LoadBMPCrop(driver, tmp.path(), crop);
    if (g_failureCount) return;
    CHECK(passthroughResult.image.rgb == plain.image.rgb,
          "passthrough transform equals plain crop");
  }

  // A transform that returns null must throw.
  {
    std::vector<uint8_t> bmp =
        buildSyntheticBMP(width, height, true, 256u, palette, indices);
    TempFile tmp = TempFile::create("loadcropnull");
    writeAllBytes(tmp.path(), bmp);
    image::ColorStreamTransform nullTransform =
        []([[maybe_unused]] kernel::ProgramBuilder &P,
           [[maybe_unused]] kernel::StreamSet *sourceImageData)
        -> kernel::StreamSet * { return nullptr; };
    expect_throw("LoadBMPCrop(null transform)", [&]() {
      image::LoadBMPCrop(driver, tmp.path(), crop, nullTransform);
    });
  }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  codegen::ParseCommandLineOptions(argc, argv,
                                   {&codegen::JIT_InfoOptions,
                                    &codegen::InstrumentationOptions});

  CPUDriver driver("bmp_loader_test");

  run_test("testGetBMP24RowStride", testGetBMP24RowStride);
  run_test("testReadBMPHeaderValid", testReadBMPHeaderValid);
  run_test("testReadBMPHeaderLena", testReadBMPHeaderLena);
  run_test("testReadBMPHeaderErrors", testReadBMPHeaderErrors);
  run_test("testReadBMPPaletteSemantics", testReadBMPPaletteSemantics);
  run_test("testParseBMPColorStreamsAndByteStreams",
           testParseBMPColorStreamsAndByteStreams, driver);
  run_test("testParseBMPBufferPaddingRemoval",
           testParseBMPBufferPaddingRemoval, driver);
  run_test("testCreateBMPColorByteStreamsValidation",
           testCreateBMPColorByteStreamsValidation, driver);
  run_test("testCropImage", testCropImage, driver);
  run_test("testCreateBMP24PixelData", testCreateBMP24PixelData);
  run_test("testWriteBMP24RoundTrip", testWriteBMP24RoundTrip);
  run_test("testCreateBMP24Image", testCreateBMP24Image);
  run_test("testWriteBMP24ImageRoundTrip", testWriteBMP24ImageRoundTrip);
  run_test("testLoadBMPCrop", testLoadBMPCrop, driver);
  run_test("testMaskImage", testMaskImage, driver);
  run_test("testMaskImageValidation", testMaskImageValidation, driver);
  run_test("testLoadBMPCropWithTransform", testLoadBMPCropWithTransform, driver);

  if (g_failureCount != 0) {
    llvm::errs() << "bmp_loader_tests: " << g_failureCount
                 << " failure(s)\n";
    return 1;
  }
  llvm::errs() << "bmp_loader_tests: all tests passed\n";
  return 0;
}
