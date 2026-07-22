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

// Try a list of candidate paths and return the first that opens, or -1.
// This makes the lena smoke test robust to whichever directory the test is
// launched from (build/tests/bin, build/, tests/, repo root, ...).
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

  if (g_failureCount != 0) {
    llvm::errs() << "bmp_loader_tests: " << g_failureCount
                 << " failure(s)\n";
    return 1;
  }
  llvm::errs() << "bmp_loader_tests: all tests passed\n";
  return 0;
}
