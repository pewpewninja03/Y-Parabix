/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cstdint>
#include <kernel/core/relationship.h>
#include <kernel/pipeline/program_builder.h>

namespace image {

/*
 * Relevant data extracted from a BMP file header.
 *
 * After readBMPHeader() returns, the file descriptor is positioned at
 * the first byte of pixel data (pixelOffset bytes from the start of the
 * file).  ParseBMPBuffer() can be called immediately afterward.
 *
 * Row order: standard BMP files store rows bottom-to-top (rowsBottomUp ==
 * true).
 */
struct BMPInfo {
  uint32_t width;
  uint32_t height;
  uint32_t rowStride;   // padded row size in bytes: ((width + 3) / 4) * 4
  uint32_t pixelOffset; // byte offset from file start to first pixel byte
  bool rowsBottomUp;    // true when the original BMP height field was positive
};

/*
 *
 * Reads BMPFileHeader (14 bytes) and BMPInfoHeader (40 bytes) from
 * an open file descriptor.  On success the fd is seeked to pixelOffset so
 * that the caller can hand it directly to ParseBMPBuffer().
 */
void readBMPHeader(int fd, BMPInfo &info);

/*
 * Wiring helper for a Parabix pipeline that reads a BMP file.
 *
 * Wires the following kernel sequence into ProgramBuilder P:
 *
 *   ReadSourceKernel(fd)          → rawStream  (StreamSet 1×8, bytes with row
 * padding) [FilterByMask with row mask]  → pixelStream (StreamSet 1×8, bytes
 * without row padding) S2PKernel                     → basisBits   (StreamSet
 * 8×1, bits)
 *
 * Output parameters:
 *   pixelStream — compact pixel bytes in native BMP row order (width × height
 * bytes) basisBits   — 8-stream bits representation of pixelStream
 *
 */
void ParseBMPBuffer(kernel::ProgramBuilder &P, kernel::Scalar *fileDescriptor,
                    const BMPInfo &info, kernel::StreamSet *&pixelStream,
                    kernel::StreamSet *&basisBits);

} // namespace image
