/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cstdint>
#include <kernel/core/relationship.h>
#include <kernel/pipeline/program_builder.h>
#include <vector>

namespace image {

struct BMPInfo {
  uint32_t width;
  uint32_t height;
  uint32_t rowStride;     // padded row size in bytes: ((width + 3) / 4) * 4
  uint32_t pixelOffset;   // byte offset from file start to first pixel byte
  uint32_t paletteOffset; // byte offset from file start to first palette entry
  uint32_t numColors;     // number of palette entries (1..256)
  bool rowsBottomUp; // true when the original BMP height field was positive

  // Each pixel contains BGR(reserves) values.
  std::vector<unsigned> bTable; // Blue  channel (palette[i*4+0])
  std::vector<unsigned> gTable; // Green channel (palette[i*4+1])
  std::vector<unsigned> rTable; // Red   channel (palette[i*4+2])
};

/*
 * Reads BMPFileHeader (14 bytes) and BMPInfoHeader (40 bytes) from
 * an open file descriptor.  On success the fd is seeked to pixelOffset so
 * that the caller can hand it directly to ParseBMPBuffer().
 *
 * For 8-bit BMPs the color table is also read into info.bTable/gTable/rTable.
 */
void readBMPHeader(int fd, BMPInfo &info);

/*
 * Create a streamset of pixels with padding removed
 */
void ParseBMPBuffer(kernel::ProgramBuilder &P, kernel::Scalar *fileDescriptor,
                    const BMPInfo &info, kernel::StreamSet *&pixelStream,
                    kernel::StreamSet *&basisBits);

/*
 * Maps each pixel index to its B, G, R color values using the tables in
 * BMPInfo.
 *
 * Output:
 *   colorStream - 24x1 BixNum StreamSet:
 *                 streams  0..7  = Blue
 *                 streams  8..15 = Green
 *                 streams 16..23 = Red
 */
void ParseBMPColorStreams(kernel::ProgramBuilder &P,
                          kernel::Scalar *fileDescriptor, const BMPInfo &info,
                          kernel::StreamSet *&colorStream);

/*
 * Crop a 24x1 B/G/R color stream using a one-bit crop mask.
 *
 * Arguments:
 *   sourceImageData - 24x1 color stream from ParseBMPColorStreams().
 *   sourceInfo      - dimensions and row order for sourceImageData.
 *   cropWidth       - output width in pixels.
 *   cropHeight      - output height in pixels.
 *   cropX           - crop rectangle left edge, in top-left image coordinates.
 *   cropY           - crop rectangle top edge, in top-left image coordinates.
 *
 * Output:
 *   croppedImageData - 24x1 color stream containing cropWidth*cropHeight
 *                      pixels in the source stream's native row order.
 */
void CropImage(kernel::ProgramBuilder &P, kernel::StreamSet *sourceImageData,
               const BMPInfo &sourceInfo, uint32_t cropWidth,
               uint32_t cropHeight, uint32_t cropX, uint32_t cropY,
               kernel::StreamSet *&croppedImageData);

} // namespace image
