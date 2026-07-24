/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cstdint>
#include <image/bmp_io.h>
#include <kernel/core/relationship.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>

namespace image {

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

void CropImage(kernel::ProgramBuilder &P, kernel::StreamSet *sourceImageData,
               const BMPInfo &sourceInfo, uint32_t cropWidth,
               uint32_t cropHeight, uint32_t cropX, uint32_t cropY,
               kernel::StreamSet *&croppedImageData);

/*
 * Black out selected pixels of a 24x1 B/G/R color stream using a 1x1 mask.
 *
 * Inputs:
 *   sourceImageData - 24x1 B/G/R color stream (same layout produced by
 *                     ParseBMPColorStreams): streams 0..7 Blue, 8..15 Green,
 *                     16..23 Red.
 *   maskImageData    - 1x1 stream with one item per pixel position, aligned
 *                     1:1 with sourceImageData. A mask bit of 1 marks the
 *                     pixel to be blacked out; a mask bit of 0 leaves the
 *                     source pixel unchanged.
 *
 * Output:
 *   maskedImageData - 24x1 B/G/R color stream with the same item count and
 *                     ordering as sourceImageData. Masked positions are
 *                     all-zero across every channel (black).
 *
 * The source and mask streams must describe images of the same shape; because
 * stream-stage arguments do not carry width/height metadata, matching
 * dimensions are the caller's responsibility. Only the structural stream
 * shapes are validated here.
 */
void MaskImage(kernel::ProgramBuilder &P, kernel::StreamSet *sourceImageData,
               kernel::StreamSet *maskImageData,
               kernel::StreamSet *&maskedImageData);


void CreateBMPColorByteStreams(kernel::ProgramBuilder &P,
                               kernel::StreamSet *sourceImageData,
                               kernel::StreamSet *redBytes,
                               kernel::StreamSet *greenBytes,
                               kernel::StreamSet *blueBytes);

struct BMPCrop {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t x = 0;
  uint32_t y = 0;
};

struct BMPCropResult {
  BMPInfo sourceInfo;
  BMP24Image image;
};

BMPCropResult LoadBMPCrop(CPUDriver &driver, const std::string &inputPath,
                          const BMPCrop &crop);

} // namespace image
