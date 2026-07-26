/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cstdint>
#include <functional>
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

using ColorStreamTransform =
    std::function<kernel::StreamSet *(kernel::ProgramBuilder &P,
                                      kernel::StreamSet *sourceImageData)>;

BMPCropResult LoadBMPCrop(CPUDriver &driver, const std::string &inputPath,
                          const BMPCrop &crop,
                          const ColorStreamTransform &transform);

} // namespace image
