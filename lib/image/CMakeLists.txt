#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: image

parabix_add_module(
NAME
    image
SRC
    bmp_io.cpp
    bmp_pipeline.cpp
    conv_filter.cpp
    conv_filter_default.cpp
    conv_filter_uniform.cpp
    conv_filter_low_rank.cpp
    conv_filter_frequency.cpp
DEPS
    pablo
    pablo.bixnum
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.basis
    kernel.util
    toolchain
)
