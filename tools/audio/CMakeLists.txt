#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: audio

parabix_add_executable(
NAME
    audioamplifier
SRC
    audioamplifier.cpp
DEPS
    kernel.io
    kernel.pipeline
    kernel.util
    toolchain
    audio
)

parabix_add_executable(
NAME
    stereo2mono
SRC
    stereo2mono.cpp
DEPS
    kernel.io
    kernel.pipeline
    kernel.util
    toolchain
    audio
)

parabix_add_executable(
NAME
    discontinuity
SRC
    discontinuity.cpp
DEPS
    kernel.io
    kernel.pipeline
    kernel.util
    toolchain
    audio
)