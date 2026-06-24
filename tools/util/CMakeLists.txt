#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: util

parabix_add_executable(
NAME
    hexlines
SRC
    hexlines.cpp
DEPS
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)


parabix_add_executable(
NAME
    hashdemo
SRC
    hashdemo.cpp
DEPS
    grep
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)


parabix_add_executable(
NAME
    packdemo
SRC
    packdemo.cpp
DEPS
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)

parabix_add_executable(
NAME
    ufilter
SRC
    ufilter.cpp
DEPS
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.unicode
    kernel.util
    toolchain
    re.cc
    re.parse
)

