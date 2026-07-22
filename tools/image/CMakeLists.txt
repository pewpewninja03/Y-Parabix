#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

parabix_add_executable(
NAME
    bmp_header_test
SRC
    bmp_header_test.cpp
DEPS
    image
    kernel.basis
    kernel.io
)
