#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.re

parabix_add_module(
NAME
    kernel.re
SRC
    regexp_kernel.cpp
DEPS
    kernel.pipeline
    kernel.basis
    kernel.bitwise
    kernel.streamutils
    kernel.unicode
    re.compile
    re.unicode
    re.transforms
    re.printer
)
