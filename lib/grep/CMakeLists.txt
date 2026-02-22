#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: grep

parabix_add_module(
NAME
    grep
SRC
    grep_engine.cpp
    grep_kernel.cpp
    grep_toolchain.cpp
    nested_grep_engine.cpp
DEPS
    kernel.pipeline
    kernel.basis
    kernel.io
    kernel.re
    kernel.scan
    kernel.streamutils
    kernel.util
    kernel.unicode
    re.parse
    re.compile
    re.unicode
    re.transforms
    re.printer
    unicode.data
)
