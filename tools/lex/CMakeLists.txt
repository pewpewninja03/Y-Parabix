#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: lex

parabix_add_executable(
NAME
    tokenizer
SRC
    tokenizer.cpp
DEPS
    pablo
    kernel.basis
    kernel.io
    kernel.unicode
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
    grep
    re.cc
    re.parse
)
