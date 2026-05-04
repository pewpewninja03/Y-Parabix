#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0
parabix_add_module(
NAME
    unicode.algo
SRC
    normalization.cpp
DEPS
    unicode.core
    unicode.data
)
