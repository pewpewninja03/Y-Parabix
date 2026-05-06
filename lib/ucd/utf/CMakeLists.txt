#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: unicode.utf

parabix_add_module(
NAME
    unicode.utf
SRC
    utf_encoder.cpp
    utf_compiler.cpp
    transchar.cpp
DEPS
    re.cc
    unicode.core
    pablo.bixnum
)
