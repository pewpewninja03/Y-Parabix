#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: json

parabix_add_module(
NAME
    jsonlib
SRC
    json_support.cpp
DEPS
    re.adt
    pablo
    grep
    kernel.bitwise
    kernel.io
    kernel.pipeline
    kernel.re
    kernel.unicode
)
