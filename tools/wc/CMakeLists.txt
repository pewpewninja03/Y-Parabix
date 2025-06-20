#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: wc

parabix_add_executable(
NAME
    wc
SRC
    wc.cpp
DEPS
    fileselect
    kernel.basis
    kernel.io
    re.cc
)

parabix_add_executable(
NAME
    ucount
SRC
    ucount.cpp
DEPS
    fileselect
    kernel.basis
    kernel.io
    re.cc
    re.parse
)

parabix_add_executable(
NAME
    gcount
SRC
    gcount.cpp
DEPS
    fileselect
    kernel.basis
    kernel.io
    kernel.unicode
    re.cc
    re.parse
)
