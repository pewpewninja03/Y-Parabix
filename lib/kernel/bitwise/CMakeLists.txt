#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.bitwise

parabix_add_module(
NAME
    kernel.bitwise
SRC
    bixnum_kernel.cpp
    bixlogic.cpp
DEPS
    pablo
    pablo.bixnum
    kernel.pipeline
)
