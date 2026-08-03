#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

parabix_add_executable(
NAME
    demo
SRC
    demo.cpp
DEPS
    image
    kernel.basis
    kernel.io
    kernel.streamutils
)

parabix_add_executable(
NAME
    bench
SRC
    bench/bench.cpp
DEPS
    image
    kernel.basis
    kernel.io
    kernel.streamutils
)
