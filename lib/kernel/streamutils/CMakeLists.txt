#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.streamutils

parabix_add_module(
NAME
    kernel.streamutils
SRC
    collapse.cpp
    deletion.cpp
    pdep_kernel.cpp
    run_index.cpp
    sentinel.cpp
    sorting.cpp
    stream_select.cpp
    stream_shift.cpp
    streams_merge.cpp
    string_insert.cpp
    swizzle.cpp
    swizzled_multiple_pdep_kernel.cpp
    until_n.cpp
    zeroextend.cpp
DEPS
    pablo.bixnum
    kernel.pipeline
    kernel.basis
)
