#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.scan

parabix_add_module(
NAME
    kernel.scan
SRC
    base.cpp
    index_generator.cpp
    line_number_generator.cpp
    line_span_generator.cpp
    reader.cpp
    scanmatchgen.cpp
DEPS
    kernel.pipeline
)
