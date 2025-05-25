#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.unicode

parabix_add_module(
NAME
    kernel.unicode
SRC
    charclasses.cpp
    boundary_kernels.cpp
    normalization.cpp
    UCD_property_kernel.cpp
    utf8gen.cpp
    utf8_decoder.cpp
    utf16gen.cpp
DEPS
    kernel.pipeline
    kernel.streamutils
    re.unicode
    pablo
    pablo.bixnum
)
