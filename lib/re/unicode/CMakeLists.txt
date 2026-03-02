#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: re.unicode

parabix_add_module(
NAME
    re.unicode
SRC
    boundaries.cpp
    casing.cpp
    decomposition.cpp
    equivalence.cpp
    re_name_resolve.cpp
    resolve_properties.cpp
    regex_passes.cpp
DEPS
    re.adt
    re.analysis
    re.transforms
    re.parse
    re.printer
    unicode.data
    unicode.algo
)
