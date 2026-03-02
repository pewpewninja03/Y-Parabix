#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: re.compile

parabix_add_module(
NAME
    re.transforms
SRC
    assertion_transformations.cpp
    exclude_CC.cpp
    name_intro.cpp
    name_lookaheads.cpp
    re_contextual_simplification.cpp
    re_minimizer.cpp
    re_multiplex.cpp
    re_reverse.cpp
    re_simplifier.cpp
    re_star_normal.cpp
    re_memoizing_transformer.cpp
    re_transformer.cpp
    reference_transform.cpp
    remove_nullable.cpp
    replaceCC.cpp
    resolve_diffs.cpp
    to_utf8.cpp
    expand_permutes.cpp
    variable_alt_promotion.cpp
DEPS
    re.analysis
    re.printer
    unicode.utf
    unicode.data
)
