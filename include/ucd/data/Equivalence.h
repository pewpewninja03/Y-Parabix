#pragma once
/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 */
#include <ucd/core/unicode_set.h>

namespace UCD {
    
enum EquivalenceOptions {Canonical = 0, Caseless = 1, Compatible = 2};
    
bool hasOption(enum EquivalenceOptions optionSet, enum EquivalenceOptions testOption);

UnicodeSet equivalentCodepoints(codepoint_t, EquivalenceOptions options = Canonical);
    
}

