/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <vector>
#include <cstdint>

namespace re {

class RE;

typedef std::vector<uint64_t> (*GrepLinesFunctionType)(re::RE *, const char * buf, size_t bufSize);

RE * resolveModesAndExternalSymbols(RE * r, bool globallyCaseInsensitive = false, GrepLinesFunctionType grep = nullptr);

RE * remove_nullable_ends(RE * r);

RE * regular_expression_passes(RE * r);

}
