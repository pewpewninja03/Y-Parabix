/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

namespace re {

class RE;

RE * resolveModesAndExternalSymbols(RE * r, bool globallyCaseInsensitive = false);

RE * excludeUnicodeLineBreak(RE * r);

RE * remove_nullable_ends(RE * r);

RE * regular_expression_passes(RE * r);

}
