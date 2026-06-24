/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <re/transforms/re_transformer.h>

namespace re {

class RE;

RE * removeNullablePrefix(RE * re);

RE * removeNullableSuffix(RE * re);

RE * zeroBoundElimination(RE * re,
                          NameTransformationMode m = NameTransformationMode::None);

RE * emptyMatchElimination(RE * r);
}
