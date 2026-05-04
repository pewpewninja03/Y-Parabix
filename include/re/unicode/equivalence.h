/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <ucd/core/unicode_set.h>
#include <ucd/data/Equivalence.h>

namespace re { class RE; class CC; class Seq; class Group;}

namespace UCD {

re::RE * addClusterMatches(re::RE * r, UCD::EquivalenceOptions options = UCD::Canonical);

re::RE * addEquivalentCodepoints(re::RE * r, UCD::EquivalenceOptions options = UCD::Canonical);

}
