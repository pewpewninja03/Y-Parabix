/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/name_intro.h>

namespace re {

/* Transform a regular expression r so that all names are
   created for all lookahead assertions. */
class LookAheadNamer final : public NameIntroduction {
public:
    LookAheadNamer() : NameIntroduction("LookAheadNamer") {}
    RE * transformAssertion (Assertion * a) override;
};
}

