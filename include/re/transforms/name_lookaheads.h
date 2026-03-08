/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once
#include <re/adt/adt.h>
#include <re/alphabet/alphabet.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/name_intro.h>

namespace re {

/* Transform a regular expression r so that all lookahead
   assertions of length more than maxLookahead are transformed into
   names to be externally defined. */
class LookAheadNamer final : public NameIntroduction {
public:
    LookAheadNamer(const cc::Alphabet & alpha = cc::Unicode, unsigned maxLookahead = 1) : 
         NameIntroduction("LookAheadNamer"), mAlphabet(alpha), mMaxLookahead(maxLookahead) {}
    RE * transformAssertion (Assertion * a) override;
private:
   const cc::Alphabet & mAlphabet;
   const unsigned mMaxLookahead;
};

/*  Given any Name, retrieves the lookahead amount associated
    with that name as created by the LookAneadHamer, or returns
    0 if the Name is not defined as a Lookahead assertion. */
unsigned NamedLookAheadAmount(const Name * n, const cc::Alphabet & alpha = cc::Unicode);
}

