/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/adt/re_utility.h>
#include <re/adt/adt.h>

namespace re {
    
RE * makeComplement(RE * s) {
  return makeDiff(makeAny(), s);
}

RE * makeZerowidthComplement(RE * s) {
    return makeDiff(makeSeq({}), s);
}

RE * makeWordBoundary() {
    return makePropertyExpression(PropertyExpression::Kind::Boundary, "word");
}

RE * makeWordNonBoundary() {
    return makeZerowidthComplement(makeWordBoundary());
}

RE * makeWordBegin() {
    auto wordC = makePropertyExpression("word");
    return makeNegativeLookBehindAssertion(wordC);
}

RE * makeWordEnd() {
    auto wordC = makePropertyExpression("word");
    return makeNegativeLookAheadAssertion(wordC);
}

RE * makeUnicodeBreak() {
    return makeAlt({makeCC(0x0A, 0x0C), makeCC(0x85), makeCC(0x2028,0x2029), makeSeq({makeCC(0x0D), makeNegativeLookAheadAssertion(makeCC(0x0A))})});
}
    
}
