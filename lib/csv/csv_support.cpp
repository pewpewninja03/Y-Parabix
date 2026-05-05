/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <csv/csv_support.h>
#include <re/adt/adt.h>
#include <re/adt/re_re.h>
#include <re/transforms/re_transformer.h>

namespace csv {

DoubleQuoteEscape::DoubleQuoteEscape(UCD::codepoint_t dqChar) : re::RE_Transformer("DoubleQuoteEscape"),
       mDQ(dqChar), mDQ_CC(nullptr), mDoubleEscape(nullptr) {}

re::CC * DoubleQuoteEscape::getDQ_CC() {
    if (mDQ_CC == nullptr) {
        mDQ_CC = re::makeCC(mDQ, &cc::Unicode);
    }
    return mDQ_CC;
}

re::RE * DoubleQuoteEscape::getDoubleEscape() {
    if (mDoubleEscape == nullptr) {
        re::CC * DQ_CC = getDQ_CC();
        mDoubleEscape = re::makeSeq({DQ_CC, DQ_CC});
    }
    return mDoubleEscape;
}

re::RE * DoubleQuoteEscape::transformCC (re::CC * cc) {
    if (cc->contains(mDQ)) {
        auto dblEsc = getDoubleEscape();
        if (cc->count() == 1) {
            return dblEsc;
        }
        return re::makeAlt({dblEsc, subtractCC(cc, mDQ_CC)});
    }
    return cc;
}

re::RE * DoubleQuoteEscape::transformAny (re::Any * a) {
    auto dblEsc = getDoubleEscape();
    return re::makeAlt({dblEsc, re::makeDiff(a, getDQ_CC())});
}

re::RE * DoubleQuoteEscape::transformName (re::Name * name) {
    re::RE * defn = name->getDefinition();
    if (!defn) return makeDiff(name, getDQ_CC());
    re::RE * d = transform(defn);
    if (d == defn) return name;
    return d;
}

re::RE * DoubleQuoteEscape::transformPropertyExpression (re::PropertyExpression * pe) {
    re::RE * defn = pe->getResolvedRE();
    if (!defn) return makeDiff(pe, getDQ_CC());
    re::RE * d = transform(defn);
    if (d == defn) return pe;
    return d;
}

}
