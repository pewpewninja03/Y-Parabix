/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/to_utf8.h>
#include <re/transforms/re_transformer.h>
#include <re/adt/adt.h>
#include <ucd/core/unicode_set.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace re {

RE * UTF8_Transformer::rangeCodeUnits(codepoint_t lo, codepoint_t hi, unsigned index, const unsigned lgth){
    const codepoint_t hunit = mEncoder.nthCodeUnit(hi, index);
    const codepoint_t lunit = mEncoder.nthCodeUnit(lo, index);
    if (index == lgth) {
        return makeCC(lunit, hunit, &cc::UTF8);
    }
    else if (hunit == lunit) {
        return makeSeq({makeCC(hunit, &cc::UTF8), rangeCodeUnits(lo, hi, index + 1, lgth)});
    }
    else {
        const unsigned suffix_mask = (static_cast<unsigned>(1) << ((lgth - index) * 6)) - 1;
        if ((hi & suffix_mask) != suffix_mask) {
            const unsigned hi_floor = (~suffix_mask) & hi;
            return makeAlt({rangeCodeUnits(hi_floor, hi, index, lgth), rangeCodeUnits(lo, hi_floor - 1, index, lgth)});
        }
        else if ((lo & suffix_mask) != 0) {
            const unsigned low_ceil = lo | suffix_mask;
            return makeAlt({rangeCodeUnits(low_ceil + 1, hi, index, lgth), rangeCodeUnits(lo, low_ceil, index, lgth)});
        }
        else {
            return makeSeq({makeCC(lunit, hunit, &cc::UTF8), rangeCodeUnits(lo, hi, index + 1, lgth)});
        }
    }
}

RE * UTF8_Transformer::rangeToUTF8(codepoint_t lo, codepoint_t hi) {
    const auto min_lgth = mEncoder.encoded_length(lo);
    const auto max_lgth = mEncoder.encoded_length(hi);
    if (min_lgth < max_lgth) {
        const auto m = mEncoder.max_codepoint_of_length(min_lgth);
        return makeAlt({rangeToUTF8(lo, m), rangeToUTF8(m + 1, hi)});
    }
    else {
        return rangeCodeUnits(lo, hi, 1, max_lgth);
    }
}

RE * UTF8_Transformer::transformCC(CC * cc) {
    if (cc->getAlphabet() != &cc::Unicode) return cc;
    std::vector<RE *> alt;
    for (const interval_t i : *cc) {
        alt.push_back(rangeToUTF8(lo_codepoint(i), hi_codepoint(i)));
    }
    RE * xfrmd = makeAlt(alt.begin(), alt.end());
    if (mInternalNaming) {
        std::string ccName = cc->canonicalName();
        return makeName(ccName, xfrmd);
    }
    return xfrmd;
}

UTF8_Transformer::UTF8_Transformer(bool useInternalNaming) :
EncodingTransformer("ToUTF8", &cc::Unicode, &cc::UTF8),
    mInternalNaming(useInternalNaming) {
    mEncoder.setCodeUnitBits(8);
}

RE * toUTF8(RE * r, bool useInternalNaming) {
    return UTF8_Transformer(useInternalNaming).transformRE(r);
}

}
