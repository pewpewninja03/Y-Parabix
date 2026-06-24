/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/remove_nullable.h>
#include <re/adt/adt.h>
#include <re/analysis/nullable.h>
#include <re/analysis/re_analysis.h>
#include <re/transforms/re_transformer.h>
#include <llvm/ADT/SmallVector.h>

/*

 A regular expression is nullable if it (a) matches the empty
 string, and (b) applies everywhere.  Note that Start (^) and
 End ($) match the empty string, but not everywhere).

*/

using namespace llvm;

namespace re {

class NullablePrefixRemover final : public RE_Transformer {
protected:
    RE * transformSeq(Seq * seq) override;
    RE * transformRep(Rep * rep) override;
    RE * transformAssertion(Assertion * a) override;
public:
    NullablePrefixRemover() : RE_Transformer("NullablePrefixRemoval") {}
private:
    SmallVector<RE *, 16> mList;
};

RE * NullablePrefixRemover::transformSeq(Seq * seq) {
    // if the sequence is empty, return it unmodified.
    if (isNullable(seq)) return seq;
    // Process the first element.
    auto i = seq->begin();
    auto e = transform(*i);
    while (isNullable(e)) {
        // Skip empty elements.
        i++;
        e = transform(*i);
    }
    // Special case: nothing skipped and first element unchanged.
    if ((i == seq->begin()) && (e == *i)) return seq;
    mList.assign(1, e);
    mList.append(++i, seq->end());
    return makeSeq(mList.begin(), mList.end());
}

RE * NullablePrefixRemover::transformRep(Rep * rep) {
    auto lb = rep->getLB();
    auto r = rep->getRE();
    if ((lb == 0) || isNullable(r)) {
        return makeSeq();
    }
    auto s = transform(r);
    if ((s == r) && (lb == rep->getUB())) return rep; // special case.  No transformation required.
    if (lb == 1) return s;
    if (lb == 2) return makeSeq({s, r});
    return makeSeq({s, makeRep(r, lb - 1, lb - 1)});
}

RE * NullablePrefixRemover::transformAssertion(Assertion * a) {
    return a;
}

RE * removeNullablePrefix(RE * r) {
    return NullablePrefixRemover().transformRE(r);
}

class NullableSuffixRemover final : public RE_Transformer {
protected:
    RE * transformSeq(Seq * seq) override;
    RE * transformRep(Rep * rep) override;
    RE * transformAssertion(Assertion * a) override;
public:
    NullableSuffixRemover() : RE_Transformer("NullableSuffixRemoval") {}
private:
    SmallVector<RE *, 16> mList;
};

RE * NullableSuffixRemover::transformSeq(Seq * seq) {
    // if the sequence is empty, return it unmodified.
    if (isNullable(seq)) return seq;
    // Process the last element.
    auto ri = seq->rbegin();
    auto r = transform(*ri);
    while (isNullable(r)) {
        // Skip empty elements.
        ri++;
        r = transform(*ri);
    }
    // Special case: nothing skipped and first element unchanged.
    if ((ri == seq->rbegin()) && (r == *ri)) return seq;
    mList.clear();
    mList.append(seq->begin(), (ri + 1).base());
    mList.push_back(r);
    return makeSeq(mList.begin(), mList.end());
}

RE * NullableSuffixRemover::transformRep(Rep * rep) {
    auto lb = rep->getLB();
    auto r = rep->getRE();
    if ((lb == 0) || isNullable(r)) {
        return makeSeq();
    }
    auto s = transform(r);
    if ((s == r) && (lb == rep->getUB())) return rep; // special case.  No transformation required.
    if (lb == 1) return s;
    if (lb == 2) return makeSeq({r, s});
    return makeSeq({makeRep(r, lb - 1, lb - 1), s});
}

RE * NullableSuffixRemover::transformAssertion(Assertion * a) {
    return a;
}
RE * removeNullableSuffix(RE * r) {
    return NullableSuffixRemover().transformRE(r);
}

class ZeroBoundElimination final : public RE_Transformer {
public:
    ZeroBoundElimination() : RE_Transformer("ZeroBoundElimination") {}
protected:
    RE * transformRep(Rep * r) override {
        if (r->getLB() > 0) return r;
        return makeAlt({makeSeq(), makeRep(r->getRE(), 1, r->getUB())});
    }
    RE * transformAssertion(Assertion * a) override {
        return a;
    }
};

RE * zeroBoundElimination(RE * re,
                          NameTransformationMode m) {
    return ZeroBoundElimination().transformRE(re, m);
}

class RemoveEmptyTransformer final : public RE_Transformer {
protected:
    RE * transformName(Name * n) override;
    RE * transformAlt(Alt * a) override;
    RE * transformSeq(Seq * seq) override;
    RE * transformRep(Rep * rep) override;
    RE * transformAssertion(Assertion * a) override;
    RE * transformStart(Start * s) override;
    RE * transformEnd(End * e) override;
public:
    RemoveEmptyTransformer() : RE_Transformer("RemoveEmptyTransformer") {}
};

RE * RemoveEmptyTransformer::transformName(Name * n) {
    auto def = n->getDefinition();
    auto e = transform(def);
    if (e == def) return n;
    return e;
}

RE * RemoveEmptyTransformer::transformRep(Rep * rep) {
    auto lb = rep->getLB();
    auto r = rep->getRE();
    auto s = transform(r);
    if ((s == r) && (lb >= 1)) return rep; // special case.  No transformation required.
    auto ub = rep->getUB();
    if (ub == 1) return s;
    auto new_lb = lb == 0 ? 1 : lb;
    return makeRep(s, new_lb, ub);
}

RE * RemoveEmptyTransformer::transformAlt(Alt * alt) {
    SmallVector<RE *, 16> elems;
    elems.reserve(alt->size());
    bool any_changed = false;
    for (RE * e : *alt) {
        RE * e1 = transform(e);
        if (e1 != e) any_changed = true;
        if (!isEmptySet(e1)) {
            elems.push_back(e1);
        }
    }
    if (!any_changed) return alt;
    if (elems.size() == 1) return elems[0];
    return makeAlt(elems.begin(), elems.end());
}

RE * RemoveEmptyTransformer::transformSeq(Seq * seq) {
    auto rg = getLengthRange(seq, &cc::Unicode);
    if (rg.first > 0) return seq;
    if (rg.second == 0) return makeAlt();
    //
    // After these two tests, we know that all elements
    // are potentially empty, but at least one has a
    // nonempty case.   Create a list of alternatives
    // each of which is a Seq based on the original
    // except that one element is modified to the nonempty
    // case.
    SmallVector<RE *, 16> alts;
    //
    // All elements are potentially empty, but
    // at least one has a nonempty case.
    //
    for (auto i = seq->begin(); i != seq->end(); ++i) {
        RE * e = *i;
        RE * e1 = transform(e);
        if (!isEmptySet(e1)) {
            SmallVector<RE *, 16> elems;
            elems.reserve(seq->size());
            for (auto j = seq->begin(); j != i; ++j) {
                elems.push_back(*j);
            }
            elems.push_back(e1);
            for (auto j = i+1; j != seq->end(); ++j) {
                elems.push_back(*j);
            }
            alts.push_back(makeSeq(elems.begin(), elems.end()));
        }
    }
    if (alts.size() == 1) return alts[0];
    return makeAlt(alts.begin(), alts.end());
}

RE * RemoveEmptyTransformer::transformAssertion(Assertion * a) {
    return makeAlt();
}

RE * RemoveEmptyTransformer::transformStart(Start * s) {
    return makeAlt();
}

RE * RemoveEmptyTransformer::transformEnd(End * e) {
    return makeAlt();
}

RE * emptyMatchElimination(RE * r) {
    return RemoveEmptyTransformer().transformRE(r);
}

}
