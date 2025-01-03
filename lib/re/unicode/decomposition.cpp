/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/unicode/decomposition.h>

#include <string>
#include <vector>
#include <llvm/Support/Casting.h>
#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/data/PropertyValueAliases.h>


using namespace llvm;
using namespace re;

namespace UCD {
    
static inline std::u32string getStringPiece(Seq * s, unsigned position) {
    unsigned pos = position;
    unsigned size = s->size();
    std::u32string rslt;
    while ((pos < size) && isa<CC>((*s)[pos])) {
        CC * cc = cast<CC>((*s)[pos]);
        if (cc->empty()) return rslt;
        if (cc->getAlphabet() != &cc::Unicode) return rslt;
        codepoint_t lo = lo_codepoint(cc->front());
        codepoint_t hi = hi_codepoint(cc->back());
        if (lo != hi) // not a singleton CC; end of the string piece.
            return rslt;
        rslt.push_back(lo);
        pos++;
    }
    return rslt;
}

class NFD_Transformer final : public re::RE_Transformer {
public:
    /* Transforme an RE so that all string pieces and character classes
     are converted to NFD form (or NFKD form if the Compatible option
     is used.  The options may also including case folding.  Example:
     NFD_Transformer(CaseFold | NFKD).transformRE(r);
    */
    NFD_Transformer(DecompositionOptions opt, NFD_Engine & engine);
protected:
    re::RE * transformCC(re::CC * cc) override;
    re::RE * transformSeq(re::Seq * seq) override;
    re::RE * transformGroup(re::Group * g) override;
private:
    NFD_Engine & mEngine;
    DecompositionOptions mOptions;
};
    
NFD_Transformer::NFD_Transformer(DecompositionOptions opt, NFD_Engine & engine) :
RE_Transformer("toNFD"),
mEngine(engine),
mOptions(opt) {
}

RE * NFD_Transformer::transformGroup(Group * g) {
    re::Group::Mode mode = g->getMode();
    re::Group::Sense sense = g->getSense();
    auto r = g->getRE();
    DecompositionOptions opts;
    if (mode == re::Group::Mode::CaseInsensitiveMode) {
        if (sense == re::Group::Sense::On) {
            opts = static_cast<DecompositionOptions>(mOptions | CaseFold);
        } else {
            opts = static_cast<DecompositionOptions>(mOptions & ~CaseFold);
        }
    } else if (mode == re::Group::Mode::CompatibilityMode) {
        if (sense == re::Group::Sense::On) {
            opts = static_cast<DecompositionOptions>(mOptions | NFKD);
        } else {
            opts = static_cast<DecompositionOptions>(mOptions & ~NFKD);
        }
    } else {
        opts = mOptions;
    }
    RE * t;
    if (opts == mOptions) {
        t = transform(r);
    } else {
        t = toNFD(r, opts);
    }
    if (t == r) return g;
    return makeGroup(mode, t, sense);
}

RE * NFD_Transformer::transformCC(CC * cc) {
    if (cc->getAlphabet() != &cc::Unicode) return cc;
    std::vector<RE *> alts;
    CC * finalCC = cc;
    for (const interval_t i : *cc) {
        for (codepoint_t cp = lo_codepoint(i); cp <= hi_codepoint(i); cp++) {
            std::u32string decomp;
            mEngine.NFD_append1(decomp, cp);
            if (decomp.size() == 1) {
                finalCC = makeCC(finalCC, makeCC(decomp[0]));
            } else {
                alts.push_back(u32string2re(decomp));
            }
        }
    }
    if (!finalCC->empty()) alts.push_back(finalCC);
    return makeAlt(alts.begin(), alts.end());
}

RE * NFD_Transformer::transformSeq(Seq * seq) {
    // find and process all string pieces
    unsigned size = seq->size();
    if (size == 0) return seq;
    std::vector<RE *> list;
    unsigned i = 0;
    bool unchanged = true;
    while (i < size) {
        std::u32string stringPiece = getStringPiece(seq, i);
        if (stringPiece.size() > 0) {
            std::u32string s;
            mEngine.NFD_append(s, stringPiece);
            if (s != stringPiece) unchanged = false;
            list.push_back(u32string2re(s));
            i += stringPiece.size();
        } else {
            RE * r = (*seq)[i];
            RE * t = transform(r);
            if (t != r) unchanged = false;
            list.push_back(t);
            i++;
        }
    }
    if (unchanged) return seq;
    return makeSeq(list.begin(), list.end());
}

RE * toNFD(RE * re, const DecompositionOptions opt) {
    NFD_Engine engine(opt);
    return NFD_Transformer(opt, engine).transformRE(re);
}

} // end namespace UCD
