#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/expand_permutes.h>
#include <re/printer/re_printer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>

using namespace llvm;

namespace re {



class ExpandPermutes final : public RE_Transformer {
public:
    ExpandPermutes(const cc::Alphabet * lengthAlpha) : 
        RE_Transformer("ExpandPermutes"), mLengthAlphabet(lengthAlpha) {}
    RE * transformPermute(Permute * p) override;
private:
    const cc::Alphabet * mLengthAlphabet;
};

RE * ExpandPermutes::transformPermute(Permute * p) {
    unsigned perm_size = p->size();
    unsigned total_length = 0;
    std::vector<RE *> alts(perm_size);
    std::vector<unsigned> alt_lgth(perm_size);
    unsigned i = 0;
    for (auto perm : *p) {
        alts[i] = perm;
        auto rg =  getLengthRange(perm, mLengthAlphabet);
        if (rg.first != rg.second) {
            llvm::report_fatal_error("Variable length permutation terms are prohibited.");
        }
        alt_lgth[i] = rg.first;
        total_length += rg.first;
        i++;
    }
    std::vector<RE *> elems(perm_size + 1);
    RE * anyAlt = makeAlt(alts.begin(), alts.end());
    elems.push_back(makeRep(anyAlt, perm_size, perm_size));
    i = 1;
    for (auto perm : *p) {
        unsigned remlgth = total_length - alt_lgth[i];
        RE * r = makeSeq({perm, makeRep(makeAny(mLengthAlphabet), 0, remlgth)});
        elems[i] = makeLookBehindAssertion(r);
        i++;
    }
    return makeSeq(elems.begin(), elems.end());
}

RE * expandPermutes(RE * r, const cc::Alphabet * lengthAlpha) {
    return ExpandPermutes(lengthAlpha).transformRE(r);
}

}
