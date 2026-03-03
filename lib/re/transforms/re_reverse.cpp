/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/re_reverse.h>

#include <map>
#include <llvm/Support/ErrorHandling.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_name.h>
#include <re/adt/re_start.h>
#include <re/adt/re_end.h>
#include <re/adt/re_any.h>
#include <re/adt/re_seq.h>
#include <re/adt/re_alt.h>
#include <re/adt/re_rep.h>
#include <re/adt/re_group.h>
#include <re/adt/re_range.h>
#include <re/adt/re_diff.h>
#include <re/adt/re_intersect.h>
#include <re/adt/re_assertion.h>
#include <re/transforms/re_transformer.h>


using namespace llvm;

namespace re {
    
using CaptureMap = std::map<std::string, std::pair<Capture *, unsigned>>;

class ReverseTransformer : public RE_Transformer {
public:
    ReverseTransformer() : RE_Transformer("Reverse") {}
    RE * transformCapture(Capture * c) override {
        std::string cname = c->getName();
        auto f = mCaptureMap.find(cname);
        if (f != mCaptureMap.end()) {
            Capture * captured = f->second.first;
            unsigned instanceCount = f->second.second;
            RE * ref = makeReference(captured->getName(), captured, instanceCount);
            f->second = std::make_pair(captured, instanceCount+1);
            return ref;
        }
        else {
            std::string newName = "\\" + std::to_string(mCaptureMap.size() + 1);
            Capture * capture = makeCapture(newName, transform(c->getCapturedRE()));
            mCaptureMap.emplace(cname, std::make_pair(capture, 0));
            return capture;
        }
    }
    RE * transformReference(Reference * r) override {
        std::string cname = r->getName();
        Capture * referent = r->getCapture();
        auto f = mCaptureMap.find(cname);
        if (f != mCaptureMap.end()) {
            Capture * captured = f->second.first;
            unsigned instanceCount = f->second.second;
            Reference * ref = makeReference(cname, captured, instanceCount);
            f->second = std::make_pair(captured, instanceCount+1);
            return ref;
        }
        else {
            std::string newName = "\\" + std::to_string(mCaptureMap.size() + 1);
            Capture * capture = makeCapture(newName, transform(referent->getCapturedRE()));
            mCaptureMap.emplace(cname, std::make_pair(capture, 0));
            return capture;
        }
    }
    RE * transformSeq (Seq * seq) override {
        std::vector<RE*> list;
        for (auto i = seq->rbegin(); i != seq->rend(); ++i) {
            list.push_back(transform(*i));
        }
        return makeSeq(list.begin(), list.end());
    }
    RE * transformStart (Start *) override { return makeEnd();}
    RE * transformEnd (End *) override { return makeStart();}
    RE * transformAssertion (Assertion * a) override {
        RE * asserted = a->getAsserted();
        RE * reversed = transform(asserted);
        return makeAssertion(reversed, Assertion::reverseKind(a->getKind()), a->getSense());
    }
    
private:
    CaptureMap mCaptureMap;
};

RE * reverse(RE * re) {
    return ReverseTransformer().transformRE(re);
}

}
