#include <re/analysis/re_name_gather.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_inspector.h>

using namespace llvm;
namespace re {
    
struct NameCollector final : public RE_Inspector {

    NameCollector(std::set<Name *> & nameSet)
    : RE_Inspector()
    , mNameSet(nameSet) {

    }

    void inspectName(Name * n) final {
        mNameSet.insert(n);
    }

private:
    std::set<Name *> & mNameSet;
};

void gatherNames(RE * const re, std::set<Name *> & nameSet) {
    NameCollector collector(nameSet);
    collector.inspectRE(re);
}

struct ExternalCollector final : public RE_Inspector {

    ExternalCollector(std::set<std::string> & externals,
                      std::set<std::string> & alphabets)
    : RE_Inspector(), mExternals(externals), mAlphabets(alphabets) {}

    void inspectName(Name * n) override {
        mExternals.insert(n->getFullName());
    }

    void inspectCC(CC * cc) override {
        auto alpha = cc->getAlphabet();
        mAlphabets.insert(alpha->getName());
    }

    std::set<std::string> & mExternals;
    std::set<std::string> & mAlphabets;
};

void gatherExternals(RE * const re,
                     std::set<std::string> & externals,
                     std::set<std::string> & alphabets) {
    ExternalCollector collector(externals, alphabets);
    collector.inspectRE(re);
}
}
