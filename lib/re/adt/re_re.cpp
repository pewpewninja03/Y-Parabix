#include <re/adt/re_re.h>
#include <re/adt/adt.h>

using namespace llvm;

namespace re {

RE::Allocator RE::mAllocator;

bool matchesEmptyString(const RE * re) {
    if (const Alt * alt = dyn_cast<Alt>(re)) {
        for (const RE * re : *alt) {
            if (matchesEmptyString(re)) {
                return true;
            }
        }
        return false;
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        for (const RE * re : *seq) {
            if (!matchesEmptyString(re)) {
                return false;
            }
        }
        return true;
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        return (rep->getLB() == 0) || matchesEmptyString(rep->getRE());
    } else if (isa<Start>(re)) {
        return true;
    } else if (isa<End>(re)) {
        return true;
    } else if (const Assertion * a = dyn_cast<Assertion>(re)) {
        return a->getSense() == Assertion::Sense::Negative;
    } else if (const Diff * diff = dyn_cast<Diff>(re)) {
        return matchesEmptyString(diff->getLH()) && !matchesEmptyString(diff->getRH());
    } else if (const Intersect * e = dyn_cast<Intersect>(re)) {
        return matchesEmptyString(e->getLH()) && matchesEmptyString(e->getRH());
    } else if (isa<Any>(re)) {
        return false;
    } else if (isa<CC>(re)) {
        return false;
    } else if (const Group * g = dyn_cast<Group>(re)) {
        return matchesEmptyString(g->getRE());
    } else if (const Name * n = dyn_cast<Name>(re)) {
        return matchesEmptyString(n->getDefinition());
    } else if (const Capture * c = dyn_cast<Capture>(re)) {
        return matchesEmptyString(c->getCapturedRE());
    }
    return false; // otherwise
}

[[noreturn]] void UnsupportedRE(const std::string & errmsg) {
    llvm::report_fatal_error(llvm::StringRef(errmsg));
}

}
