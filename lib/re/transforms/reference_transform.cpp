#include <re/transforms/reference_transform.h>

#include <re/adt/adt.h>
#include <re/analysis/capture-ref.h>
#include <re/analysis/re_analysis.h>
#include <ucd/data/PropertyAliases.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace re {


RE * FixedReferenceTransformer::transformReference(Reference * r) {
    auto rg1 = getLengthRange(r->getCapture(), &cc::Unicode);
    if (rg1.first != rg1.second) return r;
    if (rg1.first != 1) {
        llvm::report_fatal_error("Capture length > 1 is a future extension");
    }
    std::string instanceName = r->getInstanceName();
    auto mapping = mRefInfo.twixtREs.find(instanceName);
    if (mapping == mRefInfo.twixtREs.end()) return r;
    auto rg2 = getLengthRange(mapping->second, &cc::Unicode);
    if (rg2.first != rg2.second) return r;
    UCD::property_t p = r->getReferencedProperty();
    std::string pname = p == UCD::identity ? "Unicode" : UCD::getPropertyFullName(p);
    auto fixed_dist = std::to_string(rg1.first + rg2.first);
    auto matchLen = std::to_string(rg1.first);
    std::string externalName = pname + "@-" + fixed_dist + "_" + matchLen;
    return createName(externalName, r);
}
}
