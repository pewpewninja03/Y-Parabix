/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <unicode/algo/decomposition.h>
#include <vector>
#include <llvm/Support/Casting.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/data/PropertyValueAliases.h>

using namespace llvm;

namespace UCD {
    
// Constants for computation of Hangul decompositions, see Unicode Standard, section 3.12.
const codepoint_t Hangul_SBase = 0xAC00;
const codepoint_t Hangul_LBase = 0x1100;
//const codepoint_t Hangul_LMax = 0x1112;
const codepoint_t Hangul_VBase = 0x1161;
//const codepoint_t Hangul_VMax = 0x1175;
const codepoint_t Hangul_TBase = 0x11A7;
//const codepoint_t Hangul_TMax = 0x11C2;
const unsigned Hangul_TCount = 28;
const unsigned Hangul_NCount = 588;
const unsigned Hangul_SCount = 11172;

NFD_Engine::NFD_Engine(DecompositionOptions opt) :
mOptions(opt),
decompTypeObj(cast<EnumeratedPropertyObject>(getPropertyObject(dt))),
decompMappingObj(cast<StringPropertyObject>(getPropertyObject(dm))),
cccObj(cast<EnumeratedPropertyObject>(getPropertyObject(ccc))),
caseFoldObj(cast<StringOverridePropertyObject>(getPropertyObject(cf))),
canonicalMapped(decompTypeObj->GetCodepointSet(DT_ns::Can)),
cc0Set(cccObj->GetCodepointSet(CCC_ns::NR)),
selfNFKD(decompMappingObj->GetReflexiveSet()),
selfCaseFold(caseFoldObj->GetReflexiveSet()),
HangulPrecomposed(Hangul_SBase, Hangul_SBase + Hangul_SCount - 1) {

}

bool hasOption(enum DecompositionOptions optionSet, enum DecompositionOptions testOption) {
    return (testOption & optionSet) != 0;
}
    
bool NFD_Engine::reordering_needed(std::u32string & prefix, codepoint_t suffix_cp) {
    if (prefix.empty()) return false;
    if (cc0Set.contains(suffix_cp)) return false;
    auto cc1 = cccObj->GetEnumerationValue(prefix.back());
    auto cc2 = cccObj->GetEnumerationValue(suffix_cp);
    return cc1 > cc2;
}

void NFD_Engine::NFD_append1(std::u32string & NFD_string, codepoint_t cp) {
    if (HangulPrecomposed.contains(cp)) {
        // Apply NFD normalization; no NFKD or casefolding required
        auto SIndex = cp - Hangul_SBase;
        auto LIndex = SIndex / Hangul_NCount;
        auto VIndex = (SIndex % Hangul_NCount) / Hangul_TCount;
        auto TIndex = SIndex % Hangul_TCount;
        NFD_string.push_back(Hangul_LBase + LIndex);
        NFD_string.push_back(Hangul_VBase + VIndex);
        if (TIndex > 0) {
            NFD_string.push_back(Hangul_TBase + TIndex);
        }
    } else if (canonicalMapped.contains(cp)) {
        std::u32string dms = decompMappingObj->GetU32StringValue(cp);
        // Recursive normalization may be necessary.
        NFD_append(NFD_string, dms);
        // After canonical mappings are handled, canonical ordering may be required.
        // This should be done before casefolding.
    } else if (reordering_needed(NFD_string, cp)) {
        // Reorder the last two characters - recursion will handle
        // rare multiposition reordering.
        std::u32string reordered({cp, NFD_string.back()});
        NFD_string.pop_back();
        NFD_append(NFD_string, reordered);
    } else if (hasOption(mOptions, CaseFold) && !selfCaseFold.contains(cp)) {
        std::u32string dms = caseFoldObj->GetU32StringValue(cp);
        NFD_append(NFD_string, dms);
    } else if (hasOption(mOptions, NFKD) && (!selfNFKD.contains(cp))) {
        std::u32string dms = decompMappingObj->GetU32StringValue(cp);
        NFD_append(NFD_string, dms);
    } else {
        NFD_string.push_back(cp);
    }
}

void NFD_Engine::NFD_append(std::u32string & NFD_string, std::u32string & to_convert) {
    for (unsigned i = 0; i < to_convert.size(); i++) {
        NFD_append1(NFD_string, to_convert[i]);
    }
}

std::u32string NFD_Engine::decompose (std::u32string to_decompose) {
    std::u32string rslt;
    NFD_append(rslt, to_decompose);
    return rslt;
}
} // end namespace UCD
