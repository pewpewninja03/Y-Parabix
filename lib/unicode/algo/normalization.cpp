/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <unicode/algo/normalization.h>
#include <vector>
#include <llvm/Support/Casting.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/data/PropertyValueAliases.h>

using namespace llvm;

namespace UCD {
    
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
HangulPrecomposed(Hangul::SBase, Hangul::SBase + Hangul::SCount - 1) {

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
        auto SIndex = cp - Hangul::SBase;
        auto LIndex = SIndex / Hangul::NCount;
        auto VIndex = (SIndex % Hangul::NCount) / Hangul::TCount;
        auto TIndex = SIndex % Hangul::TCount;
        NFD_string.push_back(Hangul::LBase + LIndex);
        NFD_string.push_back(Hangul::VBase + VIndex);
        if (TIndex > 0) {
            NFD_string.push_back(Hangul::TBase + TIndex);
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

void NFD_Engine::prepareUnicodeBitTranslationData() {
    auto cps = decompMappingObj->GetExplicitCps();
    std::vector<unicode::TranslationMap> xmaps;
    for (auto cp : cps) {
        std::u32string decomp;
        NFD_append1(decomp, cp);
        while (xmaps.size() < decomp.length()) {
            xmaps.push_back(unicode::TranslationMap());
        }
        unsigned diff = decomp.length() - 1;
        unsigned bit = 0;
        while (diff > 0) {
            if ((diff & 1UL) == 1UL) {
                while (decomposition_insert_length_bixnum_sets.size() <= bit) {
                    decomposition_insert_length_bixnum_sets.push_back(UnicodeSet());
                }
                decomposition_insert_length_bixnum_sets[bit].insert(cp);
            }
            diff >>= 1;
            bit++;
        }
        for (unsigned i = 0; i < decomp.length(); i++) {
            xmaps[i].emplace(cp, decomp[i]);
        }
    }
    UCD::UnicodeSet Hangul_Precomposed(Hangul::SBase, Hangul::MaxPrecomposed);
    UCD::UnicodeSet Hangul_Precomposed_LV;
    for (UCD::codepoint_t cp = Hangul::SBase; cp <= Hangul::MaxPrecomposed; cp += Hangul::TCount) {
        Hangul_Precomposed_LV.insert(cp);
    }
    UCD::UnicodeSet Hangul_Precomposed_LVT = Hangul_Precomposed - Hangul_Precomposed_LV;
    // Each LV requires expansion by a single position, each LVT by two positions.
    decomposition_insert_length_bixnum_sets[0] = decomposition_insert_length_bixnum_sets[0] + Hangul_Precomposed_LV;
    decomposition_insert_length_bixnum_sets[1] = decomposition_insert_length_bixnum_sets[1] + Hangul_Precomposed_LVT;
    bitXlatSets.resize(xmaps.size());
    bitXlatSets[0] = unicode::ComputeBitTranslationSets(xmaps[0], unicode::XlateMode::XorBit);
    for (unsigned j = 1; j < xmaps.size(); j++) {
        bitXlatSets[j] = unicode::ComputeBitTranslationSets(xmaps[j], unicode::XlateMode::LiteralBit);
    }
}

unicode::BitTranslationSets NFD_Engine::UnicodeInsertLengthBixNumSets() {
    if (decomposition_insert_length_bixnum_sets.size() == 0) {
        prepareUnicodeBitTranslationData();
    }
    return decomposition_insert_length_bixnum_sets;
}

std::vector<unicode::BitTranslationSets> NFD_Engine::UnicodeBitTransformSets() {
    if (decomposition_insert_length_bixnum_sets.size() == 0) {
        prepareUnicodeBitTranslationData();
    }
    return bitXlatSets;
}

} // end namespace UCD
