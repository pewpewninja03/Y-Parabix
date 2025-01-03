/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 */

#include <unicode/data/PropertyObjects.h>
#include <string>
#include <locale>
#include <codecvt>
#include <sstream>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#include <unicode/core/unicode_set.h>
#include <unicode/data/PropertyObjectTable.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Twine.h>
#include <util/aligned_allocator.h>
#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>

using namespace llvm;

namespace UCD {

std::string canonicalize_value_name(const std::string & prop_or_val) {
    std::locale loc;
    std::stringstream s;

    for (char c : prop_or_val) {
        if ((c != '_') && (c != ' ') && (c != '-')) {
            s << std::tolower(c, loc);
        }
    }
    return s.str();
}

const std::string & PropertyObject::GetPropertyValueGrepString() {
    report_fatal_error("Property Value Grep String unsupported.");
}

const UnicodeSet PropertyObject::GetCodepointSet(const std::string &) {
    report_fatal_error(Twine("Property ") + UCD::getPropertyFullName(the_property) + " unsupported.");
}

const UnicodeSet PropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    report_fatal_error(Twine("GetCodepointSetMatchingPattern for ") + UCD::getPropertyFullName(the_property) + " unsupported.");
}

const std::u32string PropertyObject::GetU32StringValue(codepoint_t cp) const {
    std::string s = GetStringValue(cp);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(s);
#pragma GCC diagnostic pop
}

const std::string PropertyObject::GetStringValue(codepoint_t cp) const {
    report_fatal_error("GetStringValue unsupported");
}

const UnicodeSet PropertyObject::GetPropertyIntersection(PropertyObject * p) {
    return UnicodeSet();
}

const UnicodeSet PropertyObject::GetReflexiveSet() const {
    return UnicodeSet();
}

const UnicodeSet PropertyObject::GetNullSet() const {
    return UnicodeSet();
}

const UnicodeSet EnumeratedPropertyObject::GetCodepointSet(const std::string & value_spec) {
    const int property_enum_val = GetPropertyValueEnumCode(value_spec);
    if (property_enum_val < 0) {
        report_fatal_error(Twine("Enumerated Property ") + UCD::getPropertyFullName(the_property) + ": unknown value: " + value_spec);
    }
    return GetCodepointSet(property_enum_val);
}

const UnicodeSet EnumeratedPropertyObject::GetCodepointSet(const int property_enum_val) const {
    assert (property_enum_val >= 0);
    UCD::UnicodeSet s;
    if ((getPropertyCode() == UCD::property_t::age) && (property_enum_val > 0)) {
        // Special logic for the age property:  \p{age=x.y} includes
        // all codepoints defined as of version x.y, i.e., whose age
        // property is numerically less than or equal to x.y.
        for (int a = 1; a <= property_enum_val; a++) {
            s.insert(*(property_value_sets[a]));
        }
        return s;
    }
    return *(property_value_sets[property_enum_val]);
}

const UnicodeSet EnumeratedPropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    AlignedAllocator<char, 64> alloc;
    std::vector<std::string> accumulatedValues;

    const std::string & str = GetPropertyValueGrepString();

    const unsigned segmentSize = 8;
    const auto n = str.length();
    const auto w = 256 * segmentSize;
    const auto m = w - (n % w);

    char * aligned = alloc.allocate(n + m, 0);
    std::memcpy(aligned, str.data(), n);
    std::memset(aligned + n, 0, m);
    std::vector<uint64_t> matchedEnums = grep(re, aligned, n);
    alloc.deallocate(aligned, 0);
    const unsigned enumCount = GetEnumCount();
    UCD::UnicodeSet s;
    for (const auto v : matchedEnums) {
        s.insert(GetCodepointSet(v % enumCount));
    }
    return s;
}

std::vector<UnicodeSet> & EnumeratedPropertyObject::GetEnumerationBasisSets() {
    // Return the previously computed vector of basis sets, if it exists.
    if (LLVM_UNLIKELY(enumeration_basis_sets.empty())) {
        // Otherwise compute and return.
        // Basis set i is the set of all codepoints whose numerical enumeration code e
        // has bit i set, i.e., (e >> i) & 1 == 1.
        unsigned basis_count = 1;
        while ((1UL << basis_count) < independent_enum_count) {
            basis_count++;
        }
        for (unsigned i = 0; i < basis_count; i++) {
            enumeration_basis_sets.push_back(UnicodeSet());
            for (unsigned e = 0; e < independent_enum_count; e++) {
                if (((e >> i) & 1UL) == 1UL) {
                    enumeration_basis_sets[i] = enumeration_basis_sets[i] + *property_value_sets[e];
                }
            }
        }
    }
    return enumeration_basis_sets;
}

unsigned EnumeratedPropertyObject::GetEnumerationValue(codepoint_t cp) {
    unsigned enum_value = 0;
    if (LLVM_UNLIKELY(enumeration_basis_sets.empty())) {
        GetEnumerationBasisSets();
    }
    unsigned basis_count = enumeration_basis_sets.size();
    for (unsigned i = 0; i < basis_count; i++) {
        if (enumeration_basis_sets[i].contains(cp)) {
            enum_value += 1 << i;
        }
    }
    return enum_value;
}

const std::string & EnumeratedPropertyObject::GetPropertyValueGrepString() {
    if (LLVM_LIKELY(mPropertyValueGrepString.empty())) {
        std::stringstream buffer;
        for (unsigned i = 0; i != property_value_full_names.size(); i++) {
            buffer << property_value_full_names[i] + "\n";
        }
        for (unsigned i = 0; i != property_value_enum_names.size(); i++) {
            buffer << property_value_enum_names[i] + "\n";
        }
        for (unsigned i = 0; i != property_value_aliases.size(); i++) {
            buffer << property_value_aliases[i] + "\n";
        }
        mPropertyValueGrepString = buffer.str();
    }
    return mPropertyValueGrepString;
}

int EnumeratedPropertyObject::GetPropertyValueEnumCode(const std::string & value_spec) {
    // The canonical full names are not stored in the precomputed alias map,
    // to save space in the executable.   Add them if the property is used.
    if (uninitialized) {
        for (unsigned i = 0; i != property_value_full_names.size(); i++) {
            enum_name_map.insert({canonicalize_value_name(property_value_full_names[i]), i});
        }
        for (unsigned i = 0; i != property_value_enum_names.size(); i++) {
            enum_name_map.insert({canonicalize_value_name(property_value_enum_names[i]), i});
        }
        for (unsigned i = 0; i != property_value_aliases.size(); i++) {
            enum_name_map.insert({canonicalize_value_name(property_value_aliases[i]), i});
        }
        uninitialized = false;
    }
    const auto valit = enum_name_map.find(canonicalize_value_name(value_spec));
    if (valit == enum_name_map.end())
        return -1;
    return valit->second;
}

PropertyObject::iterator ExtensionPropertyObject::begin() const {
    if (const auto * obj = dyn_cast<EnumeratedPropertyObject>(getPropertyObject(base_property))) {
        return obj->begin();
    }
    report_fatal_error("Iterators unsupported for this type of PropertyObject.");
}

PropertyObject::iterator ExtensionPropertyObject::end() const {
    if (const auto * obj = dyn_cast<EnumeratedPropertyObject>(getPropertyObject(base_property))) {
        return obj->end();
    }
    report_fatal_error("Iterators unsupported for this type of PropertyObject.");
}

const UnicodeSet ExtensionPropertyObject::GetCodepointSet(const std::string & value_spec) {
    int property_enum_val = GetPropertyValueEnumCode(value_spec);
    if (property_enum_val == -1) {
        report_fatal_error(Twine("Extension Property ") + UCD::getPropertyFullName(the_property) +  ": unknown value: " + value_spec);
    }
    return GetCodepointSet(property_enum_val);
}

const UnicodeSet ExtensionPropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    AlignedAllocator<char, 64> alloc;
    std::vector<std::string> accumulatedValues;

    UCD::EnumeratedPropertyObject * baseObj = llvm::cast<UCD::EnumeratedPropertyObject>(getPropertyObject(base_property));

    const std::string & str = baseObj->GetPropertyValueGrepString();
    const unsigned segmentSize = 8;
    const auto n = str.length();
    const auto w = 256 * segmentSize;
    const auto m = w - (n % w);

    char * aligned = alloc.allocate(n + m, 0);
    std::memcpy(aligned, str.data(), n);
    std::memset(aligned + n, 0, m);
    std::vector<uint64_t> matchedEnums = grep(re, aligned, n);
    alloc.deallocate(aligned, 0);
    UCD::UnicodeSet a;
    int enumCount = baseObj->GetEnumCount();
    for (const auto v : matchedEnums) {
        a.insert(GetCodepointSet(v % enumCount));
    }
    return a;
}

const UnicodeSet & ExtensionPropertyObject::GetCodepointSet(const int property_enum_val) const {
    assert (property_enum_val >= 0);
    return *(property_value_sets[property_enum_val]);
}

int ExtensionPropertyObject::GetPropertyValueEnumCode(const std::string & value_spec) {
    return cast<EnumeratedPropertyObject>(getPropertyObject(base_property))->GetPropertyValueEnumCode(value_spec);
}

const std::string & ExtensionPropertyObject::GetPropertyValueGrepString() {
    return getPropertyObject(base_property)->GetPropertyValueGrepString();
}

const UnicodeSet BinaryPropertyObject::GetCodepointSet(const std::string & value_spec) {
    return GetCodepointSet(GetPropertyValueEnumCode(value_spec));
}

int BinaryPropertyObject::GetPropertyValueEnumCode(const std::string & value_spec) {
    int property_enum_val = Binary_ns::Y;
    if (value_spec.length() != 0) {
        auto valit = Binary_ns::aliases_only_map.find(canonicalize_value_name(value_spec));
        if (valit == Binary_ns::aliases_only_map.end()) {
            report_fatal_error(Twine("Binary Property ") + UCD::getPropertyFullName(the_property) +  ": bad value: " + value_spec);
        }
        property_enum_val = valit->second;
    }
    return property_enum_val;
}

const UnicodeSet & BinaryPropertyObject::GetCodepointSet(const int property_enum_val) {
    if (property_enum_val == Binary_ns::Y) {
        return mY;
    }
    if (mN.get() == nullptr) {
        mN = std::make_unique<UnicodeSet>(~mY);
    }
    return *mN;
}

const UnicodeSet BinaryPropertyObject::GetPropertyIntersection(PropertyObject * p) {
    if (BinaryPropertyObject * b = dyn_cast<BinaryPropertyObject>(p)) {
        return mY & b->GetCodepointSet(UCD::Binary_ns::Y);
    } else return UnicodeSet();
}

const std::string & BinaryPropertyObject::GetPropertyValueGrepString() {
    if (mPropertyValueGrepString.empty()) {
        std::stringstream buffer;
        for (const auto & prop : Binary_ns::aliases_only_map) {
            buffer << std::get<0>(prop) + "\n";
        }
        mPropertyValueGrepString = buffer.str();
    }
    return mPropertyValueGrepString;
}

const unsigned firstCodepointLengthAndVal(const std::string & s, codepoint_t & cp) {
    size_t lgth = s.length();
    cp = 0;
    if (lgth == 0) return 0;
    unsigned char s0 = s[0];
    cp = static_cast<codepoint_t>(s0);
    if (s0 < 0x80) return 1;
    if (lgth == 1) return 0;  // invalid UTF-8
    cp = ((cp & 0x1F) << 6) | (s[1] & 0x3F);
    if ((s0 >= 0xC2) && (s0 <= 0xDF)) return 2;
    if (lgth == 2) return 0;  // invalid UTF-8
    cp = ((cp & 0x3FFF) << 6) | (s[2] & 0x3F);
    if ((s0 >= 0xE0) && (s0 <= 0xEF)) return 3;
    if (lgth == 3) return 0;  // invalid UTF-8
    cp = ((cp & 0x7FFF) << 6) | (s[3] & 0x3F);
    if ((s0 >= 0xF0) && (s0 <= 0xF4)) return 4;
    return 0;
}

const UnicodeSet NumericPropertyObject::GetCodepointSet(const std::string & value_spec) {
    if (value_spec == "NaN") {
        return mNaNCodepointSet;
    } else {
        UnicodeSet result_set;
        unsigned val_bytes = value_spec.length();
        const char * value_str = value_spec.c_str();
        const char * search_str = mStringBuffer;
        unsigned buffer_line = 0;
        while (buffer_line < mExplicitCps.size()) {
            const char * eol = strchr(search_str, '\n');
            unsigned len = eol - search_str;
            if ((len == val_bytes) && (memcmp(search_str, value_str, len) == 0)) {
                result_set.insert(mExplicitCps[buffer_line]);
            }
            buffer_line++;
            search_str = eol+1;
        }
        return result_set;
    }
}

const UnicodeSet NumericPropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    UCD::UnicodeSet matched;
    std::vector<uint64_t> matchedLines = grep(re, mStringBuffer, mBufSize);
    for (const auto v : matchedLines) {
        matched.insert(mExplicitCps[v]);
    }
    return matched;
}

const UnicodeSet CodePointPropertyObject::GetCodepointSet(const std::string & value_spec) {
    unsigned val_bytes = value_spec.length();
    codepoint_t cp;
    unsigned cp_bytes = firstCodepointLengthAndVal(value_spec, cp);
    if (val_bytes != cp_bytes) {
        return UnicodeSet();  // empty set
    }
    UnicodeSet result_set;
    if (mSelfCodepointSet.contains(cp)) {
        result_set.insert(cp);
    }
    for (auto & p : mExplicitCodepointMap) {
        if (p.second == cp) {
            result_set.insert(p.first);
        }
    }
    return result_set;
}

const UnicodeSet CodePointPropertyObject::GetPropertyIntersection(PropertyObject * p) {
    UnicodeSet intersection;
    if (isa<CodePointPropertyObject>(p) || isa<StringPropertyObject>(p) || isa<StringOverridePropertyObject>(p)) {
        intersection = (mNullCodepointSet & p->GetNullSet()) + (mSelfCodepointSet & p->GetReflexiveSet());
        for (auto & mapping : mExplicitCodepointMap) {
            if (GetStringValue(mapping.first) == p->GetStringValue(mapping.first)) {
                intersection.insert(mapping.first);
            }
        }
        return intersection;
    } else return UnicodeSet();
}

const UnicodeSet CodePointPropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    const re::CC * matchable = re::matchableCodepoints(re);
    UCD::UnicodeSet matched(*matchable & mSelfCodepointSet);
    for (auto & p : mExplicitCodepointMap) {
        if (matched.contains(p.second)) {
            matched.insert(p.first);
        }
    }
    return matched;
}

const UnicodeSet CodePointPropertyObject::GetNullSet() const {
    return mNullCodepointSet;
}

const UnicodeSet CodePointPropertyObject::GetReflexiveSet() const {
    return mSelfCodepointSet;
}

const codepoint_t CodePointPropertyObject::GetCodePointValue(codepoint_t cp) const {
    if (mSelfCodepointSet.contains(cp)) {
        return cp;
    }
    auto f = mExplicitCodepointMap.find(cp);
    if (f != mExplicitCodepointMap.end()) {
        return f->second;
    }
    llvm::report_fatal_error("codepoint property value not found");
}

const std::u32string CodePointPropertyObject::GetU32StringValue(codepoint_t cp) const {
    std::u32string s(1, GetCodePointValue(cp));
    return s;
}

const std::string CodePointPropertyObject::GetStringValue(codepoint_t cp) const {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::u32string s(1, GetCodePointValue(cp));
    return conv.to_bytes(s);
#pragma GCC diagnostic pop
}

std::vector<UCD::UnicodeSet> & CodePointPropertyObject::GetBitTransformSets() {
    // Return the previously computed vector of bit transformation sets, if it exists.
    if (bit_xform_sets.empty()) {
        bit_xform_sets = unicode::ComputeBitTranslationSets(mExplicitCodepointMap);
    }
    return bit_xform_sets;
}

void CodePointPropertyObject::compute_u8_movement() {
    u8_insertion_bixnum = unicode::ComputeUTF8_insertionBixNum(mExplicitCodepointMap);
    u8_deletion_bixnum = unicode::ComputeUTF8_deletionBixNum(mExplicitCodepointMap);
    u8_movement_initialized = true;
}

std::vector<UnicodeSet> & CodePointPropertyObject::GetUTF8insertionBixNum() {
    if (!u8_movement_initialized) {
        compute_u8_movement();
    }
    return u8_insertion_bixnum;
}

std::vector<UnicodeSet> & CodePointPropertyObject::GetUTF8deletionBixNum() {
    if (!u8_movement_initialized) {
        compute_u8_movement();
    }
    return u8_deletion_bixnum;
}

const UnicodeSet StringPropertyObject::GetCodepointSet(const std::string & value_spec) {
    if (value_spec.empty()) {
        return mNullCodepointSet;
    } else {
        UnicodeSet result_set;
        unsigned val_bytes = value_spec.length();
        codepoint_t cp;
        if (val_bytes == firstCodepointLengthAndVal(value_spec, cp)) {
            if (mSelfCodepointSet.contains(cp)) {
                result_set.insert(cp);
            }
        }
        const char * value_str = value_spec.c_str();
        const char * search_str = mStringBuffer;
        unsigned buffer_line = 0;
        while (buffer_line < mExplicitCps.size()) {
            const char * eol = strchr(search_str, '\n');
            unsigned len = eol - search_str;
            if ((len == val_bytes) && (memcmp(search_str, value_str, len) == 0)) {
                result_set.insert(mExplicitCps[buffer_line]);
            }
            buffer_line++;
            search_str = eol+1;
        }
        return result_set;
    }
}

const UnicodeSet StringPropertyObject::GetPropertyIntersection(PropertyObject * p) {
    UnicodeSet intersection;
    if (isa<CodePointPropertyObject>(p)) {
        return p->GetPropertyIntersection(this);
    }
    if (isa<StringPropertyObject>(p) || isa<StringOverridePropertyObject>(p)) {
        intersection = (mNullCodepointSet & p->GetNullSet()) + (mSelfCodepointSet & p->GetReflexiveSet());
        for (unsigned i = 0; i < mExplicitCps.size(); i++) {
            if (GetStringValue(mExplicitCps[i]) == p->GetStringValue(mExplicitCps[i])) {
                intersection.insert(mExplicitCps[i]);
            }
        }
        return intersection;
    } else return UnicodeSet();
}

const UnicodeSet StringPropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    UCD::UnicodeSet matched(*re::matchableCodepoints(re) & mSelfCodepointSet);
    if (re::matchesEmptyString(re)) {
        matched.insert(mNullCodepointSet);
    }
    const unsigned bufSize = mStringOffsets[mExplicitCps.size()];
    std::vector<uint64_t> matchedLines = grep(re, mStringBuffer, bufSize);
    for (const auto v : matchedLines) {
        matched.insert(mExplicitCps[v]);
    }
    return matched;
}

const UnicodeSet StringPropertyObject::GetNullSet() const {
    return mNullCodepointSet;
}

const UnicodeSet StringPropertyObject::GetReflexiveSet() const {
    return mSelfCodepointSet;
}

const std::u32string StringPropertyObject::GetU32StringValue(codepoint_t cp) const {
    if (mNullCodepointSet.contains(cp)) return std::u32string();
    if (mSelfCodepointSet.contains(cp)) {
        std::u32string s(1, cp);
        return s;
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    std::string s = GetStringValue(cp);
    return conv.from_bytes(s);
#pragma GCC diagnostic pop
}

const std::string StringPropertyObject::GetStringValue(codepoint_t cp) const {
    if (mNullCodepointSet.contains(cp)) return "";
    if (mSelfCodepointSet.contains(cp)) {
        std::u32string s(1, cp);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        return conv.to_bytes(s);
#pragma GCC diagnostic pop
    }
    // Otherwise, binary search through the explicit cps to find the index.
    // string index.
    unsigned lo = 0;
    unsigned hi = mExplicitCps.size()-1;
    while (lo < hi) {
        unsigned mid = (lo + hi)/2;
        if (cp <= mExplicitCps[mid]) hi = mid;
        else lo = mid + 1;
    }
    // Now lo == hi is the index of the desired string.
    unsigned offset = mStringOffsets[lo];
    unsigned lgth = mStringOffsets[lo+1] - offset - 1;
    return std::string(&mStringBuffer[offset], lgth);
}

const UnicodeSet StringOverridePropertyObject::GetCodepointSet(const std::string & value_spec) {
    // First step: get the codepoints from the base object and then remove any overridden ones.
    UnicodeSet result_set = getPropertyObject(mBaseProperty)->GetCodepointSet(value_spec) - mOverriddenSet;
    // Now search for additional entries.
    unsigned val_bytes = value_spec.length();
    const char * value_str = value_spec.c_str();
    const char * search_str = mStringBuffer;
    unsigned buffer_line = 0;
    while (buffer_line < mExplicitCps.size()) {
        const char * eol = strchr(search_str, '\n');
        unsigned len = eol - search_str;
        if ((len == val_bytes) && (memcmp(search_str, value_str, len) == 0)) {
            result_set.insert(mExplicitCps[buffer_line]);
        }
        buffer_line++;
        search_str = eol+1;
    }
    return result_set;
}

const UnicodeSet StringOverridePropertyObject::GetNullSet() const {
    return getPropertyObject(mBaseProperty)->GetNullSet() - mOverriddenSet;
}

const UnicodeSet StringOverridePropertyObject::GetReflexiveSet() const {
    return getPropertyObject(mBaseProperty)->GetReflexiveSet() - mOverriddenSet;
}

const std::u32string StringOverridePropertyObject::GetU32StringValue(codepoint_t cp) const {
    if (!mOverriddenSet.contains(cp)) return getPropertyObject(mBaseProperty)->GetU32StringValue(cp);
    std::string s = GetStringValue(cp);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(s);
#pragma GCC diagnostic pop
}

const std::string StringOverridePropertyObject::GetStringValue(codepoint_t cp) const {
    if (!mOverriddenSet.contains(cp)) return getPropertyObject(mBaseProperty)->GetStringValue(cp);
    // Otherwise, binary search through the explicit cps to find the index.
    // string index.
    unsigned lo = 0;
    unsigned hi = mExplicitCps.size()-1;
    while (lo < hi) {
        unsigned mid = (lo + hi)/2;
        if (cp <= mExplicitCps[mid]) hi = mid;
        else lo = mid + 1;
    }
    // Now lo == hi is the index of the desired string.
    unsigned offset = mStringOffsets[lo];
    unsigned lgth = mStringOffsets[lo+1] - offset - 1;
    return std::string(&mStringBuffer[offset], lgth);
}

const UnicodeSet StringOverridePropertyObject::GetPropertyIntersection(PropertyObject * p) {
    UnicodeSet intersection = getPropertyObject(mBaseProperty)->GetPropertyIntersection(p) - mOverriddenSet;
    if (isa<CodePointPropertyObject>(p)) {
        return p->GetPropertyIntersection(this);
    }
    if (isa<StringPropertyObject>(p) || isa<StringOverridePropertyObject>(p)) {
        for (unsigned i = 0; i < mExplicitCps.size(); i++) {
            if (GetStringValue(mExplicitCps[i]) == p->GetStringValue(mExplicitCps[i])) {
                intersection.insert(mExplicitCps[i]);
            }
        }
        return intersection;
    } else return UnicodeSet();
}

const UnicodeSet StringOverridePropertyObject::GetCodepointSetMatchingPattern(re::RE * re, GrepLinesFunctionType grep) {
    PropertyObject * baseObj = getPropertyObject(mBaseProperty);
    UCD::UnicodeSet s = baseObj->GetCodepointSetMatchingPattern(re, grep) - mOverriddenSet;
    const unsigned bufSize = mStringOffsets[mExplicitCps.size()];
    std::vector<uint64_t> matchedLines = grep(re, mStringBuffer, bufSize);
    for (const auto v : matchedLines) {
        s.insert(mExplicitCps[v]);
    }
    return s;
}

const std::string & ObsoletePropertyObject::GetPropertyValueGrepString() {
    report_fatal_error(Twine("Property ") + UCD::getPropertyFullName(the_property) + " is obsolete.");
}

const UnicodeSet ObsoletePropertyObject::GetCodepointSet(const std::string &) {
    report_fatal_error(Twine("Property ") + UCD::getPropertyFullName(the_property) + " is obsolete.");
}

}
