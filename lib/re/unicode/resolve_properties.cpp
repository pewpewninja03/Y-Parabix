/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <re/unicode/resolve_properties.h>

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/adt/re_name.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/analysis/re_inspector.h>
#include <re/parse/parser.h>
#include <re/compile/re_compiler.h>
#include <re/unicode/boundaries.h>
#include <ucd/data/PropertyAliases.h>
#include <ucd/data/PropertyObjects.h>
#include <ucd/data/PropertyObjectTable.h>
#include <ucd/data/PropertyValueAliases.h>
#include <util/aligned_allocator.h>

using namespace UCD;
using namespace re;
using namespace llvm;

namespace UCD {
    
void UnicodePropertyExpressionError(const std::string & errmsg) {
    llvm::report_fatal_error(llvm::StringRef(errmsg));
}

struct PropertyResolver : public RE_Transformer {
    PropertyResolver(GrepLinesFunctionType grep) : RE_Transformer("PropertyResolver"), mGrep(grep) {}
    RE * transformPropertyExpression (PropertyExpression * exp) override;
    RE * resolveCC(std::string val, bool is_negated);
    RE * resolveBoundary(std::string val, bool is_negated);
    
    
private:
    GrepLinesFunctionType mGrep;
    int mPropCode;
    PropertyObject * mPropObj;
    
};

RE * PropertyResolver::resolveCC (std::string value, bool is_negated) {
    RE * resolved = nullptr;
    if ((value.length() > 0) && (value[0] == '/')) {
        if (mGrep == nullptr)
            llvm::report_fatal_error("Recursive property expression found, but no grep function supplied");
        re::RE * propValueRe = re::RE_Parser::parse(value.substr(1), re::DEFAULT_MODE, re::PCRE, false);
        return makeCC(mPropObj->GetCodepointSetMatchingPattern(propValueRe, mGrep), &cc::Unicode);
    }
    else if ((value.length() > 0) && (value[0] == '@')) {
        // resolve a @property@ or @identity@ expression.
        std::string otherProp = canonicalize_value_name(value.substr(1));
        auto propObj2 = getPropertyObject(getPropertyCode(otherProp));
        resolved = makeCC(mPropObj->GetPropertyIntersection(propObj2), &cc::Unicode);
    } else {
        resolved = makeCC(mPropObj->GetCodepointSet(value), &cc::Unicode);
    }
    if (is_negated) {
        resolved = makeDiff(makeAny(), resolved);
    }
    return resolved;
}

RE * PropertyResolver::resolveBoundary (std::string val, bool is_negated) {
    RE * resolved = nullptr;
    if (BoundaryPropertyObject * b = dyn_cast<BoundaryPropertyObject>(mPropObj)) {
        resolved = b->GetBoundaryExpression();
        if (resolved  == nullptr) {
            if (mPropCode == UCD::g) { // Grapheme cluster boundary
                resolved = generateGraphemeClusterBoundaryRule();
            } else if (mPropCode == UCD::w) { // Unicode word boundary
                resolved = generateWordBoundaryRule();
            }
            b->SetBoundaryExpression(resolved);
        }
        if (is_negated) {
            resolved = makeDiff(makeAny(), resolved);
        }
    } else if (isa<EnumeratedPropertyObject>(mPropObj) && (val == "")) {
        // Boundary between codepoints with any two different values for an
        // enumerated property.
        // TODO:  Pass in the operator, so that negated boundaries are generated in simplified form.
        resolved = EnumeratedPropertyBoundary(cast<EnumeratedPropertyObject>(mPropObj));
        if (is_negated) {
            resolved = makeDiff(makeAny(), resolved);
        }
    } else {
        std::string propName = getPropertyFullName(static_cast<property_t>(mPropCode));
        PropertyExpression * codepointProp = makePropertyExpression(propName, val);
        codepointProp->setPropertyCode(mPropCode);
        codepointProp->setResolvedRE(resolveCC(val, false));
        RE * a = makeLookAheadAssertion(codepointProp);
        RE * na = makeNegativeLookAheadAssertion(codepointProp);
        RE * b = makeLookBehindAssertion(codepointProp);
        RE * nb = makeNegativeLookBehindAssertion(codepointProp);
        if (is_negated) {
            resolved = makeAlt({makeSeq({b, a}), makeSeq({nb, na})});
        } else {
            resolved = makeAlt({makeSeq({b, na}), makeSeq({nb, a})});
        }
    }
    return resolved;
}

RE * PropertyResolver::transformPropertyExpression (PropertyExpression * exp) {
    mPropCode = exp->getPropertyCode();
    PropertyExpression::Operator op = exp->getOperator();
    std::string val = exp->getValueString();
    if (mPropCode < 0) {
        UnicodePropertyExpressionError("Property '" + exp->getPropertyIdentifier() + "' unlinked");
    }
    mPropObj = getPropertyObject(static_cast<UCD::property_t>(mPropCode));
    if (exp->getKind() == PropertyExpression::Kind::Boundary) {
        exp->setResolvedRE(resolveBoundary(val, op == PropertyExpression::Operator::NEq));
    } else {
        exp->setResolvedRE(resolveCC(val, op == PropertyExpression::Operator::NEq));
    }
    return exp;
}

RE * resolveProperties(RE * r, GrepLinesFunctionType grep) {
    return PropertyResolver(grep).transformRE(r);
}

    
struct PropertyLinker : public RE_Transformer {
    PropertyLinker() : RE_Transformer("PropertyLinker") {}
    RE * transformPropertyExpression (PropertyExpression * exp) override {
        // If already linked, simply return.
        if (exp->getPropertyCode() >= 0) return exp;
        std::string id = exp->getPropertyIdentifier();
        std::string canon = UCD::canonicalize_value_name(id);
        // In the case of a property expression without a value,
        // we may have a general category, script or some other special cases.
        if (exp->getValueString() == "") {
            const auto & gcObj = cast<EnumeratedPropertyObject>(getPropertyObject(gc));
            int valcode = gcObj->GetPropertyValueEnumCode(canon);
            if (valcode >= 0) {
                // Found a general category.
                exp->setValueString(gcObj->GetValueFullName(valcode));
                exp->setPropertyIdentifier(getPropertyFullName(gc));
                exp->setPropertyCode(gc);
                return exp;
            }
            const auto & scObj = cast<EnumeratedPropertyObject>(getPropertyObject(sc));
            valcode = scObj->GetPropertyValueEnumCode(canon);
            if (valcode >= 0) {
                // Found a script.
                exp->setValueString(scObj->GetValueFullName(valcode));
                exp->setPropertyIdentifier(getPropertyFullName(sc));
                exp->setPropertyCode(sc);
                return exp;
            }
            if (canon == "ascii") {  // block:ascii special case
                exp->setValueString("ascii");
                exp->setPropertyIdentifier(getPropertyFullName(blk));
                exp->setPropertyCode(blk);
                return exp;
            }
            if (canon == "assigned") {  // cn:n special case
                // general category != unassigned
                exp->setValueString("unassigned");
                exp->setPropertyIdentifier(getPropertyFullName(gc));
                exp->setOperator(PropertyExpression::Operator::NEq);
                exp->setPropertyCode(gc);
                return exp;
            }
            if (canon == "any") return makeAny();
        }
        auto propCode = UCD::getPropertyCode(canon);
        if (propCode != UCD::Undefined) {
            exp->setPropertyCode(propCode);
            return exp;
        }
        return exp;
    }
};

RE * linkProperties(RE * r) {
    return PropertyLinker().transformRE(r);
}

struct PropertyReferencePromotion : public RE_Transformer {
    PropertyReferencePromotion() : RE_Transformer("PropertyReferencePromotion") {}
    RE * transformPropertyExpression (PropertyExpression * exp) override {
        int prop_code = exp->getPropertyCode();
        if (prop_code < 0) return exp;  // No property code - leave unchanged.
        RE * defn = exp->getResolvedRE();
        if (defn == nullptr) return exp;
        if (Reference * ref = dyn_cast<Reference>(defn)) {
            ref -> setReferencedProperty(static_cast<UCD::property_t>(prop_code));
            return ref;
        }
        return exp;
    }
};

RE * promotePropertyReferences(RE * r) {
    return PropertyReferencePromotion().transformRE(r);
}

struct PropertyStandardization : public RE_Transformer {
    PropertyStandardization() : RE_Transformer("PropertyStandardization") {}
    RE * transformPropertyExpression (PropertyExpression * exp) override {
        int prop_code = exp->getPropertyCode();
        if (prop_code < 0) return exp;  // No property code - leave unchanged.
        PropertyExpression::Operator op = exp->getOperator();
        std::string val_str = exp->getValueString();
        std::string canon = UCD::canonicalize_value_name(val_str);
        auto * propObj = getPropertyObject(static_cast<UCD::property_t>(prop_code));
        if (auto * obj = dyn_cast<EnumeratedPropertyObject>(propObj)) {
            if (canon == "") return exp;  // No value to standardize
            int val_code = obj->GetPropertyValueEnumCode(canon);
            if (val_code < 0) return exp;
            exp->setValueString(obj->GetValueFullName(val_code));
            return exp;
        }
        if (auto * obj = dyn_cast<BinaryPropertyObject>(propObj)) {
            int val_code = obj->GetPropertyValueEnumCode(canon);
            // Standardize binary properties to positive form with an empty value string.
            if (val_code < 0) return exp;
            bool eq_F = (op == PropertyExpression::Operator::Eq) && (val_code == 0);
            bool ne_T = (op == PropertyExpression::Operator::NEq) && (val_code == 1);
            if (eq_F || ne_T) {
                // negated property.
                exp->setOperator(PropertyExpression::Operator::NEq);
                exp->setValueString("");
            } else { /*  if (eq_T || ne_F)  positive properties.  */
                exp->setOperator(PropertyExpression::Operator::Eq);
                exp->setValueString("");
            }
            return exp;
        }
        return exp;
    }
};

RE * standardizeProperties(RE * r) {
    return PropertyStandardization().transformRE(r);
}

struct SimplePropertyInliner : public RE_Transformer {
    const unsigned MAX_CC_COUNT_TO_INLINE = 1;
    const unsigned MAX_BOUNDARY_ALTS_TO_INLINE = 2;
    SimplePropertyInliner() : RE_Transformer("SimplePropertyInliner") {}
    RE * transformPropertyExpression (PropertyExpression * exp) override {
        if (exp->getKind() == PropertyExpression::Kind::Codepoint) {
            if (CC * cc = dyn_cast<CC>(exp->getResolvedRE())) {
                if (cc->count() <= MAX_CC_COUNT_TO_INLINE) {
                    return cc;
                }
            }
        }
        else {
            if (Alt * a = dyn_cast<Alt>(exp->getResolvedRE())) {
                if (a->size() <= MAX_BOUNDARY_ALTS_TO_INLINE) {
                    return a;
                }
            }
        }
        return exp;
    }
};

RE * inlineSimpleProperties(RE * r) {
    return SimplePropertyInliner().transformRE(r);
}

using PropertySet = std::set<UCD::property_t>;
struct EnumBasisRequiredCollector : public RE_Inspector {
    EnumBasisRequiredCollector(PropertySet & enums) : RE_Inspector(),
    mEnumSet(enums) {}

    void inspectPropertyExpression(PropertyExpression * pe) {
        auto id = static_cast<UCD::property_t>(pe->getPropertyCode());
        PropertyObject * propObj = getPropertyObject(id);
        if (isa<EnumeratedPropertyObject>(propObj)) {
            if (pe->getKind() == PropertyExpression::Kind::Boundary) {
                mEnumSet.insert(id);
            }
            RE * defn = pe->getResolvedRE();
            if (defn && isa<Reference>(defn)) {
                mEnumSet.insert(id);
            }
        }
    }
    PropertySet & mEnumSet;
};

PropertySet propertiesRequiringBasisSet(RE * r) {
    PropertySet ps;
    EnumBasisRequiredCollector(ps).inspectRE(r);
    return ps;
}

using PropertyAlphabetMap = std::map<UCD::property_t, cc::Alphabet *>;

struct EnumeratedPropertyMultiplexer : public RE_Transformer {
    EnumeratedPropertyMultiplexer(PropertyAlphabetMap & propertiesToMultiplex)
        : RE_Transformer("EnumeratedPropertyMultiplexer"), mPropertiesToMultiplex(propertiesToMultiplex) {}
    RE * transformPropertyExpression (PropertyExpression * exp) override {
        auto id = static_cast<UCD::property_t>(exp->getPropertyCode());
        auto f = mPropertiesToMultiplex.find(id);
        if (f == mPropertiesToMultiplex.end()) return exp;
        cc::Alphabet * enumAlphabet = f->second;
        PropertyExpression::Operator op = exp->getOperator();
        std::string val_str = exp->getValueString();
        PropertyObject * propObj = getPropertyObject(id);
        if (auto * obj = dyn_cast<EnumeratedPropertyObject>(propObj)) {
            std::string propName = getPropertyFullName(id);
            int val_code = obj->GetPropertyValueEnumCode(val_str);
            if (val_code < 0) return exp;  // TODO: deal with recursive regexp
            re::CC * enumCC = makeCC(enumAlphabet);
            if (op == PropertyExpression::Operator::Eq) {
                enumCC->insert(val_code);
            } else if (op == PropertyExpression::Operator::NEq) {
                for (int i = 0; i < obj->GetEnumCount(); i++) {
                    if (i != val_code) enumCC->insert(i);
                }
            }
            return enumCC;
        }
        return exp;
    }
private:
    PropertyAlphabetMap mPropertiesToMultiplex;
};

RE * enumeratedPropertiesToCCs(PropertySet propertyCodes, RE * r) {
    PropertyAlphabetMap propertyMap;
    for (auto c : propertyCodes) {
        PropertyObject * propObj = getPropertyObject(c);
        if (auto * obj = dyn_cast<EnumeratedPropertyObject>(propObj)) {
            std::string alphabetName = "UCD:" + getPropertyFullName(c);
            auto enumCount = obj->GetEnumCount();
            std::vector<CC *> enumCCs;
            for (int i = 0; i < enumCount; i++) {
                enumCCs.push_back(re::makeCC(obj->GetCodepointSet(i)));
            }
            propertyMap.emplace(c, cc::makeMultiplexedAlphabet(alphabetName, enumCCs));
        }
    }
    return EnumeratedPropertyMultiplexer(propertyMap).transformRE(r);
}

PropertyExternalizer::PropertyExternalizer() :
    NameIntroduction("PropertyExternalizer") {}

RE * PropertyExternalizer::transformPropertyExpression (PropertyExpression * exp) {
    PropertyExpression::Operator op = exp->getOperator();
    std::string id = exp->getPropertyIdentifier();
    std::string val_str = exp->getValueString();
    std::string op_str = val_str == "" ? "" : ":";
    if (op == PropertyExpression::Operator::NEq) op_str = "!" + op_str;
    //RE * defn = exp->getResolvedRE();
    std::string theName = id + op_str + val_str;
    if (exp->getKind() == PropertyExpression::Kind::Codepoint) {
        return createName(theName, exp);
    } else {
        theName = "\\b{" + theName + "}";
        return createName(theName, exp);
    }
}

RE * externalizeProperties(RE * r) {
    return PropertyExternalizer().transformRE(r);
}

struct AnyExternalizer : public RE_Transformer {
    AnyExternalizer() : RE_Transformer("AnyExternalizer") {}
    RE * transformAny(re::Any * a) override {
        Name * externName = makeName("u8index");
        externName->setDefinition(a);
        return externName;
    }
};

RE * externalizeAnyNodes(RE * r) {
    return AnyExternalizer().transformRE(r);
}

RE * linkAndResolve(RE * r, GrepLinesFunctionType grep) {
    RE * linked = linkProperties(r);
    linked = promotePropertyReferences(r);
    RE * std = standardizeProperties(linked);
    return resolveProperties(std, grep);
}


}
