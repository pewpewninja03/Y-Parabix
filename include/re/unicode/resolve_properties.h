#pragma once

#include <string>
#include <set>
#include <llvm/Support/Compiler.h>
#include <ucd/data/PropertyAliases.h>
#include <ucd/data/PropertyObjects.h>
#include <re/transforms/name_intro.h>
#include <re/transforms/re_transformer.h>

namespace re {
    class RE;
    class Name;
    class PropertyExpression;
}

namespace UCD {

[[noreturn]] void UnicodePropertyExpressionError(const std::string & errmsg);

/*  Link all property expression nodes to their property_enum code, and
    standardize the property name.   */
re::RE * linkProperties(re::RE * r);

/*  Convert all property expression to standardized form, using the
    full name of any enumerated properties. */
re::RE * standardizeProperties(re::RE * r);

/*  Resolve and store the equivalent regexp for all property expressions.
    Whenever property values are expressed by regular expression, use the
    passed in grep function to perform the resolution.  */
re::RE * resolveProperties(re::RE * r, GrepLinesFunctionType grep = nullptr);

/*  Link, standardize and resolve properties.  */
re::RE * linkAndResolve(re::RE * r, GrepLinesFunctionType grep = nullptr);

/*  Replace very simple codepoint properties (e.g. ASCII) with the equivalent CC. */
re::RE * inlineSimpleProperties(re::RE * r);

/*  Determine enumerated properties that require a basis set for implementation, such as an enumerated property in a boundary property expression or property reference.
    */
using PropertySet = std::set<UCD::property_t>;
PropertySet propertiesRequiringBasisSet(re::RE * r);

/*  Convert enumerated properties to CC expressions with a MultiplexedAlphabet */
re::RE * enumeratedPropertiesToCCs(PropertySet propertyCodes, re::RE * r);

/*  Create named externals for all property expressions.  */
class PropertyExternalizer : public re::NameIntroduction {
public:
    PropertyExternalizer();
protected:
    re::RE * transformPropertyExpression (re::PropertyExpression * exp) override;
};

re::RE * externalizeProperties(re::RE * r);

/*  Create named external for Any nodes.  */
re::RE * externalizeAnyNodes(re::RE * r);

}

