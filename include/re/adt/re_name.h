#pragma once

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_re.h>
#include <unicode/data/PropertyAliases.h>
#include <llvm/ADT/Twine.h>
#include <sstream>
namespace UCD {
    class UnicodeSet;
}

namespace re {

using length_t = std::string::size_type;
class Name : public RE {
public:
    enum NameFlags {
        None = 0,
        External = 1
    };
    static inline bool classof(const RE * re) {
        return re->getClassTypeId() == ClassTypeId::Name;
    }
    static inline bool classof(const void *) {
        return false;
    }
    std::string getNamespace() const;
    bool hasNamespace() const;
    std::string getName() const;
    std::string getFullName() const;
    RE * getDefinition() const;
    void setExternal() {
        mFlags |= NameFlags::External;
    }
    bool isExternal() const {
        return (mFlags & NameFlags::External) != 0;
    }
    bool operator<(const Name & other) const;
    bool operator<(const CC & other) const;
    bool operator>(const CC & other) const;
    void setDefinition(RE * definition);
    virtual ~Name() {}
protected:
    friend Name * makeName(const std::string & name, RE * cc);
    friend Name * makeZeroWidth(const std::string & name, RE * zerowidth);
    friend Name * makeName(CC * const cc);
    friend Name * makeName(const std::string &, RE *);
    friend Name * makeName(const std::string &, const std::string &, RE *);
    Name(const char * nameSpace, const length_t namespaceLength, const char * name, const length_t nameLength, RE * defn)
    : RE(ClassTypeId::Name)
    , mNamespaceLength(namespaceLength)
    , mNamespace(replicateString(nameSpace, namespaceLength))
    , mNameLength(nameLength)
    , mName(replicateString(name, nameLength))
    , mDefinition(defn)
    , mFlags(NameFlags::None) {

    }

private:
    const length_t      mNamespaceLength;
    const char * const  mNamespace;
    const length_t      mNameLength;
    const char * const  mName;
    RE *                mDefinition;
    size_t              mFlags;
};

inline std::string Name::getNamespace() const {
    return std::string(mNamespace, mNamespaceLength);
}

inline bool Name::hasNamespace() const {
    return (mNamespaceLength != 0);
}

inline std::string Name::getName() const {
    return std::string(mName, mNameLength);
}

inline std::string Name::getFullName() const {
    if (hasNamespace()) return getNamespace() + ":" + getName();
    else return getName();
}

inline RE * Name::getDefinition() const {
    return mDefinition;
}

inline void Name::setDefinition(RE * definition) {
    assert (definition != nullptr);
    assert (definition != this);
    mDefinition = definition;
}

inline bool Name::operator < (const Name & other) const {
    if (LLVM_LIKELY(mDefinition && other.mDefinition && llvm::isa<CC>(mDefinition) && llvm::isa<CC>(other.mDefinition))) {
        return *llvm::cast<CC>(mDefinition) < *llvm::cast<CC>(other.mDefinition);
    } else if (mNamespaceLength < other.mNamespaceLength) {
        return true;
    } else if (mNamespaceLength > other.mNamespaceLength) {
        return false;
    } else if (mNameLength < other.mNameLength) {
        return true;
    } else if (mNameLength > other.mNameLength) {
        return false;
    }
    const auto diff = std::memcmp(mNamespace, other.mNamespace, mNamespaceLength);
    if (diff < 0) {
        return true;
    } else if (diff > 0) {
        return false;
    }
    return (std::memcmp(mName, other.mName, mNameLength) < 0);
}

inline bool Name::operator < (const CC & other) const {
    if (mDefinition && llvm::isa<CC>(mDefinition)) {
        return *llvm::cast<CC>(mDefinition) < other;
    }
    return RE::ClassTypeId::Name < RE::ClassTypeId::CC;
}

inline bool Name::operator > (const CC & other) const {
    if (mDefinition && llvm::isa<CC>(mDefinition)) {
        return other < *llvm::cast<CC>(mDefinition);
    }
    return RE::ClassTypeId::CC < RE::ClassTypeId::Name;
}

inline Name * makeName(const std::string & name, RE * defn = nullptr) {
    return new Name(nullptr, 0, name.c_str(), name.length(), defn);
}

inline Name * makeName(const std::string & property, const std::string & value, RE * defn = nullptr) {
    return new Name(property.c_str(), property.length(), value.c_str(), value.length(), defn);
}

inline Name * makeName(CC * const cc) {
    const std::string name = cc->canonicalName();
    return new Name(nullptr, 0, name.c_str(), name.length(), cc);
}

inline Name * makeZeroWidth(const std::string & name, RE * zerowidth = NULL) {
    return new Name(nullptr, 0, name.c_str(), name.length(), zerowidth);
}

template <typename To, typename FromTy> bool defined(FromTy * e) {
    if (llvm::isa<To>(e)) return true;
    if (llvm::isa<re::Name>(e)) {
        re::RE * def = llvm::cast<re::Name>(e)->getDefinition();
        return def && defined<To, FromTy>(def);
    }
    return false;
}

template <typename To, typename FromTy> To * defCast(FromTy * e) {
    if (llvm::isa<To>(e)) return llvm::cast<To>(e);
    if (llvm::isa<re::Name>(e)) {
        re::RE * def = llvm::cast<re::Name>(e)->getDefinition();
        if (def) return defCast<To, FromTy>(def);
    }
    return nullptr;
}

[[ noreturn ]]
inline void UndefinedNameError(const re::Name * n) {
    llvm::report_fatal_error(llvm::Twine("Error: Undefined name in regular expression: \"") + n->getName() + "\".");
}

class Capture : public RE {
public:
    std::string getName() const {
        return std::string(mName, mNameLength);
    }
    RE * getCapturedRE() const {return mCapturedRE;}
    static Capture * Create(std::string name, RE * captured) {
        return new Capture(name.c_str(), name.length(), captured);
    }
    RE_SUBTYPE(Capture)
private:
    Capture(const char * name, const length_t nameLength, RE * captured): RE(ClassTypeId::Capture)
    , mNameLength(nameLength)
    , mName(replicateString(name, nameLength))
    , mCapturedRE(captured) {}
    const length_t mNameLength;
    const char * const mName;
    RE * mCapturedRE;
};

inline Capture * makeCapture(std::string name, RE * captured) {
    return Capture::Create(name, captured);
}

class Reference : public RE {
public:
    std::string getName() const  {
        return std::string(mName, mNameLength);
    }
    Capture * getCapture() const {return mCapture;}
    unsigned getInstance() const {return mInstance;}
    std::string getInstanceName() const  {
        return std::string(mName, mNameLength) + "." + std::to_string(mInstance);
    }
    UCD::property_t getReferencedProperty() const {return mReferencedProperty;}
    void setReferencedProperty(UCD::property_t p) {mReferencedProperty = p;}
    static Reference * Create(std::string name, Capture * capture, unsigned instance, UCD::property_t p = UCD::identity) {
        return new Reference(name.c_str(), name.length(), capture, instance, p);
    }
    RE_SUBTYPE(Reference)
private:
    Reference(const char * name, const length_t nameLength,
              Capture * capture, unsigned instance, UCD::property_t p = UCD::identity) :
    RE(ClassTypeId::Reference)
    , mNameLength(nameLength)
    , mName(replicateString(name, nameLength))
    , mCapture(capture)
    , mInstance(instance)
    , mReferencedProperty(p) {}
    const length_t mNameLength;
    const char * const mName;
    Capture * mCapture;
    unsigned mInstance;
    UCD::property_t mReferencedProperty;
};

inline Reference * makeReference(std::string name, Capture * capture, unsigned instance){
    return Reference::Create(name, capture, instance);
}

class PropertyExpression : public RE {
public:
    enum class Kind {Codepoint, Boundary};
    enum class Operator {Eq, NEq};

    PropertyExpression::Kind getKind() const { return mKind;}
    std::string getPropertyIdentifier() const { return mIdentifier;}
    PropertyExpression::Operator getOperator() const { return mOperator;}
    std::string getValueString() const { return mValue;}
    int getPropertyCode() const { return mPropertyCode;}
    RE * getResolvedRE() const { return mResolvedRE;}
    std::string getFullName();

    static PropertyExpression * Create(PropertyExpression::Kind k,
                                       std::string id,
                                       PropertyExpression::Operator op = PropertyExpression::Operator::Eq,
                                       std::string val = "") {
        return new PropertyExpression(k, id, op, val);
    }
    RE_SUBTYPE(PropertyExpression)

    void setPropertyIdentifier(std::string id) {mIdentifier = id;}
    void setOperator(PropertyExpression::Operator op) {mOperator = op;}
    void setValueString(std::string v) {mValue = v;}
    void setPropertyCode(int code) {mPropertyCode = code;}
    void setResolvedRE(RE * re) {mResolvedRE = re;}

private:
    PropertyExpression(Kind k, std::string id, PropertyExpression::Operator op, std::string v):
       RE(ClassTypeId::PropertyExpression),
       mKind(k), mIdentifier(id), mOperator(op), mValue(v), mPropertyCode(-1), mResolvedRE(nullptr) {}
    PropertyExpression::Kind mKind;
    std::string mIdentifier;
    PropertyExpression::Operator mOperator;
    std::string mValue;
    int mPropertyCode;
    RE * mResolvedRE;
};

inline std::string PropertyExpression::getFullName() {
    std::stringstream s;
    s << mIdentifier;
    if (mOperator == PropertyExpression::Operator::NEq) {
        s << "!";
    }
    if (mValue != "") {
        s << ":";
    }
    s << mValue;
    return s.str();
}

inline PropertyExpression * makePropertyExpression(PropertyExpression::Kind k, std::string ident, PropertyExpression::Operator op = PropertyExpression::Operator::Eq, std::string v = "") {
    return PropertyExpression::Create(k, ident, op, v);
}

inline PropertyExpression * makePropertyExpression(std::string ident, std::string v = "", PropertyExpression::Operator op = PropertyExpression::Operator::Eq) {
    return PropertyExpression::Create(PropertyExpression::Kind::Codepoint, ident, op, v);
}

inline PropertyExpression * makeBoundaryExpression(std::string ident, std::string v = "", PropertyExpression::Operator op = PropertyExpression::Operator::Eq) {
    return PropertyExpression::Create(PropertyExpression::Kind::Boundary, ident, op, v);
}

inline void UnresolvedPropertyExpressionError(const PropertyExpression * pe) {
    std::string prop = pe->getPropertyIdentifier();
    llvm::report_fatal_error(llvm::Twine("Error: Unresolved property expression in RE: ") + prop);
}


}
