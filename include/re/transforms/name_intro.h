/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <map>
#include <re/transforms/re_transformer.h>
#include <unicode/utf/utf_encoder.h>

namespace cc {class Alphabet;}

namespace re {
class RE; class Name; class Alt; class Seq;

class NameIntroduction : public RE_Transformer {
public:
    NameIntroduction(std::string xfrmName) : RE_Transformer(xfrmName) {}
    std::map<std::string, RE *> mNameMap;
protected:
    Name * createName(std::string, RE * defn);
    void showProcessing() override;
};

class VariableLengthCCNamer final : public NameIntroduction {
public:
    VariableLengthCCNamer(unsigned UTF_bits = 8);
protected:
    RE * transformCC (CC * cc) override;
private:
    UTF_Encoder mEncoder;
};

class FixedSpanNamer final : public NameIntroduction {
public:
    FixedSpanNamer(const cc::Alphabet * a);
protected:
    RE * transform (RE * r) override;
    void processOneAlt(RE * r);
private:
    const cc::Alphabet * mAlphabet;
    std::string mLgthPrefix;
    std::map<int, std::vector<RE *>> mFixedLengthAlts;
    std::vector<RE *> mNewAlts;
};

class UniquePrefixNamer final : public NameIntroduction {
public:
    UniquePrefixNamer();
protected:
    RE * transform (RE * r) override;
};

class Repeated_CC_Seq_Namer final : public NameIntroduction {
public:
    Repeated_CC_Seq_Namer();
    std::map<std::string, std::pair<re::CC *, unsigned>> mInfoMap;
protected:
    RE * transform (RE * r) override;
private:
    std::string mPrefix;
    unsigned mGenSym;
    std::string genSym();
};

RE * canonicalizeExternals(RE * r, const std::vector<std::string> & external_names);

}
