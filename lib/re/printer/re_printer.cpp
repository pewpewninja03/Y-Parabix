/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/printer/re_printer.h>

//Regular Expressions
#include <re/adt/re_re.h>
#include <re/adt/re_alt.h>
#include <re/adt/re_any.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_name.h>
#include <re/adt/re_end.h>
#include <re/adt/re_rep.h>
#include <re/adt/re_seq.h>
#include <re/adt/re_start.h>
#include <re/adt/re_range.h>
#include <re/adt/re_diff.h>
#include <re/adt/re_intersect.h>
#include <re/adt/re_assertion.h>
#include <re/adt/re_group.h>
#include <re/adt/re_permute.h>
#include <re/alphabet/alphabet.h>
#include <ucd/data/PropertyAliases.h>
#include <llvm/Support/raw_ostream.h>

using namespace re;
using namespace llvm;

struct REStringBuilder {

    REStringBuilder() : tmp(), out(tmp) {}

    void buildString(const RE * re);

    std::string toString() {
        out.flush();
        tmp.shrink_to_fit();
        return tmp;
    }

 private:
    std::string tmp;
    raw_string_ostream out;
};

void REStringBuilder::buildString(const RE * re) {

    tmp.reserve(100);

    if (LLVM_UNLIKELY(re == nullptr)) {
        out << "<NULL>";
    } else if (const Alt* re_alt = dyn_cast<const Alt>(re)) {
        out << "(Alt[";
        bool comma = false;
        for (const RE * re : *re_alt) {
            if (comma) {
                out << ',';
            }
            buildString(re);
            comma = true;
        }
        out << "])";
    } else if (const CC* re_cc = dyn_cast<const CC>(re)) {
        out << "CC \"";
        out << re_cc->canonicalName();
        out << "\" ";
    } else if (const Name* re_name = dyn_cast<const Name>(re)) {
        if (re_name->isExternal()) {
            out << "Name \"";
            if (re_name->hasNamespace()) {
                out << re_name->getNamespace() << ':';
            }
            out << re_name->getName();
            out << '\"';
        } else {
            out << "Internal[";
            buildString(re_name->getDefinition());
            out << "]";
        }
    } else if (const Capture * c = dyn_cast<const Capture>(re)) {
        out << "Capture \"";
        out << c->getName();
        out << "\" ";
        out << "=(";
        buildString(c->getCapturedRE());
        out << ')';
    } else if (const Reference * r = dyn_cast<const Reference>(re)) {
        out << "Ref \"";
        out << r->getName();
        out << ".";
        out << r->getInstance();
        UCD::property_t p = r->getReferencedProperty();
        if (p != UCD::identity) {
            out << ":" << UCD::getPropertyFullName(p);
        }
        out << '\"';
    } else if (const Range* rg = dyn_cast<const Range>(re)) {
        out << "Range (";
        buildString(rg->getLo());
        out << " , ";
        buildString(rg->getHi());
        out << ')';
    } else if (const Assertion * a = dyn_cast<const Assertion>(re)) {
        if (a->getSense() == Assertion::Sense::Negative) {
            out << "Negative";
        }
        switch (a->getKind()) {
            case Assertion::Kind::LookAhead:
                out << "LookAhead";
                break;
            case Assertion::Kind::LookBehind:
                out << "LookBehind";
                break;
        }
        out << "Assertion(";
        buildString(a->getAsserted());
        out << ')';
    } else if (const Diff* diff = dyn_cast<const Diff>(re)) {
        out << "Diff (";
        buildString(diff->getLH());
        out << " , ";
        buildString(diff->getRH());
        out << ')';
    } else if (const Intersect* x = dyn_cast<const Intersect>(re)) {
        out << "Intersect (";
        buildString(x->getLH());
        out << " , ";
        buildString(x->getRH());
        out << ')';
    } else if (isa<const End>(re)) {
        out << "End";
    } else if (const Rep* re_rep = dyn_cast<const Rep>(re)) {
        out <<  "Rep(";
        buildString(re_rep->getRE());
        out << ',' << re_rep->getLB() << ',';
        if (re_rep->getUB() == Rep::UNBOUNDED_REP) {
            out << "Unbounded";
        } else {
            out << re_rep->getUB();
        }
        out << ')';
    } else if (const Seq* re_seq = dyn_cast<const Seq>(re)) {
        out << "(Seq[";
        bool comma = false;
        for (const RE * re : *re_seq) {
            if (comma) {
                out << ',';
            }
            buildString(re);
            comma = true;
        }
        out << "])";
    } else if (const Permute* p = dyn_cast<const Permute>(re)) {
        out << "(Permute[";
        bool comma = false;
        for (const RE * re : *p) {
            if (comma) {
                out << ',';
            }
            buildString(re);
            comma = true;
        }
        out << "])";
    } else if (const Group * g = dyn_cast<const Group>(re)) {
        out << "Group(";
        if (g->getMode() == Group::Mode::GraphemeMode) {
            out << ((g->getSense() == Group::Sense::On) ? "+g:" : "-g:");
        }
        else if (g->getMode() == Group::Mode::CaseInsensitiveMode) {
            out << ((g->getSense() == Group::Sense::On) ? "+i:" : "-i:");
        }
        else if (g->getMode() == Group::Mode::CompatibilityMode) {
            out << ((g->getSense() == Group::Sense::On) ? "+K:" : "-K:");
        }
        buildString(g->getRE());
        out << ')';
    } else if (const PropertyExpression * pe = dyn_cast<const PropertyExpression>(re)) {
        if (pe->getKind() == PropertyExpression::Kind::Boundary) {
            out << "Boundary(";
        } else {
            out << "Property(";
        }
        out << pe->getPropertyIdentifier();
        PropertyExpression::Operator op = pe->getOperator();
        std::string val_str = pe->getValueString();
        if ((val_str != "") || (op != PropertyExpression::Operator::Eq)) {
            if (op == PropertyExpression::Operator::Eq) {
                out << ':';
            } else if (op == PropertyExpression::Operator::NEq) {
                out << "!=";
            }
            out << pe->getValueString();
        }
        out << ')';
    } else if (isa<const Start>(re)) {
        out << "Start";
    } else if (const Any * a = dyn_cast<Any>(re)) {
        out << "Any(";
        out << a->getAlphabet()->getName();
        out << ")";
    } else {
        out << "???";
    }

}

const std::string Printer_RE::PrintRE(const RE * re) {
    REStringBuilder sb;
    sb.buildString(re);
    return sb.toString();
}
