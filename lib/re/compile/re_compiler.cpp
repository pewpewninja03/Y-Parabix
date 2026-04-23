/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/compile/re_compiler.h>

#include <llvm/ADT/STLExtras.h>         // for std::make_unique
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#include <pablo/codegenstate.h>            // for PabloBuilder
#include <pablo/builder.hpp>            // for PabloBuilder
#include <pablo/pe_ones.h>              // for Ones
#include <pablo/pe_var.h>               // for Var
#include <pablo/pe_zeroes.h>            // for Zeroes
#include <re/adt/adt.h>
#include <re/cc/cc_compiler_target.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/cc/cc_compiler.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_local.h>
#include <re/analysis/cc_sequence_search.h>
#include <re/transforms/name_lookaheads.h>
#include <re/toolchain/toolchain.h>

namespace pablo { class PabloAST; }
namespace pablo { class Var; }
namespace pablo { class PabloKernel; }

using namespace pablo;
using namespace llvm;

namespace re {

using Marker = RE_Compiler::Marker;



class RE_Block_Compiler {
public:

    RE_Block_Compiler(RE_Compiler & main, PabloBuilder & pb);
    RE_Block_Compiler(RE_Block_Compiler * parent, PabloBuilder & pb);
    Marker process(RE * re, Marker marker);

private:

    Marker compile(RE * re);
    Marker compile(RE * re, Marker initialMarkers);

    Marker compileName(Name * name, Marker marker);
    Marker compileAny(Marker marker);
    Marker compileCC(CC * cc, Marker marker);
    Marker compileSeq(Seq * seq, Marker marker);
    Marker compileSeqTail(Seq::const_iterator current, const Seq::const_iterator end, int matchLenSoFar, Marker marker);
    Marker compileAlt(Alt * alt, Marker base);
    Marker compileAssertion(Assertion * a, Marker marker);
    Marker compileRep(int LB, int UB, RE * repeated, Marker marker);
    Marker compileDiff(Diff * diff, Marker marker);
    Marker compileIntersect(Intersect * x, Marker marker);
    pablo::PabloAST * consecutive_matches(pablo::PabloAST * repeated_j, int j, int repeat_count, const int match_length, pablo::PabloAST * indexStream);
    pablo::PabloAST * reachable(pablo::PabloAST * repeated, int length, int repeat_count, pablo::PabloAST * indexStream);
    std::pair<int, int> lengthRange(RE * regexp);
    Marker expandLowerBound(RE * repeated,  int lb, Marker marker, int ifGroupSize);
    Marker processUnboundedRep(RE * repeated, Marker marker);
    Marker expandUpperBound(RE * repeated, int ub, Marker marker, int ifGroupSize);

    Marker compileName(Name * name);
    Marker compileStart(Marker marker);
    Marker compileEnd(Marker marker);

    PabloAST * getCompiledCC(CC * cc);

    PabloAST * NextCharacter(Marker marker, PabloBuilder & pb);
    PabloAST * NextCodeUnitStream(Marker marker, PabloBuilder & pb);
    void AlignMarkers(Marker & m1, Marker & m2);

private:
    RE_Compiler &               mMain;
    RE_Block_Compiler *         mParent;
    PabloBuilder &              mPB;
    std::map<CC *, PabloAST *>  mLocallyCompiledCCs;
};

using Position = RE_Compiler::Marker::Position;

inline Marker RE_Block_Compiler::compile(RE * const re) {
    return process(re, Marker(mPB.createOnes(), Position::AtNextCodeUnit));
}

inline Marker RE_Block_Compiler::compile(RE * const re, Marker initialMarkers) {
    return process(re, initialMarkers);
}

PabloAST * ScanToIndex(PabloAST * cursor, PabloAST * indexStrm, PabloBuilder & pb) {
    return pb.createOr(pb.createAnd(cursor, indexStrm),
                       pb.createScanTo(pb.createAnd(pb.createNot(indexStrm), cursor), indexStrm),
                       "ScanToIndex");
}

std::pair<int, int> RE_Block_Compiler::lengthRange(RE * regexp) {
    if (Name * n = dyn_cast<Name>(regexp)) {
        const auto & nameString = n->getFullName();
        auto f = mMain.mExternalNameMap.find(nameString);
        if (f == mMain.mExternalNameMap.end()) {
            return std::make_pair<int, int>(1, 1);  // default for non-external names
        }
        return f->second.lengthRange();
    }
    auto alphabet = mMain.mIndexingAlphabet ? mMain.mIndexingAlphabet : mMain.mCodeUnitAlphabet;
    return getLengthRange(regexp, alphabet);
}

Marker RE_Block_Compiler::process(RE * const re, Marker marker) {
    if (isa<Name>(re)) {
        return compileName(cast<Name>(re), marker);
    } else if (Capture * c = dyn_cast<Capture>(re)) {
        return process(c->getCapturedRE(), marker);
    } else if (LLVM_UNLIKELY(isa<Reference>(re))) {
        llvm::report_fatal_error("back references not supported in icgrep.");
    } else if (isa<Seq>(re)) {
        return compileSeq(cast<Seq>(re), marker);
    } else if (isa<Alt>(re)) {
        return compileAlt(cast<Alt>(re), marker);
    } else if (Rep * rep = dyn_cast<Rep>(re)) {
        return compileRep(rep->getLB(), rep->getUB(), rep->getRE(), marker);
    } else if (isa<Assertion>(re)) {
        return compileAssertion(cast<Assertion>(re), marker);
    } else if (isa<Diff>(re)) {
        return compileDiff(cast<Diff>(re), marker);
    } else if (isa<Intersect>(re)) {
        return compileIntersect(cast<Intersect>(re), marker);
    } else if (isa<Start>(re)) {
        return compileStart(marker);
    } else if (isa<End>(re)) {
        return compileEnd(marker);
    } else if (isa<CC>(re)) {
        // CCs may be passed through the toolchain directly to the compiler.
        return compileCC(cast<CC>(re), marker);
    } else if (isa<Any>(re)) {
        // CCs may be passed through the toolchain directly to the compiler.
        return compileAny(marker);
    } else {
        UnsupportedRE("RE Compiler failed to process " + Printer_RE::PrintRE(re));
    }
}

Marker RE_Block_Compiler::compileAny(Marker marker) {
    PabloAST * nextPos = NextCharacter(marker, mPB);
    return Marker(mPB.createAnd(nextPos, mMain.mMatchable, "Any"));
}

Marker RE_Block_Compiler::compileCC(CC * const cc, Marker marker) {
    if (cc->empty()) {
        return Marker(mPB.createZeroes());
    }
    const cc::Alphabet * a = cc->getAlphabet();
    PabloAST * ccStrm = getCompiledCC(cc);
    if (ccStrm == nullptr) {
        unsigned i = 0;
        while (i < mMain.mAlphabets.size() && (a != mMain.mAlphabets[i])) i++;
        if (i < mMain.mAlphabets.size()) {
            //llvm::errs() << "Found alphabet: " << i << ", " << mMain.mAlphabets[i]->getName() << "\n";
            ccStrm = mPB.createAnd(mMain.mMatchable, mMain.mAlphabetCompilers[i]->compileCC(cc, mPB));
            mLocallyCompiledCCs.emplace(cc, ccStrm);
        } else {
            llvm::report_fatal_error(llvm::StringRef("Alphabet ") + a->getName() + " has no CC compiler, codeUnitAlphabet = " + mMain.mCodeUnitAlphabet->getName() + "\n in compiling RE: " + Printer_RE::PrintRE(cc) + "\n");
        }
    }
    PabloAST * nextPos = nullptr;
    if (a == mMain.mCodeUnitAlphabet) {
        nextPos = NextCodeUnitStream(marker, mPB);
    } else {
        nextPos = NextCharacter(marker, mPB);
    }
    return Marker(mPB.createAnd(nextPos, ccStrm, "cc_" + cc->canonicalName()));
}

Marker RE_Block_Compiler::compileName(Name * const name, Marker marker) {
    const auto & nameString = name->getFullName();
    auto f = mMain.mExternalNameMap.find(nameString);
    if (f == mMain.mExternalNameMap.end()) {
        // Names which are not in the external map are required to
        // be unit length Unicode REs, expressed as a code unit
        // sequences (e.g., sequences of UTF-8 CCs).
        RE * defn = name->getDefinition();
        if (!defn) {
            llvm::report_fatal_error(llvm::StringRef("RE compiler cannot find name as external or definition: ") + nameString);
        }
        if (mMain.mIndexingAlphabet) {
            // If we have an indexing alphabet, then the marker may be aligned
            // at the final byte of the code unit sequence.  We compile the
            // definition and align the marker based on the final position of
            // the compiled code unit sequence sequence.
            auto nameMarker = compile(defn, Marker(mPB.createOnes(), Position::AtNextChar));
            PabloAST * nextPos = marker.stream();
            if (marker.position() == Position::AtEnd) {
                nextPos = mPB.createIndexedAdvance(nextPos, mMain.mIndexStream, 1);
            }
            return Marker(mPB.createAnd(nextPos, nameMarker.stream(), nameString), nameMarker.position());
        } else {
            return compile(defn, marker);
        }
    }
    auto ext = f->second;
    unsigned amt = NamedLookAheadAmount(name, *mMain.mCodeUnitAlphabet);
    if (amt > 0) {
        // Named lookahead expression.
        if (marker.position() == Position::AtEnd) {
            return Marker(mPB.createAnd(marker.stream(), mPB.createLookahead(ext.stream(), amt)));
        } else {
            PabloAST * nextPos = NextCharacter(marker, mPB);
            return Marker(mPB.createAnd(nextPos, mPB.createLookahead(ext.stream(), amt - 1)), Position::AtNextChar);
        }
    }
    auto externalLength = ext.minLength();
    //llvm::errs() << "External: " << nameString << ", lgth " << externalLength << ", offset " << ext.offset() << "\n"; 
    if (ext.fromFirst() && (externalLength == ext.maxLength())) {
        // We have an external marker whose offset is from the
        // start of the external matched string; adjust to final position.
        auto adv = externalLength - 1 - ext.offset();
        if (marker.position() == Position::AtNextChar) {
            adv++;
        }
        PabloAST * extFinal = ext.stream();
        if (adv > 0) {
            extFinal = mPB.createIndexedAdvance(extFinal, mMain.mIndexStream, adv);
        }
        return Marker(mPB.createAnd(extFinal, marker.stream(), "la_" + nameString), marker.position());
    }
    if (marker.stream() == mMain.mIndexStream) {
        // We are at the beginning of a regular expression;
        // the external marker should become the new marker,
        // if it is at a matchable position.
        if (ext.offset() > 0) {
            return Marker(ext.stream(), Position::AtNextChar);
        }
        return Marker(mPB.createAnd(mMain.mMatchable, ext.stream()));
    }
    if (externalLength != ext.maxLength()) {
        llvm::report_fatal_error(llvm::StringRef("Variable length external not in initial position:  ")  + nameString);
    }
    PabloAST * nextPos = NextCharacter(marker, mPB);
    auto external_adv = externalLength + ext.offset() - 1;
    if (external_adv > 0) {
        nextPos = mPB.createIndexedAdvance(nextPos, mMain.mIndexStream, external_adv);
    }
    PabloAST * extStream = mPB.createAnd(nextPos, ext.stream(), "ext_" + nameString);
    if (ext.offset() == 0) {
        return Marker(mPB.createAnd(mMain.mMatchable, extStream));
    }
    return Marker(extStream, Position::AtNextChar);
}

Marker RE_Block_Compiler::compileSeq(Seq * const seq, Marker marker) {

    // if-hierarchies are not inserted within unbounded repetitions
    if (mMain.mStarDepth > 0) {
        for (RE * re : *seq) {
            marker = process(re, marker);
        }
        return marker;
    } else {
        return compileSeqTail(seq->cbegin(), seq->cend(), 0, marker);
    }
}

Marker RE_Block_Compiler::compileSeqTail(Seq::const_iterator current, const Seq::const_iterator end, const int matchLenSoFar, Marker marker) {
    if (current == end) {
        return marker;
    } else if (matchLenSoFar < IfInsertionGap) {
        RE * r = *current;
        marker = process(r, marker);
        return compileSeqTail(++current, end, matchLenSoFar + minMatchLength(r), marker);
    } else {
        Var * m = mPB.createVar("m", mPB.createZeroes());
        auto nested = mPB.createScope();
        RE_Block_Compiler subcompiler(mMain, nested);
        Marker m1 = subcompiler.compileSeqTail(current, end, 0, marker);
        nested.createAssign(m, m1.stream());
        mPB.createIf(marker.stream(), nested);
        return Marker(m, m1.position());
    }
}

Marker RE_Block_Compiler::compileAlt(Alt * const alt, const Marker base) {
    std::vector<PabloAST *>  accum(1, mPB.createZeroes());
    // The following may be useful to force a common Advance rather than separate
    // Advances in each alternative.
    for (RE * re : *alt) {
        Marker m = process(re, base);
        const unsigned o = static_cast<unsigned>(m.position());
        while (o >= accum.size()) {accum.push_back(mPB.createZeroes());}
        accum[o] = mPB.createOr(accum[o], m.stream(), "offset" + std::to_string(o) + "_alt");
    }
    if (accum.size() == 1) {
        // Only have accumulated AtEnd results.
        return Marker(accum[0]);
    } else {
        PabloAST * accumNext = mPB.createIndexedAdvance(accum[0], mMain.mIndexStream, 1);
        if (!isa<Zeroes>(accum[1])) {
            accumNext = mPB.createOr(accumNext, ScanToIndex(accum[1], mMain.mIndexStream, mPB));
        }
        if (accum.size() == 3) {
            accumNext = mPB.createOr(accumNext, accum[2]);
        }
        return Marker(accumNext, Position::AtNextChar);
    }
}

Marker RE_Block_Compiler::compileAssertion(Assertion * const a, Marker marker) {
    RE * asserted = a->getAsserted();
    if (a->getKind() == Assertion::Kind::LookBehind) {
        Marker lookback = compile(asserted);
        AlignMarkers(marker, lookback);
        PabloAST * lb = lookback.stream();
        if (a->getSense() == Assertion::Sense::Negative) {
            lb = mPB.createAnd(mPB.createNot(lb), mMain.mIndexStream);
        }
        return Marker(mPB.createAnd(marker.stream(), lb, "lookback"), marker.position());
    }
    // Lookahead assertions.
    auto lengths = lengthRange(asserted);
    // Zero-width assertions
    if (lengths.second == 0) {
        Marker lookahead = compile(asserted);
        AlignMarkers(marker, lookahead);
        PabloAST * la = lookahead.stream();
        if (a->getSense() == Assertion::Sense::Negative) {
            la = mPB.createNot(la);
        }
        return Marker(mPB.createAnd(marker.stream(), la, "lookahead"), marker.position());
    }
    Marker lookahead = compile(asserted);
    if (LLVM_LIKELY((lengths.second == 1) && (lookahead.position() == Position::AtEnd))) {
        Marker lookahead = compile(asserted);
        PabloAST * la = lookahead.stream();
        if (a->getSense() == Assertion::Sense::Negative) {
            la = mPB.createNot(la);
            if (mMain.mIndexStream) {
                la = mPB.createAnd(la, mMain.mIndexStream);
            }
        }
        PabloAST * following = NextCharacter(marker, mPB);
        return Marker(mPB.createAnd(following, la, "lookahead"), Position::AtNextChar);
    }
    // If the lookahead expression is an externally defined Name, we
    // may be able to use lookahead operations.
    if (Name * n = dyn_cast<Name>(asserted)) {
        const auto & nameString = n->getFullName();
        auto f = mMain.mExternalNameMap.find(nameString);
        if (f != mMain.mExternalNameMap.end()) {
            auto ext = f->second;
            auto extStream = ext.stream();
            if (ext.fromFirst()) {
                // We have an external marker whose offset is from the
                // start of the external matched string, enabling lookahead.
                if ((marker.position() == Position::AtNextChar) && (ext.offset() == 0)) {
                    // The current marker is already aligned with the
                    // external marker.
                    return Marker(mPB.createAnd(marker.stream(), extStream), marker.position());
                }
                auto ahead = ext.offset() + 1;
                if (marker.position() == Position::AtNextChar) {
                    ahead--;
                }
                PabloAST * extLookahead = mPB.createLookahead(extStream, ahead);
                if (a->getSense() == Assertion::Sense::Negative) {
                    extLookahead = mPB.createNot(extLookahead);
                }
                return Marker(mPB.createAnd(marker.stream(), extLookahead), marker.position());
            } else {
                PabloAST * following = NextCharacter(marker, mPB);
                auto extLength = ext.minLength();
                if (extLength == ext.maxLength()) {
                    auto ahead = extLength + ext.offset() - 1;
                    PabloAST * extLookahead = mPB.createLookahead(extStream, ahead);
                    if (a->getSense() == Assertion::Sense::Negative) {
                        extLookahead = mPB.createNot(extLookahead);
                    }
                    return Marker(mPB.createAnd(following, extLookahead), Position::AtNextChar);
                }
            }
        }
    }
    llvm::errs() << "lengths.second = " << lengths.second << "\n";
    UnsupportedRE("Unsupported lookahead assertion:" + Printer_RE::PrintRE(a));
}

inline bool alignedUnicodeLength(const RE * const lh, const RE * const rh) {
    const auto lhl = getLengthRange(lh, &cc::Unicode);
    const auto rhl = getLengthRange(rh, &cc::Unicode);
    //llvm::errs() << "lhl: (" << lhl.first << ", " << lhl.second << ")\n";
    //llvm::errs() << "rhl: (" << rhl.first << ", " << rhl.second << ")\n";
    return (lhl.first == lhl.second && lhl.first == rhl.first && lhl.second == rhl.second);
}

Marker RE_Block_Compiler::compileDiff(Diff * diff, Marker marker) {
    RE * const lh = diff->getLH();
    RE * const rh = diff->getRH();
    if (true) {
        Marker t1 = process(lh, marker);
        Marker t2 = process(rh, marker);
        AlignMarkers(t1, t2);
        return Marker(mPB.createAnd(t1.stream(), mPB.createNot(t2.stream()), "diff"), t1.position());
    }
    UnsupportedRE("Unsupported Diff operands: " + Printer_RE::PrintRE(diff));
}

Marker RE_Block_Compiler::compileIntersect(Intersect * const x, Marker marker) {
    RE * const lh = x->getLH();
    RE * const rh = x->getRH();
    if (alignedUnicodeLength(lh, rh)) {
        Marker t1 = process(lh, marker);
        Marker t2 = process(rh, marker);
        AlignMarkers(t1, t2);
        return Marker(mPB.createAnd(t1.stream(), t2.stream(), "intersect"), t1.position());
    }
    UnsupportedRE("Unsupported Intersect operands: " + Printer_RE::PrintRE(x));
}

bool CharacteristicSubexpressionAnalysis(RE * repeated, RE * &E1, RE * &C, RE * &E2) {
    if (isa<CC>(repeated)) {
        E1 = makeSeq();
        E2 = makeSeq();
        C = repeated;
        return true;
    }
    if (Seq * s = dyn_cast<Seq>(repeated)) {
        unsigned i = 0;
        while (i < s->size()) {
            unsigned j = i;
            std::vector<CC *> CC_seq;
            while (j < s->size()) {
                RE * item = (*s)[j];
                if (CC * cc = dyn_cast<CC>(item)) {
                    CC_seq.push_back(cc);
                    j++;
                } else if (const Name * name = dyn_cast<Name>(item)) {
                    RE * defn = name->getDefinition();
                    if (CC * cc = dyn_cast<CC>(defn)) {
                        CC_seq.push_back(cc);
                        j++;
                    } else break;
                } else break;
            }
            // If we found a nonempty CC_seq, determine if it is a characteristic
            // expression.
            if (j > i) {
                E1 = makeSeq(s->begin(), s->begin()+i);
                E2 = makeSeq(s->begin()+j, s->end());
                // Form E2 E1, where the original seq s is E1 CC_seq E2
                RE * E2_E1 = makeSeq({E2, E1});
                if (!CC_Sequence_Search(CC_seq, E2_E1)) {
                    // C is a characteristic subexpression
                    C = makeSeq(s->begin()+i, s->begin()+j);
                    return true;
                }
            }
            i = j+1;
        }
        return false;
    }
    return false;
}

Marker RE_Block_Compiler::compileRep(int lb, int ub, RE * repeated, Marker marker) {
    // Always handle cases with small lower bound and no upper bound by expansion.
    //llvm::errs() << "compileRep(" << Printer_RE::PrintRE(repeated) << ", " << lb << ", " << ub << ")\n";
    if ((lb <= 2) && (ub == Rep::UNBOUNDED_REP)) {
        Marker at_lb = expandLowerBound(repeated, lb, marker, IfInsertionGap);
        return processUnboundedRep(repeated, at_lb);
    }
    // Similarly handle cases with small upper bound by expansion.
    if ((ub != Rep::UNBOUNDED_REP) && (ub <= 2)) {
        Marker at_lb = expandLowerBound(repeated, lb, marker, IfInsertionGap);
        if (lb == ub) return at_lb;
        return expandUpperBound(repeated, ub - lb, at_lb, IfInsertionGap);
    }
    // Handle the case of lb == 0 specially.
    if (lb == 0) {
        Marker at_least_one = compileRep(1, ub, repeated, marker);
        AlignMarkers(marker, at_least_one);
        return Marker(mPB.createOr(marker.stream(), at_least_one.stream(), "none_or_1+"), at_least_one.position());
    }
    if (LLVM_LIKELY(!AlgorithmOptionIsSet(DisableLog2BoundedRepetition))) {
        // Check for a regular expression that satisfies on of the special conditions that
        // allow implementation using the log2 technique.
        auto lengths = getLengthRange(repeated, mMain.mCodeUnitAlphabet);
        //llvm::errs() << "getLengthRange(repeated, mMain.mCodeUnitAlphabet) = " << lengths.first << ", " << lengths.second << "\n";
        int rpt = lb;
        if ((lengths.first == 1) && (lengths.second == 1)) {
            PabloAST * cc = compile(repeated).stream();
            if (lb > 0) {
                PabloAST * cc_lb = consecutive_matches(cc, 1, rpt, lengths.first, nullptr);
                auto lb_lgth = lengths.first * rpt;
                if (marker.position() != Position::AtEnd) {
                    marker = Marker(NextCharacter(marker, mPB));
                    lb_lgth--;
                }
                PabloAST * marker_fwd = mPB.createAdvance(marker.stream(), lb_lgth, "marker_fwd");
                marker = Marker(mPB.createAnd(marker_fwd, cc_lb, "lowerbound"));
            }
            if (ub == Rep::UNBOUNDED_REP) {
                marker = processUnboundedRep(repeated, marker);
            } else if (lb < ub) {
                //marker = processBoundedRep(repeated, ub - lb, marker, IfInsertionGap);
                PabloAST * cursor = marker.stream();
                PabloAST * upperLimitMask = reachable(cursor, lengths.first, ub - lb, nullptr);
                PabloAST * masked = mPB.createAnd(cc, upperLimitMask, "masked");
                PabloAST * bounded = mPB.createAnd(mPB.createMatchStar(cursor, masked), upperLimitMask, "bounded");
                marker = Marker(bounded);
            }
            return marker;
        }
        if (mMain.mIndexingAlphabet) {
            auto lengths = getLengthRange(repeated, mMain.mIndexingAlphabet);
            //llvm::errs() << "getLengthRange(repeated, getIndexingAlphabet) = " << lengths.first << ", " << lengths.second << "\n";
            if ((lengths.first == 1) && (lengths.second == 1)) {
                PabloAST * cc = compile(repeated).stream();
                PabloAST * cursor = marker.stream();
                if (marker.position() != Position::AtEnd) {
                    cursor = mPB.createAnd(cc, NextCharacter(marker, mPB));
                    rpt -= 1;
                }
                PabloAST * cc_lb = consecutive_matches(cc, 1, rpt, 1, mMain.mIndexStream);
                PabloAST * marker_fwd = mPB.createIndexedAdvance(cursor, mMain.mIndexStream, rpt);
                PabloAST * at_lb = mPB.createAnd(marker_fwd, cc_lb, "lowerbound");
                if (ub == Rep::UNBOUNDED_REP) {
                    return processUnboundedRep(repeated, Marker(at_lb));
                }
                PabloAST * upperLimitMask = reachable(at_lb, 1, ub - lb, mMain.mIndexStream);
                PabloAST * masked = mPB.createAnd(cc, upperLimitMask, "masked");
                masked = mPB.createOr(masked, mPB.createNot(mMain.mIndexStream));
                PabloAST * bounded = mPB.createAnd(mPB.createMatchStar(at_lb, masked), upperLimitMask, "bounded");
                return Marker(bounded);
            }
        }
        RE * C;
        RE * E1;
        RE * E2;
        if (CharacteristicSubexpressionAnalysis(repeated, E1, C, E2)) {
            //llvm::errs() << "E1 = " << Printer_RE::PrintRE(E1) << "\n";
            //llvm::errs() << "C = " << Printer_RE::PrintRE(C) << "\n";
            //llvm::errs() << "E2 = " << Printer_RE::PrintRE(E2) << "\n";
            // Process an initial half iteration upto and including a match to C.
            Marker M1 = process(E1, marker);
            Marker half_mark = process(C, M1);
            assert(half_mark.position() == Position::AtEnd && "RE compiler error: characteristic subexpression with nonzero offset");
            //
            // Prepare the stream marking positions represent full repetitions.
            RE * C_E2_E1_C = makeSeq({C, E2, E1, C});
            PabloAST * iteration_marks = compile(C_E2_E1_C).stream();
            //
            // Prepare the matches to the characteristic subexpression C as the index stream.
            PabloAST * idx = compile(C).stream();
            //
            //  Restrict to positions with at least lb-1 iterations.
            if (lb > 1) {
                PabloAST * at_least_lb = consecutive_matches(iteration_marks, 1, lb - 1, 1, idx);
                PabloAST * marker_fwd = mPB.createIndexedAdvance(half_mark.stream(), idx, lb - 1);
                half_mark = Marker(mPB.createAnd(marker_fwd, at_least_lb, "lower_bound"));
            }
            if (ub == Rep::UNBOUNDED_REP) {
                // Finish the half-iteration for the lower bound, then continue with unbounded.
                Marker at_lb = process(E2, half_mark);
                return processUnboundedRep(repeated, at_lb);
            }
            if (ub == lb) {
                // No additional iterations, but finish the half-iteration.
                return process(E2, half_mark);
            }
            PabloAST * at_lb = half_mark.stream();
            PabloAST * upperLimitMask = reachable(at_lb, 1, ub - lb, idx);
            PabloAST * masked = mPB.createAnd(mPB.createOr(iteration_marks, at_lb, "reachable"), upperLimitMask, "masked");
            PabloAST * fill = mPB.createOr(masked, mPB.createNot(idx), "fill");
            PabloAST * bounded = mPB.createAnd(mPB.createMatchStar(at_lb, fill), masked, "bounded");
            return process(E2, Marker(bounded));
        }
    }
    if (lb > 0) {
        marker = expandLowerBound(repeated, lb, marker, IfInsertionGap);
    }
    if (ub == Rep::UNBOUNDED_REP) {
        marker = processUnboundedRep(repeated, marker);
    } else if (lb < ub) {
        marker = expandUpperBound(repeated, ub - lb, marker, IfInsertionGap);
    }
    return marker;
}

/*
   Given a stream |repeated_j| marking positions associated with |j| consecutive matches to an item
   of length |match_length| compute a stream marking |repeat_count| consecutive occurrences of such items.
*/

PabloAST * RE_Block_Compiler::consecutive_matches(PabloAST * const repeated_j, const int j, const int repeat_count, const int match_length, PabloAST * const indexStream) {
    if (j == repeat_count) {
        return repeated_j;
    }
    const int i = std::min(j, repeat_count - j);
    const int k = j + i;
    if (j > IfInsertionGap) {
        Var * repeated = mPB.createVar("repeated", mPB.createZeroes());
        auto nested = mPB.createScope();
        RE_Block_Compiler subcompiler(mMain, nested);
        PabloAST * adv_i = nested.createIndexedAdvance(repeated_j, indexStream, i * match_length);
        PabloAST * repeated_k = nested.createAnd(repeated_j, adv_i, "at" + std::to_string(k) + "of" + std::to_string(repeat_count));
        nested.createAssign(repeated, subcompiler.consecutive_matches(repeated_k, k, repeat_count, match_length, indexStream));
        mPB.createIf(repeated_j, nested);
        return repeated;
    } else {
        PabloAST * adv_i = mPB.createIndexedAdvance(repeated_j, indexStream, i * match_length);
        PabloAST * repeated_k = mPB.createAnd(repeated_j, adv_i, "at" + std::to_string(k) + "of" + std::to_string(repeat_count));
        return consecutive_matches(repeated_k, k, repeat_count, match_length, indexStream);
    }
}


inline PabloAST * RE_Block_Compiler::reachable(PabloAST * const repeated, const int length, const int repeat_count, PabloAST * const indexStream) {
    if (repeat_count == 0) {
        return repeated;
    }
    const int total_length = repeat_count * length;
    PabloAST * const v2 = mPB.createIndexedAdvance(repeated, indexStream, length);
    PabloAST * reachable = mPB.createOr(repeated, v2, "within1");
    int i = length;
    while ((i * 2) < total_length) {
        PabloAST * const extension = mPB.createIndexedAdvance(reachable, indexStream, i);
        i *= 2;
        reachable = mPB.createOr(reachable, extension, "within" + std::to_string(i));
    }
    if (LLVM_LIKELY(i < total_length)) {
        PabloAST * const extension = mPB.createIndexedAdvance(reachable, indexStream, total_length - i);
        reachable = mPB.createOr(reachable, extension, "within" + std::to_string(total_length));
    }
    return reachable;
}

Marker RE_Block_Compiler::expandLowerBound(RE * const repeated, const int lb, Marker marker, const int ifGroupSize) {
    //llvm::errs() << "expandLowerBound(" << Printer_RE::PrintRE(repeated) << ", " << lb << ")\n";
    if (LLVM_UNLIKELY(lb == 0)) {
        return marker;
    } else if (LLVM_UNLIKELY(lb == 1)) {
        return process(repeated, marker);
    }
    const auto group = ifGroupSize < lb ? ifGroupSize : lb;
    for (auto i = 0; i < group; i++) {
        marker = process(repeated, marker);
    }
    if (lb == group) {
        return marker;
    }
    Var * m = mPB.createVar("m", mPB.createZeroes());
    auto nested = mPB.createScope();
    RE_Block_Compiler subcompiler(mMain, nested);
    Marker m1 = subcompiler.expandLowerBound(repeated, lb - group, marker, ifGroupSize * 2);
    nested.createAssign(m, m1.stream());
    mPB.createIf(marker.stream(), nested);
    return Marker(m, m1.position());
}

Marker RE_Block_Compiler::expandUpperBound(RE * const repeated, const int ub, Marker marker, const int ifGroupSize) {
    //llvm::errs() << "expandUpperBound(" << Printer_RE::PrintRE(repeated) << ", " << ub << ")\n";
    if (LLVM_UNLIKELY(ub == 0)) {
        return marker;
    }
    const auto group = ifGroupSize < ub ? ifGroupSize : ub;
    for (auto i = 0; i < group; i++) {
        Marker a = process(repeated, marker);
        Marker m = marker;
        AlignMarkers(a, m);
        marker = Marker(mPB.createOr(a.stream(), m.stream(), "ub_combine"), a.position());
    }
    if (ub == group) {
        return marker;
    }
    Var * const m1a = mPB.createVar("m", mPB.createZeroes());
    auto nested = mPB.createScope();
    RE_Block_Compiler subcompiler(mMain, nested);
    Marker m1 = subcompiler.expandUpperBound(repeated, ub - group, marker, ifGroupSize * 2);
    nested.createAssign(m1a, m1.stream());
    mPB.createIf(marker.stream(), nested);
    return Marker(m1a, m1.position());
}

Marker RE_Block_Compiler::processUnboundedRep(RE * const repeated, Marker marker) {
    // always use PostPosition markers for unbounded repetition.
    PabloAST * base = NextCharacter(marker, mPB);
    if (LLVM_LIKELY(!AlgorithmOptionIsSet(DisableMatchStar))) {
        auto lengths = getLengthRange(repeated, mMain.mCodeUnitAlphabet);
        //llvm::errs() << "getLengthRange(repeated, mMain.mCodeUnitAlphabet) = " << lengths.first << ", " << lengths.second << "\n";
        if ((lengths.first == 1) && (lengths.second == 1)) {
            PabloAST * mask = compile(repeated).stream();
            mask = mPB.createOr(mask, mPB.createNot(mMain.mIndexStream));
            // The post position character may land on the initial byte of a multi-byte character. Combine them with the masked range.
            PabloAST * unbounded = mPB.createMatchStar(base, mask, "unbounded");
            return Marker(mPB.createAnd(unbounded, mMain.mIndexStream, "unbounded"), Position::AtNextChar);
        }
        if (mMain.mIndexingAlphabet) {
            auto lengths = getLengthRange(repeated, mMain.mIndexingAlphabet);
            //llvm::errs() << "getLengthRange(repeated, getIndexingAlphabet) = " << lengths.first << ", " << lengths.second << "\n";
            if ((lengths.first == 1) && (lengths.second == 1)) {
                PabloAST * mask = compile(repeated).stream();
                mask = mPB.createOr(mask, mPB.createNot(mMain.mIndexStream));
                PabloAST * unbounded = mPB.createMatchStar(base, mask);
                return Marker(mPB.createAnd(unbounded, mMain.mIndexStream, "unbounded"), Position::AtNextChar);
            }
        }
    }
    if (mMain.mStarDepth > 0){
        PabloBuilder * const outer = mPB.getParent();
        Var * starPending = outer->createVar("pending", outer->createZeroes());
        Var * starAccum = outer->createVar("accum", outer->createZeroes());
        mMain.mStarDepth++;
        PabloAST * m1 = mPB.createOr(base, starPending);
        PabloAST * m2 = mPB.createOr(base, starAccum);
        Marker result = process(repeated, Marker(m1, Position::AtNextChar));
        PabloAST * loopComputation = NextCharacter(result, mPB);
        mPB.createAssign(starPending, mPB.createAnd(loopComputation, mPB.createNot(m2)));
        mPB.createAssign(starAccum, mPB.createOr(loopComputation, m2));
        mMain.mWhileTest = mPB.createOr(mMain.mWhileTest, starPending);
        mMain.mStarDepth--;
        return Marker(mPB.createOr(base, starAccum, "unbounded"), Position::AtNextChar);
    } else {
        Var * whileTest = mPB.createVar("test", base);
        Var * whilePending = mPB.createVar("pending", base);
        Var * whileAccum = mPB.createVar("accum", base);
        mMain.mWhileTest = mPB.createZeroes();
        auto wb = mPB.createScope();
        RE_Block_Compiler subcompiler(mMain, wb);
        mMain.mStarDepth++;
        Marker result = subcompiler.process(repeated, Marker(whilePending, Position::AtNextChar));
        PabloAST * loopComputation = NextCharacter(result, wb);
        wb.createAssign(whilePending, wb.createAnd(loopComputation, wb.createNot(whileAccum)));
        wb.createAssign(whileAccum, wb.createOr(loopComputation, whileAccum));
        wb.createAssign(whileTest, wb.createOr(mMain.mWhileTest, whilePending));
        mPB.createWhile(whileTest, wb);
        mMain.mStarDepth--;
        return Marker(whileAccum, Position::AtNextChar);
    }
}

inline Marker RE_Block_Compiler::compileStart(Marker marker) {
    return Marker(mMain.mRegionStart, Position::AtNextCodeUnit);
}

inline Marker RE_Block_Compiler::compileEnd(Marker marker) {
    if (marker.position() == Position::AtEnd) {
        PabloAST * last = mPB.createLookahead(mMain.mRegionFollow, 1);
        return Marker(mPB.createAnd(marker.stream(), last, "EOT_match"));
    }
    PabloAST * nextPos = NextCharacter(marker, mPB);
    PabloAST * const EOT_match = mPB.createAnd(mMain.mRegionFollow, nextPos, "EOT_follow");
    return Marker(EOT_match, Position::AtNextChar);
}

PabloAST * RE_Block_Compiler::NextCharacter(Marker marker, PabloBuilder & pb) {
    if (marker.position() == Position::AtEnd) {
        return pb.createIndexedAdvance(marker.stream(), mMain.mIndexStream, 1);
    } else if  (marker.position() == Marker::Position::AtNextCodeUnit) {
        return ScanToIndex(marker.stream(), mMain.mIndexStream, pb);
    }
    return marker.stream();
}

inline PabloAST * RE_Block_Compiler::NextCodeUnitStream(Marker marker, PabloBuilder & pb) {
    if (marker.position() == Marker::Position::AtEnd) {
        return pb.createAdvance(marker.stream(), 1);
    }
    return marker.stream();
}

inline void RE_Block_Compiler::AlignMarkers(Marker & m1, Marker & m2) {
    if (m1.position() == m2.position()) return;
    if ((m1.position() == Position::AtEnd) && (m2.position() == Position::AtNextCodeUnit)) {
        m1 = Marker(NextCodeUnitStream(m1, mPB), Position::AtNextCodeUnit);
    } else if ((m1.position() == Position::AtEnd) && (m2.position() == Position::AtNextChar)) {
        m1 = Marker(NextCharacter(m1, mPB), Position::AtNextChar);
    } else if ((m1.position() == Position::AtNextCodeUnit) && (m2.position() == Position::AtNextChar)) {
        m1 = Marker(NextCharacter(m1, mPB), Position::AtNextChar);
    } else {
        AlignMarkers(m2, m1);
    }
}

PabloAST * RE_Block_Compiler::getCompiledCC(CC * cc) {
    if (mParent) {
        PabloAST * compiled = mParent->getCompiledCC(cc);
        if (compiled) return compiled;
    }
    auto f = mLocallyCompiledCCs.find(cc);
    if (f != mLocallyCompiledCCs.end()) {
        return f->second;
    }
    return nullptr;
}

RE_Block_Compiler::RE_Block_Compiler(RE_Compiler & main, PabloBuilder & pb)
: mMain(main)
, mParent(nullptr)
, mPB(pb) {
}

RE_Block_Compiler::RE_Block_Compiler(RE_Block_Compiler * parent, PabloBuilder & pb)
: mMain(parent->mMain)
, mParent(parent)
, mPB(pb) {
}
void RE_Compiler::addAlphabet(const cc::Alphabet * a, std::vector<pablo::PabloAST *> basis_set) {
    mAlphabets.push_back(a);
    mBasisSets.push_back(basis_set);
    bool useDirectCC = cast<VectorType>(basis_set[0]->getType())->getElementType()->getIntegerBitWidth() > 1;
    std::unique_ptr<cc::CC_Compiler> ccc;
    if (useDirectCC) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(basis_set[0]);
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(basis_set);
    }
    mAlphabetCompilers.push_back(std::move(ccc));
}

void RE_Compiler::setIndexing(const cc::Alphabet * indexingAlphabet, PabloAST * indexStream) {
    PabloBuilder pb(mEntryScope);
    mIndexingAlphabet = indexingAlphabet;
    mIndexStream = indexStream;
}
    
void RE_Compiler::addPrecompiled(std::string precompiledName, ExternalStream precompiled) {
    mExternalNameMap.emplace(precompiledName, precompiled);
}

Marker RE_Compiler::compileRE(RE * const re) {
    pablo::PabloBuilder mPB(mEntryScope);
    RE_Block_Compiler blockCompiler(*this, mPB);
    return blockCompiler.process(re, Marker(mIndexStream, Position::AtNextChar));
}

Marker RE_Compiler::compileRE(RE * const re, Marker initialMarkers) {
    pablo::PabloBuilder pb(mEntryScope);
    //  An important use case for an initial set of cursors to be passed in
    //  is that the initial cursors are computed from a prefix of an RE such
    //  that there is a high probability of all cursors remaining in a block
    //  are zeroed.   We therefore embed processing logic in an if-test,
    //  dependent on the initial cursors.
    Var * m = pb.createVar("m", pb.createZeroes());
    auto nested = pb.createScope();
    RE_Block_Compiler blockCompiler(*this, nested);
    Marker m1 = blockCompiler.process(re, initialMarkers);
    //Marker m1 = process(re, initialMarkers, nested);
    nested.createAssign(m, m1.stream());
    pb.createIf(initialMarkers.stream(), nested);
    return Marker(m, m1.position());
}

RE_Compiler::RE_Compiler(PabloBlock * scope,
                         PabloAST * regionStart,
                         PabloAST * regionFollow,
                         const cc::Alphabet * codeUnitAlphabet)
: mEntryScope(scope)
, mCodeUnitAlphabet(codeUnitAlphabet)
, mIndexingAlphabet(nullptr)
, mIndexStream(nullptr)
, mRegionStart(regionStart)
, mRegionFollow(regionFollow)
, mMatchable(nullptr)
, mWhileTest(nullptr)
, mStarDepth(0) {
    PabloBuilder pb(mEntryScope);
    mIndexStream = pb.createOnes();
    if (regionStart == nullptr) {
        PabloAST * advOnes = pb.createAdvance(pb.createOnes(), 1);
        mRegionStart = pb.createNot(advOnes);
        mRegionFollow = pb.createAtEOF(advOnes);
        mMatchable = pb.createOnes();
    } else {
        mMatchable = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {regionStart, regionFollow});
    }
}

} // end of namespace re
