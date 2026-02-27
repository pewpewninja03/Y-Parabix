/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/regex_passes.h>

#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/analysis/validation.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/name_lookaheads.h>
#include <re/analysis/re_analysis.h>
#include <re/transforms/assertion_transformations.h>
#include <re/transforms/re_contextual_simplification.h>
#include <re/transforms/re_minimizer.h>
#include <re/transforms/re_simplifier.h>
#include <re/transforms/re_star_normal.h>
#include <re/transforms/resolve_diffs.h>
#include <re/transforms/remove_nullable.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/casing.h>
#include <re/unicode/decomposition.h>
#include <re/unicode/equivalence.h>
#include <re/unicode/re_name_resolve.h>
#include <re/unicode/resolve_properties.h>
#include <re/toolchain/toolchain.h>
#include <toolchain/toolchain.h>

using namespace llvm;
using namespace re;

namespace re {

RE * resolveModesAndExternalSymbols(RE * r, bool globallyCaseInsensitive, GrepLinesFunctionType grep) {
    if (PrintOptionIsSet(ShowAllREs) || PrintOptionIsSet(ShowREs)) {
        errs() << "Parser:\n" << Printer_RE::PrintRE(r) << '\n';
    }
    r = resolveEscapeNames(r);
    r = resolveGraphemeMode(r, false /* not in grapheme mode at top level*/);
    r = UCD::linkAndResolve(r, grep);
    r = removeUnneededCaptures(r);
    r = UCD::inlineSimpleProperties(r);
    //r = resolveBoundaryProperties(r);
    validateNamesDefined(r);
    if (UnicodeLevel2IsSet() && validateAlphabet(&cc::Unicode, r)) {
        r = UCD::toNFD(r);
        r = UCD::addClusterMatches(r);
        r = UCD::addEquivalentCodepoints(r);
    }
    r = resolveCaseInsensitiveMode(r, globallyCaseInsensitive);
    //r = expandBoundaryAssertions(r);
    //r = simplifyAssertions(r);
    //r = lookaheadPromotion(r);
    return r;
}

RE * excludeUnicodeLineBreak(RE * r) {
    r = exclude_CC(r, re::makeCC(re::makeCC(0x0A, 0x0D), re::makeCC(re::makeCC(0x85), re::makeCC(0x2028, 0x2029))));
    if (PrintOptionIsSet(ShowAllREs)) {
        errs() << "excludeUnicodeLineBreak:\n" << Printer_RE::PrintRE(r) << '\n';
    }
    return r;
}

RE * remove_nullable_ends(RE * re) {
    RE * r = re;
    r = removeNullablePrefix(r);
    r = removeNullableSuffix(r);
    return r;
}

RE * regular_expression_passes(RE * re) {
    //Optimization passes to simplify the AST.
    RE * r = re;
    r = convertToStarNormalForm(r);
    if (codegen::OptLevel > CodeGenOptLevel::Less) {
        r = minimizeRE(r);
    } else {
        r = simplifyRE(r);
    }
    r = resolveDiffs(r);
    r = resolveAnchors(r, makeAlt());
    return r;
}

} // namespace re
