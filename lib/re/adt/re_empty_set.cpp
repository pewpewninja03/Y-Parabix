/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/adt/adt.h>
#include <re/adt/re_empty_set.h>

using namespace llvm;

namespace re {

bool isEmptySet(RE * const re) {
    if (const Alt * a = dyn_cast<Alt>(re)) {
        return a->empty();
    } else if (const CC * cc = dyn_cast<CC>(re)) {
        return cc->empty();
    }
    return false;
}

RE * makeEmptySet() {
    return makeAlt();
}

}
