/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <unicode/algo/normalization.h>

namespace re { class RE; }

namespace UCD {

/* Transforme an RE so that all string pieces and character classes
 are converted to NFD form (or NFKD form if the Compatible option
 is used.  The options may also including case folding. Examples:
 nfd_re = toNFD(r);
 nfkdi_re = toNFD(r, CaseFold | NFKD);
 */

re::RE * toNFD(re::RE * re, const DecompositionOptions opt = DecompositionOptions::NFD);

}
