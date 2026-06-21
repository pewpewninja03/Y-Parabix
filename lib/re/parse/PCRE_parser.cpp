/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/parse/PCRE_parser.h>

#include <re/parse/parser_helper.h>

namespace re{

    // \< and \> removed
    const uint64_t setEscapeCharacters = bit3C('b') | bit3C('p') | bit3C('q') | bit3C('d') | bit3C('w') | bit3C('s') | bit3C('B') |
                                         bit3C('P') | bit3C('Q') | bit3C('D') | bit3C('W') | bit3C('S') | bit3C('N') | bit3C('X') | bit3C('A');

    bool PCRE_Parser::isSetEscapeChar(char c) {
        return c >= 0x3C && c <= 0x7B && ((setEscapeCharacters >> (c - 0x3C)) & 1) == 1;
    }

}
