/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <vector>                       // for vector<>::iterator
#include <boost/container/flat_map.hpp>
#include <pablo/builder.hpp>
#include <re/adt/adt.h>              // for Seq
#include <re/alphabet/alphabet.h>

namespace cc { class CC_Compiler; class Alphabet;}
namespace pablo { class PabloAST; }
namespace pablo { class PabloBlock; }
namespace pablo { class Var; }

namespace re {

class RE_Block_Compiler;

class RE_Compiler {
    friend class RE_Block_Compiler;
    public:

/*   The regular expression compiler works in terms of three fundamental bit stream
     concepts: barrier streams, index streams and marker streams.

     It is often desirable to consider that the input stream is divided
     into separate matching regions, such that any matched string must be
     wholly contained within one region.   For example, in grep-style
     matching, the matching regions are the individual lines of the line,
     and matches do not extend across more than one line.
     A barrier stream is used to separate the input into regions broken by
     positions marked by 1 bits.   Thus a matched substring will always
     correspond to a consecutive run of 0 bits in the barrier stream.

     Index streams mark positions corresponding to whole matching units.
     For example, if the matching units are UTF-8 sequences, then the index
     stream identifies the last byte of each UTF-8 sequence denoting a single
     Unicode codepoint.   If the matching units are UTF-16 sequences then the
     index stream will have one bits on every UTF-16 code unit that denotes
     a full Unicode character itself, as well as on the final code unit of
     a surrogate pair.

     Index streams may be defined with respect to a particular regular expression.
     For example, an index stream with respect to an regular expression involved
     in a bounded repetition may mark the last code unit of substrings matching
     a so-called characteristic subexpression of the bounded repetition.
     As a characteristic subexpression is a subexpression that must be matched
     once and only once for the repetition, the index stream may be used to
     count the number of repetitions.

     If the unit of matching is defined in terms of code units (e.g., UTF-8 bytes
     or UTF-16 double bytes) then the correspoinding index stream is the stream
     of all one bits.

     Marker streams represent the results of matching steps.
     Markers have an offset with respect to compiled regular expressions.
     Offset 0 means that each marker bit is placed at a position corresponding
     to the last code unit of a matched substring.

     Offsets are expressed with respect to an index stream.   An offset N > 0
     means that the marker is placed N positions further along the index stream
     than the last code unit of the matched substring.

     An offset of 1 with an index stream of all ones means that marker stream
     is placed on the code unit immediately after the last code unit matched.  The offset
     1 naturally arises whenever a regular expression can match the empty string.

     Non-zero offsets also arise in matching lookahead assertions.  For example,
     Unicode boundary assertions such as word boundaries or grapheme cluster
     boundaries typically involving lookahead at one or two Unicode characters.
     If the underlying streams are based on UTF-8 code units, then the index
     stream for such an offset is the the stream marking the final bytes of
     UTF-8 code unit sequences.  */

    class Marker {
    public:
        enum class Position : unsigned {AtEnd, AtNextCodeUnit, AtNextChar};
        Marker(pablo::PabloAST * strm, Position p = Position::AtEnd) : mPosition(p), mStream(strm) {}
        Marker & operator =(const Marker &) = default;
        Position position() {return mPosition;}
        pablo::PabloAST * stream() {return mStream;}
    private:
        Position mPosition;
        pablo::PabloAST * mStream;
    };

    //
    // The regular expression compiler may include one or more externally
    // defined Names.  Each name has a length range and may also have a
    // nonzero offset.   The length range indicates the range of string
    // lengths that are considered to be matched by the named RE object.
    // Normally, the min and max values of the range must be the same for
    // successful compilation, but but variable length ranges may be used
    // as the first element of a regular expression.  The offset is
    // normally zero, indicating that the marker is placed on the last
    // matched character of the named object.   An offset of 1 is used
    // when the length is zero or potentially zero (e.g., zero-width assertions).
    // When the from_first parameter is true, offsets are counted from
    // the first matched character; these are intended for externals
    // used for lookahead assertions.
    //
    class ExternalStream {
    public:
        ExternalStream(pablo::PabloAST * s, unsigned offset, std::pair<int, int> lgthRange, bool from_first = false) :
            mStream(s), mOffset(offset), mLengthRange(lgthRange), mFromFirst(from_first) {}
        ExternalStream & operator = (const ExternalStream &) = default;
        pablo::PabloAST * stream() {return mStream;}
        unsigned offset() {return mOffset;}
        std::pair<int, int> lengthRange() {return mLengthRange;}
        unsigned minLength() {return mLengthRange.first;}
        unsigned maxLength() {return mLengthRange.second;}
        bool fromFirst() {return mFromFirst;}
    private:
        pablo::PabloAST * mStream;
        unsigned mOffset;
        std::pair<int, int> mLengthRange;
        bool mFromFirst;
    };

    void addPrecompiled(std::string externalName, ExternalStream s);

    RE_Compiler(pablo::PabloBlock * scope,
                pablo::PabloAST * regionStartStream,
                pablo::PabloAST * regionFollowStream,
                const cc::Alphabet * codeUnitAlphabet = &cc::UTF8);

    void setIndexing(const cc::Alphabet * indexingAlphabet, pablo::PabloAST * idxStream);
    
    //
    // The CCs (character classes) within a regular expression are generally
    // expressed using a single alphabet.   But multiple alphabets may be
    // used under some circumstances.   For example, regular expressions for
    // Unicode may use both the Unicode alphabet for full Unicode characters
    // as well as the Byte alphabet for the individual code units of UTF-8.
    // In other cases, a multiplexed alphabet may be used for a certain
    // subexpression, for example, if the subexpression involves a local
    // language or a capture-backreference combination.
    //
    // Alphabets are added as needed using the addAlphabet method, giving both
    // the alphabet value and the set of parallel bit streams that comprise
    // a basis for the coded alphabet values.

    void addAlphabet(const cc::Alphabet * a, std::vector<pablo::PabloAST* > basis_set);

    void addAlphabet(const std::shared_ptr<cc::Alphabet> & a, std::vector<pablo::PabloAST* > basis_set) {
        addAlphabet(a.get(), basis_set);
    }
    
    Marker compileRE(RE * re);
    
    Marker compileRE(RE * re, Marker initialMarkers);

private:
    using ExternalNameMap = std::map<std::string, ExternalStream>;
    pablo::PabloBlock * const                       mEntryScope;
    const cc::Alphabet *                            mCodeUnitAlphabet;
    const cc::Alphabet *                            mIndexingAlphabet;
    pablo::PabloAST *                               mIndexStream;
    pablo::PabloAST *                               mRegionStart;
    pablo::PabloAST *                               mRegionFollow;
    pablo::PabloAST *                               mMatchable;
    std::vector<const cc::Alphabet *>               mAlphabets;
    std::vector<std::vector<pablo::PabloAST *>>     mBasisSets;
    std::vector<std::unique_ptr<cc::CC_Compiler>>   mAlphabetCompilers;
    pablo::PabloAST *                               mWhileTest;
    int                                             mStarDepth;
    ExternalNameMap                                 mExternalNameMap;
};

}

