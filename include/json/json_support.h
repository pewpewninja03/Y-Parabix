/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 *
 */
#pragma once

#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/program_builder.h>

using PipelineBuilder = kernel::PipelineBuilder;
using StreamSet = kernel::StreamSet;

namespace json {

//
// Certain characters within JSON strings must be escaped, including
// double quotes, backslashes and ASCII controls 0x00-0X1F.
// The escape sequences for backslash and several common control characters
// generate two character escape sequences (insertion of one position required).
//
// \ -> \\, 0x08 (BS) -> \b, 0x09 (TAB) -> \t, 0x0A(LF) -> \n, 0x0A(FF) -> \f, 0x0D(CR) -> \r
//
// Other controls generate 6-character escape sequences of the form \u00xy
// where xy are the two hexadecimal digits of the control sequence value.
//
// Inputs:  
//    Basis - a set of basis bit streams 
//    stringmask - a mask marking data to be included in JSON strings
//    EscapedBasis - a newly created StreamSet of the same number of
//	      streams as basis.

void EscapeStringSpecials(PipelineBuilder & P, StreamSet * Basis, StreamSet * stringMask, StreamSet * EscapedBasis);

}
