/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
#include <kernel/pipeline/pipeline_builder.h>

namespace kernel {

//
//  Perform bitwise logical inversion of a mask stream.   Both mask and
//  inverted are streamsets of a single stream.

void Invert(PipelineBuilder & P, StreamSet * mask, StreamSet * inverted);

//
//  Combining kernels use a bitwise logic operation (Or, Xor, or And) to
//  combine streams from a source stream set with respective streams from
//  a combining stream set.   The number of streams in the combining stream
//  set may be fewer than those of the source stream set, in which the
//  higher indexed streams in the output are reproduced unmodified from
//  the source.
//
//  Combining kernels can be employed in two styles: Functional and InOut.
//  In the Functional style, the output streamset is produced as a distinct
//  set of streams.  In the InOut style, the input streamset is modified
//  to produce the output, without requiring reallocation.
//

enum class CombiningKind {Functional, InOut};

void OrCombine(PipelineBuilder & P,
               StreamSet * source, StreamSet * toCombine,
               StreamSet * combined, CombiningKind k = CombiningKind::Functional);

void XorCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k = CombiningKind::Functional);

void AndCombine(PipelineBuilder & P,
                StreamSet * source, StreamSet * toCombine,
                StreamSet * combined, CombiningKind k = CombiningKind::Functional);
}
