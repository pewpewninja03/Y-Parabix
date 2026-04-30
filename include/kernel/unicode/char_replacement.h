/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <vector>
#include <kernel/pipeline/pipeline_builder.h>
#include <unicode/utf/transchar.h>

kernel::StreamSet *  U21_CharToShortStringPipeline(kernel::PipelineBuilder & P,
    unicode::BitTranslationSets & insert_length_bixnum,
    std::vector<unicode::BitTranslationSets> & char_xlat_bitsets_by_position,
    kernel::StreamSet * U21_basis);

