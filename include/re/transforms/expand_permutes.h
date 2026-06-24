#pragma once

namespace cc { class Alphabet;}
namespace re {

class RE;

RE * expandPermutes(RE * re, const cc::Alphabet * lengthAlpha = &cc::Unicode);

}
