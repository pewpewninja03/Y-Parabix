#pragma once

#include <string>
#include <set>
#include <vector>

namespace re {

    class RE; class Name;

    void gatherNames(RE * const re, std::set<Name *> & mNameSet);

    void gatherExternals(RE * const re,
                         std::set<std::string> & names,
                         std::set<std::string> & alphabets);

}
