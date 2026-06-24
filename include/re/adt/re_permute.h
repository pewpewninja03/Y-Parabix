/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <vector>
#include <llvm/Support/Casting.h>
#include <re/adt/re_re.h>
#include <util/slab_allocator.h>

namespace re {

class Permute : public RE, public std::vector<RE*, ProxyAllocator<RE *>> {
public:
    static inline bool classof(const RE * re) {
        return re->getClassTypeId() == ClassTypeId::Permute;
    }
    static inline bool classof(const void *) {
        return false;
    }
    template<typename iterator> static Permute * Create(const iterator begin, const iterator end) {return new Permute(begin, end);}
protected:
    template<typename iterator> friend RE * makePermute(iterator, iterator);
    Permute() : RE(ClassTypeId::Permute), std::vector<RE*, ProxyAllocator<RE *>>(mAllocator) {}
    template<typename iterator>
    Permute(const iterator begin, const iterator end)
    : RE(ClassTypeId::Permute), std::vector<RE*, ProxyAllocator<RE *>>(begin, end, mAllocator) { }
};

template<typename iterator>
RE * makePermute(const iterator begin, const iterator end) {
    return new Permute(begin, end);
}

inline RE * makePermute(std::initializer_list<RE *> list) {
    return makePermute(list.begin(), list.end());
}
}
