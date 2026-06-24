#pragma once

#include "graphs.h"
#include <boost/iterator/iterator_facade.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <boost/icl/interval_set.hpp>
#include <random>
#include <kernel/core/kernel_compiler.h>
#include <boost/interprocess/mapped_region.hpp>

using boost::intrusive::detail::floor_log2;
using boost::intrusive::detail::ceil_log2;
using boost::intrusive::detail::ceil_pow2;
using boost::intrusive::detail::is_pow2;

inline unsigned getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

using IntervalSet = boost::icl::interval_set<size_t>;

using Interval = IntervalSet::interval_type;

namespace kernel {

using namespace boost;
using namespace llvm;

template <typename T, unsigned n = 16>
using Vec = SmallVector<T, n>;


struct CompilerAllocator : public SlabAllocator<> {
    template<typename Type = uint8_t>
    inline Type * allocate(const size_type n, const_pointer = nullptr) noexcept {
        static_assert(sizeof(Type) > 0, "Cannot allocate a zero-length type.");
        if (LLVM_UNLIKELY(n == 0)) {
            return nullptr;
        }
        assert ("A memory leak will occur whenever the SlabAllocator allocates 0 items" && n > 0);
        auto ptr = static_cast<Type *>(mAllocator.Allocate(n * sizeof(Type), sizeof(void*)));
        assert ("allocator returned a null pointer. Function was likely called before Allocator creation!" && ptr);
        return ptr;
    }

    template<typename Type = uint8_t>
    inline Type * aligned_allocate(const size_type n, const size_t align, const_pointer = nullptr) noexcept {
        static_assert(sizeof(Type) > 0, "Cannot allocate a zero-length type.");
        if (LLVM_UNLIKELY(n == 0)) {
            return nullptr;
        }
        auto ptr = static_cast<Type *>(mAllocator.Allocate(n * sizeof(Type), align));
        assert ("allocator returned a null pointer. Function was likely called before Allocator creation!" && ptr);
        return ptr;
    }
};

using pipeline_random_engine = std::default_random_engine;

// Many of the topological orderings of the graphs are simply
// a reverse traversal through the nodes of the graph.
// This class is just a small optimization for such orderings.
struct reverse_traversal {

    struct iterator : public boost::iterator_facade<
        iterator, const size_t, boost::forward_traversal_tag> {

    friend struct reverse_traversal;
    friend class boost::iterator_core_access;

        iterator() = default;

        explicit iterator(size_t n) : counter(n) { }

    private:

        void increment() {
            --counter;
        }

        bool equal(iterator const& other) const {
            return this->counter == other.counter;
        }

        const size_t & dereference() const { return counter; }

    private:
        size_t counter;
    };

    inline iterator begin() const {
        return iterator(N - 1);
    }

    inline iterator end() const {
        return iterator(-1);
    }

    inline unsigned size() const {
        return N;
    }

    reverse_traversal(const size_t n) : N(n) { }

private:

    const size_t N;

};

struct forward_traversal {

    struct iterator : public boost::iterator_facade<
        iterator, const size_t, boost::forward_traversal_tag> {

    friend struct reverse_traversal;
    friend class boost::iterator_core_access;

        iterator() = default;

        explicit iterator(size_t n) : counter(n) { }

    private:

        void increment() {
            ++counter;
        }

        bool equal(iterator const& other) const {
            return this->counter == other.counter;
        }

        const size_t & dereference() const { return counter; }

    private:
        size_t counter;
    };

    inline iterator begin() const {
        return iterator(0);
    }

    inline iterator end() const {
        return iterator(N);
    }

    inline unsigned size() const {
        return N;
    }

    forward_traversal(const size_t n) : N(n) { }

private:

    const size_t N;

};

template <typename T>
struct FixedVector {
    FixedVector(const size_t First, const size_t Last, CompilerAllocator & A)
    : mArray(A.allocate<T>(Last - First + 1U) - First)
    #ifndef NDEBUG
    , mFirst(First)
    , mLast(Last)
    #endif
    {
        reset(First, Last);
    }

    FixedVector(const size_t Size, CompilerAllocator & A)
    : mArray(A.allocate<T>(Size))
    #ifndef NDEBUG
    , mFirst(0)
    , mLast(Size - 1U)
    #endif
    {
        reset(0, Size - 1U);
    }

    inline T operator[](const size_t index) const {
        assert ("index exceeds allocated bounds!" && index >= mFirst && index <= mLast);
        return mArray[index];
    }

    inline T & operator[](const size_t index) {
        assert ("index exceeds allocated bounds!" && index >= mFirst && index <= mLast);
        return mArray[index];
    }

    inline void reset(const size_t First, const size_t Last) {
        assert ("invalid range!" && First <= Last);
        assert ("range exceeds allocated bounds!" && mFirst <= First && mLast >= Last);
        std::fill_n(mArray + First, (Last - First) + 1U, T{});
    }

private:
    T * const mArray;
    #ifndef NDEBUG
    const size_t mFirst;
    const size_t mLast;
    #endif
};


struct StreamSetInputPort {
    operator StreamSetPort () const noexcept {
        return StreamSetPort{Type, Number};
    }
    StreamSetInputPort() = default;
    StreamSetInputPort(const StreamSetPort port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    StreamSetInputPort(const StreamSetPort & port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    StreamSetInputPort(const RelationshipType & port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    explicit StreamSetInputPort(const StreamSetInputPort & port)
    : Number(port.Number) {

    }
    static constexpr PortType Type = PortType::Input;
    unsigned Number = 0;
};

struct StreamSetOutputPort {
    operator StreamSetPort() const noexcept {
        return StreamSetPort{Type, Number};
    }
    StreamSetOutputPort() = default;
    StreamSetOutputPort(const StreamSetPort port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    StreamSetOutputPort(const StreamSetPort & port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    StreamSetOutputPort(const RelationshipType & port)
    : Number(port.Number) {
        assert (port.Type == Type);
    }
    explicit StreamSetOutputPort(const StreamSetOutputPort & port)
    : Number(port.Number) {

    }
    static constexpr PortType Type = PortType::Output;
    unsigned Number = 0;
};

template <typename T>
struct InputPortVector {
    inline InputPortVector(const size_t n, CompilerAllocator & A)
    : mArray(0, n, A) {
    }
    inline T operator[](const StreamSetPort port) const {
        assert (port.Type == PortType::Input);
        return mArray[port.Number];
    }
    inline T & operator[](const StreamSetPort port) {
        assert (port.Type == PortType::Input);
        return mArray[port.Number];
    }
    inline T operator[](const StreamSetInputPort port) const {
        return mArray[port.Number];
    }
    inline T & operator[](const StreamSetInputPort port) {
        return mArray[port.Number];
    }
    inline void reset(const size_t n) {
        mArray.reset(0, n);
    }
private:
    FixedVector<T> mArray;
};

template <typename T>
struct OutputPortVector {
    inline OutputPortVector(const size_t n, CompilerAllocator & A)
    : mArray(0, n, A) {
    }
    inline T operator[](const StreamSetPort port) const {
        assert (port.Type == PortType::Output);
        return mArray[port.Number];
    }
    inline T & operator[](const StreamSetPort port) {
        assert (port.Type == PortType::Output);
        return mArray[port.Number];
    }
    inline T operator[](const StreamSetOutputPort port) const {
        return mArray[port.Number];
    }
    inline T & operator[](const StreamSetOutputPort port) {
        return mArray[port.Number];
    }
    inline void reset(const size_t n) {
        mArray.reset(0, n);
    }
private:
    FixedVector<T> mArray;
};

class PipelineCommonGraphFunctions {
public:

    PipelineCommonGraphFunctions(const RelationshipGraph & sg, const BufferGraph & bg)
    : mStreamGraphRef(sg)
    , mBufferGraphRef(bg) {

    }

    LLVM_READNONE RelationshipGraph::edge_descriptor getReferenceEdge(const size_t kernel, const StreamSetPort port) const;
    LLVM_READNONE unsigned getReferenceBufferVertex(const size_t kernel, const StreamSetPort port) const;
    LLVM_READNONE const StreamSetPort getReference(const size_t kernel, const StreamSetPort port) const;

    LLVM_READNONE unsigned getInputBufferVertex(const size_t kernel, const StreamSetPort inputPort) const;
    LLVM_READNONE StreamSetBuffer * getInputBuffer(const size_t kernel, const StreamSetPort inputPort) const;
    LLVM_READNONE const Binding & getInputBinding(const size_t kernel, const StreamSetPort inputPort) const;
    LLVM_READNONE const BufferPort & getInputPort(const size_t kernel, const StreamSetPort inputPort) const;
    LLVM_READNONE const BufferGraph::edge_descriptor getInput(const size_t kernel, const StreamSetPort outputPort) const;


    LLVM_READNONE unsigned getOutputBufferVertex(const size_t kernel, const StreamSetPort outputPort) const;
    LLVM_READNONE StreamSetBuffer * getOutputBuffer(const size_t kernel, const StreamSetPort outputPort) const;
    LLVM_READNONE const Binding & getOutputBinding(const size_t kernel, const StreamSetPort outputPort) const;
    LLVM_READNONE const BufferPort & getOutputPort(const size_t kernel, const StreamSetPort outputPort) const;
    LLVM_READNONE const BufferGraph::edge_descriptor getOutput(const size_t kernel, const StreamSetPort outputPort) const;


    LLVM_READNONE unsigned numOfStreamInputs(const unsigned kernel) const;
    LLVM_READNONE unsigned numOfStreamOutputs(const unsigned kernel) const;

    LLVM_READNONE const Binding & getBinding(const unsigned kernel, const StreamSetPort port) const;
    LLVM_READNONE const Kernel * getKernel(const size_t index) const;

    LLVM_READNONE bool mayHaveNonLinearIO(const size_t kernel) const;

    LLVM_READNONE bool isKernelStateFree(const size_t kernel) const;

    LLVM_READNONE bool isKernelFamilyCall(const size_t kernel) const;

private:

    const RelationshipGraph &   mStreamGraphRef;
    const BufferGraph &         mBufferGraphRef;

};

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

// NOTE: these graph functions not safe for general use since they are intended for inspection of *edge-immutable* graphs.

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor first_in_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (in_degree(u, G) >= 1);
    return *in_edges(u, G).first;
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor in_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (in_degree(u, G) == 1);
    return first_in_edge(u, G);
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::vertex_descriptor parent(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    return source(in_edge(u, G), G);
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor first_out_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (out_degree(u, G) >= 1);
    return *out_edges(u, G).first;
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::edge_descriptor out_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (out_degree(u, G) == 1);
    return first_out_edge(u, G);
}

template <typename Graph>
LLVM_READNONE
inline typename graph_traits<Graph>::vertex_descriptor child(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    return target(out_edge(u, G), G);
}

template <typename Graph>
LLVM_READNONE
inline bool is_parent(const typename graph_traits<Graph>::vertex_descriptor u,
                      const typename graph_traits<Graph>::vertex_descriptor v,
                      const Graph & G) {
    return parent(u, G) == v;
}

template <typename Graph>
LLVM_READNONE
inline bool has_child(const typename graph_traits<Graph>::vertex_descriptor u,
                      const typename graph_traits<Graph>::vertex_descriptor v,
                      const Graph & G) {
    for (const auto e : make_iterator_range(out_edges(u, G))) {
        if (target(e, G) == v) {
            return true;
        }
    }
    return false;
}

template <typename IntTy>
inline IntTy round_down_to(const IntTy x, const IntTy y) {
    assert(is_power_2(y));
    return x & -y;
}

template <typename IntTy>
inline IntTy round_up_to(const IntTy x, const IntTy y) {
    assert(is_power_2(y));
    return (x + y - 1) & -y;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReferenceBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCommonGraphFunctions::getReferenceBufferVertex(const size_t kernel, const StreamSetPort inputPort) const {
    return parent(source(getReferenceEdge(kernel, inputPort), mStreamGraphRef), mStreamGraphRef);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReference
 ** ------------------------------------------------------------------------------------------------------------- */
inline const StreamSetPort PipelineCommonGraphFunctions::getReference(const size_t kernel, const StreamSetPort inputPort) const {
    return mStreamGraphRef[getReferenceEdge(kernel, inputPort)];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCommonGraphFunctions::getInputBufferVertex(const size_t kernel, const StreamSetPort inputPort) const {
    assert (inputPort.Type == PortType::Input);
    return source(getInput(kernel, inputPort), mBufferGraphRef);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
inline StreamSetBuffer * PipelineCommonGraphFunctions::getInputBuffer(const size_t kernel, const StreamSetPort inputPort) const {
    return mBufferGraphRef[getInputBufferVertex(kernel, inputPort)].Buffer;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getNumOfStreamInputs
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCommonGraphFunctions::numOfStreamInputs(const unsigned kernel) const {
    return in_degree(kernel, mStreamGraphRef);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getNumOfStreamOutputs
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCommonGraphFunctions::numOfStreamOutputs(const unsigned kernel) const {
    return out_degree(kernel, mStreamGraphRef);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getKernel
 ** ------------------------------------------------------------------------------------------------------------- */
inline const Kernel * PipelineCommonGraphFunctions::getKernel(const size_t index) const {
    return mStreamGraphRef[index].Kernel;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isKernelFamily
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool PipelineCommonGraphFunctions::isKernelFamilyCall(const size_t index) const {
    return (mStreamGraphRef[index].Flags & RelationshipNodeFlag::IndirectFamily) != 0;
}

struct IllustratedStreamSet {
    const size_t StreamSet;
    const IllustratorTypeId IllustratorType;
    std::string Name;
    const std::array<char, 2> ReplacementCharacter;
    IllustratedStreamSet(const size_t streamSet, const IllustratorBinding & bind)
    : StreamSet(streamSet)
    , IllustratorType(bind.IllustratorType)
    , Name(bind.Name)
    , ReplacementCharacter(bind.ReplacementCharacter) {

    }
};

using IllustratedStreamSetMap = std::vector<IllustratedStreamSet>;

inline static IntervalSet parseCommaDelimitedList(const std::string & list)  {
    const auto len = list.size();
    IntervalSet parsedList;
    if (len > 0) {
        size_t pos = 0;
        for (;;) {
            auto p = pos;
            size_t rangeStart = 0;
            size_t num = 0;
            bool hasRangeStart = false;
            for (; p < len; ++p) {
                const auto ch = list.at(p);
                if (ch == '-') {
                    if (LLVM_LIKELY(!hasRangeStart)) {
                        rangeStart = num;
                        num = 0;
                        hasRangeStart = true;
                        continue;
                    }
                } else if ('0' <= ch && ch <= '9') {
                    num *= 10;
                    num += ch - '0';
                    continue;
                } else if (ch == ',') {
                    break;
                }
                report_fatal_error(StringRef{"Illegal comma delimited list given: "} + list);
            }

            rangeStart = hasRangeStart ? rangeStart : num;
            parsedList.add(Interval::closed(rangeStart, num));
            if (p == len) {
                break;
            }
            pos = p + 1;
        }
    }
    return parsedList;
}


}

