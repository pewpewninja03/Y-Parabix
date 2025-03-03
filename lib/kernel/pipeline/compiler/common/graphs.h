#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#include <boost/graph/adjacency_list.hpp>
#pragma GCC diagnostic pop

#include <boost/graph/adjacency_matrix.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/container/flat_map.hpp>

#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/core/kernel_compiler.h>
#include <kernel/core/refwrapper.h>
#include <util/extended_boost_graph_containers.h>
#include <toolchain/toolchain.h>
#include <boost/range/adaptor/reversed.hpp>
#if (BOOST_VERSION < 106500)
#include <boost/math/common_factor_rt.hpp>
using namespace boost::math;
#else
#include <boost/integer/common_factor_rt.hpp>
#endif
#include <boost/dynamic_bitset.hpp>
#include <llvm/IR/ValueMap.h>
#include <llvm/ADT/BitVector.h>
#include <llvm/Support/raw_ostream.h>
#include <util/small_flat_set.hpp>


using namespace boost;
using namespace llvm;
using boost::container::flat_set;
using boost::container::flat_map;

namespace kernel {

#include <util/enum_flags.hpp>

using BindingRef = RefWrapper<Binding>;
using PortType = Kernel::PortType;
using StreamSetPort = Kernel::StreamSetPort;
using AttrId = Attribute::KindId;
using Rational = ProcessingRate::Rational;
using RateId = ProcessingRate::KindId;
using Scalars = PipelineKernel::Scalars;
using Kernels = PipelineKernel::Kernels;
using CallBinding = PipelineKernel::CallBinding;
using CallBindings = PipelineKernel::CallBindings;
using CallRef = RefWrapper<CallBinding>;
using LengthAssertion = PipelineKernel::LengthAssertion;
using LengthAssertions = PipelineKernel::LengthAssertions;
using ArgIterator = KernelCompiler::ArgIterator;
using InitArgTypes = KernelCompiler::InitArgTypes;

using StreamSetId = unsigned;

enum RelationshipNodeFlag {
    IndirectFamily = 1
    , ImplicitlyAdded = 2
    , IsSideEffecting = 4
};

struct RelationshipNode {

    enum RelationshipNodeType : unsigned {
        IsNil
        , IsKernel
        , IsStreamSet
        , IsScalar
        , IsCallee
        , IsBinding
    } Type;

    unsigned Flags;

    union {

        const kernel::Kernel * Kernel;
        kernel::Relationship * Relationship;
        CallRef                Callee;
        BindingRef             Binding;

    };

    bool operator == (const RelationshipNode & rn) const {
        return (Type == rn.Type) && (Kernel == rn.Kernel);
    }

    bool operator < (const RelationshipNode & rn) const {
        if(static_cast<unsigned>(Type) < static_cast<unsigned>(rn.Type)) {
            return true;
        }
        if (Type == rn.Type && Kernel < rn.Kernel) {
            return true;
        }
        return false;
    }

    static_assert(sizeof(Kernel) == sizeof(Relationship), "pointer size inequality?");
    static_assert(sizeof(Kernel) == sizeof(Callee), "pointer size inequality?");
    static_assert(sizeof(Kernel) == sizeof(Binding), "pointer size inequality?");

    explicit RelationshipNode() noexcept
        : Type(IsNil), Flags(0U), Kernel(nullptr) { }
    explicit RelationshipNode(std::nullptr_t) noexcept
        : Type(IsNil), Flags(0U), Kernel(nullptr)  { }
    explicit RelationshipNode(RelationshipNodeType typeId, not_null<const kernel::Kernel *> kernel, const unsigned flags = 0U) noexcept
        : Type(IsKernel), Flags(flags), Kernel(kernel) { assert (typeId == IsKernel); }
    explicit RelationshipNode(RelationshipNodeType typeId, not_null<kernel::Relationship *> relationship, const unsigned flags = 0U) noexcept
        : Type(typeId), Flags(flags), Relationship(relationship) { assert (typeId == IsStreamSet || typeId == IsScalar); }
    explicit RelationshipNode(RelationshipNodeType typeId, not_null<const CallBinding *> callee, const unsigned flags = 0U) noexcept
        : Type(IsCallee), Flags(flags), Callee(callee) { assert (typeId == IsCallee); }
    explicit RelationshipNode(RelationshipNodeType typeId, not_null<const kernel::Binding *> ref, const unsigned flags = 0U) noexcept
        : Type(IsBinding), Flags(flags), Binding(ref) { assert (typeId == IsBinding); }
    explicit RelationshipNode(const RelationshipNode & rn) noexcept
        : Type(rn.Type), Flags(rn.Flags), Kernel(rn.Kernel) { }

    RelationshipNode & operator = (const RelationshipNode & other) {
        Type = other.Type;
        Kernel = other.Kernel;
        Flags = other.Flags;
        return *this;
    }

};

enum class ReasonType : unsigned {
    None
    // -----------------------------
    , Explicit
    // -----------------------------
    , ImplicitRegionSelector
    , ImplicitPopCount
    , ImplicitTruncatedSource
    // -----------------------------
    , Reference
    // -----------------------------
    , OrderingConstraint
};

struct RelationshipType : public StreamSetPort {
    ReasonType Reason;

    explicit RelationshipType(PortType type, unsigned number, ReasonType reason = ReasonType::Explicit)
    : StreamSetPort(type, number), Reason(reason) { }

    explicit RelationshipType(StreamSetPort port, ReasonType reason = ReasonType::Explicit)
    : StreamSetPort(port), Reason(reason) { }

    explicit RelationshipType(ReasonType reason = ReasonType::None)
    : StreamSetPort(), Reason(reason) { }

    RelationshipType & operator = (const RelationshipType &) = default;

    bool operator == (const RelationshipType & rn) const {
        return (Number == rn.Number) && (Reason == rn.Reason) && (Type == rn.Type);
    }

    bool operator < (const RelationshipType & rn) const {
        if (LLVM_LIKELY(Reason == rn.Reason)) {
            if (LLVM_LIKELY(Type == rn.Type)) {
                return Number < rn.Number;
            }
            return static_cast<unsigned>(Type) < static_cast<unsigned>(rn.Type);
        }
        return static_cast<unsigned>(Reason) < static_cast<unsigned>(rn.Reason);
    }

};

using RelationshipGraph = adjacency_list<vecS, vecS, bidirectionalS, RelationshipNode, RelationshipType, no_property>;

struct ProgramGraph : public RelationshipGraph {
    using Vertex = RelationshipGraph::vertex_descriptor;

    template <typename T>
    inline Vertex add(RelationshipNode::RelationshipNodeType typeId, T key, const unsigned flags = 0) {
        return __add(RelationshipNode{typeId, key, flags});
    }

    template <typename T>
    inline Vertex set(RelationshipNode::RelationshipNodeType typeId, T key, Vertex v) {
        return __set(RelationshipNode{typeId, key}, v);
    }

    template <typename T>
    inline Vertex find(RelationshipNode::RelationshipNodeType typeId, T key) {
        return __find(RelationshipNode{typeId, key});
    }

    template <typename T>
    inline Vertex addOrFind(RelationshipNode::RelationshipNodeType typeId, T key, const bool permitAdd = true) {
        return __addOrFind(RelationshipNode{typeId, key}, permitAdd);
    }

    RelationshipGraph & Graph() {
        return static_cast<RelationshipGraph &>(*this);
    }

    const RelationshipGraph & Graph() const {
        return static_cast<const RelationshipGraph &>(*this);
    }

    ProgramGraph() = default;

    ProgramGraph(size_t n) noexcept : RelationshipGraph(n) { }

    ProgramGraph(ProgramGraph && G) noexcept
    : RelationshipGraph(std::move(G.Graph()))
    , mMap(std::move(G.mMap)) {

    }

    ProgramGraph & operator=(ProgramGraph && G) = default;

private:

    BOOST_NOINLINE Vertex __add(const RelationshipNode & key) {
        assert ("adding an existing relationship key!" && mMap.find(key.Kernel) == mMap.end());
        const auto v = add_vertex(key, *this);
        mMap.emplace(key.Kernel, v);
        assert ((*this)[v] == key);
        assert (__find(key) == v);
        return v;
    }

    BOOST_NOINLINE Vertex __set(const RelationshipNode & key, const Vertex v) {
        auto f = mMap.find(key.Kernel);
        if (LLVM_UNLIKELY(f == mMap.end())) {
            mMap.emplace(key.Kernel, v);
        } else {
            f->second = v;
        }
        assert ((*this)[v] == key);
        assert (__find(key) == v);
        return v;
    }

    BOOST_NOINLINE Vertex __find(const RelationshipNode & key) const {
        const auto f = mMap.find(key.Kernel);
        if (LLVM_LIKELY(f != mMap.end())) {
            const auto v = f->second;
            assert ((*this)[v] == key);
            return v;
        }
        llvm_unreachable("could not find node in relationship graph");
    }

    BOOST_NOINLINE Vertex __addOrFind(const RelationshipNode & key, const bool permitAdd) {
        const auto f = mMap.find(key.Kernel);
        if (f != mMap.end()) {
            const auto v = f->second;
            assert ((*this)[v] == key);
            return v;
        }
        if (LLVM_LIKELY(permitAdd)) {
            return __add(key);
        }
        llvm_unreachable("could not find node in relationship graph");
    }

    flat_map<const void *, Vertex> mMap;
};

enum BufferType : unsigned {
    External = 1
    , Unowned = 2
    , Shared = 4
    , Returned = 8
    , Truncated = 16
    , CrossThreaded = 32
    , InOutRedirect = 64
    // ------------------
    , HasIllustratedStreamset = 512
    , StartsNestedSynchronizationRegion = 1024
};

ENABLE_ENUM_FLAGS(BufferType)

enum BufferLocality {
    ThreadLocal
    , PartitionLocal
    , GloballyShared
    , ConstantShared
    , ZeroElementsOrWidth
};

enum KernelFlags {
    PermitSegmentSizeSlidingWindowing = 1
};

struct BufferNode {
    StreamSetBuffer * Buffer = nullptr;
    unsigned Type = 0;
    bool IsLinear = false;

    BufferLocality Locality = BufferLocality::ThreadLocal;

    unsigned LookBehind = 0;
    unsigned MaxAdd = 0;

    unsigned BufferStart = 0;
    unsigned BufferEnd = 0;

    bool RequiresUnderflow = false;

    unsigned RequiredCapacity = 0;
    unsigned PartialSumSpanLength = 0;

    unsigned OutputItemCountId = 0;
    unsigned LockId = 0;


    bool permitSlidingWindow() const {
        return (Type & KernelFlags::PermitSegmentSizeSlidingWindowing) != 0;
    }

    bool isOwned() const {
        return (Type & BufferType::Unowned) == 0;
    }

    bool isUnowned() const {
        return (Type & BufferType::Unowned) != 0;
    }

    bool isInternal() const {
        return (Type & BufferType::External) == 0;
    }

    bool isExternal() const {
        return (Type & BufferType::External) != 0;
    }

    bool isShared() const {
        return (Type & BufferType::Shared) != 0;
    }

    bool isReturned() const {
        return (Type & BufferType::Returned) != 0;
    }

    bool isTruncated() const {
        return (Type & BufferType::Truncated) != 0;
    }

    bool isCrossThreaded() const {
        return (Type & BufferType::CrossThreaded) != 0;
    }

    bool isInOutRedirect() const {
        return (Type & BufferType::InOutRedirect) != 0;
    }


    bool startsNestedSynchronizationRegion() const {
        return (Type & BufferType::StartsNestedSynchronizationRegion) != 0;
    }

    bool isThreadLocal() const {
        return (Locality == BufferLocality::ThreadLocal);
    }

    bool isNonThreadLocal() const {
        return (Locality != BufferLocality::ThreadLocal);
    }

    bool isConstant() const {
        return (Locality == BufferLocality::ConstantShared);
    }

    bool hasZeroElementsOrWidth() const {
        return (Locality == BufferLocality::ZeroElementsOrWidth);
    }

    bool isDeallocatable() const {
        return !(isUnowned() || isThreadLocal() ||isConstant() || isTruncated() || isInOutRedirect() || isReturned());
    }
};

enum BufferPortType : unsigned {
    IsPrincipal = 1,
    IsFixed = 2,
    IsZeroExtended = 4,
    IsDeferred = 8,
    IsRelative = 16,
    IsShared = 32,
    IsManaged = 64,
    CanModifySegmentLength = 128,
    IsCrossThreaded = 256,
    Illustrated = 512
};

struct BufferPort {

    RelationshipType Port;
    BindingRef Binding;
    Rational Minimum;
    Rational Maximum;
    unsigned Flags = 0;


    unsigned SymbolicRateId = 0U;

    // binding attributes
    unsigned Add = 0;
    unsigned Truncate = 0;
    unsigned Delay = 0;
    unsigned LookAhead = 0;
    unsigned LookBehind = 0;

    //bool mCanModifySegmentLength = false;

    int TransitiveAdd = 0;

    bool isPrincipal() const {
        return (Flags & BufferPortType::IsPrincipal) != 0;
    }

    bool isCrossThreaded() const {
        return (Flags & BufferPortType::IsCrossThreaded) != 0;
    }

    bool isFixed() const {
        return (Flags & BufferPortType::IsFixed) != 0;
    }

    bool isZeroExtended() const {
        return (Flags & BufferPortType::IsZeroExtended) != 0;
    }

    bool isDeferred() const {
        return (Flags & BufferPortType::IsDeferred) != 0;
    }

    bool isRelative() const {
        return (Flags & BufferPortType::IsRelative) != 0;
    }

    bool isShared() const {
        return (Flags & BufferPortType::IsShared) != 0;
    }

    bool isManaged() const {
        return (Flags & BufferPortType::IsManaged) != 0;
    }

    bool canModifySegmentLength() const {
        return (Flags & BufferPortType::CanModifySegmentLength) != 0;
    }

    bool isIllustrated() const {
        return (Flags & BufferPortType::Illustrated) != 0;
    }

    bool operator < (const BufferPort & rn) const {
        if (LLVM_LIKELY(Port.Type == rn.Port.Type)) {
            return Port.Number < rn.Port.Number;
        }
        return static_cast<unsigned>(Port.Type) < static_cast<unsigned>(rn.Port.Type);
    }

    const ProcessingRate & getRate() const {
        return Binding.get().getRate();
    }

    const kernel::AttributeSet & getAttributes() const {
        return Binding.get().getAttributes();
    }

    BufferPort() = default;

    BufferPort(RelationshipType port, const struct Binding & binding,
               Rational minRate, Rational maxRate)
    : Port(port), Binding(binding)
    , Minimum(minRate), Maximum(maxRate) {

    }

};

using BufferGraph = adjacency_list<vecS, vecS, bidirectionalS, BufferNode, BufferPort>;

struct ConsumerNode {
//    mutable Value * Consumed = nullptr;
//    mutable PHINode * PhiNode = nullptr;
};

struct ConsumerEdge {

    enum ConsumerTypeFlags : unsigned {
        None = 0
        , UpdateConsumedCount = 1
        , WriteConsumedCount = 2
        , UpdateExternalCount = 4
    };

    unsigned Port = 0;
    unsigned Index = 0;
    unsigned Flags = ConsumerEdge::None;

    ConsumerEdge() = default;

    ConsumerEdge(const StreamSetPort port, const unsigned index, const unsigned flags)
    : Port(port.Number), Index(index), Flags(flags) { }
};

using ConsumerGraph = adjacency_list<vecS, vecS, bidirectionalS, ConsumerNode, ConsumerEdge>;

using PartialSumStepFactorGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

enum TerminationSignal : unsigned {
    None = KernelBuilder::TerminationCode::None
    , Aborted = KernelBuilder::TerminationCode::Terminated
    , Fatal = KernelBuilder::TerminationCode::Fatal
    , Completed = Aborted | Fatal
};

enum TerminationCheckFlag : unsigned {
    Soft = 1
    , Hard = 2
};

using TerminationChecks = std::vector<unsigned>;

using TerminationPropagationGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, bool>;

enum CountingType : unsigned {
    Unknown = 0
    , Positive = 1
    , Negative = 2
    , Both = Positive | Negative
};

ENABLE_ENUM_FLAGS(CountingType)


template <typename T>
using OwningVector = std::vector<std::unique_ptr<T>>;

using KernelIdVector = std::vector<unsigned>;

using OrderingDAWG = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

struct PartitionData {

    KernelIdVector          Kernels;
    std::vector<Rational>   Repetitions;
    OrderingDAWG            Orderings;
    Rational                ExpectedStridesPerSegment{1};
    Rational                StridesPerSegmentCoV{0};
    unsigned                LinkedGroupId = 0;

};

using PartitionGraph = adjacency_list<vecS, vecS, bidirectionalS, PartitionData, StreamSetId>;

using PartitionDependencyGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, no_property>;

struct PartitionDataflowEdge {
    unsigned    KernelId;
    Rational    Expected;

    PartitionDataflowEdge() = default;

    PartitionDataflowEdge(unsigned id, Rational expected)
    : KernelId(id)
    , Expected(expected) {

    }

};

using PartitionDataflowGraph = adjacency_list<vecS, vecS, bidirectionalS, Rational, PartitionDataflowEdge>;

using PartitionOrderingGraph = adjacency_list<hash_setS, vecS, bidirectionalS, std::vector<unsigned>, double>;

struct PartitionOrdering {
    PartitionOrderingGraph  Graph;
    KernelIdVector          Kernels;
    const unsigned          NumOfKernelSets;

    PartitionOrdering(PartitionOrderingGraph && graph, unsigned numOfKernelSets, flat_set<unsigned> && kernels)
    : Graph(graph)
    , Kernels(kernels.begin(), kernels.end())
    , NumOfKernelSets(numOfKernelSets) {

    }
};

struct SchedulingNode {

    enum NodeType {
        IsKernel = 0
        , IsStreamSet = 1
        , IsExternal = 2
    };

    NodeType Type = NodeType::IsKernel;
    Rational Size;

    SchedulingNode() = default;

    SchedulingNode(NodeType ty, Rational size = Rational{0})
    : Type(ty)
    , Size(size) {

    }
};

using SchedulingGraph = adjacency_list<vecS, vecS, bidirectionalS, SchedulingNode, Rational>;

struct InternallyGeneratedStreamSetGraphNode {
    mutable Value * StreamSet = nullptr;
    mutable Value * RunLength = nullptr;
};

using InternallyGeneratedStreamSetGraph = adjacency_list<vecS, vecS, bidirectionalS, InternallyGeneratedStreamSetGraphNode, unsigned>;

struct FamilyScalarData {
    mutable Value * SharedStateObject = nullptr;
    mutable Value * allocateSharedInternalStreamSetsFuncPointer = nullptr;
    mutable Value * initializeThreadLocalFuncPointer = nullptr;
    mutable Value * allocateThreadLocalFuncPointer = nullptr;
    mutable Value * doSegmentFuncPointer = nullptr;
    mutable Value * finalizeThreadLocalFuncPointer = nullptr;
    mutable Value * finalizeFuncPointer = nullptr;

    unsigned CaptureFlags = 0;
    unsigned InputNum = 0;
    unsigned PassedParamNum = 0;


    enum {
        CaptureSharedStateObject = 1
        , CaptureThreadLocal = 2
        , CaptureAllocateInternal = 4
        , CaptureStoreInKernelState = 8
    };

    FamilyScalarData(unsigned inputNum, unsigned passedParam, unsigned flags)
    : CaptureFlags(flags)
    , InputNum(inputNum)
    , PassedParamNum(passedParam)
    {

    }

    FamilyScalarData() = default;
};

using FamilyScalarGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, FamilyScalarData>;

using ZeroInputGraph = adjacency_list<vecS, vecS, directedS, no_property, unsigned>;

using InOutGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, no_property>;

}

