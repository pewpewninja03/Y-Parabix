/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/illustrator/illustrator.h>
#include <kernel/illustrator/illustrator_binding.h>
#include <llvm/Support/raw_os_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <util/slab_allocator.h>
#include <boost/intrusive/detail/math.hpp>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/graph/adjacency_list.hpp>
#include "../pipeline/compiler/analysis/lexographic_ordering.hpp" // TODO: <- move this to a util folder we use this in the final version
#include <mutex>
#include <stack>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <stdio.h>
#endif

using namespace boost;
using namespace boost::container;
using boost::intrusive::detail::floor_log2;
using boost::intrusive::detail::is_pow2;
using namespace llvm;

#define ELEMENTS_PER_ALLOCATION 1024

inline static size_t udiv(const size_t x, const size_t y) {
    assert (is_pow2(y));
    const unsigned z = x >> floor_log2(y);
    assert (z == (x / y));
    return z;
}

inline static size_t ceil_udiv(const size_t x, const size_t y) {
    return udiv(((x - 1) | (y - 1)) + 1, y);
}

inline size_t get_terminal_width(const size_t fileDefault) {
#if defined(_WIN32)
    #warning UNTESTED CODE
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD type = GetFileType(h);
    if (type == FILE_TYPE_DISK) {
        return fileDefault;
    }
    GetConsoleScreenBufferInfo(h, &csbi);
    return = csbi.srWindow.Right-csbi.srWindow.Left+1;
#elif defined(__linux__) || defined(__APPLE__)
    if (isatty(STDERR_FILENO)) {
        struct winsize w;
        ioctl(STDERR_FILENO, TIOCGWINSZ, &w);
        auto len = w.ws_col;
        return len == 0 ? fileDefault : len;
    } else {
        return fileDefault;
    }
#endif // Windows/Linux
}

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

namespace kernel {

using MemoryOrdering = KernelBuilder::MemoryOrdering;

using StreamDataKey = const void *;

struct StreamDataStateObject;

struct StreamDataCapture;

struct StreamDataStateObject;

struct StreamDataElement {
    size_t StrideNum;
    size_t From;
    size_t To;
    uint8_t * Data;
    size_t * LoopIndex;
    size_t SequenceNum;
    #ifndef NDEBUG
    const StreamDataCapture * Capture;
    #endif
};

struct StreamDataChunk {
    StreamDataElement Data[ELEMENTS_PER_ALLOCATION];
    StreamDataChunk * Next = nullptr;
};

using StreamDataAllocator = SlabAllocator<uint8_t, 1024 * 1024>;

using LoopVector = SmallVector<size_t, 8 + 1>;

struct StreamDataCapture {
    const char * const StreamName;
    StreamDataCapture * Next = nullptr;

    size_t CurrentIndex = 0;

    StreamDataChunk Root;
    StreamDataChunk * Current = nullptr;

    const size_t Rows;
    const size_t Cols;
    const size_t ItemWidth;
    const MemoryOrdering Ordering;
    const IllustratorTypeId IllustratorType;
    const char Replacement0;
    const char Replacement1;
    const size_t * const LoopIdArray;
    #ifndef NDEBUG
    const StreamDataStateObject * StateObject;
    #endif

    inline void append(StreamDataStateObject * stateObjectEntry,
                       const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to, const size_t blockWidth);

    StreamDataCapture(const char * streamName, size_t rows, size_t cols, size_t iw, uint8_t ordering,
                      IllustratorTypeId typeId, const char rep0, const char rep1,
                      const size_t * loopIdArray)
    : StreamName(streamName), Current(&Root)
    , Rows(rows), Cols(cols), ItemWidth(iw), Ordering((MemoryOrdering)ordering)
    , IllustratorType(typeId), Replacement0(rep0), Replacement1(rep1)
    , LoopIdArray(loopIdArray) {
        assert (Ordering == MemoryOrdering::ColumnMajor || Ordering == MemoryOrdering::RowMajor);
        assert (is_pow2(ItemWidth));
    }
};

struct StreamDataStateObject {
    const char * KernelName;
    StreamDataCapture First;
    size_t SequenceLength = 0;
    LoopVector LoopIteration;
    bool InKernel = false;

    inline void enterKernel() {
        assert (!InKernel);
        assert (LoopIteration.empty());
        InKernel = true;
    }

    inline void enterLoop() {
        assert (InKernel);
        LoopIteration.emplace_back(0); // future iteration num
    }

    inline void iterateLoop() {
        assert (InKernel);
        assert (!LoopIteration.empty());
        LoopIteration.back()++;
    }

    inline void exitLoop() {
        assert (InKernel);
        assert (!LoopIteration.empty());
        LoopIteration.pop_back();
    }

    inline void exitKernel() {
        assert (InKernel);
        assert (LoopIteration.empty());
        InKernel = false;
    }

    StreamDataStateObject(const char * kernelName, const char * streamName, size_t rows, size_t cols, size_t iw, uint8_t ordering,
                      IllustratorTypeId typeId, const char rep0, const char rep1,
                      const size_t * loopIdArray)
    : KernelName(kernelName)
    , First(streamName, rows, cols, iw, ordering, typeId, rep0, rep1, loopIdArray) {

    }

    // Even when executed in multi-threaded mode, each kernel instance is guaranteed to be executed
    // in lock step manner. To ensure this, the pipeline disables state-free/data-parallel execution
    // when anything is illustrated. Thus we can safely use a single allocator per instance.
    StreamDataAllocator InternalAllocator;
};

using StreamDataEntry = std::tuple<const char *, const void *, const StreamDataCapture *>;

class StreamDataIllustrator {
public:

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerStreamDataCapture
 ** ------------------------------------------------------------------------------------------------------------- */
inline void registerStreamDataCapture(const char * kernelName, const char * streamName, const void * stateObject,
                                      const size_t rows, const size_t cols, const size_t itemWidth, const uint8_t memoryOrdering,
                                      const IllustratorTypeId illustratorType, const char replacement0, const char replacement1,
                                      const size_t * loopIdArray) {

 //   std::lock_guard<std::mutex> L(AllocatorLock);
    StreamDataCapture * newCapture = nullptr;
    #ifndef NDEBUG
    StreamDataStateObject * so = nullptr;
    #endif
    auto r = RegisteredStateObjects.find(stateObject);
    if (r == RegisteredStateObjects.end()) {
        StreamDataStateObject * newStateObjectEntry = GroupAllocator.allocate<StreamDataStateObject>(1);
        new (newStateObjectEntry) StreamDataStateObject(kernelName, streamName, rows, cols, itemWidth, memoryOrdering, illustratorType, replacement0, replacement1, loopIdArray);
        #ifndef NDEBUG
        so = newStateObjectEntry;
        #endif
        newStateObjectEntry->KernelName = kernelName;
        RegisteredStateObjects.emplace(stateObject, newStateObjectEntry);
        newCapture = &newStateObjectEntry->First;
    } else {
        StreamDataStateObject * existingStateObjectEntry = r->second;
        #ifndef NDEBUG
        so = existingStateObjectEntry;
        #endif
        StreamDataCapture * current = &existingStateObjectEntry->First;
        for (;;) {
            if (current->StreamName == streamName) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << "Illustrator error: multiple instances of " << streamName << " was registered for " << kernelName << "\n";
                report_fatal_error(msg.str());
            }
            assert (strcmp(current->StreamName, streamName) != 0);
            StreamDataCapture * next = current->Next;
            if (next == nullptr) {
                break;
            }
            current = next;
        }
        newCapture = GroupAllocator.allocate<StreamDataCapture>(1);
        new (newCapture) StreamDataCapture(streamName, rows, cols, itemWidth, memoryOrdering, illustratorType, replacement0, replacement1, loopIdArray);
        current->Next = newCapture;
    }
    assert (newCapture);
    #ifndef NDEBUG
    newCapture->StateObject = so;
    #endif
    // return id instead and force kernel to store it?
    InstallOrderCaptures.emplace_back(streamName, stateObject, newCapture);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief doStreamDataCapture
 ** ------------------------------------------------------------------------------------------------------------- */
inline void doStreamDataCapture(const char * streamName, const void * stateObject,
                                const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to,
                                const size_t blockWidth) {

    StreamDataStateObject * const stateObjectEntry = getStateObject(stateObject);
    StreamDataCapture * current = &stateObjectEntry->First;
    for (;;) {
        assert (current);
        if (current->StreamName == streamName) {
            break;
        }
        assert (strcmp(current->StreamName, streamName) != 0);
        current = current->Next;
    }

    current->append(stateObjectEntry, strideNum, streamData, from, to, blockWidth);
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterKernel
 ** ------------------------------------------------------------------------------------------------------------- */
inline void enterKernel(const void * stateObject) {
    getStateObject(stateObject)->enterKernel();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterLoop
 ** ------------------------------------------------------------------------------------------------------------- */
inline void enterLoop(const void * stateObject) {
    getStateObject(stateObject)->enterLoop();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief iterateLoop
 ** ------------------------------------------------------------------------------------------------------------- */
inline void iterateLoop(const void * stateObject) {
    getStateObject(stateObject)->iterateLoop();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief exitLoop
 ** ------------------------------------------------------------------------------------------------------------- */
inline void exitLoop(const void * stateObject) {
    getStateObject(stateObject)->exitLoop();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief enterKernel
 ** ------------------------------------------------------------------------------------------------------------- */
inline void exitKernel(const void * stateObject) {
    getStateObject(stateObject)->exitKernel();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getStateObject
 ** ------------------------------------------------------------------------------------------------------------- */
inline StreamDataStateObject * getStateObject(const void * address) const {
    auto r = RegisteredStateObjects.find(address);
    assert (r != RegisteredStateObjects.end());
    return r->second;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief displayCapturedData
 ** ------------------------------------------------------------------------------------------------------------- */
inline void displayCapturedData(const size_t blockWidth) const {
    struct KernelNameNode {
        StringRef Label;
        size_t NumOfCopies;
        size_t CurrentCopyNum;

        KernelNameNode(StringRef kernelName)
        : Label(kernelName)
        , NumOfCopies(0)
        , CurrentCopyNum(0) {

        }
    };

    struct StreamNameNode {
        StringRef Label;
        std::vector<KernelNameNode> Children;

        StreamNameNode(StringRef streamName, StringRef kernelName)
        : Label(streamName) {
            Children.emplace_back(kernelName);
        }
    };

    const auto n = InstallOrderCaptures.size();

    const auto numOfStateObjects = RegisteredStateObjects.size();

    // Generate the ordering of the captures

    struct CaptureData {
        std::string Name;
        std::vector<size_t> MaxIterations;
        const StreamDataCapture & Capture;
        std::vector<std::string> SubField;

        CaptureData(const StreamDataCapture * capture)
        : Capture(*capture) {
            assert (capture->Ordering == MemoryOrdering::RowMajor);
            assert (&Capture == capture);
        }
    };

    struct StateObjects {
        StreamDataStateObject * StateObject;
        std::vector<CaptureData> Capture;
    };

    std::vector<StateObjects> StateObjects(numOfStateObjects);

    flat_map<const void *, size_t> M;
    for (size_t i = 0, j = 0; i < n; ++i) {
        const void * stateObject;
        const StreamDataCapture * capture;
        std::tie(std::ignore, stateObject, capture) = InstallOrderCaptures[i];

        const auto f = M.find(stateObject);
        size_t k = 0;
        if (f == M.end()) {
            k = j++;
            assert (k < numOfStateObjects);
            StateObjects[k].StateObject = getStateObject(stateObject);
            M.emplace(stateObject, k);
        } else {
            k = f->second;
            assert (k < numOfStateObjects);
            assert (StateObjects[k].StateObject == getStateObject(stateObject));
        }
        assert (capture->StateObject == StateObjects[k].StateObject);
        StateObjects[k].Capture.emplace_back(capture);
        assert (StateObjects[k].Capture.back().Capture.Ordering == MemoryOrdering::RowMajor);
    }
    assert (M.size() == numOfStateObjects);

    #define TREE_NODE std::numeric_limits<size_t>::max()

    struct DisplayNode {
        size_t Index;
        std::vector<const StreamDataElement *> Elements;
        size_t CurrentElementPosition = 0;

        DisplayNode(size_t index = TREE_NODE) : Index(index) {}
    };

    using DisplayOrderGraph = adjacency_list<vecS, vecS, bidirectionalS, DisplayNode>;

    std::vector<DisplayOrderGraph> subgraphs(numOfStateObjects);

    // When reassembling the output data, we must be aware that the source of the printed data could come from different
    // strides. Moreover, "if" statements mean there could be gaps in the data and/or "whiles" may not process some
    // iterations that prior or subsequent strides will process. Collect all of the data we can and organize it into a
    // tree structure in which the leaves contain the data for a particular loop iteration's captured values.

    for (size_t stateObjectId = 0; stateObjectId < numOfStateObjects; ++stateObjectId) {

        auto & C = StateObjects[stateObjectId].Capture;
        const auto numOfCapturedValues = C.size();
        assert (numOfCapturedValues < TREE_NODE);

        auto & D = subgraphs[stateObjectId];
        const auto root = add_vertex(TREE_NODE, D);
        assert (root == 0);
        assert (D[0].Index == TREE_NODE);

        std::vector<size_t> V;

        flat_map<std::pair<size_t, size_t>, size_t> B;

        for (size_t captureNum = 0; captureNum < numOfCapturedValues; ++captureNum) {
            auto & R = C[captureNum];
            const auto & G = R.Capture;


            assert (G.Ordering == MemoryOrdering::RowMajor);

            // fill in the data for the i-th illustrated streamset

            const StreamDataChunk * current = &(G.Root);
            size_t index = 0;

            size_t limit = (current->Next == nullptr) ? G.CurrentIndex : ELEMENTS_PER_ALLOCATION;
            if (LLVM_UNLIKELY(limit == 0)) {
                continue;
            }

            std::vector<size_t> pos;
            flat_map<std::vector<size_t>, size_t> A;

            auto & maxIterations = R.MaxIterations;

            for (;;) {

                assert (current);
                assert (index < limit);
                const StreamDataElement & E = current->Data[index];

                assert (E.Capture == &G);

                size_t priorIndex = root;
                if (LLVM_UNLIKELY(E.LoopIndex)) {
                    for (size_t k = 0;;++k) {
                        const auto j = E.LoopIndex[k];
                        if (j || k == 0) {
                            const auto key = std::make_pair(k, j);
                            const auto f = B.find(key);
                            size_t u = 0;
                            if (f == B.end()) {
                                u = add_vertex(TREE_NODE, D);
                                add_edge(priorIndex, u, D);
                                B.emplace(key, u);
                            } else {
                                u = f->second;
                                assert (edge(priorIndex, u, D).second);
                            }
                            priorIndex = u;
                        }
                        if (j == 0) {
                            break;
                        }
                        if (maxIterations.size() <= k) {
                            maxIterations.resize(k + 1);
                        }
                        auto & M = maxIterations[k];
                        M = std::max(M, j);
                    }
                }

                assert (pos.empty());
                if (LLVM_UNLIKELY(E.LoopIndex)) {
                    for (size_t k = 0;;++k) {
                        const auto j = E.LoopIndex[k];
                        if (j == 0) {
                            break;
                        }
                        pos.push_back(j);
                    }
                }

                const auto f = A.find(pos);
                size_t elemVertex = 0;
                if (f == A.end()) {
                    elemVertex = add_vertex(captureNum, D);
                    add_edge(priorIndex, elemVertex, D);
                    A.emplace(std::vector<size_t>{pos.begin(), pos.end()}, elemVertex);
                    V.push_back(elemVertex);
                } else {
                    elemVertex = f->second;
                    assert (D[elemVertex].Index == captureNum);
                    assert (edge(priorIndex, elemVertex, D).second);
                }
                pos.clear();

                assert (D[elemVertex].Index < numOfCapturedValues);
                D[elemVertex].Elements.push_back(&E);

                if (LLVM_UNLIKELY(++index == limit)) {
                    current = current->Next;
                    index = 0;
                    if (LLVM_UNLIKELY(current == nullptr)) {
                        break;
                    }
                    limit = (current->Next == nullptr) ? G.CurrentIndex : ELEMENTS_PER_ALLOCATION;
                    assert (limit > 0);
                }

            }
        }

        // Identify disjoint loop nests as those in which every iteration of one occurs prior to
        // any iteration of a later one.

        const auto m = V.size();
        for (size_t i = 1; i < m; ++i) {
            const auto vi = V[i];
            assert (D[vi].Index != TREE_NODE);
            const auto & A = D[vi].Elements;
            assert (!A.empty());
            for (size_t j = 0; j < i; ++j) {
                const auto vj = V[j];
                assert (vi != vj);
                const auto & B = D[vj].Elements;
                assert (D[vj].Index != TREE_NODE);
                assert (!B.empty());
                assert (D[vi].Index != D[vj].Index);
                size_t a = 0, b = 0;
                for(;;) {
                    assert (A[a] != B[b]);
                    const auto sa = A[a]->StrideNum;
                    const auto sb = B[b]->StrideNum;
                    // We need to find the first stride that both of these process execute and compare
                    // the sequence nums. NOTE: this only compares entries with the same loop indices
                    // and thus should have occurred in the same program order.
                    if (LLVM_UNLIKELY(sa < sb)) {
                        if (++a == A.size()) {
                            break;
                        }
                    } else if (LLVM_UNLIKELY(sb < sa)) {
                        if (++b == B.size()) {
                            break;
                        }
                    } else {
                        auto va = vi, vb = vj;
                        assert (A[a]->Data != B[b]->Data || A[a]->Data == nullptr);
                        if (A[a]->SequenceNum > B[b]->SequenceNum) {
                            std::swap(va, vb);
                        } else {
                            assert (A[a]->SequenceNum < B[b]->SequenceNum);
                        }
                        add_edge(va, vb, D);
                        break;
                    }
                }
            }
        }

        // compute a lexical ordering of the graph
        std::vector<size_t> L;
        const auto total = num_vertices(D);
        L.reserve(total);
        lexical_ordering(D, L);

        std::vector<size_t> O(total);
        for (size_t i = 0; i < total; ++i) {
            O[L[i]] = i;
        }

        L.clear();
        for (size_t i = 0; i < total; ++i) {
            DisplayNode & Di = D[i];
            if (Di.Index == TREE_NODE) {
                assert (L.empty());
                for (const auto e : make_iterator_range(out_edges(i, D))) {
                    L.push_back(target(e, D));
                }
                clear_out_edges(i, D);
                std::sort(L.begin(), L.end(), [&](const size_t a, const size_t b) {
                    return O[a] < O[b];
                });
                for (auto v : L) {
                    add_edge(i, v, D);
                }
                L.clear();
            } else {
                clear_out_edges(i, D);
            }
        }
    }

    // Initialize some position information and determine whether the streamNames are unique so that we can make it
    // easier for the user to read the output.

    std::vector<StreamNameNode> roots;

    assert (is_pow2(blockWidth));

    for (size_t i = 0; i < n; ++i) {
        const char * streamName; const void * stateObject;
        const StreamDataCapture * capture;
        std::tie(streamName, stateObject, capture) = InstallOrderCaptures[i];
        StreamDataStateObject * const stateObjectEntry = getStateObject(stateObject);

        // determine if the streamName is unique or if we need
        const auto m = roots.size();

        StringRef curStreamName{streamName};
        StringRef curKernelName{stateObjectEntry->KernelName};

        for (size_t j = 0; j < m; ++j) {
            auto & A = roots[j];
            if (LLVM_UNLIKELY(curStreamName.compare(A.Label) == 0)) {
                const auto l = A.Children.size();
                for (size_t k = 0; k < l; ++k) {
                    auto & B = A.Children[k];
                    if (LLVM_UNLIKELY(curKernelName.compare(B.Label) == 0)) {
                        B.NumOfCopies++;
                        goto updated_trie;
                    }
                }
                // add new kernel name entry
                A.Children.emplace_back(curKernelName);
            }
        }
        roots.emplace_back(curStreamName, curKernelName);
updated_trie:
        continue;
    }



    // Initialize some information and construct the displayed names

    const auto m = roots.size();

    size_t longestNameLength = 0;
    size_t maxNumOfRows = 0;
    size_t maxNumOfCapturesPerStateObject = 0;

    for (size_t i = 0; i < numOfStateObjects; ++i) {
        auto & s = StateObjects[i];
        StreamDataStateObject * const stateObjectEntry = s.StateObject;
        const char * kernelName = stateObjectEntry->KernelName;
        maxNumOfCapturesPerStateObject = std::max(maxNumOfCapturesPerStateObject, s.Capture.size());

        for (CaptureData & cd : s.Capture) {

            auto & capture = cd.Capture;

            if (capture.IllustratorType == IllustratorTypeId::BixNum) {
                const auto r = capture.Rows;
                const auto bixNumRows = ceil_udiv(r, 4);
                cd.SubField.resize(bixNumRows);
                maxNumOfRows = std::max(maxNumOfRows, bixNumRows);
                for (unsigned j = 0; j < bixNumRows; ++j) {
                    const auto a = (bixNumRows - j - 1) * 4;
                    const auto b = std::min<size_t>(a + 3, r);
                    cd.SubField[j] = "[" + std::to_string(a) + "-" + std::to_string(b) + "]";
                }
            } else {
                const auto r = capture.Rows;
                cd.SubField.resize(r);
                maxNumOfRows = std::max(maxNumOfRows, r);
                if (r == 1) {
                    cd.SubField[0] = "";
                } else {
                    for (unsigned j = 0; j < r; ++j) {
                        cd.SubField[j] = "[" + std::to_string(j) + "]";
                    }
                }
            }


            size_t lengthOfMaxIterations = 0;
            const auto maxIterationSize = cd.MaxIterations.size();
            if (maxIterationSize > 0) {
                // {#,#,...}
                lengthOfMaxIterations = maxIterationSize + 2 - 1;
                for (size_t j = 0; j < maxIterationSize; ++j) {
                    lengthOfMaxIterations += (size_t)std::ceil(std::log10(cd.MaxIterations[j]));
                }
            }

            StringRef curStreamName{capture.StreamName};
            StringRef curKernelName{kernelName};

            for (size_t j = 0; j < m; ++j) {
                auto & A = roots[j];
                if (LLVM_UNLIKELY(curStreamName.compare(A.Label) == 0)) {
                    const auto l = A.Children.size();
                    for (size_t k = 0; k < l; ++k) {
                        auto & B = A.Children[k];
                        if (LLVM_UNLIKELY(curKernelName.compare(B.Label) == 0)) {
                            const auto c = B.NumOfCopies;
                            if (l == 1 && c == 0) {
                                cd.Name = curStreamName.str();
                            } else {
                                std::string name;
                                if (LLVM_LIKELY(l != 1)) {
                                    name = curKernelName.str() + ".";
                                }
                                name += curStreamName.str();
                                if (c > 0) {
                                    name += std::to_string(++B.CurrentCopyNum);
                                }
                                cd.Name = name;
                            }
                            size_t maxFieldLength = 0;
                            for (unsigned j = 0; j < cd.SubField.size(); ++j) {
                                auto & Sj = cd.SubField[j];
                                maxFieldLength = std::max(maxFieldLength, Sj.size());
                            }
                            auto length = cd.Name.length() + maxFieldLength + lengthOfMaxIterations;
                            longestNameLength = std::max(longestNameLength, length);
                            goto calculate_max_loop_depth;
                        }
                    }
                }
            }
            llvm_unreachable("Failed to locate stream group in trie?");
    calculate_max_loop_depth:
            continue;
        }
    }

    // display item aligned data

    std::vector<std::vector<char>> FormattedOutput(maxNumOfRows);

    const size_t consoleWidth = get_terminal_width(blockWidth);

    size_t charsPerRow = codegen::IllustratorDisplay;
    if (charsPerRow == 0) {
        if (LLVM_LIKELY(longestNameLength + 3 < consoleWidth)) {
            charsPerRow = consoleWidth - (longestNameLength + 3);
        } else {
            // TODO: what is a good default here for when we cannot even fit the names in the console?
            charsPerRow = 100;
        }
    }

    for (size_t i = 0; i < maxNumOfRows; ++i) {
        FormattedOutput[i].resize(charsPerRow);
    }

    SmallVector<char, 32> iterationVec;

    size_t startPosition = 0;

    auto & out = errs();

    for (;;) {

        bool noRowCompletelyFilled = true;

        const auto endPosition = startPosition + charsPerRow;

        for (size_t stateObjectId = 0; stateObjectId < numOfStateObjects; ++stateObjectId) {

            auto & C = StateObjects[stateObjectId].Capture;
            auto & D = subgraphs[stateObjectId];

            std::function<bool(size_t, bool, size_t)> parse_tree = [&](const size_t node, bool isFirst, const size_t depth) {

                assert (node < num_vertices(D));
                DisplayNode & Di = D[node];
                if (Di.Index == TREE_NODE) {

                    bool first = false;// (depth > 0);
                    bool any = false;
                    for (const auto e : make_iterator_range(out_edges(node, D))) {
                        const auto j = target(e, D);
                        any |= parse_tree(j, first, depth + 1);
                        first = false;
                    }
                    return any;

                } else {

                    assert (out_degree(node, D) == 0);

                    assert (Di.Index < C.size());

                    auto & R = C[Di.Index];
                    const auto & G = R.Capture;
                    const auto & F = R.SubField;

                    const auto numOfRows = F.size();

                    assert (G.Ordering == MemoryOrdering::RowMajor);

                    size_t scale = G.ItemWidth;
                    if (G.IllustratorType == IllustratorTypeId::ByteData) {
                        scale /= CHAR_BIT;
                    }

                    for (size_t i = 0; i < numOfRows; ++i) {
                        auto & toFill = FormattedOutput[i];
                        std::fill(toFill.begin(), toFill.end(), 0);
                    }

                    auto position = startPosition;

                    size_t end = 0;
                    const StreamDataElement * E = nullptr;

                    while (Di.CurrentElementPosition < Di.Elements.size()) {

                        E = Di.Elements[Di.CurrentElementPosition];

                        // each chunk is aligned in blockWidth x itemWidth bits

                        const auto from = E->From * scale;
                        const auto to = (E->To * scale);

                        if (to <= position) {
                            Di.CurrentElementPosition++;
                            continue;
                        }

                        if (from >= endPosition) {
                            break;
                        }

                        assert (position >= from);
                        assert (position < to);

                        const auto rowSize = G.Cols * (blockWidth * G.ItemWidth) / CHAR_BIT;

                        const auto chunkSize = G.Rows * rowSize;

                        assert (from <= position);
                        const auto offset = position - from;

                        const uint8_t * blockData = E->Data + ((offset / blockWidth) * chunkSize);
                        const size_t readStart = (offset & (blockWidth - 1));
                        const size_t writeStart = (position % charsPerRow);

                        assert (readStart <= blockWidth);
                        assert (writeStart <= charsPerRow);

                        const size_t blockDataLimit = std::min(blockWidth - readStart, charsPerRow - writeStart);

                        assert (position <= to);

                        const size_t length = std::min(blockDataLimit, to - position);
                        assert (length > 0);
                        assert ((readStart + length) <= blockWidth);
                        assert ((writeStart + length) <= charsPerRow);

                        if (G.IllustratorType == IllustratorTypeId::Bitstream) {

                            const char zeroCh = G.Replacement0;
                            const char oneCh = G.Replacement1;
                            for (size_t j = 0; j < numOfRows; ++j) {

                                assert (j < FormattedOutput.size());
                                auto & toFill = FormattedOutput[j];
                                assert (toFill.size() == charsPerRow);

                                const uint8_t * rowData = blockData + (j * rowSize);
                                for (size_t k = 0; k < length; ++k) {
                                    const auto in = (readStart + k);
                                    assert (in < blockWidth);
                                    const uint8_t v = rowData[in / CHAR_BIT] & (1UL << (in & (CHAR_BIT - 1)));
                                    const auto ch = (v == 0) ? zeroCh : oneCh;
                                    const auto out = (writeStart + k);
                                    assert (out < charsPerRow);
                                    toFill[out] = ch;
                                }
                            }

                        } else if (G.IllustratorType == IllustratorTypeId::BixNum) {

                            const char hexBase = G.Replacement0 - 10;

                            for (unsigned j = 0; j < numOfRows; ++j) {

                                const auto s = (j * 4);
                                assert (s < G.Rows);
                                const auto t = std::min(G.Rows - s, 4UL);
                                assert (j < FormattedOutput.size());
                                const auto x = numOfRows - j - 1;
                                assert (x < FormattedOutput.size());
                                auto & toFill = FormattedOutput[x];
                                assert (toFill.size() == charsPerRow);

                                for (size_t r = 0; r < t; ++r) {
                                    const uint8_t * rowData = blockData + ((s + r) * rowSize);
                                    for (size_t k = 0; k < length; ++k) {
                                        const auto in = (readStart + k);
                                        assert (in < blockWidth);
                                        assert ((1 << (in & 7)) < 256);
                                        const uint8_t v = ( rowData[in >> 3UL] & (1 << (in & 7)) ) != 0;
                                        assert (v == 0 || v == 1);
                                        const auto out = (writeStart + k);
                                        assert (out < charsPerRow);
                                        const auto z = v << r;
                                        assert (z < 256);
                                        assert ((toFill[out] & z) == 0);
                                        toFill[out] |= z;
                                    }
                                }

                                for (size_t k = 0; k < length; ++k) {
                                    const auto out = (writeStart + k);
                                    assert (out < charsPerRow);
                                    auto & c = toFill[out];
                                    assert (c < 16);
                                    if (c < 10) {
                                        c += '0';
                                    } else {
                                        c += hexBase;
                                    }
                                }

                            }

                        } else if (G.IllustratorType == IllustratorTypeId::ByteData) {

                            const char nonAsciiRep = G.Replacement0;
                            for (size_t j = 0; j < numOfRows; ++j) {

                                assert (j < FormattedOutput.size());
                                auto & toFill = FormattedOutput[j];
                                assert (toFill.size() == charsPerRow);

                                const uint8_t * rowData = blockData + (j * rowSize);
                                for (size_t k = 0; k < length; ++k) {
                                    const auto in = (readStart + k);
                                    assert (in < blockWidth);
                                    auto ch = rowData[in];
                                    const auto out = (writeStart + k);
                                    if (LLVM_UNLIKELY((ch < 32) || (ch > 126))) {
                                        switch (ch) {
                                            case '\t': case '\n': case '\r':
                                                ch = ' ';
                                                break;
                                            default:
                                                ch = nonAsciiRep;
                                        }
                                    }
                                    assert (out < charsPerRow);
                                    toFill[out] = ch;
                                }
                            }
                        }

                        position += length;

                        end = writeStart + length;
                        assert (end <= charsPerRow);
                        if (end == charsPerRow) {
                            noRowCompletelyFilled = false;
                            break;
                        }

                    }

                    assert (E || end == 0);

                    StringRef indices;

                    if (LLVM_LIKELY(E)) {
                        const auto numIterations = R.MaxIterations.size();
                        if (LLVM_UNLIKELY(numIterations > 0)) {
                            iterationVec.clear();
                            raw_svector_ostream iterationStr(iterationVec);
                            char joiner = '{';
                            for (size_t j = 0; j < numIterations; ++j) {
                                const auto v = E->LoopIndex[j];
                                assert (v);
                                iterationStr << joiner << v;
                                joiner = ',';
                            }
                            iterationStr << '}';
                            indices = iterationStr.str();
                        }
                    }

                    for (size_t j = 0; j < numOfRows; ++j) {
                        const auto & Fj = F[j];
                        out.indent(longestNameLength - R.Name.size() - indices.size() - Fj.size());
                        out << R.Name << Fj << indices << " | ";
                        auto & output = FormattedOutput[j];
                        for (size_t i = 0; i < end; ++i) {
                            out << output[i];
                        }
                        out << '\n';
                    }

                    return (end > 0);
                }
            };

            parse_tree(0, false, 0);

        }

        startPosition = endPosition;

        out << '\n';

        if (noRowCompletelyFilled) {
            break;
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief deconstructor
 ** ------------------------------------------------------------------------------------------------------------- */
~StreamDataIllustrator() {
    // slab allocated but group deconstructor needs to be called too
    for (auto & rc : RegisteredStateObjects) {
        rc.second->~StreamDataStateObject();
    }
}

private:

SlabAllocator<StreamDataStateObject, sizeof(StreamDataStateObject) * 64> GroupAllocator;
flat_map<StreamDataKey, StreamDataStateObject *> RegisteredStateObjects;
std::vector<StreamDataEntry> InstallOrderCaptures;

};

inline void StreamDataCapture::append(StreamDataStateObject * stateObjectEntry,
                   const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to, const size_t blockWidth) {
    assert (to >= from);

    auto & A = stateObjectEntry->InternalAllocator;

    StreamDataChunk * C = Current;
    if (LLVM_UNLIKELY(CurrentIndex == ELEMENTS_PER_ALLOCATION)) {
        StreamDataChunk * N = new (A.aligned_allocate(sizeof(StreamDataChunk), sizeof(size_t))) StreamDataChunk{};
        assert (N);
        C->Next = N;
        Current = N;
        CurrentIndex = 0;
        C = N;
    }
    assert (CurrentIndex < ELEMENTS_PER_ALLOCATION);
    StreamDataElement & E = C->Data[CurrentIndex++];
    E.StrideNum = strideNum;
    E.To = to;
    E.SequenceNum = (++stateObjectEntry->SequenceLength);
    #ifndef NDEBUG
    E.Capture = this;
    #endif
    assert (stateObjectEntry == StateObject);
    assert (from <= to);

    auto modFrom = from;

    if (LLVM_UNLIKELY(from == to)) {
        E.Data = nullptr;
    } else if (LLVM_UNLIKELY(ItemWidth >= CHAR_BIT && Rows == 1 && Cols == 1)) {
        // may be unsafe; do not memcpy any data that isn't explicitly given
        assert (ItemWidth % CHAR_BIT == 0);
        assert (ItemWidth > 0);
        const auto offset = udiv(from * ItemWidth, CHAR_BIT);
        const uint8_t * start = streamData + offset;
        const auto ItemBytes = (ItemWidth / CHAR_BIT);
        const size_t length  = (to - from) * ItemBytes;
        assert ((length % ItemBytes) == 0);
        E.Data = A.aligned_allocate(length, ItemBytes);
        assert (E.Data);
        std::memcpy(E.Data, start, length);
    } else {
        // each "block" of streamData will contain blockWidth items, regardless of the item width.
        const size_t blockSize = (Rows * Cols * ItemWidth * blockWidth) / CHAR_BIT;
        if (LLVM_UNLIKELY(blockSize == 0)) {
            E.Data = nullptr;
        } else {
            modFrom = from & (~(blockWidth - 1UL));
            const auto offset = (udiv(from, blockWidth) * blockSize);
            const uint8_t * start = streamData + offset;
            const auto end = (modFrom & (blockWidth - 1)) + (to - modFrom);
            const auto length = udiv(end + blockWidth - 1, blockWidth) * blockSize;
            assert (length > 0);
            E.Data = A.aligned_allocate(length, blockWidth / CHAR_BIT);
            assert (E.Data);
            std::memcpy(E.Data, start, length);
        }
    }

    E.From = modFrom;

    if (stateObjectEntry->InKernel) {
        const auto & L = stateObjectEntry->LoopIteration;
        const auto n = L.size();
        size_t * const V = (size_t*)A.aligned_allocate(n + 1, sizeof(size_t));
        assert (V);
        for (size_t i = 0; i < n; ++i) {
            V[i] = L[i]; assert (L[i]);
        }
        V[n] = 0;
        E.LoopIndex = V;
    } else {
        assert (stateObjectEntry->LoopIteration.empty());
        E.LoopIndex = nullptr;
    }

}


extern "C"
StreamDataIllustrator * createStreamDataIllustrator() {
    return new StreamDataIllustrator();
}

// Each kernel can verify that the display Name of every illustrated value is locally unique but since multiple instances
// of a kernel can be instantiated, we also need the address of the state object to identify each value. Additionally, the
// presence of family kernels means we cannot guarantee that all kernels will be compiled at the same time so we cannot
// number the illustrated values at compile time.

extern "C"
void illustratorRegisterCapturedData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                     const size_t rows, const size_t cols, const size_t itemWidth, const uint8_t memoryOrdering,
                                     const uint8_t illustratorTypeId, const char replacement0, const char replacement1, const size_t * loopIdArray) {
    illustrator->registerStreamDataCapture(kernelName, streamName, stateObject, rows, cols, itemWidth, memoryOrdering, (IllustratorTypeId)illustratorTypeId, replacement0, replacement1, loopIdArray);
}

extern "C"
void illustratorCaptureStreamData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                  const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to, const size_t blockWidth) {
    assert (illustrator);
    assert (is_pow2(blockWidth));
    illustrator->doStreamDataCapture(streamName, stateObject, strideNum, streamData, from, to, blockWidth);
}

extern "C"
void illustratorEnterKernel(StreamDataIllustrator * illustrator, const void * stateObject) {
    illustrator->enterKernel(stateObject);
}

extern "C"
void illustratorEnterLoop(StreamDataIllustrator * illustrator, const void * stateObject, const size_t loopId) {
    illustrator->enterLoop(stateObject);
}

extern "C"
void illustratorIterateLoop(StreamDataIllustrator * illustrator, const void * stateObject) {
    illustrator->iterateLoop(stateObject);
}


extern "C"
void illustratorExitLoop(StreamDataIllustrator * illustrator, const void * stateObject) {
    illustrator->exitLoop(stateObject);
}

extern "C"
void illustratorExitKernel(StreamDataIllustrator * illustrator, const void * stateObject) {
    illustrator->exitKernel(stateObject);
}

extern "C"
void illustratorDisplayCapturedData(const StreamDataIllustrator * illustrator, const size_t blockWidth) {
    assert (illustrator);
    assert (is_pow2(blockWidth));
    illustrator->displayCapturedData(blockWidth);
}

extern "C"
void destroyStreamDataIllustrator(StreamDataIllustrator * illustrator) {
    delete illustrator;
}


}
