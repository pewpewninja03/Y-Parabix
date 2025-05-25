#include "pipeline_analysis.hpp"
#include <llvm/Support/Format.h>
#include <llvm/ADT/DenseMap.h>
#include <bitset>

#include <atomic>
#include <chrono>
#include <thread>
#include <queue>

namespace kernel {

template<typename FitnessValueType = size_t>
class PermutationBasedEvolutionaryAlgorithmWorker;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief PermutationBasedEvolutionaryAlgorithm
 *
 * Both the partition scheduling algorithm and whole program scheduling algorithm rely on the following class.
 * Within it is a genetic algorithm designed to find a minimum memory schedule of a given SchedulingGraph.
 * However, the phenotype of of the partition algorithm is a topological ordering and the phenotype of the
 * whole program is a hamiltonian path. This consitutes a significant enough difference that it is difficult
 * to call with only one function. Instead both the "initGA" and "repair" functions are implemented within
 * the actual scheduling functions.
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType = size_t>
class PermutationBasedEvolutionaryAlgorithm {
public:

    using CandidateLengthType = unsigned;

    using Vertex = unsigned;

    using Candidate = std::vector<Vertex>;

//    using Queue = boost::lockfree::queue<Candidate>;

    using Candidates = std::map<Candidate, FitnessValueType>;

    using Individual = typename Candidates::const_iterator;

    using Population = std::vector<Individual>;

    using WorkerType = PermutationBasedEvolutionaryAlgorithmWorker<FitnessValueType>;

    using WorkerPtr = std::unique_ptr<WorkerType>;

    struct FitnessValueEvaluator {
        constexpr static bool eval(const FitnessValueType a,const FitnessValueType b) {
            return a < b;
        }
    };

    struct FitnessComparator {
        bool operator()(const Individual & a,const Individual & b) const{
            return FitnessValueEvaluator::eval(a->second, b->second);
        }
    };

public:

    constexpr static size_t BITS_PER_SIZET = (CHAR_BIT * sizeof(size_t));

    struct permutation_bitset {

        permutation_bitset(const size_t N)
        : _value((N + BITS_PER_SIZET - 1) / BITS_PER_SIZET, 0) {

        }

        void randomize(pipeline_random_engine & rng) {
            std::uniform_int_distribution<size_t> distribution(std::numeric_limits<size_t>::min(), std::numeric_limits<size_t>::max());
            for (auto & a : _value) {
                a = distribution(rng);
            }
        }

        bool test(unsigned i) const {
            return (_value[i / BITS_PER_SIZET] & (i & (BITS_PER_SIZET - 1))) != 0;
        }

    private:
        SmallVector<size_t, 4> _value;
    };

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief initGA
     ** ------------------------------------------------------------------------------------------------------------- */
    virtual bool initGA(Population & initial) { return false; };

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief runGA
     ** ------------------------------------------------------------------------------------------------------------- */
    template <bool allowDuplicates>
    const PermutationBasedEvolutionaryAlgorithm & runGA();

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getResult
     ** ------------------------------------------------------------------------------------------------------------- */
    OrderingDAWG getResult() const;

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getBestFitnessValue
     ** ------------------------------------------------------------------------------------------------------------- */
    FitnessValueType getBestFitnessValue() const {
        assert (std::is_sorted(population.begin(), population.end(), FitnessComparator{}));
        return population.front()->second;
    }

protected:

    virtual WorkerPtr makeWorker(pipeline_random_engine & rng) = 0;

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief insertCandidate
     *
     * NOTE: not threadsafe; intended just for initGA function.
     ** ------------------------------------------------------------------------------------------------------------- */
    bool insertCandidate(Candidate && candidate, Population & population);

    virtual ~PermutationBasedEvolutionaryAlgorithm() { };

private:

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief make_trie
     ** ------------------------------------------------------------------------------------------------------------- */
    void make_trie(const Candidate & C, OrderingDAWG & O) const;

protected:

    PermutationBasedEvolutionaryAlgorithm(CandidateLengthType candidateLength
                                         , const unsigned maxInitTime
                                         , const unsigned maxInitCandidates
                                         , const unsigned maxRunTime
                                         , const unsigned maxCandidates
                                         , const unsigned maxStallRounds
                                         , const unsigned threadCount
                                         , pipeline_random_engine & rng)
    : candidateLength(candidateLength)
    , MaxInitTime(maxInitTime)
    , MaxInitCandidates(maxInitCandidates)
    , MaxRunTime(maxRunTime)
    , maxCandidates(maxCandidates)
    , averageStallThreshold(3)
    , maxStallThreshold(3)
    , maxStallGenerations(maxStallRounds)
    , threadCount(threadCount)
    , rng(rng) {
        population.reserve(maxCandidates * 3);
    }

protected:



    const CandidateLengthType candidateLength;
    const unsigned MaxInitTime;
    const unsigned MaxInitCandidates;
    const unsigned MaxRunTime;
    const unsigned maxCandidates;

    const FitnessValueType averageStallThreshold;
    const FitnessValueType maxStallThreshold;
    const unsigned maxStallGenerations;

    const unsigned threadCount;

    WorkerPtr mainWorker;

    Population population;

    std::map<Candidate, FitnessValueType> candidates;

    pipeline_random_engine & rng;

};

namespace {

struct TASLock {
    inline void lock() {
        while(_lock.exchange(true, std::memory_order_acquire));
    }
    inline void unlock() {
        _lock.store(false, std::memory_order_release);
    }
private:
    std::atomic<bool> _lock = {false};
};

template <typename T>
struct WorkQueue {

    inline bool empty() const {
        // NOTE: not thread safe; used only for debugging.
        return _queue.empty();
    }

    WorkQueue() = default;

    inline bool pop(T & c) {
        std::lock_guard<TASLock> lock(_lock);
        if (_queue.empty()) {
            return false;
        }
        c.swap(_queue.front());
        _queue.pop();
        return true;
    }

    inline void push(T && item) {
       std::lock_guard<TASLock> lock(_lock);
        _queue.push(std::move(item));
    }

    inline size_t size() {
        std::lock_guard<TASLock> lock(_lock);
        return _queue.size();
    }

private:
    mutable std::queue<T> _queue;
    TASLock _lock;
};

template<typename T, typename U>
inline T abs_subtract(const T a, const U b) {
    if (a < b) {
        return b - a;
    } else {
        return a - b;
    }
}

template <bool allowDuplicates>
struct CopyDataType {
    using Type = std::vector<unsigned>;

    CopyDataType(size_t n)
    : _data(n, 0) {

    }

    void set(unsigned i) {
        llvm_unreachable("cannot use this method");
    }

    bool test(unsigned i) {
        llvm_unreachable("cannot use this method");
    }

    void reset() {
        llvm_unreachable("cannot use this method");
    }

    unsigned & operator[](const unsigned i) {
        return _data[i];
    }
private:
    std::vector<unsigned> _data;

};

template <>
struct CopyDataType<false> {
    using Type = BitVector;

    CopyDataType(size_t n)
    : _data(n, false) {

    }

    void set(unsigned i) {
        _data.set(i);
    }

    bool test(unsigned i) {
        return _data.test(i);
    }

    void reset() {
        _data.reset();
    }

    unsigned & operator[](const unsigned i) {
        llvm_unreachable("cannot use this method");
    }
private:
    BitVector _data;
};

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runGA
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType>
template <bool allowDuplicates>
const PermutationBasedEvolutionaryAlgorithm<FitnessValueType> & PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::runGA() {

    using namespace std::chrono;



    assert (candidateLength > 0);

    population.reserve(3 * maxCandidates);
    assert (population.empty());

    Population nextGeneration;
    nextGeneration.reserve(3 * maxCandidates);

    std::vector<std::thread> threads;

    WorkQueue<Candidate> workQueue;

    bool finishedProcessing = false;

    TASLock candidateMapLock;
    TASLock nextGenLock;

    auto processCandidate = [&](WorkerPtr & worker, Candidate && C, pipeline_random_engine & rng) {
        assert (C.size() == candidateLength);
        worker->repair(C, rng);
        candidateMapLock.lock();
        const auto f = candidates.emplace(std::move(C), 0);
        candidateMapLock.unlock();
        if (LLVM_LIKELY(f.second)) {
            f.first->second = worker->fitness(f.first->first, rng);
        }
        nextGenLock.lock();
        nextGeneration.emplace_back(f.first);
        nextGenLock.unlock();
    };

    std::atomic<size_t> activeThreads{0};

    for (unsigned i = 1; i < threadCount; ++i) {
        threads.emplace_back([&]() {
            pipeline_random_engine threadRng(rng());
            auto worker = makeWorker(threadRng);
            for (;;) {
                Candidate C;
                if (workQueue.pop(C)) {
                    assert (C.size() == candidateLength);
                    activeThreads.fetch_add(1, std::memory_order_seq_cst);
                    processCandidate(worker, std::move(C), threadRng);
                    activeThreads.fetch_add(-1, std::memory_order_seq_cst);
                } else { // sleep 1/10 ms then check if we're finished.
                    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
                    assert (activeThreads.load(std::memory_order_relaxed) < threadCount);
                    if (finishedProcessing) {
                        break;
                    }
                }
            }
        });
    }

    mainWorker = makeWorker(rng);

    const auto start = system_clock::now();

    if (LLVM_UNLIKELY(initGA(nextGeneration))) {
        population.swap(nextGeneration);
        goto enumerated_entire_search_space;
    }

    BEGIN_SCOPED_REGION

    const auto initLimit = start + seconds(MaxInitTime);

    assert (MaxInitCandidates <= maxCandidates);

    if (nextGeneration.size() < MaxInitCandidates) {
        auto n = MaxInitCandidates - nextGeneration.size();
        while (--n && system_clock::now() < initLimit) {

            Candidate C(candidateLength);
            mainWorker->newCandidate(C, rng);
            workQueue.push(std::move(C));

        }
    }

    for (;;) {
        Candidate C;
        if (workQueue.pop(C)) {
            assert (C.size() == candidateLength);
            processCandidate(mainWorker, std::move(C), rng);
        } else {
            assert (workQueue.empty());
            for (;;) {
                const auto c = activeThreads.load(std::memory_order_relaxed);
                assert (c < threadCount);
                if (c == 0) break;
            }
            break;
        }
    }

    assert (workQueue.empty());
    assert (population.empty());
    assert (nextGeneration.size() > 0);
    assert (candidateLength > 0);

    // acquire then release the lock to ensure that all of the threads are clear
    nextGenLock.lock();
    nextGenLock.unlock();

    population.swap(nextGeneration);


    if (LLVM_UNLIKELY(candidateLength < 2)) {
        goto enumerated_entire_search_space;
    }

    permutation_bitset bitString(candidateLength);
    CopyDataType<allowDuplicates> uncopied(candidateLength);

    std::uniform_real_distribution<double> zeroToOneReal(0.0, 1.0);

    constexpr FitnessValueType minFitVal = std::numeric_limits<FitnessValueType>::min();
    constexpr FitnessValueType maxFitVal = std::numeric_limits<FitnessValueType>::max();
    constexpr FitnessValueType worstFitnessValue = FitnessValueEvaluator::eval(minFitVal, maxFitVal) ? maxFitVal : minFitVal;

    unsigned averageStallCount = 0;
    unsigned bestStallCount = 0;

    FitnessValueType priorAverageFitness{worstFitnessValue};
    FitnessValueType priorBestFitness{worstFitnessValue};

    std::vector<double> weights(maxCandidates * 3);

    flat_set<unsigned> chosen;
    chosen.reserve(maxCandidates);

    const auto maxTime = seconds(MaxRunTime);
    const auto maxTimeVal = maxTime.count();
    const auto limit = start + maxTime;

    for (unsigned g = 0; ; ++g) {

        const auto now = system_clock::now();
        if (now >= limit) break;

        nextGeneration.clear();

        // nextGeneration.assign(population.begin(), population.end());
        const auto populationSize = population.size();
        assert (populationSize > 1);

        Rational T{(limit - now).count(), maxTimeVal};
        const auto s = std::max(averageStallCount, bestStallCount);
        Rational S{maxStallGenerations - s, maxStallGenerations};
        const auto d = std::min(S, T);
        const double currentMutationRate = ((double)d.numerator() / (double)d.denominator()) + 0.03;
        const double currentCrossoverRate = 1.0 - currentMutationRate;

        // CROSSOVER:

        for (unsigned i = 1; i < populationSize; ++i) {
            for (unsigned j = 0; j < i; ++j) {
                if (zeroToOneReal(rng) <= currentCrossoverRate) {

                    const Candidate & A = population[i]->first;
                    const Candidate & B = population[j]->first;

                    // generate a random bit string
                    bitString.randomize(rng);

                    if (allowDuplicates) {


                        auto crossover_dup = [&](const Candidate & A, const Candidate & B, const bool selector) {

                            Candidate C(candidateLength);

                            assert (candidateLength > 1);
                            assert (C.size() == candidateLength);

                            #ifdef NDEBUG
                            for (unsigned k = 0; k < candidateLength; ++k) {
                                assert (uncopied[k] == 0);
                            }
                            #endif

                            for (unsigned k = 0; k < candidateLength; ++k) {
                                const auto v = A[k];
                                if (bitString.test(k) == selector) {
                                    C[k] = v;
                                } else {
                                    uncopied[v]++;
                                }
                            }

                            for (unsigned k = 0U, p = -1U; k < candidateLength; ++k) {
                                if (bitString.test(k) != selector) {
                                    // V contains 1-bits for every entry we did not
                                    // directly copy from A into C. We now insert them
                                    // into C in the same order as they are in B.
                                    for (;;){
                                        ++p;
                                        assert (p < candidateLength);
                                        const auto v = B[p];
                                        assert (v < candidateLength);
                                        if (uncopied[v] > 0) {
                                            C[k] = v;
                                            uncopied[v]--;
                                            break;
                                        }
                                    }

                                }
                            }

                            workQueue.push(std::move(C));

                        };

                        crossover_dup(A, B, true);

                        crossover_dup(B, A, false);

                    } else {

                        auto crossover = [&](const Candidate & A, const Candidate & B, const bool selector) {

                            Candidate C(candidateLength);

                            assert (candidateLength > 1);
                            assert (C.size() == candidateLength);

                            uncopied.reset();

                            #ifndef NDEBUG
                            unsigned count = 0;
                            #endif

                            for (unsigned k = 0; k < candidateLength; ++k) {
                                if (bitString.test(k) == selector) {
                                    const auto v = A[k];
                                    assert (v < candidateLength);
                                    assert ("candidate contains duplicate values?" && !uncopied.test(v));
                                    uncopied.set(v);
                                    #ifndef NDEBUG
                                    ++count;
                                    #endif
                                } else {
                                    C[k] = A[k];
                                }
                            }

                            for (unsigned k = 0U, p = -1U; k < candidateLength; ++k) {
                                const auto t = bitString.test(k);
                                if (t == selector) {
                                    // V contains 1-bits for every entry we did not
                                    // directly copy from A into C. We now insert them
                                    // into C in the same order as they are in B.
                                    #ifndef NDEBUG
                                    assert (count-- > 0);
                                    #endif
                                    for (;;){
                                        ++p;
                                        assert (p < candidateLength);
                                        const auto v = B[p];
                                        assert (v < candidateLength);
                                        if (uncopied.test(v)) {
                                            break;
                                        }
                                    }
                                    C[k] = B[p];
                                }
                            }

                            assert (count == 0);

                            workQueue.push(std::move(C));

                        };

                        crossover(A, B, true);

                        crossover(B, A, false);
                    }

                }
            }
        }

        // MUTATION:

        for (unsigned i = 0; i < populationSize; ++i) {
            if (zeroToOneReal(rng) <= currentMutationRate) {

                auto & A = population[i];

                Candidate C{A->first};

                const auto a = std::uniform_int_distribution<unsigned>{0, candidateLength - 2}(rng);
                const auto b = std::uniform_int_distribution<unsigned>{a + 1, candidateLength - 1}(rng);
                std::shuffle(C.begin() + a, C.begin() + b, rng);

                workQueue.push(std::move(C));

            }
        }

        for (;;) {
            Candidate C;
            if (workQueue.pop(C)) {
                assert (C.size() == candidateLength);
                processCandidate(mainWorker, std::move(C), rng);
            } else {
                assert (workQueue.empty());
                for (;;) {
                    const auto c = activeThreads.load(std::memory_order_relaxed);
                    assert (c < threadCount);
                    if (c == 0) break;
                }
                break;
            }
        }

        // acquire then release the lock to ensure that all of the threads are clear
        nextGenLock.lock();
        nextGenLock.unlock();

        assert (workQueue.empty());
        while (LLVM_UNLIKELY(nextGeneration.size() < 2)) {
            nextGeneration.push_back(population.back());
            population.pop_back();
        }
        population.clear();
        const auto newPopulationSize = nextGeneration.size();
        assert (newPopulationSize > 1);

        FitnessValueType sumOfGenerationalFitness{0};
        auto minFitness = maxFitVal;
        auto maxFitness = minFitVal;

        for (const auto & I : nextGeneration) {
            const auto fitness = I->second;
            sumOfGenerationalFitness += fitness;
            if (minFitness > fitness) {
                minFitness = fitness;
            }
            if (maxFitness < fitness) {
                maxFitness = fitness;
            }
        }

        const double averageGenerationFitness = ((double)sumOfGenerationalFitness) / ((double)newPopulationSize);

        FitnessValueType bestGenerationalFitness = maxFitness;
        if (FitnessValueEvaluator::eval(minFitness, maxFitness)) {
            bestGenerationalFitness = minFitness;
        }

        bool hasStalled = false;

        if (abs_subtract(averageGenerationFitness, priorAverageFitness) <= static_cast<double>(averageStallThreshold)) {
            if (++averageStallCount == maxStallGenerations) {
                hasStalled = true;
            }
        } else {
            averageStallCount = 0;
        }
        assert (averageStallCount <= maxStallGenerations);

        if (abs_subtract(bestGenerationalFitness, priorBestFitness) <= maxStallThreshold) {
            if (++bestStallCount == maxStallGenerations) {
                hasStalled = true;
            }
        } else {
            bestStallCount = 0;
        }
        assert (bestStallCount <= maxStallGenerations);

        // BOLTZMANN SELECTION:
        if (newPopulationSize <= maxCandidates) {
            population.swap(nextGeneration);
        } else {

            if (LLVM_UNLIKELY(minFitness == maxFitness)) {
                std::shuffle(nextGeneration.begin(), nextGeneration.end(), rng);
                for (unsigned i = 0; i < maxCandidates; ++i) {
                    population.emplace_back(nextGeneration[i]);
                }
            } else {

                // Calculate the variance for the annealing factor

                double sumDiffOfSquares = 0.0;
                for (unsigned i = 0; i < newPopulationSize; ++i) {
                    const auto w = nextGeneration[i]->second;
                    const auto d = (double)w - (double)averageGenerationFitness;
                    sumDiffOfSquares += d * d;
                }

                double beta;
                if (LLVM_LIKELY(sumDiffOfSquares == 0)) {
                    beta = 4.0;
                } else {
                    beta = std::sqrt((double)newPopulationSize / sumDiffOfSquares);
                }

                if (weights.size() < newPopulationSize) {
                    weights.resize(newPopulationSize);
                }

                assert (chosen.empty());

                if (newPopulationSize > maxCandidates) {
                    const auto weights_end = weights.begin() + newPopulationSize;
                    double sumX = 0.0;
                    unsigned fittestIndividual = 0;
                    const double r = beta / (double)(maxFitness - minFitness);
                    for (unsigned i = 0; i < newPopulationSize; ++i) {
                        const auto itr = nextGeneration[i];
                        assert (itr->first.size() == candidateLength);
                        const auto w = itr->second;
                        assert (w >= bestGenerationalFitness);
                        if (w == bestGenerationalFitness) {
                            fittestIndividual = i;
                        }
                        const double x = std::exp((double)(w - minFitness) * r);
                        const auto y = std::max(x, std::numeric_limits<double>::min());
                        sumX += y;
                        weights[i] = sumX;
                    }

                    assert (sumX > 0);

                    // ELITISM: always keep the fittest candidate for the next generation
                    chosen.insert(fittestIndividual);

                    std::uniform_real_distribution<double> selector(0, sumX);
                    while (chosen.size() < maxCandidates) {

                        const auto d = selector(rng);
                        assert (d < sumX);
                        const auto f = std::upper_bound(weights.begin(), weights_end, d);
                        assert (f != weights_end);
                        const unsigned j = std::distance(weights.begin(), f);
                        assert (j < newPopulationSize);
                        chosen.insert(j);
                    }
                } else {
                    for (size_t j = 0; j < newPopulationSize; ++j) {
                        chosen.insert(j);
                    }
                }
                for (unsigned i : chosen) {
                    assert (i < newPopulationSize);
                    const auto itr = nextGeneration[i];
                    assert (itr->first.size() == candidateLength);
                    population.push_back(itr);
                }
                chosen.clear();
            }

        }

        assert (population.size() > 0);

        if (LLVM_UNLIKELY(hasStalled)) break;

        priorAverageFitness = averageGenerationFitness;
        priorBestFitness = bestGenerationalFitness;
    }

    END_SCOPED_REGION

enumerated_entire_search_space:

    finishedProcessing = true;

    for (auto & t : threads) {
        t.join();
    }

    std::sort(population.begin(), population.end(), FitnessComparator{});

    return *this;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief insertCandidate
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType>
bool PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::insertCandidate(Candidate && candidate, Population & population) {
    assert (candidate.size() == candidateLength);
    #ifndef NDEBUG
    BitVector check(candidateLength);
    for (unsigned i = 0; i != candidateLength; ++i) {
        const auto v = candidate[i];
        assert ("invalid candidate #" && v < candidateLength);
        check.set(v);
    }
    assert ("duplicate candidate #" && (check.count() == candidateLength));
    #endif
    // NOTE: do not erase candidates or switch the std::map to something else without
    // verifying whether the population iterators are being invalidated.
    const auto f = candidates.emplace(std::move(candidate), 0);
    if (LLVM_LIKELY(f.second)) {
        f.first->second = mainWorker->fitness(f.first->first, rng);
    }
    assert (f.first != candidates.end());
    if (f.second) {
        population.emplace_back(f.first);
    }
    return f.second;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getResult
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType>
OrderingDAWG PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::getResult() const {
    assert (std::is_sorted(population.begin(), population.end(), FitnessComparator{}));

    // Construct a trie of all possible best (lowest) orderings of this partition

    auto i = population.begin();
    const auto end = population.end();
    OrderingDAWG result(1);
    const auto bestWeight = (*i)->second;
    do {
        make_trie((*i)->first, result);
    } while ((++i != end) && (bestWeight == (*i)->second));

    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief make_trie
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType>
void PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::make_trie(const Candidate & C, OrderingDAWG & O) const {
    assert (num_vertices(O) > 0);
    assert (C.size() == candidateLength);
    auto u = 0;

    for (unsigned i = 0; i != candidateLength; ) {
        const auto j = C[i];
        assert (j < candidateLength);
        for (const auto e : make_iterator_range(out_edges(u, O))) {
            if (O[e] == j) {
                u = target(e, O);
                goto in_trie;
            }
        }
        BEGIN_SCOPED_REGION
        const auto v = add_vertex(O);
        add_edge(u, v, j, O);
        u = v;
        END_SCOPED_REGION
in_trie:    ++i;
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief SchedulingAnalysisWorker
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FitnessValueType>
class PermutationBasedEvolutionaryAlgorithmWorker {
public:
    using Candidate = typename PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::Candidate;
    using Population = typename PermutationBasedEvolutionaryAlgorithm<FitnessValueType>::Population;

    virtual void repair(Candidate & candidate, pipeline_random_engine & rng) = 0;

    virtual FitnessValueType fitness(const Candidate & candidate, pipeline_random_engine & rng)  = 0;

    virtual void newCandidate(Candidate & candidate, pipeline_random_engine & rng) {
        std::iota(candidate.begin(), candidate.end(), 0);
        std::shuffle(candidate.begin(), candidate.end(), rng);
    }

    virtual ~PermutationBasedEvolutionaryAlgorithmWorker() { };

};


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief BitStringBasedHarmonySearch
 ** ------------------------------------------------------------------------------------------------------------- */
class BitStringBasedHarmonySearch {
public:

    struct Candidate {

        using BitWord = uintptr_t;

        static constexpr BitWord BITWORD_SIZE{sizeof(BitWord) * CHAR_BIT};

        explicit Candidate(const size_t N)
        : _value((N + BITWORD_SIZE - 1) / BITWORD_SIZE + 1, 0) {
            assert (N > 0);
            assert (_value.size() > 0);
        }

        Candidate(const Candidate & other)
        : _value(other._value) {

        }

        Candidate(Candidate && other)
        : _value(std::move(other._value)) {

        }

        Candidate & operator=(const Candidate & other ) {
            _value = other._value;
            return *this;
        }

        Candidate & operator=(Candidate && other ) {
            _value = std::move(other._value);
            return *this;
        }

        void set(const BitWord i, const bool value) {
            constexpr BitWord ZERO{0};
            constexpr BitWord ONE{1};
            auto & V = _value[i / BITWORD_SIZE];
            const auto mask = BitWord(1) << (i & (BITWORD_SIZE - ONE));
            V = (V | (value ? mask : ZERO)) & ~(value ? ZERO : mask);
        }

        void reset() {
            for (auto & v : _value) {
                v = 0;
            }
        }

        bool test(const BitWord i) const {
            constexpr BitWord ONE{1};
            const auto & V = _value[i / BITWORD_SIZE];
            const auto mask = BitWord(1) << (i & (BITWORD_SIZE - ONE));
            return (V & mask) != 0;
        }

        size_t count() const {
            size_t c = 0;
            for (auto v : _value) {
                c += std::bitset<BITWORD_SIZE>{v}.count();
            }
            return c;
        }

        size_t hash() const {
            std::size_t seed = 0;
            for (const auto & v : _value) {
                boost::hash_combine(seed, v);
            }
            return seed;
        }

        bool operator==(const Candidate & other) const {
            assert (other._value.size() == _value.size());
            auto i = _value.begin();
            auto j = other._value.begin();
            const auto end = _value.end();
            for (; i != end; ++i, ++j) {
                if (*i != *j) return false;
            }
            return true;
        }

        void swap(Candidate & other) {
            other._value.swap(_value);
        }

    private:
        SmallVector<size_t, 4> _value;
    };


    struct __CandidateHash {
        size_t operator()(const Candidate & key) const {
           return key.hash();
        }
    };


    using FitnessValueType = double;

    using Candidates = std::unordered_map<Candidate, FitnessValueType, __CandidateHash>;

    using Individual = typename Candidates::const_iterator;

    using CandidateList = std::vector<Candidate>;

    using Population = std::vector<Individual>;

    struct FitnessValueEvaluator {
        constexpr static bool eval(const FitnessValueType a,const FitnessValueType b) {
            return a > b;
        }
    };

    struct FitnessComparator {
        bool operator()(const Individual & a,const Individual & b) const{
            return FitnessValueEvaluator::eval(a->second, b->second);
        }
    };

public:

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief runHarmonySearch
     ** ------------------------------------------------------------------------------------------------------------- */
    const BitStringBasedHarmonySearch & runHarmonySearch();

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief removeLeastFitCandidates
     ** ------------------------------------------------------------------------------------------------------------- */
    void updatePopulation(Population & nextGeneration);

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getResult
     ** ------------------------------------------------------------------------------------------------------------- */
    Candidate getResult() const {
        assert (std::is_sorted(population.begin(), population.end(), FitnessComparator{}));
        return population.front()->first;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getBestFitnessValue
     ** ------------------------------------------------------------------------------------------------------------- */
    FitnessValueType getBestFitnessValue() const {
        assert (std::is_sorted(population.begin(), population.end(), FitnessComparator{}));
        return population.front()->second;
    }

protected:

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief insertCandidate
     ** ------------------------------------------------------------------------------------------------------------- */
    bool insertCandidate(const Candidate & candidate, Population & population);

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief initGA
     ** ------------------------------------------------------------------------------------------------------------- */
    virtual bool initialize(Population & initialPopulation) = 0;

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repairCandidate
     ** ------------------------------------------------------------------------------------------------------------- */
    virtual void repairCandidate(Candidate &) { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    virtual FitnessValueType fitness(const Candidate & candidate) = 0;

protected:

    BitStringBasedHarmonySearch(const unsigned candidateLength
                          , const unsigned maxRounds
                          , const unsigned maxCandidates
                          , const FitnessValueType averageStallThreshold
                          , const unsigned maxStallGenerations
                          , pipeline_random_engine & rng)
    : candidateLength(candidateLength)
    , maxRounds(maxRounds)
    , maxCandidates(maxCandidates)
    , averageStallThreshold(averageStallThreshold)
    , maxStallGenerations(maxStallGenerations)
    , rng(rng) {

    }

public:

    const unsigned candidateLength;
    const unsigned maxRounds;
    const unsigned maxCandidates;

    const FitnessValueType averageStallThreshold;
    const unsigned maxStallGenerations;

    constexpr static double MinHMCR = 0.8;
    constexpr static double MaxHMCR = 1.0;
    constexpr static double CosAngularFrequency = 0.3141592653589793238462643383279502884197169399375105820974944592;
    constexpr static double CosAmplitude = (MaxHMCR - MinHMCR) / 2.0;
    constexpr static double CosShift = (MaxHMCR + MinHMCR) / 2.0;

    FitnessValueType bestResult = 0;


    Population population;

    Candidates candidates;

    pipeline_random_engine & rng;

};

}
