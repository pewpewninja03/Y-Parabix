#include "evolutionary_algorithm.hpp"
#include <boost/lockfree/queue.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <queue>

using namespace std::chrono;

using TimePoint = time_point<system_clock, seconds>;
using Candidate = kernel::PermutationBasedEvolutionaryAlgorithm::Candidate;

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

using CandidateQueue = WorkQueue<Candidate>;

template<typename T>
T abs_subtract(const T a, const T b) {
    if (a < b) {
        return b - a;
    } else {
        return a - b;
    }
}

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runGA
 ** ------------------------------------------------------------------------------------------------------------- */
const PermutationBasedEvolutionaryAlgorithm & PermutationBasedEvolutionaryAlgorithm::runGA() {

    assert (candidateLength > 0);

    population.reserve(3 * maxCandidates);
    assert (population.empty());

    Population nextGeneration;
    nextGeneration.reserve(3 * maxCandidates);

    std::vector<std::thread> threads;

    CandidateQueue workQueue;

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
                    std::this_thread::sleep_for(nanoseconds(100));
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

    BitVector uncopied(candidateLength);

    std::uniform_real_distribution<double> zeroToOneReal(0.0, 1.0);

    constexpr auto minFitVal = std::numeric_limits<FitnessValueType>::min();
    constexpr auto maxFitVal = std::numeric_limits<FitnessValueType>::max();
    constexpr auto worstFitnessValue = FitnessValueEvaluator::eval(minFitVal, maxFitVal) ? maxFitVal : minFitVal;

    unsigned averageStallCount = 0;
    unsigned bestStallCount = 0;

    double priorAverageFitness = worstFitnessValue;
    FitnessValueType priorBestFitness = worstFitnessValue;

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

        FitnessValueType sumOfGenerationalFitness = 0.0;
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
                    const auto d = w - averageGenerationFitness;
                    sumDiffOfSquares += d * d;
                }

                double beta;
                if (LLVM_LIKELY(sumDiffOfSquares == 0)) {
                    beta = 4.0;
                } else {
                    beta = std::sqrt(newPopulationSize / sumDiffOfSquares);
                }

                if (weights.size() < newPopulationSize) {
                    weights.resize(newPopulationSize);
                }

                const auto weights_end = weights.begin() + newPopulationSize;

                assert (chosen.empty());

                auto sumX = 0.0;
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

        // errs() << "averageGenerationFitness=" << averageGenerationFitness << "\n";
        // errs() << "bestGenerationalFitness=" << bestGenerationalFitness << "\n";

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
bool PermutationBasedEvolutionaryAlgorithm::insertCandidate(Candidate && candidate, Population & population) {
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
OrderingDAWG PermutationBasedEvolutionaryAlgorithm::getResult() const {
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
void PermutationBasedEvolutionaryAlgorithm::make_trie(const Candidate & C, OrderingDAWG & O) const {
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
 * @brief newCandidate
 ** ------------------------------------------------------------------------------------------------------------- */
void PermutationBasedEvolutionaryAlgorithmWorker::newCandidate(Candidate & C, pipeline_random_engine & rng) {
    std::iota(C.begin(), C.end(), 0);
    std::shuffle(C.begin(), C.end(), rng);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runHarmonySearch
 ** ------------------------------------------------------------------------------------------------------------- */
const BitStringBasedHarmonySearch & BitStringBasedHarmonySearch::runHarmonySearch() {

    assert (candidateLength > 1);

    population.reserve(maxCandidates);

    if (LLVM_UNLIKELY(initialize(population))) {
        goto enumerated_entire_search_space;
    }

    BEGIN_SCOPED_REGION

    std::uniform_real_distribution<double> zeroToOneReal(0.0, 1.0);

    std::uniform_int_distribution<unsigned> zeroOrOneInt(0, 1);

    FitnessValueType sumOfGenerationalFitness = 0.0;

    for (const auto & I : population) {
        const auto fitness = I->second;
        sumOfGenerationalFitness += fitness;
    }


    constexpr auto minFitVal = std::numeric_limits<FitnessValueType>::min();
    constexpr auto maxFitVal = std::numeric_limits<FitnessValueType>::max();
    constexpr auto worstFitnessValue = FitnessValueEvaluator::eval(minFitVal, maxFitVal) ? maxFitVal : minFitVal;

    unsigned averageStallCount = 0;

    double priorAverageFitness = worstFitnessValue;


    Population nextGeneration;
    nextGeneration.reserve(maxCandidates);

    Candidate newCandidate(candidateLength);

    for (unsigned round = 0; round < maxRounds; ++round) {

        const auto populationSize = population.size();

        assert (populationSize <= maxCandidates);

        auto considerationRate = CosAmplitude * std::cos(CosAngularFrequency * (double)round) + CosShift;

        std::uniform_int_distribution<unsigned> upToN(0, populationSize - 1);

        for (unsigned j = 0; j < candidateLength; ++j) {
            if (zeroToOneReal(rng) < considerationRate) {
                const auto k = upToN(rng);
                const bool v = population[k]->first.test(j);
                newCandidate.set(j, v);
            } else {
                newCandidate.set(j, zeroOrOneInt(rng));
            }
        }

        repairCandidate(newCandidate);

        const auto f = candidates.insert(std::make_pair(newCandidate, 0));
        if (LLVM_LIKELY(f.second)) {
            const auto val = fitness(f.first->first);
            if (val >= population.front()->second) {
                sumOfGenerationalFitness += val;
                f.first->second = val;
                bestResult = std::max(bestResult, val);
                std::pop_heap(population.begin(), population.end(), FitnessComparator{});
                const auto & worst = population.back();
                sumOfGenerationalFitness -= worst->second;
                population.pop_back();
                population.emplace_back(f.first);
                std::push_heap(population.begin(), population.end(), FitnessComparator{});
            }
        }

        const auto n = population.size();

        const double averageGenerationFitness = ((double)sumOfGenerationalFitness) / ((double)n);

        if (abs_subtract(averageGenerationFitness, priorAverageFitness) <= static_cast<double>(averageStallThreshold)) {
            if (++averageStallCount == maxStallGenerations) {
                break;
            }
        } else {
            averageStallCount = 0;
        }
        assert (averageStallCount < maxStallGenerations);

    }

    END_SCOPED_REGION

enumerated_entire_search_space:

    std::sort_heap(population.begin(), population.end(), FitnessComparator{});

    return *this;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief removeLeastFitCandidates
 ** ------------------------------------------------------------------------------------------------------------- */
void BitStringBasedHarmonySearch::updatePopulation(Population & nextGeneration) {
    for (const auto & I : nextGeneration) {
        assert (population.size() <= maxCandidates);
        if (population.size() == maxCandidates) {
            if (I->second >= population.front()->second) {
                std::pop_heap(population.begin(), population.end(), FitnessComparator{});
                population.pop_back();
            } else {
                // New item exceeds the weight of the heaviest candiate
                // in the population.
                continue;
            }
        }
        population.emplace_back(I);
        std::push_heap(population.begin(), population.end(), FitnessComparator{});
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief insertCandidate
 ** ------------------------------------------------------------------------------------------------------------- */
bool BitStringBasedHarmonySearch::insertCandidate(const Candidate & candidate, Population & population) {
    const auto f = candidates.insert(std::make_pair(candidate, 0));
    if (LLVM_LIKELY(f.second)) {
        const auto val = fitness(f.first->first);
        f.first->second = val;
        bestResult = std::max(bestResult, val);
        population.emplace_back(f.first);
        std::push_heap(population.begin(), population.end(), FitnessComparator{});
        return true;
    }
    return false;
}

}
