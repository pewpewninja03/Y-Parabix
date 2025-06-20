#include "pipeline_analysis.hpp"
#include <llvm/Support/Format.h>
#include <llvm/ADT/DenseMap.h>
#include <bitset>

namespace kernel {

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
class PermutationBasedEvolutionaryAlgorithm {
public:

    using CandidateLengthType = unsigned;

    using Vertex = unsigned;

    using Candidate = std::vector<Vertex>;

//    using Queue = boost::lockfree::queue<Candidate>;

    using FitnessValueType = size_t;

    using Candidates = std::map<Candidate, FitnessValueType>;

    using Individual = typename Candidates::const_iterator;

    using Population = std::vector<Individual>;

    using WorkerPtr = std::unique_ptr<PermutationBasedEvolutionaryAlgorithmWorker>;

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

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief SchedulingAnalysisWorker
 ** ------------------------------------------------------------------------------------------------------------- */
class PermutationBasedEvolutionaryAlgorithmWorker {
public:
    using Candidate = PermutationBasedEvolutionaryAlgorithm::Candidate;
    using Population = PermutationBasedEvolutionaryAlgorithm::Population;

    virtual void repair(Candidate & candidate, pipeline_random_engine & rng) = 0;

    virtual size_t fitness(const Candidate & candidate, pipeline_random_engine & rng)  = 0;

    virtual void newCandidate(Candidate & candidate, pipeline_random_engine & rng);

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
