#include "evolutionary_algorithm.hpp"

namespace kernel {

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
