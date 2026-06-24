#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief simpleSchedulePartitionedProgram
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::simpleSchedulePartitionedProgram(PartitionGraph & P, pipeline_random_engine & /* rng */) {

    const auto numOfPartitions = num_vertices(P);
    size_t u = -1U;
    for (unsigned i = 0; i < numOfPartitions; ++i) {
        for (const auto v : P[i].Kernels) {
            if (u != -1U) {
                add_edge(u, v, RelationshipType{ReasonType::OrderingConstraint}, Relationships);
            }
            u = v;
        }
    }

}

}
