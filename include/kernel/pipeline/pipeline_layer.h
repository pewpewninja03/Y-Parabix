#ifndef PIPELINE_LAYER_H
#define PIPELINE_LAYER_H

#include <vector>

namespace kernel {

class StreamSet;

class PipelinePhaseBoundary {

public:

    PipelinePhaseBoundary() = default;

    void addExpectedIORatio(const StreamSet * streamSet, const double requirement) {
        mRestrictions.emplace_back(streamSet, requirement);
    }

    const std::vector<std::pair<const StreamSet *, double>> & getRestrictions() const {
        return mRestrictions;
    }

private:

    std::vector<std::pair<const StreamSet *, double>> mRestrictions;

};

}

#endif // PIPELINE_LAYER_H
