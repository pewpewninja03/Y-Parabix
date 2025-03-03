#include "grep_engine.h"

namespace grep {

class NestedInternalSearchEngine {
    typedef void (*GrepFunctionType)(const char * buffer, const size_t length, MatchAccumulator &);
public:

    NestedInternalSearchEngine(BaseDriver & driver);

    ~NestedInternalSearchEngine();

    void setRecordBreak(GrepRecordBreakKind b) {mGrepRecordBreak = b;}

    void setCaseInsensitive()  {mCaseInsensitive = true;}

    void push(const re::PatternVector & REs);

    void pop();

    void doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum);

private:
    GrepRecordBreakKind mGrepRecordBreak;
    bool mCaseInsensitive;
    BaseDriver & mGrepDriver;

    std::vector<GrepFunctionType>   mMainMethod;
    std::vector<kernel::Kernel *>   mNested;



};


}
