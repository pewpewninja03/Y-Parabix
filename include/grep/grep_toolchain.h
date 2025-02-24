#pragma once

namespace grep {

extern int Threads;
extern bool UnicodeIndexing;
extern bool PropertyKernels;
extern bool MultithreadedSimpleRE;
extern int ScanMatchBlocks;
extern int MatchCoordinateBlocks;
extern int FileBatchSegments;
extern unsigned ByteCClimit;
extern bool TraceFiles;
extern bool ShowExternals;
extern bool UseByteFilterByMask;
extern bool UseNestedColourizationPipeline;
}

