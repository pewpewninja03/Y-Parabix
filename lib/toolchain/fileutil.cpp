/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <toolchain/fileutil.h>
#include <string>
#include <boost/filesystem.hpp>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <util/aligned_allocator.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>

using namespace llvm;
bool NoOSFileCaching;
static cl::opt<bool, true> OptNoOSCaching("disable-file-caching", cl::location(NoOSFileCaching),
                                         cl::desc("Disable OS file caching."), cl::init(false));

bool canMMap(const std::string & fileName) {
    if (fileName == "-") return false;
    namespace fs = boost::filesystem;
    fs::path p(fileName);
    boost::system::error_code errc;
    fs::file_status s = fs::status(p, errc);
    return !errc && is_regular_file(s);
}

AlignedFileBuffer::AlignedFileBuffer() {}

void AlignedFileBuffer::load(const std::string & fileName, bool preferMMap) {
    bool useMMap = preferMMap && canMMap(fileName);
    int32_t fd;
    int flags = O_RDONLY;
#ifdef __linux__
    if (NoOSFileCaching) {
        flags |= O_DIRECT;
    }
#endif
    if (fileName == "-") {
        fd = STDIN_FILENO;
    } else {
        fd = open(fileName.c_str(), flags);
        if (fd == -1) {
            mBufSize = -1;
            return;
        }
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        mBufSize = -1;
        return;
    }
    useMMap = useMMap && (st.st_size > 0);
    if (useMMap) {
        auto mmap_ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (LLVM_UNLIKELY(mmap_ptr == MAP_FAILED)) {
            mBufSize = -1;
            mDidMMap = false;
        } else {
            mBuffer = reinterpret_cast<char *>(mmap_ptr);
            mBufSize = st.st_size;
            mDidMMap = true;
        }
    } else {
        mAlignedSize = (st.st_size + mAlignment - 1) & -mAlignment;
        mBuffer = mAlloc.allocate(mAlignedSize, 0);
        if (mBuffer == nullptr) {
            report_fatal_error("Unable to allocate aligned file buffer");
        }
        mBufSize = read(fd, mBuffer, st.st_size);
        mDidMMap = false;
    }
    close(fd);
}

size_t AlignedFileBuffer::getBufSize() {
    return mBufSize;
}

char * AlignedFileBuffer::getBuf() {
    return mBuffer;
}

void AlignedFileBuffer::release() {
    if (mDidMMap) {
        munmap(reinterpret_cast<void *>(mBuffer), mBufSize);
    } else if (mBufSize > 0) {
        mAlloc.deallocate(mBuffer, 0);
    }
}

