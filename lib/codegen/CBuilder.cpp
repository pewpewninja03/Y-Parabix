/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <codegen/CBuilder.h>
#include <kernel/pipeline/driver/driver.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Format.h>
#include <toolchain/toolchain.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cstdarg>
#include <cstdlib>
#include <sys/mman.h>
#include <unistd.h>
#include <stdio.h>
#include <boost/format.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <boost/filesystem.hpp>
#include <cxxabi.h>
using boost::intrusive::detail::floor_log2;
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(10, 0, 0)
#include <llvm/Support/Alignment.h>
#endif
#include <unistd.h>

static constexpr unsigned NON_HUGE_PAGE_SIZE = 4096;

static constexpr auto ALIGNED_ALLOC_NAME = "std_aligned_alloc";

#ifdef HAS_ADDRESS_SANITIZER
#include <llvm/Analysis/AliasAnalysis.h>
#include <sanitizer/asan_interface.h>
#endif

#if defined(__i386__)
#define PRIdsz PRId32
#define PRIxsz PRIx32
#else
#define PRIdsz PRId64
#define PRIxsz PRIx64
#endif

#ifdef ENABLE_LIBBACKTRACE
#include <backtrace-supported.h>
#if BACKTRACE_SUPPORTED == 1
#include <backtrace.h>
#ifdef HAS_LIBUNWIND
#include <unwind.h>
#endif
#ifdef HAS_BACKTRACE
#include <execinfo.h>
#endif
#include <cxxabi.h>
#else
#undef ENABLE_LIBBACKTRACE
#endif
#endif

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(10, 0, 0)
    typedef unsigned            AlignType;
#else
    typedef llvm::Align         AlignType;
#endif

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
    using FixedVectorType = llvm::VectorType;
#else
    using FixedVectorType = llvm::FixedVectorType;
#endif

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

using namespace llvm;

static int accumulatedFreeCalls = 0;

extern "C" void free_debug_wrapper(void * ptr) {
    if (accumulatedFreeCalls < codegen::FreeCallBisectLimit) {
        accumulatedFreeCalls++;
        free(ptr);
    }
}

Value * CBuilder::CreateURem(Value * const number, Value * const divisor, const Twine Name) {
    if (ConstantInt * const c = dyn_cast<ConstantInt>(divisor)) {
        const auto d = c->getZExtValue();
        assert ("CreateURem divisor cannot be 0!" && d);
        if (is_power_2(d)) {
            if (LLVM_UNLIKELY(d == 1)) {
                return ConstantInt::getNullValue(number->getType());
            } else {
                return CreateAnd(number, ConstantInt::get(number->getType(), d - 1), Name);
            }
        }
    }
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CreateAssert(divisor, "CreateURem divisor cannot be 0!");
    }
    return Insert(BinaryOperator::CreateURem(number, divisor), Name);
}

Value * CBuilder::CreateUDiv(Value * const number, Value * const divisor, const Twine Name) {
    if (ConstantInt * c = dyn_cast<ConstantInt>(divisor)) {
        const auto d = c->getZExtValue();
        assert ("CreateUDiv divisor cannot be 0!" && d);
        if (is_power_2(d)) {
            if (d > 1) {
                return CreateLShr(number, ConstantInt::get(divisor->getType(), floor_log2(d)), Name);
            } else {
                return number;
            }
        }
    }
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CreateAssert(divisor, "CreateUDiv divisor cannot be 0!");
    }
    return Insert(BinaryOperator::CreateUDiv(number, divisor), Name);
}

Value * CBuilder::CreateCeilUDiv(Value * const number, Value * const divisor, const Twine Name) {
    assert (number->getType() == divisor->getType());
    if (LLVM_LIKELY(isa<Constant>(divisor))) {
        if (LLVM_UNLIKELY(cast<Constant>(divisor)->isOneValue())) {
            return number;
        }
    }
    // TODO: verify the ASM reuses the remainder from the initial division
    IntegerType * const intTy = cast<IntegerType>(number->getType());
    Value * const quot = CreateUDiv(number, divisor);
    Value * const rem = CreateURem(number, divisor);
    return CreateAdd(quot, CreateZExt(CreateIsNotNull(rem), intTy), Name);
}

Value * CBuilder::CreateRoundDown(Value * const number, Value * const divisor, const Twine Name) {
    if (isa<ConstantInt>(divisor)) {
        const auto d = cast<ConstantInt>(divisor)->getZExtValue();
        if (LLVM_UNLIKELY(d == 1)) {
            return number;
        } else if (is_power_2(d)) {
            return CreateAnd(number, ConstantExpr::getNeg(cast<ConstantInt>(divisor)));
        }
    }
    return CreateMul(CreateUDiv(number, divisor), divisor, Name);
}

Value * CBuilder::CreateRoundUp(Value * const number, Value * const divisor, const Twine Name) {
    if (isa<ConstantInt>(divisor)) {
        const auto d = cast<ConstantInt>(divisor)->getZExtValue();
        if (LLVM_UNLIKELY(d == 1)) {
            return number;
        } else if (is_power_2(d)) {
            Constant * const ONE = ConstantInt::get(divisor->getType(), 1);
            Constant * const toAdd = ConstantExpr::getSub(cast<ConstantInt>(divisor), ONE);
            return CreateAnd(CreateAdd(number, toAdd), ConstantExpr::getNeg(cast<ConstantInt>(divisor)));
        }
    }
    return CreateMul(CreateCeilUDiv(number, divisor), divisor, Name);
}


Value * CBuilder::CreateSaturatingAdd(Value * const a, Value * const b, const Twine Name) {
    // TODO: this seems to be an intrinsic in later versions of LLVM. Determine which.
    Value * const c = CreateAdd(a, b);
    Constant * const max = ConstantInt::getAllOnesValue(a->getType());
    return CreateSelect(CreateICmpULT(c, a), max, c, Name);
}

Value * CBuilder::CreateSaturatingSub(Value * const a, Value * const b, const Twine Name) {
    // TODO: this seems to be an intrinsic in later versions of LLVM. Determine which.
    Value * const c = CreateSub(a, b);
    Constant * const min = ConstantInt::getNullValue(a->getType());
    return CreateSelect(CreateICmpULT(a, b), min, c, Name);
}

Value * CBuilder::CreateOpenCall(Value * filename, Value * oflag, Value * mode) {
    Module * const m = getModule();
    IntegerType * const int32Ty = getInt32Ty();
    PointerType * const int8PtrTy = getInt8PtrTy();
    FunctionType * openTy = FunctionType::get(int32Ty, {int8PtrTy, int32Ty, int32Ty}, false);
    Function * openFn = m->getFunction("open");
    if (openFn == nullptr) {
        openFn = Function::Create(openTy, Function::ExternalLinkage, "open", m);
    }
    return CreateCall(openTy, openFn, {filename, oflag, mode});
}

// ssize_t write(int fildes, const void *buf, size_t nbyte);
Value * CBuilder::CreateWriteCall(Value * fileDescriptor, Value * buf, Value * nbyte) {
    PointerType * const voidPtrTy = getVoidPtrTy();
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    IntegerType * const int32Ty = getInt32Ty();
    FunctionType * writeTy = FunctionType::get(sizeTy, {int32Ty, voidPtrTy, sizeTy}, false);
    Function * write = m->getFunction("write");
    if (write == nullptr) {
        write = Function::Create(writeTy, Function::ExternalLinkage, "write", m);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
        write->addAttribute(2U, Attribute::NoAlias);
#else
    //TODO: update for LLVM14+
#endif
    }
    buf = CreatePointerCast(buf, voidPtrTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(buf, nbyte, "CreateWriteCall");
    }
    return CreateCall(writeTy, write, {fileDescriptor, buf, nbyte});
}

Value * CBuilder::CreateReadCall(Value * fileDescriptor, Value * buf, Value * nbyte) {
    PointerType * const voidPtrTy = getVoidPtrTy();
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    IntegerType * const int32Ty = getInt32Ty();
    FunctionType * readTy = FunctionType::get(sizeTy, {int32Ty, voidPtrTy, sizeTy}, false);
    Function * readFn = m->getFunction("read");
    if (readFn == nullptr) {
        readFn = Function::Create(readTy, Function::ExternalLinkage, "read", m);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
       readFn->addAttribute(2U, Attribute::NoAlias);
#else
    //TODO: update for LLVM14+
#endif
    }
    buf = CreatePointerCast(buf, voidPtrTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(buf, nbyte, "CreateReadCall");
    }
    return CreateCall(readTy, readFn, {fileDescriptor, buf, nbyte});
}

Value * CBuilder::CreateCloseCall(Value * fileDescriptor) {
    Module * const m = getModule();
    IntegerType * int32Ty = getInt32Ty();
    FunctionType * fty = FunctionType::get(int32Ty, {int32Ty}, true);
    Function * closeFn = m->getFunction("close");
    if (closeFn == nullptr) {
        closeFn = Function::Create(fty, Function::ExternalLinkage, "close", m);
    }
    return CreateCall(fty, closeFn, fileDescriptor);
}

Value * CBuilder::CreateUnlinkCall(Value * path) {
    Module * const m = getModule();
    FunctionType * fty = FunctionType::get(getInt32Ty(), {getInt8PtrTy()}, false);
    Function * unlinkFunc = m->getFunction("unlink");
    if (unlinkFunc == nullptr) {
        unlinkFunc = Function::Create(fty, Function::ExternalLinkage, "unlink", m);
        unlinkFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(fty, unlinkFunc, path);
}

Value * CBuilder::CreatePosixFAllocate(Value * fileDescriptor, Value * offset, Value * len) {
#if defined(__APPLE__) || defined(_WIN32)
    return nullptr;
#elif defined(__linux__)
    Module * const m = getModule();
    IntegerType * sizeTy = getSizeTy();
    FunctionType * fty = FunctionType::get(sizeTy, {getInt32Ty(), sizeTy, sizeTy}, false);
    Function * fPosixFAllocate = m->getFunction("posix_fallocate");
    if (fPosixFAllocate == nullptr) {
        fPosixFAllocate = Function::Create(fty, Function::ExternalLinkage, "posix_fallocate", m);
    }
    return CreateCall(fty, fPosixFAllocate, {fileDescriptor, offset, len});
#endif
}

Value * CBuilder::CreateFSync(Value * fileDescriptor) {
    Module * const m = getModule();
    IntegerType * int32Ty = getInt32Ty();
    FunctionType * fty = FunctionType::get(int32Ty, {int32Ty}, true);
    Function * fSync = m->getFunction("fsync");
    if (fSync == nullptr) {
        fSync = Function::Create(fty, Function::ExternalLinkage, "fsync", m);
    }
    return CreateCall(fty, fSync, fileDescriptor);
}



Value * CBuilder::CreateMkstempCall(Value * ftemplate) {
    Module * const m = getModule();
    FunctionType * const fty = FunctionType::get(getInt32Ty(), {getInt8PtrTy()}, false);
    Function * mkstempFn = m->getFunction("mkstemp");
    if (LLVM_UNLIKELY(mkstempFn == nullptr)) {
        mkstempFn = Function::Create(fty, Function::ExternalLinkage, "mkstemp", m);
    }
    return CreateCall(fty, mkstempFn, ftemplate);
}

Value * CBuilder::CreateStrlenCall(Value * str) {
    Module * const m = getModule();
    FunctionType * const fty = FunctionType::get(getSizeTy(), {getInt8PtrTy()}, false);
    Function * strlenFn = m->getFunction("strlen");
    if (LLVM_UNLIKELY(strlenFn == nullptr)) {
        strlenFn = Function::Create(fty, Function::ExternalLinkage, "strlen", m);
    }
    return CreateCall(fty, strlenFn, str);
}

Function * CBuilder::GetPrintf() {
    Module * const m = getModule();
    Function * printf = m->getFunction("printf");
    if (LLVM_UNLIKELY(printf == nullptr)) {
        FunctionType * const fty = FunctionType::get(getInt32Ty(), {getInt8PtrTy()}, true);
        printf = Function::Create(fty, Function::ExternalLinkage, "printf", m);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
        printf->addAttribute(1, Attribute::NoAlias);
#else
    //TODO: update for LLVM14+
#endif
    }
    return printf;
}

CallInst * CBuilder::__CreatePrintfCall(Value * const format, std::initializer_list<Value *> args) {
    SmallVector<Value *, 8> argVals(1);
    argVals[0] = format;
    argVals.append(args);
    Function * printFn = GetPrintf();
    return CreateCall(printFn->getFunctionType(), printFn, argVals);
}

Function * CBuilder::GetDprintf() {
    Module * const m = getModule();
    Function * dprintf = m->getFunction("dprintf");
    if (LLVM_UNLIKELY(dprintf == nullptr)) {
        FunctionType * fty = FunctionType::get(getInt32Ty(), {getInt32Ty(), getInt8PtrTy()}, true);
        dprintf = Function::Create(fty, Function::ExternalLinkage, "dprintf", m);
    }
    return dprintf;
}

CallInst * CBuilder::__CreateDprintfCall(Value * const fd, Value * const format, std::initializer_list<Value *> args) {
    SmallVector<Value *, 9> argVals(2);
    argVals[0] = fd;
    argVals[1] = format;
    argVals.append(args);
    Function * printFn = GetDprintf();
    return CreateCall(printFn->getFunctionType(), printFn, argVals);
}

CallInst * CBuilder::__CreateSprintfCall(Value * const str, Value * const format, std::initializer_list<Value *> args) {
    Module * const m = getModule();
    PointerType * int8PtrTy = getInt8PtrTy();
    FunctionType * fty = FunctionType::get(getInt32Ty(), {int8PtrTy, int8PtrTy}, true);
    Function * sprintf = m->getFunction("sprintf");
    if (LLVM_UNLIKELY(sprintf == nullptr)) {
        sprintf = Function::Create(fty, Function::ExternalLinkage, "sprintf", m);
    }
    SmallVector<Value *, 8> argVals(2);
    argVals[0] = str;
    argVals[1] = format;
    argVals.append(args);
    return CreateCall(fty, sprintf, argVals);
}

void CBuilder::CallPrintIntCond(StringRef name, Value * const value, Value * const cond, const STD_FD fd) {
    BasicBlock * const insertBefore = GetInsertBlock()->getNextNode();
    BasicBlock* const callBlock = CreateBasicBlock("callBlock", insertBefore);
    BasicBlock* const exitBlock = CreateBasicBlock("exitBlock", insertBefore);
    CreateCondBr(cond, callBlock, exitBlock);
    SetInsertPoint(callBlock);
    CallPrintInt(name, value, fd);
    CreateBr(exitBlock);
    SetInsertPoint(exitBlock);
}

CallInst * CBuilder::CallPrintInt(StringRef name, Value * const value, const STD_FD fd) {
    Module * const m = getModule();
    IntegerType * const int64Ty = getInt64Ty();
    FunctionType * const FT = FunctionType::get(getVoidTy(), { getInt32Ty(), getInt8PtrTy(), int64Ty }, false);
    Function * printRegister = m->getFunction("print_int");
    if (LLVM_UNLIKELY(printRegister == nullptr)) {
        auto ip = saveIP();
        Function * printFn = Function::Create(FT, Function::InternalLinkage, "print_int", m);
        auto arg = printFn->arg_begin();
        BasicBlock * entry = BasicBlock::Create(getContext(), "entry", printFn);
        SetInsertPoint(entry);
        Value * const fdInt = &*(arg++);
        fdInt->setName("fd");
        Value * const name = &*(arg++);
        name->setName("name");
        Value * value = &*arg;
        value->setName("value");
        std::vector<Value *> args(4);
        args[0] = fdInt;
        args[1] = GetString("%-40s = %" PRIx64 "\n");
        args[2] = name;
        args[3] = value;
        Function * DprintFn = GetDprintf();
        CreateCall(DprintFn->getFunctionType(), DprintFn, args);
        CreateFSync(fdInt);
        CreateRetVoid();
        printRegister = printFn;
        restoreIP(ip);
    }
    Value * num = value;
    Type * const t = value->getType();
    if (t->isPointerTy()) {
        num = CreatePtrToInt(value, int64Ty);
    } else if (t->isIntegerTy()) {
        if (t->getIntegerBitWidth() < 64) {
            num = CreateZExt(value, int64Ty);
        }
    } else {
        assert (!"CallPrintInt was given a non-integer/non-pointer value.");
        report_fatal_error("CallPrintInt was given a non-integer/non-pointer value.");
    }
    assert (num->getType()->isIntegerTy() && num->getType()->getIntegerBitWidth() == 64);
    return CreateCall(FT, printRegister, {getInt32(static_cast<uint32_t>(fd)), GetString(name), num});
}




Value * CBuilder::CreateMalloc(Value * size) {
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    FunctionType * fty = FunctionType::get(voidPtrTy, {sizeTy}, false);
    Function * f = m->getFunction("malloc");
    if (f == nullptr) {
        f = Function::Create(fty, Function::ExternalLinkage, "malloc", m);
        f->setCallingConv(CallingConv::C);
        f->setReturnDoesNotAlias();
    }
    size = CreateZExtOrTrunc(size, sizeTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        __CreateAssert(CreateIsNotNull(size), "CreateMalloc: 0-byte malloc is implementation defined", {});
    }
    CallInst * const ptr = CreateCall(fty, f, size);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        __CreateAssert(CreateIsNotNull(ptr), "CreateMalloc: returned null pointer", {});
    }
    CreateMemZero(ptr, size, 1);
    return ptr;
}

Value * CBuilder::CreatePageAlignedMalloc(Type * const type, Value * const ArraySize, const unsigned addressSpace) {
    return CreateAlignedMalloc(type, ArraySize, addressSpace, unsigned{NON_HUGE_PAGE_SIZE});
}

Value * CBuilder::CreatePageAlignedMalloc(Value * const size) {
    const auto alignment = unsigned{NON_HUGE_PAGE_SIZE};
    Constant * align = ConstantInt::get(size->getType(), NON_HUGE_PAGE_SIZE);
    Value * const alignedSize = CreateRoundUp(size, align);
    return CreateAlignedMalloc(alignedSize, alignment);
}

Value * CBuilder::CreateCacheAlignedMalloc(Type * const type, Value * const ArraySize, const unsigned addressSpace) {
    return CreateAlignedMalloc(type, ArraySize, addressSpace, getCacheAlignment());
}

Value * CBuilder::CreateCacheAlignedMalloc(Value * const size) {
    const auto alignment = getCacheAlignment();
    Constant * align = ConstantInt::get(size->getType(), alignment);
    Value * const alignedSize = CreateRoundUp(size, align);
    return CreateAlignedMalloc(alignedSize, alignment);
}

Value * CBuilder::CreateAlignedMalloc(Type * const type, Value * const ArraySize, const unsigned addressSpace, const unsigned alignment) {

    IntegerType * const sizeTy = getSizeTy();
    assert (type);
    Value * size = getTypeSize(type);
    if (ArraySize) {
        size = CreateMul(size, CreateZExtOrTrunc(ArraySize, sizeTy));
    }
    ConstantInt * const align = ConstantInt::get(sizeTy, alignment);
    size = CreateRoundUp(size, align);
    return CreatePointerCast(CreateAlignedMalloc(size, alignment), type->getPointerTo(addressSpace));
}

Value * CBuilder::CreateAlignedMalloc(Value * size, const unsigned alignment) {
    if (LLVM_UNLIKELY(!is_power_2(alignment))) {
        report_fatal_error("CreateAlignedMalloc: alignment must be a power of 2");
    }
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    ConstantInt * const align = ConstantInt::get(sizeTy, alignment);
    size = CreateZExtOrTrunc(size, sizeTy);

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const ZERO = ConstantInt::get(sizeTy, 0);
        __CreateAssert(CreateICmpNE(size, ZERO),
                       "CreateAlignedMalloc: 0-byte malloc "
                       "is implementation defined", {});

        __CreateAssert(CreateICmpEQ(CreateURem(size, align), ZERO),
                       "CreateAlignedMalloc: allocation size (%" PRIu64 ") must be an "
                       "integral multiple of alignment (%" PRIu64 ").", {size, align});
    }

    Function * fAlignedAlloc = m->getFunction(ALIGNED_ALLOC_NAME); assert (fAlignedAlloc);
    Value * ptr = CreateCall(fAlignedAlloc->getFunctionType(), fAlignedAlloc, {align, size});
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        __CreateAssert(CreateIsNotNull(ptr), "CreateAlignedMalloc: returned null when attempting "
                                             "to allocate %" PRIu64 " bytes at %" PRIu64 " alignment (out of memory?)",
                                             {size, align});
    }
    CreateMemZero(ptr, size, alignment);
    return ptr;
}

Value * CBuilder::CreateRealloc(Type * const type, Value * const base, Value * const ArraySize) {
    Value * size = getTypeSize(type);
    if (ArraySize) {
        size = CreateMul(size, CreateZExtOrTrunc(ArraySize, size->getType()));
    }
    return CreatePointerCast(CreateRealloc(base, size), type->getPointerTo());
}

Value * CBuilder::CreateRealloc(Value * const base, Value * const size) {
    assert ("Ptr is not a pointer type" && base->getType()->isPointerTy());
    assert ("Size is not an integer" && size->getType()->isIntegerTy());
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    FunctionType * fty = FunctionType::get(voidPtrTy, {voidPtrTy, sizeTy}, false);
    Function * f = m->getFunction("realloc");
    if (f == nullptr) {
        f = Function::Create(fty, Function::ExternalLinkage, "realloc", m);
        f->setCallingConv(CallingConv::C);
        f->setReturnDoesNotAlias();
    }    
    Value * basePtr = CreatePointerCast(base, voidPtrTy);
    CallInst * const ci = CreateCall(fty, f, {basePtr, CreateZExtOrTrunc(size, sizeTy)});
    Value * ptr = CreatePointerCast(ci, base->getType());
    return ptr;
}

void CBuilder::CreateFree(Value * const ptr) {
    assert (ptr->getType()->isPointerTy());
    Module * const m = getModule();
    Type * const voidPtrTy =  getVoidPtrTy();
    Value * castPtr = CreatePointerCast(ptr, voidPtrTy);
    if (codegen::FreeCallBisectLimit >= 0) {
        FunctionType * fty = FunctionType::get(getVoidTy(), {voidPtrTy}, false);
        Function * dispatcher = m->getFunction("free_debug_wrapper");
        if (dispatcher == nullptr) {
            dispatcher = Function::Create(fty, Function::ExternalLinkage, "free_debug_wrapper", m);
            dispatcher->setCallingConv(CallingConv::C);
            assert (dispatcher);
            CreateCall(fty, dispatcher, castPtr);
        }
    } else {
        FunctionType * fty = FunctionType::get(getVoidTy(), {voidPtrTy}, false);
        Function * f = m->getFunction("free");
        if (f == nullptr) {
            f = Function::Create(fty, Function::ExternalLinkage, "free", m);
            f->setCallingConv(CallingConv::C);
        }
        CreateCall(fty, f, castPtr);
    }
}

Value * CBuilder::CreateAnonymousMMap(Value * size, const unsigned flags) {
    PointerType * const voidPtrTy = getVoidPtrTy();
    IntegerType * const intTy = getInt32Ty();
    IntegerType * const sizeTy = getSizeTy();
    size = CreateZExtOrTrunc(size, sizeTy);
    ConstantInt * const prot =  ConstantInt::get(intTy, PROT_READ | PROT_WRITE);
    ConstantInt * const intflags =  ConstantInt::get(intTy, MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | flags);
    ConstantInt * const fd =  ConstantInt::get(intTy, -1ULL);
    Constant * const offset = ConstantInt::get(sizeTy, 0);
    return CreateMMap(ConstantPointerNull::getNullValue(voidPtrTy), size, prot, intflags, fd, offset);
}

#if __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(2,6,22)
#undef MAP_POPULATE_FLAGS
#define HAS_MAP_POPULATE
#endif
#endif

Value * CBuilder::CreateFileSourceMMap(Value * fd, Value * size) {
    PointerType * const voidPtrTy = getVoidPtrTy();
    IntegerType * const intTy = getInt32Ty();
    fd = CreateZExtOrTrunc(fd, intTy);
    IntegerType * const sizeTy = getSizeTy();
    size = CreateZExtOrTrunc(size, sizeTy);
    ConstantInt * const prot =  ConstantInt::get(intTy, PROT_READ);
    #ifdef HAS_MAP_POPULATE
    #define MMAP_FLAGS MAP_POPULATE|MAP_NONBLOCK|MAP_PRIVATE
    #else
    #define MMAP_FLAGS MAP_PRIVATE
    #endif
    ConstantInt * const flags =  ConstantInt::get(intTy, MMAP_FLAGS);
    Constant * const offset = ConstantInt::get(sizeTy, 0);
    return CreateMMap(ConstantPointerNull::getNullValue(voidPtrTy), size, prot, flags, fd, offset);
}

Value * CBuilder::CreateMMap(Value * const addr, Value * size, Value * const prot, Value * const flags, Value * const fd, Value * const offset) {
    Module * const m = getModule();
    Function * fMMap = m->getFunction("mmap");
    if (LLVM_UNLIKELY(fMMap == nullptr)) {
        PointerType * const voidPtrTy = getVoidPtrTy();
        IntegerType * const intTy = getInt32Ty();
        IntegerType * const sizeTy = getSizeTy();
        FunctionType * fty = FunctionType::get(voidPtrTy, {voidPtrTy, sizeTy, intTy, intTy, intTy, sizeTy}, false);
        fMMap = Function::Create(fty, Function::ExternalLinkage, "mmap", m);
    }
    FixedArray<Value *, 6> args;
    args[0] = addr;
    args[1] = size;
    args[2] = prot;
    args[3] = flags;
    args[4] = fd;
    args[5] = offset;

    Value * ptr = CreateCall(fMMap->getFunctionType(), fMMap, args);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(m);
        IntegerType * const intTy = getIntPtrTy(DL);
        Value * success = CreateICmpNE(CreatePtrToInt(addr, intTy), ConstantInt::get(intTy, (uint64_t)MAP_FAILED));
        CreateAssert(success, "CreateMMap: mmap failed to allocate memory");
    }
    return ptr;
}

Value * CBuilder::CreateMemFdCreate(Value * const name, Value * const flags) {
    Module * const m = getModule();
    Function * fShmOpen = m->getFunction("memfd_create");
    if (LLVM_UNLIKELY(fShmOpen == nullptr)) {
        IntegerType * const intTy = getInt32Ty();
        FunctionType * fty = FunctionType::get(intTy, {getInt8PtrTy(), intTy}, false);
        fShmOpen = Function::Create(fty, Function::ExternalLinkage, "memfd_create", m);
    }
    Value * retVal = CreateCall(fShmOpen->getFunctionType(), fShmOpen, {name, flags});
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(m);
        Value * success = CreateICmpNE(retVal, ConstantInt::get(getInt32Ty(), -1ULL));
        CreateAssert(success, "CreateMemFdCreate: failed to create anonymous memory file");
    }
    return retVal;
}


Value * CBuilder::CreateFTruncate(Value * const fd, Value * size) {
    Module * const m = getModule();
    Function * fTruncate = m->getFunction("ftruncate");
    if (LLVM_UNLIKELY(fTruncate == nullptr)) {
        IntegerType * const intTy = getInt32Ty();
        IntegerType * const offTy = getIntNTy(sizeof(off_t) * 8);
        FunctionType * fty = FunctionType::get(intTy, {intTy, offTy}, false);
        fTruncate = Function::Create(fty, Function::ExternalLinkage, "ftruncate", m);
    }
    Value * retVal = CreateCall(fTruncate->getFunctionType(), fTruncate, {fd, size});
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(m);
        Value * success = CreateICmpNE(retVal, ConstantInt::get(getInt32Ty(), -1ULL));
        __CreateAssert(success, "CreateFTruncate: failed to truncate fd", {});
    }
    return retVal;
}


/**
 * @brief CBuilder::CreateMAdvise
 * @param addr
 * @param length
 * @param advice
 *
 * Note: this funcition can fail if a kernel resource was temporarily unavailable. Test if this is more than a simple hint and handle accordingly.
 *
 *  ADVICE_NORMAL
 *      No special treatment. This is the default.
 *  ADVICE_RANDOM
 *      Expect page references in random order. (Hence, read ahead may be less useful than normally.)
 *  ADVICE_SEQUENTIAL
 *      Expect page references in sequential order. (Hence, pages in the given range can be aggressively read ahead, and may be freed
 *      soon after they are accessed.)
 *  ADVICE_WILLNEED
 *      Expect access in the near future. (Hence, it might be a good idea to read some pages ahead.)
 *  ADVICE_DONTNEED
 *      Do not expect access in the near future. (For the time being, the application is finished with the given range, so the kernel
 *      can free resources associated with it.) Subsequent accesses of pages in this range will succeed, but will result either in
 *      reloading of the memory contents from the underlying mapped file (see mmap(2)) or zero-fill-on-demand pages for mappings`
 *      without an underlying file.
 *
 * @return Value indicating success (0) or failure (-1).
 */
Value * CBuilder::CreateMAdvise(Value * addr, Value * length, const int advice) {
    Triple T(mTriple);
    Value * result = nullptr;

    if (T.isOSLinux() || T.isTargetMachineMac()) {
        Module * const m = getModule();
        IntegerType * const intTy = getInt32Ty();
        IntegerType * const sizeTy = getSizeTy();
        PointerType * const voidPtrTy = getVoidPtrTy();
        FunctionType * fty = FunctionType::get(intTy, {voidPtrTy, sizeTy, intTy}, false);
        Function * MAdviseFunc = m->getFunction("madvise");
        if (LLVM_UNLIKELY(MAdviseFunc == nullptr)) {
            MAdviseFunc = Function::Create(fty, Function::ExternalLinkage, "madvise", m);
        }
        addr = CreatePointerCast(addr, voidPtrTy);
        length = CreateZExtOrTrunc(length, sizeTy);
        result = CreateCall(fty, MAdviseFunc, {addr, length, ConstantInt::get(intTy, advice)});
    }
    return result;
}

#ifndef MREMAP_MAYMOVE
#define MREMAP_MAYMOVE	1
#endif

Value * CBuilder::CreateMRemap(Value * addr, Value * oldSize, Value * newSize) {
    Triple T(mTriple);
    Value * ptr = nullptr;
    if (T.isOSLinux()) {
        Module * const m = getModule();
        DataLayout DL(m);
        PointerType * const voidPtrTy = getVoidPtrTy();
        IntegerType * const sizeTy = getSizeTy();
        IntegerType * const intTy = getIntPtrTy(DL);
        FunctionType * fty = FunctionType::get(voidPtrTy, {voidPtrTy, sizeTy, sizeTy, intTy}, false);
        Function * fMRemap = m->getFunction("mremap");
        if (LLVM_UNLIKELY(fMRemap == nullptr)) {
            fMRemap = Function::Create(fty, Function::ExternalLinkage, "mremap", m);
        }
        addr = CreatePointerCast(addr, voidPtrTy);
        oldSize = CreateZExtOrTrunc(oldSize, sizeTy);
        newSize = CreateZExtOrTrunc(newSize, sizeTy);
        ConstantInt * const flags = ConstantInt::get(intTy, MREMAP_MAYMOVE);
        ptr = CreateCall(fty, fMRemap, {addr, oldSize, newSize, flags});
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * success = CreateICmpNE(CreatePtrToInt(addr, intTy), ConstantInt::getAllOnesValue(intTy)); // MAP_FAILED = -1
            CreateAssert(success, "CreateMRemap: mremap failed to allocate memory");
        }
    } else { // no OS mremap support
        ptr = CreateAnonymousMMap(newSize);
        CreateMemCpy(ptr, addr, oldSize, getPageSize());
        CreateMUnmap(addr, oldSize);
    }
    return ptr;
}

Value * CBuilder::CreateMUnmap(Value * addr, Value * len) {
    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    Module * const m = getModule();
    FunctionType * const fty = FunctionType::get(sizeTy, {voidPtrTy, sizeTy}, false);
    Function * munmapFunc = m->getFunction("munmap");
    if (LLVM_UNLIKELY(munmapFunc == nullptr)) {
        munmapFunc = Function::Create(fty, Function::ExternalLinkage, "munmap", m);
    }
    len = CreateZExtOrTrunc(len, sizeTy);
    addr = CreatePointerCast(addr, voidPtrTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = getIntPtrTy(DL);
        CreateAssert(len, "CreateMUnmap: length cannot be 0");
        Value * const addrValue = CreatePtrToInt(addr, intPtrTy);
        Value * const pageOffset = CreateURem(addrValue, ConstantInt::get(intPtrTy, getPageSize()));
        CreateAssertZero(pageOffset, "CreateMUnmap: addr must be a multiple of the page size");
        Value * const boundCheck = CreateICmpULT(addrValue, CreateSub(ConstantInt::getAllOnesValue(intPtrTy), CreateZExtOrTrunc(len, intPtrTy)));
        CreateAssert(boundCheck, "CreateMUnmap: addresses in [addr, addr+len) are outside the valid address space range");
    }
    return CreateCall(fty, munmapFunc, {addr, len});
}

Value * CBuilder::CreateMProtect(Value * addr, Value * size, const Protect protect) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        // mprotect() changes the access protections for the calling process's
        // memory pages containing any part of the address range in the interval
        // [addr, addr+len-1].  addr must be aligned to a page boundary.

        // mprotect(): POSIX.1-2001, POSIX.1-2008, SVr4.  POSIX says that the
        // behavior of mprotect() is unspecified if it is applied to a region of
        // memory that was not obtained via mmap(2).

        // On Linux, it is always permissible to call mprotect() on any address
        // in a process's address space (except for the kernel vsyscall area).
        // In particular, it can be used to change existing code mappings to be
        // writable. (NOTE: does not appear to be true on UBUNTU 16.04, 16.10 or 18.04)

        DataLayout DL(getModule());
        IntegerType * const intPtrTy = getIntPtrTy(DL);
        Constant * const pageSize = ConstantInt::get(intPtrTy, getPageSize());
        CreateAssertZero(CreateURem(CreatePtrToInt(addr, intPtrTy), pageSize), "CreateMProtect: addr must be aligned to page boundary");
    }

    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    IntegerType * const int32Ty = getInt32Ty();

    Module * const m = getModule();
    FunctionType * const fty = FunctionType::get(sizeTy, {voidPtrTy, sizeTy, int32Ty}, false);
    Function * mprotectFunc = m->getFunction("mprotect");
    if (LLVM_UNLIKELY(mprotectFunc == nullptr)) {
        mprotectFunc = Function::Create(fty, Function::ExternalLinkage, "mprotect", m);
    }
    addr = CreatePointerCast(addr, voidPtrTy);
    size = CreateZExtOrTrunc(size, sizeTy);
    Value * const result = CreateCall(fty, mprotectFunc, {addr, size, ConstantInt::get(int32Ty, (int)protect)});
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CreateAssertZero(result, "CreateMProtect: could not change the permission of the given address range");
    }
    return result;
}

IntegerType * LLVM_READNONE CBuilder::getIntAddrTy() const {
    return IntegerType::get(getContext(), sizeof(intptr_t) * 8);
}

PointerType * LLVM_READNONE CBuilder::getVoidPtrTy(const unsigned AddressSpace) const {
    //return PointerType::get(Type::getVoidTy(getContext()), AddressSpace);
    return PointerType::get(Type::getInt8Ty(getContext()), AddressSpace);
}


Value * CBuilder::CreateAtomicFetchAndAdd(Value * const val, Value * const ptr, MaybeAlign align) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const Size = getTypeSize(val->getType());
        CheckAddress(ptr, Size, "CreateAtomicFetchAndAdd: ptr");
    }
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(13, 0, 0)
    return CreateAtomicRMW(AtomicRMWInst::Add, ptr, val, AtomicOrdering::AcquireRelease);
#elif LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(16, 0, 0)
    return CreateAtomicRMW(AtomicRMWInst::Add, ptr, val, None, AtomicOrdering::AcquireRelease);
#else
    return CreateAtomicRMW(AtomicRMWInst::Add, ptr, val, align, AtomicOrdering::AcquireRelease);
#endif
}

Value * CBuilder::CreateAtomicFetchAndSub(Value * const val, Value * const ptr, MaybeAlign align) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const Size = getTypeSize(val->getType());
        CheckAddress(ptr, Size, "CreateAtomicFetchAndSub: ptr");
    }
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(13, 0, 0)
    return CreateAtomicRMW(AtomicRMWInst::Sub, ptr, val, AtomicOrdering::AcquireRelease);
#elif LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(16, 0, 0)
    return CreateAtomicRMW(AtomicRMWInst::Add, ptr, val, None, AtomicOrdering::AcquireRelease);
#else
    return CreateAtomicRMW(AtomicRMWInst::Sub, ptr, val, align, AtomicOrdering::AcquireRelease);
#endif
}

LoadInst * CBuilder::CreateAtomicLoadAcquire(Type * type, Value * ptr) {
    const auto alignment = type->getPrimitiveSizeInBits() / 8;
    LoadInst * inst = CreateAlignedLoad(type, ptr, alignment, true);
    inst->setOrdering(AtomicOrdering::Acquire);
    return inst;
}

StoreInst * CBuilder::CreateAtomicStoreRelease(Value * val, Value * ptr) {
    const auto alignment = val->getType()->getPrimitiveSizeInBits() / 8;
    StoreInst * inst = CreateAlignedStore(val, ptr, alignment, true);
    inst->setOrdering(AtomicOrdering::Release);
    return inst;
}

void CBuilder::setNontemporal(StoreInst * s) {
    s->setMetadata(LLVMContext::MD_nontemporal,
                   MDNode::get(getContext(), {ConstantAsMetadata::get(getInt32(1))}));
}

Value * CBuilder::CreatePrefetch(Value * ptr, PrefetchRW mode, unsigned locality, CacheType c) {
    Function * prefetchIntrin = Intrinsic::getDeclaration(getModule(), Intrinsic::prefetch);
    Value * modeVal = getInt32(mode == PrefetchRW::Read ? 0 : 1);
    Value * localityVal = getInt32(locality > 3 ? 3 : locality);
    Value * cacheKind = getInt32(c == CacheType::Instruction ? 0 : 1);
    return CreateCall(prefetchIntrin->getFunctionType(), prefetchIntrin, {CreateBitCast(ptr, getInt8PtrTy()), modeVal, localityVal, cacheKind});
}

PointerType * LLVM_READNONE CBuilder::getFILEptrTy() {
    if (mFILEtype == nullptr) {
        mFILEtype = StructType::create(getContext(), "struct._IO_FILE");
    }
    return mFILEtype->getPointerTo();
}

Value * CBuilder::CreateFOpenCall(Value * filename, Value * mode) {
    Module * const m = getModule();
    FunctionType * fty = FunctionType::get(getFILEptrTy(), {getInt8Ty()->getPointerTo(), getInt8Ty()->getPointerTo()}, false);
    Function * fOpenFunc = m->getFunction("fopen");
    if (fOpenFunc == nullptr) {
        FunctionType * fty = FunctionType::get(getFILEptrTy(), {getInt8Ty()->getPointerTo(), getInt8Ty()->getPointerTo()}, false);
        fOpenFunc = Function::Create(fty, Function::ExternalLinkage, "fopen", m);
        fOpenFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(fty, fOpenFunc, {filename, mode});
}

Value * CBuilder::CreateFReadCall(Value * ptr, Value * size, Value * nitems, Value * stream) {
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    FunctionType * fty = FunctionType::get(sizeTy, {voidPtrTy, sizeTy, sizeTy, getFILEptrTy()}, false);
    Function * fReadFunc = m->getFunction("fread");
    if (fReadFunc == nullptr) {
        fReadFunc = Function::Create(fty, Function::ExternalLinkage, "fread", m);
        fReadFunc->setCallingConv(CallingConv::C);
    }
    ptr = CreatePointerCast(ptr, voidPtrTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(ptr, CreateMul(size, nitems), "CreateFReadCall");
    }
    return CreateCall(fty, fReadFunc, {ptr, size, nitems, stream});
}

Value * CBuilder::CreateFWriteCall(Value * ptr, Value * size, Value * nitems, Value * stream) {
    Module * const m = getModule();
    IntegerType * const sizeTy = getSizeTy();
    PointerType * const voidPtrTy = getVoidPtrTy();
    FunctionType * fty = FunctionType::get(sizeTy, {voidPtrTy, sizeTy, sizeTy, getFILEptrTy()}, false);
    Function * fWriteFunc = m->getFunction("fwrite");
    if (fWriteFunc == nullptr) {
        fWriteFunc = Function::Create(fty, Function::ExternalLinkage, "fwrite", m);
        fWriteFunc->setCallingConv(CallingConv::C);
    }
    ptr = CreatePointerCast(ptr, voidPtrTy);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(ptr, CreateMul(size, nitems), "CreateFReadCall");
    }
    return CreateCall(fty, fWriteFunc, {ptr, size, nitems, stream});
}

Value * CBuilder::CreateFCloseCall(Value * stream) {
    Module * const m = getModule();
    FunctionType * fty = FunctionType::get(getInt32Ty(), {getFILEptrTy()}, false);
    Function * fCloseFunc = m->getFunction("fclose");
    if (fCloseFunc == nullptr) {
        fCloseFunc = Function::Create(fty, Function::ExternalLinkage, "fclose", m);
        fCloseFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(fty, fCloseFunc, {stream});
}

Value * CBuilder::CreateRenameCall(Value * oldName, Value * newName) {
    Module * const m = getModule();
    FunctionType * fty = FunctionType::get(getInt32Ty(), {getInt8PtrTy(), getInt8PtrTy()}, false);
    Function * renameFunc = m->getFunction("rename");
    if (renameFunc == nullptr) {
        renameFunc = Function::Create(fty, Function::ExternalLinkage, "rename", m);
        renameFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(fty, renameFunc, {oldName, newName});
}

Value * CBuilder::CreateRemoveCall(Value * path) {
    Module * const m = getModule();
    FunctionType * fty = FunctionType::get(getInt32Ty(), {getInt8PtrTy()}, false);
    Function * removeFunc = m->getFunction("remove");
    if (removeFunc == nullptr) {
        removeFunc = Function::Create(fty, Function::ExternalLinkage, "remove", m);
        removeFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(fty, removeFunc, {path});
}

struct __backtrace_data {
    const char * FileName = nullptr;
    const char * FunctionName = nullptr;
    size_t LineNo = 0;
} __attribute__((packed));

extern "C"
BOOST_NOINLINE
void __report_failure_v(const char * name, const char * fmt, const __backtrace_data * const trace[], const uint32_t traceLength, va_list & args) {
    // colourize the output if and only if stderr is piped to the terminal
    const auto colourize = isatty(STDERR_FILENO) == 1;
    raw_fd_ostream out(STDERR_FILENO, false);

    if (trace) {

        SmallVector<char, 4096> tmp;
        raw_svector_ostream trace_string(tmp);
        for (uint32_t i = 0; i < traceLength; ++i) {
            const __backtrace_data * const T = trace[i];
            if (T->FunctionName) {
                trace_string << StringRef{T->FunctionName} << ' ';
            }
            if (T->FileName) {
                trace_string << '(' << StringRef{T->FileName} << ':' << T->LineNo << ")";
            }
            trace_string << '\n';

        }
        if (colourize) {
            out.changeColor(raw_fd_ostream::WHITE, true);
        }
        out << "Compilation Stacktrace:\n";
        if (colourize) {
            out.resetColor();
        }
        out << trace_string.str();
    }
    if (name) {
        if (colourize) {
            out.changeColor(raw_fd_ostream::RED, true);
        }
        out << name << ": ";
    }
    if (colourize) {
        out.changeColor(raw_fd_ostream::WHITE, true);
    }
    char buffer[1024] = {0};
    const auto m = std::vsnprintf(buffer, 1024, fmt, args);
    out.write(buffer, m);
    if (trace == nullptr) {
        if (colourize) {
            out.changeColor(raw_fd_ostream::WHITE, true);
        }
        out << "\nNo debug symbols loaded.\n";
    }
    if (codegen::TaskThreads > 1 || codegen::SegmentThreads > 1) {
        if (colourize) {
            out.changeColor(raw_fd_ostream::BLUE, true);
        }
        out << " (Thread # ";
        out.write_hex(reinterpret_cast<unsigned long>(pthread_self()));
        out << ")";
    }
    if (colourize) {
        out.resetColor();
    }
    out << "\n\n";
    out.flush();
}

extern "C"
BOOST_NOINLINE
void __report_failure(const char * name, const char * fmt, const __backtrace_data * const trace[], const uint32_t traceLength, ...) {
    va_list args;
    va_start(args, traceLength);
    __report_failure_v(name, fmt, trace, traceLength, args);
    va_end(args);
    exit (-1);
}

#ifdef ENABLE_LIBBACKTRACE
extern "C"
BOOST_NOINLINE
int __backtrace_callback(void * data, uintptr_t /* pc */, const char *filename, int lineno, const char *function) {
    auto pc_data = reinterpret_cast<__backtrace_data*>(data);
    pc_data->FileName = filename;
    pc_data->FunctionName = function;
    pc_data->LineNo = lineno;
    return (lineno == 0) ? 0 : 1;
}

extern "C"
BOOST_NOINLINE
void __backtrace_syminfo_callback(void * data, uintptr_t /* pc */, const char *symname, uintptr_t symval, uintptr_t symsize) {
    auto pc_data = reinterpret_cast<__backtrace_data*>(data);
    pc_data->FileName = nullptr;
    pc_data->FunctionName = symname;
    pc_data->LineNo = 0;
}

extern "C"
BOOST_NOINLINE
void __backtrace_ignored_error_callback(void *data, const char *msg, int errnum) {
    auto pc_data = reinterpret_cast<__backtrace_data*>(data);
    pc_data->FileName = nullptr;
    pc_data->FunctionName = nullptr;
    pc_data->LineNo = 0;
}

using CallStack = SmallVector<uintptr_t, 64>;

#ifdef HAS_LIBUNWIND
static _Unwind_Reason_Code
__unwind_callback (struct _Unwind_Context *context, void *data) {
    CallStack * callstack = static_cast<CallStack *>(data);
    uintptr_t pc;
    #ifdef HAVE_GETIPINFO
    int ip_before_insn = 0;
    pc = _Unwind_GetIPInfo (context, &ip_before_insn);
    if (!ip_before_insn) {
      --pc;
    }
    #else
    pc = _Unwind_GetIP (context);
    #endif
    callstack->push_back(pc);
    return _URC_NO_REASON;
}
#endif // HAS_LIBUNWIND

#endif // ENABLE_LIBBACKTRACE

constexpr StringRef __BACKTRACE_STRUCT_NAME{"__bkstruct"};

void CBuilder::__CreateAssert(Value * const assertion, const Twine format, std::initializer_list<Value *> params) {

    if (LLVM_UNLIKELY(isa<Constant>(assertion))) {
        if (LLVM_LIKELY(!cast<Constant>(assertion)->isNullValue())) {
            return;
        }
    }

    Module * const m = getModule();
    LLVMContext & C = getContext();

    Function * assertFunc = m->getFunction("assert");
    if (LLVM_UNLIKELY(assertFunc == nullptr)) {

        auto ip = saveIP();
        IntegerType * const int1Ty = getInt1Ty();
        IntegerType * const int32Ty = getInt32Ty();
        PointerType * const int8PtrTy = getInt8PtrTy();
        PointerType * const int8PtrPtrTy = int8PtrTy->getPointerTo();
        // va_list is platform specific but since we are not directly modifying
        // any use of this type in LLVM code, just ensure it is large enough.
        ArrayType * const vaListTy = ArrayType::get(getInt8Ty(), sizeof(va_list));
        Type * const voidTy = getVoidTy();

        FixedArray<Type *, 3> fields;
        fields[0] = int8PtrTy;
        fields[1] = int8PtrTy;
        fields[2] = getSizeTy();
        StructType * const structTy = StructType::create(C, fields, __BACKTRACE_STRUCT_NAME, true);
        assert (getTypeSize(structTy)->getLimitedValue() == sizeof(__backtrace_data));

        PointerType * const structPtrTy = structTy->getPointerTo();

        FixedArray<Type *, 5> params;
        params[0] = int1Ty;
        params[1] = int8PtrTy;
        params[2] = int8PtrTy;
        params[3] = structPtrTy;
        params[4] = int32Ty;

        FunctionType * fty = FunctionType::get(voidTy, params, true);
        assertFunc = Function::Create(fty, Function::PrivateLinkage, "assert", m);
        BasicBlock * const entry = BasicBlock::Create(C, "", assertFunc);
        BasicBlock * const failure = BasicBlock::Create(C, "", assertFunc);
        BasicBlock * const success = BasicBlock::Create(C, "", assertFunc);
        auto arg = assertFunc->arg_begin();
        arg->setName("assertion");
        Value * assertion = &*arg++;
        arg->setName("name");
        Value * name = &*arg++;
        arg->setName("msg");
        Value * msg = &*arg++;
        arg->setName("trace");
        Value * trace = &*arg++;
        arg->setName("depth");
        Value * depth = &*arg++;
        SetInsertPoint(entry);
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        assertFunc->setHasUWTable();
        #else
        assertFunc->setUWTableKind(UWTableKind::Default);
        #endif
        assertFunc->setPersonalityFn(getDefaultPersonalityFunction());

        Value * const vaList = CreatePointerCast(CreateAlignedAlloca(vaListTy, mCacheLineAlignment), int8PtrTy);
        FunctionType * vaFuncTy = FunctionType::get(voidTy, { int8PtrTy }, false);
        Function * const vaStart = Function::Create(vaFuncTy, Function::ExternalLinkage, "llvm.va_start", m);
        Function * const vaEnd = Function::Create(vaFuncTy, Function::ExternalLinkage, "llvm.va_end", m);

        CreateCondBr(assertion, success, failure);

        SetInsertPoint(failure);

        params[0] = int8PtrTy;
        params[1] = int8PtrTy;
        params[2] = structPtrTy;
        params[3] = int32Ty;
        params[4] = int8PtrTy;

        FunctionType * const rfTy = FunctionType::get(voidTy, params, false);
        Function * const reportFn = mDriver->addLinkFunction(m, "__report_failure_v", rfTy,
                                                             reinterpret_cast<void *>(&__report_failure_v));
        reportFn->setCallingConv(CallingConv::C);

        CreateCall(vaFuncTy, vaStart, vaList);

        FixedArray<Value *, 5> args;
        args[0] = name;
        args[1] = msg;
        args[2] = trace;
        args[3] = depth;
        args[4] = vaList;

        CreateCall(rfTy, reportFn, args);
        CreateCall(vaFuncTy, vaEnd, vaList);

        Function * alloc_exception = getAllocateException();
        Value * const exception = CreateCall(alloc_exception->getFunctionType(), alloc_exception, { getTypeSize(int8PtrTy) } );
        Constant * const nil = ConstantPointerNull::get(int8PtrTy);
        IRBuilder<>::CreateStore(nil, CreateBitCast(exception, int8PtrPtrTy));
        // NOTE: the second argument is supposed to point to a std::type_info object.
        // The external value Clang passes into it resolves to "null" when RTTI is disabled.
        // This appears to work here but ought to be verified.
        Function * throwFn = getThrow();
        FunctionType * throwTy = throwFn->getFunctionType();
        CreateCall(throwTy, throwFn, { exception, nil, nil });
        CreateUnreachable();

        SetInsertPoint(success);
        CreateRetVoid();

        restoreIP(ip);
    }

    PointerType * const structPtrTy = cast<PointerType>(assertFunc->getArg(3)->getType());

    Constant * trace = nullptr;
    ConstantInt * depth = nullptr;

    #ifdef ENABLE_LIBBACKTRACE
    if (mBacktraceState) {
        // TODO: implement a tree structure and traverse from the bottom up, stopping when it
        // determines the correct parent?


        #ifdef HAS_LIBUNWIND
        CallStack callstack;
        _Unwind_Backtrace (__unwind_callback, &callstack);
        const auto n = callstack.size();
        #endif
        #ifdef HAS_BACKTRACE
        CallStack callstack(64);
        size_t n = 0;
        for (;;) {
            n = backtrace(reinterpret_cast<void **>(callstack.data()), callstack.size());
            if (LLVM_LIKELY(n < callstack.capacity())) {
                break;
            }
            callstack.resize(n * 2);
        }
        #endif

        SmallVector<Constant *, 64> traceArray(n);
        const auto state = reinterpret_cast<backtrace_state *>(mBacktraceState);


        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        StructType * const structTy = cast<StructType>(structPtrTy->getPointerElementType());
        #else
        StructType * const structTy = StructType::getTypeByName(getContext(), __BACKTRACE_STRUCT_NAME);
        #endif
        assert (getTypeSize(structTy)->getLimitedValue() == sizeof(__backtrace_data));

        char * demangled = nullptr;
        size_t bufferLength = 0;

        for (unsigned i = 0; i < n; ++i) {
            const auto pc = callstack[i];

            SmallVector<char, 16> tmp;
            raw_svector_ostream nm(tmp);
            nm << "__sym_" << pc;

            GlobalVariable * symbol = m->getGlobalVariable(nm.str(), true);

            if (symbol == nullptr) {

                __backtrace_data data;
                auto r = backtrace_pcinfo(state, pc, &__backtrace_callback, &__backtrace_ignored_error_callback, &data);
                if (LLVM_UNLIKELY(r == 0)) {
                    r = backtrace_syminfo(state, pc, &__backtrace_syminfo_callback, &__backtrace_ignored_error_callback, &data);
                }
                FixedArray<Constant *, 3> values;
                if (data.FileName) {
                    values[0] = GetString(data.FileName);
                } else {
                    values[0] = ConstantPointerNull::get(getInt8PtrTy());
                }
                Constant * funcName = nullptr;
                if (data.FunctionName) {
                    int status;
                    demangled = abi::__cxa_demangle(data.FunctionName, demangled, &bufferLength, &status);
                    if (LLVM_LIKELY(status == 0)) {
                        funcName = GetString(demangled);
                    } else {
                        funcName = GetString(data.FunctionName);
                    }
                } else {
                    funcName = ConstantPointerNull::get(getInt8PtrTy());
                }
                values[1] = funcName;
                values[2] = getSize(data.LineNo);
                Constant * const symbolStruct = ConstantStruct::get(structTy, values);
                assert (getTypeSize(symbolStruct->getType())->getLimitedValue() == sizeof(__backtrace_data));
                symbol = new GlobalVariable(*m, structTy, true, GlobalVariable::PrivateLinkage, symbolStruct, nm.str());
            }
            traceArray[i] = symbol;
        }

        ArrayType * traceTy = ArrayType::get(structPtrTy, n);
        assert (getTypeSize(traceTy)->getLimitedValue() == sizeof(__backtrace_data *) * n);
        trace = ConstantArray::get(traceTy, traceArray);
        trace = new GlobalVariable(*m, trace->getType(), true, GlobalVariable::PrivateLinkage, trace);
        trace = ConstantExpr::getPointerCast(trace, structPtrTy);
        depth = getInt32(n);
        free(demangled);
    } else {
    #endif
        trace = ConstantPointerNull::get(structPtrTy);
        depth = getInt32(0);
    #ifdef ENABLE_LIBBACKTRACE
    }
    #endif



    Value * const name = GetString(getKernelName());
    SmallVector<char, 1024> tmp;
    const StringRef fmt = format.toStringRef(tmp);
    // TODO: add a check that the number of var_args == number of message args?
    SmallVector<Value *, 12> args(5);
    args[0] = assertion; assert (assertion);
    args[1] = name; assert (name);
    args[2] = GetString(fmt);
    args[3] = trace; assert (trace);
    args[4] = depth; assert (depth);
    args.append(params);
    IRBuilder<>::CreateCall(assertFunc->getFunctionType(), assertFunc, args);
}

void CBuilder::CreateExit(const int exitCode) {
    Module * const m = getModule();
    Function * exit = m->getFunction("exit");
    if (LLVM_UNLIKELY(exit == nullptr)) {
        FunctionType * fty = FunctionType::get(getVoidTy(), {getInt32Ty()}, false);
        exit = Function::Create(fty, Function::ExternalLinkage, "exit", m);
        exit->setDoesNotReturn();
        exit->setDoesNotThrow();
    }
    CreateCall(exit->getFunctionType(), exit, getInt32(exitCode));
}

BasicBlock * CBuilder::CreateBasicBlock(const StringRef name, BasicBlock * insertBefore) {
    return BasicBlock::Create(getContext(), name, GetInsertBlock()->getParent(), insertBefore);
}

bool CBuilder::supportsIndirectBr() const {
    return !codegen::DebugOptionIsSet(codegen::DisableIndirectBranch);
}

BranchInst * CBuilder::CreateLikelyCondBr(Value * Cond, BasicBlock * True, BasicBlock * False, const int probability) {
    MDBuilder mdb(getContext());
    if (probability < 0 || probability > 100) {
        report_fatal_error("branch weight probability must be in [0,100]");
    }
    return CreateCondBr(Cond, True, False, mdb.createBranchWeights(probability, 100 - probability));
}

Value * CBuilder::CreatePopcount(Value * bits) {
    Function * ctpopFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::ctpop, bits->getType());
    return CreateCall(ctpopFunc->getFunctionType(), ctpopFunc, bits);
}

Value * CBuilder::CreateCountForwardZeroes(Value * value, const Twine Name, const no_conversion<bool> guaranteedNonZero) {
    if (LLVM_UNLIKELY(guaranteedNonZero && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CreateAssert(value, "CreateCountForwardZeroes: value cannot be zero!");
    }
    Function * cttzFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::cttz, value->getType());
    return CreateCall(cttzFunc->getFunctionType(), cttzFunc, {value, getInt1(guaranteedNonZero)}, Name);
}

Value * CBuilder::CreateCountReverseZeroes(Value * value, const Twine Name, const no_conversion<bool> guaranteedNonZero) {
    if (LLVM_UNLIKELY(guaranteedNonZero && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CreateAssert(value, "CreateCountReverseZeroes: value cannot be zero!");
    }
    Function * ctlzFunc = Intrinsic::getDeclaration(getModule(), Intrinsic::ctlz, value->getType());
    return CreateCall(ctlzFunc->getFunctionType(), ctlzFunc, {value, getInt1(guaranteedNonZero)}, Name);
}

Value * CBuilder::CreateResetLowestBit(Value * bits, const Twine Name) {
    return CreateAnd(bits, CreateSub(bits, ConstantInt::get(bits->getType(), 1)), Name);
}

Value * CBuilder::CreateIsolateLowestBit(Value * bits, const Twine Name) {
    return CreateAnd(bits, CreateNeg(bits), Name);
}

Value * CBuilder::CreateMaskToLowestBitInclusive(Value * bits, const Twine Name) {
    return CreateXor(bits, CreateSub(bits, ConstantInt::get(bits->getType(), 1)), Name);
}

Value * CBuilder::CreateMaskToLowestBitExclusive(Value * bits, const Twine Name) {
    return CreateAnd(CreateSub(bits, ConstantInt::get(bits->getType(), 1)), CreateNot(bits), Name);
}

Value * CBuilder::CreateZeroHiBitsFrom(Value * bits, Value * pos, const Twine Name) {
    Type * Ty = bits->getType();
    Constant * one = Constant::getIntegerValue(Ty, APInt(Ty->getScalarSizeInBits(), 1));
    Value * mask = CreateSub(CreateShl(one, pos), one);
    return CreateAnd(bits, mask, Name);
}

Value * CBuilder::CreateExtractBitField(Value * bits, Value * start, Value * length, const Twine Name) {
    Constant * One = ConstantInt::get(bits->getType(), 1);
    return CreateAnd(CreateLShr(bits, start), CreateSub(CreateShl(One, length), One), Name);
}

Value * CBuilder::CreateCeilLog2(Value * value, const Twine Name) {
    IntegerType * ty = cast<IntegerType>(value->getType());
    Value * m = CreateCountReverseZeroes(CreateSub(value, ConstantInt::get(ty, 1)));
    return CreateSub(ConstantInt::get(m->getType(), ty->getBitWidth()), m, Name);
}

Value * CBuilder::CreateLog2(Value * value, const Twine Name) {
    IntegerType * ty = cast<IntegerType>(value->getType());
    Value * m = CreateCountReverseZeroes(value);
    return CreateSub(ConstantInt::get(m->getType(), ty->getBitWidth() - 1), m, Name);
}

Constant * CBuilder::GetString(StringRef Str) {
    Module * const m = getModule(); assert (m);
    GlobalVariable * ptr = m->getGlobalVariable(Str, true);
    if (ptr == nullptr) {
        ptr = CreateGlobalString(Str, Str, 0, m);
    }
    return ConstantExpr::getPointerCast(ptr, getInt8PtrTy());
}

Value * CBuilder::CreateReadCycleCounter() {
    Module * const m = getModule();
    Function * cycleCountFunc = Intrinsic::getDeclaration(m, Intrinsic::readcyclecounter);
    return CreateCall(cycleCountFunc->getFunctionType(), cycleCountFunc, std::vector<Value *>({}));
}

Function * CBuilder::LinkFunction(StringRef name, FunctionType * type, void * functionPtr) const {
    assert (mDriver);
    return mDriver->addLinkFunction(getModule(), name, type, functionPtr);
}

#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
Value * CBuilder::CreateGEP(Type * Ty, Value * Ptr, ArrayRef<Value *> IdxList, const Twine & Name, bool IsInBounds) {
    assert (Ty->canLosslesslyBitCastTo(Ptr->getType()->getPointerElementType()));
    if (IsInBounds) {
        return IRBuilder<>::CreateInBoundsGEP(Ty, CreatePointerCast(Ptr, Ty->getPointerTo()), IdxList, Name);
    } else {
        return IRBuilder<>::CreateGEP(Ty, CreatePointerCast(Ptr, Ty->getPointerTo()), IdxList, Name);
    }
}
#endif

LoadInst * CBuilder::CreateLoad(Type * type, Value * Ptr, const char * Name) {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    assert (type->canLosslesslyBitCastTo(Ptr->getType()->getPointerElementType()));
    #endif
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr, getTypeSize(type), "CreateLoad");
    }
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
    return IRBuilder<>::CreateLoad(IRBuilder<>::CreatePointerCast(Ptr, type->getPointerTo()), Name);
    #else
    return IRBuilder<>::CreateLoad(type, Ptr, Name);
    #endif
}

LoadInst * CBuilder::CreateLoad(Type * type, Value *Ptr, const Twine Name) {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    assert (type->canLosslesslyBitCastTo(Ptr->getType()->getPointerElementType()));
    #endif
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr, getTypeSize(type), "CreateLoad");
    }
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
    return IRBuilder<>::CreateLoad(IRBuilder<>::CreatePointerCast(Ptr, type->getPointerTo()), Name);
    #else
    return IRBuilder<>::CreateLoad(type, Ptr, Name);
    #endif
}

LoadInst * CBuilder::CreateLoad(Type * type, Value * Ptr, bool isVolatile, const Twine Name) {
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    assert (type->canLosslesslyBitCastTo(Ptr->getType()->getPointerElementType()));
    #endif
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr, getTypeSize(type), "CreateLoad");
    }
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
    return IRBuilder<>::CreateLoad(IRBuilder<>::CreatePointerCast(Ptr, type->getPointerTo()), Name);
    #else
    return IRBuilder<>::CreateLoad(type, Ptr, isVolatile, Name);
    #endif
}

StoreInst * CBuilder::CreateStore(Value * Val, Value * Ptr, bool isVolatile) {
    assert ("Ptr (Arg2) was expected to be a pointer type" &&
            Ptr->getType()->isPointerTy());
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    assert (Val->getType()->canLosslesslyBitCastTo(Ptr->getType()->getPointerElementType()));
    #endif
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr, getTypeSize(Val->getType()), "CreateStore");
    }
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    Val = IRBuilder<>::CreateBitCast(Val, Ptr->getType()->getPointerElementType());
    #endif
    return IRBuilder<>::CreateStore(Val, Ptr, isVolatile);
}

inline bool CBuilder::hasAddressSanitizer() const {
    return mDriver && mDriver->hasExternalFunction("__asan_region_is_poisoned");
}

LoadInst * CBuilder::CreateAlignedLoad(Type * type, Value * Ptr, const unsigned Align, const char * Name) {
    assert (Align > 0);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
        ConstantInt * align = ConstantInt::get(intPtrTy, Align);
        Value * alignmentOffset = CreateURem(CreatePtrToInt(Ptr, intPtrTy), align);
        CreateAssertZero(alignmentOffset, "CreateAlignedLoad: pointer (%" PRIxsz ") is misaligned (%" PRIdsz ")", Ptr, align);
    }
    LoadInst * LI = CreateLoad(type, Ptr, Name);
    LI->setAlignment(AlignType{Align});
    return LI;
}

LoadInst * CBuilder::CreateAlignedLoad(Type * type, Value * Ptr, const unsigned Align, const Twine Name) {
    assert (Align > 0);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
        ConstantInt * align = ConstantInt::get(intPtrTy, Align);
        Value * alignmentOffset = CreateURem(CreatePtrToInt(Ptr, intPtrTy), align);
        CreateAssertZero(alignmentOffset, "CreateAlignedLoad: pointer " + Name + " (%" PRIxsz ") is misaligned (%" PRIdsz ")", Ptr, align);
    }
    LoadInst * LI = CreateLoad(type, Ptr, Name);
    LI->setAlignment(AlignType{Align});
    return LI;
}

LoadInst * CBuilder::CreateAlignedLoad(Type * type, Value * Ptr, const unsigned Align, bool isVolatile, const Twine Name) {
    assert (Align > 0);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
        ConstantInt * align = ConstantInt::get(intPtrTy, Align);
        Value * alignmentOffset = CreateURem(CreatePtrToInt(Ptr, intPtrTy), align);
        CreateAssertZero(alignmentOffset, "CreateAlignedLoad: pointer (%" PRIxsz ") is misaligned (%" PRIdsz ")", Ptr, align);
    }
    LoadInst * LI = CreateLoad(type, Ptr, isVolatile, Name);
    LI->setAlignment(AlignType{Align});
    return LI;
}

StoreInst * CBuilder::CreateAlignedStore(Value * Val, Value * Ptr, const unsigned Align, bool isVolatile) {
    assert (Align > 0);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
        ConstantInt * align = ConstantInt::get(intPtrTy, Align);
        Value * alignmentOffset = CreateURem(CreatePtrToInt(Ptr, intPtrTy), align);
        CreateAssertZero(alignmentOffset, "CreateAlignedStore: pointer (%" PRIxsz ") is misaligned (%" PRIdsz ")", Ptr, align);
    }
    StoreInst * SI = CreateStore(Val, Ptr, isVolatile);
    SI->setAlignment(AlignType{Align});
    return SI;
}

Value * CBuilder::CreateMemChr(Value * ptr, Value * byteVal, Value * num) {
    Module * const m = getModule();
    Function * memchrFn = m->getFunction("memchr");
    if (memchrFn == nullptr) {
        IntegerType * const int32Ty = getInt32Ty();
        IntegerType * const sizeTy = getSizeTy();
        PointerType * const voidPtrTy = getVoidPtrTy();
        FunctionType * memchrTy = FunctionType::get(voidPtrTy, {voidPtrTy, int32Ty, sizeTy}, false);
        memchrFn = Function::Create(memchrTy, Function::ExternalLinkage, "memchr", m);
    }
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(ptr, num, "CreateMemChr: Src");
    }
    return CreateCall(memchrFn->getFunctionType(), memchrFn, {ptr, byteVal, num});
}

CallInst * CBuilder::CreateMemMove(Value * Dst, Value * Src, Value *Size, const unsigned Align, bool isVolatile,
                                   MDNode *TBAATag, MDNode *ScopeTag, MDNode *NoAliasTag) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Src, Size, "CreateMemMove: Src");
        CheckAddress(Dst, Size, "CreateMemMove: Dst");
        // If the call to this intrinisic has an alignment value that is not 0 or 1, then the caller
        // guarantees that both the source and destination pointers are aligned to that boundary.
        if (Align > 1) {
            DataLayout DL(getModule());
            IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
            Value * intSrc = CreatePtrToInt(Src, intPtrTy);
            Value * intDst = CreatePtrToInt(Dst, intPtrTy);
            ConstantInt * align = ConstantInt::get(intPtrTy, Align);
            CreateAssertZero(CreateURem(intSrc, align), "CreateMemMove: Src pointer is misaligned");
            CreateAssertZero(CreateURem(intDst, align), "CreateMemMove: Dst pointer is misaligned");

        }
    }
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(7, 0, 0)
    return IRBuilder<>::CreateMemMove(Dst, AlignType{Align}, Src, AlignType{Align}, Size, isVolatile, TBAATag, ScopeTag, NoAliasTag);
#else
    return IRBuilder<>::CreateMemMove(Dst, Src, Size, AlignType{Align}, isVolatile, TBAATag, ScopeTag, NoAliasTag);
#endif
}

CallInst * CBuilder::CreateMemCpy(Value *Dst, Value *Src, Value *Size, const unsigned Align, bool isVolatile,
                                  MDNode *TBAATag, MDNode *TBAAStructTag, MDNode *ScopeTag, MDNode *NoAliasTag) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Src, Size, "CreateMemCpy: Src");
        CheckAddress(Dst, Size, "CreateMemCpy: Dst");
        DataLayout DL(getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
        Value * const intSrc = CreatePtrToInt(Src, intPtrTy);
        Value * const intDst = CreatePtrToInt(Dst, intPtrTy);
        // If the call to this intrinisic has an alignment value that is not 0 or 1, then the caller
        // guarantees that both the source and destination pointers are aligned to that boundary.
        if (Align > 1) {
            ConstantInt * align = ConstantInt::get(intPtrTy, Align);
            CreateAssertZero(CreateURem(intSrc, align), "CreateMemCpy: Src is misaligned");
            CreateAssertZero(CreateURem(intDst, align), "CreateMemCpy: Dst is misaligned");
        }
        Value * const intSize = CreateZExtOrTrunc(Size, intPtrTy);
        Value * const srcEndsBeforeDst = CreateICmpULE(CreateAdd(intSrc, intSize), intDst);
        Value * const dstEndsBeforeSrc = CreateICmpULE(CreateAdd(intDst, intSize), intSrc);
        Value * const nonOverlapping = CreateOr(srcEndsBeforeDst, dstEndsBeforeSrc);
        CreateAssert(nonOverlapping, "CreateMemCpy: overlapping ranges is undefined");
    }
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(7, 0, 0)
    return IRBuilder<>::CreateMemCpy(Dst, AlignType{Align}, Src, AlignType{Align}, Size, isVolatile, TBAATag, TBAAStructTag, ScopeTag, NoAliasTag);
#else
    return IRBuilder<>::CreateMemCpy(Dst, Src, Size, AlignType{Align}, isVolatile, TBAATag, TBAAStructTag, ScopeTag, NoAliasTag);
#endif
}

CallInst * CBuilder::CreateMemSet(Value * Ptr, Value * Val, Value * Size, const unsigned Align,
                       bool isVolatile, MDNode * TBAATag, MDNode * ScopeTag, MDNode * NoAliasTag) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr, Size, "CreateMemSet");
        if (Align > 1) {
            DataLayout DL(getModule());
            IntegerType * const intPtrTy = DL.getIntPtrType(getContext());
            Value * intPtr = CreatePtrToInt(Ptr, intPtrTy);
            ConstantInt * align = ConstantInt::get(intPtrTy, Align);
            CreateAssertZero(CreateURem(intPtr, align), "CreateMemSet: Ptr is misaligned");
        }
    }
    return IRBuilder<>::CreateMemSet(Ptr, Val, Size, AlignType{Align}, isVolatile, TBAATag, ScopeTag, NoAliasTag);
}

CallInst * CBuilder::CreateMemCmp(Value * Ptr1, Value * Ptr2, Value * Num) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        CheckAddress(Ptr1, Num, "CreateMemCmp: Ptr1");
        CheckAddress(Ptr2, Num, "CreateMemCmp: Ptr2");
    }
    Module * const m = getModule();
    Function * f = m->getFunction("memcmp");
    PointerType * const voidPtrTy = getVoidPtrTy();
    IntegerType * const sizeTy = getSizeTy();
    if (f == nullptr) {
        FunctionType * const fty = FunctionType::get(getInt32Ty(), {voidPtrTy, voidPtrTy, sizeTy}, false);
        f = Function::Create(fty, Function::ExternalLinkage, "memcmp", m);
        f->setCallingConv(CallingConv::C);
    }
    Ptr1 = CreatePointerCast(Ptr1, voidPtrTy);
    Ptr2 = CreatePointerCast(Ptr2, voidPtrTy);
    Num = CreateZExtOrTrunc(Num, sizeTy);
    return CreateCall(f->getFunctionType(), f, {Ptr1, Ptr2, Num});
}

AllocaInst * CBuilder::CreateAllocaAtEntryPoint(Type * Ty, Value * ArraySize, const Twine Name) {

    auto BB = GetInsertBlock();
    auto F = BB->getParent();
    auto entryBlock = F->begin();
    if (LLVM_UNLIKELY(entryBlock == F->end())) {
        report_fatal_error("CreateAllocaAtEntryPoint cannot create a value in an empty function");
    }
    const auto & DL = F->getParent()->getDataLayout();
    const auto addrSize = DL.getAllocaAddrSpace();
    auto const first = entryBlock->getFirstNonPHIOrDbgOrLifetime();
    AllocaInst * alloca = nullptr;
    if (LLVM_UNLIKELY(first == nullptr)) {
        alloca = new AllocaInst(Ty, addrSize, ArraySize, Name, &*entryBlock);
    } else {
        alloca = new AllocaInst(Ty, addrSize, ArraySize, Name, first);
    }
    return alloca;
}


AllocaInst * CBuilder::CreateAlignedAlloca(Type * const Ty, const unsigned Align, Value * const ArraySize) {
    AllocaInst * const alloca = IRBuilder<>::CreateAlloca(Ty, ArraySize);
    alloca->setAlignment(AlignType{Align});
    return alloca;
}

AllocaInst * CBuilder::CreateAlignedAllocaAtEntryPoint(llvm::Type * const Ty, const unsigned alignment, llvm::Value * const ArraySize) {
    auto BB = GetInsertBlock();
    auto F = BB->getParent();
    auto entryBlock = F->begin();
    if (LLVM_UNLIKELY(entryBlock == F->end())) {
        report_fatal_error("CreateAllocaAtEntryPoint cannot create a value in an empty function");
    }
    const auto & DL = F->getParent()->getDataLayout();
    const auto addrSize = DL.getAllocaAddrSpace();
    auto const first = entryBlock->getFirstNonPHIOrDbgOrLifetime();
    AllocaInst * alloca = nullptr;
    if (LLVM_UNLIKELY(first == nullptr)) {
        alloca = new AllocaInst(Ty, addrSize, ArraySize, "", &*entryBlock);
    } else {
        alloca = new AllocaInst(Ty, addrSize, ArraySize, "", first);
    }
    alloca->setAlignment(AlignType{alignment});
    return alloca;
}

Value * CBuilder::CreateExtractElement(Value * Vec, Value *Idx, const Twine Name) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        if (LLVM_UNLIKELY(!isa<FixedVectorType>(Vec->getType()))) {
            report_fatal_error("CreateExtractElement: Vec argument is not a vector type");
        }
        const auto n = cast<FixedVectorType>(Vec->getType())->getNumElements();
        Constant * const Size = ConstantInt::get(Idx->getType(), n);
        // exctracting an element from a position that exceeds the length of the vector is undefined
        __CreateAssert(CreateICmpULT(Idx, Size), "CreateExtractElement: Idx (%" PRIdsz ") is greater than Vec size (%" PRIdsz ")", { Idx, Size });
    }
    return IRBuilder<>::CreateExtractElement(Vec, Idx, Name);
}

Value * CBuilder::CreateInsertElement(Value * Vec, Value * NewElt, Value * Idx, const Twine Name) {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        if (LLVM_UNLIKELY(!isa<FixedVectorType>(Vec->getType()))) {
            report_fatal_error("CreateInsertElement: Vec argument is not a vector type");
        }
        const auto n = cast<FixedVectorType>(Vec->getType())->getNumElements();
        Constant * const Size = ConstantInt::get(Idx->getType(), n);
        // inserting an element into a position that exceeds the length of the vector is undefined
        __CreateAssert(CreateICmpULT(Idx, Size), "CreateInsertElement: Idx (%" PRIdsz ") is greater than Vec size (%" PRIdsz ")", { Idx, Size });
    }
    return IRBuilder<>::CreateInsertElement(Vec, NewElt, Idx, Name);
}

CallInst * CBuilder::CreateSRandCall(Value * randomSeed) {
    Module * const m = getModule();
    Function * srandFunc = m->getFunction("srand");
    if (srandFunc == nullptr) {
        FunctionType * fty = FunctionType::get(getVoidTy(), {getInt32Ty()}, false);
        srandFunc = Function::Create(fty, Function::ExternalLinkage, "srand", m);
        srandFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(srandFunc->getFunctionType(), srandFunc, randomSeed);
}

CallInst * CBuilder::CreateRandCall() {
    Module * const m = getModule();
    Function * randFunc = m->getFunction("rand");
    if (randFunc == nullptr) {
        FunctionType * fty = FunctionType::get(getInt32Ty(), false);
        randFunc = Function::Create(fty, Function::ExternalLinkage, "rand", m);
        randFunc->setCallingConv(CallingConv::C);
    }
    return CreateCall(randFunc->getFunctionType(), randFunc, {});
}

unsigned CBuilder::getPageSize() {
    return boost::interprocess::mapped_region::get_page_size();
}

BasicBlock * CBuilder::WriteDefaultRethrowBlock() {

    const auto ip = saveIP();

    BasicBlock * const current = GetInsertBlock();
    Function * const f = current->getParent();

    f->setPersonalityFn(getDefaultPersonalityFunction());
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    f->setHasUWTable();
    #else
    f->setUWTableKind(UWTableKind::Default);
    #endif

    LLVMContext & C = getContext();

    BasicBlock * const handleCatch = BasicBlock::Create(C, "__catch", f);
    BasicBlock * const handleRethrow = BasicBlock::Create(C, "__rethrow", f);
    BasicBlock * const handleResume = BasicBlock::Create(C, "__resume", f);
    BasicBlock * const handleExit = BasicBlock::Create(C, "__exit", f);
    BasicBlock * const handleUnreachable = BasicBlock::Create(C, "__unreachable", f);

    PointerType * const int8PtrTy = getInt8PtrTy();
    IntegerType * const int32Ty = getInt32Ty();
    StructType * const caughtResultType = StructType::get(C, { int8PtrTy, int32Ty });
    Constant * const catchAny = ConstantPointerNull::get(int8PtrTy);

    SetInsertPoint(handleCatch);
    LandingPadInst * const caughtResult = CreateLandingPad(caughtResultType, 1);
    caughtResult->addClause(catchAny);
    caughtResult->setCleanup(false);
    Value * const exception = CreateExtractValue(caughtResult, 0);
    Function * catchFn = getBeginCatch();
    FunctionType * catchTy = catchFn->getFunctionType();
    CallInst * beginCatch = CreateCall(catchTy, catchFn, {exception});
    beginCatch->setTailCall(true);
    InvokeInst * const rethrowInst = CreateInvoke(getRethrow(), handleUnreachable, handleRethrow);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
    beginCatch->addAttribute(-1, Attribute::NoUnwind);
    rethrowInst->addAttribute(-1, Attribute::NoReturn);
#else
    beginCatch->setDoesNotThrow();
    rethrowInst->setDoesNotReturn();
#endif

    SetInsertPoint(handleRethrow);
    LandingPadInst * const caughtResult2 = CreateLandingPad(caughtResultType, 1);
    caughtResult2->setCleanup(true);
    CreateInvoke(getEndCatch(), handleResume, handleExit);

    SetInsertPoint(handleResume);
    CreateResume(caughtResult2);

    SetInsertPoint(handleExit);
    LandingPadInst * const caughtResult3 = CreateLandingPad(caughtResultType, 1);
    caughtResult3->addClause(catchAny);
    Value * const exception3 = CreateExtractValue(caughtResult3, 0);
    CallInst * beginCatch2 = CreateCall(catchTy, catchFn, {exception3});
    beginCatch2->setTailCall(true);
#if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(14, 0, 0)
    beginCatch2->addAttribute(-1, Attribute::NoUnwind);
#else
    beginCatch2->setDoesNotThrow();
#endif
    // should call std::terminate
    CreateExit(-1);
    CreateBr(handleUnreachable);

    SetInsertPoint(handleUnreachable);
    CreateUnreachable();

    restoreIP(ip);

    return handleCatch;
}

Function * CBuilder::getDefaultPersonalityFunction() {
    const auto GNU_NAME = "__gxx_personality_v0";
    Module * const m = getModule();
    Function * personalityFn = m->getFunction(GNU_NAME);
    if (personalityFn == nullptr) {
        FunctionType * const fTy = FunctionType::get(getInt32Ty(), true);
        personalityFn = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
    }
    return personalityFn;
}

Function * CBuilder::getAllocateException() {
    const auto GNU_NAME = "__cxa_allocate_exception";
    Module * const m = getModule();
    Function * cxa_alloc_excp = m->getFunction(GNU_NAME);
    if (cxa_alloc_excp == nullptr) {
        PointerType * const int8PtrTy = getInt8PtrTy();
        IntegerType * const int64Ty = getInt64Ty();
        FunctionType * const fTy = FunctionType::get(int8PtrTy, { int64Ty }, false);
        cxa_alloc_excp = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
        cxa_alloc_excp->addFnAttr(Attribute::NoUnwind);
    }
    return cxa_alloc_excp;
}

Function * CBuilder::getThrow() {
    const auto GNU_NAME = "__cxa_throw";
    Module * const m = getModule();
    Function * cxa_throw = m->getFunction(GNU_NAME);
    if (cxa_throw == nullptr) {
        Type * const voidTy = getVoidTy();
        PointerType * const int8PtrTy = getInt8PtrTy();
        FunctionType * const fTy = FunctionType::get(voidTy, { int8PtrTy, int8PtrTy, int8PtrTy }, false);
        cxa_throw = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
        cxa_throw->setDoesNotReturn();
    }
    return cxa_throw;
}

Function * CBuilder::getBeginCatch() {
    const auto GNU_NAME = "__cxa_begin_catch";
    Module * const m = getModule();
    Function * cxa_begin = m->getFunction(GNU_NAME);
    if (cxa_begin == nullptr) {
        PointerType * const int8PtrTy = getInt8PtrTy();
        FunctionType * const fTy = FunctionType::get(int8PtrTy, { int8PtrTy }, false);
        cxa_begin = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
        cxa_begin->setDoesNotThrow();
        cxa_begin->setDoesNotRecurse();
    }
    return cxa_begin;
}

Function * CBuilder::getEndCatch() {
    const auto GNU_NAME = "__cxa_end_catch";
    Module * const m = getModule();
    Function * cxa_end = m->getFunction(GNU_NAME);
    if (cxa_end == nullptr) {
        FunctionType * const fTy = FunctionType::get(getVoidTy(), false);
        cxa_end = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
    }
    return cxa_end;
}

Function * CBuilder::getRethrow() {
    const auto GNU_NAME = "__cxa_rethrow";
    Module * const m = getModule();
    Function * cxa_rethrow = m->getFunction(GNU_NAME);
    if (cxa_rethrow == nullptr) {
        FunctionType * const fTy = FunctionType::get(getVoidTy(), false);
        cxa_rethrow = Function::Create(fTy, Function::ExternalLinkage, GNU_NAME, m);
        cxa_rethrow->setDoesNotReturn();
    }
    return cxa_rethrow;
}

AllocaInst * CBuilder::resolveStackAddress(Value * Ptr) {
    for (;;) {
        if (GetElementPtrInst * gep = dyn_cast<GetElementPtrInst>(Ptr)) {
            Ptr = gep->getPointerOperand();
        } else if (CastInst * ci = dyn_cast<CastInst>(Ptr)) {
            Ptr = ci->getOperand(0);
        } else {
            return dyn_cast<AllocaInst>(Ptr);
        }
    }
}

static inline bool notConstantZeroArraySize(const AllocaInst * const Base) {
    if (const Constant * const as = dyn_cast_or_null<Constant>(Base->getArraySize())) {
        return !as->isNullValue();
    }
    return false;
}


void CBuilder::CheckAddress(Value * const Ptr, Value * const Size, Constant * const Name) {

    __CreateAssert(CreateOr(CreateIsNotNull(Ptr), CreateIsNull(Size)), "%s was given a null address", {Name});


    if (AllocaInst * Base = resolveStackAddress(Ptr)) {
        IntegerType * const intPtrTy = getIntPtrTy(getModule()->getDataLayout());

        Value * sz = getTypeSize(Base->getAllocatedType(), intPtrTy);
        if (notConstantZeroArraySize(Base)) {
            sz = CreateMul(sz, CreateZExtOrTrunc(Base->getArraySize(), intPtrTy));
        }
        Value * const p = CreatePtrToInt(Ptr, intPtrTy);
        Value * const s = CreatePtrToInt(Base, intPtrTy);
        Value * const w = CreateAdd(p, CreateZExtOrTrunc(Size, intPtrTy));
        Value * const e = CreateAdd(s, sz);
        Value * const valid = CreateAnd(CreateICmpUGE(p, s), CreateICmpULE(w, e));
        __CreateAssert(valid, "%s was given an invalid stack address (%" PRIx64 ")", {Name, p});
    }

    #ifdef HAS_ADDRESS_SANITIZER
//    if (LLVM_UNLIKELY(hasAddressSanitizer())) {
        Module * const m = getModule();
        PointerType * const voidPtrTy = getVoidPtrTy();
        IntegerType * const sizeTy = getSizeTy();
        Function * isPoisoned = m->getFunction("__asan_region_is_poisoned");
        if (LLVM_UNLIKELY(isPoisoned == nullptr)) {
            FunctionType * const funcTy = FunctionType::get(voidPtrTy, {voidPtrTy, sizeTy}, false);
            isPoisoned = LinkFunction( "__asan_region_is_poisoned", funcTy, (void*)__asan_region_is_poisoned);
            isPoisoned->setCallingConv(CallingConv::C);
            isPoisoned->setReturnDoesNotAlias();
        }
        Value * const addr = CreatePointerCast(Ptr, voidPtrTy);
        Value * const firstPoisoned = CreateCall(isPoisoned->getFunctionType(), isPoisoned, { addr, CreateTrunc(Size, sizeTy) });
        Value * const valid = CreateICmpEQ(firstPoisoned, ConstantPointerNull::get(voidPtrTy));
        IntegerType * const intPtrTy = getIntPtrTy(getModule()->getDataLayout());
        Value * const startInt = CreatePtrToInt(Ptr, intPtrTy);
        Value * const firstPoisonedInt = CreatePtrToInt(firstPoisoned, intPtrTy);
        Value * const offset = CreateSub(firstPoisonedInt, startInt);
        __CreateAssert(valid, "%s was given an unallocated %" PRIuMAX "-byte memory address 0x%" PRIxPTR " (first poisoned=%" PRIuMAX ")", {Name, Size, Ptr, offset});
//    }
    #endif
}

#ifdef ENABLE_LIBBACKTRACE
extern "C"
BOOST_NOINLINE
int __backtrace_check_valid_return_callback(void * data, uintptr_t, const char *filename, int lineno, const char *function) {
    if (lineno == 0 || *filename == '\0') {
        *reinterpret_cast<bool*>(data) = true;
    }
    return 0;
}

extern "C"
BOOST_NOINLINE
void __backtrace_set_true_on_error_callback(void *data, const char *msg, int errnum) {
    *reinterpret_cast<bool*>(data) = true;
}
#endif

CBuilder::CBuilder(LLVMContext & C)
: IRBuilder<>(C)
, mCacheLineAlignment(64)
, mSizeType(IntegerType::get(getContext(), sizeof(size_t) * 8))
, mFILEtype(nullptr)
, mDriver(nullptr) {
    #ifdef ENABLE_LIBBACKTRACE
    if (LLVM_UNLIKELY(codegen::AnyAssertionOptionIsSet())) {
        auto p = boost::filesystem::absolute(codegen::ProgramName).lexically_normal().native();
        bool error = false;
        mBacktraceState = backtrace_create_state(p.c_str(), 0, __backtrace_set_true_on_error_callback, &error);
        if (error) {
            mBacktraceState = nullptr;
        }
    } else {
        mBacktraceState = nullptr;
    }
    #endif
}

struct RemoveRedundantAssertionsPass : public ModulePass {
    static char ID;
    RemoveRedundantAssertionsPass() : ModulePass(ID) { }

    void getAnalysisUsage(AnalysisUsage & AU) const override {
        #ifdef HAS_ADDRESS_SANITIZER
        AU.addRequired<DominatorTreeWrapperPass>();
        AU.addRequired<AAResultsWrapperPass>();
        AU.addPreserved<AAResultsWrapperPass>();
        #endif
    }

    virtual bool runOnModule(Module &M) override;
};

ModulePass * createRemoveRedundantAssertionsPass() {
    return new RemoveRedundantAssertionsPass();
}

char RemoveRedundantAssertionsPass::ID = 0;

bool RemoveRedundantAssertionsPass::runOnModule(Module & M) {

    Function * const assertFunc = M.getFunction("assert");
    if (LLVM_UNLIKELY(assertFunc == nullptr)) {
        return false;
    }
    bool modified = false;
    DenseSet<Value *> assertions;
    bool discoveredStaticFailure = false;

    #ifdef HAS_ADDRESS_SANITIZER
    Function * const isPoisoned = M.getFunction("__asan_region_is_poisoned");
    DenseSet<CallInst *> isPoisonedCalls;
    #endif

    for (Function & F : M) {

        // ignore declarations
        if (F.empty()) continue;

        #ifdef HAS_ADDRESS_SANITIZER
        DominatorTree & DT = getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
        AAResults & AA = getAnalysis<AAResultsWrapperPass>(F).getAAResults();
        isPoisonedCalls.clear();
        #endif

        assertions.clear();
        // scan through each instruction and remove any trivially true or redundant assertions.
        for (auto & B : F) { 
            for (auto i = B.begin(); i != B.end(); ) {
                Instruction & inst = *i;
                if (LLVM_UNLIKELY(isa<CallInst>(inst))) {
                    CallInst & ci = cast<CallInst>(inst);
                    auto isIndirectCall = [&]() {
                        return ci.isIndirectCall();
                    };
                    if (!(ci.getCalledFunction() || isIndirectCall())) {
                        auto & out = llvm::errs();
                        B.print(out);
                        errs() << "\n\n";
                        ci.print(out);
                    }

                    assert ("null pointer for function call?" && (ci.getCalledFunction() || isIndirectCall()));
                    // if we're using address sanitizer, try to determine whether we're
                    // rechecking the same address
                    if (isIndirectCall()) {
                        /* do nothing */
                    }
                    #ifdef HAS_ADDRESS_SANITIZER
                    else if (ci.getCalledFunction() == isPoisoned) {
                        bool alreadyProven = false;
                        // To support non-constant lengths, we'd need a way of proving whether one
                        // length is (symbolically) always less than or equal to another or v.v..
                        ConstantInt * const length = dyn_cast<ConstantInt>(ci.getArgOperand(1));
                        if (length) {
                            Value * const ptr = ci.getArgOperand(0);
                            for (CallInst * prior : isPoisonedCalls) {
                                if (DT.dominates(prior, &ci)) {
                                    Value * const priorPtr = prior->getArgOperand(0);
                                    ConstantInt * const priorLength = cast<ConstantInt>(prior->getArgOperand(1));
                                    const auto result = AA.alias(ptr, length->getLimitedValue(), priorPtr, priorLength->getLimitedValue());
                                    if (LLVM_UNLIKELY(result == AliasResult::MustAlias)) {
                                        alreadyProven = true;
                                        break;
                                    }
                                }
                            }
                            if (alreadyProven) {
                                // __asan_region_is_poisoned returns the address of first poisoned
                                // byte to indicate failure. Replace any dominated checks with null
                                // to indicate success.
                                PointerType * const voidPtrTy = cast<PointerType>(isPoisoned->getReturnType());
                                ci.replaceAllUsesWith(ConstantPointerNull::get(voidPtrTy));
                                modified = true;
                            } else {
                                isPoisonedCalls.insert(&ci);
                            }
                        }
                    }
                    #endif
                    else if (ci.getCalledFunction() == assertFunc) {
                        assert (ci.getNumOperands() >= 5);
                        bool remove = false;
                        Value * const check = ci.getOperand(0);
                        Constant * static_check = nullptr;
                        if (isa<Constant>(check)) {
                            static_check = cast<Constant>(check);
                        } else if (LLVM_LIKELY(isa<ICmpInst>(check))) {
                            ICmpInst * const icmp = cast<ICmpInst>(check);
                            Value * const op0 = icmp->getOperand(0);
                            Value * const op1 = icmp->getOperand(1);
                            if (LLVM_UNLIKELY(isa<Constant>(op0) && isa<Constant>(op1))) {
                                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(19, 0, 0)
                                static_check = ConstantExpr::getICmp(icmp->getPredicate(), cast<Constant>(op0), cast<Constant>(op1));
                                #else
                                static_check = ConstantFoldCompareInstruction(icmp->getPredicate(), cast<Constant>(op0), cast<Constant>(op1));
                                #endif
                                ci.setOperand(0, static_check);
                            }
                        }
                        if (static_check) {
                            if (LLVM_UNLIKELY(static_check->isNullValue())) {
                                // show any static failures with their compilation context
                                auto extract = [&](const Value * a) -> const char * {
                                    assert (a);
                                    a = a->stripPointerCasts();
                                    if (LLVM_LIKELY(isa<GlobalVariable>(a))) {
                                        const GlobalVariable * const g = cast<GlobalVariable>(a);
                                        const Constant * const gi = g->getInitializer(); assert (gi);
                                        if (LLVM_LIKELY(isa<ConstantDataArray>(gi))) {
                                            const ConstantDataArray * const d = cast<ConstantDataArray>(gi);
                                            // StringRef does not "own" the data
                                            const StringRef ref = d->getRawDataValues();
                                            return ref.data();
                                        } else {
                                            errs() << "WARNING: \"";
                                            ci.print(errs());
                                            errs() << "\" has an unexpected trace log format\n";
                                            return nullptr;
                                        }
                                    } else {
                                        assert (isa<ConstantPointerNull>(a) || isa<Function>(a));    
                                    }
                                    return nullptr;
                                };

                                const char * const name = extract(ci.getArgOperand(1)); assert (name);
                                const char * const msg = extract(ci.getArgOperand(2)); assert (msg);
                                const auto trace = reinterpret_cast<const __backtrace_data * const *>(extract(ci.getArgOperand(3)));
                                const uint32_t n = trace ? cast<ConstantInt>(ci.getOperand(4))->getLimitedValue() : 0UL;

                                // since we may not necessarily be able to statically evaluate every varadic param,
                                // attempt to fill in what constants we can and report <any> for all others.
                                boost::format fmt(msg);
                                for (unsigned i = 5; i < ci.getNumOperands(); ++i) {
                                    Value * const arg = ci.getOperand(i); assert (arg);
                                    if (LLVM_LIKELY(isa<Constant>(arg))) {
                                        if (LLVM_LIKELY(isa<ConstantInt>(arg))) {
                                            const auto v = cast<ConstantInt>(arg)->getLimitedValue();
                                            fmt % v;
                                        } else if (isa<ConstantFP>(arg)) {
                                            const auto & f = cast<ConstantFP>(arg)->getValueAPF();
                                            fmt % f.convertToDouble();
                                        } else {
                                            Type * const ty = arg->getType();
                                            if (ty->isPointerTy()) {
                                                const char * const argC = extract(arg);
                                                if (argC) {
                                                    fmt % argC;
                                                }
                                            } else {
                                                fmt % "<unknown constant type>";
                                            }
                                        }
                                    } else {
                                        fmt % "<any value>";
                                    }
                                }

                                F.print(errs()); errs() << "\n\n";

                                SmallVector<char, 1024> tmp;
                                raw_svector_ostream out(tmp);
                                out << "STATIC FAILURE: " << fmt.str();
                                __report_failure(name, out.str().data(), trace, n);
                                discoveredStaticFailure = true;
                            } else {
                                remove = true;
                            }
                        } else { // non-static check

                            // a duplicate check will never be executed if true
                            if (assertions.count(check)) {
                                remove = true;
                            } else if (isa<PHINode>(check)) {
                                // if all incoming values of this PHINode have been checked,
                                // then this is an implicit duplicate check.
                                PHINode * const phi = cast<PHINode>(check);
                                const auto n = phi->getNumIncomingValues();
                                bool allChecked = true;
                                for (auto i = 0U; i != n; ++i) {
                                    if (assertions.count(phi->getIncomingValue(i)) == 0) {
                                        allChecked = false;
                                        break;
                                    }
                                }
                                if (allChecked) {
                                    assertions.insert(check);
                                    remove = true;
                                }
                            }
                        }

                        if (remove) {
                            i = ci.eraseFromParent();
                            RecursivelyDeleteTriviallyDeadInstructions(check);
                            modified = true;
                            continue;
                        } else {
                            assertions.insert(check);
                        }

                    }
                }
                ++i;
            }
        }

        // if we have any runtime assertions, replace the calls to each with an invoke.
        if (LLVM_UNLIKELY(assertions.empty())) continue;

        // Gather all assertions function calls from the remaining conditions
        SmallVector<CallInst *, 64> assertList;
        assertList.reserve(assertions.size());
        for (Value * s : assertions) {
            // It's possible (but unlikely) that the assertion condition is
            // used for some purpose other than the assertion call; scan
            // through and find each. We know, however, there must be exactly
            // one assertion call for each condition.
            for (User * u : s->users()) {
                if (isa<CallInst>(u)) {
                    CallInst * const c = cast<CallInst>(u);
                    if (LLVM_LIKELY(c->getCalledFunction() == assertFunc)) {
                        assertList.push_back(c);
                        break;
                    }
                }
            }
        }
        assert (assertList.size() == assertions.size());
        assertions.clear();


        // Replace the assertion function with a "try/throw" block
        CBuilder builder(M.getContext());
        builder.setModule(&M);
        builder.SetInsertPoint(&F.front());
        BasicBlock * const rethrow = builder.WriteDefaultRethrowBlock();

        SmallVector<Value *, 16> args;

        for (CallInst * ci : assertList) {
            BasicBlock * const bb = ci->getParent();
            const auto n = ci->getNumOperands();
            assert (n >= 5);
            args.clear();
            for (unsigned i = 0; i < n; ++i) {
                args.push_back(ci->getOperand(i));
            }
            auto next = ci->eraseFromParent();
            // note: split automatically inserts an unconditional branch to the new block
            BasicBlock * const assertFinally = bb->splitBasicBlock(next, "__assert_ok");
            assert(dyn_cast_or_null<BranchInst>(bb->getTerminator()));
            bb->getTerminator()->eraseFromParent();
            builder.SetInsertPoint(bb);
            builder.CreateInvoke(assertFunc, assertFinally, rethrow, args);
            assert(dyn_cast_or_null<InvokeInst>(bb->getTerminator()));
        }
    }

    if (LLVM_UNLIKELY(discoveredStaticFailure)) {
        exit(-1);
    }

    return modified;
}

ConstantInt * LLVM_READNONE CBuilder::getTypeSize(Type * type, IntegerType * valType) const {
    // ConstantExpr::getSizeOf was creating an infinite(?) loop when folding the value for some complex structs
    // until replaced with this in LLVM 12.
    const DataLayout & dl = getModule()->getDataLayout();
    if (valType == nullptr) {
        valType = getSizeTy();
    }
    return ConstantInt::get(valType, getTypeSize(dl, type));
}

uintptr_t LLVM_READNONE CBuilder::getTypeSize(const llvm::DataLayout & DL, llvm::Type * type) {
    uintptr_t size = 0;
    if (LLVM_LIKELY(type != nullptr)) {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
        size = DL.getTypeAllocSize(type);
        #elif LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(16, 0, 0)
        size = DL.getTypeAllocSize(type).getFixedSize();
        #else
        size = DL.getTypeAllocSize(type).getFixedValue();
        #endif
    }
    return size;
}

void CBuilder::linkAllNecessaryExternalFunctions() const {
    assert (mDriver);
    assert (mModule);
    // void* aligned_alloc( std::size_t alignment, std::size_t size );

    IntegerType * const sizeTy = getSizeTy();

    FixedArray<Type *, 2> params;
    params[0] = sizeTy;
    params[1] = sizeTy;
    FunctionType * fty = FunctionType::get(getVoidPtrTy(), params, false);
    mDriver->addLinkFunction(mModule, ALIGNED_ALLOC_NAME, fty, (void*)std::aligned_alloc);

}

std::string CBuilder::getKernelName() const {
    return "cbuilder";
}
