/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>               // for StdOutKernel_
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <llvm/ADT/StringRef.h>
#include <kernel/pipeline/program_builder.h>
#include <fcntl.h>
#include <boost/intrusive/detail/math.hpp>
#include <llvm/Support/raw_ostream.h>  // For debugging output

using boost::intrusive::detail::floor_log2;
#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory PackDemoOptions("Pack Demo Options", "Pack demo options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(PackDemoOptions));

enum class PackOption {packh, packl};
class PackKernel final : public MultiBlockKernel {
public:
    PackKernel(LLVMTypeSystemInterface & ts,
              StreamSet * const i16Stream,
              Scalar * Char);
protected:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
private:
    PackOption mOption;
};

std::string packOptionString(PackOption opt) {
    if (opt == PackOption::packh) return "packh";
    return "packl";
}

PackKernel::PackKernel(LLVMTypeSystemInterface & ts, StreamSet * const i16Stream, Scalar * Char) //to make sure the code isn't delected, scaler inside the kernel state, to makse sure teh code isn't deleted
: MultiBlockKernel(ts, "Lemier",
{Binding{"i16Stream", i16Stream}},
                   {}, {}, {Binding{"Char", Char}}, {}) {}



const uint8_t lookup_table[16] = {
        0, 0, 0, 0, 0, 0, 0x26, 0,
        0, 0, 0, 0, 0x3c, 0xd, 0, 0
    };
// Table lookup
// Perform SIMD-based lookup for multiple values at once
//llvm::Value* lookup(KernelBuilder & b, llvm::Value* val_a) {
//    
//    // Define lookup values (only nonzero ones)
//    const uint8_t keys[3] = { 0x26, 0x3C, 0x0D };
//
//    // Load key values into SIMD vectors
//    llvm::Value* key_vec1 = b.simd_fill(8, b.getInt8(keys[0]));
//    //b.CallPrintRegister("a", key_vec1);
//    llvm::Value* key_vec2 = b.simd_fill(8, b.getInt8(keys[1]));
//    //b.CallPrintRegister("b", key_vec2);
//    llvm::Value* key_vec3 = b.simd_fill(8, b.getInt8(keys[2]));
//    //b.CallPrintRegister("c", key_vec3);
//
//    // Compare input vector with all key vectors simultaneously
//    llvm::Value* match1 = b.simd_eq(8, val_a, key_vec1);
//    //b.CallPrintRegister("d", match1);
//    llvm::Value* match2 = b.simd_eq(8, val_a, key_vec2);
//    //b.CallPrintRegister("e", match2);
//    llvm::Value* match3 = b.simd_eq(8, val_a, key_vec3);
//    //b.CallPrintRegister("f", match3);
//
//    // Combine all matches using bitwise OR
//    llvm::Value* match_result = b.simd_or(match1, b.simd_or(match2, match3));
//
//    return match_result;  // Return final match result
//}


// Function to generate the lookup logic in SIMD
//llvm::Value* generate_neon_lookup_logic(KernelBuilder &b, llvm::Value* const numOfStrides) {
//    
//    // Load input vector
//    llvm::Value* val_a = b.loadInputStreamPack("i8Stream", b.getSize(0), numOfStrides);
//
//    // Perform lookup in one step (fully parallel)
//    llvm::Value* result = lookup(b, val_a);
//
//    return result; // Return the final match results
//}

//Generic form of the lookup table

llvm::Value* lookup(KernelBuilder & b, llvm::Value* val_a, const std::vector<uint8_t>& keys) {
    if (keys.empty()) {
        return b.getInt8(0); // Return 0 if there are no keys to compare
    }

    llvm::Value* match_result = nullptr;

    // Iterate through all keys
    for (size_t i = 0; i < keys.size(); ++i) {
        
        // Load key into SIMD vector
        llvm::Value* key_vec = b.simd_fill(8, b.getInt8(keys[i]));
        
        // Compare input vector with current keys
        llvm::Value* match = b.simd_eq(8, val_a, key_vec);

        // total matches
        if (match_result) {
            match_result = b.simd_or(match_result, match);
        } else {
            match_result = match; // First match assignment
        }
    }

    return match_result;
}


// Pairwise addition

llvm::Value * simd_pairwise_sum(KernelBuilder & b, unsigned fw, llvm::Value * Val_a, llvm::Value * Val_b) {

        
    
        // Extract upper 16 bits
        llvm::Value * highA = b.simd_srli(2*fw, Val_a, fw);  // fw to make it work for any power of 2**k = fw shift -16 bits , fw = 16
        llvm::Value * highB = b.simd_srli(2*fw, Val_b, fw);

        // Mask with all elements set to 0xFFFF
        llvm::Value * mask = b.simd_fill(2*fw, b.getIntN(2*fw, (1ULL<<fw) - 1ULL));  // 0xFFFF = 16 - 1s
        
      
        // Mask lower 16 bits
        llvm::Value * lowA = b.simd_and(Val_a, mask);
        llvm::Value * lowB = b.simd_and(Val_b, mask);

        // Sum the upper and lower 16-bit parts for pairwise sum
        llvm::Value * sumA = b.simd_add(2*fw, lowA, highA);
        llvm::Value * sumB = b.simd_add(2*fw, lowB, highB);

        llvm::Value * result = b.hsimd_packl(2*fw, sumA, sumB);  // hsimd_packl - takes the lower half and packs them into a single vector
      
        return result;
    }


//packed pairwise sum
//
llvm::Value * simd_packed_pairwise_sum(KernelBuilder & b, llvm::Value * matches1, llvm::Value * matches2, llvm::Value * matches3, llvm::Value * matches4) {
    // Define bit mask (same as bit_mask in the provided NEON code)
    const uint8_t bit_mask_values[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                         0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
    llvm::Value* bit_mask = b.simd_fill(8, b.getInt8(0));
    for (int i = 0; i < 16; i++) {
        bit_mask = b.CreateInsertElement(bit_mask, b.getInt8(bit_mask_values[i]), i);
    }

    // Apply bit mask to matches
    llvm::Value *masked1 = b.simd_and(matches1, bit_mask);
    llvm::Value *masked2 = b.simd_and(matches2, bit_mask);
    llvm::Value *masked3 = b.simd_and(matches3, bit_mask);
    llvm::Value *masked4 = b.simd_and(matches4, bit_mask);
    
    // Sum the intermediate results
    // llvm::Value *sum_final = simd_pairwise_sum(b, fw, sum1, sum2);

    // Perform pairwise sums
    llvm::Value *sum0 = b.hsimd_pairwisesum(8, masked1, masked2);
    llvm::Value *sum1 = b.hsimd_pairwisesum(8, masked3, masked4);
    sum0 = b.hsimd_pairwisesum(8, sum0, sum1);
    sum0 = b.hsimd_pairwisesum(8, sum0, sum0);  // Equivalent to the next level of pairwise summation
    
    sum0 = b.fwCast(64, sum0);
    // Convert to 64-bit value by extracting lane
    //llvm::Value *matches_64 = b.bitCast(sum0, llvm::Type::getInt64Ty(b.getContext()));
    Value * matches_64 = b.CreateExtractElement(sum0, b.getInt32(0));
    
    return matches_64;
}
llvm::Value* lookup(KernelBuilder & b, llvm::Value* val_a) {
    
    // Define lookup table: 16 values where only specific positions contain valid keys
    const uint8_t keys[16] = { 0, 0, 0, 0, 0, 0, 0x26, 0, 0, 0, 0, 0, 0x3C, 0x0D, 0, 0 };

    // Load the keys into a constant SIMD vector
    std::vector<llvm::Constant*> key_constants;
    for (size_t i = 0; i < 16; ++i) {
        key_constants.push_back(llvm::ConstantInt::get(b.getInt8Ty(), keys[i]));
    }
    llvm::Value* key_vector = llvm::ConstantVector::get(key_constants);

    // Create a mask vector 0x0F to extract low 4 bits from input
    llvm::Value* v0f = b.simd_fill(16, b.getInt8(0x0F));

    // AND input vector with 0x0F to get index_vector
    llvm::Value* index_vector = b.simd_and(val_a, v0f);

    // DISA's mvmd_shuffle to simulate table lookup
    llvm::Value* shuffled_keys = b.mvmd_shuffle(8, key_vector, index_vector);

    // Compare input values with looked-up keys
    llvm::Value* match_result = b.simd_eq(8, val_a, shuffled_keys);

    return match_result; // Final matches
}





void PackKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    
    
    const unsigned fw = 8;
    const unsigned inputPacksPerStride = 2*fw;
    const unsigned outputPacksPerStride = fw;

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * packLoop = b.CreateBasicBlock("packLoop");
    BasicBlock * packFinalize = b.CreateBasicBlock("packFinalize");
    Constant * const ZERO = b.getSize(0);
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(floor_log2(getStride()/b.getBitBlockWidth())));
        llvm::errs() << "stride = " << getStride() << "\n";
    }
    Value * initialCount = b.getProcessedItemCount("i16Stream"); //prior to entering the generateblock logic function, how many itesm have been processed
    b.CreateBr(packLoop);  //creat  a branch to the basic block packloop
    b.SetInsertPoint(packLoop);
    
    PHINode * processedItemCount = b.CreatePHI(b.getSizeTy(), 2);
    processedItemCount->addIncoming(initialCount, entry);
    
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);
    Value * fleInput[8];
    for (unsigned i = 0; i < 8; i++) {   //advance function of lemier, load data
    
        fleInput[i] = b.loadInputStreamPack("i16Stream", ZERO, b.getInt32(i), blockOffsetPhi);  // assigning the initial array to be a value we loded from register
    }
    //
    
    Value * matches_all[2]; //2 64 bit integers , what holds the final matches, each match is 64 bit integer
    // calling the packwise and sum logic 
    for (unsigned i = 0; i < 2; i++) {   // transforming the data, first 64 bytes
        Value * val_matches[4];
        
        
        for (unsigned j = 0; j < 4; j++) { //first 4 vectors
            
            val_matches[j] = lookup(b, fleInput[i*4+j]);
            
            
        }
        matches_all[i] = simd_packed_pairwise_sum(b, val_matches[0], val_matches[1], val_matches[2], val_matches[3]);
    
    }
    Value * startPos = processedItemCount;

    for (unsigned i = 0; i < 2; i++) {
        BasicBlock * Entry = b.GetInsertBlock();
        
        
        BasicBlock * findNextMatch = b.CreateBasicBlock("findNextMatch", packFinalize);
        
        BasicBlock * findNextMatchExit = b.CreateBasicBlock("findNextMatchExit", packFinalize);
        
        Value * anyMatch = b.CreateIsNotNull(matches_all[i]); // is the value non zero
        b.CreateCondBr(anyMatch, findNextMatch, findNextMatchExit); //if matches is true, WHICH BASIC BLOCK TO ENTER to find the next match AND WHICH CODE TO RUN
        b.SetInsertPoint(findNextMatch);
        
        // finding trailing zeros , (how many zeros we passed to get to the 1)
        PHINode * matchesAllPhi = b.CreatePHI(b.getSizeTy(), 2);
        matchesAllPhi->addIncoming(matches_all[i], Entry);  //packLoop -initially entered this lop from
                                                               //matches_all[0] - matches_all(2 values) - first set of values
        
        PHINode * offSetPhi = b.CreatePHI(b.getSizeTy(), 2);  // number of things we have parsed prior to finding the first bit of matches_all
        offSetPhi->addIncoming(ZERO, Entry);
        
     //Lemier code
    //    int off = __builtin_ctzll(matches);
    //    matches >>= off;
    //    offset += off;
    //    return true;
        
        //int off = __builtin_ctzll(matches);
        Value * off = b.CreateCountForwardZeroes(matchesAllPhi);   //how many zeroes we have passed in matchesAllPhi prior to the first 1
        
        //matches >>= off;
        Value * newMatchesVal = b.CreateLShr(matchesAllPhi, off);
        
        //offset += off;  // find the position
        //matchesAllPhi->addIncoming(newMatchesVal, findNextMatch); // going back to the basic block
        
        Value * newOffSet = b.CreateAdd(offSetPhi, off);  // adding the phi nodes, this will be the next position
        
        //offSetPhi->addIncoming(newOffSet, findNextMatch);  // new value of the new offsetphi position
        
        // Get logic to get the positions
        
        Value * filePos = b.CreateAdd(newOffSet, startPos);  // finding the positin of the value we wanna add to  Scalar * Char
        
        Value * filePointer = b.getRawInputPointer("i16Stream", filePos);  // pointer to the byte at filePos position
        
        //benchmark logic, benchmark calls get, outer program to dervive the function
        Value * fileVal =b.CreateLoad(b.getInt8Ty(), filePointer); //8 bit integer that is loaded at the found position
        
        b.setScalarField("Char", fileVal);  //saving the fileVal in Char Scalar
        
        
        //CONSUME LOGIC OF LEMIER
        
        Value *  newMatchesVal_2= b.CreateLShr(newMatchesVal, b.getSize(1));  // consume logic of lemier, mover right by 1
        
        matchesAllPhi->addIncoming(newMatchesVal_2, findNextMatch); //incoming value , consume logic of lemier
        
        Value * newOffSet_2 = b.CreateAdd(newOffSet, b.getSize(1));  // adding the phi nodes, this will be the next position
        
        offSetPhi->addIncoming(newOffSet_2, findNextMatch);  // new value of the new offsetphi position
        
        // resatarting condition of the while loop, if we are at the end of 64bytes
        Value * anyMatch2 = b.CreateIsNotNull(newMatchesVal_2);
        b.CreateCondBr(anyMatch2, findNextMatch, findNextMatchExit);
        
        b.SetInsertPoint(findNextMatchExit); //When we have exausted all the matches in the 64bytes, then we exit the loop
        
        startPos = b.CreateAdd(startPos, b.getSize(64));  //updated or the second loop ()
        
    }
    BasicBlock * currentExit = b.GetInsertBlock();
    processedItemCount->addIncoming(startPos,currentExit);
   
    
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, currentExit);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

    b.CreateCondBr(moreToDo, packLoop, packFinalize);
    b.SetInsertPoint(packFinalize);
}

typedef uint8_t (*PackDemoFunctionType)(uint32_t fd);

PackDemoFunctionType packdemo_gen (CPUDriver & driver) {
    
    auto P = CreatePipeline(driver, Input<uint32_t>{"inputFileDecriptor"}, Output<uint8_t>{"Char"});
    
    Scalar * fileDescriptor = P.getInputScalar("inputFileDecriptor");
    
//    // Read source data into a 16-bit stream
//    StreamSet * const i16Stream = P.CreateStreamSet(1, 16);
//    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, i16Stream);
//
//    // Pairwise addition
//    StreamSet * const i8Stream = P.CreateStreamSet(1, 8);
//    P.CreateKernelCall<PackKernel>(i16Stream, i8Stream, PackOption::packl);
//    SHOW_BYTES(i8Stream);
//
//    StreamSet * const BasisBitsL = P.CreateStreamSet(8, 1);
//    P.CreateKernelCall<S2PKernel>(i8Stream, BasisBitsL);
//    SHOW_BIXNUM(BasisBitsL);
//
//    // Output the final pairwise summed stream
//    P.CreateKernelCall<StdOutKernel>(i8Stream);
//
//    return P.compile();
//
//}

    
    //source data
    
    StreamSet * const i16Stream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<ReadSourceKernel>(fileDescriptor, i16Stream);

    //StreamSet * const packedStreamL = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<PackKernel>(i16Stream, P.getOutputScalar("Char"));
    

//    StreamSet * const BasisBitsL = P.CreateStreamSet(8, 1);
//    P.CreateKernelCall<S2PKernel>(packedStreamL, BasisBitsL);
//    SHOW_BIXNUM(BasisBitsL);

//    StreamSet * const packedStreamH = P.CreateStreamSet(1, 8);
//    P.CreateKernelCall<PackKernel>(i16Stream, packedStreamH, PackOption::packh);
//    SHOW_BYTES(packedStreamH);

//    StreamSet * const BasisBitsH = P.CreateStreamSet(8, 1);
//    P.CreateKernelCall<S2PKernel>(packedStreamH, BasisBitsH);
//    SHOW_BIXNUM(BasisBitsH);

//    P.CreateKernelCall<StdOutKernel>(packedStreamL);

    return P.compile();
}



int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&PackDemoOptions, &codegen::JIT_InfoOptions, &codegen::InstrumentationOptions});
    CPUDriver driver("packdemo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        PackDemoFunctionType func = nullptr;
        func = packdemo_gen(driver);
        auto r = func(fd);
        close(fd);
        llvm::errs() << r << '\n' ;
    }
    return 0;
}
