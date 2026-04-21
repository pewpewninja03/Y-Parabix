/*
 * CMPT 473 - Dimension 1: lib/kernel/bitwise/ coverage tests
 *
 * Purpose: Drive bixlogic.cpp and bixnum_kernel.cpp through the existing
 *          Parabix kernel-test framework.  Before these tests, the
 *          lib/kernel/bitwise/ module was only 46.2% line-covered; none of
 *          the InvertKernel, BitwiseCombine, ZeroByMaskKernel, or bixnum::
 *          Add/Sub/Mul/EQ/NEQ/UGT/UGE/ULT/ULE kernels had a dedicated
 *          exercise from a test binary.
 *
 * Methodology (white-box.txt, section 4):
 *          branch coverage subsumes statement coverage.  Each test case
 *          below is designed to force the BitwiseCombine loop into its
 *          "i < toCombine.size()" branch (first stream combined) and its
 *          "i >= toCombine.size()" branch (remaining streams passed
 *          through unchanged) so both edges of the dispatch CFG are
 *          exercised.  The three-way switch inside BitwiseCombine
 *          (Or / Xor / And) is covered once each by the OrCombine,
 *          XorCombine, and AndCombine test cases.
 *
 * Convention for bit-stream notation:
 *          '1' = bit set, '0' or '.' = bit clear.
 *          Stream k of a BixNum represents bit k (LSB = stream 0) of the
 *          numeric value at each position.
 */

#include <testing/testing.h>
#include <kernel/bitwise/bixlogic.h>
#include <kernel/bitwise/bixnum_kernel.h>

using namespace kernel;
using namespace testing;

// ── Invert: NOT each bit of a single stream ──────────────────────────────────
auto inv_in  = BinaryStream({"10101010"});
auto inv_exp = BinaryStream({"01010101"});
TEST_CASE(test_invert, inv_in, inv_exp) {
    auto R = P.CreateStreamSet(1);
    Invert(P, Input<0>(T), R);
    AssertEQ(P, R, Input<1>(T));
}

// ── OrCombine: result[0] = source[0] | toCombine[0]; result[1] = source[1] ──
//   11001100 | 00110011 = 11111111
auto or_src  = BinaryStreamSet({"11001100", "10101010"});
auto or_comb = BinaryStreamSet({"00110011"});
auto or_exp  = BinaryStreamSet({"11111111", "10101010"});
TEST_CASE(test_or_combine, or_src, or_comb, or_exp) {
    auto R = P.CreateStreamSet(2);
    OrCombine(P, Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

// ── XorCombine: result[0] = source[0] ^ toCombine[0]; source[1] passthrough ─
//   11001100 ^ 10101010 = 01100110
auto xor_src  = BinaryStreamSet({"11001100", "10101010"});
auto xor_comb = BinaryStreamSet({"10101010"});
auto xor_exp  = BinaryStreamSet({"01100110", "10101010"});
TEST_CASE(test_xor_combine, xor_src, xor_comb, xor_exp) {
    auto R = P.CreateStreamSet(2);
    XorCombine(P, Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

// ── AndCombine: result[0] = source[0] & toCombine[0]; source[1] passthrough ─
//   11110000 & 10101010 = 10100000
auto and_src  = BinaryStreamSet({"11110000", "10101010"});
auto and_comb = BinaryStreamSet({"10101010"});
auto and_exp  = BinaryStreamSet({"10100000", "10101010"});
TEST_CASE(test_and_combine, and_src, and_comb, and_exp) {
    auto R = P.CreateStreamSet(2);
    AndCombine(P, Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

// ── ZeroByMask: AND mask with every stream in source ─────────────────────────
//   mask=11110000, src[0]=11001100, src[1]=10101010
//   result[0] = 11000000, result[1] = 10100000
auto zbm_mask = BinaryStream({"11110000"});
auto zbm_src  = BinaryStreamSet({"11001100", "10101010"});
auto zbm_exp  = BinaryStreamSet({"11000000", "10100000"});
TEST_CASE(test_zero_by_mask, zbm_mask, zbm_src, zbm_exp) {
    auto R = P.CreateStreamSet(2);
    ZeroByMask(P, Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

// ── BixNum 2-bit domain [0,1,2,3] at positions 0..3 ──────────────────────────
//   position 0 1 2 3  value 0 1 2 3
//   bit0     0 1 0 1  → "0101"
//   bit1     0 0 1 1  → "0011"
auto bix = BinaryStreamSet({"0101", "0011"});

// ── Add_immediate: [0,1,2,3] + 1 mod 4 = [1,2,3,0]
auto addi_exp = BinaryStreamSet({"1010", "0110"});
TEST_CASE(test_add_immediate, bix, addi_exp) {
    auto R = P.CreateStreamSet(2);
    P.CreateKernelCall<bixnum::Add_immediate>(Input<0>(T), 1U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── Sub_immediate: [0,1,2,3] - 1 mod 4 = [3,0,1,2] ───────────────────────────
auto subi_exp = BinaryStreamSet({"1010", "1001"});
TEST_CASE(test_sub_immediate, bix, subi_exp) {
    auto R = P.CreateStreamSet(2);
    P.CreateKernelCall<bixnum::Sub_immediate>(Input<0>(T), 1U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── Mul_immediate: [0,1,2,3] * 3 mod 4 = [0,3,2,1] ───────────────────────────
//   bit0: 0 1 0 1 → "0101"
//   bit1: 0 1 1 0 → "0110"
auto muli_exp = BinaryStreamSet({"0101", "0110"});
TEST_CASE(test_mul_immediate, bix, muli_exp) {
    auto R = P.CreateStreamSet(2);
    P.CreateKernelCall<bixnum::Mul_immediate>(Input<0>(T), 3U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── EQ_immediate ==2: only position 2 matches → "0010" ──────────────────────
auto eqi_exp = BinaryStream({"0010"});
TEST_CASE(test_eq_immediate, bix, eqi_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::EQ_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── NEQ_immediate !=2: positions 0,1,3 → "1101" ─────────────────────────────
auto neqi_exp = BinaryStream({"1101"});
TEST_CASE(test_neq_immediate, bix, neqi_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::NEQ_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── ULT_immediate <2: positions 0,1 → "1100" ────────────────────────────────
auto ulti_exp = BinaryStream({"1100"});
TEST_CASE(test_ult_immediate, bix, ulti_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::ULT_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── UGE_immediate >=2: positions 2,3 → "0011" ───────────────────────────────
auto ugei_exp = BinaryStream({"0011"});
TEST_CASE(test_uge_immediate, bix, ugei_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::UGE_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── UGT_immediate >2: position 3 only → "0001" ──────────────────────────────
auto ugti_exp = BinaryStream({"0001"});
TEST_CASE(test_ugt_immediate, bix, ugti_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::UGT_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── ULE_immediate <=2: positions 0,1,2 → "1110" ─────────────────────────────
auto ulei_exp = BinaryStream({"1110"});
TEST_CASE(test_ule_immediate, bix, ulei_exp) {
    auto R = P.CreateStreamSet(1);
    P.CreateKernelCall<bixnum::ULE_immediate>(Input<0>(T), 2U, R);
    AssertEQ(P, R, Input<1>(T));
}

// ── Add (BixNum + BixNum): [0,1,2,3] + [3,2,1,0] mod 4 = [3,3,3,3] ──────────
//   addend [3,2,1,0]:
//     pos 0: 3 → bit0=1, bit1=1
//     pos 1: 2 → bit0=0, bit1=1
//     pos 2: 1 → bit0=1, bit1=0
//     pos 3: 0 → bit0=0, bit1=0
//   → add_b = {"1010", "1100"}
//   sum [3,3,3,3] → bit0 all ones, bit1 all ones
auto add_b   = BinaryStreamSet({"1010", "1100"});
auto add_exp = BinaryStreamSet({"1111", "1111"});
TEST_CASE(test_add_binary, bix, add_b, add_exp) {
    auto R = P.CreateStreamSet(2);
    P.CreateKernelCall<bixnum::Add>(Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

// ── Sub (BixNum - BixNum): x - x = 0 for all positions ──────────────────────
auto sub_exp = BinaryStreamSet({"0000", "0000"});
TEST_CASE(test_sub_binary, bix, bix, sub_exp) {
    auto R = P.CreateStreamSet(2);
    P.CreateKernelCall<bixnum::Sub>(Input<0>(T), Input<1>(T), R);
    AssertEQ(P, R, Input<2>(T));
}

RUN_TESTS(
    CASE(test_invert),
    CASE(test_or_combine),
    CASE(test_xor_combine),
    CASE(test_and_combine),
    CASE(test_zero_by_mask),
    CASE(test_add_immediate),
    CASE(test_sub_immediate),
    CASE(test_mul_immediate),
    CASE(test_eq_immediate),
    CASE(test_neq_immediate),
    CASE(test_ult_immediate),
    CASE(test_uge_immediate),
    CASE(test_ugt_immediate),
    CASE(test_ule_immediate),
    CASE(test_add_binary),
    CASE(test_sub_binary),
)
