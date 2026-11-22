#include "test_utils.h"
#include "gemm.h"
#include "tensor.h"
#include "cpu_features.h"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using tile_runtime::Tensor;
using tile_runtime::gemm_naive;
using tile_runtime::gemm_tiled;
using tile_runtime::gemm_parallel;
using tile_runtime::gemm_avx;
using tile_runtime::gemm_avx512;
#ifdef TILE_HAS_STD_SIMD
using tile_runtime::gemm_simd;
#endif
using tile_runtime::gemm_parallel_simd;
using tile_runtime::gemm_parallel_avx512;

// ---------------------------------------------------------------------------
// The kernels under test
// ---------------------------------------------------------------------------

// Every kernel has this signature once gemm_naive is given a block size it
// ignores, so one table can push all of them through the same checks.
using Kernel = void (*)(const Tensor& A, const Tensor& B, Tensor& C, size_t block_size);

struct KernelCase {
    const char* name;
    Kernel run;
    bool uses_block_size;  // false for gemm_naive: one run per shape is enough
    bool parallel;         // true for the OpenMP kernels
    bool supported;        // false when this CPU cannot run the kernel
};

static void run_naive(const Tensor& A, const Tensor& B, Tensor& C, size_t /*block_size*/) {
    gemm_naive(A, B, C);
}

static const std::vector<KernelCase>& all_kernels() {
    static const std::vector<KernelCase> kernels = [] {
        const tile_runtime::CpuFeatures& cpu = tile_runtime::CpuFeatures::detect();
        const bool avx2 = cpu.avx2 && cpu.fma;
        // The AVX-512 sources are compiled with -mavx512vl as well as -mavx512f.
        const bool avx512 = cpu.avx512f && cpu.avx512vl && cpu.fma;
        return std::vector<KernelCase>{
            {"gemm_naive", run_naive, false, false, true},
            {"gemm_tiled", gemm_tiled, true, false, true},
            {"gemm_parallel", gemm_parallel, true, true, true},
            {"gemm_avx", gemm_avx, true, false, avx2},
            {"gemm_avx512", gemm_avx512, true, false, avx512},
#ifdef TILE_HAS_STD_SIMD
            // Compiled with -mavx2 -mfma, so it needs the same CPU as gemm_avx.
            {"gemm_simd", gemm_simd, true, false, avx2},
#endif
            {"gemm_parallel_simd", gemm_parallel_simd, true, true, avx2},
            {"gemm_parallel_avx512", gemm_parallel_avx512, true, true, avx512},
        };
    }();
    return kernels;
}

static void print_kernels_under_test() {
    std::cout << "kernels under test:";
    for (const KernelCase& kernel : all_kernels()) {
        std::cout << " " << kernel.name;
        if (!kernel.supported) {
            std::cout << " (skipped: this CPU cannot run it)";
        }
    }
    std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// Reference product and tolerance
// ---------------------------------------------------------------------------

// A * B computed in double, independently of every kernel, together with the
// sum of |a_ik * b_kj| behind each element. That sum scales the tolerance.
struct Reference {
    size_t M;
    size_t K;
    size_t N;
    std::vector<double> product;    // M x N, row-major
    std::vector<double> magnitude;  // M x N, row-major
};

static Reference reference_product(const Tensor& A, const Tensor& B) {
    Reference ref{A.rows(), A.cols(), B.cols(), {}, {}};
    ref.product.assign(ref.M * ref.N, 0.0);
    ref.magnitude.assign(ref.M * ref.N, 0.0);

    const float* a = A.data();
    const float* b = B.data();
    for (size_t i = 0; i < ref.M; ++i) {
        for (size_t k = 0; k < ref.K; ++k) {
            const double a_ik = a[i * ref.K + k];
            for (size_t j = 0; j < ref.N; ++j) {
                const double term = a_ik * static_cast<double>(b[k * ref.N + j]);
                ref.product[i * ref.N + j] += term;
                ref.magnitude[i * ref.N + j] += std::fabs(term);
            }
        }
    }
    return ref;
}

// True when `got` is within the rounding error a single-precision dot product
// of K terms can pick up.
//
// Each term goes through one multiply and then at most K + 1 additions (the
// rest of its k-chunk, then one addition per later chunk as the chunks are
// folded into C). Every one of those operations rounds by at most
// FLT_EPSILON / 2 relative to a partial sum no larger than `magnitude`, so
// the error is bounded by about (K + 2) * FLT_EPSILON / 2 * magnitude,
// whatever order the kernel adds the terms in. The tolerance here is four
// times that bound: loose enough never to fail on rounding, tight enough that
// one dropped or doubled term is far outside it.
//
// The comparison is written as "difference <= tolerance" so that a NaN result
// is rejected: any comparison with NaN is false.
static bool within_rounding_error(double got, double expected, size_t K, double magnitude) {
    const double tolerance = 2.0 * static_cast<double>(K + 2) * FLT_EPSILON * magnitude;
    return std::fabs(got - expected) <= tolerance;
}

// A sweep stops once it has reported this many failures, so a broken kernel
// gives a readable log instead of thousands of lines.
constexpr int MAX_FAILURES_PER_SWEEP = 10;

static bool too_many_failures(int failures_before) {
    return g_test_failures - failures_before >= MAX_FAILURES_PER_SWEEP;
}

// Compares one kernel's output with the reference. Records at most one
// failure per call and prints what is needed to reproduce it.
static bool matches_reference(const KernelCase& kernel, size_t block_size,
                              const Reference& ref, const Tensor& C) {
    const float* c = C.data();
    for (size_t i = 0; i < ref.M; ++i) {
        for (size_t j = 0; j < ref.N; ++j) {
            const size_t at = i * ref.N + j;
            if (!within_rounding_error(c[at], ref.product[at], ref.K, ref.magnitude[at])) {
                std::cerr << "FAIL: " << kernel.name << "  A is " << ref.M << "x" << ref.K
                          << ", B is " << ref.K << "x" << ref.N
                          << ", block_size=" << block_size
                          << "  C(" << i << ", " << j << ") = " << c[at]
                          << ", expected " << ref.product[at] << std::endl;
                ++g_test_failures;
                return false;
            }
        }
    }
    return true;
}

// An output tensor filled with NaN. Every kernel must overwrite C, so a
// kernel that adds into C, or leaves an element unwritten, still has NaN
// there, and every comparison in this file rejects NaN.
static Tensor nan_filled(size_t rows, size_t cols) {
    return Tensor(rows, cols, std::numeric_limits<float>::quiet_NaN());
}

// Runs one kernel and checks the result. C starts as NaN so every output
// element must be overwritten to pass the comparison.
static void check_kernel(const KernelCase& kernel, const Tensor& A, const Tensor& B,
                         const Reference& ref, size_t block_size) {
    Tensor C = nan_filled(ref.M, ref.N);
    kernel.run(A, B, C, block_size);
    matches_reference(kernel, block_size, ref, C);
}

// Runs every supported kernel on one random M x K times K x N product, at
// each of the given block sizes.
static void check_shape(size_t M, size_t K, size_t N,
                        const std::vector<size_t>& block_sizes, unsigned seed) {
    Tensor A(M, K), B(K, N);
    A.randomize(seed);
    B.randomize(seed + 1);
    const Reference ref = reference_product(A, B);

    const int failures_before = g_test_failures;
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        for (size_t block_size : block_sizes) {
            check_kernel(kernel, A, B, ref, block_size);
            if (!kernel.uses_block_size || too_many_failures(failures_before)) {
                break;
            }
        }
    }
}

static Tensor tensor_from(size_t rows, size_t cols, const std::vector<float>& values) {
    Tensor t(rows, cols);
    std::copy(values.begin(), values.end(), t.data());
    return t;
}

// ---------------------------------------------------------------------------
// Checks on the test helpers themselves
// ---------------------------------------------------------------------------

void test_nearly_equal_rejects_nan() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    ASSERT_TRUE(!nearly_equal(nan, 1.0f));
    ASSERT_TRUE(!nearly_equal(1.0f, nan));
    ASSERT_TRUE(!nearly_equal(nan, nan));
}

void test_nearly_equal_rejects_infinity() {
    const float inf = std::numeric_limits<float>::infinity();
    ASSERT_TRUE(!nearly_equal(inf, 1.0f));
    ASSERT_TRUE(!nearly_equal(inf, inf));
    ASSERT_TRUE(!nearly_equal(-inf, inf));
}

void test_nearly_equal_uses_relative_tolerance() {
    ASSERT_TRUE(nearly_equal(1.0f, 1.0f));
    ASSERT_TRUE(nearly_equal(1000.0f, 1000.05f));   // 5e-5 of the value
    ASSERT_TRUE(!nearly_equal(1000.0f, 1000.5f));   // 5e-4 of the value
    ASSERT_TRUE(nearly_equal(0.0f, 0.00005f));      // absolute below 1
    ASSERT_TRUE(!nearly_equal(0.0f, 0.001f));
}

void test_rounding_bound_rejects_nan_and_missing_terms() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Three terms of 0.25 each: the product is 0.75 and so is the magnitude.
    ASSERT_TRUE(within_rounding_error(0.75, 0.75, 3, 0.75));
    ASSERT_TRUE(within_rounding_error(0.75 + 1e-7, 0.75, 3, 0.75));
    ASSERT_TRUE(!within_rounding_error(0.5, 0.75, 3, 0.75));  // one term dropped
    ASSERT_TRUE(!within_rounding_error(nan, 0.75, 3, 0.75));
    // With no terms at all, the only acceptable result is exactly zero.
    ASSERT_TRUE(within_rounding_error(0.0, 0.0, 0, 0.0));
    ASSERT_TRUE(!within_rounding_error(1e-30, 0.0, 0, 0.0));
}

// ---------------------------------------------------------------------------
// Every kernel, same checks
// ---------------------------------------------------------------------------

// Compares C with literal values worked out by hand, so these cases do not
// rest on the reference helper.
static void expect_values(const KernelCase& kernel, size_t block_size, const Tensor& C,
                          const std::vector<float>& expected) {
    for (size_t at = 0; at < expected.size(); ++at) {
        if (!nearly_equal(C.data()[at], expected[at])) {
            std::cerr << "FAIL: " << kernel.name << "  block_size=" << block_size
                      << "  element " << at << " = " << C.data()[at]
                      << ", expected " << expected[at] << std::endl;
            ++g_test_failures;
            return;
        }
    }
}

void test_hand_computed_products() {
    // | 1 2 |   | 5 6 |   | 19 22 |
    // | 3 4 | x | 7 8 | = | 43 50 |
    const Tensor A2 = tensor_from(2, 2, {1, 2, 3, 4});
    const Tensor B2 = tensor_from(2, 2, {5, 6, 7, 8});
    const std::vector<float> C2 = {19, 22, 43, 50};

    // A (3x2) x B (2x4) -> C (3x4)
    // Row 0: (1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8) = (11, 14, 17, 20)
    // Row 1: (3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8) = (23, 30, 37, 44)
    // Row 2: (5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8) = (35, 46, 57, 68)
    const Tensor A3 = tensor_from(3, 2, {1, 2, 3, 4, 5, 6});
    const Tensor B3 = tensor_from(2, 4, {1, 2, 3, 4, 5, 6, 7, 8});
    const std::vector<float> C3 = {11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68};

    // 1x1: 3 * 7 = 21
    const Tensor A1 = tensor_from(1, 1, {3});
    const Tensor B1 = tensor_from(1, 1, {7});
    const std::vector<float> C1 = {21};

    const size_t block_sizes[] = {1, 2, 16};
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        for (size_t block_size : block_sizes) {
            Tensor C = nan_filled(2, 2);
            kernel.run(A2, B2, C, block_size);
            expect_values(kernel, block_size, C, C2);

            Tensor D = nan_filled(3, 4);
            kernel.run(A3, B3, D, block_size);
            expect_values(kernel, block_size, D, C3);

            Tensor E = nan_filled(1, 1);
            kernel.run(A1, B1, E, block_size);
            expect_values(kernel, block_size, E, C1);
        }
    }
}

// Records a failure unless `got` holds exactly the same values as `expected`.
// The values are printed with enough digits to tell two different floats
// apart, since a mismatch here can be in the last bit.
static void expect_exact(const KernelCase& kernel, size_t block_size, const char* what,
                         const Tensor& got, const Tensor& expected) {
    for (size_t at = 0; at < expected.size(); ++at) {
        if (!(got.data()[at] == expected.data()[at])) {
            const std::streamsize precision_before =
                std::cerr.precision(std::numeric_limits<float>::max_digits10);
            std::cerr << "FAIL: " << kernel.name << "  " << what << ", result is "
                      << expected.rows() << "x" << expected.cols()
                      << ", block_size=" << block_size
                      << "  element " << at << " = " << got.data()[at]
                      << ", expected exactly " << expected.data()[at] << std::endl;
            std::cerr.precision(precision_before);
            ++g_test_failures;
            return;
        }
    }
}

// Multiplying by an identity matrix involves only exact operations (x * 1,
// x * 0 and adding zero), so every kernel must give A back exactly, with the
// identity on either side.
void test_identity_returns_the_input_exactly() {
    struct Shape { size_t rows, cols; };
    const Shape shapes[] = {{1, 1}, {5, 17}, {17, 5}, {33, 9}};
    const size_t block_sizes[] = {1, 4, 16, 64};

    unsigned seed = 4000;
    for (const Shape& shape : shapes) {
        Tensor A(shape.rows, shape.cols);
        A.randomize(seed++);
        Tensor left(shape.rows, shape.rows), right(shape.cols, shape.cols);
        for (size_t i = 0; i < shape.rows; ++i) left.at(i, i) = 1.0f;
        for (size_t i = 0; i < shape.cols; ++i) right.at(i, i) = 1.0f;

        for (const KernelCase& kernel : all_kernels()) {
            if (!kernel.supported) {
                continue;
            }
            for (size_t block_size : block_sizes) {
                Tensor C = nan_filled(shape.rows, shape.cols);
                kernel.run(left, A, C, block_size);
                expect_exact(kernel, block_size, "identity * A", C, A);

                Tensor D = nan_filled(shape.rows, shape.cols);
                kernel.run(A, right, D, block_size);
                expect_exact(kernel, block_size, "A * identity", D, A);
            }
        }
    }
}

// Small shapes, every kernel, many block sizes.
//
// M from 1 to 9 gives zero, one and two groups of 4 rows, each with every
// count of leftover rows. N from 1 to 33 gives every AVX2 column tail (1 to 7)
// and every AVX-512 mask (1 to 15), with and without full vectors before them.
// Block sizes below 8 and 16 make tiles narrower than one vector, and K takes
// values below, equal to and above the block size.
void test_small_shapes_match_reference() {
    const std::vector<size_t> block_sizes = {1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 32, 64};
    const size_t k_values[] = {1, 2, 3, 8, 17};

    const int failures_before = g_test_failures;
    unsigned seed = 1000;
    for (size_t M = 1; M <= 9; ++M) {
        for (size_t N = 1; N <= 33; ++N) {
            for (size_t K : k_values) {
                check_shape(M, K, N, block_sizes, seed);
                seed += 2;
                if (too_many_failures(failures_before)) {
                    return;
                }
            }
        }
    }
}

// Larger shapes that split into many tiles, including tiles cut short at the
// right and bottom edges. 48, 64 and 128 are the block sizes the benchmark
// sweeps.
void test_multi_tile_shapes_match_reference() {
    struct Shape { size_t M, K, N; };
    const Shape shapes[] = {
        {64, 64, 64}, {100, 100, 100}, {37, 53, 71}, {129, 65, 130},
        {24, 7, 24},  {40, 9, 40},     {18, 11, 37}, {3, 5, 17},
    };
    const std::vector<size_t> block_sizes = {8, 12, 16, 20, 32, 48, 64, 128};

    const int failures_before = g_test_failures;
    unsigned seed = 2000;
    for (const Shape& shape : shapes) {
        check_shape(shape.M, shape.K, shape.N, block_sizes, seed);
        seed += 2;
        if (too_many_failures(failures_before)) {
            return;
        }
    }
}

// The k loop is split into chunks of the block size as well. For each block
// size this takes K just below it, equal to it, twice it and one past twice
// it, plus K = 1 and K = 2, where the SIMD kernels' prefetch looks past the
// last row of B. M = 6 leaves two rows after a group of 4, and N = 19 leaves
// three columns after full vectors.
void test_k_chunks_match_reference() {
    const size_t chunk_sizes[] = {4, 8, 16};

    const int failures_before = g_test_failures;
    unsigned seed = 3000;
    for (size_t block : chunk_sizes) {
        const size_t k_values[] = {1, 2, block - 1, block, 2 * block, 2 * block + 1};
        for (size_t K : k_values) {
            check_shape(6, K, 19, {block}, seed);
            seed += 2;
            if (too_many_failures(failures_before)) {
                return;
            }
        }
    }
}

// Matrices with a zero dimension are legal inputs. With K = 0, C starts as
// NaN and the zero tolerance requires every element to be overwritten with
// exact zero, even across several tiles. With M = 0 or N = 0 there is nothing
// to compute and the kernel must return without dividing by a zero tile count.
void test_empty_shapes() {
    struct Shape { size_t M, K, N; };
    const Shape shapes[] = {{3, 0, 5}, {0, 5, 3}, {3, 5, 0}, {0, 0, 0}, {9, 0, 33}};
    const std::vector<size_t> block_sizes = {1, 4};

    unsigned seed = 5000;
    for (const Shape& shape : shapes) {
        check_shape(shape.M, shape.K, shape.N, block_sizes, seed);
        seed += 2;
    }
}

// Records a failure unless the kernel rejects these shapes.
static void expect_invalid_argument(const KernelCase& kernel, const char* what,
                                    const Tensor& A, const Tensor& B, Tensor& C,
                                    size_t block_size) {
    try {
        kernel.run(A, B, C, block_size);
    } catch (const std::invalid_argument&) {
        return;
    }
    std::cerr << "FAIL: " << kernel.name << "  expected std::invalid_argument for "
              << what << std::endl;
    ++g_test_failures;
}

void test_shape_mismatch_throws() {
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        {
            Tensor A(2, 3), B(4, 2), C(2, 2);
            expect_invalid_argument(kernel, "A.cols() != B.rows()", A, B, C, 16);
        }
        {
            Tensor A(2, 3), B(3, 4), C(3, 4);
            expect_invalid_argument(kernel, "C with the wrong number of rows", A, B, C, 16);
        }
        {
            Tensor A(2, 3), B(3, 4), C(2, 5);
            expect_invalid_argument(kernel, "C with the wrong number of columns", A, B, C, 16);
        }
    }
}

// The product must be independent of the previous contents of C. A finite
// starting value makes any unintended addition visible in the failure output.
void test_every_kernel_overwrites_previous_output() {
    struct Shape { size_t M, K, N, block_size; };
    const Shape shapes[] = {
        {3, 3, 3, 16}, {9, 17, 33, 4}, {37, 53, 71, 16}, {40, 19, 72, 8},
    };

    const int failures_before = g_test_failures;
    unsigned seed = 8000;
    for (const Shape& shape : shapes) {
        Tensor A(shape.M, shape.K), B(shape.K, shape.N);
        A.randomize(seed++);
        B.randomize(seed++);
        const Reference ref = reference_product(A, B);

        for (const KernelCase& kernel : all_kernels()) {
            if (!kernel.supported) {
                continue;
            }
            Tensor C(shape.M, shape.N, 999.0f);
            kernel.run(A, B, C, shape.block_size);
            matches_reference(kernel, shape.block_size, ref, C);
            if (too_many_failures(failures_before)) {
                return;
            }
        }
    }
}

// C starts at zero on purpose. Only the contents of C change between calls,
// so a kernel that adds into C reports a doubled value instead of NaN.
void test_calling_twice_gives_the_same_result() {
    struct Shape { size_t M, K, N, block_size; };
    const Shape shapes[] = {
        {3, 3, 3, 16}, {9, 17, 33, 4}, {37, 53, 71, 16}, {40, 19, 72, 8},
    };

    const int failures_before = g_test_failures;
    unsigned seed = 9000;
    for (const Shape& shape : shapes) {
        Tensor A(shape.M, shape.K), B(shape.K, shape.N);
        A.randomize(seed++);
        B.randomize(seed++);

        for (const KernelCase& kernel : all_kernels()) {
            if (!kernel.supported) {
                continue;
            }
            Tensor C(shape.M, shape.N);
            kernel.run(A, B, C, shape.block_size);
            const Tensor first = C;
            kernel.run(A, B, C, shape.block_size);
            expect_exact(kernel, shape.block_size,
                         "second call on the same C against the first call", C, first);
            if (too_many_failures(failures_before)) {
                return;
            }
        }
    }
}

// Diagnostics must identify the entry point and both relevant shapes.
static void expect_shape_message(const KernelCase& kernel, const Tensor& A,
                                 const Tensor& B, Tensor& C,
                                 const char* first, const char* second) {
    std::string message = "no exception";
    try {
        kernel.run(A, B, C, 16);
    } catch (const std::invalid_argument& error) {
        message = error.what();
        if (message.find(kernel.name) != std::string::npos &&
            message.find(first) != std::string::npos &&
            message.find(second) != std::string::npos) {
            return;
        }
    }
    std::cerr << "FAIL: " << kernel.name << "  unexpected diagnostic: "
              << message << std::endl;
    ++g_test_failures;
}

// Shape errors must give callers enough context to locate the bad arguments.
void test_error_messages_name_the_kernel_and_the_shapes() {
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        Tensor A(2, 3), B(4, 2), C(2, 2);
        expect_shape_message(kernel, A, B, C, "2x3", "4x2");
        Tensor D(3, 4), E(3, 5);
        expect_shape_message(kernel, A, D, E, "3x5", "2x4");
    }
}

// Rejected output aliases must leave both input tensors untouched.
void test_output_aliasing_an_input_throws() {
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        Tensor S(5, 5), T(5, 5);
        S.randomize(10000);
        T.randomize(10001);
        const Tensor S_before = S;
        const Tensor T_before = T;
        expect_invalid_argument(kernel, "C is the same tensor as A", S, T, S, 16);
        expect_exact(kernel, 16, "A after the rejected call", S, S_before);
        expect_invalid_argument(kernel, "C is the same tensor as B", S, T, T, 16);
        expect_exact(kernel, 16, "B after the rejected call", T, T_before);
    }
}

// Sharing the read-only inputs is legal even though output aliases are not.
void test_inputs_may_alias_each_other() {
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported) {
            continue;
        }
        for (size_t block_size : {size_t(4), size_t(16)}) {
            Tensor S(17, 17);
            S.randomize(11000);
            const Reference ref = reference_product(S, S);
            Tensor C = nan_filled(17, 17);
            kernel.run(S, S, C, block_size);
            matches_reference(kernel, block_size, ref, C);
        }
    }
}

// Oversized blocks must still compute the single output tile without wrapping.
void test_huge_block_size_matches_reference() {
    const size_t max = std::numeric_limits<size_t>::max();
    const std::vector<size_t> blocks = {max, max - 1, max / 2 + 1, size_t(1) << 20};
    check_shape(37, 53, 71, blocks, 12000);
    check_shape(9, 17, 33, blocks, 12002);
    check_shape(2, 3, 2, blocks, 12004);
}

// Zero cannot advance tile traversal and must be rejected before computing.
void test_zero_block_size_throws() {
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported || !kernel.uses_block_size) {
            continue;
        }
        Tensor A(4, 3), B(3, 5);
        Tensor C = nan_filled(4, 5);
        expect_invalid_argument(kernel, "block_size == 0", A, B, C, 0);
    }
}

// ---------------------------------------------------------------------------
// Parallel kernels
// ---------------------------------------------------------------------------

// Each pair executes the same tile kernel on the same tiles, so scheduling
// must leave every output value exactly equal to the serial result.
void test_parallel_kernels_match_their_serial_twin() {
    struct KernelPair { Kernel serial, parallel; };
    const KernelPair pairs[] = {
        {gemm_tiled, gemm_parallel},
        {gemm_avx, gemm_parallel_simd},
        {gemm_avx512, gemm_parallel_avx512},
    };
    struct Shape { size_t M, K, N, block_size; };
    const Shape shapes[] = {
        {40, 19, 72, 8}, {37, 53, 71, 16}, {9, 17, 33, 4}, {129, 65, 130, 64},
    };

    const auto& kernels = all_kernels();
    const auto find_kernel = [&kernels](Kernel run) {
        return std::find_if(kernels.begin(), kernels.end(),
                            [run](const KernelCase& kernel) { return kernel.run == run; });
    };

    unsigned seed = 7000;
    for (const KernelPair& pair : pairs) {
        const auto serial = find_kernel(pair.serial);
        const auto parallel = find_kernel(pair.parallel);
        ASSERT_TRUE(serial != kernels.end() && parallel != kernels.end());
        if (serial == kernels.end() || parallel == kernels.end() ||
            !serial->supported || !parallel->supported) {
            continue;
        }

        const std::string what = std::string("against ") + serial->name;
        for (const Shape& shape : shapes) {
            Tensor A(shape.M, shape.K), B(shape.K, shape.N);
            A.randomize(seed++);
            B.randomize(seed++);

            Tensor serial_result = nan_filled(shape.M, shape.N);
            Tensor parallel_result = nan_filled(shape.M, shape.N);
            serial->run(A, B, serial_result, shape.block_size);
            parallel->run(A, B, parallel_result, shape.block_size);
            expect_exact(*parallel, shape.block_size, what.c_str(), parallel_result, serial_result);
        }
    }
}

#ifdef _OPENMP
// Each output tile is computed by exactly one thread, so a parallel kernel
// must give the same values whatever the thread count. The three shapes split
// into 45 tiles, 2 tiles and 1 tile, so some runs have more threads than
// tiles.
void test_parallel_kernels_ignore_thread_count() {
    struct Shape { size_t M, K, N, block_size; };
    const Shape shapes[] = {{40, 19, 72, 8}, {8, 19, 16, 8}, {5, 19, 6, 64}};
    const int thread_counts[] = {1, 2, 3, 4, 7};

    const int threads_before = omp_get_max_threads();
    unsigned seed = 6000;
    for (const KernelCase& kernel : all_kernels()) {
        if (!kernel.supported || !kernel.parallel) {
            continue;
        }
        for (const Shape& shape : shapes) {
            Tensor A(shape.M, shape.K), B(shape.K, shape.N);
            A.randomize(seed);
            B.randomize(seed + 1);
            seed += 2;
            const Reference ref = reference_product(A, B);

            Tensor with_one_thread(shape.M, shape.N);
            for (int threads : thread_counts) {
                omp_set_num_threads(threads);
                Tensor C = nan_filled(shape.M, shape.N);
                kernel.run(A, B, C, shape.block_size);
                matches_reference(kernel, shape.block_size, ref, C);

                if (threads == 1) {
                    with_one_thread = C;
                } else {
                    const std::string what =
                        "result with " + std::to_string(threads) + " threads against 1 thread";
                    expect_exact(kernel, shape.block_size, what.c_str(), C, with_one_thread);
                }
            }
        }
    }
    omp_set_num_threads(threads_before);
}
#endif

int main() {
    print_kernels_under_test();

    std::cout << "test_gemm (test helpers):" << std::endl;
    RUN_TEST(test_nearly_equal_rejects_nan);
    RUN_TEST(test_nearly_equal_rejects_infinity);
    RUN_TEST(test_nearly_equal_uses_relative_tolerance);
    RUN_TEST(test_rounding_bound_rejects_nan_and_missing_terms);

    std::cout << "test_gemm (every kernel):" << std::endl;
    RUN_TEST(test_hand_computed_products);
    RUN_TEST(test_identity_returns_the_input_exactly);
    RUN_TEST(test_small_shapes_match_reference);
    RUN_TEST(test_multi_tile_shapes_match_reference);
    RUN_TEST(test_k_chunks_match_reference);
    RUN_TEST(test_empty_shapes);
    RUN_TEST(test_shape_mismatch_throws);
    RUN_TEST(test_every_kernel_overwrites_previous_output);
    RUN_TEST(test_calling_twice_gives_the_same_result);
    RUN_TEST(test_error_messages_name_the_kernel_and_the_shapes);
    RUN_TEST(test_output_aliasing_an_input_throws);
    RUN_TEST(test_inputs_may_alias_each_other);
    RUN_TEST(test_huge_block_size_matches_reference);
    RUN_TEST(test_zero_block_size_throws);

    std::cout << "test_gemm (parallel kernels):" << std::endl;
    RUN_TEST(test_parallel_kernels_match_their_serial_twin);
#ifdef _OPENMP
    RUN_TEST(test_parallel_kernels_ignore_thread_count);
#endif

    TEST_SUMMARY("test_gemm");
}
