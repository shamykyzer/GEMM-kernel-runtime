#include "test_utils.h"
#include "tensor.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

using tile_runtime::Tensor;

static_assert(std::is_nothrow_move_constructible<Tensor>::value, "Tensor moves must not throw");
static_assert(std::is_nothrow_move_assignable<Tensor>::value, "Tensor moves must not throw");
static_assert(std::is_copy_constructible<Tensor>::value, "Tensor stays copyable");
static_assert(std::is_copy_assignable<Tensor>::value, "Tensor stays copyable");

void test_default_construction() {
    Tensor t;
    ASSERT_EQ(t.rows(), size_t(0));
    ASSERT_EQ(t.cols(), size_t(0));
    ASSERT_EQ(t.size(), size_t(0));
}

void test_shape_construction() {
    Tensor t(3, 4);
    ASSERT_EQ(t.rows(), size_t(3));
    ASSERT_EQ(t.cols(), size_t(4));
    ASSERT_EQ(t.size(), size_t(12));
    // Default-initialized to zero
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(t.at(i, j), 0.0f);
}

void test_init_value_construction() {
    Tensor t(2, 3, 5.0f);
    ASSERT_EQ(t.size(), size_t(6));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            ASSERT_NEAR(t.at(i, j), 5.0f);
}

void test_at_read_write() {
    Tensor t(2, 2);
    t.at(0, 0) = 1.0f;
    t.at(0, 1) = 2.0f;
    t.at(1, 0) = 3.0f;
    t.at(1, 1) = 4.0f;
    ASSERT_NEAR(t.at(0, 0), 1.0f);
    ASSERT_NEAR(t.at(0, 1), 2.0f);
    ASSERT_NEAR(t.at(1, 0), 3.0f);
    ASSERT_NEAR(t.at(1, 1), 4.0f);
}

void test_at_out_of_bounds() {
    Tensor t(2, 3);
    ASSERT_THROWS(t.at(2, 0), std::out_of_range);
    ASSERT_THROWS(t.at(0, 3), std::out_of_range);
    ASSERT_THROWS(t.at(100, 100), std::out_of_range);
}

void test_fill() {
    Tensor t(3, 3);
    t.fill(7.0f);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            ASSERT_NEAR(t.at(i, j), 7.0f);
}

void test_zero() {
    Tensor t(2, 2, 9.0f);
    t.zero();
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            ASSERT_NEAR(t.at(i, j), 0.0f);
}

void test_randomize_deterministic() {
    Tensor a(4, 4);
    Tensor b(4, 4);
    a.randomize(42);
    b.randomize(42);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            ASSERT_NEAR(a.at(i, j), b.at(i, j));
}

void test_randomize_different_seeds() {
    Tensor a(4, 4);
    Tensor b(4, 4);
    a.randomize(1);
    b.randomize(2);
    bool any_different = false;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 4; ++j)
            if (std::fabs(a.at(i, j) - b.at(i, j)) > 1e-6f)
                any_different = true;
    ASSERT_TRUE(any_different);
}

void test_copy_independence() {
    Tensor a(2, 2, 1.0f);
    Tensor b = a;
    b.at(0, 0) = 99.0f;
    ASSERT_NEAR(a.at(0, 0), 1.0f);
    ASSERT_NEAR(b.at(0, 0), 99.0f);

    Tensor c(2, 2);
    c = a;
    c.at(1, 1) = 88.0f;
    ASSERT_NEAR(a.at(1, 1), 1.0f);
    ASSERT_NEAR(c.at(1, 1), 88.0f);
}

void test_data_pointer() {
    Tensor t(2, 3);
    t.at(1, 2) = 42.0f;
    // Row-major: element (1,2) is at offset 1*3+2 = 5
    ASSERT_NEAR(t.data()[5], 42.0f);
}

void test_aligned_allocation() {
    Tensor t(64, 64);
    uintptr_t addr = reinterpret_cast<uintptr_t>(t.data());
    ASSERT_TRUE((addr % 64) == 0);

    // Small tensor should also be aligned
    Tensor small(1, 1);
    uintptr_t small_addr = reinterpret_cast<uintptr_t>(small.data());
    ASSERT_TRUE((small_addr % 64) == 0);

    // Non-power-of-2 dimensions
    Tensor odd(7, 13);
    uintptr_t odd_addr = reinterpret_cast<uintptr_t>(odd.data());
    ASSERT_TRUE((odd_addr % 64) == 0);
}

// Wrapped element counts must not let the reported shape exceed storage.
void test_size_overflow_throws() {
    const size_t half = std::numeric_limits<size_t>::max() / 2 + 1;
    ASSERT_THROWS(Tensor(half, 2), std::length_error);
    ASSERT_THROWS(Tensor(2, half), std::length_error);
    ASSERT_THROWS(Tensor(half, 2, 1.0f), std::length_error);
    ASSERT_THROWS(Tensor(half + 1, 2), std::length_error);
    Tensor t(std::numeric_limits<size_t>::max(), 0);
    ASSERT_EQ(t.size(), size_t(0));
}

// A moved source must describe the empty storage it owns after the transfer.
void test_move_leaves_source_empty() {
    Tensor a(2, 3, 1.0f);
    Tensor b(std::move(a));
    ASSERT_EQ(b.rows(), size_t(2));
    ASSERT_EQ(b.cols(), size_t(3));
    ASSERT_EQ(b.size(), size_t(6));
    ASSERT_NEAR(b.at(1, 2), 1.0f);
    ASSERT_EQ(a.rows(), size_t(0));
    ASSERT_EQ(a.cols(), size_t(0));
    ASSERT_EQ(a.size(), size_t(0));

    Tensor c(4, 4);
    Tensor d(2, 3, 5.0f);
    c = std::move(d);
    ASSERT_EQ(c.rows(), size_t(2));
    ASSERT_EQ(c.cols(), size_t(3));
    ASSERT_EQ(c.size(), size_t(6));
    ASSERT_NEAR(c.at(1, 2), 5.0f);
    ASSERT_EQ(d.rows(), size_t(0));
    ASSERT_EQ(d.cols(), size_t(0));
    ASSERT_EQ(d.size(), size_t(0));
}

int main() {
    std::cout << "test_tensor:" << std::endl;
    RUN_TEST(test_default_construction);
    RUN_TEST(test_shape_construction);
    RUN_TEST(test_init_value_construction);
    RUN_TEST(test_at_read_write);
    RUN_TEST(test_at_out_of_bounds);
    RUN_TEST(test_fill);
    RUN_TEST(test_zero);
    RUN_TEST(test_randomize_deterministic);
    RUN_TEST(test_randomize_different_seeds);
    RUN_TEST(test_copy_independence);
    RUN_TEST(test_data_pointer);
    RUN_TEST(test_aligned_allocation);
    RUN_TEST(test_size_overflow_throws);
    RUN_TEST(test_move_leaves_source_empty);
    TEST_SUMMARY("test_tensor");
}
