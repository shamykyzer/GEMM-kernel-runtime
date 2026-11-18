#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

static int g_test_failures = 0;

#define ASSERT_TRUE(expr)                                                  \
    do {                                                                   \
        if (!(expr)) {                                                     \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #expr << std::endl;                          \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define ASSERT_EQ(a, b)                                                    \
    do {                                                                   \
        if ((a) != (b)) {                                                  \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #a " == " #b << "  (" << (a)                 \
                      << " vs " << (b) << ")" << std::endl;               \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

constexpr float TEST_TOLERANCE = 1e-4f;

// True when a and b are both finite and differ by at most `tolerance`,
// measured relative to the larger magnitude (absolute for values below 1).
//
// NaN and infinity never count as near, not even to themselves. Every ordered
// comparison with NaN is false, so the result has to be phrased as "finite and
// within tolerance". Phrasing it as "not too far apart" lets NaN through, and
// a kernel that produced NaN would then pass every test.
inline bool nearly_equal(float a, float b, float tolerance = TEST_TOLERANCE) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
        return false;
    }
    const float diff = std::fabs(a - b);
    const float scale = std::fmax(1.0f, std::fmax(std::fabs(a), std::fabs(b)));
    return diff <= tolerance * scale;
}

#define ASSERT_NEAR(a, b)                                                  \
    do {                                                                   \
        float _a = (a), _b = (b);                                         \
        if (!nearly_equal(_a, _b)) {                                       \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  " #a " ~= " #b << "  (" << _a                 \
                      << " vs " << _b << ", diff="                         \
                      << std::fabs(_a - _b) << ")" << std::endl;           \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define ASSERT_THROWS(expr, exception_type)                                \
    do {                                                                   \
        bool _caught = false;                                              \
        try { expr; } catch (const exception_type&) { _caught = true; }    \
        if (!_caught) {                                                    \
            std::cerr << "FAIL: " << __FILE__ << ":" << __LINE__           \
                      << "  expected " #exception_type " from " #expr      \
                      << std::endl;                                        \
            ++g_test_failures;                                             \
        }                                                                  \
    } while (0)

#define RUN_TEST(fn)                                                       \
    do {                                                                   \
        std::cout << "  " #fn "... ";                                      \
        int _before = g_test_failures;                                     \
        fn();                                                              \
        std::cout << (g_test_failures == _before ? "OK" : "FAILED")        \
                  << std::endl;                                            \
    } while (0)

#define TEST_SUMMARY(suite)                                                \
    do {                                                                   \
        if (g_test_failures == 0)                                          \
            std::cout << suite ": all tests passed." << std::endl;         \
        else                                                               \
            std::cerr << suite ": " << g_test_failures                     \
                      << " failure(s)." << std::endl;                      \
        return g_test_failures == 0 ? 0 : 1;                              \
    } while (0)
