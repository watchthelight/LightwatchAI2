#pragma once

#include <lightwatch/config.hpp>

#if LIGHTWATCH_HAS_SSE4 || (LIGHTWATCH_ARCH_X64 && !LIGHTWATCH_PLATFORM_MACOS)

#include <cstddef>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4.1

namespace lightwatch::simd::sse {

// Process 4 floats at a time with SSE

// Matrix multiplication with SSE
// Uses blocking for better cache utilization
inline void matmul(const float* a, const float* b, float* c,
                   size_t M, size_t N, size_t K) {
    // Initialize C to zero
    for (size_t i = 0; i < M * N; ++i) {
        c[i] = 0.0f;
    }

    constexpr size_t BLOCK = 32;

    // Blocked matrix multiply
    for (size_t ii = 0; ii < M; ii += BLOCK) {
        for (size_t kk = 0; kk < K; kk += BLOCK) {
            for (size_t jj = 0; jj < N; jj += BLOCK) {
                size_t i_end = (ii + BLOCK < M) ? ii + BLOCK : M;
                size_t k_end = (kk + BLOCK < K) ? kk + BLOCK : K;
                size_t j_end = (jj + BLOCK < N) ? jj + BLOCK : N;

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        __m128 a_ik = _mm_set1_ps(a[i * K + k]);

                        size_t j = jj;
                        // Process 4 elements at a time
                        for (; j + 4 <= j_end; j += 4) {
                            __m128 b_kj = _mm_loadu_ps(&b[k * N + j]);
                            __m128 c_ij = _mm_loadu_ps(&c[i * N + j]);
                            c_ij = _mm_add_ps(c_ij, _mm_mul_ps(a_ik, b_kj));
                            _mm_storeu_ps(&c[i * N + j], c_ij);
                        }
                        // Handle remainder
                        for (; j < j_end; ++j) {
                            c[i * N + j] += a[i * K + k] * b[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// Element-wise addition
inline void add(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Element-wise multiplication
inline void mul(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(c + i, _mm_mul_ps(va, vb));
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// Scale
inline void scale(const float* a, float scalar, float* c, size_t n) {
    __m128 vs = _mm_set1_ps(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(c + i, _mm_mul_ps(va, vs));
    }
    for (; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
}

// ReLU
inline void relu(const float* a, float* c, size_t n) {
    __m128 zero = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(c + i, _mm_max_ps(zero, va));
    }
    for (; i < n; ++i) {
        c[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
    }
}

// Sum reduction
inline float sum(const float* a, size_t n) {
    __m128 vsum = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        vsum = _mm_add_ps(vsum, va);
    }

    // Horizontal sum
    float result[4];
    _mm_storeu_ps(result, vsum);
    float total = result[0] + result[1] + result[2] + result[3];

    for (; i < n; ++i) {
        total += a[i];
    }
    return total;
}

// Max reduction
inline float max(const float* a, size_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    __m128 vmax = _mm_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        vmax = _mm_max_ps(vmax, va);
    }

    // Horizontal max
    float result[4];
    _mm_storeu_ps(result, vmax);
    float m = result[0];
    if (result[1] > m) m = result[1];
    if (result[2] > m) m = result[2];
    if (result[3] > m) m = result[3];

    for (; i < n; ++i) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

}  // namespace lightwatch::simd::sse

#endif  // LIGHTWATCH_HAS_SSE4
