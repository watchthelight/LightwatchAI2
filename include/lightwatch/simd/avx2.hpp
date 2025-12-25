#pragma once

#include <lightwatch/config.hpp>

#if LIGHTWATCH_HAS_AVX2

#include <cstddef>
#include <immintrin.h>
#include <limits>

namespace lightwatch::simd::avx2 {

// Process 8 floats at a time with AVX2

// Matrix multiplication with AVX2 and cache blocking
inline void matmul(const float* a, const float* b, float* c,
                   size_t M, size_t N, size_t K) {
    // Initialize C to zero
    for (size_t i = 0; i < M * N; ++i) {
        c[i] = 0.0f;
    }

    constexpr size_t BLOCK = 64;  // Larger block for AVX2

    // Blocked matrix multiply
    for (size_t ii = 0; ii < M; ii += BLOCK) {
        for (size_t kk = 0; kk < K; kk += BLOCK) {
            for (size_t jj = 0; jj < N; jj += BLOCK) {
                size_t i_end = (ii + BLOCK < M) ? ii + BLOCK : M;
                size_t k_end = (kk + BLOCK < K) ? kk + BLOCK : K;
                size_t j_end = (jj + BLOCK < N) ? jj + BLOCK : N;

                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        __m256 a_ik = _mm256_set1_ps(a[i * K + k]);

                        size_t j = jj;
                        // Process 8 elements at a time
                        for (; j + 8 <= j_end; j += 8) {
                            __m256 b_kj = _mm256_loadu_ps(&b[k * N + j]);
                            __m256 c_ij = _mm256_loadu_ps(&c[i * N + j]);
                            c_ij = _mm256_fmadd_ps(a_ik, b_kj, c_ij);
                            _mm256_storeu_ps(&c[i * N + j], c_ij);
                        }
                        // Handle remainder with scalar
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
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Element-wise multiplication
inline void mul(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// Scale
inline void scale(const float* a, float scalar, float* c, size_t n) {
    __m256 vs = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(c + i, _mm256_mul_ps(va, vs));
    }
    for (; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
}

// ReLU
inline void relu(const float* a, float* c, size_t n) {
    __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(c + i, _mm256_max_ps(zero, va));
    }
    for (; i < n; ++i) {
        c[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
    }
}

// Sum reduction
inline float sum(const float* a, size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vsum = _mm256_add_ps(vsum, va);
    }

    // Horizontal sum: reduce 8 floats to 1
    __m128 hi = _mm256_extractf128_ps(vsum, 1);
    __m128 lo = _mm256_castps256_ps128(vsum);
    __m128 sum4 = _mm_add_ps(hi, lo);
    __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));

    float total = _mm_cvtss_f32(sum1);

    for (; i < n; ++i) {
        total += a[i];
    }
    return total;
}

// Max reduction
inline float max(const float* a, size_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();

    __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        vmax = _mm256_max_ps(vmax, va);
    }

    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(vmax, 1);
    __m128 lo = _mm256_castps256_ps128(vmax);
    __m128 max4 = _mm_max_ps(hi, lo);
    __m128 max2 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
    __m128 max1 = _mm_max_ss(max2, _mm_shuffle_ps(max2, max2, 1));

    float m = _mm_cvtss_f32(max1);

    for (; i < n; ++i) {
        if (a[i] > m) m = a[i];
    }
    return m;
}

}  // namespace lightwatch::simd::avx2

#endif  // LIGHTWATCH_HAS_AVX2
