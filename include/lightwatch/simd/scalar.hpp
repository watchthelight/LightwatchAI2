#pragma once

#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>

namespace lightwatch::simd::scalar {

// Matrix multiplication: C = A @ B
// A is MxK, B is KxN, C is MxN
inline void matmul(const float* a, const float* b, float* c,
                   size_t M, size_t N, size_t K) {
    // Initialize C to zero
    for (size_t i = 0; i < M * N; ++i) {
        c[i] = 0.0f;
    }

    // Naive triple loop with cache-friendly ordering (ikj)
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a_ik = a[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                c[i * N + j] += a_ik * b[k * N + j];
            }
        }
    }
}

// Element-wise addition: c = a + b
inline void add(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// Element-wise multiplication: c = a * b
inline void mul(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// Scale: c = a * scalar
inline void scale(const float* a, float scalar, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] * scalar;
    }
}

// Exponential: c = exp(a)
inline void exp(const float* a, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = std::exp(a[i]);
    }
}

// ReLU: c = max(0, a)
inline void relu(const float* a, float* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = std::max(0.0f, a[i]);
    }
}

// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
inline void gelu(const float* a, float* c, size_t n) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float coeff = 0.044715f;

    for (size_t i = 0; i < n; ++i) {
        float x = a[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        c[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

// Sum reduction
inline float sum(const float* a, size_t n) {
    float result = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        result += a[i];
    }
    return result;
}

// Max reduction
inline float max(const float* a, size_t n) {
    if (n == 0) return -std::numeric_limits<float>::infinity();
    float result = a[0];
    for (size_t i = 1; i < n; ++i) {
        if (a[i] > result) result = a[i];
    }
    return result;
}

}  // namespace lightwatch::simd::scalar
