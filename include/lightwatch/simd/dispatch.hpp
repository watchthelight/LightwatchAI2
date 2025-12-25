#pragma once

#include <cstddef>
#include <cmath>
#include <lightwatch/config.hpp>
#include <lightwatch/simd/scalar.hpp>

#if LIGHTWATCH_ARCH_X64
  #if LIGHTWATCH_HAS_AVX2
    #include <lightwatch/simd/avx2.hpp>
  #endif
  #if LIGHTWATCH_HAS_SSE4 || LIGHTWATCH_ARCH_X64
    // SSE available on all x64
  #endif
#endif

namespace lightwatch::simd {

enum class Backend {
    SCALAR,
    SSE,
    AVX2,
    AVX512
};

// Detect best available backend at runtime
inline Backend detect_backend() {
#if LIGHTWATCH_ARCH_X64
    #if LIGHTWATCH_HAS_AVX2
        return Backend::AVX2;
    #elif LIGHTWATCH_HAS_SSE4
        return Backend::SSE;
    #else
        return Backend::SCALAR;
    #endif
#else
    // ARM or other architectures - use scalar
    return Backend::SCALAR;
#endif
}

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::SCALAR: return "Scalar";
        case Backend::SSE: return "SSE";
        case Backend::AVX2: return "AVX2";
        case Backend::AVX512: return "AVX512";
        default: return "Unknown";
    }
}

// Get singleton backend (detected once at first call)
inline Backend get_backend() {
    static Backend backend = detect_backend();
    return backend;
}

// Dispatch functions - route to best implementation

inline void matmul(const float* a, const float* b, float* c,
                   size_t M, size_t N, size_t K) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        avx2::matmul(a, b, c, M, N, K);
        return;
    }
#endif
    // Fall back to scalar (handles all cases including ARM)
    scalar::matmul(a, b, c, M, N, K);
}

inline void add(const float* a, const float* b, float* c, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        avx2::add(a, b, c, n);
        return;
    }
#endif
    scalar::add(a, b, c, n);
}

inline void mul(const float* a, const float* b, float* c, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        avx2::mul(a, b, c, n);
        return;
    }
#endif
    scalar::mul(a, b, c, n);
}

inline void scale(const float* a, float s, float* c, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        avx2::scale(a, s, c, n);
        return;
    }
#endif
    scalar::scale(a, s, c, n);
}

inline void exp(const float* a, float* c, size_t n) {
    // exp is not trivially vectorizable without approximations
    // Using scalar implementation for accuracy
    scalar::exp(a, c, n);
}

inline void relu(const float* a, float* c, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        avx2::relu(a, c, n);
        return;
    }
#endif
    scalar::relu(a, c, n);
}

inline void gelu(const float* a, float* c, size_t n) {
    // GELU uses tanh which is complex to vectorize
    // Using scalar implementation for accuracy
    scalar::gelu(a, c, n);
}

inline float sum(const float* a, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        return avx2::sum(a, n);
    }
#endif
    return scalar::sum(a, n);
}

inline float max(const float* a, size_t n) {
#if LIGHTWATCH_HAS_AVX2
    if (get_backend() == Backend::AVX2) {
        return avx2::max(a, n);
    }
#endif
    return scalar::max(a, n);
}

}  // namespace lightwatch::simd
