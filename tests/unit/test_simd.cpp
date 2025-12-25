// Phase 04: SIMD Operations Tests

#include <lightwatch/simd/dispatch.hpp>
#include <lightwatch/simd/scalar.hpp>
#include <iostream>
#include <cmath>
#include <vector>

using namespace lightwatch::simd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test backend detection
bool test_phase_04_detect_backend() {
    Backend backend = detect_backend();

    // Just check it returns a valid enum value
    if (backend != Backend::SCALAR &&
        backend != Backend::SSE &&
        backend != Backend::AVX2 &&
        backend != Backend::AVX512) {
        std::cerr << "detect_backend: invalid enum" << std::endl;
        return false;
    }

    std::cout << "test_phase_04_detect_backend: PASSED (backend: "
              << backend_name(backend) << ")" << std::endl;
    return true;
}

// Test SIMD add matches scalar
bool test_phase_04_simd_add() {
    const size_t n = 1024;
    std::vector<float> a(n), b(n), c_simd(n), c_scalar(n);

    // Initialize with test data
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.1f;
        b[i] = static_cast<float>(n - i) * 0.1f;
    }

    // Run both implementations
    add(a.data(), b.data(), c_simd.data(), n);
    scalar::add(a.data(), b.data(), c_scalar.data(), n);

    // Compare results
    for (size_t i = 0; i < n; ++i) {
        if (!float_eq(c_simd[i], c_scalar[i])) {
            std::cerr << "simd_add: mismatch at " << i << ": "
                      << c_simd[i] << " vs " << c_scalar[i] << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_04_simd_add: PASSED" << std::endl;
    return true;
}

// Test SIMD mul matches scalar
bool test_phase_04_simd_mul() {
    const size_t n = 1024;
    std::vector<float> a(n), b(n), c_simd(n), c_scalar(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.01f;
        b[i] = static_cast<float>(i % 10) + 1.0f;
    }

    mul(a.data(), b.data(), c_simd.data(), n);
    scalar::mul(a.data(), b.data(), c_scalar.data(), n);

    for (size_t i = 0; i < n; ++i) {
        if (!float_eq(c_simd[i], c_scalar[i])) {
            std::cerr << "simd_mul: mismatch at " << i << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_04_simd_mul: PASSED" << std::endl;
    return true;
}

// Test SIMD matmul matches scalar
bool test_phase_04_simd_matmul() {
    const size_t M = 32, N = 32, K = 32;
    std::vector<float> a(M * K), b(K * N), c_simd(M * N), c_scalar(M * N);

    // Initialize with test data
    for (size_t i = 0; i < M * K; ++i) {
        a[i] = static_cast<float>(i % 7) * 0.1f;
    }
    for (size_t i = 0; i < K * N; ++i) {
        b[i] = static_cast<float>(i % 11) * 0.1f;
    }

    matmul(a.data(), b.data(), c_simd.data(), M, N, K);
    scalar::matmul(a.data(), b.data(), c_scalar.data(), M, N, K);

    // Compare with higher tolerance for accumulated error
    for (size_t i = 0; i < M * N; ++i) {
        if (!float_eq(c_simd[i], c_scalar[i], 1e-4f)) {
            std::cerr << "simd_matmul: mismatch at " << i << ": "
                      << c_simd[i] << " vs " << c_scalar[i] << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_04_simd_matmul: PASSED" << std::endl;
    return true;
}

// Test SIMD exp matches std::exp
bool test_phase_04_simd_exp() {
    const size_t n = 256;
    std::vector<float> a(n), c_simd(n), c_ref(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.02f - 2.0f;  // Range [-2, ~3]
    }

    exp(a.data(), c_simd.data(), n);

    for (size_t i = 0; i < n; ++i) {
        c_ref[i] = std::exp(a[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        if (!float_eq(c_simd[i], c_ref[i], 1e-4f)) {
            std::cerr << "simd_exp: mismatch at " << i << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_04_simd_exp: PASSED" << std::endl;
    return true;
}

// Test SIMD GELU matches reference
bool test_phase_04_simd_gelu() {
    const size_t n = 256;
    std::vector<float> a(n), c_simd(n);

    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.04f - 5.0f;  // Range [-5, 5]
    }

    gelu(a.data(), c_simd.data(), n);

    // Reference GELU implementation
    auto ref_gelu = [](float x) {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        return 0.5f * x * (1.0f + std::tanh(inner));
    };

    for (size_t i = 0; i < n; ++i) {
        float ref = ref_gelu(a[i]);
        if (!float_eq(c_simd[i], ref, 1e-4f)) {
            std::cerr << "simd_gelu: mismatch at " << i << ": "
                      << c_simd[i] << " vs " << ref << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_04_simd_gelu: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_04_detect_backend()) ++failures;
    if (!test_phase_04_simd_add()) ++failures;
    if (!test_phase_04_simd_mul()) ++failures;
    if (!test_phase_04_simd_matmul()) ++failures;
    if (!test_phase_04_simd_exp()) ++failures;
    if (!test_phase_04_simd_gelu()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 04 tests passed" << std::endl;
    return 0;
}
