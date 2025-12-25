// Phase 03: Tensor Core Tests

#include <lightwatch/tensor.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace lightwatch;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test tensor creation
bool test_phase_03_tensor_create() {
    Tensor<float> t({2, 3});

    if (t.numel() != 6) {
        std::cerr << "tensor_create: numel != 6" << std::endl;
        return false;
    }
    if (t.ndim() != 2) {
        std::cerr << "tensor_create: ndim != 2" << std::endl;
        return false;
    }
    if (t.size(0) != 2 || t.size(1) != 3) {
        std::cerr << "tensor_create: shape mismatch" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_create: PASSED" << std::endl;
    return true;
}

// Test zeros factory
bool test_phase_03_tensor_zeros() {
    auto t = Tensor<float>::zeros({3, 4});

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (t({i, j}) != 0.0f) {
                std::cerr << "tensor_zeros: non-zero element" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_zeros: PASSED" << std::endl;
    return true;
}

// Test ones factory
bool test_phase_03_tensor_ones() {
    auto t = Tensor<float>::ones({2, 2});

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (t({i, j}) != 1.0f) {
                std::cerr << "tensor_ones: non-one element" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_ones: PASSED" << std::endl;
    return true;
}

// Test element access
bool test_phase_03_tensor_element_access() {
    auto t = Tensor<float>::zeros({3, 4});

    t({0, 1}) = 5.0f;

    if (!float_eq(t.at({0, 1}), 5.0f)) {
        std::cerr << "tensor_element_access: value mismatch" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_element_access: PASSED" << std::endl;
    return true;
}

// Test reshape
bool test_phase_03_tensor_reshape() {
    auto t = Tensor<float>::ones({2, 3});
    t({0, 0}) = 1.0f;
    t({0, 1}) = 2.0f;
    t({0, 2}) = 3.0f;
    t({1, 0}) = 4.0f;
    t({1, 1}) = 5.0f;
    t({1, 2}) = 6.0f;

    auto r = t.reshape({3, 2});

    if (r.shape() != Shape{3, 2}) {
        std::cerr << "tensor_reshape: wrong shape" << std::endl;
        return false;
    }
    if (r.numel() != 6) {
        std::cerr << "tensor_reshape: wrong numel" << std::endl;
        return false;
    }

    // Check data is preserved (row-major order)
    if (!float_eq(r({0, 0}), 1.0f) || !float_eq(r({0, 1}), 2.0f)) {
        std::cerr << "tensor_reshape: data not preserved" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_reshape: PASSED" << std::endl;
    return true;
}

// Test transpose
bool test_phase_03_tensor_transpose() {
    auto t = Tensor<float>::zeros({2, 3});
    t({0, 0}) = 1.0f;
    t({0, 1}) = 2.0f;
    t({0, 2}) = 3.0f;
    t({1, 0}) = 4.0f;
    t({1, 1}) = 5.0f;
    t({1, 2}) = 6.0f;

    auto r = t.transpose(0, 1);

    if (r.shape() != Shape{3, 2}) {
        std::cerr << "tensor_transpose: wrong shape" << std::endl;
        return false;
    }
    if (!float_eq(r({0, 0}), 1.0f) || !float_eq(r({1, 0}), 2.0f) || !float_eq(r({0, 1}), 4.0f)) {
        std::cerr << "tensor_transpose: wrong values" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_transpose: PASSED" << std::endl;
    return true;
}

// Test slice
bool test_phase_03_tensor_slice() {
    auto t = Tensor<float>::zeros({4, 3});
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            t({i, j}) = static_cast<float>(i * 3 + j);
        }
    }

    auto s = t.slice(0, 1, 3);

    if (s.shape() != Shape{2, 3}) {
        std::cerr << "tensor_slice: wrong shape " << s.size(0) << "x" << s.size(1) << std::endl;
        return false;
    }
    if (!float_eq(s({0, 0}), 3.0f) || !float_eq(s({1, 0}), 6.0f)) {
        std::cerr << "tensor_slice: wrong values" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_slice: PASSED" << std::endl;
    return true;
}

// Test addition
bool test_phase_03_tensor_add() {
    auto a = Tensor<float>::ones({2, 3});
    auto b = Tensor<float>::ones({2, 3});
    auto c = a + b;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!float_eq(c({i, j}), 2.0f)) {
                std::cerr << "tensor_add: wrong value" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_add: PASSED" << std::endl;
    return true;
}

// Test scalar multiplication
bool test_phase_03_tensor_mul() {
    auto t = Tensor<float>::ones({2, 3});
    auto r = t * 2.0f;

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!float_eq(r({i, j}), 2.0f)) {
                std::cerr << "tensor_mul: wrong value" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_mul: PASSED" << std::endl;
    return true;
}

// Test 2D matmul
bool test_phase_03_tensor_matmul_2d() {
    auto a = Tensor<float>::ones({2, 3});
    auto b = Tensor<float>::ones({3, 4});
    auto c = matmul(a, b);

    if (c.shape() != Shape{2, 4}) {
        std::cerr << "tensor_matmul_2d: wrong shape" << std::endl;
        return false;
    }

    // ones @ ones = 3.0 (sum of 3 ones)
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (!float_eq(c({i, j}), 3.0f)) {
                std::cerr << "tensor_matmul_2d: wrong value at " << i << "," << j << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_matmul_2d: PASSED" << std::endl;
    return true;
}

// Test sum reduction
bool test_phase_03_tensor_sum() {
    auto t = Tensor<float>::ones({2, 3});
    auto s = t.sum(1);

    if (s.shape() != Shape{2}) {
        std::cerr << "tensor_sum: wrong shape" << std::endl;
        return false;
    }
    if (!float_eq(s({0}), 3.0f) || !float_eq(s({1}), 3.0f)) {
        std::cerr << "tensor_sum: wrong values" << std::endl;
        return false;
    }

    std::cout << "test_phase_03_tensor_sum: PASSED" << std::endl;
    return true;
}

// Test broadcasting
bool test_phase_03_tensor_broadcast() {
    auto a = Tensor<float>::ones({2, 1});
    auto b = Tensor<float>::ones({1, 3});
    auto c = a + b;

    if (c.shape() != Shape{2, 3}) {
        std::cerr << "tensor_broadcast: wrong shape" << std::endl;
        return false;
    }

    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (!float_eq(c({i, j}), 2.0f)) {
                std::cerr << "tensor_broadcast: wrong value" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_03_tensor_broadcast: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_03_tensor_create()) ++failures;
    if (!test_phase_03_tensor_zeros()) ++failures;
    if (!test_phase_03_tensor_ones()) ++failures;
    if (!test_phase_03_tensor_element_access()) ++failures;
    if (!test_phase_03_tensor_reshape()) ++failures;
    if (!test_phase_03_tensor_transpose()) ++failures;
    if (!test_phase_03_tensor_slice()) ++failures;
    if (!test_phase_03_tensor_add()) ++failures;
    if (!test_phase_03_tensor_mul()) ++failures;
    if (!test_phase_03_tensor_matmul_2d()) ++failures;
    if (!test_phase_03_tensor_sum()) ++failures;
    if (!test_phase_03_tensor_broadcast()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 03 tests passed (12/12)" << std::endl;
    return 0;
}
