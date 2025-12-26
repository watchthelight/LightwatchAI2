// Phase 25: Gradient Clipping Tests

#include <lightwatch/optim/clip.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::optim;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Clip by norm with scaling
bool test_phase_25_clip_norm() {
    // Create parameters with gradient that has L2 norm = 2.0
    // 4 elements each with grad = 1.0: sqrt(4 * 1^2) = 2.0
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    Tensor<float> grad_data({4});
    for (size_t i = 0; i < 4; ++i) {
        grad_data.data()[i] = 1.0f;  // norm = 2.0
    }
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    // Clip to max_norm=1.0, should scale by 0.5
    float original_norm = clip_grad_norm_(params, 1.0f);

    // Check original norm was 2.0
    if (!float_eq(original_norm, 2.0f, 0.01f)) {
        std::cerr << "clip_norm: original norm should be 2.0, got " << original_norm << std::endl;
        return false;
    }

    // Check gradients were scaled by 0.5
    for (size_t i = 0; i < 4; ++i) {
        if (!float_eq(param.grad().data()[i], 0.5f, 0.01f)) {
            std::cerr << "clip_norm: grads should be 0.5 after clipping" << std::endl;
            return false;
        }
    }

    // Verify new norm is ~1.0
    float new_norm = grad_norm(params);
    if (!float_eq(new_norm, 1.0f, 0.01f)) {
        std::cerr << "clip_norm: new norm should be 1.0, got " << new_norm << std::endl;
        return false;
    }

    std::cout << "test_phase_25_clip_norm: PASSED" << std::endl;
    return true;
}

// Test 2: Clip by norm with no-op (norm already below max)
bool test_phase_25_clip_norm_no_op() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Set gradient with small norm = 0.5
    // 4 elements each with grad = 0.25: sqrt(4 * 0.25^2) = sqrt(0.25) = 0.5
    Tensor<float> grad_data({4});
    for (size_t i = 0; i < 4; ++i) {
        grad_data.data()[i] = 0.25f;
    }
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    // Clip to max_norm=1.0, should not change grads
    float original_norm = clip_grad_norm_(params, 1.0f);

    // Check original norm
    if (!float_eq(original_norm, 0.5f, 0.01f)) {
        std::cerr << "clip_norm_no_op: original norm should be 0.5, got " << original_norm << std::endl;
        return false;
    }

    // Check gradients are unchanged
    for (size_t i = 0; i < 4; ++i) {
        if (!float_eq(param.grad().data()[i], 0.25f, 0.01f)) {
            std::cerr << "clip_norm_no_op: grads should remain 0.25" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_25_clip_norm_no_op: PASSED" << std::endl;
    return true;
}

// Test 3: Clip by value
bool test_phase_25_clip_value() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Set gradients with values outside [-0.1, 0.1]
    Tensor<float> grad_data({4});
    grad_data.data()[0] = 0.5f;
    grad_data.data()[1] = -0.3f;
    grad_data.data()[2] = 0.05f;  // Already in range
    grad_data.data()[3] = -0.05f; // Already in range
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    clip_grad_value_(params, 0.1f);

    // Check all grads are in [-0.1, 0.1]
    float* g = const_cast<float*>(param.grad().data());
    if (!float_eq(g[0], 0.1f)) {
        std::cerr << "clip_value: g[0] should be 0.1, got " << g[0] << std::endl;
        return false;
    }
    if (!float_eq(g[1], -0.1f)) {
        std::cerr << "clip_value: g[1] should be -0.1, got " << g[1] << std::endl;
        return false;
    }
    if (!float_eq(g[2], 0.05f)) {
        std::cerr << "clip_value: g[2] should remain 0.05, got " << g[2] << std::endl;
        return false;
    }
    if (!float_eq(g[3], -0.05f)) {
        std::cerr << "clip_value: g[3] should remain -0.05, got " << g[3] << std::endl;
        return false;
    }

    std::cout << "test_phase_25_clip_value: PASSED" << std::endl;
    return true;
}

// Test 4: Compute gradient norm (L2)
bool test_phase_25_grad_norm() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Gradient: [3, 4, 0, 0] -> norm = sqrt(9 + 16) = 5
    Tensor<float> grad_data({4});
    grad_data.data()[0] = 3.0f;
    grad_data.data()[1] = 4.0f;
    grad_data.data()[2] = 0.0f;
    grad_data.data()[3] = 0.0f;
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    float norm = grad_norm(params, 2.0f);

    if (!float_eq(norm, 5.0f)) {
        std::cerr << "grad_norm: L2 norm should be 5.0, got " << norm << std::endl;
        return false;
    }

    std::cout << "test_phase_25_grad_norm: PASSED" << std::endl;
    return true;
}

// Test 5: L1 norm
bool test_phase_25_l1_norm() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Gradient: [1, -2, 3, -4] -> L1 norm = 1 + 2 + 3 + 4 = 10
    Tensor<float> grad_data({4});
    grad_data.data()[0] = 1.0f;
    grad_data.data()[1] = -2.0f;
    grad_data.data()[2] = 3.0f;
    grad_data.data()[3] = -4.0f;
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    float norm = grad_norm(params, 1.0f);

    if (!float_eq(norm, 10.0f)) {
        std::cerr << "l1_norm: L1 norm should be 10.0, got " << norm << std::endl;
        return false;
    }

    std::cout << "test_phase_25_l1_norm: PASSED" << std::endl;
    return true;
}

// Test 6: Linf (max) norm
bool test_phase_25_linf_norm() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Gradient: [1, -5, 3, -2] -> Linf norm = 5
    Tensor<float> grad_data({4});
    grad_data.data()[0] = 1.0f;
    grad_data.data()[1] = -5.0f;
    grad_data.data()[2] = 3.0f;
    grad_data.data()[3] = -2.0f;
    param.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param};

    float norm = grad_norm(params, std::numeric_limits<float>::infinity());

    if (!float_eq(norm, 5.0f)) {
        std::cerr << "linf_norm: Linf norm should be 5.0, got " << norm << std::endl;
        return false;
    }

    std::cout << "test_phase_25_linf_norm: PASSED" << std::endl;
    return true;
}

// Test 7: Multiple parameters
bool test_phase_25_multi_param() {
    Tensor<float> param1_data({4});
    Tensor<float> param2_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param1_data.data()[i] = 1.0f;
        param2_data.data()[i] = 2.0f;
    }
    Variable param1(param1_data, true);
    Variable param2(param2_data, true);

    // Each has gradient [1, 1, 1, 1]
    // Total L2 norm = sqrt(8 * 1^2) = sqrt(8) ≈ 2.83
    Tensor<float> grad_data({4});
    for (size_t i = 0; i < 4; ++i) {
        grad_data.data()[i] = 1.0f;
    }
    param1.accumulate_grad(grad_data);
    param2.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param1, &param2};

    float norm = grad_norm(params, 2.0f);
    float expected = std::sqrt(8.0f);

    if (!float_eq(norm, expected, 0.01f)) {
        std::cerr << "multi_param: norm should be " << expected << ", got " << norm << std::endl;
        return false;
    }

    // Clip to 1.0
    clip_grad_norm_(params, 1.0f);

    // Verify new norm is 1.0
    float new_norm = grad_norm(params, 2.0f);
    if (!float_eq(new_norm, 1.0f, 0.01f)) {
        std::cerr << "multi_param: clipped norm should be 1.0, got " << new_norm << std::endl;
        return false;
    }

    std::cout << "test_phase_25_multi_param: PASSED" << std::endl;
    return true;
}

// Test 8: Skip params without gradients
bool test_phase_25_skip_no_grad() {
    Tensor<float> param1_data({4});
    Tensor<float> param2_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param1_data.data()[i] = 1.0f;
        param2_data.data()[i] = 2.0f;
    }
    Variable param1(param1_data, true);
    Variable param2(param2_data, true);  // No grad set

    // Only param1 has gradient
    Tensor<float> grad_data({4});
    for (size_t i = 0; i < 4; ++i) {
        grad_data.data()[i] = 1.0f;  // norm = 2.0
    }
    param1.accumulate_grad(grad_data);

    std::vector<Variable*> params = {&param1, &param2};

    float norm = grad_norm(params, 2.0f);

    // Only param1 contributes: sqrt(4) = 2.0
    if (!float_eq(norm, 2.0f, 0.01f)) {
        std::cerr << "skip_no_grad: norm should be 2.0, got " << norm << std::endl;
        return false;
    }

    // Clipping should not crash
    clip_grad_norm_(params, 1.0f);
    clip_grad_value_(params, 0.5f);

    std::cout << "test_phase_25_skip_no_grad: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 25: Gradient Clipping Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_25_clip_norm()) ++failures;
    if (!test_phase_25_clip_norm_no_op()) ++failures;
    if (!test_phase_25_clip_value()) ++failures;
    if (!test_phase_25_grad_norm()) ++failures;
    if (!test_phase_25_l1_norm()) ++failures;
    if (!test_phase_25_linf_norm()) ++failures;
    if (!test_phase_25_multi_param()) ++failures;
    if (!test_phase_25_skip_no_grad()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 25 tests passed (8/8) ===" << std::endl;
    return 0;
}
