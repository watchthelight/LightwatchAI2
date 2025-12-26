// Phase 22: SGD Optimizer Tests

#include <lightwatch/optim/sgd.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::optim;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Basic SGD step
bool test_phase_22_sgd_basic() {
    // Create a simple parameter
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;  // Initial value = 1
    }
    Variable param(param_data, true);

    // Set gradient manually
    Tensor<float> grad_data({4});
    for (size_t i = 0; i < 4; ++i) {
        grad_data.data()[i] = 2.0f;  // Gradient = 2
    }
    param.data().add_(Tensor<float>::zeros(param.shape()));  // Ensure grad exists
    param.accumulate_grad(grad_data);

    // Create optimizer with lr=0.1
    SGDOptions opts;
    opts.lr = 0.1f;
    SGD sgd({&param}, opts);

    // Before step: param = 1.0
    float before = param.data().data()[0];
    if (!float_eq(before, 1.0f)) {
        std::cerr << "sgd_basic: initial param should be 1.0" << std::endl;
        return false;
    }

    // Step: param -= lr * grad = 1.0 - 0.1 * 2.0 = 0.8
    sgd.step();

    float after = param.data().data()[0];
    if (!float_eq(after, 0.8f)) {
        std::cerr << "sgd_basic: after step, param should be 0.8, got " << after << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_basic: PASSED" << std::endl;
    return true;
}

// Test 2: SGD with momentum
bool test_phase_22_sgd_momentum() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 0.1f;
    opts.momentum = 0.9f;
    SGD sgd({&param}, opts);

    // Step 1: Set gradient = 1
    Tensor<float> grad1({4});
    for (size_t i = 0; i < 4; ++i) {
        grad1.data()[i] = 1.0f;
    }
    param.accumulate_grad(grad1);
    sgd.step();

    // After step 1: v = 0.9*0 + 1 = 1
    // param = 1 - 0.1 * 1 = 0.9
    float after1 = param.data().data()[0];

    // Step 2: Set gradient = 1 again
    sgd.zero_grad();
    param.accumulate_grad(grad1);
    sgd.step();

    // After step 2: v = 0.9*1 + 1 = 1.9
    // param = 0.9 - 0.1 * 1.9 = 0.71
    float after2 = param.data().data()[0];

    // Momentum should cause larger update in second step
    float update1 = 1.0f - after1;   // ≈ 0.1
    float update2 = after1 - after2; // ≈ 0.19

    if (update2 <= update1) {
        std::cerr << "sgd_momentum: second update should be larger than first" << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_momentum: PASSED" << std::endl;
    return true;
}

// Test 3: SGD with weight decay
bool test_phase_22_sgd_weight_decay() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 10.0f;  // Large initial value
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 0.1f;
    opts.weight_decay = 0.1f;  // 10% weight decay
    SGD sgd({&param}, opts);

    // Set zero gradient - only weight decay should affect param
    Tensor<float> grad({4});
    grad.zero_();
    param.accumulate_grad(grad);

    float before = param.data().data()[0];
    sgd.step();
    float after = param.data().data()[0];

    // Weight decay: param -= lr * wd * param = 10 - 0.1 * 0.1 * 10 = 9.9
    float expected = before - opts.lr * opts.weight_decay * before;
    if (!float_eq(after, expected, 0.001f)) {
        std::cerr << "sgd_weight_decay: expected " << expected << ", got " << after << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_weight_decay: PASSED" << std::endl;
    return true;
}

// Test 4: SGD with Nesterov momentum
bool test_phase_22_sgd_nesterov() {
    Tensor<float> param_data1({4});
    Tensor<float> param_data2({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data1.data()[i] = 1.0f;
        param_data2.data()[i] = 1.0f;
    }
    Variable param_regular(param_data1, true);
    Variable param_nesterov(param_data2, true);

    SGDOptions opts_regular;
    opts_regular.lr = 0.1f;
    opts_regular.momentum = 0.9f;
    opts_regular.nesterov = false;

    SGDOptions opts_nesterov;
    opts_nesterov.lr = 0.1f;
    opts_nesterov.momentum = 0.9f;
    opts_nesterov.nesterov = true;

    SGD sgd_regular({&param_regular}, opts_regular);
    SGD sgd_nesterov({&param_nesterov}, opts_nesterov);

    // Run a few steps
    Tensor<float> grad({4});
    for (size_t i = 0; i < 4; ++i) {
        grad.data()[i] = 1.0f;
    }

    for (int step = 0; step < 3; ++step) {
        sgd_regular.zero_grad();
        sgd_nesterov.zero_grad();
        param_regular.accumulate_grad(grad);
        param_nesterov.accumulate_grad(grad);
        sgd_regular.step();
        sgd_nesterov.step();
    }

    // Nesterov and regular momentum should give different results
    float regular = param_regular.data().data()[0];
    float nesterov = param_nesterov.data().data()[0];

    if (float_eq(regular, nesterov)) {
        std::cerr << "sgd_nesterov: Nesterov should differ from regular momentum" << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_nesterov: PASSED" << std::endl;
    return true;
}

// Test 5: zero_grad functionality
bool test_phase_22_zero_grad() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    // Set gradient
    Tensor<float> grad({4});
    for (size_t i = 0; i < 4; ++i) {
        grad.data()[i] = 5.0f;
    }
    param.accumulate_grad(grad);

    // Verify gradient is set
    if (!param.has_grad()) {
        std::cerr << "zero_grad: param should have grad before zero_grad" << std::endl;
        return false;
    }

    SGDOptions opts;
    opts.lr = 0.1f;
    SGD sgd({&param}, opts);

    // Zero gradients
    sgd.zero_grad();

    // Gradient should now be zero
    if (param.has_grad()) {
        for (size_t i = 0; i < 4; ++i) {
            if (param.grad().data()[i] != 0.0f) {
                std::cerr << "zero_grad: grad should be 0 after zero_grad" << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_22_zero_grad: PASSED" << std::endl;
    return true;
}

// Test 6: Multiple parameters
bool test_phase_22_sgd_multi_param() {
    Tensor<float> param1_data({4});
    Tensor<float> param2_data({8});
    for (size_t i = 0; i < 4; ++i) {
        param1_data.data()[i] = 1.0f;
    }
    for (size_t i = 0; i < 8; ++i) {
        param2_data.data()[i] = 2.0f;
    }

    Variable param1(param1_data, true);
    Variable param2(param2_data, true);

    SGDOptions opts;
    opts.lr = 0.5f;
    SGD sgd({&param1, &param2}, opts);

    // Set gradients
    Tensor<float> grad1({4});
    Tensor<float> grad2({8});
    for (size_t i = 0; i < 4; ++i) {
        grad1.data()[i] = 1.0f;
    }
    for (size_t i = 0; i < 8; ++i) {
        grad2.data()[i] = 0.5f;
    }
    param1.accumulate_grad(grad1);
    param2.accumulate_grad(grad2);

    sgd.step();

    // param1 = 1 - 0.5 * 1 = 0.5
    // param2 = 2 - 0.5 * 0.5 = 1.75
    if (!float_eq(param1.data().data()[0], 0.5f)) {
        std::cerr << "sgd_multi_param: param1 should be 0.5" << std::endl;
        return false;
    }
    if (!float_eq(param2.data().data()[0], 1.75f)) {
        std::cerr << "sgd_multi_param: param2 should be 1.75" << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_multi_param: PASSED" << std::endl;
    return true;
}

// Test 7: Learning rate adjustment
bool test_phase_22_sgd_lr_adjust() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 0.1f;
    SGD sgd({&param}, opts);

    // Verify initial LR
    if (!float_eq(sgd.get_lr(), 0.1f)) {
        std::cerr << "sgd_lr_adjust: initial LR should be 0.1" << std::endl;
        return false;
    }

    // Change LR
    sgd.set_lr(0.01f);

    if (!float_eq(sgd.get_lr(), 0.01f)) {
        std::cerr << "sgd_lr_adjust: LR should be 0.01 after set_lr" << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_lr_adjust: PASSED" << std::endl;
    return true;
}

// Test 8: Skip parameters without gradients
bool test_phase_22_sgd_no_grad() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 0.5f;
    SGD sgd({&param}, opts);

    // Don't set gradient - step should not change param
    float before = param.data().data()[0];
    sgd.step();
    float after = param.data().data()[0];

    if (!float_eq(before, after)) {
        std::cerr << "sgd_no_grad: param should not change without gradient" << std::endl;
        return false;
    }

    std::cout << "test_phase_22_sgd_no_grad: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 22: SGD Optimizer Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_22_sgd_basic()) ++failures;
    if (!test_phase_22_sgd_momentum()) ++failures;
    if (!test_phase_22_sgd_weight_decay()) ++failures;
    if (!test_phase_22_sgd_nesterov()) ++failures;
    if (!test_phase_22_zero_grad()) ++failures;
    if (!test_phase_22_sgd_multi_param()) ++failures;
    if (!test_phase_22_sgd_lr_adjust()) ++failures;
    if (!test_phase_22_sgd_no_grad()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 22 tests passed (8/8) ===" << std::endl;
    return 0;
}
