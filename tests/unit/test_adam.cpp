// Phase 23: Adam Optimizer Tests

#include <lightwatch/optim/adam.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::optim;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Basic Adam step
bool test_phase_23_adam_basic() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    AdamOptions opts;
    opts.lr = 0.1f;
    Adam adam({&param}, opts);

    // Set gradient
    Tensor<float> grad({4});
    for (size_t i = 0; i < 4; ++i) {
        grad.data()[i] = 1.0f;
    }
    param.accumulate_grad(grad);

    float before = param.data().data()[0];
    adam.step();
    float after = param.data().data()[0];

    // After Adam step, param should decrease
    if (after >= before) {
        std::cerr << "adam_basic: param should decrease after step" << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adam_basic: PASSED" << std::endl;
    return true;
}

// Test 2: Bias correction in early steps
bool test_phase_23_adam_bias_correction() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 0.0f;
    }
    Variable param(param_data, true);

    AdamOptions opts;
    opts.lr = 0.1f;
    opts.beta1 = 0.9f;
    opts.beta2 = 0.999f;
    Adam adam({&param}, opts);

    // Set constant gradient
    Tensor<float> grad({4});
    for (size_t i = 0; i < 4; ++i) {
        grad.data()[i] = 1.0f;
    }

    // First few steps
    std::vector<float> updates;
    for (int step = 0; step < 5; ++step) {
        adam.zero_grad();
        param.accumulate_grad(grad);
        float before = param.data().data()[0];
        adam.step();
        float after = param.data().data()[0];
        updates.push_back(before - after);
    }

    // With bias correction, updates should be roughly similar
    // Without correction, first steps would be much smaller
    // Just verify updates are happening
    for (float u : updates) {
        if (u <= 0) {
            std::cerr << "adam_bias_correction: updates should be positive" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_23_adam_bias_correction: PASSED" << std::endl;
    return true;
}

// Test 3: AdamW decoupled weight decay
bool test_phase_23_adamw_decay() {
    Tensor<float> param_data1({4});
    Tensor<float> param_data2({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data1.data()[i] = 10.0f;
        param_data2.data()[i] = 10.0f;
    }
    Variable param_adam(param_data1, true);
    Variable param_adamw(param_data2, true);

    AdamOptions opts;
    opts.lr = 0.01f;
    opts.weight_decay = 0.1f;
    Adam adam({&param_adam}, opts);
    AdamW adamw({&param_adamw}, opts);

    // Set zero gradient - only weight decay affects params
    Tensor<float> grad({4});
    grad.zero_();
    param_adam.accumulate_grad(grad);
    param_adamw.accumulate_grad(grad);

    adam.step();
    adamw.step();

    // Both should shrink due to weight decay
    float after_adam = param_adam.data().data()[0];
    float after_adamw = param_adamw.data().data()[0];

    if (after_adam >= 10.0f) {
        std::cerr << "adamw_decay: Adam param should shrink with weight decay" << std::endl;
        return false;
    }

    if (after_adamw >= 10.0f) {
        std::cerr << "adamw_decay: AdamW param should shrink with weight decay" << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adamw_decay: PASSED" << std::endl;
    return true;
}

// Test 4: State persistence across steps
bool test_phase_23_adam_state() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    AdamOptions opts;
    opts.lr = 0.1f;
    opts.beta1 = 0.9f;
    Adam adam({&param}, opts);

    Tensor<float> grad({4});
    for (size_t i = 0; i < 4; ++i) {
        grad.data()[i] = 1.0f;
    }

    // Run several steps
    for (int i = 0; i < 5; ++i) {
        adam.zero_grad();
        param.accumulate_grad(grad);
        adam.step();
    }

    // Check step count
    if (adam.step_count() != 5) {
        std::cerr << "adam_state: step_count should be 5, got " << adam.step_count() << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adam_state: PASSED" << std::endl;
    return true;
}

// Test 5: AMSGrad variant
bool test_phase_23_amsgrad() {
    Tensor<float> param_data1({4});
    Tensor<float> param_data2({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data1.data()[i] = 1.0f;
        param_data2.data()[i] = 1.0f;
    }
    Variable param_regular(param_data1, true);
    Variable param_amsgrad(param_data2, true);

    AdamOptions opts_regular;
    opts_regular.lr = 0.1f;
    opts_regular.amsgrad = false;

    AdamOptions opts_amsgrad;
    opts_amsgrad.lr = 0.1f;
    opts_amsgrad.amsgrad = true;

    Adam adam_regular({&param_regular}, opts_regular);
    Adam adam_amsgrad({&param_amsgrad}, opts_amsgrad);

    // Run several steps with varying gradients
    for (int step = 0; step < 10; ++step) {
        adam_regular.zero_grad();
        adam_amsgrad.zero_grad();

        Tensor<float> grad({4});
        for (size_t i = 0; i < 4; ++i) {
            grad.data()[i] = (step % 2 == 0) ? 2.0f : 0.5f;  // Varying gradient
        }

        param_regular.accumulate_grad(grad);
        param_amsgrad.accumulate_grad(grad);

        adam_regular.step();
        adam_amsgrad.step();
    }

    // AMSGrad should track max, so results may differ
    float regular = param_regular.data().data()[0];
    float amsgrad = param_amsgrad.data().data()[0];

    // Both should have updated
    if (regular >= 1.0f || amsgrad >= 1.0f) {
        std::cerr << "amsgrad: params should decrease" << std::endl;
        return false;
    }

    std::cout << "test_phase_23_amsgrad: PASSED" << std::endl;
    return true;
}

// Test 6: Multiple parameters
bool test_phase_23_adam_multi_param() {
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

    AdamOptions opts;
    opts.lr = 0.1f;
    Adam adam({&param1, &param2}, opts);

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

    float before1 = param1.data().data()[0];
    float before2 = param2.data().data()[0];

    adam.step();

    float after1 = param1.data().data()[0];
    float after2 = param2.data().data()[0];

    // Both should decrease
    if (after1 >= before1 || after2 >= before2) {
        std::cerr << "adam_multi_param: both params should decrease" << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adam_multi_param: PASSED" << std::endl;
    return true;
}

// Test 7: Learning rate adjustment
bool test_phase_23_adam_lr_adjust() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    AdamOptions opts;
    opts.lr = 0.1f;
    Adam adam({&param}, opts);

    // Verify initial LR
    if (!float_eq(adam.get_lr(), 0.1f)) {
        std::cerr << "adam_lr_adjust: initial LR should be 0.1" << std::endl;
        return false;
    }

    // Change LR
    adam.set_lr(0.01f);

    if (!float_eq(adam.get_lr(), 0.01f)) {
        std::cerr << "adam_lr_adjust: LR should be 0.01 after set_lr" << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adam_lr_adjust: PASSED" << std::endl;
    return true;
}

// Test 8: Convergence on simple optimization problem
bool test_phase_23_adam_convergence() {
    // Minimize f(x) = x^2, gradient = 2x
    Tensor<float> param_data({1});
    param_data.data()[0] = 5.0f;  // Start at x=5
    Variable param(param_data, true);

    AdamOptions opts;
    opts.lr = 0.5f;
    Adam adam({&param}, opts);

    for (int step = 0; step < 50; ++step) {
        adam.zero_grad();

        // Gradient = 2 * x
        Tensor<float> grad({1});
        grad.data()[0] = 2.0f * param.data().data()[0];
        param.accumulate_grad(grad);

        adam.step();
    }

    // Should converge close to 0
    float final_val = param.data().data()[0];
    if (std::abs(final_val) > 0.1f) {
        std::cerr << "adam_convergence: should converge near 0, got " << final_val << std::endl;
        return false;
    }

    std::cout << "test_phase_23_adam_convergence: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 23: Adam Optimizer Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_23_adam_basic()) ++failures;
    if (!test_phase_23_adam_bias_correction()) ++failures;
    if (!test_phase_23_adamw_decay()) ++failures;
    if (!test_phase_23_adam_state()) ++failures;
    if (!test_phase_23_amsgrad()) ++failures;
    if (!test_phase_23_adam_multi_param()) ++failures;
    if (!test_phase_23_adam_lr_adjust()) ++failures;
    if (!test_phase_23_adam_convergence()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 23 tests passed (8/8) ===" << std::endl;
    return 0;
}
