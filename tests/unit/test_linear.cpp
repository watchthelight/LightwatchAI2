// Phase 11: Linear Layer Tests

#include <lightwatch/nn/linear.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: Basic 2D shape
bool test_phase_11_linear_shape() {
    Linear linear(10, 5);

    // Input: {2, 10}
    Tensor<float> input_data = Tensor<float>::randn({2, 10});
    Variable input(input_data, true);

    auto output = linear.forward(input);

    // Output should be {2, 5}
    if (output.shape().size() != 2) {
        std::cerr << "linear_shape: expected 2D output" << std::endl;
        return false;
    }

    if (output.shape()[0] != 2 || output.shape()[1] != 5) {
        std::cerr << "linear_shape: expected {2, 5}, got {"
                  << output.shape()[0] << ", " << output.shape()[1] << "}" << std::endl;
        return false;
    }

    // Check values are finite
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "linear_shape: NaN/Inf in output" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_11_linear_shape: PASSED" << std::endl;
    return true;
}

// Test 2: 3D input (batch dimensions)
bool test_phase_11_linear_3d() {
    Linear linear(10, 5);

    // Input: {4, 8, 10}
    Tensor<float> input_data = Tensor<float>::randn({4, 8, 10});
    Variable input(input_data, true);

    auto output = linear.forward(input);

    // Output should be {4, 8, 5}
    if (output.shape().size() != 3) {
        std::cerr << "linear_3d: expected 3D output" << std::endl;
        return false;
    }

    if (output.shape()[0] != 4 || output.shape()[1] != 8 || output.shape()[2] != 5) {
        std::cerr << "linear_3d: expected {4, 8, 5}, got {"
                  << output.shape()[0] << ", " << output.shape()[1]
                  << ", " << output.shape()[2] << "}" << std::endl;
        return false;
    }

    std::cout << "test_phase_11_linear_3d: PASSED" << std::endl;
    return true;
}

// Test 3: Linear without bias
bool test_phase_11_linear_no_bias() {
    Linear linear(10, 5, false);

    // Check has_bias returns false
    if (linear.has_bias()) {
        std::cerr << "linear_no_bias: has_bias should be false" << std::endl;
        return false;
    }

    // Get parameters - should only have weight
    auto params = linear.parameters();
    if (params.size() != 1) {
        std::cerr << "linear_no_bias: expected 1 parameter, got " << params.size() << std::endl;
        return false;
    }

    // Should still work in forward pass
    Tensor<float> input_data = Tensor<float>::randn({3, 10});
    Variable input(input_data, true);

    auto output = linear.forward(input);

    if (output.shape()[0] != 3 || output.shape()[1] != 5) {
        std::cerr << "linear_no_bias: wrong output shape" << std::endl;
        return false;
    }

    std::cout << "test_phase_11_linear_no_bias: PASSED" << std::endl;
    return true;
}

// Test 4: Backward pass
bool test_phase_11_linear_backward() {
    Linear linear(10, 5);

    Tensor<float> input_data = Tensor<float>::randn({4, 10});
    Variable input(input_data, true);

    auto output = linear.forward(input);

    // Backward
    output.backward();

    // Check weight gradients exist
    if (!linear.weight.has_grad()) {
        std::cerr << "linear_backward: weight should have grad" << std::endl;
        return false;
    }

    // Check weight grad shape
    if (linear.weight.grad().shape()[0] != 5 || linear.weight.grad().shape()[1] != 10) {
        std::cerr << "linear_backward: wrong weight grad shape" << std::endl;
        return false;
    }

    // Check bias gradients exist
    if (!linear.bias().has_grad()) {
        std::cerr << "linear_backward: bias should have grad" << std::endl;
        return false;
    }

    // Check bias grad shape
    if (linear.bias().grad().shape()[0] != 5) {
        std::cerr << "linear_backward: wrong bias grad shape" << std::endl;
        return false;
    }

    // Check input gradients exist
    if (!input.has_grad()) {
        std::cerr << "linear_backward: input should have grad" << std::endl;
        return false;
    }

    // Check gradients are finite
    for (size_t i = 0; i < linear.weight.grad().numel(); ++i) {
        if (std::isnan(linear.weight.grad().data()[i])) {
            std::cerr << "linear_backward: NaN in weight grad" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_11_linear_backward: PASSED" << std::endl;
    return true;
}

// Test 5: Parameters returned correctly
bool test_phase_11_linear_params() {
    Linear linear_with_bias(10, 5, true);
    Linear linear_no_bias(10, 5, false);

    // With bias: 2 parameters
    auto params_with = linear_with_bias.parameters();
    if (params_with.size() != 2) {
        std::cerr << "linear_params: expected 2 params with bias, got "
                  << params_with.size() << std::endl;
        return false;
    }

    // Without bias: 1 parameter
    auto params_without = linear_no_bias.parameters();
    if (params_without.size() != 1) {
        std::cerr << "linear_params: expected 1 param without bias, got "
                  << params_without.size() << std::endl;
        return false;
    }

    // Check parameter sizes
    // Weight: out_features * in_features = 5 * 10 = 50
    // Bias: out_features = 5
    size_t total_with = 0;
    for (auto* p : params_with) {
        total_with += p->numel();
    }
    if (total_with != 55) {  // 50 + 5
        std::cerr << "linear_params: expected 55 total params, got " << total_with << std::endl;
        return false;
    }

    size_t total_without = 0;
    for (auto* p : params_without) {
        total_without += p->numel();
    }
    if (total_without != 50) {
        std::cerr << "linear_params: expected 50 params without bias, got " << total_without << std::endl;
        return false;
    }

    std::cout << "test_phase_11_linear_params: PASSED" << std::endl;
    return true;
}

// Additional test: Xavier initialization bounds
bool test_phase_11_linear_init() {
    Linear linear(100, 50);

    // Xavier limit should be sqrt(6/(100+50)) = sqrt(0.04) ≈ 0.2
    float expected_limit = std::sqrt(6.0f / 150.0f);

    float max_val = 0.0f;
    for (size_t i = 0; i < linear.weight.numel(); ++i) {
        max_val = std::max(max_val, std::abs(linear.weight.data().data()[i]));
    }

    // Max value should be close to limit (with some tolerance for random sampling)
    if (max_val > expected_limit * 1.1f) {
        std::cerr << "linear_init: weight values exceed Xavier bounds" << std::endl;
        return false;
    }

    // Bias should be zeros
    for (size_t i = 0; i < linear.bias().numel(); ++i) {
        if (linear.bias().data().data()[i] != 0.0f) {
            std::cerr << "linear_init: bias should be zeros" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_11_linear_init: PASSED" << std::endl;
    return true;
}

// Test: Numerical correctness
bool test_phase_11_linear_values() {
    Linear linear(2, 2, true);

    // Set known weights and bias
    linear.weight.data().data()[0] = 1.0f;  // [0,0]
    linear.weight.data().data()[1] = 2.0f;  // [0,1]
    linear.weight.data().data()[2] = 3.0f;  // [1,0]
    linear.weight.data().data()[3] = 4.0f;  // [1,1]

    linear.bias().data().data()[0] = 0.5f;
    linear.bias().data().data()[1] = 1.5f;

    // Input: [1, 1]
    Tensor<float> input_data({1, 2});
    input_data.data()[0] = 1.0f;
    input_data.data()[1] = 1.0f;
    Variable input(input_data, false);

    auto output = linear.forward(input);

    // y = x @ W^T + b
    // W^T = [[1, 3], [2, 4]]
    // x @ W^T = [1*1 + 1*2, 1*3 + 1*4] = [3, 7]
    // y = [3 + 0.5, 7 + 1.5] = [3.5, 8.5]

    if (!float_eq(output.data().data()[0], 3.5f)) {
        std::cerr << "linear_values: output[0] should be 3.5, got "
                  << output.data().data()[0] << std::endl;
        return false;
    }

    if (!float_eq(output.data().data()[1], 8.5f)) {
        std::cerr << "linear_values: output[1] should be 8.5, got "
                  << output.data().data()[1] << std::endl;
        return false;
    }

    std::cout << "test_phase_11_linear_values: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 11: Linear Layer Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_11_linear_shape()) ++failures;
    if (!test_phase_11_linear_3d()) ++failures;
    if (!test_phase_11_linear_no_bias()) ++failures;
    if (!test_phase_11_linear_backward()) ++failures;
    if (!test_phase_11_linear_params()) ++failures;
    if (!test_phase_11_linear_init()) ++failures;
    if (!test_phase_11_linear_values()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 11 tests passed (7/7) ===" << std::endl;
    return 0;
}
