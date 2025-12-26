// Phase 12: Activation Functions Tests

#include <lightwatch/nn/activations.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: ReLU forward
bool test_phase_12_relu() {
    ReLU relu;

    Tensor<float> input_data({3});
    input_data.data()[0] = -1.0f;
    input_data.data()[1] = 0.0f;
    input_data.data()[2] = 1.0f;
    Variable input(input_data, false);

    auto output = relu.forward(input);

    // Expected: [0, 0, 1]
    if (!float_eq(output.data().data()[0], 0.0f)) {
        std::cerr << "relu: expected 0 for -1, got " << output.data().data()[0] << std::endl;
        return false;
    }
    if (!float_eq(output.data().data()[1], 0.0f)) {
        std::cerr << "relu: expected 0 for 0, got " << output.data().data()[1] << std::endl;
        return false;
    }
    if (!float_eq(output.data().data()[2], 1.0f)) {
        std::cerr << "relu: expected 1 for 1, got " << output.data().data()[2] << std::endl;
        return false;
    }

    std::cout << "test_phase_12_relu: PASSED" << std::endl;
    return true;
}

// Test 2: ReLU backward
bool test_phase_12_relu_backward() {
    ReLU relu;

    Tensor<float> input_data({4});
    input_data.data()[0] = -2.0f;
    input_data.data()[1] = -0.5f;
    input_data.data()[2] = 0.5f;
    input_data.data()[3] = 2.0f;
    Variable input(input_data, true);

    auto output = relu.forward(input);
    output.backward();

    // Gradient should be 1 where input > 0, else 0
    // Expected: [0, 0, 1, 1]
    if (!float_eq(input.grad().data()[0], 0.0f)) {
        std::cerr << "relu_backward: grad[0] should be 0" << std::endl;
        return false;
    }
    if (!float_eq(input.grad().data()[1], 0.0f)) {
        std::cerr << "relu_backward: grad[1] should be 0" << std::endl;
        return false;
    }
    if (!float_eq(input.grad().data()[2], 1.0f)) {
        std::cerr << "relu_backward: grad[2] should be 1" << std::endl;
        return false;
    }
    if (!float_eq(input.grad().data()[3], 1.0f)) {
        std::cerr << "relu_backward: grad[3] should be 1" << std::endl;
        return false;
    }

    std::cout << "test_phase_12_relu_backward: PASSED" << std::endl;
    return true;
}

// Test 3: GELU
bool test_phase_12_gelu() {
    GELU gelu;

    Tensor<float> input_data({3});
    input_data.data()[0] = 0.0f;
    input_data.data()[1] = 1.0f;
    input_data.data()[2] = 2.0f;
    Variable input(input_data, true);

    auto output = gelu.forward(input);

    // GELU(0) = 0
    if (!float_eq(output.data().data()[0], 0.0f, 1e-3f)) {
        std::cerr << "gelu: GELU(0) should be ~0, got " << output.data().data()[0] << std::endl;
        return false;
    }

    // GELU(1) ≈ 0.841
    if (!float_eq(output.data().data()[1], 0.841f, 0.01f)) {
        std::cerr << "gelu: GELU(1) should be ~0.841, got " << output.data().data()[1] << std::endl;
        return false;
    }

    // GELU(2) ≈ 1.954
    if (!float_eq(output.data().data()[2], 1.954f, 0.01f)) {
        std::cerr << "gelu: GELU(2) should be ~1.954, got " << output.data().data()[2] << std::endl;
        return false;
    }

    // Test backward
    output.backward();
    if (!input.has_grad()) {
        std::cerr << "gelu: should have gradients" << std::endl;
        return false;
    }

    std::cout << "test_phase_12_gelu: PASSED" << std::endl;
    return true;
}

// Test 4: Softmax sums to 1
bool test_phase_12_softmax() {
    Softmax softmax(-1);  // Last dimension

    Tensor<float> input_data({3});
    input_data.data()[0] = 1.0f;
    input_data.data()[1] = 2.0f;
    input_data.data()[2] = 3.0f;
    Variable input(input_data, true);

    auto output = softmax.forward(input);

    // Sum should be 1
    float sum = 0.0f;
    for (size_t i = 0; i < output.numel(); ++i) {
        sum += output.data().data()[i];
    }

    if (!float_eq(sum, 1.0f, 1e-3f)) {
        std::cerr << "softmax: sum should be 1, got " << sum << std::endl;
        return false;
    }

    // Values should be positive and in order
    if (output.data().data()[0] >= output.data().data()[1] ||
        output.data().data()[1] >= output.data().data()[2]) {
        std::cerr << "softmax: values should be in ascending order" << std::endl;
        return false;
    }

    std::cout << "test_phase_12_softmax: PASSED" << std::endl;
    return true;
}

// Test 5: Softmax numerical stability
bool test_phase_12_softmax_stable() {
    Softmax softmax(-1);

    // Large values that would overflow without numerical stability
    Tensor<float> input_data({3});
    input_data.data()[0] = 1000.0f;
    input_data.data()[1] = 1001.0f;
    input_data.data()[2] = 1002.0f;
    Variable input(input_data, false);

    auto output = softmax.forward(input);

    // Check no NaN or Inf
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "softmax_stable: NaN/Inf detected" << std::endl;
            return false;
        }
    }

    // Sum should still be 1
    float sum = 0.0f;
    for (size_t i = 0; i < output.numel(); ++i) {
        sum += output.data().data()[i];
    }

    if (!float_eq(sum, 1.0f, 1e-3f)) {
        std::cerr << "softmax_stable: sum should be 1, got " << sum << std::endl;
        return false;
    }

    std::cout << "test_phase_12_softmax_stable: PASSED" << std::endl;
    return true;
}

// Test 6: Sigmoid
bool test_phase_12_sigmoid() {
    Sigmoid sigmoid;

    Tensor<float> input_data({3});
    input_data.data()[0] = 0.0f;
    input_data.data()[1] = -10.0f;
    input_data.data()[2] = 10.0f;
    Variable input(input_data, true);

    auto output = sigmoid.forward(input);

    // sigmoid(0) = 0.5
    if (!float_eq(output.data().data()[0], 0.5f)) {
        std::cerr << "sigmoid: sigmoid(0) should be 0.5, got " << output.data().data()[0] << std::endl;
        return false;
    }

    // sigmoid(-10) ≈ 0
    if (output.data().data()[1] > 0.01f) {
        std::cerr << "sigmoid: sigmoid(-10) should be ~0" << std::endl;
        return false;
    }

    // sigmoid(10) ≈ 1
    if (output.data().data()[2] < 0.99f) {
        std::cerr << "sigmoid: sigmoid(10) should be ~1" << std::endl;
        return false;
    }

    std::cout << "test_phase_12_sigmoid: PASSED" << std::endl;
    return true;
}

// Test 7: Tanh
bool test_phase_12_tanh() {
    Tanh tanh_mod;

    Tensor<float> input_data({3});
    input_data.data()[0] = 0.0f;
    input_data.data()[1] = -10.0f;
    input_data.data()[2] = 10.0f;
    Variable input(input_data, true);

    auto output = tanh_mod.forward(input);

    // tanh(0) = 0
    if (!float_eq(output.data().data()[0], 0.0f)) {
        std::cerr << "tanh: tanh(0) should be 0" << std::endl;
        return false;
    }

    // tanh(-10) ≈ -1
    if (!float_eq(output.data().data()[1], -1.0f, 0.01f)) {
        std::cerr << "tanh: tanh(-10) should be ~-1" << std::endl;
        return false;
    }

    // tanh(10) ≈ 1
    if (!float_eq(output.data().data()[2], 1.0f, 0.01f)) {
        std::cerr << "tanh: tanh(10) should be ~1" << std::endl;
        return false;
    }

    std::cout << "test_phase_12_tanh: PASSED" << std::endl;
    return true;
}

// Test 8: SiLU (Swish)
bool test_phase_12_silu() {
    SiLU silu;

    Tensor<float> input_data({3});
    input_data.data()[0] = 0.0f;
    input_data.data()[1] = 1.0f;
    input_data.data()[2] = -1.0f;
    Variable input(input_data, true);

    auto output = silu.forward(input);

    // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
    if (!float_eq(output.data().data()[0], 0.0f)) {
        std::cerr << "silu: SiLU(0) should be 0" << std::endl;
        return false;
    }

    // SiLU(1) = 1 * sigmoid(1) ≈ 0.731
    float expected_1 = 1.0f / (1.0f + std::exp(-1.0f));
    if (!float_eq(output.data().data()[1], expected_1, 0.01f)) {
        std::cerr << "silu: SiLU(1) should be ~" << expected_1 << ", got " << output.data().data()[1] << std::endl;
        return false;
    }

    // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.269
    float expected_neg1 = -1.0f / (1.0f + std::exp(1.0f));
    if (!float_eq(output.data().data()[2], expected_neg1, 0.01f)) {
        std::cerr << "silu: SiLU(-1) should be ~" << expected_neg1 << std::endl;
        return false;
    }

    std::cout << "test_phase_12_silu: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 12: Activation Functions Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_12_relu()) ++failures;
    if (!test_phase_12_relu_backward()) ++failures;
    if (!test_phase_12_gelu()) ++failures;
    if (!test_phase_12_softmax()) ++failures;
    if (!test_phase_12_softmax_stable()) ++failures;
    if (!test_phase_12_sigmoid()) ++failures;
    if (!test_phase_12_tanh()) ++failures;
    if (!test_phase_12_silu()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 12 tests passed (8/8) ===" << std::endl;
    return 0;
}
