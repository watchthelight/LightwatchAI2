// Phase 13: Layer Normalization Tests

#include <lightwatch/nn/normalization.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: LayerNorm output mean should be approximately beta (0 by default)
bool test_phase_13_layernorm_mean() {
    LayerNorm ln(4);

    // Input with different means per row
    Tensor<float> input_data({2, 4});
    input_data.data()[0] = 1.0f;
    input_data.data()[1] = 2.0f;
    input_data.data()[2] = 3.0f;
    input_data.data()[3] = 4.0f;
    input_data.data()[4] = 10.0f;
    input_data.data()[5] = 20.0f;
    input_data.data()[6] = 30.0f;
    input_data.data()[7] = 40.0f;
    Variable input(input_data, false);

    auto output = ln.forward(input);

    // Check each row has mean ≈ 0 (beta)
    for (size_t row = 0; row < 2; ++row) {
        float mean = 0.0f;
        for (size_t col = 0; col < 4; ++col) {
            mean += output.data().data()[row * 4 + col];
        }
        mean /= 4.0f;

        if (!float_eq(mean, 0.0f, 0.01f)) {
            std::cerr << "layernorm_mean: row " << row << " mean should be ~0, got " << mean << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_13_layernorm_mean: PASSED" << std::endl;
    return true;
}

// Test 2: LayerNorm output std should be approximately gamma (1 by default)
bool test_phase_13_layernorm_std() {
    LayerNorm ln(4);

    Tensor<float> input_data({2, 4});
    input_data.data()[0] = 1.0f;
    input_data.data()[1] = 2.0f;
    input_data.data()[2] = 3.0f;
    input_data.data()[3] = 4.0f;
    input_data.data()[4] = 10.0f;
    input_data.data()[5] = 20.0f;
    input_data.data()[6] = 30.0f;
    input_data.data()[7] = 40.0f;
    Variable input(input_data, false);

    auto output = ln.forward(input);

    // Check each row has std ≈ 1 (gamma)
    for (size_t row = 0; row < 2; ++row) {
        // Compute mean
        float mean = 0.0f;
        for (size_t col = 0; col < 4; ++col) {
            mean += output.data().data()[row * 4 + col];
        }
        mean /= 4.0f;

        // Compute variance
        float var = 0.0f;
        for (size_t col = 0; col < 4; ++col) {
            float diff = output.data().data()[row * 4 + col] - mean;
            var += diff * diff;
        }
        var /= 4.0f;
        float std_val = std::sqrt(var);

        // Note: LayerNorm uses Bessel's correction (n-1) or not depending on implementation
        // Allow some tolerance
        if (std_val < 0.5f || std_val > 1.5f) {
            std::cerr << "layernorm_std: row " << row << " std should be ~1, got " << std_val << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_13_layernorm_std: PASSED" << std::endl;
    return true;
}

// Test 3: LayerNorm backward pass
bool test_phase_13_layernorm_backward() {
    LayerNorm ln(4);

    Tensor<float> input_data({2, 4});
    for (size_t i = 0; i < 8; ++i) {
        input_data.data()[i] = static_cast<float>(i + 1);
    }
    Variable input(input_data, true);

    auto output = ln.forward(input);
    output.backward();

    // Check that gradients exist and are finite
    if (!input.has_grad()) {
        std::cerr << "layernorm_backward: input should have grad" << std::endl;
        return false;
    }

    if (!ln.weight.has_grad()) {
        std::cerr << "layernorm_backward: weight should have grad" << std::endl;
        return false;
    }

    if (!ln.bias.has_grad()) {
        std::cerr << "layernorm_backward: bias should have grad" << std::endl;
        return false;
    }

    // Check for NaN/Inf
    for (size_t i = 0; i < input.grad().numel(); ++i) {
        if (std::isnan(input.grad().data()[i]) || std::isinf(input.grad().data()[i])) {
            std::cerr << "layernorm_backward: NaN/Inf in input grad" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_13_layernorm_backward: PASSED" << std::endl;
    return true;
}

// Test 4: RMSNorm
bool test_phase_13_rmsnorm() {
    RMSNorm rms(4);

    Tensor<float> input_data({2, 4});
    input_data.data()[0] = 1.0f;
    input_data.data()[1] = 2.0f;
    input_data.data()[2] = 3.0f;
    input_data.data()[3] = 4.0f;
    input_data.data()[4] = 2.0f;
    input_data.data()[5] = 4.0f;
    input_data.data()[6] = 6.0f;
    input_data.data()[7] = 8.0f;
    Variable input(input_data, true);

    auto output = rms.forward(input);

    // Check output shape
    if (output.shape()[0] != 2 || output.shape()[1] != 4) {
        std::cerr << "rmsnorm: wrong output shape" << std::endl;
        return false;
    }

    // For each row, compute expected RMS and check
    for (size_t row = 0; row < 2; ++row) {
        float sum_sq = 0.0f;
        for (size_t col = 0; col < 4; ++col) {
            float val = input_data.data()[row * 4 + col];
            sum_sq += val * val;
        }
        float expected_rms = std::sqrt(sum_sq / 4.0f + rms.eps());

        // Output should be input / rms (since weight is 1)
        for (size_t col = 0; col < 4; ++col) {
            float expected = input_data.data()[row * 4 + col] / expected_rms;
            float actual = output.data().data()[row * 4 + col];
            if (!float_eq(actual, expected, 0.01f)) {
                std::cerr << "rmsnorm: row " << row << " col " << col
                          << " expected " << expected << " got " << actual << std::endl;
                return false;
            }
        }
    }

    // Test backward
    output.backward();
    if (!input.has_grad()) {
        std::cerr << "rmsnorm: input should have grad" << std::endl;
        return false;
    }

    std::cout << "test_phase_13_rmsnorm: PASSED" << std::endl;
    return true;
}

// Test 5: Numerical stability with very small variance
bool test_phase_13_eps_stability() {
    LayerNorm ln(4);

    // All values the same - variance is 0!
    Tensor<float> input_data({1, 4});
    input_data.data()[0] = 5.0f;
    input_data.data()[1] = 5.0f;
    input_data.data()[2] = 5.0f;
    input_data.data()[3] = 5.0f;
    Variable input(input_data, true);

    auto output = ln.forward(input);

    // Check no NaN/Inf
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "eps_stability: NaN/Inf with zero variance" << std::endl;
            return false;
        }
    }

    // Output should be all zeros (since x - mean = 0, normalized = 0, then * gamma + beta = beta = 0)
    for (size_t i = 0; i < output.numel(); ++i) {
        if (!float_eq(output.data().data()[i], 0.0f, 0.01f)) {
            std::cerr << "eps_stability: output should be ~0 when all inputs equal" << std::endl;
            return false;
        }
    }

    // Test backward
    output.backward();
    for (size_t i = 0; i < input.grad().numel(); ++i) {
        if (std::isnan(input.grad().data()[i]) || std::isinf(input.grad().data()[i])) {
            std::cerr << "eps_stability: NaN/Inf in grad with zero variance" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_13_eps_stability: PASSED" << std::endl;
    return true;
}

// Test 6: LayerNorm parameters
bool test_phase_13_layernorm_params() {
    LayerNorm ln(64);

    auto params = ln.parameters();

    // Should have 2 parameters: weight and bias
    if (params.size() != 2) {
        std::cerr << "layernorm_params: expected 2 params, got " << params.size() << std::endl;
        return false;
    }

    // Total params should be 64 + 64 = 128
    size_t total = 0;
    for (auto* p : params) {
        total += p->numel();
    }
    if (total != 128) {
        std::cerr << "layernorm_params: expected 128 total params, got " << total << std::endl;
        return false;
    }

    std::cout << "test_phase_13_layernorm_params: PASSED" << std::endl;
    return true;
}

// Test 7: RMSNorm parameters
bool test_phase_13_rmsnorm_params() {
    RMSNorm rms(64);

    auto params = rms.parameters();

    // Should have 1 parameter: weight (no bias in RMSNorm)
    if (params.size() != 1) {
        std::cerr << "rmsnorm_params: expected 1 param, got " << params.size() << std::endl;
        return false;
    }

    // Total params should be 64
    if (params[0]->numel() != 64) {
        std::cerr << "rmsnorm_params: expected 64 params, got " << params[0]->numel() << std::endl;
        return false;
    }

    std::cout << "test_phase_13_rmsnorm_params: PASSED" << std::endl;
    return true;
}

// Test 8: 3D input
bool test_phase_13_layernorm_3d() {
    LayerNorm ln(4);

    // 3D input: {batch, seq, hidden}
    Tensor<float> input_data({2, 3, 4});
    for (size_t i = 0; i < 24; ++i) {
        input_data.data()[i] = static_cast<float>(i + 1);
    }
    Variable input(input_data, true);

    auto output = ln.forward(input);

    // Check shape preserved
    if (output.shape().size() != 3 ||
        output.shape()[0] != 2 ||
        output.shape()[1] != 3 ||
        output.shape()[2] != 4) {
        std::cerr << "layernorm_3d: wrong output shape" << std::endl;
        return false;
    }

    // Check each position is normalized
    for (size_t b = 0; b < 2; ++b) {
        for (size_t s = 0; s < 3; ++s) {
            float mean = 0.0f;
            for (size_t h = 0; h < 4; ++h) {
                mean += output.data().data()[b * 12 + s * 4 + h];
            }
            mean /= 4.0f;

            if (!float_eq(mean, 0.0f, 0.1f)) {
                std::cerr << "layernorm_3d: mean not ~0 at batch=" << b << ", seq=" << s << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_13_layernorm_3d: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 13: Layer Normalization Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_13_layernorm_mean()) ++failures;
    if (!test_phase_13_layernorm_std()) ++failures;
    if (!test_phase_13_layernorm_backward()) ++failures;
    if (!test_phase_13_rmsnorm()) ++failures;
    if (!test_phase_13_eps_stability()) ++failures;
    if (!test_phase_13_layernorm_params()) ++failures;
    if (!test_phase_13_rmsnorm_params()) ++failures;
    if (!test_phase_13_layernorm_3d()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 13 tests passed (8/8) ===" << std::endl;
    return 0;
}
