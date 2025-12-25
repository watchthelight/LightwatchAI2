// Phase 09: Positional Encoding Tests

#include <lightwatch/nn/positional.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test sinusoidal encoding shape
bool test_phase_09_sinusoidal_shape() {
    SinusoidalPE pe(100, 64);

    auto encoding = pe.get_encoding(10);

    if (encoding.shape().size() != 2) {
        std::cerr << "sinusoidal_shape: expected 2D" << std::endl;
        return false;
    }

    if (encoding.shape()[0] != 10 || encoding.shape()[1] != 64) {
        std::cerr << "sinusoidal_shape: expected {10, 64}" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_sinusoidal_shape: PASSED" << std::endl;
    return true;
}

// Test sinusoidal encoding values at position 0
bool test_phase_09_sinusoidal_values() {
    SinusoidalPE pe(100, 64);

    auto encoding = pe.get_encoding(1);

    // At position 0, sin(0) = 0 for all even indices
    // cos(0) = 1 for all odd indices
    for (size_t i = 0; i < 64; i += 2) {
        float sin_val = encoding({0, i});
        if (!float_eq(sin_val, 0.0f)) {
            std::cerr << "sinusoidal_values: sin at pos 0 should be 0, got "
                      << sin_val << " at " << i << std::endl;
            return false;
        }

        if (i + 1 < 64) {
            float cos_val = encoding({0, i + 1});
            if (!float_eq(cos_val, 1.0f)) {
                std::cerr << "sinusoidal_values: cos at pos 0 should be 1, got "
                          << cos_val << " at " << i + 1 << std::endl;
                return false;
            }
        }
    }

    std::cout << "test_phase_09_sinusoidal_values: PASSED" << std::endl;
    return true;
}

// Test sinusoidal forward pass
bool test_phase_09_sinusoidal_forward() {
    SinusoidalPE pe(100, 32);

    // Create input
    Variable input(Tensor<float>::ones({2, 10, 32}), true);
    auto output = pe.forward(input);

    // Check shape preserved
    auto shape = output.data().shape();
    if (shape.size() != 3 || shape[0] != 2 || shape[1] != 10 || shape[2] != 32) {
        std::cerr << "sinusoidal_forward: shape mismatch" << std::endl;
        return false;
    }

    // Check values are not all ones (encoding was added)
    bool found_non_one = false;
    for (size_t i = 0; i < output.numel(); ++i) {
        if (!float_eq(output.data().data()[i], 1.0f)) {
            found_non_one = true;
            break;
        }
    }

    if (!found_non_one) {
        std::cerr << "sinusoidal_forward: encoding not added" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_sinusoidal_forward: PASSED" << std::endl;
    return true;
}

// Test RoPE shape preservation
bool test_phase_09_rope_shape() {
    RoPE rope(64, 1024);  // head_dim=64, max_seq=1024

    // Create q and k tensors: {batch, heads, seq, head_dim}
    Tensor<float> q = Tensor<float>::randn({2, 8, 16, 64});
    Tensor<float> k = Tensor<float>::randn({2, 8, 16, 64});

    auto [q_rot, k_rot] = rope.apply(q, k);

    // Check shapes preserved
    if (q_rot.shape() != q.shape()) {
        std::cerr << "rope_shape: q shape changed" << std::endl;
        return false;
    }

    if (k_rot.shape() != k.shape()) {
        std::cerr << "rope_shape: k shape changed" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_rope_shape: PASSED" << std::endl;
    return true;
}

// Test RoPE rotation properties
bool test_phase_09_rope_rotation() {
    RoPE rope(4, 100);  // Small head_dim for testing

    // Create simple tensors
    Tensor<float> q({1, 1, 1, 4});  // Single position
    q.data()[0] = 1.0f;
    q.data()[1] = 0.0f;
    q.data()[2] = 1.0f;
    q.data()[3] = 0.0f;

    Tensor<float> k = q.clone();

    // Apply at position 0
    auto [q0, k0] = rope.apply(q, k, 0);

    // At position 0, angle=0, so cos=1, sin=0, no rotation
    if (!float_eq(q0.data()[0], 1.0f) || !float_eq(q0.data()[1], 0.0f)) {
        std::cerr << "rope_rotation: position 0 should not rotate" << std::endl;
        return false;
    }

    // Apply at position 1 - should rotate
    auto [q1, k1] = rope.apply(q, k, 1);

    // Values should have changed
    if (float_eq(q1.data()[0], q.data()[0]) && float_eq(q1.data()[1], q.data()[1])) {
        std::cerr << "rope_rotation: position 1 should rotate" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_rope_rotation: PASSED" << std::endl;
    return true;
}

// Test ALiBi shape
bool test_phase_09_alibi_shape() {
    ALiBi alibi(8);  // 8 heads

    auto bias = alibi.get_bias(16);

    // Shape should be {8, 16, 16}
    if (bias.shape().size() != 3) {
        std::cerr << "alibi_shape: expected 3D" << std::endl;
        return false;
    }

    if (bias.shape()[0] != 8 || bias.shape()[1] != 16 || bias.shape()[2] != 16) {
        std::cerr << "alibi_shape: expected {8, 16, 16}" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_alibi_shape: PASSED" << std::endl;
    return true;
}

// Test ALiBi slopes (geometric sequence)
bool test_phase_09_alibi_slopes() {
    ALiBi alibi(8);

    const auto& slopes = alibi.slopes();

    // Slopes should be 2^(-8/8), 2^(-16/8), 2^(-24/8), ...
    // = 2^(-1), 2^(-2), 2^(-3), ...
    // = 0.5, 0.25, 0.125, ...
    float expected[] = {0.5f, 0.25f, 0.125f, 0.0625f,
                        0.03125f, 0.015625f, 0.0078125f, 0.00390625f};

    for (size_t i = 0; i < 8; ++i) {
        if (!float_eq(slopes.data()[i], expected[i], 1e-4f)) {
            std::cerr << "alibi_slopes: mismatch at " << i
                      << " expected " << expected[i]
                      << " got " << slopes.data()[i] << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_09_alibi_slopes: PASSED" << std::endl;
    return true;
}

// Test ALiBi bias values
bool test_phase_09_alibi_bias_values() {
    ALiBi alibi(4);

    auto bias = alibi.get_bias(4);

    // At diagonal (i == j), bias should be 0
    for (size_t h = 0; h < 4; ++h) {
        for (size_t i = 0; i < 4; ++i) {
            if (!float_eq(bias({h, i, i}), 0.0f)) {
                std::cerr << "alibi_bias_values: diagonal should be 0" << std::endl;
                return false;
            }
        }
    }

    // Below diagonal (i > j), bias should be negative (penalty for distance)
    // bias[h][2][0] = -slope[h] * 2
    float slope0 = alibi.slopes().data()[0];
    if (!float_eq(bias({0, 2, 0}), -slope0 * 2.0f, 1e-4f)) {
        std::cerr << "alibi_bias_values: distance penalty wrong" << std::endl;
        return false;
    }

    std::cout << "test_phase_09_alibi_bias_values: PASSED" << std::endl;
    return true;
}

int main() {
    int failures = 0;

    if (!test_phase_09_sinusoidal_shape()) ++failures;
    if (!test_phase_09_sinusoidal_values()) ++failures;
    if (!test_phase_09_sinusoidal_forward()) ++failures;
    if (!test_phase_09_rope_shape()) ++failures;
    if (!test_phase_09_rope_rotation()) ++failures;
    if (!test_phase_09_alibi_shape()) ++failures;
    if (!test_phase_09_alibi_slopes()) ++failures;
    if (!test_phase_09_alibi_bias_values()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "All Phase 09 tests passed (8/8)" << std::endl;
    return 0;
}
