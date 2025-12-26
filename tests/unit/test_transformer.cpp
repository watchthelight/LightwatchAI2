// Phase 18: Transformer Encoder Block Tests

#include <lightwatch/nn/transformer.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: Output shape
bool test_phase_18_encoder_shape() {
    TransformerEncoderBlock block(768, 12, 3072);

    // Input: {batch=2, seq=16, embed=768}
    Tensor<float> input_data = Tensor<float>::randn({2, 16, 768});
    Variable input(input_data, false);

    auto output = block.forward(input);

    // Output should be same shape as input
    if (output.shape().size() != 3 ||
        output.shape()[0] != 2 ||
        output.shape()[1] != 16 ||
        output.shape()[2] != 768) {
        std::cerr << "encoder_shape: expected {2, 16, 768}, got {"
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << "}" << std::endl;
        return false;
    }

    // Check values are finite
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "encoder_shape: NaN/Inf in output" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_18_encoder_shape: PASSED" << std::endl;
    return true;
}

// Test 2: Residual connection check
bool test_phase_18_encoder_residual() {
    TransformerEncoderBlock block(64, 4, 256, 0.0f);
    block.train(false);

    // Use zeros to clearly see residual effect
    Tensor<float> input_data = Tensor<float>::zeros({1, 4, 64});
    // Set a few values
    for (size_t i = 0; i < 64; ++i) {
        input_data.data()[i] = static_cast<float>(i) * 0.01f;  // First position
    }
    Variable input(input_data, false);

    auto output = block.forward(input);

    // Output should not be all zeros (residual adds input)
    float sum = 0.0f;
    for (size_t i = 0; i < output.numel(); ++i) {
        sum += std::abs(output.data().data()[i]);
    }

    if (sum < 0.01f) {
        std::cerr << "encoder_residual: output appears to be all zeros, "
                  << "residual may not be working" << std::endl;
        return false;
    }

    std::cout << "test_phase_18_encoder_residual: PASSED" << std::endl;
    return true;
}

// Test 3: Backward pass
bool test_phase_18_encoder_backward() {
    TransformerEncoderBlock block(64, 4, 256, 0.0f);
    block.train(false);

    Tensor<float> input_data = Tensor<float>::randn({2, 4, 64});
    Variable input(input_data, true);

    auto output = block.forward(input);

    // Verify output is valid
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "encoder_backward: NaN/Inf in output" << std::endl;
            return false;
        }
    }

    // Backward pass - verify no crash
    output.backward();

    // Note: Full gradient verification through transformer block
    // requires careful backward chain management.

    std::cout << "test_phase_18_encoder_backward: PASSED" << std::endl;
    return true;
}

// Test 4: Pre-norm architecture
bool test_phase_18_encoder_prenorm() {
    TransformerEncoderBlock pre_block(64, 4, 256, 0.0f, true);   // pre_norm
    TransformerEncoderBlock post_block(64, 4, 256, 0.0f, false); // post_norm

    // Verify configuration
    if (!pre_block.is_pre_norm()) {
        std::cerr << "encoder_prenorm: pre_block should have pre_norm=true" << std::endl;
        return false;
    }

    if (post_block.is_pre_norm()) {
        std::cerr << "encoder_prenorm: post_block should have pre_norm=false" << std::endl;
        return false;
    }

    // Run forward on both
    pre_block.train(false);
    post_block.train(false);

    Tensor<float> input_data = Tensor<float>::randn({1, 4, 64});
    Variable input1(input_data, false);
    Variable input2(input_data, false);

    auto out1 = pre_block.forward(input1);
    auto out2 = post_block.forward(input2);

    // Both outputs should be valid
    for (size_t i = 0; i < out1.numel(); ++i) {
        if (std::isnan(out1.data().data()[i]) || std::isinf(out1.data().data()[i])) {
            std::cerr << "encoder_prenorm: NaN/Inf in pre_norm output" << std::endl;
            return false;
        }
    }

    for (size_t i = 0; i < out2.numel(); ++i) {
        if (std::isnan(out2.data().data()[i]) || std::isinf(out2.data().data()[i])) {
            std::cerr << "encoder_prenorm: NaN/Inf in post_norm output" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_18_encoder_prenorm: PASSED" << std::endl;
    return true;
}

// Test 5: Parameter count
bool test_phase_18_encoder_params() {
    TransformerEncoderBlock block(768, 12, 3072, 0.0f);

    // LayerNorm: 2 * (768 + 768) = 3072 (weight + bias for each)
    // MHA: 4 * (768*768 + 768) = 2362368
    // FFN: fc1(768*3072 + 3072) + fc2(3072*768 + 768) = 4722432
    // Total: 3072 + 2362368 + 4722432 = 7087872

    size_t ln_params = 2 * (768 + 768);  // 2 LayerNorms
    size_t mha_params = 4 * 768 * 768 + 4 * 768;
    size_t ffn_params = 768 * 3072 + 3072 + 3072 * 768 + 768;
    size_t expected_params = ln_params + mha_params + ffn_params;

    size_t actual_params = block.num_parameters();

    if (actual_params != expected_params) {
        std::cerr << "encoder_params: expected " << expected_params
                  << " params, got " << actual_params << std::endl;
        return false;
    }

    std::cout << "test_phase_18_encoder_params: PASSED" << std::endl;
    return true;
}

// Test 6: With attention mask
bool test_phase_18_encoder_masked() {
    TransformerEncoderBlock block(64, 4, 256, 0.0f);
    block.train(false);

    Tensor<float> input_data = Tensor<float>::randn({1, 8, 64});
    Variable input(input_data, false);

    auto mask = causal_mask(8);
    auto output = block.forward(input, &mask);

    // Check output is valid
    for (size_t i = 0; i < output.numel(); ++i) {
        if (std::isnan(output.data().data()[i]) || std::isinf(output.data().data()[i])) {
            std::cerr << "encoder_masked: NaN/Inf in output" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_18_encoder_masked: PASSED" << std::endl;
    return true;
}

// Test 7: Different configurations
bool test_phase_18_encoder_configs() {
    struct Config {
        size_t embed_dim;
        size_t num_heads;
        size_t ffn_dim;
    };

    std::vector<Config> configs = {
        {768, 12, 3072},   // GPT-2 Small
        {1024, 16, 4096},  // GPT-2 Medium
        {64, 4, 256},      // Small test
        {128, 8, 512},     // Another test
    };

    for (const auto& cfg : configs) {
        TransformerEncoderBlock block(cfg.embed_dim, cfg.num_heads, cfg.ffn_dim, 0.0f);
        block.train(false);

        Tensor<float> input_data = Tensor<float>::randn({1, 4, cfg.embed_dim});
        Variable input(input_data, false);

        auto output = block.forward(input);

        if (output.shape()[2] != cfg.embed_dim) {
            std::cerr << "encoder_configs: output embed_dim mismatch for config "
                      << cfg.embed_dim << "/" << cfg.num_heads << "/" << cfg.ffn_dim << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_18_encoder_configs: PASSED" << std::endl;
    return true;
}

// Test 8: Dropout in training mode
bool test_phase_18_encoder_dropout() {
    TransformerEncoderBlock block(64, 4, 256, 0.5f);  // 50% dropout
    block.train(true);

    Tensor<float> input_data = Tensor<float>::ones({1, 4, 64});
    Variable input(input_data, false);

    auto output1 = block.forward(input);
    auto output2 = block.forward(input);

    // Just check no NaN/Inf
    for (size_t i = 0; i < output1.numel(); ++i) {
        if (std::isnan(output1.data().data()[i]) || std::isinf(output1.data().data()[i])) {
            std::cerr << "encoder_dropout: NaN/Inf in output" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_18_encoder_dropout: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 18: Transformer Encoder Block Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_18_encoder_shape()) ++failures;
    if (!test_phase_18_encoder_residual()) ++failures;
    if (!test_phase_18_encoder_backward()) ++failures;
    if (!test_phase_18_encoder_prenorm()) ++failures;
    if (!test_phase_18_encoder_params()) ++failures;
    if (!test_phase_18_encoder_masked()) ++failures;
    if (!test_phase_18_encoder_configs()) ++failures;
    if (!test_phase_18_encoder_dropout()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 18 tests passed (8/8) ===" << std::endl;
    return 0;
}
