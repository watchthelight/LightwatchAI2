// Phase 34: Greedy Decode Tests

#include <lightwatch/generate.hpp>
#include <lightwatch/init.hpp>
#include <iostream>
#include <vector>

using namespace lightwatch;
using namespace lightwatch::models;

// Create a small test model
GPT2 create_test_model() {
    GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.embed_dim = 32;
    cfg.num_heads = 2;
    cfg.num_layers = 1;
    cfg.ffn_dim = 128;
    cfg.dropout_p = 0.0f;  // No dropout for deterministic tests

    GPT2 model(cfg);

    // Initialize with fixed seed for reproducibility
    init::InitConfig init_cfg;
    init_cfg.seed = 42;
    init::init_gpt2_weights(model, init_cfg);

    return model;
}

// Test 1: Greedy shape - output extends input
bool test_phase_34_greedy_shape() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3, 4, 5};  // 5 tokens

    GenerateConfig config;
    config.max_new_tokens = 10;
    config.early_stop = false;  // Don't stop at EOS for this test

    auto output = generate_greedy(model, prompt, config);

    // Should have original 5 + 10 new = 15 tokens
    if (output.size() != 15) {
        std::cerr << "greedy_shape: expected 15 tokens, got " << output.size() << std::endl;
        return false;
    }

    // First 5 tokens should match prompt
    for (size_t i = 0; i < 5; ++i) {
        if (output[i] != prompt[i]) {
            std::cerr << "greedy_shape: prompt not preserved at position " << i << std::endl;
            return false;
        }
    }

    // All tokens should be valid (< vocab_size)
    for (auto token : output) {
        if (token < 0 || token >= 100) {
            std::cerr << "greedy_shape: invalid token " << token << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_34_greedy_shape: PASSED" << std::endl;
    return true;
}

// Test 2: Deterministic - same input gives same output
bool test_phase_34_greedy_deterministic() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {10, 20, 30};

    GenerateConfig config;
    config.max_new_tokens = 5;
    config.early_stop = false;

    auto output1 = generate_greedy(model, prompt, config);
    auto output2 = generate_greedy(model, prompt, config);

    if (output1.size() != output2.size()) {
        std::cerr << "greedy_deterministic: different lengths" << std::endl;
        return false;
    }

    for (size_t i = 0; i < output1.size(); ++i) {
        if (output1[i] != output2[i]) {
            std::cerr << "greedy_deterministic: mismatch at position " << i << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_34_greedy_deterministic: PASSED" << std::endl;
    return true;
}

// Test 3: EOS stopping
bool test_phase_34_greedy_eos() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3};

    GenerateConfig config;
    config.max_new_tokens = 50;  // Large limit
    config.eos_token_id = 0;     // Use token 0 as EOS
    config.early_stop = true;

    auto output = generate_greedy(model, prompt, config);

    // Should have stopped before max (or at max if no EOS generated)
    // The important thing is it doesn't crash
    if (output.size() < 3) {
        std::cerr << "greedy_eos: output too short" << std::endl;
        return false;
    }

    // Check if it stopped at EOS
    bool has_eos = false;
    for (size_t i = 3; i < output.size(); ++i) {
        if (output[i] == config.eos_token_id) {
            has_eos = true;
            // EOS should be the last token (if early_stop is true)
            if (i != output.size() - 1) {
                std::cerr << "greedy_eos: EOS not at end" << std::endl;
                return false;
            }
            break;
        }
    }

    // It's okay if no EOS was generated - model might not produce it
    // The test passes either way as long as generation works correctly

    std::cout << "test_phase_34_greedy_eos: PASSED (output_len=" << output.size()
              << ", has_eos=" << has_eos << ")" << std::endl;
    return true;
}

// Test 4: Max tokens limit
bool test_phase_34_greedy_max_tokens() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {5, 10, 15};

    GenerateConfig config;
    config.max_new_tokens = 10;
    config.early_stop = false;  // Don't stop at EOS

    auto output = generate_greedy(model, prompt, config);

    // Should have exactly prompt + max_new_tokens
    size_t expected_len = prompt.size() + config.max_new_tokens;
    if (output.size() != expected_len) {
        std::cerr << "greedy_max_tokens: expected " << expected_len
                  << " tokens, got " << output.size() << std::endl;
        return false;
    }

    // Test with different values
    config.max_new_tokens = 5;
    output = generate_greedy(model, prompt, config);
    expected_len = prompt.size() + 5;
    if (output.size() != expected_len) {
        std::cerr << "greedy_max_tokens: expected " << expected_len
                  << " (with max=5), got " << output.size() << std::endl;
        return false;
    }

    std::cout << "test_phase_34_greedy_max_tokens: PASSED" << std::endl;
    return true;
}

// Test 5: Streaming callback
bool test_phase_34_streaming() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2};

    GenerateConfig config;
    config.max_new_tokens = 5;
    config.early_stop = false;

    std::vector<TokenId> streamed_tokens;
    auto callback = [&streamed_tokens](TokenId token) {
        streamed_tokens.push_back(token);
    };

    generate_greedy_streaming(model, prompt, callback, config);

    // Should have received exactly max_new_tokens callbacks
    if (streamed_tokens.size() != config.max_new_tokens) {
        std::cerr << "streaming: expected " << config.max_new_tokens
                  << " callbacks, got " << streamed_tokens.size() << std::endl;
        return false;
    }

    // Compare with non-streaming version
    auto full_output = generate_greedy(model, prompt, config);
    auto generated = get_generated_tokens(full_output, prompt.size());

    if (generated.size() != streamed_tokens.size()) {
        std::cerr << "streaming: length mismatch with non-streaming" << std::endl;
        return false;
    }

    for (size_t i = 0; i < generated.size(); ++i) {
        if (generated[i] != streamed_tokens[i]) {
            std::cerr << "streaming: token mismatch at position " << i << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_34_streaming: PASSED" << std::endl;
    return true;
}

// Test 6: Empty prompt handling
bool test_phase_34_empty_prompt() {
    auto model = create_test_model();

    std::vector<TokenId> empty_prompt;

    GenerateConfig config;
    config.max_new_tokens = 5;

    auto output = generate_greedy(model, empty_prompt, config);

    // Empty prompt should return empty output
    if (!output.empty()) {
        std::cerr << "empty_prompt: expected empty output" << std::endl;
        return false;
    }

    std::cout << "test_phase_34_empty_prompt: PASSED" << std::endl;
    return true;
}

// Test 7: Argmax helper function
bool test_phase_34_argmax() {
    float data1[] = {1.0f, 3.0f, 2.0f, 0.5f};
    if (argmax(data1, 4) != 1) {
        std::cerr << "argmax: wrong result for data1" << std::endl;
        return false;
    }

    float data2[] = {5.0f, 1.0f, 2.0f};
    if (argmax(data2, 3) != 0) {
        std::cerr << "argmax: wrong result for data2 (max at start)" << std::endl;
        return false;
    }

    float data3[] = {1.0f, 2.0f, 10.0f};
    if (argmax(data3, 3) != 2) {
        std::cerr << "argmax: wrong result for data3 (max at end)" << std::endl;
        return false;
    }

    float data4[] = {-1.0f, -0.5f, -2.0f};
    if (argmax(data4, 3) != 1) {
        std::cerr << "argmax: wrong result for negative values" << std::endl;
        return false;
    }

    std::cout << "test_phase_34_argmax: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 34: Greedy Decode Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_34_greedy_shape()) ++failures;
    if (!test_phase_34_greedy_deterministic()) ++failures;
    if (!test_phase_34_greedy_eos()) ++failures;
    if (!test_phase_34_greedy_max_tokens()) ++failures;
    if (!test_phase_34_streaming()) ++failures;
    if (!test_phase_34_empty_prompt()) ++failures;
    if (!test_phase_34_argmax()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 34 tests passed (7/7) ===" << std::endl;
    return 0;
}
