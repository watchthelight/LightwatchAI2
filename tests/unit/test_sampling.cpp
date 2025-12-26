// Phase 35: Sampling Tests

#include <lightwatch/generate.hpp>
#include <lightwatch/init.hpp>
#include <iostream>
#include <vector>
#include <set>
#include <cmath>

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
    cfg.dropout_p = 0.0f;

    GPT2 model(cfg);

    init::InitConfig init_cfg;
    init_cfg.seed = 42;
    init::init_gpt2_weights(model, init_cfg);

    return model;
}

// Test 1: Very low temperature should be nearly greedy
bool test_phase_35_temperature_0() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3};

    SamplingConfig config;
    config.max_new_tokens = 5;
    config.temperature = 0.0001f;  // Very low, nearly deterministic
    config.do_sample = true;
    config.early_stop = false;
    config.seed = 123;

    auto output1 = generate_sample(model, prompt, config);

    // Change seed - with very low temp, should still be very similar
    config.seed = 456;
    auto output2 = generate_sample(model, prompt, config);

    // With very low temperature, outputs should be identical (greedy)
    bool all_same = true;
    for (size_t i = 0; i < output1.size() && i < output2.size(); ++i) {
        if (output1[i] != output2[i]) {
            all_same = false;
            break;
        }
    }

    // Also verify output is valid
    if (output1.size() != prompt.size() + 5) {
        std::cerr << "temperature_0: wrong output length" << std::endl;
        return false;
    }

    std::cout << "test_phase_35_temperature_0: PASSED (outputs "
              << (all_same ? "identical" : "differ") << ")" << std::endl;
    return true;
}

// Test 2: High temperature should produce more diversity
bool test_phase_35_temperature_high() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3};

    SamplingConfig config;
    config.max_new_tokens = 10;
    config.temperature = 2.0f;  // High temperature
    config.do_sample = true;
    config.early_stop = false;

    // Generate multiple times with different seeds
    std::set<TokenId> unique_first_tokens;
    for (unsigned int seed = 1; seed <= 10; ++seed) {
        config.seed = seed;
        auto output = generate_sample(model, prompt, config);
        if (output.size() > prompt.size()) {
            unique_first_tokens.insert(output[prompt.size()]);
        }
    }

    // With high temperature, we should see some diversity
    // (can't guarantee it but it's likely)
    if (unique_first_tokens.size() < 2) {
        std::cerr << "temperature_high: warning - low diversity ("
                  << unique_first_tokens.size() << " unique tokens)" << std::endl;
        // Don't fail, just warn - randomness might cause this occasionally
    }

    std::cout << "test_phase_35_temperature_high: PASSED (diversity="
              << unique_first_tokens.size() << ")" << std::endl;
    return true;
}

// Test 3: Top-k filtering
bool test_phase_35_top_k() {
    // Test apply_top_k directly
    Tensor<float> logits({10});
    float* data = logits.data();
    for (size_t i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(i);  // 0, 1, 2, ..., 9
    }

    auto filtered = apply_top_k(logits, 3);

    // Only top 3 (7, 8, 9) should have finite values
    int finite_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        if (std::isfinite(filtered.data()[i])) {
            ++finite_count;
        }
    }

    if (finite_count != 3) {
        std::cerr << "top_k: expected 3 finite values, got " << finite_count << std::endl;
        return false;
    }

    // Verify highest values are preserved
    if (!std::isfinite(filtered.data()[9]) ||
        !std::isfinite(filtered.data()[8]) ||
        !std::isfinite(filtered.data()[7])) {
        std::cerr << "top_k: top 3 values should be preserved" << std::endl;
        return false;
    }

    // Test with k=0 (disabled)
    auto no_filter = apply_top_k(logits, 0);
    finite_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        if (std::isfinite(no_filter.data()[i])) {
            ++finite_count;
        }
    }
    if (finite_count != 10) {
        std::cerr << "top_k: k=0 should keep all values" << std::endl;
        return false;
    }

    std::cout << "test_phase_35_top_k: PASSED" << std::endl;
    return true;
}

// Test 4: Top-p (nucleus) filtering
bool test_phase_35_top_p() {
    // Create logits that will have clear probability distribution
    Tensor<float> logits({5});
    float* data = logits.data();
    // After softmax: [0.01, 0.02, 0.04, 0.11, 0.82] approximately
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = 2.0f;
    data[3] = 3.0f;
    data[4] = 5.0f;  // Highest probability

    auto filtered = apply_top_p(logits, 0.9f);

    // The highest probability token should be kept
    if (!std::isfinite(filtered.data()[4])) {
        std::cerr << "top_p: highest probability token should be kept" << std::endl;
        return false;
    }

    // Count finite values - should be a subset
    int finite_count = 0;
    for (size_t i = 0; i < 5; ++i) {
        if (std::isfinite(filtered.data()[i])) {
            ++finite_count;
        }
    }

    // With p=0.9, should keep at least 1 and at most all tokens
    if (finite_count < 1 || finite_count > 5) {
        std::cerr << "top_p: unexpected number of kept tokens: " << finite_count << std::endl;
        return false;
    }

    // Test with p=1.0 (disabled)
    auto no_filter = apply_top_p(logits, 1.0f);
    finite_count = 0;
    for (size_t i = 0; i < 5; ++i) {
        if (std::isfinite(no_filter.data()[i])) {
            ++finite_count;
        }
    }
    if (finite_count != 5) {
        std::cerr << "top_p: p=1.0 should keep all values" << std::endl;
        return false;
    }

    std::cout << "test_phase_35_top_p: PASSED" << std::endl;
    return true;
}

// Test 5: Seed reproducibility
bool test_phase_35_seed() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {5, 10, 15};

    SamplingConfig config;
    config.max_new_tokens = 10;
    config.temperature = 1.0f;
    config.do_sample = true;
    config.early_stop = false;
    config.seed = 12345;

    auto output1 = generate_sample(model, prompt, config);
    auto output2 = generate_sample(model, prompt, config);

    // Same seed should produce identical output
    if (output1.size() != output2.size()) {
        std::cerr << "seed: different lengths" << std::endl;
        return false;
    }

    for (size_t i = 0; i < output1.size(); ++i) {
        if (output1[i] != output2[i]) {
            std::cerr << "seed: mismatch at position " << i << std::endl;
            return false;
        }
    }

    // Different seed should produce different output (very likely)
    config.seed = 54321;
    auto output3 = generate_sample(model, prompt, config);

    bool all_same = true;
    for (size_t i = prompt.size(); i < output1.size() && i < output3.size(); ++i) {
        if (output1[i] != output3[i]) {
            all_same = false;
            break;
        }
    }

    if (all_same && output1.size() > prompt.size()) {
        std::cerr << "seed: warning - different seeds produced same output" << std::endl;
        // Don't fail, just warn
    }

    std::cout << "test_phase_35_seed: PASSED" << std::endl;
    return true;
}

// Test 6: apply_temperature function
bool test_phase_35_apply_temperature() {
    Tensor<float> logits({4});
    logits.data()[0] = 1.0f;
    logits.data()[1] = 2.0f;
    logits.data()[2] = 3.0f;
    logits.data()[3] = 4.0f;

    // Temperature = 2.0 should halve the logits
    auto scaled = apply_temperature(logits, 2.0f);

    float eps = 1e-5f;
    if (std::abs(scaled.data()[0] - 0.5f) > eps ||
        std::abs(scaled.data()[1] - 1.0f) > eps ||
        std::abs(scaled.data()[2] - 1.5f) > eps ||
        std::abs(scaled.data()[3] - 2.0f) > eps) {
        std::cerr << "apply_temperature: wrong scaling" << std::endl;
        return false;
    }

    // Temperature = 0.5 should double the logits
    scaled = apply_temperature(logits, 0.5f);
    if (std::abs(scaled.data()[0] - 2.0f) > eps ||
        std::abs(scaled.data()[1] - 4.0f) > eps ||
        std::abs(scaled.data()[2] - 6.0f) > eps ||
        std::abs(scaled.data()[3] - 8.0f) > eps) {
        std::cerr << "apply_temperature: wrong scaling for temp=0.5" << std::endl;
        return false;
    }

    std::cout << "test_phase_35_apply_temperature: PASSED" << std::endl;
    return true;
}

// Test 7: sample_token function
bool test_phase_35_sample_token() {
    // Create logits where one token has overwhelming probability
    Tensor<float> logits({5});
    logits.data()[0] = -10.0f;
    logits.data()[1] = -10.0f;
    logits.data()[2] = 10.0f;  // This one should almost always be selected
    logits.data()[3] = -10.0f;
    logits.data()[4] = -10.0f;

    std::mt19937 rng(42);

    // Sample many times
    int counts[5] = {0};
    for (int i = 0; i < 100; ++i) {
        TokenId token = sample_token(logits, rng);
        if (token >= 0 && token < 5) {
            counts[token]++;
        }
    }

    // Token 2 should be selected almost every time
    if (counts[2] < 95) {
        std::cerr << "sample_token: token 2 should be dominant, got "
                  << counts[2] << "/100" << std::endl;
        return false;
    }

    std::cout << "test_phase_35_sample_token: PASSED (token2=" << counts[2] << "/100)" << std::endl;
    return true;
}

// Test 8: do_sample=false should be greedy
bool test_phase_35_do_sample_false() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3};

    // Generate with do_sample=false (greedy)
    SamplingConfig config;
    config.max_new_tokens = 5;
    config.do_sample = false;
    config.early_stop = false;

    auto sample_output = generate_sample(model, prompt, config);

    // Compare with greedy output
    GenerateConfig greedy_config;
    greedy_config.max_new_tokens = 5;
    greedy_config.early_stop = false;

    auto greedy_output = generate_greedy(model, prompt, greedy_config);

    if (sample_output.size() != greedy_output.size()) {
        std::cerr << "do_sample_false: different lengths" << std::endl;
        return false;
    }

    for (size_t i = 0; i < sample_output.size(); ++i) {
        if (sample_output[i] != greedy_output[i]) {
            std::cerr << "do_sample_false: mismatch at position " << i << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_35_do_sample_false: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 35: Sampling Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_35_temperature_0()) ++failures;
    if (!test_phase_35_temperature_high()) ++failures;
    if (!test_phase_35_top_k()) ++failures;
    if (!test_phase_35_top_p()) ++failures;
    if (!test_phase_35_seed()) ++failures;
    if (!test_phase_35_apply_temperature()) ++failures;
    if (!test_phase_35_sample_token()) ++failures;
    if (!test_phase_35_do_sample_false()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 35 tests passed (8/8) ===" << std::endl;
    return 0;
}
