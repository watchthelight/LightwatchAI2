// Phase 33: Weight Initialization Tests

#include <lightwatch/init.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::init;

bool float_eq(float a, float b, float eps = 0.01f) {
    return std::abs(a - b) < eps;
}

// Test 1: Check weight std is approximately 0.02
bool test_phase_33_init_std() {
    // Create a small model for testing
    models::GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.embed_dim = 64;
    cfg.num_heads = 4;
    cfg.num_layers = 2;
    cfg.ffn_dim = 256;

    models::GPT2 model(cfg);

    // Initialize with default config
    InitConfig init_cfg;
    init_cfg.std = 0.02f;
    init_cfg.seed = 42;
    init_gpt2_weights(model, init_cfg);

    // Check embedding weight std
    auto& wte = model.embedding.wte().weight;
    auto stats = compute_stats(wte.data().data(), wte.numel());

    // Mean should be close to 0
    if (std::abs(stats.mean) > 0.01f) {
        std::cerr << "init_std: wte mean = " << stats.mean << " (expected ~0)" << std::endl;
        return false;
    }

    // Std should be close to 0.02
    if (!float_eq(stats.std, 0.02f, 0.005f)) {
        std::cerr << "init_std: wte std = " << stats.std << " (expected ~0.02)" << std::endl;
        return false;
    }

    std::cout << "test_phase_33_init_std: PASSED (mean=" << stats.mean
              << ", std=" << stats.std << ")" << std::endl;
    return true;
}

// Test 2: Check residual layer scaling
bool test_phase_33_init_residual() {
    models::GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.embed_dim = 64;
    cfg.num_heads = 4;
    cfg.num_layers = 4;  // 4 layers for clear scaling
    cfg.ffn_dim = 256;

    models::GPT2 model(cfg);

    // Initialize with depth scaling
    InitConfig init_cfg;
    init_cfg.std = 0.02f;
    init_cfg.scale_by_depth = true;
    init_cfg.seed = 42;
    init_gpt2_weights(model, init_cfg);

    // Expected residual std: 0.02 / sqrt(2 * 4) = 0.02 / sqrt(8) ≈ 0.00707
    float expected_residual_std = 0.02f / std::sqrt(2.0f * 4.0f);

    // Check fc2 weight (residual projection)
    auto& fc2 = model.layers[0]->ffn.fc2.weight;
    auto stats = compute_stats(fc2.data().data(), fc2.numel());

    // Mean should be close to 0
    if (std::abs(stats.mean) > 0.01f) {
        std::cerr << "init_residual: fc2 mean = " << stats.mean << " (expected ~0)" << std::endl;
        return false;
    }

    // Std should be close to expected_residual_std
    if (!float_eq(stats.std, expected_residual_std, 0.003f)) {
        std::cerr << "init_residual: fc2 std = " << stats.std
                  << " (expected ~" << expected_residual_std << ")" << std::endl;
        return false;
    }

    // Check that non-residual layer (fc1) has normal std
    auto& fc1 = model.layers[0]->ffn.fc1.weight;
    auto fc1_stats = compute_stats(fc1.data().data(), fc1.numel());

    if (!float_eq(fc1_stats.std, 0.02f, 0.005f)) {
        std::cerr << "init_residual: fc1 std = " << fc1_stats.std
                  << " (expected ~0.02)" << std::endl;
        return false;
    }

    std::cout << "test_phase_33_init_residual: PASSED (fc2_std=" << stats.std
              << ", expected=" << expected_residual_std << ")" << std::endl;
    return true;
}

// Test 3: Check LayerNorm initialization
bool test_phase_33_init_layernorm() {
    models::GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.embed_dim = 64;
    cfg.num_heads = 4;
    cfg.num_layers = 2;
    cfg.ffn_dim = 256;

    models::GPT2 model(cfg);

    InitConfig init_cfg;
    init_cfg.seed = 42;
    init_gpt2_weights(model, init_cfg);

    // Check ln_f weight (should be all 1s)
    auto& ln_weight = model.ln_f.weight;
    for (size_t i = 0; i < ln_weight.numel(); ++i) {
        float val = ln_weight.data().data()[i];
        if (!float_eq(val, 1.0f, 1e-5f)) {
            std::cerr << "init_layernorm: weight[" << i << "] = " << val
                      << " (expected 1.0)" << std::endl;
            return false;
        }
    }

    // Check ln_f bias (should be all 0s)
    auto& ln_bias = model.ln_f.bias;
    for (size_t i = 0; i < ln_bias.numel(); ++i) {
        float val = ln_bias.data().data()[i];
        if (!float_eq(val, 0.0f, 1e-5f)) {
            std::cerr << "init_layernorm: bias[" << i << "] = " << val
                      << " (expected 0.0)" << std::endl;
            return false;
        }
    }

    // Check layer ln1 weights
    auto& ln1_weight = model.layers[0]->ln1.weight;
    for (size_t i = 0; i < ln1_weight.numel(); ++i) {
        float val = ln1_weight.data().data()[i];
        if (!float_eq(val, 1.0f, 1e-5f)) {
            std::cerr << "init_layernorm: ln1.weight[" << i << "] = " << val
                      << " (expected 1.0)" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_33_init_layernorm: PASSED" << std::endl;
    return true;
}

// Test 4: Check reproducibility with same seed
bool test_phase_33_init_seed() {
    models::GPT2Config cfg;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 32;
    cfg.embed_dim = 64;
    cfg.num_heads = 4;
    cfg.num_layers = 2;
    cfg.ffn_dim = 256;

    // Create two models
    models::GPT2 model1(cfg);
    models::GPT2 model2(cfg);

    // Initialize with same seed
    InitConfig init_cfg;
    init_cfg.seed = 12345;

    init_gpt2_weights(model1, init_cfg);
    init_gpt2_weights(model2, init_cfg);

    // Check that wte weights are identical
    auto& wte1 = model1.embedding.wte().weight;
    auto& wte2 = model2.embedding.wte().weight;

    for (size_t i = 0; i < std::min(size_t(100), wte1.numel()); ++i) {
        float v1 = wte1.data().data()[i];
        float v2 = wte2.data().data()[i];
        if (!float_eq(v1, v2, 1e-6f)) {
            std::cerr << "init_seed: wte[" << i << "] differs: " << v1 << " vs " << v2 << std::endl;
            return false;
        }
    }

    // Check that fc1 weights are identical
    auto& fc1_1 = model1.layers[0]->ffn.fc1.weight;
    auto& fc1_2 = model2.layers[0]->ffn.fc1.weight;

    for (size_t i = 0; i < std::min(size_t(100), fc1_1.numel()); ++i) {
        float v1 = fc1_1.data().data()[i];
        float v2 = fc1_2.data().data()[i];
        if (!float_eq(v1, v2, 1e-6f)) {
            std::cerr << "init_seed: fc1[" << i << "] differs: " << v1 << " vs " << v2 << std::endl;
            return false;
        }
    }

    // Verify different seeds give different results
    models::GPT2 model3(cfg);
    init_cfg.seed = 99999;
    init_gpt2_weights(model3, init_cfg);

    auto& wte3 = model3.embedding.wte().weight;
    bool all_same = true;
    for (size_t i = 0; i < std::min(size_t(10), wte1.numel()); ++i) {
        if (!float_eq(wte1.data().data()[i], wte3.data().data()[i], 1e-6f)) {
            all_same = false;
            break;
        }
    }

    if (all_same) {
        std::cerr << "init_seed: different seeds produced same weights!" << std::endl;
        return false;
    }

    std::cout << "test_phase_33_init_seed: PASSED" << std::endl;
    return true;
}

// Test 5: Individual layer init functions
bool test_phase_33_individual_init() {
    // Test init_linear
    nn::Linear layer(64, 128, true);
    init_linear(layer, 0.05f);

    auto stats = compute_stats(layer.weight.data().data(), layer.weight.numel());
    if (!float_eq(stats.std, 0.05f, 0.01f)) {
        std::cerr << "individual_init: linear std = " << stats.std << " (expected ~0.05)" << std::endl;
        return false;
    }

    // Check bias is zeros (bias is already zero-initialized by Linear)
    const auto& bias = layer.bias();
    for (size_t i = 0; i < bias.numel(); ++i) {
        if (std::abs(bias.data().data()[i]) > 1e-6f) {
            std::cerr << "individual_init: linear bias not zero" << std::endl;
            return false;
        }
    }

    // Test init_layer_norm
    nn::LayerNorm ln(64);
    init_layer_norm(ln);

    for (size_t i = 0; i < ln.weight.numel(); ++i) {
        if (!float_eq(ln.weight.data().data()[i], 1.0f, 1e-5f)) {
            std::cerr << "individual_init: ln weight != 1" << std::endl;
            return false;
        }
    }

    for (size_t i = 0; i < ln.bias.numel(); ++i) {
        if (!float_eq(ln.bias.data().data()[i], 0.0f, 1e-5f)) {
            std::cerr << "individual_init: ln bias != 0" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_33_individual_init: PASSED" << std::endl;
    return true;
}

// Test 6: RandomGen class
bool test_phase_33_random_gen() {
    RandomGen rng1(42);
    RandomGen rng2(42);

    std::vector<float> data1(100);
    std::vector<float> data2(100);

    rng1.fill_normal(data1.data(), 100, 0.0f, 1.0f);
    rng2.fill_normal(data2.data(), 100, 0.0f, 1.0f);

    // Same seed should give same values
    for (size_t i = 0; i < 100; ++i) {
        if (!float_eq(data1[i], data2[i], 1e-6f)) {
            std::cerr << "random_gen: same seed gave different values" << std::endl;
            return false;
        }
    }

    // Test fill_constant
    rng1.fill_constant(data1.data(), 100, 3.14f);
    for (size_t i = 0; i < 100; ++i) {
        if (!float_eq(data1[i], 3.14f, 1e-6f)) {
            std::cerr << "random_gen: fill_constant failed" << std::endl;
            return false;
        }
    }

    // Test reseed
    rng1.reseed(42);
    rng1.fill_normal(data1.data(), 100, 0.0f, 1.0f);

    // Should match data2 again
    for (size_t i = 0; i < 100; ++i) {
        if (!float_eq(data1[i], data2[i], 1e-6f)) {
            std::cerr << "random_gen: reseed didn't restore sequence" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_33_random_gen: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 33: Weight Initialization Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_33_init_std()) ++failures;
    if (!test_phase_33_init_residual()) ++failures;
    if (!test_phase_33_init_layernorm()) ++failures;
    if (!test_phase_33_init_seed()) ++failures;
    if (!test_phase_33_individual_init()) ++failures;
    if (!test_phase_33_random_gen()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 33 tests passed (6/6) ===" << std::endl;
    return 0;
}
