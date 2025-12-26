// Phase 36: KV-Cache Tests

#include <lightwatch/generate.hpp>
#include <lightwatch/cache.hpp>
#include <lightwatch/init.hpp>
#include <iostream>
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
    cfg.num_layers = 2;
    cfg.ffn_dim = 128;
    cfg.dropout_p = 0.0f;

    GPT2 model(cfg);

    init::InitConfig init_cfg;
    init_cfg.seed = 42;
    init::init_gpt2_weights(model, init_cfg);

    return model;
}

// Test 1: Cache update increases seq_len
bool test_phase_36_cache_update() {
    size_t num_layers = 2;
    size_t num_heads = 4;
    size_t head_dim = 16;
    size_t max_seq_len = 32;
    size_t batch_size = 1;

    KVCache cache(num_layers, num_heads, head_dim, max_seq_len, batch_size);

    if (cache.seq_len() != 0) {
        std::cerr << "cache_update: initial seq_len should be 0" << std::endl;
        return false;
    }

    // Create K, V tensors for one position
    Tensor<float> new_k({batch_size, num_heads, 3, head_dim});
    Tensor<float> new_v({batch_size, num_heads, 3, head_dim});

    // Fill with test data
    for (size_t i = 0; i < new_k.numel(); ++i) {
        new_k.data()[i] = static_cast<float>(i) * 0.01f;
        new_v.data()[i] = static_cast<float>(i) * 0.02f;
    }

    // Update cache for layer 0
    cache.update(0, new_k, new_v);
    cache.update(1, new_k, new_v);
    cache.advance(3);

    if (cache.seq_len() != 3) {
        std::cerr << "cache_update: seq_len should be 3 after update, got "
                  << cache.seq_len() << std::endl;
        return false;
    }

    // Verify we can get the cached values back
    auto [k, v] = cache.get(0);

    if (k.shape()[2] != 3 || v.shape()[2] != 3) {
        std::cerr << "cache_update: retrieved cache has wrong seq_len" << std::endl;
        return false;
    }

    std::cout << "test_phase_36_cache_update: PASSED" << std::endl;
    return true;
}

// Test 2: Cached generation produces same output as non-cached
bool test_phase_36_cache_equivalence() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3, 4, 5};

    // Generate without cache
    SamplingConfig config;
    config.max_new_tokens = 5;
    config.do_sample = false;  // Greedy for deterministic comparison
    config.early_stop = false;

    auto output_no_cache = generate_sample(model, prompt, config);

    // Generate with cache
    auto cache = create_cache_for_model(model);
    auto output_with_cache = generate_with_cache(model, prompt, cache, config);

    // Outputs should be identical
    if (output_no_cache.size() != output_with_cache.size()) {
        std::cerr << "cache_equivalence: different output lengths ("
                  << output_no_cache.size() << " vs "
                  << output_with_cache.size() << ")" << std::endl;
        return false;
    }

    for (size_t i = 0; i < output_no_cache.size(); ++i) {
        if (output_no_cache[i] != output_with_cache[i]) {
            std::cerr << "cache_equivalence: mismatch at position " << i
                      << " (" << output_no_cache[i] << " vs "
                      << output_with_cache[i] << ")" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_36_cache_equivalence: PASSED" << std::endl;
    return true;
}

// Test 3: Cache reset allows independent generations
bool test_phase_36_cache_reset() {
    auto model = create_test_model();

    std::vector<TokenId> prompt1 = {1, 2, 3};
    std::vector<TokenId> prompt2 = {10, 20, 30};

    SamplingConfig config;
    config.max_new_tokens = 5;
    config.do_sample = false;
    config.early_stop = false;

    auto cache = create_cache_for_model(model);

    // First generation
    auto output1 = generate_with_cache(model, prompt1, cache, config);

    // Cache should be reset internally, generate again with different prompt
    auto output2 = generate_with_cache(model, prompt2, cache, config);

    // Verify outputs are independent
    if (output1.size() != prompt1.size() + 5 ||
        output2.size() != prompt2.size() + 5) {
        std::cerr << "cache_reset: wrong output lengths" << std::endl;
        return false;
    }

    // Verify prompts are preserved
    for (size_t i = 0; i < prompt1.size(); ++i) {
        if (output1[i] != prompt1[i]) {
            std::cerr << "cache_reset: prompt1 not preserved" << std::endl;
            return false;
        }
    }

    for (size_t i = 0; i < prompt2.size(); ++i) {
        if (output2[i] != prompt2[i]) {
            std::cerr << "cache_reset: prompt2 not preserved" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_36_cache_reset: PASSED" << std::endl;
    return true;
}

// Test 4: Cache memory calculation
bool test_phase_36_cache_memory() {
    size_t num_layers = 12;
    size_t num_heads = 12;
    size_t head_dim = 64;
    size_t max_seq_len = 1024;
    size_t batch_size = 1;

    KVCache cache(num_layers, num_heads, head_dim, max_seq_len, batch_size);

    // Expected: 2 (k+v) * 12 layers * 1 batch * 12 heads * 1024 seq * 64 head_dim * 4 bytes
    size_t expected = 2 * num_layers * batch_size * num_heads * max_seq_len * head_dim * sizeof(float);

    size_t actual = cache.memory_bytes();

    if (actual != expected) {
        std::cerr << "cache_memory: expected " << expected << " bytes, got "
                  << actual << std::endl;
        return false;
    }

    // Memory should be O(seq) not O(seq^2)
    // Verify by comparing with doubled seq_len
    KVCache cache2(num_layers, num_heads, head_dim, max_seq_len * 2, batch_size);
    size_t doubled = cache2.memory_bytes();

    if (doubled != expected * 2) {
        std::cerr << "cache_memory: memory not scaling linearly with seq_len" << std::endl;
        return false;
    }

    std::cout << "test_phase_36_cache_memory: PASSED (cache=" << actual / 1024 / 1024
              << " MB for 1024 seq)" << std::endl;
    return true;
}

// Test 5: Cache accessors
bool test_phase_36_cache_accessors() {
    size_t num_layers = 4;
    size_t num_heads = 8;
    size_t head_dim = 32;
    size_t max_seq_len = 64;
    size_t batch_size = 2;

    KVCache cache(num_layers, num_heads, head_dim, max_seq_len, batch_size);

    if (cache.num_layers() != num_layers) {
        std::cerr << "cache_accessors: wrong num_layers" << std::endl;
        return false;
    }

    if (cache.num_heads() != num_heads) {
        std::cerr << "cache_accessors: wrong num_heads" << std::endl;
        return false;
    }

    if (cache.head_dim() != head_dim) {
        std::cerr << "cache_accessors: wrong head_dim" << std::endl;
        return false;
    }

    if (cache.max_seq_len() != max_seq_len) {
        std::cerr << "cache_accessors: wrong max_seq_len" << std::endl;
        return false;
    }

    if (cache.batch_size() != batch_size) {
        std::cerr << "cache_accessors: wrong batch_size" << std::endl;
        return false;
    }

    std::cout << "test_phase_36_cache_accessors: PASSED" << std::endl;
    return true;
}

// Test 6: Create cache for model
bool test_phase_36_create_cache() {
    auto model = create_test_model();
    auto cache = create_cache_for_model(model);

    const auto& cfg = model.config();

    if (cache.num_layers() != cfg.num_layers) {
        std::cerr << "create_cache: wrong num_layers" << std::endl;
        return false;
    }

    if (cache.num_heads() != cfg.num_heads) {
        std::cerr << "create_cache: wrong num_heads" << std::endl;
        return false;
    }

    if (cache.max_seq_len() != cfg.max_seq_len) {
        std::cerr << "create_cache: wrong max_seq_len" << std::endl;
        return false;
    }

    size_t expected_head_dim = cfg.embed_dim / cfg.num_heads;
    if (cache.head_dim() != expected_head_dim) {
        std::cerr << "create_cache: wrong head_dim" << std::endl;
        return false;
    }

    std::cout << "test_phase_36_create_cache: PASSED" << std::endl;
    return true;
}

// Test 7: Benchmark function
bool test_phase_36_benchmark() {
    auto model = create_test_model();

    std::vector<TokenId> prompt = {1, 2, 3};

    auto stats = benchmark_generation(model, prompt, 3, false);

    if (stats.tokens_generated != 3) {
        std::cerr << "benchmark: wrong token count" << std::endl;
        return false;
    }

    if (stats.total_time_ms <= 0) {
        std::cerr << "benchmark: invalid time" << std::endl;
        return false;
    }

    if (stats.tokens_per_second <= 0) {
        std::cerr << "benchmark: invalid tok/s" << std::endl;
        return false;
    }

    std::cout << "test_phase_36_benchmark: PASSED ("
              << stats.tokens_per_second << " tok/s)" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 36: KV-Cache Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_36_cache_update()) ++failures;
    if (!test_phase_36_cache_equivalence()) ++failures;
    if (!test_phase_36_cache_reset()) ++failures;
    if (!test_phase_36_cache_memory()) ++failures;
    if (!test_phase_36_cache_accessors()) ++failures;
    if (!test_phase_36_create_cache()) ++failures;
    if (!test_phase_36_benchmark()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 36 tests passed (7/7) ===" << std::endl;
    return 0;
}
