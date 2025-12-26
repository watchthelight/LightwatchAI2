// Phase 32: Model Config Tests

#include <lightwatch/model_config.hpp>
#include <iostream>
#include <cmath>
#include <filesystem>

using namespace lightwatch;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Load preset config
bool test_phase_32_load_preset() {
    auto config = ModelConfig::preset("gpt2-small");

    if (config.model_type != "gpt2") {
        std::cerr << "load_preset: wrong model_type" << std::endl;
        return false;
    }

    if (config.gpt2.vocab_size != 50257) {
        std::cerr << "load_preset: wrong vocab_size" << std::endl;
        return false;
    }

    if (config.gpt2.embed_dim != 768) {
        std::cerr << "load_preset: wrong embed_dim" << std::endl;
        return false;
    }

    if (config.gpt2.num_heads != 12) {
        std::cerr << "load_preset: wrong num_heads" << std::endl;
        return false;
    }

    if (config.gpt2.num_layers != 12) {
        std::cerr << "load_preset: wrong num_layers" << std::endl;
        return false;
    }

    if (config.gpt2.ffn_dim != 3072) {
        std::cerr << "load_preset: wrong ffn_dim" << std::endl;
        return false;
    }

    std::cout << "test_phase_32_load_preset: PASSED" << std::endl;
    return true;
}

// Test 2: JSON roundtrip
bool test_phase_32_json_roundtrip() {
    ModelConfig original;
    original.model_type = "gpt2";
    original.gpt2.vocab_size = 1000;
    original.gpt2.max_seq_len = 128;
    original.gpt2.embed_dim = 256;
    original.gpt2.num_heads = 8;
    original.gpt2.num_layers = 4;
    original.gpt2.ffn_dim = 1024;
    original.gpt2.dropout_p = 0.05f;
    original.gpt2.attn_dropout_p = 0.05f;
    original.gpt2.tie_weights = false;

    // Serialize to JSON
    std::string json = original.to_json();

    // Parse back
    auto parsed = ModelConfig::from_json(json);

    // Verify all fields match
    if (parsed.model_type != original.model_type) {
        std::cerr << "json_roundtrip: model_type mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.vocab_size != original.gpt2.vocab_size) {
        std::cerr << "json_roundtrip: vocab_size mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.max_seq_len != original.gpt2.max_seq_len) {
        std::cerr << "json_roundtrip: max_seq_len mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.embed_dim != original.gpt2.embed_dim) {
        std::cerr << "json_roundtrip: embed_dim mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.num_heads != original.gpt2.num_heads) {
        std::cerr << "json_roundtrip: num_heads mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.num_layers != original.gpt2.num_layers) {
        std::cerr << "json_roundtrip: num_layers mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.ffn_dim != original.gpt2.ffn_dim) {
        std::cerr << "json_roundtrip: ffn_dim mismatch" << std::endl;
        return false;
    }

    if (!float_eq(parsed.gpt2.dropout_p, original.gpt2.dropout_p)) {
        std::cerr << "json_roundtrip: dropout_p mismatch" << std::endl;
        return false;
    }

    if (!float_eq(parsed.gpt2.attn_dropout_p, original.gpt2.attn_dropout_p)) {
        std::cerr << "json_roundtrip: attn_dropout_p mismatch" << std::endl;
        return false;
    }

    if (parsed.gpt2.tie_weights != original.gpt2.tie_weights) {
        std::cerr << "json_roundtrip: tie_weights mismatch" << std::endl;
        return false;
    }

    std::cout << "test_phase_32_json_roundtrip: PASSED" << std::endl;
    return true;
}

// Test 3: Validation - invalid configs
bool test_phase_32_validation() {
    // Test: embed_dim not divisible by num_heads
    {
        ModelConfig config;
        config.model_type = "gpt2";
        config.gpt2.vocab_size = 1000;
        config.gpt2.max_seq_len = 128;
        config.gpt2.embed_dim = 100;  // Not divisible by 12
        config.gpt2.num_heads = 12;
        config.gpt2.num_layers = 4;
        config.gpt2.ffn_dim = 400;

        bool threw = false;
        try {
            validate_config(config);
        } catch (const ConfigError& e) {
            threw = true;
            std::string msg = e.what();
            if (msg.find("divisible") == std::string::npos) {
                std::cerr << "validation: wrong error message for divisibility" << std::endl;
                return false;
            }
        }

        if (!threw) {
            std::cerr << "validation: should throw for embed_dim not divisible by num_heads" << std::endl;
            return false;
        }
    }

    // Test: zero vocab_size
    {
        ModelConfig config;
        config.gpt2.vocab_size = 0;
        config.gpt2.max_seq_len = 128;
        config.gpt2.embed_dim = 256;
        config.gpt2.num_heads = 8;
        config.gpt2.num_layers = 4;
        config.gpt2.ffn_dim = 1024;

        bool threw = false;
        try {
            validate_config(config);
        } catch (const ConfigError&) {
            threw = true;
        }

        if (!threw) {
            std::cerr << "validation: should throw for zero vocab_size" << std::endl;
            return false;
        }
    }

    // Test: invalid dropout
    {
        ModelConfig config;
        config.gpt2.vocab_size = 1000;
        config.gpt2.max_seq_len = 128;
        config.gpt2.embed_dim = 256;
        config.gpt2.num_heads = 8;
        config.gpt2.num_layers = 4;
        config.gpt2.ffn_dim = 1024;
        config.gpt2.dropout_p = 1.5f;  // Invalid

        bool threw = false;
        try {
            validate_config(config);
        } catch (const ConfigError&) {
            threw = true;
        }

        if (!threw) {
            std::cerr << "validation: should throw for invalid dropout" << std::endl;
            return false;
        }
    }

    // Test: valid config should pass
    {
        auto config = ModelConfig::preset("gpt2-small");
        try {
            validate_config(config);
        } catch (const ConfigError& e) {
            std::cerr << "validation: gpt2-small should be valid: " << e.what() << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_32_validation: PASSED" << std::endl;
    return true;
}

// Test 4: Custom config from JSON
bool test_phase_32_custom_config() {
    std::string json = R"({
        "model_type": "gpt2",
        "vocab_size": 5000,
        "max_seq_len": 512,
        "embed_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "ffn_dim": 2048,
        "dropout_p": 0.2,
        "attn_dropout_p": 0.15,
        "tie_weights": false
    })";

    auto config = ModelConfig::from_json(json);

    if (config.gpt2.vocab_size != 5000) {
        std::cerr << "custom_config: wrong vocab_size" << std::endl;
        return false;
    }

    if (config.gpt2.max_seq_len != 512) {
        std::cerr << "custom_config: wrong max_seq_len" << std::endl;
        return false;
    }

    if (config.gpt2.embed_dim != 512) {
        std::cerr << "custom_config: wrong embed_dim" << std::endl;
        return false;
    }

    if (config.gpt2.num_heads != 8) {
        std::cerr << "custom_config: wrong num_heads" << std::endl;
        return false;
    }

    if (config.gpt2.num_layers != 6) {
        std::cerr << "custom_config: wrong num_layers" << std::endl;
        return false;
    }

    if (config.gpt2.ffn_dim != 2048) {
        std::cerr << "custom_config: wrong ffn_dim" << std::endl;
        return false;
    }

    if (!float_eq(config.gpt2.dropout_p, 0.2f)) {
        std::cerr << "custom_config: wrong dropout_p" << std::endl;
        return false;
    }

    if (!float_eq(config.gpt2.attn_dropout_p, 0.15f)) {
        std::cerr << "custom_config: wrong attn_dropout_p" << std::endl;
        return false;
    }

    if (config.gpt2.tie_weights != false) {
        std::cerr << "custom_config: wrong tie_weights" << std::endl;
        return false;
    }

    // Validate the custom config
    try {
        validate_config(config);
    } catch (const ConfigError& e) {
        std::cerr << "custom_config: validation failed: " << e.what() << std::endl;
        return false;
    }

    std::cout << "test_phase_32_custom_config: PASSED" << std::endl;
    return true;
}

// Test 5: File save/load roundtrip
bool test_phase_32_file_roundtrip() {
    std::string test_path = "/tmp/test_config.json";

    ModelConfig original = ModelConfig::preset("gpt2-medium");

    // Save to file
    original.save(test_path);

    // Load from file
    auto loaded = ModelConfig::load(test_path);

    // Verify
    if (loaded.gpt2.embed_dim != original.gpt2.embed_dim) {
        std::cerr << "file_roundtrip: embed_dim mismatch" << std::endl;
        return false;
    }

    if (loaded.gpt2.num_layers != original.gpt2.num_layers) {
        std::cerr << "file_roundtrip: num_layers mismatch" << std::endl;
        return false;
    }

    // Cleanup
    std::filesystem::remove(test_path);

    std::cout << "test_phase_32_file_roundtrip: PASSED" << std::endl;
    return true;
}

// Test 6: Load from configs directory
bool test_phase_32_load_from_file() {
    try {
        auto config = ModelConfig::load("configs/gpt2-small.json");

        if (config.gpt2.vocab_size != 50257) {
            std::cerr << "load_from_file: wrong vocab_size" << std::endl;
            return false;
        }

        if (config.gpt2.embed_dim != 768) {
            std::cerr << "load_from_file: wrong embed_dim" << std::endl;
            return false;
        }

        validate_config(config);

    } catch (const ConfigError& e) {
        std::cerr << "load_from_file: " << e.what() << std::endl;
        return false;
    }

    std::cout << "test_phase_32_load_from_file: PASSED" << std::endl;
    return true;
}

// Test 7: Invalid JSON handling
bool test_phase_32_invalid_json() {
    // Test invalid JSON syntax
    {
        bool threw = false;
        try {
            ModelConfig::from_json("{ invalid json }");
        } catch (const ConfigError& e) {
            threw = true;
            std::string msg = e.what();
            if (msg.find("parse error") == std::string::npos) {
                std::cerr << "invalid_json: wrong error message" << std::endl;
                return false;
            }
        }

        if (!threw) {
            std::cerr << "invalid_json: should throw for invalid JSON" << std::endl;
            return false;
        }
    }

    // Test unknown model type
    {
        bool threw = false;
        try {
            ModelConfig::from_json(R"({"model_type": "unknown_model"})");
        } catch (const ConfigError& e) {
            threw = true;
        }

        if (!threw) {
            std::cerr << "invalid_json: should throw for unknown model type" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_32_invalid_json: PASSED" << std::endl;
    return true;
}

// Test 8: Preset variants
bool test_phase_32_presets() {
    auto small = ModelConfig::preset("gpt2-small");
    auto medium = ModelConfig::preset("gpt2-medium");
    auto large = ModelConfig::preset("gpt2-large");
    auto xl = ModelConfig::preset("gpt2-xl");

    // Verify dimensions increase
    if (small.gpt2.embed_dim >= medium.gpt2.embed_dim) {
        std::cerr << "presets: small.embed_dim should < medium.embed_dim" << std::endl;
        return false;
    }

    if (medium.gpt2.embed_dim >= large.gpt2.embed_dim) {
        std::cerr << "presets: medium.embed_dim should < large.embed_dim" << std::endl;
        return false;
    }

    if (large.gpt2.embed_dim >= xl.gpt2.embed_dim) {
        std::cerr << "presets: large.embed_dim should < xl.embed_dim" << std::endl;
        return false;
    }

    // All presets should be valid
    try {
        validate_config(small);
        validate_config(medium);
        validate_config(large);
        validate_config(xl);
    } catch (const ConfigError& e) {
        std::cerr << "presets: validation failed: " << e.what() << std::endl;
        return false;
    }

    // Test unknown preset
    bool threw = false;
    try {
        ModelConfig::preset("unknown-preset");
    } catch (const ConfigError&) {
        threw = true;
    }

    if (!threw) {
        std::cerr << "presets: should throw for unknown preset" << std::endl;
        return false;
    }

    std::cout << "test_phase_32_presets: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 32: Model Config Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_32_load_preset()) ++failures;
    if (!test_phase_32_json_roundtrip()) ++failures;
    if (!test_phase_32_validation()) ++failures;
    if (!test_phase_32_custom_config()) ++failures;
    if (!test_phase_32_file_roundtrip()) ++failures;
    if (!test_phase_32_load_from_file()) ++failures;
    if (!test_phase_32_invalid_json()) ++failures;
    if (!test_phase_32_presets()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 32 tests passed (8/8) ===" << std::endl;
    return 0;
}
