// Phase 37: Serialization Tests

#include <lightwatch/serialize.hpp>
#include <lightwatch/models/gpt.hpp>
#include <lightwatch/init.hpp>
#include <iostream>
#include <cmath>
#include <filesystem>
#include <cstring>

using namespace lightwatch;
using namespace lightwatch::models;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

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

// Test 1: Save and load weights
bool test_phase_37_save_load() {
    std::string path = "/tmp/test_weights.lwbin";

    // Create model and get state
    auto model1 = create_test_model();
    auto state1 = model1.state_dict();

    // Save weights
    try {
        save_weights(path, model1);
    } catch (const SerializeError& e) {
        std::cerr << "save_load: save failed: " << e.what() << std::endl;
        return false;
    }

    // Create new model
    auto model2 = create_test_model();

    // Modify weights to ensure they're different
    for (auto& [name, tensor] : model2.state_dict()) {
        for (size_t i = 0; i < tensor.numel(); ++i) {
            tensor.data()[i] = 0.0f;
        }
    }

    // Load weights
    try {
        load_weights(path, model2);
    } catch (const SerializeError& e) {
        std::cerr << "save_load: load failed: " << e.what() << std::endl;
        return false;
    }

    // Compare states
    auto state2 = model2.state_dict();

    for (const auto& [name, tensor1] : state1) {
        auto it = state2.find(name);
        if (it == state2.end()) {
            std::cerr << "save_load: missing tensor '" << name << "'" << std::endl;
            return false;
        }

        const auto& tensor2 = it->second;
        if (tensor1.numel() != tensor2.numel()) {
            std::cerr << "save_load: size mismatch for '" << name << "'" << std::endl;
            return false;
        }

        for (size_t i = 0; i < std::min(size_t(100), tensor1.numel()); ++i) {
            if (!float_eq(tensor1.data()[i], tensor2.data()[i])) {
                std::cerr << "save_load: value mismatch for '" << name
                          << "' at index " << i << std::endl;
                return false;
            }
        }
    }

    // Cleanup
    std::filesystem::remove(path);

    std::cout << "test_phase_37_save_load: PASSED" << std::endl;
    return true;
}

// Test 2: Check header format
bool test_phase_37_header() {
    std::string path = "/tmp/test_header.lwbin";

    auto model = create_test_model();
    save_weights(path, model);

    // Read header
    auto header = read_header(path);

    // Check magic
    if (std::memcmp(header.magic, LWBIN_MAGIC, 4) != 0) {
        std::cerr << "header: wrong magic" << std::endl;
        return false;
    }

    // Check version
    if (header.version != LWBIN_VERSION) {
        std::cerr << "header: wrong version " << header.version << std::endl;
        return false;
    }

    // Check tensor count
    auto state_dict = model.state_dict();
    if (header.tensor_count != state_dict.size()) {
        std::cerr << "header: wrong tensor count " << header.tensor_count
                  << " (expected " << state_dict.size() << ")" << std::endl;
        return false;
    }

    // Cleanup
    std::filesystem::remove(path);

    std::cout << "test_phase_37_header: PASSED" << std::endl;
    return true;
}

// Test 3: Validate weights against model
bool test_phase_37_validate() {
    std::string path = "/tmp/test_validate.lwbin";

    auto model = create_test_model();
    save_weights(path, model);

    // Valid file should validate
    if (!validate_weights(path, model)) {
        std::cerr << "validate: should pass for same model" << std::endl;
        return false;
    }

    // Create model with different architecture
    GPT2Config cfg2;
    cfg2.vocab_size = 50;  // Different
    cfg2.max_seq_len = 32;
    cfg2.embed_dim = 64;   // Different
    cfg2.num_heads = 4;    // Different
    cfg2.num_layers = 2;
    cfg2.ffn_dim = 256;    // Different

    GPT2 model2(cfg2);

    // Should fail validation
    if (validate_weights(path, model2)) {
        std::cerr << "validate: should fail for different architecture" << std::endl;
        return false;
    }

    // Cleanup
    std::filesystem::remove(path);

    std::cout << "test_phase_37_validate: PASSED" << std::endl;
    return true;
}

// Test 4: Inspect weights file
bool test_phase_37_inspect() {
    std::string path = "/tmp/test_inspect.lwbin";

    auto model = create_test_model();
    save_weights(path, model);

    auto tensors = inspect_weights(path);

    auto state_dict = model.state_dict();

    // Should have same number of tensors
    if (tensors.size() != state_dict.size()) {
        std::cerr << "inspect: wrong tensor count" << std::endl;
        return false;
    }

    // Each tensor should have valid metadata
    for (const auto& meta : tensors) {
        auto it = state_dict.find(meta.name);
        if (it == state_dict.end()) {
            std::cerr << "inspect: unknown tensor '" << meta.name << "'" << std::endl;
            return false;
        }

        const auto& model_shape = it->second.shape();
        if (model_shape.size() != meta.shape.size()) {
            std::cerr << "inspect: ndims mismatch for '" << meta.name << "'" << std::endl;
            return false;
        }

        for (size_t i = 0; i < model_shape.size(); ++i) {
            if (static_cast<int64_t>(model_shape[i]) != meta.shape[i]) {
                std::cerr << "inspect: shape mismatch for '" << meta.name << "'" << std::endl;
                return false;
            }
        }
    }

    // Cleanup
    std::filesystem::remove(path);

    std::cout << "test_phase_37_inspect: PASSED" << std::endl;
    return true;
}

// Test 5: is_valid_lwbin function
bool test_phase_37_is_valid() {
    std::string valid_path = "/tmp/test_valid.lwbin";
    std::string invalid_path = "/tmp/test_invalid.txt";

    auto model = create_test_model();
    save_weights(valid_path, model);

    // Valid .lwbin file
    if (!is_valid_lwbin(valid_path)) {
        std::cerr << "is_valid: should return true for valid file" << std::endl;
        return false;
    }

    // Create invalid file
    {
        std::ofstream f(invalid_path);
        f << "This is not a .lwbin file";
    }

    if (is_valid_lwbin(invalid_path)) {
        std::cerr << "is_valid: should return false for invalid file" << std::endl;
        return false;
    }

    // Non-existent file
    if (is_valid_lwbin("/tmp/nonexistent_file.lwbin")) {
        std::cerr << "is_valid: should return false for non-existent file" << std::endl;
        return false;
    }

    // Cleanup
    std::filesystem::remove(valid_path);
    std::filesystem::remove(invalid_path);

    std::cout << "test_phase_37_is_valid: PASSED" << std::endl;
    return true;
}

// Test 6: Error handling
bool test_phase_37_errors() {
    // Try to load non-existent file
    auto model = create_test_model();

    bool caught = false;
    try {
        load_weights("/tmp/nonexistent_weights.lwbin", model);
    } catch (const SerializeError&) {
        caught = true;
    }

    if (!caught) {
        std::cerr << "errors: should throw for non-existent file" << std::endl;
        return false;
    }

    // Try to load invalid file
    std::string invalid_path = "/tmp/invalid_weights.bin";
    {
        std::ofstream f(invalid_path, std::ios::binary);
        f << "INVALID MAGIC";
    }

    caught = false;
    try {
        load_weights(invalid_path, model);
    } catch (const SerializeError&) {
        caught = true;
    }

    if (!caught) {
        std::cerr << "errors: should throw for invalid magic" << std::endl;
        return false;
    }

    // Cleanup
    std::filesystem::remove(invalid_path);

    std::cout << "test_phase_37_errors: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 37: Serialization Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_37_save_load()) ++failures;
    if (!test_phase_37_header()) ++failures;
    if (!test_phase_37_validate()) ++failures;
    if (!test_phase_37_inspect()) ++failures;
    if (!test_phase_37_is_valid()) ++failures;
    if (!test_phase_37_errors()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 37 tests passed (6/6) ===" << std::endl;
    return 0;
}
