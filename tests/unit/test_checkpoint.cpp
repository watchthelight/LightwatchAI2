// Phase 26: Checkpointing Tests

#include <lightwatch/checkpoint.hpp>
#include <lightwatch/nn/linear.hpp>
#include <lightwatch/optim/sgd.hpp>
#include <iostream>
#include <cmath>
#include <cstdio>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::optim;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Simple test model
class SimpleModel : public Module {
public:
    SimpleModel() : fc1(4, 8), fc2(8, 2) {
        // Linear already registers its own parameters, but we need them
        // accessible through our module. Just register the submodules'
        // params manually under qualified names.
        register_parameter("fc1.weight", fc1.weight);
        register_parameter("fc1.bias", fc1.bias());
        register_parameter("fc2.weight", fc2.weight);
        register_parameter("fc2.bias", fc2.bias());
    }

    Variable forward(const Variable& input) override {
        auto h = fc1.forward(input);
        return fc2.forward(h);
    }

    Linear fc1;
    Linear fc2;
};

// Test 1: Save and load model state
bool test_phase_26_save_load() {
    const std::string path = "/tmp/test_ckpt_1.ckpt";

    // Create model with known values
    SimpleModel model1;
    model1.fc1.weight.data().fill_(1.0f);
    model1.fc1.bias().data().fill_(0.5f);
    model1.fc2.weight.data().fill_(2.0f);
    model1.fc2.bias().data().fill_(0.1f);

    // Create optimizer
    SGDOptions opts;
    opts.lr = 0.01f;
    SGD sgd1(model1.parameters(), opts);

    // Save
    save_checkpoint(path, model1, sgd1, 10, 1000, 0.5f, "{\"model\":\"test\"}");

    // Load into new model
    SimpleModel model2;
    SGD sgd2(model2.parameters(), opts);

    Checkpoint ckpt = load_checkpoint(path);

    // Verify metadata
    if (ckpt.epoch != 10) {
        std::cerr << "save_load: epoch should be 10" << std::endl;
        return false;
    }
    if (ckpt.step != 1000) {
        std::cerr << "save_load: step should be 1000" << std::endl;
        return false;
    }
    if (!float_eq(ckpt.loss, 0.5f)) {
        std::cerr << "save_load: loss should be 0.5" << std::endl;
        return false;
    }
    if (ckpt.config_json != "{\"model\":\"test\"}") {
        std::cerr << "save_load: config mismatch" << std::endl;
        return false;
    }

    // Restore and verify model state
    restore_checkpoint(ckpt, model2, sgd2);

    if (!float_eq(model2.fc1.weight.data().data()[0], 1.0f)) {
        std::cerr << "save_load: fc1.weight should be 1.0" << std::endl;
        return false;
    }
    if (!float_eq(model2.fc2.weight.data().data()[0], 2.0f)) {
        std::cerr << "save_load: fc2.weight should be 2.0" << std::endl;
        return false;
    }

    // Cleanup
    std::remove(path.c_str());

    std::cout << "test_phase_26_save_load: PASSED" << std::endl;
    return true;
}

// Test 2: Optimizer state preservation
bool test_phase_26_optimizer_state() {
    const std::string path = "/tmp/test_ckpt_2.ckpt";

    SimpleModel model1;
    SGDOptions opts;
    opts.lr = 0.1f;
    opts.momentum = 0.9f;
    SGD sgd1(model1.parameters(), opts);

    // Run a few steps to build up momentum
    for (int i = 0; i < 5; ++i) {
        Tensor<float> input({1, 4});
        for (size_t j = 0; j < 4; ++j) {
            input.data()[j] = 1.0f;
        }
        Variable x(input, false);
        auto y = model1.forward(x);

        // Manual gradient
        sgd1.zero_grad();
        Tensor<float> grad({1, 2});
        grad.data()[0] = 1.0f;
        grad.data()[1] = 1.0f;
        y.accumulate_grad(grad);
        sgd1.step();
    }

    // Save state
    save_checkpoint(path, model1, sgd1, 5, 500, 0.3f);

    // Get optimizer state before save
    auto state1 = sgd1.state_dict();

    // Load into new model
    SimpleModel model2;
    SGD sgd2(model2.parameters(), opts);
    Checkpoint ckpt = load_checkpoint(path);
    restore_checkpoint(ckpt, model2, sgd2);

    // Verify optimizer state was restored
    auto state2 = sgd2.state_dict();

    if (state1.size() != state2.size()) {
        std::cerr << "optimizer_state: state dict size mismatch" << std::endl;
        return false;
    }

    // Cleanup
    std::remove(path.c_str());

    std::cout << "test_phase_26_optimizer_state: PASSED" << std::endl;
    return true;
}

// Test 3: Metadata preservation
bool test_phase_26_metadata() {
    const std::string path = "/tmp/test_ckpt_3.ckpt";

    SimpleModel model;
    SGDOptions opts;
    SGD sgd(model.parameters(), opts);

    int test_epoch = 42;
    int test_step = 12345;
    float test_loss = 0.123456f;
    std::string test_config = "{\"learning_rate\": 0.001, \"batch_size\": 32}";

    save_checkpoint(path, model, sgd, test_epoch, test_step, test_loss, test_config);

    Checkpoint ckpt = load_checkpoint(path);

    if (ckpt.epoch != test_epoch) {
        std::cerr << "metadata: epoch mismatch" << std::endl;
        return false;
    }
    if (ckpt.step != test_step) {
        std::cerr << "metadata: step mismatch" << std::endl;
        return false;
    }
    if (!float_eq(ckpt.loss, test_loss, 1e-6f)) {
        std::cerr << "metadata: loss mismatch" << std::endl;
        return false;
    }
    if (ckpt.config_json != test_config) {
        std::cerr << "metadata: config mismatch" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_metadata: PASSED" << std::endl;
    return true;
}

// Test 4: Atomic write (temp file cleanup)
bool test_phase_26_atomic() {
    const std::string path = "/tmp/test_ckpt_4.ckpt";
    const std::string temp_path = path + ".tmp";

    SimpleModel model;
    SGDOptions opts;
    SGD sgd(model.parameters(), opts);

    // Save checkpoint
    save_checkpoint(path, model, sgd, 1, 1, 0.0f);

    // Verify temp file doesn't exist
    std::ifstream temp_check(temp_path);
    if (temp_check.good()) {
        std::cerr << "atomic: temp file should not exist after save" << std::endl;
        return false;
    }

    // Verify main file exists
    std::ifstream main_check(path);
    if (!main_check.good()) {
        std::cerr << "atomic: main checkpoint file should exist" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_atomic: PASSED" << std::endl;
    return true;
}

// Test 5: Missing key handling (partial state dict)
bool test_phase_26_missing_key() {
    const std::string path = "/tmp/test_ckpt_5.ckpt";

    SimpleModel model1;
    model1.fc1.weight.data().fill_(5.0f);
    model1.fc2.weight.data().fill_(10.0f);

    SGDOptions opts;
    SGD sgd1(model1.parameters(), opts);

    save_checkpoint(path, model1, sgd1, 1, 1, 0.0f);

    // Load into model and manually remove a key
    Checkpoint ckpt = load_checkpoint(path);

    // Create a new model with different structure
    SimpleModel model2;
    float original_fc2 = model2.fc2.weight.data().data()[0];

    // Manually create partial state dict (only fc1)
    std::unordered_map<std::string, Tensor<float>> partial;
    partial["fc1.weight"] = ckpt.model_state["fc1.weight"];
    partial["fc1.bias"] = ckpt.model_state["fc1.bias"];

    model2.load_state_dict(partial);

    // fc1 should be updated
    if (!float_eq(model2.fc1.weight.data().data()[0], 5.0f)) {
        std::cerr << "missing_key: fc1.weight should be 5.0" << std::endl;
        return false;
    }

    // fc2 should be unchanged (key was missing)
    if (float_eq(model2.fc2.weight.data().data()[0], 10.0f)) {
        std::cerr << "missing_key: fc2.weight should remain original" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_missing_key: PASSED" << std::endl;
    return true;
}

// Test 6: Checkpoint file existence check
bool test_phase_26_exists() {
    const std::string path = "/tmp/test_ckpt_6.ckpt";

    // Should not exist
    if (checkpoint_exists(path)) {
        std::cerr << "exists: file should not exist initially" << std::endl;
        return false;
    }

    // Create checkpoint
    SimpleModel model;
    SGDOptions opts;
    SGD sgd(model.parameters(), opts);
    save_checkpoint(path, model, sgd, 1, 1, 0.0f);

    // Should exist now
    if (!checkpoint_exists(path)) {
        std::cerr << "exists: file should exist after save" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_exists: PASSED" << std::endl;
    return true;
}

// Test 7: Invalid file handling
bool test_phase_26_invalid_file() {
    const std::string path = "/tmp/test_ckpt_7.ckpt";

    // Write garbage to file
    std::ofstream out(path, std::ios::binary);
    out.write("garbage", 7);
    out.close();

    bool caught = false;
    try {
        load_checkpoint(path);
    } catch (const std::runtime_error& e) {
        caught = true;
    }

    if (!caught) {
        std::cerr << "invalid_file: should throw on invalid file" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_invalid_file: PASSED" << std::endl;
    return true;
}

// Test 8: Empty config string
bool test_phase_26_empty_config() {
    const std::string path = "/tmp/test_ckpt_8.ckpt";

    SimpleModel model;
    SGDOptions opts;
    SGD sgd(model.parameters(), opts);

    // Save with empty config
    save_checkpoint(path, model, sgd, 1, 1, 0.0f, "");

    Checkpoint ckpt = load_checkpoint(path);

    if (!ckpt.config_json.empty()) {
        std::cerr << "empty_config: config should be empty" << std::endl;
        return false;
    }

    std::remove(path.c_str());

    std::cout << "test_phase_26_empty_config: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 26: Checkpointing Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_26_save_load()) ++failures;
    if (!test_phase_26_optimizer_state()) ++failures;
    if (!test_phase_26_metadata()) ++failures;
    if (!test_phase_26_atomic()) ++failures;
    if (!test_phase_26_missing_key()) ++failures;
    if (!test_phase_26_exists()) ++failures;
    if (!test_phase_26_invalid_file()) ++failures;
    if (!test_phase_26_empty_config()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 26 tests passed (8/8) ===" << std::endl;
    return 0;
}
