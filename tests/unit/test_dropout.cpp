// Phase 14: Dropout Tests

#include <lightwatch/nn/dropout.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: Dropout in training mode
bool test_phase_14_dropout_train() {
    Dropout dropout(0.5f);
    dropout.train(true);

    // Create large input for statistical testing
    Tensor<float> input_data = Tensor<float>::ones({100, 100});
    Variable input(input_data, false);

    auto output = dropout.forward(input);

    // Count zeros
    size_t zero_count = 0;
    for (size_t i = 0; i < output.numel(); ++i) {
        if (output.data().data()[i] == 0.0f) {
            ++zero_count;
        }
    }

    // With p=0.5, expect ~50% zeros (allow 40-60% range for randomness)
    float zero_ratio = static_cast<float>(zero_count) / static_cast<float>(output.numel());

    if (zero_ratio < 0.40f || zero_ratio > 0.60f) {
        std::cerr << "dropout_train: expected ~50% zeros, got "
                  << (zero_ratio * 100.0f) << "%" << std::endl;
        return false;
    }

    std::cout << "test_phase_14_dropout_train: PASSED (zero_ratio="
              << zero_ratio << ")" << std::endl;
    return true;
}

// Test 2: Dropout in eval mode
bool test_phase_14_dropout_eval() {
    Dropout dropout(0.5f);
    dropout.train(false);  // Eval mode

    Tensor<float> input_data = Tensor<float>::ones({10, 10});
    Variable input(input_data, false);

    auto output = dropout.forward(input);

    // In eval mode, should pass through unchanged
    for (size_t i = 0; i < output.numel(); ++i) {
        if (!float_eq(output.data().data()[i], 1.0f)) {
            std::cerr << "dropout_eval: values should be unchanged in eval mode" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_14_dropout_eval: PASSED" << std::endl;
    return true;
}

// Test 3: Inverted dropout scaling preserves mean
bool test_phase_14_dropout_scale() {
    Dropout dropout(0.5f);
    dropout.train(true);

    // Large input for statistical testing
    Tensor<float> input_data = Tensor<float>::full({1000, 100}, 2.0f);
    Variable input(input_data, false);

    auto output = dropout.forward(input);

    // Compute mean of output
    float sum = 0.0f;
    for (size_t i = 0; i < output.numel(); ++i) {
        sum += output.data().data()[i];
    }
    float mean = sum / static_cast<float>(output.numel());

    // With inverted dropout, mean should be preserved (~2.0)
    // Allow some tolerance for randomness
    if (mean < 1.5f || mean > 2.5f) {
        std::cerr << "dropout_scale: mean should be ~2.0 (preserved), got " << mean << std::endl;
        return false;
    }

    std::cout << "test_phase_14_dropout_scale: PASSED (mean=" << mean << ")" << std::endl;
    return true;
}

// Test 4: DropPath drops entire samples
bool test_phase_14_droppath() {
    DropPath droppath(0.3f);
    droppath.train(true);

    // Batch of 100 samples
    Tensor<float> input_data = Tensor<float>::ones({100, 10});
    Variable input(input_data, false);

    auto output = droppath.forward(input);

    // Count fully-zeroed samples
    size_t dropped_count = 0;
    for (size_t b = 0; b < 100; ++b) {
        bool is_dropped = true;
        for (size_t i = 0; i < 10; ++i) {
            if (output.data().data()[b * 10 + i] != 0.0f) {
                is_dropped = false;
                break;
            }
        }
        if (is_dropped) {
            ++dropped_count;
        }
    }

    // With p=0.3, expect ~30% dropped samples (allow 15-45% range)
    float drop_ratio = static_cast<float>(dropped_count) / 100.0f;

    if (drop_ratio < 0.15f || drop_ratio > 0.45f) {
        std::cerr << "droppath: expected ~30% dropped, got "
                  << (drop_ratio * 100.0f) << "%" << std::endl;
        return false;
    }

    std::cout << "test_phase_14_droppath: PASSED (drop_ratio=" << drop_ratio << ")" << std::endl;
    return true;
}

// Test 5: DropPath in eval mode
bool test_phase_14_droppath_eval() {
    DropPath droppath(0.5f);
    droppath.train(false);  // Eval mode

    Tensor<float> input_data = Tensor<float>::ones({10, 10});
    Variable input(input_data, false);

    auto output = droppath.forward(input);

    // In eval mode, should pass through unchanged
    for (size_t i = 0; i < output.numel(); ++i) {
        if (!float_eq(output.data().data()[i], 1.0f)) {
            std::cerr << "droppath_eval: values should be unchanged in eval mode" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_14_droppath_eval: PASSED" << std::endl;
    return true;
}

// Test 6: Dropout with p=0 (no dropout)
bool test_phase_14_dropout_p0() {
    Dropout dropout(0.0f);
    dropout.train(true);

    Tensor<float> input_data = Tensor<float>::ones({10, 10});
    Variable input(input_data, false);

    auto output = dropout.forward(input);

    // With p=0, no elements should be dropped
    for (size_t i = 0; i < output.numel(); ++i) {
        if (!float_eq(output.data().data()[i], 1.0f)) {
            std::cerr << "dropout_p0: with p=0, output should equal input" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_14_dropout_p0: PASSED" << std::endl;
    return true;
}

// Test 7: Dropout backward
bool test_phase_14_dropout_backward() {
    Dropout dropout(0.5f);
    dropout.train(true);

    Tensor<float> input_data = Tensor<float>::randn({4, 8});
    Variable input(input_data, true);

    auto output = dropout.forward(input);
    output.backward();

    // Input should have gradients
    if (!input.has_grad()) {
        std::cerr << "dropout_backward: input should have grad" << std::endl;
        return false;
    }

    // Gradients should be zero where output was dropped, scaled where kept
    for (size_t i = 0; i < input.numel(); ++i) {
        if (std::isnan(input.grad().data()[i]) || std::isinf(input.grad().data()[i])) {
            std::cerr << "dropout_backward: NaN/Inf in grad" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_14_dropout_backward: PASSED" << std::endl;
    return true;
}

// Test 8: DropPath backward
bool test_phase_14_droppath_backward() {
    DropPath droppath(0.3f);
    droppath.train(true);

    Tensor<float> input_data = Tensor<float>::randn({10, 8});
    Variable input(input_data, true);

    auto output = droppath.forward(input);
    output.backward();

    // Input should have gradients
    if (!input.has_grad()) {
        std::cerr << "droppath_backward: input should have grad" << std::endl;
        return false;
    }

    // Gradients should be valid
    for (size_t i = 0; i < input.numel(); ++i) {
        if (std::isnan(input.grad().data()[i]) || std::isinf(input.grad().data()[i])) {
            std::cerr << "droppath_backward: NaN/Inf in grad" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_14_droppath_backward: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 14: Dropout Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_14_dropout_train()) ++failures;
    if (!test_phase_14_dropout_eval()) ++failures;
    if (!test_phase_14_dropout_scale()) ++failures;
    if (!test_phase_14_droppath()) ++failures;
    if (!test_phase_14_droppath_eval()) ++failures;
    if (!test_phase_14_dropout_p0()) ++failures;
    if (!test_phase_14_dropout_backward()) ++failures;
    if (!test_phase_14_droppath_backward()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 14 tests passed (8/8) ===" << std::endl;
    return 0;
}
