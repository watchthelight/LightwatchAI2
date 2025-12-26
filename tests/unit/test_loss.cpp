// Phase 21: Loss Functions Tests

#include <lightwatch/nn/loss.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::nn;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-4f) {
    return std::abs(a - b) < eps;
}

// Test 1: CrossEntropyLoss basic functionality
bool test_phase_21_ce_basic() {
    CrossEntropyLoss ce;

    // logits: {1, 2, 4} - batch 1, seq 2, vocab 4
    Tensor<float> logits({1, 2, 4});
    for (size_t i = 0; i < 8; ++i) {
        logits.data()[i] = static_cast<float>(i);  // 0,1,2,3 and 4,5,6,7
    }

    // targets: {1, 2}
    Tensor<TokenId> targets({1, 2});
    targets.data()[0] = 2;  // First token should predict class 2
    targets.data()[1] = 3;  // Second token should predict class 3

    Variable input(logits, false);
    auto loss = ce.forward(input, targets);

    // Loss should be a scalar
    if (loss.shape().size() != 1 || loss.shape()[0] != 1) {
        std::cerr << "ce_basic: loss should be scalar" << std::endl;
        return false;
    }

    // Loss should be positive
    if (loss.data().data()[0] <= 0) {
        std::cerr << "ce_basic: loss should be positive" << std::endl;
        return false;
    }

    std::cout << "test_phase_21_ce_basic: PASSED (loss=" << loss.data().data()[0] << ")" << std::endl;
    return true;
}

// Test 2: Perfect prediction should have near-zero loss
bool test_phase_21_ce_correct_pred() {
    CrossEntropyLoss ce;

    // logits: make target class have very high score
    Tensor<float> logits({1, 1, 4});
    logits.data()[0] = -100.0f;
    logits.data()[1] = -100.0f;
    logits.data()[2] = 100.0f;  // Target class
    logits.data()[3] = -100.0f;

    Tensor<TokenId> targets({1, 1});
    targets.data()[0] = 2;

    Variable input(logits, false);
    auto loss = ce.forward(input, targets);

    // Loss should be very close to 0
    if (loss.data().data()[0] > 0.01f) {
        std::cerr << "ce_correct_pred: loss should be near 0, got "
                  << loss.data().data()[0] << std::endl;
        return false;
    }

    std::cout << "test_phase_21_ce_correct_pred: PASSED" << std::endl;
    return true;
}

// Test 3: Uniform distribution should give log(vocab_size) loss
bool test_phase_21_ce_uniform() {
    CrossEntropyLoss ce;

    size_t vocab_size = 100;
    Tensor<float> logits({1, 1, vocab_size});
    for (size_t i = 0; i < vocab_size; ++i) {
        logits.data()[i] = 0.0f;  // Uniform logits -> uniform probability
    }

    Tensor<TokenId> targets({1, 1});
    targets.data()[0] = 42;  // Any target

    Variable input(logits, false);
    auto loss = ce.forward(input, targets);

    // Expected loss = log(vocab_size)
    float expected_loss = std::log(static_cast<float>(vocab_size));
    float actual_loss = loss.data().data()[0];

    if (!float_eq(actual_loss, expected_loss, 0.01f)) {
        std::cerr << "ce_uniform: expected loss " << expected_loss
                  << ", got " << actual_loss << std::endl;
        return false;
    }

    std::cout << "test_phase_21_ce_uniform: PASSED" << std::endl;
    return true;
}

// Test 4: Ignore index should be skipped
bool test_phase_21_ce_ignore() {
    CrossEntropyLoss ce(-100);  // ignore_index = -100

    Tensor<float> logits({1, 3, 4});
    for (size_t i = 0; i < 12; ++i) {
        logits.data()[i] = 0.0f;
    }

    // First and third tokens are valid, second is ignored
    Tensor<TokenId> targets({1, 3});
    targets.data()[0] = 0;
    targets.data()[1] = -100;  // Ignored
    targets.data()[2] = 1;

    Variable input(logits, false);
    auto loss = ce.forward(input, targets);

    // Loss should be log(4) since uniform distribution
    float expected_loss = std::log(4.0f);
    float actual_loss = loss.data().data()[0];

    if (!float_eq(actual_loss, expected_loss, 0.01f)) {
        std::cerr << "ce_ignore: expected loss " << expected_loss
                  << ", got " << actual_loss << std::endl;
        return false;
    }

    std::cout << "test_phase_21_ce_ignore: PASSED" << std::endl;
    return true;
}

// Test 5: Label smoothing
bool test_phase_21_ce_smoothing() {
    float eps = 0.1f;
    CrossEntropyLoss ce_smooth(eps);
    CrossEntropyLoss ce_no_smooth(0.0f);

    // Perfect prediction
    Tensor<float> logits({1, 1, 4});
    logits.data()[0] = -100.0f;
    logits.data()[1] = -100.0f;
    logits.data()[2] = 100.0f;  // Target class
    logits.data()[3] = -100.0f;

    Tensor<TokenId> targets({1, 1});
    targets.data()[0] = 2;

    Variable input1(logits, false);
    Variable input2(logits, false);
    auto loss_smooth = ce_smooth.forward(input1, targets);
    auto loss_no_smooth = ce_no_smooth.forward(input2, targets);

    // With smoothing, loss should be higher than without
    if (loss_smooth.data().data()[0] <= loss_no_smooth.data().data()[0] + 0.001f) {
        std::cerr << "ce_smoothing: smoothed loss should be higher" << std::endl;
        return false;
    }

    std::cout << "test_phase_21_ce_smoothing: PASSED" << std::endl;
    return true;
}

// Test 6: Backward pass
bool test_phase_21_ce_backward() {
    CrossEntropyLoss ce;

    Tensor<float> logits({2, 4, 8});
    for (size_t i = 0; i < logits.numel(); ++i) {
        logits.data()[i] = static_cast<float>(i % 8) - 3.5f;
    }

    Tensor<TokenId> targets({2, 4});
    for (size_t i = 0; i < 8; ++i) {
        targets.data()[i] = i % 8;
    }

    Variable input(logits, true);
    auto loss = ce.forward(input, targets);

    // Backward
    loss.backward();

    // Input should have gradients
    if (!input.has_grad()) {
        std::cerr << "ce_backward: input should have grad" << std::endl;
        return false;
    }

    // Check gradients are finite
    for (size_t i = 0; i < input.grad().numel(); ++i) {
        if (std::isnan(input.grad().data()[i]) || std::isinf(input.grad().data()[i])) {
            std::cerr << "ce_backward: NaN/Inf in grad" << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_21_ce_backward: PASSED" << std::endl;
    return true;
}

// Test 7: NLLLoss basic
bool test_phase_21_nll_basic() {
    NLLLoss nll;

    // Already log probabilities
    Tensor<float> log_probs({1, 2, 4});
    // Position 0: equal log probs
    log_probs.data()[0] = std::log(0.25f);
    log_probs.data()[1] = std::log(0.25f);
    log_probs.data()[2] = std::log(0.25f);
    log_probs.data()[3] = std::log(0.25f);
    // Position 1: also equal
    log_probs.data()[4] = std::log(0.25f);
    log_probs.data()[5] = std::log(0.25f);
    log_probs.data()[6] = std::log(0.25f);
    log_probs.data()[7] = std::log(0.25f);

    Tensor<TokenId> targets({1, 2});
    targets.data()[0] = 0;
    targets.data()[1] = 2;

    Variable input(log_probs, false);
    auto loss = nll.forward(input, targets);

    // Expected loss = -log(0.25) = log(4) ≈ 1.386
    float expected = -std::log(0.25f);
    float actual = loss.data().data()[0];

    if (!float_eq(actual, expected, 0.01f)) {
        std::cerr << "nll_basic: expected " << expected << ", got " << actual << std::endl;
        return false;
    }

    std::cout << "test_phase_21_nll_basic: PASSED" << std::endl;
    return true;
}

// Test 8: CrossEntropyLoss with different batch sizes
bool test_phase_21_ce_batches() {
    CrossEntropyLoss ce;

    // Test with various batch sizes
    std::vector<size_t> batch_sizes = {1, 2, 4, 8};

    for (size_t batch : batch_sizes) {
        Tensor<float> logits({batch, 4, 10});
        for (size_t i = 0; i < logits.numel(); ++i) {
            logits.data()[i] = static_cast<float>(i % 10) - 4.5f;
        }

        Tensor<TokenId> targets({batch, 4});
        for (size_t i = 0; i < batch * 4; ++i) {
            targets.data()[i] = i % 10;
        }

        Variable input(logits, true);
        auto loss = ce.forward(input, targets);

        if (loss.data().data()[0] <= 0) {
            std::cerr << "ce_batches: loss should be positive for batch " << batch << std::endl;
            return false;
        }
    }

    std::cout << "test_phase_21_ce_batches: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 21: Loss Functions Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_21_ce_basic()) ++failures;
    if (!test_phase_21_ce_correct_pred()) ++failures;
    if (!test_phase_21_ce_uniform()) ++failures;
    if (!test_phase_21_ce_ignore()) ++failures;
    if (!test_phase_21_ce_smoothing()) ++failures;
    if (!test_phase_21_ce_backward()) ++failures;
    if (!test_phase_21_nll_basic()) ++failures;
    if (!test_phase_21_ce_batches()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 21 tests passed (8/8) ===" << std::endl;
    return 0;
}
