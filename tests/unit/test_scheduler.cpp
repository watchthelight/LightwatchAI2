// Phase 24: Learning Rate Scheduler Tests

#include <lightwatch/optim/scheduler.hpp>
#include <lightwatch/optim/sgd.hpp>
#include <iostream>
#include <cmath>

using namespace lightwatch;
using namespace lightwatch::optim;
using namespace lightwatch::autograd;

bool float_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) < eps;
}

// Test 1: Cosine annealing at half cycle
bool test_phase_24_cosine() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;  // Base LR = 1.0
    SGD sgd({&param}, opts);

    CosineAnnealingLR scheduler(sgd, 100, 0.0f);  // T_max=100, eta_min=0

    // Run 50 steps (half cycle)
    for (int i = 0; i < 50; ++i) {
        scheduler.step();
    }

    // At t=50, T_max=100:
    // lr = 0 + (1 - 0) * (1 + cos(π * 50 / 100)) / 2
    // lr = (1 + cos(π/2)) / 2 = (1 + 0) / 2 = 0.5
    float expected_lr = 0.5f;
    float actual_lr = scheduler.get_last_lr();

    if (!float_eq(actual_lr, expected_lr, 0.01f)) {
        std::cerr << "cosine: at half cycle, LR should be ~0.5, got " << actual_lr << std::endl;
        return false;
    }

    // Also verify optimizer LR was updated
    if (!float_eq(sgd.get_lr(), actual_lr)) {
        std::cerr << "cosine: optimizer LR should match scheduler" << std::endl;
        return false;
    }

    std::cout << "test_phase_24_cosine: PASSED" << std::endl;
    return true;
}

// Test 2: Cosine annealing at end of cycle
bool test_phase_24_cosine_end() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    float eta_min = 0.001f;
    CosineAnnealingLR scheduler(sgd, 100, eta_min);

    // Run full 100 steps
    for (int i = 0; i < 100; ++i) {
        scheduler.step();
    }

    // At t=100, T_max=100:
    // lr = eta_min + (1 - eta_min) * (1 + cos(π)) / 2
    // lr = eta_min + (1 - eta_min) * (1 - 1) / 2 = eta_min
    float actual_lr = scheduler.get_last_lr();

    if (!float_eq(actual_lr, eta_min, 0.001f)) {
        std::cerr << "cosine_end: at end of cycle, LR should be eta_min=" << eta_min
                  << ", got " << actual_lr << std::endl;
        return false;
    }

    std::cout << "test_phase_24_cosine_end: PASSED" << std::endl;
    return true;
}

// Test 3: Warmup linear increase
bool test_phase_24_warmup() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    WarmupLR scheduler(sgd, 10, 0.0f);  // 10 warmup steps, start at 0

    // Initial LR should be 0
    if (!float_eq(scheduler.get_last_lr(), 0.0f)) {
        std::cerr << "warmup: initial LR should be 0" << std::endl;
        return false;
    }

    // Track LRs through warmup
    std::vector<float> lrs;
    for (int i = 0; i < 10; ++i) {
        scheduler.step();
        lrs.push_back(scheduler.get_last_lr());
    }

    // LRs should increase linearly: 0.1, 0.2, ..., 1.0
    for (int i = 0; i < 10; ++i) {
        float expected = static_cast<float>(i + 1) / 10.0f;
        if (!float_eq(lrs[i], expected, 0.01f)) {
            std::cerr << "warmup: step " << (i + 1) << " LR should be " << expected
                      << ", got " << lrs[i] << std::endl;
            return false;
        }
    }

    // After warmup, LR should stay at base
    scheduler.step();
    if (!float_eq(scheduler.get_last_lr(), 1.0f)) {
        std::cerr << "warmup: after warmup, LR should be 1.0" << std::endl;
        return false;
    }

    std::cout << "test_phase_24_warmup: PASSED" << std::endl;
    return true;
}

// Test 4: Step decay
bool test_phase_24_step() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    StepLR scheduler(sgd, 10, 0.1f);  // step_size=10, gamma=0.1

    // Steps 1-9: epoch=0, lr = 1.0 * 0.1^0 = 1.0
    for (int i = 0; i < 9; ++i) {
        scheduler.step();
        if (!float_eq(scheduler.get_last_lr(), 1.0f)) {
            std::cerr << "step: steps 1-9 should have LR 1.0" << std::endl;
            return false;
        }
    }

    // Steps 10-19: epoch=1, lr = 1.0 * 0.1^1 = 0.1
    scheduler.step();  // Step 10
    if (!float_eq(scheduler.get_last_lr(), 0.1f)) {
        std::cerr << "step: step 10 should drop LR to 0.1, got "
                  << scheduler.get_last_lr() << std::endl;
        return false;
    }

    for (int i = 0; i < 9; ++i) {
        scheduler.step();  // Steps 11-19
        if (!float_eq(scheduler.get_last_lr(), 0.1f)) {
            std::cerr << "step: steps 11-19 should have LR 0.1" << std::endl;
            return false;
        }
    }

    // Step 20: epoch=2, lr = 1.0 * 0.1^2 = 0.01
    scheduler.step();
    if (!float_eq(scheduler.get_last_lr(), 0.01f)) {
        std::cerr << "step: step 20 should drop LR to 0.01, got "
                  << scheduler.get_last_lr() << std::endl;
        return false;
    }

    std::cout << "test_phase_24_step: PASSED" << std::endl;
    return true;
}

// Test 5: Chained scheduler (warmup then cosine)
bool test_phase_24_chained() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    // 10 warmup steps, 110 total steps, start at 0, end at 0
    ChainedScheduler scheduler(sgd, 10, 110, 0.0f, 0.0f);

    // Initial LR should be 0
    if (!float_eq(scheduler.get_last_lr(), 0.0f)) {
        std::cerr << "chained: initial LR should be 0" << std::endl;
        return false;
    }

    // Warmup phase: steps 1-10
    for (int i = 0; i < 10; ++i) {
        scheduler.step();
    }

    // After warmup, should be at base LR
    if (!float_eq(scheduler.get_last_lr(), 1.0f, 0.01f)) {
        std::cerr << "chained: after warmup, LR should be ~1.0, got "
                  << scheduler.get_last_lr() << std::endl;
        return false;
    }

    // Cosine phase: 100 more steps (steps 11-110)
    // At step 60 (50 steps into cosine), should be at ~0.5
    for (int i = 0; i < 50; ++i) {
        scheduler.step();
    }

    float mid_lr = scheduler.get_last_lr();
    if (!float_eq(mid_lr, 0.5f, 0.05f)) {
        std::cerr << "chained: at cosine midpoint, LR should be ~0.5, got "
                  << mid_lr << std::endl;
        return false;
    }

    // Complete the cycle
    for (int i = 0; i < 50; ++i) {
        scheduler.step();
    }

    float end_lr = scheduler.get_last_lr();
    if (!float_eq(end_lr, 0.0f, 0.01f)) {
        std::cerr << "chained: at end, LR should be ~0, got " << end_lr << std::endl;
        return false;
    }

    std::cout << "test_phase_24_chained: PASSED" << std::endl;
    return true;
}

// Test 6: Warmup with non-zero start factor
bool test_phase_24_warmup_start_factor() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    // Start at 0.1 * base_lr = 0.1
    WarmupLR scheduler(sgd, 10, 0.1f);

    // Initial LR should be 0.1
    if (!float_eq(scheduler.get_last_lr(), 0.1f)) {
        std::cerr << "warmup_start_factor: initial LR should be 0.1" << std::endl;
        return false;
    }

    // After 5 steps: factor = 0.1 + 0.9 * 0.5 = 0.55
    for (int i = 0; i < 5; ++i) {
        scheduler.step();
    }

    if (!float_eq(scheduler.get_last_lr(), 0.55f, 0.01f)) {
        std::cerr << "warmup_start_factor: at step 5, LR should be ~0.55, got "
                  << scheduler.get_last_lr() << std::endl;
        return false;
    }

    std::cout << "test_phase_24_warmup_start_factor: PASSED" << std::endl;
    return true;
}

// Test 7: Cosine continues past T_max (clamped)
bool test_phase_24_cosine_clamp() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    float eta_min = 0.001f;
    CosineAnnealingLR scheduler(sgd, 100, eta_min);

    // Run 150 steps (past T_max)
    for (int i = 0; i < 150; ++i) {
        scheduler.step();
    }

    // Should stay at eta_min
    if (!float_eq(scheduler.get_last_lr(), eta_min, 0.001f)) {
        std::cerr << "cosine_clamp: past T_max, LR should stay at eta_min" << std::endl;
        return false;
    }

    std::cout << "test_phase_24_cosine_clamp: PASSED" << std::endl;
    return true;
}

// Test 8: Multiple scheduler step count tracking
bool test_phase_24_step_count() {
    Tensor<float> param_data({4});
    for (size_t i = 0; i < 4; ++i) {
        param_data.data()[i] = 1.0f;
    }
    Variable param(param_data, true);

    SGDOptions opts;
    opts.lr = 1.0f;
    SGD sgd({&param}, opts);

    CosineAnnealingLR scheduler(sgd, 100);

    for (int i = 0; i < 25; ++i) {
        scheduler.step();
    }

    if (scheduler.get_step_count() != 25) {
        std::cerr << "step_count: should be 25, got " << scheduler.get_step_count() << std::endl;
        return false;
    }

    std::cout << "test_phase_24_step_count: PASSED" << std::endl;
    return true;
}

int main() {
    std::cout << "=== Phase 24: LR Scheduler Tests ===" << std::endl;

    int failures = 0;

    if (!test_phase_24_cosine()) ++failures;
    if (!test_phase_24_cosine_end()) ++failures;
    if (!test_phase_24_warmup()) ++failures;
    if (!test_phase_24_step()) ++failures;
    if (!test_phase_24_chained()) ++failures;
    if (!test_phase_24_warmup_start_factor()) ++failures;
    if (!test_phase_24_cosine_clamp()) ++failures;
    if (!test_phase_24_step_count()) ++failures;

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }

    std::cout << "=== All Phase 24 tests passed (8/8) ===" << std::endl;
    return 0;
}
