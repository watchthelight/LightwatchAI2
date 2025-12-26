// Phase 24: Learning Rate Schedulers

#pragma once

#include <lightwatch/optim/optimizer.hpp>
#include <cmath>
#include <algorithm>

namespace lightwatch {
namespace optim {

// Base class for all LR schedulers
class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer)
        : optimizer_(optimizer)
        , step_count_(0)
        , last_lr_(optimizer.get_lr()) {}

    virtual ~LRScheduler() = default;

    // Advance the scheduler by one step and update optimizer LR
    virtual void step() = 0;

    // Get the last computed learning rate
    float get_last_lr() const { return last_lr_; }

    // Get current step count
    int get_step_count() const { return step_count_; }

protected:
    void apply_lr(float lr) {
        last_lr_ = lr;
        optimizer_.set_lr(lr);
    }

    Optimizer& optimizer_;
    int step_count_;
    float last_lr_;
};

// Cosine Annealing Learning Rate Scheduler
// lr = eta_min + (base_lr - eta_min) * (1 + cos(π * t / T_max)) / 2
class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0f)
        : LRScheduler(optimizer)
        , T_max_(T_max)
        , eta_min_(eta_min)
        , base_lr_(optimizer.get_lr()) {}

    void step() override {
        ++step_count_;

        // Clamp step count to T_max for calculation
        int t = std::min(step_count_, T_max_);

        // Cosine annealing formula
        float cos_val = std::cos(M_PI * static_cast<float>(t) / static_cast<float>(T_max_));
        float lr = eta_min_ + (base_lr_ - eta_min_) * (1.0f + cos_val) / 2.0f;

        apply_lr(lr);
    }

    // Accessors
    int T_max() const { return T_max_; }
    float eta_min() const { return eta_min_; }
    float base_lr() const { return base_lr_; }

private:
    int T_max_;
    float eta_min_;
    float base_lr_;
};

// Linear Warmup Learning Rate Scheduler
// lr = start_factor * base_lr + (1 - start_factor) * base_lr * t / warmup_steps
// Simplifies to: lr = base_lr * (start_factor + (1 - start_factor) * t / warmup_steps)
class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer, int warmup_steps, float start_factor = 0.0f)
        : LRScheduler(optimizer)
        , warmup_steps_(warmup_steps)
        , start_factor_(start_factor)
        , base_lr_(optimizer.get_lr()) {
        // Set initial LR to start_factor * base_lr
        apply_lr(start_factor_ * base_lr_);
    }

    void step() override {
        ++step_count_;

        float lr;
        if (step_count_ >= warmup_steps_) {
            // After warmup, use base LR
            lr = base_lr_;
        } else {
            // Linear interpolation: start_factor -> 1.0
            float progress = static_cast<float>(step_count_) / static_cast<float>(warmup_steps_);
            float factor = start_factor_ + (1.0f - start_factor_) * progress;
            lr = base_lr_ * factor;
        }

        apply_lr(lr);
    }

    // Accessors
    int warmup_steps() const { return warmup_steps_; }
    float start_factor() const { return start_factor_; }
    float base_lr() const { return base_lr_; }
    bool is_warmup_complete() const { return step_count_ >= warmup_steps_; }

private:
    int warmup_steps_;
    float start_factor_;
    float base_lr_;
};

// Step Learning Rate Scheduler
// lr = base_lr * gamma^(step_count // step_size)
class StepLR : public LRScheduler {
public:
    StepLR(Optimizer& optimizer, int step_size, float gamma = 0.1f)
        : LRScheduler(optimizer)
        , step_size_(step_size)
        , gamma_(gamma)
        , base_lr_(optimizer.get_lr()) {}

    void step() override {
        ++step_count_;

        // Calculate which "epoch" we're in
        int epoch = step_count_ / step_size_;
        float lr = base_lr_ * std::pow(gamma_, static_cast<float>(epoch));

        apply_lr(lr);
    }

    // Accessors
    int step_size() const { return step_size_; }
    float gamma() const { return gamma_; }
    float base_lr() const { return base_lr_; }

private:
    int step_size_;
    float gamma_;
    float base_lr_;
};

// Chained Scheduler: Combines multiple schedulers in sequence
// Usage: Warmup for N steps, then cosine annealing
class ChainedScheduler : public LRScheduler {
public:
    ChainedScheduler(Optimizer& optimizer,
                     int warmup_steps,
                     int total_steps,
                     float start_factor = 0.0f,
                     float eta_min = 0.0f)
        : LRScheduler(optimizer)
        , warmup_steps_(warmup_steps)
        , total_steps_(total_steps)
        , start_factor_(start_factor)
        , eta_min_(eta_min)
        , base_lr_(optimizer.get_lr()) {
        // Set initial LR
        apply_lr(start_factor_ * base_lr_);
    }

    void step() override {
        ++step_count_;

        float lr;
        if (step_count_ <= warmup_steps_) {
            // Warmup phase: linear increase
            float progress = static_cast<float>(step_count_) / static_cast<float>(warmup_steps_);
            float factor = start_factor_ + (1.0f - start_factor_) * progress;
            lr = base_lr_ * factor;
        } else {
            // Cosine annealing phase
            int cosine_step = step_count_ - warmup_steps_;
            int cosine_total = total_steps_ - warmup_steps_;
            int t = std::min(cosine_step, cosine_total);

            float cos_val = std::cos(M_PI * static_cast<float>(t) / static_cast<float>(cosine_total));
            lr = eta_min_ + (base_lr_ - eta_min_) * (1.0f + cos_val) / 2.0f;
        }

        apply_lr(lr);
    }

    // Accessors
    int warmup_steps() const { return warmup_steps_; }
    int total_steps() const { return total_steps_; }
    float start_factor() const { return start_factor_; }
    float eta_min() const { return eta_min_; }
    float base_lr() const { return base_lr_; }
    bool is_warmup_complete() const { return step_count_ >= warmup_steps_; }

private:
    int warmup_steps_;
    int total_steps_;
    float start_factor_;
    float eta_min_;
    float base_lr_;
};

}  // namespace optim
}  // namespace lightwatch
