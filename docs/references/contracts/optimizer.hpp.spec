// LightwatchAI2 API Contract: Optimizer
// Defined by: Phase 22
// Consumers: 23-26, 29
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "autograd.hpp"
#include <vector>
#include <unordered_map>
#include <string>

namespace lightwatch::optim {

struct OptimizerOptions {
    float lr = 1e-3;
    float weight_decay = 0.0;
};

class Optimizer {
public:
    explicit Optimizer(std::vector<autograd::Variable*> params, OptimizerOptions options = {});
    virtual ~Optimizer() = default;

    // Perform one optimization step
    virtual void step() = 0;

    // Zero all parameter gradients
    virtual void zero_grad();

    // Parameter group management
    void add_param_group(std::vector<autograd::Variable*> params, OptimizerOptions options = {});

    // Learning rate access
    float get_lr() const;
    void set_lr(float lr);

    // State access (for checkpointing)
    virtual std::unordered_map<std::string, Tensor<float>> state_dict() const;
    virtual void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict);

protected:
    struct ParamGroup {
        std::vector<autograd::Variable*> params;
        OptimizerOptions options;
    };

    std::vector<ParamGroup> param_groups_;

    // Per-parameter state (momentum buffers, Adam moments, etc.)
    std::unordered_map<autograd::Variable*, std::unordered_map<std::string, Tensor<float>>> state_;
};

// SGD with momentum
struct SGDOptions : OptimizerOptions {
    float momentum = 0.0;
    bool nesterov = false;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<autograd::Variable*> params, SGDOptions options = {});
    void step() override;

private:
    SGDOptions options_;
};

// Adam / AdamW
struct AdamOptions : OptimizerOptions {
    float beta1 = 0.9;
    float beta2 = 0.999;
    float eps = 1e-8;
    bool amsgrad = false;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;

private:
    AdamOptions options_;
    int step_count_ = 0;
};

class AdamW : public Adam {
public:
    AdamW(std::vector<autograd::Variable*> params, AdamOptions options = {});
    void step() override;
};

// Learning rate schedulers
class LRScheduler {
public:
    explicit LRScheduler(Optimizer& optimizer);
    virtual ~LRScheduler() = default;

    virtual void step() = 0;
    float get_last_lr() const;

protected:
    Optimizer& optimizer_;
    int step_count_ = 0;
    float last_lr_;
};

class CosineAnnealingLR : public LRScheduler {
public:
    CosineAnnealingLR(Optimizer& optimizer, int T_max, float eta_min = 0.0);
    void step() override;

private:
    int T_max_;
    float eta_min_;
    float base_lr_;
};

class WarmupLR : public LRScheduler {
public:
    WarmupLR(Optimizer& optimizer, int warmup_steps, float start_factor = 0.0);
    void step() override;

private:
    int warmup_steps_;
    float start_factor_;
    float base_lr_;
};

}  // namespace lightwatch::optim
