// Phase 22: Optimizer Base Class

#pragma once

#include <lightwatch/autograd.hpp>
#include <vector>
#include <unordered_map>
#include <string>

namespace lightwatch {
namespace optim {

struct OptimizerOptions {
    float lr = 1e-3f;
    float weight_decay = 0.0f;
};

struct ParamGroup {
    std::vector<autograd::Variable*> params;
    float lr;
    float weight_decay;
};

// Base class for all optimizers
class Optimizer {
public:
    explicit Optimizer(std::vector<autograd::Variable*> params,
                       OptimizerOptions options = {})
        : lr_(options.lr)
        , weight_decay_(options.weight_decay) {
        ParamGroup group;
        group.params = std::move(params);
        group.lr = options.lr;
        group.weight_decay = options.weight_decay;
        param_groups_.push_back(std::move(group));
    }

    virtual ~Optimizer() = default;

    // Perform single optimization step
    virtual void step() = 0;

    // Zero all parameter gradients
    virtual void zero_grad() {
        for (auto& group : param_groups_) {
            for (auto* param : group.params) {
                param->zero_grad();
            }
        }
    }

    // Learning rate accessors
    float get_lr() const { return lr_; }

    void set_lr(float lr) {
        lr_ = lr;
        for (auto& group : param_groups_) {
            group.lr = lr;
        }
    }

    // Access parameter groups
    std::vector<ParamGroup>& param_groups() { return param_groups_; }
    const std::vector<ParamGroup>& param_groups() const { return param_groups_; }

protected:
    float lr_;
    float weight_decay_;
    std::vector<ParamGroup> param_groups_;

    // Per-parameter state (e.g., momentum buffer)
    std::unordered_map<autograd::Variable*,
                       std::unordered_map<std::string, Tensor<float>>> state_;
};

}  // namespace optim
}  // namespace lightwatch
