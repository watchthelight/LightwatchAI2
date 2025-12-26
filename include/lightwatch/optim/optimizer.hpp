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

    // State dict for serialization (optimizer state)
    // Returns a flattened map: "param_idx.key" -> tensor
    std::unordered_map<std::string, Tensor<float>> state_dict() const {
        std::unordered_map<std::string, Tensor<float>> dict;
        int param_idx = 0;
        for (const auto& group : param_groups_) {
            for (const auto* param : group.params) {
                auto it = state_.find(const_cast<autograd::Variable*>(param));
                if (it != state_.end()) {
                    for (const auto& kv : it->second) {
                        std::string key = std::to_string(param_idx) + "." + kv.first;
                        dict[key] = kv.second;
                    }
                }
                ++param_idx;
            }
        }
        return dict;
    }

    // Load state dict
    void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict) {
        int param_idx = 0;
        for (auto& group : param_groups_) {
            for (auto* param : group.params) {
                std::string prefix = std::to_string(param_idx) + ".";
                for (const auto& kv : dict) {
                    if (kv.first.substr(0, prefix.size()) == prefix) {
                        std::string key = kv.first.substr(prefix.size());
                        state_[param][key] = kv.second;
                    }
                }
                ++param_idx;
            }
        }
    }

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
