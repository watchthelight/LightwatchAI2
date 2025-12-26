// Phase 22: SGD Optimizer

#pragma once

#include <lightwatch/optim/optimizer.hpp>

namespace lightwatch {
namespace optim {

struct SGDOptions : OptimizerOptions {
    float momentum = 0.0f;
    bool nesterov = false;

    SGDOptions& with_lr(float lr_val) {
        lr = lr_val;
        return *this;
    }

    SGDOptions& with_momentum(float m) {
        momentum = m;
        return *this;
    }

    SGDOptions& with_weight_decay(float wd) {
        weight_decay = wd;
        return *this;
    }

    SGDOptions& with_nesterov(bool n) {
        nesterov = n;
        return *this;
    }
};

class SGD : public Optimizer {
public:
    SGD(std::vector<autograd::Variable*> params, SGDOptions options = {})
        : Optimizer(std::move(params),
                    OptimizerOptions{options.lr, options.weight_decay})
        , options_(options) {}

    void step() override {
        for (auto& group : param_groups_) {
            float lr = group.lr;
            float wd = group.weight_decay;
            float momentum = options_.momentum;
            bool nesterov = options_.nesterov;

            for (auto* param : group.params) {
                if (!param->has_grad()) {
                    continue;
                }

                Tensor<float>& data = param->data();
                const Tensor<float>& grad = param->grad();

                // Apply weight decay: grad += wd * param (L2 regularization)
                // Note: we apply to param directly, not grad
                if (wd != 0.0f) {
                    for (size_t i = 0; i < data.numel(); ++i) {
                        data.data()[i] -= lr * wd * data.data()[i];
                    }
                }

                if (momentum != 0.0f) {
                    // Get or create momentum buffer
                    auto& param_state = state_[param];
                    if (param_state.find("momentum_buffer") == param_state.end()) {
                        // Initialize momentum buffer to zeros
                        param_state["momentum_buffer"] = Tensor<float>(grad.shape());
                        param_state["momentum_buffer"].zero_();
                    }

                    Tensor<float>& buf = param_state["momentum_buffer"];

                    // v = momentum * v + grad
                    for (size_t i = 0; i < buf.numel(); ++i) {
                        buf.data()[i] = momentum * buf.data()[i] + grad.data()[i];
                    }

                    if (nesterov) {
                        // param -= lr * (grad + momentum * v)
                        for (size_t i = 0; i < data.numel(); ++i) {
                            data.data()[i] -= lr * (grad.data()[i] + momentum * buf.data()[i]);
                        }
                    } else {
                        // param -= lr * v
                        for (size_t i = 0; i < data.numel(); ++i) {
                            data.data()[i] -= lr * buf.data()[i];
                        }
                    }
                } else {
                    // Basic SGD: param -= lr * grad
                    for (size_t i = 0; i < data.numel(); ++i) {
                        data.data()[i] -= lr * grad.data()[i];
                    }
                }
            }
        }
    }

private:
    SGDOptions options_;
};

}  // namespace optim
}  // namespace lightwatch
