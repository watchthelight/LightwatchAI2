// Phase 23: Adam and AdamW Optimizers

#pragma once

#include <lightwatch/optim/optimizer.hpp>
#include <cmath>

namespace lightwatch {
namespace optim {

struct AdamOptions : OptimizerOptions {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    bool amsgrad = false;

    AdamOptions& with_lr(float lr_val) {
        lr = lr_val;
        return *this;
    }

    AdamOptions& with_betas(float b1, float b2) {
        beta1 = b1;
        beta2 = b2;
        return *this;
    }

    AdamOptions& with_weight_decay(float wd) {
        weight_decay = wd;
        return *this;
    }

    AdamOptions& with_eps(float e) {
        eps = e;
        return *this;
    }

    AdamOptions& with_amsgrad(bool a) {
        amsgrad = a;
        return *this;
    }
};

class Adam : public Optimizer {
public:
    Adam(std::vector<autograd::Variable*> params, AdamOptions options = {})
        : Optimizer(std::move(params),
                    OptimizerOptions{options.lr, options.weight_decay})
        , options_(options)
        , step_count_(0) {}

    void step() override {
        ++step_count_;

        for (auto& group : param_groups_) {
            float lr = group.lr;
            float wd = group.weight_decay;
            float beta1 = options_.beta1;
            float beta2 = options_.beta2;
            float eps = options_.eps;
            bool amsgrad = options_.amsgrad;

            // Bias correction terms
            float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_count_));
            float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_count_));

            for (auto* param : group.params) {
                if (!param->has_grad()) {
                    continue;
                }

                Tensor<float>& data = param->data();
                const Tensor<float>& grad = param->grad();

                // Get or create state tensors
                auto& param_state = state_[param];
                if (param_state.find("exp_avg") == param_state.end()) {
                    param_state["exp_avg"] = Tensor<float>(grad.shape());
                    param_state["exp_avg"].zero_();
                    param_state["exp_avg_sq"] = Tensor<float>(grad.shape());
                    param_state["exp_avg_sq"].zero_();
                    if (amsgrad) {
                        param_state["max_exp_avg_sq"] = Tensor<float>(grad.shape());
                        param_state["max_exp_avg_sq"].zero_();
                    }
                }

                Tensor<float>& exp_avg = param_state["exp_avg"];
                Tensor<float>& exp_avg_sq = param_state["exp_avg_sq"];

                // Apply weight decay (L2 regularization for Adam)
                // Note: For AdamW, weight decay is decoupled
                if (wd != 0.0f && !is_adamw_) {
                    for (size_t i = 0; i < data.numel(); ++i) {
                        data.data()[i] -= lr * wd * data.data()[i];
                    }
                }

                // Update biased first moment estimate
                // m = β1 * m + (1 - β1) * g
                for (size_t i = 0; i < exp_avg.numel(); ++i) {
                    exp_avg.data()[i] = beta1 * exp_avg.data()[i] +
                                        (1.0f - beta1) * grad.data()[i];
                }

                // Update biased second raw moment estimate
                // v = β2 * v + (1 - β2) * g²
                for (size_t i = 0; i < exp_avg_sq.numel(); ++i) {
                    exp_avg_sq.data()[i] = beta2 * exp_avg_sq.data()[i] +
                                           (1.0f - beta2) * grad.data()[i] * grad.data()[i];
                }

                // Compute denominator
                if (amsgrad) {
                    Tensor<float>& max_exp_avg_sq = param_state["max_exp_avg_sq"];
                    for (size_t i = 0; i < data.numel(); ++i) {
                        // v_max = max(v_max, v)
                        max_exp_avg_sq.data()[i] = std::max(max_exp_avg_sq.data()[i],
                                                             exp_avg_sq.data()[i]);
                        // v_hat = v_max / (1 - β2^t)
                        float v_hat = max_exp_avg_sq.data()[i] / bias_correction2;
                        // m_hat = m / (1 - β1^t)
                        float m_hat = exp_avg.data()[i] / bias_correction1;
                        // param -= lr * m_hat / (sqrt(v_hat) + eps)
                        data.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                    }
                } else {
                    for (size_t i = 0; i < data.numel(); ++i) {
                        // m_hat = m / (1 - β1^t)
                        float m_hat = exp_avg.data()[i] / bias_correction1;
                        // v_hat = v / (1 - β2^t)
                        float v_hat = exp_avg_sq.data()[i] / bias_correction2;
                        // param -= lr * m_hat / (sqrt(v_hat) + eps)
                        data.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                    }
                }
            }
        }
    }

    int step_count() const { return step_count_; }

    // Access options
    const AdamOptions& options() const { return options_; }

protected:
    AdamOptions options_;
    int step_count_;
    bool is_adamw_ = false;
};

// AdamW: Adam with decoupled weight decay
class AdamW : public Adam {
public:
    AdamW(std::vector<autograd::Variable*> params, AdamOptions options = {})
        : Adam(std::move(params), options) {
        is_adamw_ = true;
    }

    void step() override {
        ++step_count_;

        for (auto& group : param_groups_) {
            float lr = group.lr;
            float wd = group.weight_decay;
            float beta1 = options_.beta1;
            float beta2 = options_.beta2;
            float eps = options_.eps;
            bool amsgrad = options_.amsgrad;

            // Bias correction terms
            float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_count_));
            float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_count_));

            for (auto* param : group.params) {
                if (!param->has_grad()) {
                    continue;
                }

                Tensor<float>& data = param->data();
                const Tensor<float>& grad = param->grad();

                // Decoupled weight decay (AdamW style)
                // Applied BEFORE the Adam update
                if (wd != 0.0f) {
                    for (size_t i = 0; i < data.numel(); ++i) {
                        data.data()[i] -= lr * wd * data.data()[i];
                    }
                }

                // Get or create state tensors
                auto& param_state = state_[param];
                if (param_state.find("exp_avg") == param_state.end()) {
                    param_state["exp_avg"] = Tensor<float>(grad.shape());
                    param_state["exp_avg"].zero_();
                    param_state["exp_avg_sq"] = Tensor<float>(grad.shape());
                    param_state["exp_avg_sq"].zero_();
                    if (amsgrad) {
                        param_state["max_exp_avg_sq"] = Tensor<float>(grad.shape());
                        param_state["max_exp_avg_sq"].zero_();
                    }
                }

                Tensor<float>& exp_avg = param_state["exp_avg"];
                Tensor<float>& exp_avg_sq = param_state["exp_avg_sq"];

                // Update biased first moment estimate
                for (size_t i = 0; i < exp_avg.numel(); ++i) {
                    exp_avg.data()[i] = beta1 * exp_avg.data()[i] +
                                        (1.0f - beta1) * grad.data()[i];
                }

                // Update biased second raw moment estimate
                for (size_t i = 0; i < exp_avg_sq.numel(); ++i) {
                    exp_avg_sq.data()[i] = beta2 * exp_avg_sq.data()[i] +
                                           (1.0f - beta2) * grad.data()[i] * grad.data()[i];
                }

                // Apply update
                if (amsgrad) {
                    Tensor<float>& max_exp_avg_sq = param_state["max_exp_avg_sq"];
                    for (size_t i = 0; i < data.numel(); ++i) {
                        max_exp_avg_sq.data()[i] = std::max(max_exp_avg_sq.data()[i],
                                                             exp_avg_sq.data()[i]);
                        float v_hat = max_exp_avg_sq.data()[i] / bias_correction2;
                        float m_hat = exp_avg.data()[i] / bias_correction1;
                        data.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                    }
                } else {
                    for (size_t i = 0; i < data.numel(); ++i) {
                        float m_hat = exp_avg.data()[i] / bias_correction1;
                        float v_hat = exp_avg_sq.data()[i] / bias_correction2;
                        data.data()[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
                    }
                }
            }
        }
    }
};

}  // namespace optim
}  // namespace lightwatch
