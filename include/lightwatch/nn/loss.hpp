// Phase 21: Loss Functions

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/autograd.hpp>
#include <cmath>

namespace lightwatch {
namespace nn {

using TokenId = int32_t;

// Negative Log Likelihood Loss
// Expects log probabilities as input
class NLLLoss : public Module {
public:
    NLLLoss(TokenId ignore_index = -100, bool reduction_mean = true)
        : ignore_index_(ignore_index)
        , reduction_mean_(reduction_mean)
    {}

    autograd::Variable forward(const autograd::Variable& input) override {
        throw std::runtime_error("NLLLoss requires targets; use forward(log_probs, targets)");
    }

    // log_probs: {batch, seq, vocab}, targets: {batch, seq}
    autograd::Variable forward(
        const autograd::Variable& log_probs,
        const Tensor<TokenId>& targets) {

        const auto& shape = log_probs.shape();
        if (shape.size() != 3) {
            throw std::runtime_error("NLLLoss: log_probs must be 3D {batch, seq, vocab}");
        }

        size_t batch_size = shape[0];
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        // Compute loss: -log_probs[target]
        float total_loss = 0.0f;
        size_t count = 0;

        const float* log_prob_data = log_probs.data().data();
        const TokenId* target_data = targets.data();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                TokenId target = target_data[b * seq_len + s];

                if (target == ignore_index_) {
                    continue;  // Skip ignored tokens
                }

                if (target < 0 || static_cast<size_t>(target) >= vocab_size) {
                    throw std::runtime_error("NLLLoss: invalid target index");
                }

                size_t idx = (b * seq_len + s) * vocab_size + target;
                total_loss -= log_prob_data[idx];
                ++count;
            }
        }

        // Apply reduction
        float loss_value = (count > 0 && reduction_mean_) ? (total_loss / count) : total_loss;

        Tensor<float> loss_tensor({1});
        loss_tensor.data()[0] = loss_value;

        autograd::Variable loss(loss_tensor, log_probs.requires_grad());

        if (autograd::is_grad_enabled() && loss.requires_grad()) {
            auto fn = std::make_shared<NLLBackward>();
            fn->batch_size = batch_size;
            fn->seq_len = seq_len;
            fn->vocab_size = vocab_size;
            fn->count = count;
            fn->reduction_mean = reduction_mean_;
            fn->ignore_index = ignore_index_;
            fn->targets = targets.clone();
            fn->inputs.push_back(log_probs.impl());
            loss.set_grad_fn(fn);
        }

        return loss;
    }

private:
    TokenId ignore_index_;
    bool reduction_mean_;

    class NLLBackward : public autograd::Function {
    public:
        size_t batch_size;
        size_t seq_len;
        size_t vocab_size;
        size_t count;
        bool reduction_mean;
        TokenId ignore_index;
        Tensor<TokenId> targets;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            // Gradient: -1/count at target position (or -1 if no reduction)
            Tensor<float> grad({batch_size, seq_len, vocab_size});
            grad.zero_();

            float scale = (reduction_mean && count > 0) ? (1.0f / count) : 1.0f;
            scale *= grad_output.data()[0];

            const TokenId* target_data = targets.data();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    TokenId target = target_data[b * seq_len + s];

                    if (target == ignore_index) continue;
                    if (target < 0 || static_cast<size_t>(target) >= vocab_size) continue;

                    size_t idx = (b * seq_len + s) * vocab_size + target;
                    grad.data()[idx] = -scale;
                }
            }

            return {grad};
        }
    };
};

// Cross Entropy Loss
// Combines log_softmax and NLLLoss
class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss(
        float label_smoothing = 0.0f,
        TokenId ignore_index = -100,
        bool reduction_mean = true)
        : label_smoothing_(label_smoothing)
        , ignore_index_(ignore_index)
        , reduction_mean_(reduction_mean)
    {}

    autograd::Variable forward(const autograd::Variable& input) override {
        throw std::runtime_error("CrossEntropyLoss requires targets; use forward(logits, targets)");
    }

    // logits: {batch, seq, vocab}, targets: {batch, seq}
    autograd::Variable forward(
        const autograd::Variable& logits,
        const Tensor<TokenId>& targets) {

        const auto& shape = logits.shape();
        if (shape.size() != 3) {
            throw std::runtime_error("CrossEntropyLoss: logits must be 3D {batch, seq, vocab}");
        }

        size_t batch_size = shape[0];
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        // Compute log_softmax for each position
        // log_softmax(x) = x - log(sum(exp(x)))
        Tensor<float> log_probs({batch_size, seq_len, vocab_size});
        const float* logit_data = logits.data().data();
        float* log_prob_data = log_probs.data();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                size_t offset = (b * seq_len + s) * vocab_size;

                // Find max for numerical stability
                float max_val = logit_data[offset];
                for (size_t v = 1; v < vocab_size; ++v) {
                    max_val = std::max(max_val, logit_data[offset + v]);
                }

                // Compute log_sum_exp
                float sum_exp = 0.0f;
                for (size_t v = 0; v < vocab_size; ++v) {
                    sum_exp += std::exp(logit_data[offset + v] - max_val);
                }
                float log_sum_exp = max_val + std::log(sum_exp);

                // log_softmax = x - log_sum_exp
                for (size_t v = 0; v < vocab_size; ++v) {
                    log_prob_data[offset + v] = logit_data[offset + v] - log_sum_exp;
                }
            }
        }

        // Compute loss
        float total_loss = 0.0f;
        size_t count = 0;
        const TokenId* target_data = targets.data();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                TokenId target = target_data[b * seq_len + s];

                if (target == ignore_index_) continue;
                if (target < 0 || static_cast<size_t>(target) >= vocab_size) continue;

                size_t offset = (b * seq_len + s) * vocab_size;

                if (label_smoothing_ > 0.0f) {
                    // With label smoothing:
                    // loss = (1 - eps) * -log_probs[target] + eps * -mean(log_probs)
                    float target_loss = -log_prob_data[offset + target];

                    float mean_log_prob = 0.0f;
                    for (size_t v = 0; v < vocab_size; ++v) {
                        mean_log_prob += log_prob_data[offset + v];
                    }
                    mean_log_prob /= vocab_size;

                    total_loss += (1.0f - label_smoothing_) * target_loss -
                                  label_smoothing_ * mean_log_prob;
                } else {
                    total_loss -= log_prob_data[offset + target];
                }
                ++count;
            }
        }

        float loss_value = (count > 0 && reduction_mean_) ? (total_loss / count) : total_loss;

        Tensor<float> loss_tensor({1});
        loss_tensor.data()[0] = loss_value;

        autograd::Variable loss(loss_tensor, logits.requires_grad());

        if (autograd::is_grad_enabled() && loss.requires_grad()) {
            auto fn = std::make_shared<CEBackward>();
            fn->batch_size = batch_size;
            fn->seq_len = seq_len;
            fn->vocab_size = vocab_size;
            fn->count = count;
            fn->reduction_mean = reduction_mean_;
            fn->ignore_index = ignore_index_;
            fn->label_smoothing = label_smoothing_;
            fn->targets = targets.clone();
            fn->log_probs = log_probs;
            fn->inputs.push_back(logits.impl());
            loss.set_grad_fn(fn);
        }

        return loss;
    }

private:
    float label_smoothing_;
    TokenId ignore_index_;
    bool reduction_mean_;

    class CEBackward : public autograd::Function {
    public:
        size_t batch_size;
        size_t seq_len;
        size_t vocab_size;
        size_t count;
        bool reduction_mean;
        TokenId ignore_index;
        float label_smoothing;
        Tensor<TokenId> targets;
        Tensor<float> log_probs;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            // Gradient of CE loss: softmax(x) - one_hot(target)
            // With label smoothing: softmax(x) - ((1-eps)*one_hot + eps/K)

            Tensor<float> grad({batch_size, seq_len, vocab_size});
            const float* log_prob_data = log_probs.data();
            float* grad_data = grad.data();

            float scale = (reduction_mean && count > 0) ? (1.0f / count) : 1.0f;
            scale *= grad_output.data()[0];

            const TokenId* target_data = targets.data();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t s = 0; s < seq_len; ++s) {
                    TokenId target = target_data[b * seq_len + s];
                    size_t offset = (b * seq_len + s) * vocab_size;

                    if (target == ignore_index) {
                        // Zero gradient for ignored positions
                        for (size_t v = 0; v < vocab_size; ++v) {
                            grad_data[offset + v] = 0.0f;
                        }
                        continue;
                    }

                    // softmax(x) = exp(log_softmax(x))
                    for (size_t v = 0; v < vocab_size; ++v) {
                        float softmax_v = std::exp(log_prob_data[offset + v]);

                        if (label_smoothing > 0.0f) {
                            float target_dist = (static_cast<size_t>(target) == v)
                                ? (1.0f - label_smoothing)
                                : 0.0f;
                            target_dist += label_smoothing / vocab_size;
                            grad_data[offset + v] = (softmax_v - target_dist) * scale;
                        } else {
                            float target_val = (static_cast<size_t>(target) == v) ? 1.0f : 0.0f;
                            grad_data[offset + v] = (softmax_v - target_val) * scale;
                        }
                    }
                }
            }

            return {grad};
        }
    };
};

}  // namespace nn
}  // namespace lightwatch
