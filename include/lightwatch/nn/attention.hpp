// Phase 15: Single-Head Attention
// Scaled dot-product attention with optional causal masking

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/nn/dropout.hpp>
#include <lightwatch/autograd.hpp>
#include <cmath>
#include <limits>

namespace lightwatch::nn {

// Utility: create causal (autoregressive) mask
// Returns lower triangular matrix where [i,j] = 1.0 if i >= j, else 0.0
inline Tensor<float> causal_mask(size_t seq_len) {
    Tensor<float> mask({seq_len, seq_len});
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            mask.data()[i * seq_len + j] = (i >= j) ? 1.0f : 0.0f;
        }
    }
    return mask;
}

// Scaled Dot-Product Attention
// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
class ScaledDotProductAttention : public Module {
public:
    explicit ScaledDotProductAttention(float dropout_p = 0.0f)
        : dropout_(dropout_p) {}

    // Main forward: q, k, v: {batch, seq_len, head_dim}
    // mask: optional {seq_len, seq_len} float mask (1.0 = attend, 0.0 = mask out)
    autograd::Variable forward(
        const autograd::Variable& query,
        const autograd::Variable& key,
        const autograd::Variable& value,
        const Tensor<float>* mask = nullptr) {

        // Get dimensions
        const auto& q_shape = query.shape();
        size_t batch_size = q_shape[0];
        size_t seq_len = q_shape[1];
        size_t head_dim = q_shape[2];

        // Compute scale factor
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Compute attention scores: Q @ K^T
        // Q: {batch, seq, head_dim}
        // K^T: {batch, head_dim, seq}
        auto key_t = transpose_12(key, batch_size, seq_len, head_dim);

        // Batched matmul: {batch, seq, head_dim} @ {batch, head_dim, seq} = {batch, seq, seq}
        auto scores = batched_matmul(query, key_t);

        // Scale scores
        Tensor<float> scaled_scores(scores.data().shape());
        for (size_t i = 0; i < scores.numel(); ++i) {
            scaled_scores.data()[i] = scores.data().data()[i] * scale;
        }
        autograd::Variable scaled(scaled_scores, scores.requires_grad());
        if (autograd::is_grad_enabled() && scaled.requires_grad()) {
            auto fn = std::make_shared<ScaleBackward>();
            fn->scale = scale;
            fn->inputs.push_back(scores.impl());
            scaled.set_grad_fn(fn);
        }

        // Apply mask if provided
        autograd::Variable masked = scaled;
        if (mask) {
            masked = apply_mask(scaled, *mask, batch_size, seq_len);
        }

        // Softmax over last dimension (keys)
        auto weights = autograd::ops::softmax(masked, -1);  // {batch, seq, seq}

        // Apply dropout to attention weights
        weights = dropout_.forward(weights);

        // Compute output: weights @ V
        // weights: {batch, seq, seq}
        // V: {batch, seq, head_dim}
        // Output: {batch, seq, head_dim}
        auto output = batched_matmul(weights, value);

        return output;
    }

    // Required override for single-input forward (not used in practice for attention)
    autograd::Variable forward(const autograd::Variable& input) override {
        // For attention, we need Q, K, V - use the same input for all (self-attention)
        return forward(input, input, input, nullptr);
    }

    float dropout_p() const { return dropout_.p(); }

private:
    Dropout dropout_;

    // Scale backward function
    class ScaleBackward : public autograd::Function {
    public:
        float scale;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            Tensor<float> grad_input(grad_output.shape());
            for (size_t i = 0; i < grad_output.numel(); ++i) {
                grad_input.data()[i] = grad_output.data()[i] * scale;
            }
            return {grad_input};
        }
    };

    // Apply attention mask
    autograd::Variable apply_mask(
        const autograd::Variable& scores,
        const Tensor<float>& mask,
        size_t batch_size,
        size_t seq_len) {

        Tensor<float> masked_scores(scores.data().shape());
        const float* score_data = scores.data().data();
        float* masked_data = masked_scores.data();

        constexpr float NEG_INF = -1e9f;  // Use large negative instead of -inf

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    size_t idx = b * seq_len * seq_len + i * seq_len + j;
                    size_t mask_idx = i * seq_len + j;

                    if (mask.data()[mask_idx] > 0.5f) {
                        masked_data[idx] = score_data[idx];
                    } else {
                        masked_data[idx] = NEG_INF;
                    }
                }
            }
        }

        autograd::Variable out(masked_scores, scores.requires_grad());

        if (autograd::is_grad_enabled() && out.requires_grad()) {
            auto fn = std::make_shared<MaskBackward>();
            fn->batch_size = batch_size;
            fn->seq_len = seq_len;
            fn->float_mask = mask.clone();  // Store copy
            fn->inputs.push_back(scores.impl());
            out.set_grad_fn(fn);
        }

        return out;
    }

    // Mask backward
    class MaskBackward : public autograd::Function {
    public:
        size_t batch_size;
        size_t seq_len;
        Tensor<float> float_mask;  // Store copy of mask

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            Tensor<float> grad_input(grad_output.shape());
            const float* grad_out = grad_output.data();
            float* grad_in = grad_input.data();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < seq_len; ++j) {
                        size_t idx = b * seq_len * seq_len + i * seq_len + j;
                        size_t mask_idx = i * seq_len + j;

                        // Gradient flows only through unmasked positions
                        grad_in[idx] = float_mask.data()[mask_idx] * grad_out[idx];
                    }
                }
            }

            return {grad_input};
        }
    };

    // Transpose dimensions 1 and 2: {batch, d1, d2} -> {batch, d2, d1}
    autograd::Variable transpose_12(
        const autograd::Variable& x,
        size_t batch_size,
        size_t d1,
        size_t d2) {

        Tensor<float> result({batch_size, d2, d1});
        const float* in_data = x.data().data();
        float* out_data = result.data();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t i = 0; i < d1; ++i) {
                for (size_t j = 0; j < d2; ++j) {
                    // x[b, i, j] -> result[b, j, i]
                    out_data[b * d2 * d1 + j * d1 + i] =
                        in_data[b * d1 * d2 + i * d2 + j];
                }
            }
        }

        autograd::Variable out(result, x.requires_grad());

        if (autograd::is_grad_enabled() && out.requires_grad()) {
            auto fn = std::make_shared<Transpose12Backward>();
            fn->batch_size = batch_size;
            fn->d1 = d1;
            fn->d2 = d2;
            fn->inputs.push_back(x.impl());
            out.set_grad_fn(fn);
        }

        return out;
    }

    class Transpose12Backward : public autograd::Function {
    public:
        size_t batch_size, d1, d2;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            // grad_output: {batch, d2, d1}
            // Need grad_input: {batch, d1, d2}
            Tensor<float> grad_input({batch_size, d1, d2});
            const float* go = grad_output.data();
            float* gi = grad_input.data();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < d2; ++i) {
                    for (size_t j = 0; j < d1; ++j) {
                        // go[b, i, j] -> gi[b, j, i]
                        gi[b * d1 * d2 + j * d2 + i] =
                            go[b * d2 * d1 + i * d1 + j];
                    }
                }
            }

            return {grad_input};
        }
    };

    // Batched matrix multiplication for 3D tensors
    // A: {batch, m, k}, B: {batch, k, n} -> {batch, m, n}
    autograd::Variable batched_matmul(
        const autograd::Variable& a,
        const autograd::Variable& b) {

        const auto& a_shape = a.shape();
        const auto& b_shape = b.shape();

        size_t batch = a_shape[0];
        size_t m = a_shape[1];
        size_t k = a_shape[2];
        size_t n = b_shape[2];

        Tensor<float> result({batch, m, n});
        result.zero_();

        const float* a_data = a.data().data();
        const float* b_data = b.data().data();
        float* out_data = result.data();

        for (size_t ba = 0; ba < batch; ++ba) {
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (size_t p = 0; p < k; ++p) {
                        sum += a_data[ba * m * k + i * k + p] *
                               b_data[ba * k * n + p * n + j];
                    }
                    out_data[ba * m * n + i * n + j] = sum;
                }
            }
        }

        autograd::Variable out(result, a.requires_grad() || b.requires_grad());

        if (autograd::is_grad_enabled() && out.requires_grad()) {
            auto fn = std::make_shared<BatchedMatmulBackward>();
            fn->save_tensor(a.data());
            fn->save_tensor(b.data());
            fn->batch = batch;
            fn->m = m;
            fn->k = k;
            fn->n = n;
            fn->inputs.push_back(a.impl());
            fn->inputs.push_back(b.impl());
            out.set_grad_fn(fn);
        }

        return out;
    }

    // Batched matmul backward
    class BatchedMatmulBackward : public autograd::Function {
    public:
        size_t batch, m, k, n;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            const auto& a = saved_tensors[0];
            const auto& b = saved_tensors[1];

            // grad_a = grad_output @ b^T
            // grad_output: {batch, m, n}, b: {batch, k, n}
            // b^T: {batch, n, k}
            // grad_a: {batch, m, k}
            Tensor<float> grad_a({batch, m, k});
            grad_a.zero_();

            const float* go = grad_output.data();
            const float* b_data = b.data();
            float* ga = grad_a.data();

            for (size_t ba = 0; ba < batch; ++ba) {
                for (size_t i = 0; i < m; ++i) {
                    for (size_t p = 0; p < k; ++p) {
                        float sum = 0.0f;
                        for (size_t j = 0; j < n; ++j) {
                            // grad_output[ba, i, j] * b[ba, p, j]
                            sum += go[ba * m * n + i * n + j] *
                                   b_data[ba * k * n + p * n + j];
                        }
                        ga[ba * m * k + i * k + p] = sum;
                    }
                }
            }

            // grad_b = a^T @ grad_output
            // a: {batch, m, k}, a^T: {batch, k, m}
            // grad_output: {batch, m, n}
            // grad_b: {batch, k, n}
            Tensor<float> grad_b({batch, k, n});
            grad_b.zero_();

            const float* a_data = a.data();
            float* gb = grad_b.data();

            for (size_t ba = 0; ba < batch; ++ba) {
                for (size_t p = 0; p < k; ++p) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0.0f;
                        for (size_t i = 0; i < m; ++i) {
                            // a[ba, i, p] * grad_output[ba, i, j]
                            sum += a_data[ba * m * k + i * k + p] *
                                   go[ba * m * n + i * n + j];
                        }
                        gb[ba * k * n + p * n + j] = sum;
                    }
                }
            }

            return {grad_a, grad_b};
        }
    };
};

}  // namespace lightwatch::nn
