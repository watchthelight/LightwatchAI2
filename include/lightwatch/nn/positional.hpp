// Phase 09: Positional Encoding
// Sinusoidal, RoPE, and ALiBi positional encoding schemes

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/tensor.hpp>
#include <lightwatch/autograd.hpp>
#include <cmath>
#include <utility>

namespace lightwatch::nn {

// Identity backward for non-trainable additions (passes gradient through unchanged)
class IdentityBackward : public ::lightwatch::autograd::Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.clone()};
    }
};

// Sinusoidal positional encoding (original Transformer)
class SinusoidalPE : public Module {
public:
    SinusoidalPE(size_t max_seq_len, size_t embed_dim)
        : max_seq_len_(max_seq_len)
        , embed_dim_(embed_dim) {

        // Precompute encodings: PE(pos, 2i) = sin(pos/10000^(2i/d))
        //                       PE(pos, 2i+1) = cos(pos/10000^(2i/d))
        encodings_ = Tensor<float>({max_seq_len, embed_dim});

        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < embed_dim; i += 2) {
                float div_term = std::exp(
                    -static_cast<float>(i) * std::log(10000.0f) /
                    static_cast<float>(embed_dim));

                encodings_({pos, i}) = std::sin(static_cast<float>(pos) * div_term);

                if (i + 1 < embed_dim) {
                    encodings_({pos, i + 1}) = std::cos(static_cast<float>(pos) * div_term);
                }
            }
        }
    }

    // Returns positional encoding for given sequence length
    Tensor<float> get_encoding(size_t seq_len) const {
        if (seq_len > max_seq_len_) {
            throw std::runtime_error(
                "Sequence length " + std::to_string(seq_len) +
                " exceeds max " + std::to_string(max_seq_len_));
        }

        return encodings_.slice(0, 0, seq_len);
    }

    // Add positional encoding to input embeddings
    // Input shape: {batch, seq_len, embed_dim} or {seq_len, embed_dim}
    autograd::Variable forward(const autograd::Variable& input) override {
        const auto& shape = input.data().shape();
        size_t seq_len;

        if (shape.size() == 2) {
            seq_len = shape[0];
        } else if (shape.size() == 3) {
            seq_len = shape[1];
        } else {
            throw std::runtime_error("SinusoidalPE: expected 2D or 3D input");
        }

        // Get encoding and broadcast add
        Tensor<float> pe = get_encoding(seq_len);
        Tensor<float> result = input.data().clone();

        if (shape.size() == 2) {
            // {seq_len, embed_dim}
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < embed_dim_; ++j) {
                    result({i, j}) += pe({i, j});
                }
            }
        } else {
            // {batch, seq_len, embed_dim}
            size_t batch_size = shape[0];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < embed_dim_; ++j) {
                        result({b, i, j}) += pe({i, j});
                    }
                }
            }
        }

        // Positional encoding is not trainable, so gradient just passes through
        ::lightwatch::autograd::Variable out(result, input.requires_grad());

        if (input.requires_grad() && ::lightwatch::autograd::is_grad_enabled()) {
            // Identity gradient - just pass through
            auto fn = std::make_shared<IdentityBackward>();
            fn->inputs.push_back(input.impl());
            out.impl()->grad_fn = fn;
        }

        return out;
    }

    size_t max_seq_len() const { return max_seq_len_; }
    size_t embed_dim() const { return embed_dim_; }

private:
    size_t max_seq_len_;
    size_t embed_dim_;
    Tensor<float> encodings_;
};

// Rotary Position Embedding (RoPE)
// Applies rotation to query and key tensors based on position
class RoPE {
public:
    RoPE(size_t head_dim, size_t max_seq_len = 2048, float base = 10000.0f)
        : head_dim_(head_dim)
        , max_seq_len_(max_seq_len)
        , base_(base) {

        // Precompute cos and sin caches
        // For each position and each pair of dimensions
        cos_cached_ = Tensor<float>({max_seq_len, head_dim / 2});
        sin_cached_ = Tensor<float>({max_seq_len, head_dim / 2});

        // Compute inverse frequencies
        // inv_freq[i] = 1 / (base^(2i/head_dim)) for i in [0, head_dim/2)
        std::vector<float> inv_freq(head_dim / 2);
        for (size_t i = 0; i < head_dim / 2; ++i) {
            inv_freq[i] = 1.0f / std::pow(base, 2.0f * static_cast<float>(i) /
                                              static_cast<float>(head_dim));
        }

        // Compute cos and sin for each position
        for (size_t pos = 0; pos < max_seq_len; ++pos) {
            for (size_t i = 0; i < head_dim / 2; ++i) {
                float angle = static_cast<float>(pos) * inv_freq[i];
                cos_cached_({pos, i}) = std::cos(angle);
                sin_cached_({pos, i}) = std::sin(angle);
            }
        }
    }

    // Apply rotation to query and key tensors
    // q, k shape: {batch, heads, seq, head_dim}
    // Returns rotated q and k with same shapes
    std::pair<Tensor<float>, Tensor<float>> apply(
        const Tensor<float>& q,
        const Tensor<float>& k,
        size_t offset = 0) const {

        auto q_shape = q.shape();
        auto k_shape = k.shape();

        if (q_shape.size() != 4 || k_shape.size() != 4) {
            throw std::runtime_error("RoPE: expected 4D tensors (batch, heads, seq, head_dim)");
        }

        size_t batch = q_shape[0];
        size_t heads = q_shape[1];
        size_t seq_len = q_shape[2];
        size_t dim = q_shape[3];

        if (dim != head_dim_) {
            throw std::runtime_error("RoPE: head_dim mismatch");
        }

        if (offset + seq_len > max_seq_len_) {
            throw std::runtime_error("RoPE: sequence too long");
        }

        Tensor<float> q_rot(q_shape);
        Tensor<float> k_rot(k_shape);

        // Apply rotation
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t s = 0; s < seq_len; ++s) {
                    size_t pos = offset + s;

                    for (size_t i = 0; i < dim / 2; ++i) {
                        float cos_val = cos_cached_({pos, i});
                        float sin_val = sin_cached_({pos, i});

                        // Get pairs (x, y) from positions (2i, 2i+1)
                        size_t idx1 = b * heads * seq_len * dim +
                                     h * seq_len * dim +
                                     s * dim + 2 * i;
                        size_t idx2 = idx1 + 1;

                        // Rotate q
                        float q1 = q.data()[idx1];
                        float q2 = q.data()[idx2];
                        q_rot.data()[idx1] = q1 * cos_val - q2 * sin_val;
                        q_rot.data()[idx2] = q1 * sin_val + q2 * cos_val;

                        // Rotate k
                        float k1 = k.data()[idx1];
                        float k2 = k.data()[idx2];
                        k_rot.data()[idx1] = k1 * cos_val - k2 * sin_val;
                        k_rot.data()[idx2] = k1 * sin_val + k2 * cos_val;
                    }
                }
            }
        }

        return {q_rot, k_rot};
    }

    size_t head_dim() const { return head_dim_; }
    size_t max_seq_len() const { return max_seq_len_; }

private:
    size_t head_dim_;
    size_t max_seq_len_;
    float base_;
    Tensor<float> cos_cached_;
    Tensor<float> sin_cached_;
};

// Attention with Linear Biases (ALiBi)
// Adds position-dependent bias to attention scores
class ALiBi {
public:
    explicit ALiBi(size_t num_heads) : num_heads_(num_heads) {
        // Compute slopes: for head i, slope = 2^(-8 * (i+1) / num_heads)
        slopes_ = Tensor<float>({num_heads});

        for (size_t i = 0; i < num_heads; ++i) {
            // Use geometric sequence: 2^(-8/n), 2^(-16/n), ...
            float power = -8.0f * static_cast<float>(i + 1) / static_cast<float>(num_heads);
            slopes_.data()[i] = std::pow(2.0f, power);
        }
    }

    // Returns bias to add to attention scores
    // Shape: {num_heads, seq_len, seq_len}
    // bias[h][i][j] = -slopes[h] * |i - j| for causal (j <= i)
    Tensor<float> get_bias(size_t seq_len) const {
        Tensor<float> bias({num_heads_, seq_len, seq_len});

        for (size_t h = 0; h < num_heads_; ++h) {
            float slope = slopes_.data()[h];

            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    // ALiBi uses distance: bias = -m * (i - j) for j <= i
                    // For causal attention, j > i positions are masked anyway
                    int dist = static_cast<int>(i) - static_cast<int>(j);
                    if (dist >= 0) {
                        bias({h, i, j}) = -slope * static_cast<float>(dist);
                    } else {
                        // Non-causal or future positions get large negative (will be masked)
                        bias({h, i, j}) = -1e9f;
                    }
                }
            }
        }

        return bias;
    }

    // Get causal bias (j <= i only)
    Tensor<float> get_causal_bias(size_t seq_len) const {
        return get_bias(seq_len);
    }

    // Get slopes for inspection
    const Tensor<float>& slopes() const { return slopes_; }
    size_t num_heads() const { return num_heads_; }

private:
    size_t num_heads_;
    Tensor<float> slopes_;
};

}  // namespace lightwatch::nn
