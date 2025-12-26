// Phase 18: Transformer Encoder Block

#pragma once

#include <lightwatch/nn/normalization.hpp>
#include <lightwatch/nn/attention.hpp>
#include <lightwatch/nn/ffn.hpp>
#include <lightwatch/nn/dropout.hpp>

namespace lightwatch {
namespace nn {

// Single transformer encoder block with pre-norm or post-norm architecture
class TransformerEncoderBlock : public Module {
public:
    TransformerEncoderBlock(
        size_t embed_dim,
        size_t num_heads,
        size_t ffn_dim,
        float dropout_p = 0.1f,
        bool pre_norm = true)
        : embed_dim_(embed_dim)
        , num_heads_(num_heads)
        , ffn_dim_(ffn_dim)
        , pre_norm_(pre_norm)
        , ln1(embed_dim)
        , attn(embed_dim, num_heads, dropout_p)
        , dropout1(dropout_p)
        , ln2(embed_dim)
        , ffn(embed_dim, ffn_dim, dropout_p)
        , dropout2(dropout_p)
    {}

    // Forward without mask (self-attention)
    autograd::Variable forward(const autograd::Variable& input) override {
        return forward(input, nullptr);
    }

    // Forward with optional attention mask
    autograd::Variable forward(
        const autograd::Variable& input,
        const Tensor<float>* mask) {

        autograd::Variable x = input;

        if (pre_norm_) {
            // Pre-norm architecture (GPT-2 style)
            // x = x + dropout(attn(ln1(x)))
            auto normed = ln1.forward(x);
            auto attn_out = attn.forward(normed, normed, normed, mask);
            auto dropped = dropout1.forward(attn_out);
            x = autograd::ops::add(x, dropped);

            // x = x + dropout(ffn(ln2(x)))
            normed = ln2.forward(x);
            auto ffn_out = ffn.forward(normed);
            dropped = dropout2.forward(ffn_out);
            x = autograd::ops::add(x, dropped);
        } else {
            // Post-norm architecture (original Transformer)
            // x = ln1(x + dropout(attn(x)))
            auto attn_out = attn.forward(x, x, x, mask);
            auto dropped = dropout1.forward(attn_out);
            x = autograd::ops::add(x, dropped);
            x = ln1.forward(x);

            // x = ln2(x + dropout(ffn(x)))
            auto ffn_out = ffn.forward(x);
            dropped = dropout2.forward(ffn_out);
            x = autograd::ops::add(x, dropped);
            x = ln2.forward(x);
        }

        return x;
    }

    // Override to count all submodule parameters
    size_t num_parameters() const {
        return ln1.num_parameters() + attn.num_parameters() +
               ln2.num_parameters() + ffn.num_parameters();
    }

    // Override train mode to propagate
    void train(bool mode = true) {
        Module::train(mode);
        ln1.train(mode);
        attn.train(mode);
        dropout1.train(mode);
        ln2.train(mode);
        ffn.train(mode);
        dropout2.train(mode);
    }

    // Override zero_grad
    void zero_grad() {
        ln1.zero_grad();
        attn.zero_grad();
        ln2.zero_grad();
        ffn.zero_grad();
    }

    size_t embed_dim() const { return embed_dim_; }
    size_t num_heads() const { return num_heads_; }
    size_t ffn_dim() const { return ffn_dim_; }
    bool is_pre_norm() const { return pre_norm_; }

    LayerNorm ln1;
    MultiHeadAttention attn;
    Dropout dropout1;
    LayerNorm ln2;
    FFN ffn;
    Dropout dropout2;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t ffn_dim_;
    bool pre_norm_;
};

// Transformer Decoder Block with causal self-attention (GPT-style)
class TransformerDecoderBlock : public Module {
public:
    TransformerDecoderBlock(
        size_t embed_dim,
        size_t num_heads,
        size_t ffn_dim,
        float dropout_p = 0.1f)
        : embed_dim_(embed_dim)
        , num_heads_(num_heads)
        , ffn_dim_(ffn_dim)
        , ln1(embed_dim)
        , attn(embed_dim, num_heads, dropout_p)
        , dropout1(dropout_p)
        , ln2(embed_dim)
        , ffn(embed_dim, ffn_dim, dropout_p)
        , dropout2(dropout_p)
    {}

    // Forward with automatic causal mask
    autograd::Variable forward(const autograd::Variable& input) override {
        size_t seq_len = input.shape()[1];
        auto mask = causal_mask(seq_len);
        return forward(input, &mask);
    }

    // Forward with custom mask (for KV-cache or custom attention patterns)
    autograd::Variable forward(
        const autograd::Variable& input,
        const Tensor<float>* mask) {

        autograd::Variable x = input;

        // Pre-norm architecture (GPT-2 style)
        // x = x + dropout(attn(ln1(x), mask))
        auto normed = ln1.forward(x);
        auto attn_out = attn.forward(normed, normed, normed, mask);
        auto dropped = dropout1.forward(attn_out);
        x = autograd::ops::add(x, dropped);

        // x = x + dropout(ffn(ln2(x)))
        normed = ln2.forward(x);
        auto ffn_out = ffn.forward(normed);
        dropped = dropout2.forward(ffn_out);
        x = autograd::ops::add(x, dropped);

        return x;
    }

    // Override to count all submodule parameters
    size_t num_parameters() const {
        return ln1.num_parameters() + attn.num_parameters() +
               ln2.num_parameters() + ffn.num_parameters();
    }

    // Override train mode to propagate
    void train(bool mode = true) {
        Module::train(mode);
        ln1.train(mode);
        attn.train(mode);
        dropout1.train(mode);
        ln2.train(mode);
        ffn.train(mode);
        dropout2.train(mode);
    }

    // Override zero_grad
    void zero_grad() {
        ln1.zero_grad();
        attn.zero_grad();
        ln2.zero_grad();
        ffn.zero_grad();
    }

    size_t embed_dim() const { return embed_dim_; }
    size_t num_heads() const { return num_heads_; }
    size_t ffn_dim() const { return ffn_dim_; }

    LayerNorm ln1;
    MultiHeadAttention attn;
    Dropout dropout1;
    LayerNorm ln2;
    FFN ffn;
    Dropout dropout2;

private:
    size_t embed_dim_;
    size_t num_heads_;
    size_t ffn_dim_;
};

}  // namespace nn
}  // namespace lightwatch
