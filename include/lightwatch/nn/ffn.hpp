// Phase 17: Feed-Forward Network Block

#pragma once

#include <lightwatch/nn/linear.hpp>
#include <lightwatch/nn/activations.hpp>
#include <lightwatch/nn/dropout.hpp>

namespace lightwatch {
namespace nn {

// Standard GPT-2 FFN: Linear -> GELU -> Linear -> Dropout
class FFN : public Module {
public:
    FFN(size_t embed_dim, size_t hidden_dim, float dropout_p = 0.0f)
        : embed_dim_(embed_dim)
        , hidden_dim_(hidden_dim)
        , fc1(embed_dim, hidden_dim, true)
        , fc2(hidden_dim, embed_dim, true)
        , gelu()
        , dropout(dropout_p)
    {}

    autograd::Variable forward(const autograd::Variable& input) override {
        // FFN: dropout(fc2(gelu(fc1(x))))
        auto h = fc1.forward(input);
        h = gelu.forward(h);
        h = fc2.forward(h);
        h = dropout.forward(h);
        return h;
    }

    // Override to include sub-linear parameters
    size_t num_parameters() const {
        return fc1.num_parameters() + fc2.num_parameters();
    }

    // Override train to propagate to submodules
    void train(bool mode = true) {
        Module::train(mode);
        fc1.train(mode);
        fc2.train(mode);
        dropout.train(mode);
    }

    // Override zero_grad to propagate to submodules
    void zero_grad() {
        fc1.zero_grad();
        fc2.zero_grad();
    }

    size_t embed_dim() const { return embed_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }

    Linear fc1;   // {embed_dim, hidden_dim}
    Linear fc2;   // {hidden_dim, embed_dim}
    GELU gelu;
    Dropout dropout;

private:
    size_t embed_dim_;
    size_t hidden_dim_;
};

// SwiGLU variant (for modern models like LLaMA)
// Uses: down_proj(silu(gate_proj(x)) * up_proj(x))
class SwiGLU : public Module {
public:
    SwiGLU(size_t embed_dim, size_t hidden_dim, float dropout_p = 0.0f)
        : embed_dim_(embed_dim)
        , hidden_dim_(hidden_dim)
        , gate_proj(embed_dim, hidden_dim, false)  // No bias in LLaMA style
        , up_proj(embed_dim, hidden_dim, false)
        , down_proj(hidden_dim, embed_dim, false)
        , dropout(dropout_p)
    {}

    autograd::Variable forward(const autograd::Variable& input) override {
        // SwiGLU: dropout(down_proj(silu(gate_proj(x)) * up_proj(x)))
        auto gate = gate_proj.forward(input);
        gate = silu.forward(gate);
        auto up = up_proj.forward(input);
        auto h = autograd::ops::mul(gate, up);
        h = down_proj.forward(h);
        h = dropout.forward(h);
        return h;
    }

    // Override to include sub-linear parameters
    size_t num_parameters() const {
        return gate_proj.num_parameters() + up_proj.num_parameters() + down_proj.num_parameters();
    }

    // Override train to propagate to submodules
    void train(bool mode = true) {
        Module::train(mode);
        gate_proj.train(mode);
        up_proj.train(mode);
        down_proj.train(mode);
        dropout.train(mode);
    }

    // Override zero_grad to propagate to submodules
    void zero_grad() {
        gate_proj.zero_grad();
        up_proj.zero_grad();
        down_proj.zero_grad();
    }

    size_t embed_dim() const { return embed_dim_; }
    size_t hidden_dim() const { return hidden_dim_; }

    Linear gate_proj;  // {embed_dim, hidden_dim}
    Linear up_proj;    // {embed_dim, hidden_dim}
    Linear down_proj;  // {hidden_dim, embed_dim}
    SiLU silu;
    Dropout dropout;

private:
    size_t embed_dim_;
    size_t hidden_dim_;
};

}  // namespace nn
}  // namespace lightwatch
