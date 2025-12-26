// Phase 13: Layer Normalization
// LayerNorm and RMSNorm for transformer normalization

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/autograd.hpp>
#include <cmath>

namespace lightwatch::nn {

// LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
class LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5f)
        : normalized_shape_(normalized_shape)
        , eps_(eps)
        , weight(Tensor<float>::ones({normalized_shape}), true)
        , bias(Tensor<float>::zeros({normalized_shape}), true) {

        register_parameter("weight", weight);
        register_parameter("bias", bias);
    }

    autograd::Variable forward(const autograd::Variable& input) override {
        // Normalize over last dimension
        return autograd::ops::layer_norm(input, weight, bias, eps_);
    }

    size_t normalized_shape() const { return normalized_shape_; }
    float eps() const { return eps_; }

    autograd::Variable weight;
    autograd::Variable bias;

private:
    size_t normalized_shape_;
    float eps_;
};

// RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
class RMSNorm : public Module {
public:
    RMSNorm(size_t normalized_shape, float eps = 1e-6f)
        : normalized_shape_(normalized_shape)
        , eps_(eps)
        , weight(Tensor<float>::ones({normalized_shape}), true) {

        register_parameter("weight", weight);
    }

    autograd::Variable forward(const autograd::Variable& input) override {
        // Compute RMS normalization
        const auto& x = input.data();
        int dim = static_cast<int>(x.ndim()) - 1;

        // Compute x^2
        Tensor<float> x_sq(x.shape());
        for (size_t i = 0; i < x.numel(); ++i) {
            x_sq.data()[i] = x.data()[i] * x.data()[i];
        }
        autograd::Variable x_squared(x_sq, input.requires_grad());

        // Compute mean(x^2) along last dimension
        auto mean_sq = autograd::ops::mean(x_squared, dim, true);

        // Compute sqrt(mean(x^2) + eps)
        Tensor<float> rms_tensor(mean_sq.data().shape());
        for (size_t i = 0; i < mean_sq.numel(); ++i) {
            rms_tensor.data()[i] = std::sqrt(mean_sq.data().data()[i] + eps_);
        }
        autograd::Variable rms(rms_tensor, mean_sq.requires_grad());

        // Divide x by rms
        auto normalized = autograd::ops::div(input, rms);

        // Multiply by weight
        return autograd::ops::mul(normalized, weight);
    }

    size_t normalized_shape() const { return normalized_shape_; }
    float eps() const { return eps_; }

    autograd::Variable weight;

private:
    size_t normalized_shape_;
    float eps_;
};

}  // namespace lightwatch::nn
