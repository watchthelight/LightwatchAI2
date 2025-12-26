// Phase 11: Dense (Linear) Layer
// Fully connected layer: y = xW^T + b

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/autograd.hpp>
#include <cmath>
#include <random>

namespace lightwatch::nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true)
        : in_features_(in_features)
        , out_features_(out_features)
        , has_bias_(bias)
        , weight(init_weight(in_features, out_features), true)
        , bias_(init_bias(out_features, bias), bias) {

        register_parameter("weight", weight);
        if (has_bias_) {
            register_parameter("bias", bias_);
        }
    }

    autograd::Variable forward(const autograd::Variable& input) override {
        // input: {*, in_features}
        // output: {*, out_features}

        // Get input shape
        const auto& in_shape = input.shape();
        size_t ndim = in_shape.size();

        if (ndim == 0 || in_shape[ndim - 1] != in_features_) {
            throw std::runtime_error("Linear: input last dimension (" +
                std::to_string(in_shape[ndim - 1]) +
                ") must match in_features (" +
                std::to_string(in_features_) + ")");
        }

        // Reshape for batched matmul if needed
        if (ndim == 2) {
            // Simple case: {batch, in_features}
            // output = input @ weight.T
            auto weight_t = autograd::ops::transpose(weight, 0, 1);
            auto output = autograd::ops::matmul(input, weight_t);

            if (has_bias_) {
                // Broadcast add bias
                output = add_bias(output, bias_);
            }
            return output;
        } else {
            // General case: {*, in_features} -> {*, out_features}
            // Flatten leading dimensions, apply linear, reshape back

            // Calculate total batch size
            size_t batch_size = 1;
            for (size_t i = 0; i < ndim - 1; ++i) {
                batch_size *= in_shape[i];
            }

            // Reshape to {batch_size, in_features}
            auto reshaped = autograd::ops::reshape(input, {batch_size, in_features_});

            // Apply linear transformation
            auto weight_t = autograd::ops::transpose(weight, 0, 1);
            auto output = autograd::ops::matmul(reshaped, weight_t);

            if (has_bias_) {
                output = add_bias(output, bias_);
            }

            // Build output shape
            Shape out_shape;
            for (size_t i = 0; i < ndim - 1; ++i) {
                out_shape.push_back(in_shape[i]);
            }
            out_shape.push_back(out_features_);

            // Reshape back
            return autograd::ops::reshape(output, out_shape);
        }
    }

    // Accessor for bias (maintains API compatibility)
    autograd::Variable& bias() { return bias_; }
    const autograd::Variable& bias() const { return bias_; }

    bool has_bias() const { return has_bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

    autograd::Variable weight;

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    autograd::Variable bias_;

    // Xavier/Glorot uniform initialization
    static Tensor<float> init_weight(size_t in_features, size_t out_features) {
        // Weight shape: {out_features, in_features}
        Tensor<float> w({out_features, in_features});

        // Xavier uniform: U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))
        float limit = std::sqrt(6.0f / static_cast<float>(in_features + out_features));
        std::uniform_real_distribution<float> dist(-limit, limit);

        for (size_t i = 0; i < w.numel(); ++i) {
            w.data()[i] = dist(lightwatch::detail::get_rng());
        }

        return w;
    }

    // Zero initialization for bias
    static Tensor<float> init_bias(size_t out_features, bool has_bias) {
        if (!has_bias) {
            return Tensor<float>();  // Empty tensor
        }
        return Tensor<float>::zeros({out_features});
    }

    // Add bias with broadcasting
    static autograd::Variable add_bias(const autograd::Variable& input,
                                        const autograd::Variable& bias) {
        // input: {batch, out_features}
        // bias: {out_features}
        // Result: input + bias (broadcast)

        const auto& in_shape = input.shape();
        size_t batch_size = in_shape[0];
        size_t features = in_shape[1];

        Tensor<float> result(in_shape);
        const float* in_data = input.data().data();
        const float* bias_data = bias.data().data();
        float* out_data = result.data();

        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t f = 0; f < features; ++f) {
                out_data[b * features + f] = in_data[b * features + f] + bias_data[f];
            }
        }

        autograd::Variable out(result, input.requires_grad() || bias.requires_grad());

        if (autograd::is_grad_enabled() && out.requires_grad()) {
            auto fn = std::make_shared<LinearBiasBackward>();
            fn->batch_size = batch_size;
            fn->inputs.push_back(input.impl());
            fn->inputs.push_back(bias.impl());
            out.set_grad_fn(fn);
        }

        return out;
    }

    // Backward for bias addition with reduction
    class LinearBiasBackward : public autograd::Function {
    public:
        size_t batch_size;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            // grad_input: same as grad_output
            // grad_bias: sum over batch dimension

            size_t features = grad_output.numel() / batch_size;
            Tensor<float> grad_bias({features});
            grad_bias.zero_();

            const float* grad_data = grad_output.data();
            float* bias_grad = grad_bias.data();

            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t f = 0; f < features; ++f) {
                    bias_grad[f] += grad_data[b * features + f];
                }
            }

            return {grad_output.clone(), grad_bias};
        }
    };
};

}  // namespace lightwatch::nn
