// Phase 14: Dropout Layers
// Dropout and DropPath regularization

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/autograd.hpp>
#include <random>

namespace lightwatch::nn {

// Dropout: Randomly zero elements with probability p
// Uses inverted dropout: scale by 1/(1-p) during training
class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1f) : p_(p) {}

    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::dropout(input, p_, is_training());
    }

    float p() const { return p_; }

private:
    float p_;
};

// DropPath (Stochastic Depth): Drop entire samples with probability p
// Used in residual connections to randomly drop entire residual branches
class DropPath : public Module {
public:
    explicit DropPath(float p = 0.1f) : p_(p) {}

    autograd::Variable forward(const autograd::Variable& input) override {
        if (!is_training() || p_ == 0.0f) {
            return autograd::Variable(input.data().clone(), input.requires_grad());
        }

        // Get batch dimension
        const auto& shape = input.shape();
        size_t batch_size = shape[0];

        // Create per-sample mask
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float scale = 1.0f / (1.0f - p_);

        Tensor<float> result(shape);
        const float* in_data = input.data().data();
        float* out_data = result.data();

        // Calculate elements per sample
        size_t elements_per_sample = input.numel() / batch_size;

        for (size_t b = 0; b < batch_size; ++b) {
            bool keep = dist(lightwatch::detail::get_rng()) > p_;
            float sample_scale = keep ? scale : 0.0f;

            for (size_t i = 0; i < elements_per_sample; ++i) {
                size_t idx = b * elements_per_sample + i;
                out_data[idx] = in_data[idx] * sample_scale;
            }
        }

        autograd::Variable out(result, input.requires_grad());

        if (autograd::is_grad_enabled() && out.requires_grad()) {
            auto fn = std::make_shared<DropPathBackward>();
            fn->batch_size = batch_size;
            fn->elements_per_sample = elements_per_sample;
            fn->result_data = result.clone();  // Save mask implicitly
            fn->scale = scale;
            fn->inputs.push_back(input.impl());
            out.set_grad_fn(fn);
        }

        return out;
    }

    float p() const { return p_; }

private:
    float p_;

    class DropPathBackward : public autograd::Function {
    public:
        size_t batch_size;
        size_t elements_per_sample;
        Tensor<float> result_data;
        float scale;

        std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
            Tensor<float> grad_input(grad_output.shape());
            const float* grad_out = grad_output.data();
            const float* result = result_data.data();
            float* grad_in = grad_input.data();

            for (size_t b = 0; b < batch_size; ++b) {
                // Check if sample was kept (first element non-zero implies kept)
                size_t base = b * elements_per_sample;
                bool was_kept = (result[base] != 0.0f);

                float sample_scale = was_kept ? scale : 0.0f;
                for (size_t i = 0; i < elements_per_sample; ++i) {
                    grad_in[base + i] = grad_out[base + i] * sample_scale;
                }
            }

            return {grad_input};
        }
    };
};

}  // namespace lightwatch::nn
