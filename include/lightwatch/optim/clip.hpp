// Phase 25: Gradient Clipping

#pragma once

#include <lightwatch/autograd.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace lightwatch {
namespace optim {

// Compute gradient norm without clipping
// norm_type: 1.0 = L1 norm, 2.0 = L2 norm, inf = Linf norm
inline float grad_norm(
    const std::vector<autograd::Variable*>& params,
    float norm_type = 2.0f) {

    if (params.empty()) {
        return 0.0f;
    }

    // Handle infinity norm separately
    if (std::isinf(norm_type)) {
        float max_val = 0.0f;
        for (const auto* param : params) {
            if (!param->has_grad()) {
                continue;
            }
            const Tensor<float>& grad = param->grad();
            for (size_t i = 0; i < grad.numel(); ++i) {
                max_val = std::max(max_val, std::abs(grad.data()[i]));
            }
        }
        return max_val;
    }

    // General Lp norm
    float total = 0.0f;
    for (const auto* param : params) {
        if (!param->has_grad()) {
            continue;
        }
        const Tensor<float>& grad = param->grad();
        for (size_t i = 0; i < grad.numel(); ++i) {
            total += std::pow(std::abs(grad.data()[i]), norm_type);
        }
    }

    return std::pow(total, 1.0f / norm_type);
}

// Clip gradients by global norm (modifies in-place)
// Returns the original total norm before clipping
inline float clip_grad_norm_(
    std::vector<autograd::Variable*>& params,
    float max_norm,
    float norm_type = 2.0f) {

    // Compute current gradient norm
    float total_norm = grad_norm(params, norm_type);

    // Avoid division by zero
    float clip_coef = max_norm / (total_norm + 1e-6f);

    // Only clip if norm exceeds max_norm
    if (clip_coef < 1.0f) {
        for (auto* param : params) {
            if (!param->has_grad()) {
                continue;
            }
            Tensor<float>& grad = const_cast<Tensor<float>&>(param->grad());
            for (size_t i = 0; i < grad.numel(); ++i) {
                grad.data()[i] *= clip_coef;
            }
        }
    }

    return total_norm;
}

// Clip gradients by value (modifies in-place)
// Clamps all gradient values to [-clip_value, clip_value]
inline void clip_grad_value_(
    std::vector<autograd::Variable*>& params,
    float clip_value) {

    for (auto* param : params) {
        if (!param->has_grad()) {
            continue;
        }
        Tensor<float>& grad = const_cast<Tensor<float>&>(param->grad());
        for (size_t i = 0; i < grad.numel(); ++i) {
            grad.data()[i] = std::clamp(grad.data()[i], -clip_value, clip_value);
        }
    }
}

}  // namespace optim
}  // namespace lightwatch
