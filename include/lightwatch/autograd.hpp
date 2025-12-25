#pragma once

#include <lightwatch/tensor.hpp>
#include <lightwatch/simd/dispatch.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <random>

namespace lightwatch::autograd {

// Thread-local gradient tracking state
namespace detail {
    inline thread_local int no_grad_count = 0;
}

inline bool is_grad_enabled() {
    return detail::no_grad_count == 0;
}

// RAII guard for disabling gradient computation
class NoGradGuard {
public:
    NoGradGuard() { ++detail::no_grad_count; }
    ~NoGradGuard() { --detail::no_grad_count; }
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

// Forward declarations
class Function;
class Variable;

// Internal shared state for Variable
struct VariableData {
    Tensor<float> data;
    Tensor<float> grad;
    bool requires_grad = false;
    bool has_grad = false;
    bool retain_grad = false;
    std::shared_ptr<Function> grad_fn;

    VariableData() = default;
    explicit VariableData(Tensor<float> d, bool rg = false)
        : data(std::move(d)), requires_grad(rg) {}
};

// Base class for all backward functions
class Function : public std::enable_shared_from_this<Function> {
public:
    virtual ~Function() = default;

    virtual std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) = 0;

    // Store inputs for backward
    std::vector<std::shared_ptr<VariableData>> inputs;
    std::vector<Tensor<float>> saved_tensors;

    void save_tensor(const Tensor<float>& t) { saved_tensors.push_back(t); }
};

// Variable: Tensor wrapper with automatic differentiation
class Variable {
public:
    Variable() : impl_(std::make_shared<VariableData>()) {}

    explicit Variable(Tensor<float> data, bool requires_grad = false)
        : impl_(std::make_shared<VariableData>(std::move(data), requires_grad)) {}

    // Data access
    Tensor<float>& data() { return impl_->data; }
    const Tensor<float>& data() const { return impl_->data; }

    // Gradient access
    Tensor<float>& grad() { return impl_->grad; }
    const Tensor<float>& grad() const { return impl_->grad; }
    bool has_grad() const { return impl_->has_grad; }
    bool requires_grad() const { return impl_->requires_grad; }
    void set_requires_grad(bool requires) { impl_->requires_grad = requires; }

    void zero_grad() {
        if (impl_->has_grad) {
            impl_->grad.zero_();
        }
    }

    // Shape delegation
    const Shape& shape() const { return impl_->data.shape(); }
    size_t size(int dim) const { return impl_->data.size(dim); }
    size_t numel() const { return impl_->data.numel(); }
    size_t ndim() const { return impl_->data.ndim(); }

    // Backward pass
    void backward() {
        Tensor<float> grad_output = Tensor<float>::ones(impl_->data.shape());
        backward(grad_output);
    }

    void backward(const Tensor<float>& grad_output);

    // Computation graph
    void set_grad_fn(std::shared_ptr<Function> fn) { impl_->grad_fn = std::move(fn); }
    std::shared_ptr<Function> grad_fn() const { return impl_->grad_fn; }

    // Detach from computation graph
    Variable detach() const {
        return Variable(impl_->data.clone(), false);
    }

    // Retain gradient for non-leaf nodes
    void retain_grad() { impl_->retain_grad = true; }

    // Check if leaf (no grad_fn)
    bool is_leaf() const { return !impl_->grad_fn; }

    // Get internal impl (for backward)
    std::shared_ptr<VariableData> impl() const { return impl_; }

    // Accumulate gradient
    void accumulate_grad(const Tensor<float>& incoming_grad) {
        if (!impl_->requires_grad) return;
        if (!impl_->has_grad) {
            impl_->grad = Tensor<float>::zeros(impl_->data.shape());
            impl_->has_grad = true;
        }
        impl_->grad.add_(incoming_grad);
    }

private:
    std::shared_ptr<VariableData> impl_;
};

// Backward function implementations
class AddBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.clone(), grad_output.clone()};
    }
};

class SubBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.clone(), -grad_output};
    }
};

class MulBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& a = saved_tensors[0];
        const auto& b = saved_tensors[1];
        return {grad_output * b, grad_output * a};
    }
};

class DivBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& a = saved_tensors[0];
        const auto& b = saved_tensors[1];
        auto grad_a = grad_output / b;
        auto grad_b = -grad_output * a / (b * b);
        return {grad_a, grad_b};
    }
};

class NegBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {-grad_output};
    }
};

class MatmulBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& a = saved_tensors[0];
        const auto& b = saved_tensors[1];
        auto grad_a = matmul(grad_output, b.transpose(0, 1));
        auto grad_b = matmul(a.transpose(0, 1), grad_output);
        return {grad_a, grad_b};
    }
};

class TransposeBackward : public Function {
public:
    int dim0, dim1;
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.transpose(dim0, dim1)};
    }
};

class ReluBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& input = saved_tensors[0];
        Tensor<float> result(grad_output.shape());
        std::vector<size_t> indices(grad_output.ndim(), 0);
        for (size_t i = 0; i < grad_output.numel(); ++i) {
            result(indices) = (input(indices) > 0.0f) ? grad_output(indices) : 0.0f;
            for (int d = static_cast<int>(grad_output.ndim()) - 1; d >= 0; --d) {
                if (++indices[static_cast<size_t>(d)] < grad_output.shape()[static_cast<size_t>(d)]) break;
                indices[static_cast<size_t>(d)] = 0;
            }
        }
        return {result};
    }
};

class SigmoidBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& output = saved_tensors[0];
        auto grad = grad_output * output * (Tensor<float>::ones(output.shape()) - output);
        return {grad};
    }
};

class TanhBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& output = saved_tensors[0];
        auto grad = grad_output * (Tensor<float>::ones(output.shape()) - output * output);
        return {grad};
    }
};

class GeluBackward : public Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& input = saved_tensors[0];
        Tensor<float> grad(input.shape());
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        constexpr float coeff = 0.044715f;

        std::vector<size_t> indices(input.ndim(), 0);
        for (size_t i = 0; i < input.numel(); ++i) {
            float x = input(indices);
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x3);
            float tanh_val = std::tanh(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float inner_deriv = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x * x);
            float deriv = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * inner_deriv;
            grad(indices) = grad_output(indices) * deriv;
            for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
                if (++indices[static_cast<size_t>(d)] < input.shape()[static_cast<size_t>(d)]) break;
                indices[static_cast<size_t>(d)] = 0;
            }
        }
        return {grad};
    }
};

class SoftmaxBackward : public Function {
public:
    int dim;
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        const auto& output = saved_tensors[0];
        auto sum_term = (grad_output * output).sum(dim, true);
        auto grad = output * (grad_output - sum_term);
        return {grad};
    }
};

class SumBackward : public Function {
public:
    Shape input_shape;
    int dim;
    bool keepdim;

    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        Tensor<float> grad(input_shape);
        if (dim == -1) {
            float val = grad_output.numel() > 0 ? grad_output.item() : 0.0f;
            grad.fill_(val);
        } else {
            std::vector<size_t> indices(input_shape.size(), 0);
            for (size_t i = 0; i < grad.numel(); ++i) {
                std::vector<size_t> grad_indices;
                for (size_t j = 0; j < input_shape.size(); ++j) {
                    if (static_cast<int>(j) == dim) {
                        if (keepdim) grad_indices.push_back(0);
                    } else {
                        grad_indices.push_back(indices[j]);
                    }
                }
                grad(indices) = grad_output(grad_indices);
                for (int d = static_cast<int>(input_shape.size()) - 1; d >= 0; --d) {
                    if (++indices[static_cast<size_t>(d)] < input_shape[static_cast<size_t>(d)]) break;
                    indices[static_cast<size_t>(d)] = 0;
                }
            }
        }
        return {grad};
    }
};

class MeanBackward : public Function {
public:
    Shape input_shape;
    int dim;
    bool keepdim;
    size_t reduce_size;

    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        SumBackward sum_bwd;
        sum_bwd.input_shape = input_shape;
        sum_bwd.dim = dim;
        sum_bwd.keepdim = keepdim;
        auto grads = sum_bwd.backward(grad_output);
        return {grads[0] / static_cast<float>(reduce_size)};
    }
};

class ReshapeBackward : public Function {
public:
    Shape original_shape;
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.reshape(original_shape)};
    }
};

class SqueezeBackward : public Function {
public:
    Shape original_shape;
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.reshape(original_shape)};
    }
};

class UnsqueezeBackward : public Function {
public:
    int dim;
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output.squeeze(dim)};
    }
};

class SliceBackward : public Function {
public:
    Shape original_shape;
    int dim;
    size_t start;

    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        Tensor<float> grad = Tensor<float>::zeros(original_shape);
        std::vector<size_t> src_indices(grad_output.ndim(), 0);
        std::vector<size_t> dst_indices(original_shape.size(), 0);

        for (size_t i = 0; i < grad_output.numel(); ++i) {
            dst_indices = src_indices;
            dst_indices[static_cast<size_t>(dim)] += start;
            grad(dst_indices) = grad_output(src_indices);

            for (int d = static_cast<int>(grad_output.ndim()) - 1; d >= 0; --d) {
                if (++src_indices[static_cast<size_t>(d)] < grad_output.shape()[static_cast<size_t>(d)]) break;
                src_indices[static_cast<size_t>(d)] = 0;
            }
        }
        return {grad};
    }
};

class DropoutBackward : public Function {
public:
    Tensor<float> mask;
    float scale;

    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        return {grad_output * mask * scale};
    }
};

// Operations namespace
namespace ops {

inline Variable add(const Variable& a, const Variable& b) {
    Tensor<float> result = a.data() + b.data();
    Variable out(result, a.requires_grad() || b.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<AddBackward>();
        fn->inputs.push_back(a.impl());
        fn->inputs.push_back(b.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable sub(const Variable& a, const Variable& b) {
    Tensor<float> result = a.data() - b.data();
    Variable out(result, a.requires_grad() || b.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SubBackward>();
        fn->inputs.push_back(a.impl());
        fn->inputs.push_back(b.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable mul(const Variable& a, const Variable& b) {
    Tensor<float> result = a.data() * b.data();
    Variable out(result, a.requires_grad() || b.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<MulBackward>();
        fn->save_tensor(a.data());
        fn->save_tensor(b.data());
        fn->inputs.push_back(a.impl());
        fn->inputs.push_back(b.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable div(const Variable& a, const Variable& b) {
    Tensor<float> result = a.data() / b.data();
    Variable out(result, a.requires_grad() || b.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<DivBackward>();
        fn->save_tensor(a.data());
        fn->save_tensor(b.data());
        fn->inputs.push_back(a.impl());
        fn->inputs.push_back(b.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable neg(const Variable& x) {
    Tensor<float> result = -x.data();
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<NegBackward>();
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable matmul(const Variable& a, const Variable& b) {
    Tensor<float> result = lightwatch::matmul(a.data(), b.data());
    Variable out(result, a.requires_grad() || b.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<MatmulBackward>();
        fn->save_tensor(a.data());
        fn->save_tensor(b.data());
        fn->inputs.push_back(a.impl());
        fn->inputs.push_back(b.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable transpose(const Variable& x, int dim0, int dim1) {
    Tensor<float> result = x.data().transpose(dim0, dim1);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<TransposeBackward>();
        fn->dim0 = dim0;
        fn->dim1 = dim1;
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable relu(const Variable& x) {
    Tensor<float> result(x.data().shape());
    const auto& input = x.data();
    std::vector<size_t> indices(input.ndim(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        float val = input(indices);
        result(indices) = val > 0.0f ? val : 0.0f;
        for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < input.shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<ReluBackward>();
        fn->save_tensor(x.data());
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable sigmoid(const Variable& x) {
    Tensor<float> result(x.data().shape());
    const auto& input = x.data();
    std::vector<size_t> indices(input.ndim(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        result(indices) = 1.0f / (1.0f + std::exp(-input(indices)));
        for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < input.shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SigmoidBackward>();
        fn->save_tensor(result);
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable tanh(const Variable& x) {
    Tensor<float> result(x.data().shape());
    const auto& input = x.data();
    std::vector<size_t> indices(input.ndim(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        result(indices) = std::tanh(input(indices));
        for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < input.shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<TanhBackward>();
        fn->save_tensor(result);
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable gelu(const Variable& x) {
    Tensor<float> result(x.data().shape());
    const auto& input = x.data();
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float coeff = 0.044715f;

    std::vector<size_t> indices(input.ndim(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        float val = input(indices);
        float x3 = val * val * val;
        float inner = sqrt_2_over_pi * (val + coeff * x3);
        result(indices) = 0.5f * val * (1.0f + std::tanh(inner));
        for (int d = static_cast<int>(input.ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < input.shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<GeluBackward>();
        fn->save_tensor(x.data());
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable silu(const Variable& x) {
    return mul(x, sigmoid(x));
}

inline Variable softmax(const Variable& x, int dim) {
    const auto& input = x.data();
    int d = dim;
    if (d < 0) d += static_cast<int>(input.ndim());

    auto max_val = input.max(dim, true);
    auto shifted = input - max_val;

    Tensor<float> exp_result(input.shape());
    std::vector<size_t> indices(input.ndim(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        exp_result(indices) = std::exp(shifted(indices));
        for (int id = static_cast<int>(input.ndim()) - 1; id >= 0; --id) {
            if (++indices[static_cast<size_t>(id)] < input.shape()[static_cast<size_t>(id)]) break;
            indices[static_cast<size_t>(id)] = 0;
        }
    }

    auto sum_exp = exp_result.sum(dim, true);
    Tensor<float> result = exp_result / sum_exp;
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SoftmaxBackward>();
        fn->dim = d;
        fn->save_tensor(result);
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable log_softmax(const Variable& x, int dim) {
    auto sm = softmax(x, dim);
    Tensor<float> result(sm.data().shape());
    std::vector<size_t> indices(sm.data().ndim(), 0);
    for (size_t i = 0; i < sm.data().numel(); ++i) {
        result(indices) = std::log(sm.data()(indices));
        for (int d = static_cast<int>(sm.data().ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < sm.data().shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable out(result, x.requires_grad());
    if (is_grad_enabled() && out.requires_grad()) {
        out.set_grad_fn(sm.grad_fn());
    }
    return out;
}

inline Variable sum(const Variable& x, int dim = -1, bool keepdim = false) {
    Tensor<float> result = x.data().sum(dim, keepdim);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SumBackward>();
        fn->input_shape = x.data().shape();
        fn->dim = dim;
        fn->keepdim = keepdim;
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable mean(const Variable& x, int dim = -1, bool keepdim = false) {
    Tensor<float> result = x.data().mean(dim, keepdim);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<MeanBackward>();
        fn->input_shape = x.data().shape();
        fn->dim = dim;
        fn->keepdim = keepdim;
        fn->reduce_size = (dim == -1) ? x.numel() : x.size(dim);
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable reshape(const Variable& x, const Shape& new_shape) {
    Tensor<float> result = x.data().reshape(new_shape);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<ReshapeBackward>();
        fn->original_shape = x.data().shape();
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable squeeze(const Variable& x, int dim = -1) {
    Tensor<float> result = x.data().squeeze(dim);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SqueezeBackward>();
        fn->original_shape = x.data().shape();
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable unsqueeze(const Variable& x, int dim) {
    Tensor<float> result = x.data().unsqueeze(dim);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<UnsqueezeBackward>();
        fn->dim = dim;
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable slice(const Variable& x, int dim, size_t start, size_t end) {
    Tensor<float> result = x.data().slice(dim, start, end);
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<SliceBackward>();
        fn->original_shape = x.data().shape();
        fn->dim = dim;
        fn->start = start;
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable index_select(const Variable& x, int dim, const Tensor<int32_t>& indices) {
    return x;
}

inline Variable dropout(const Variable& x, float p, bool training) {
    if (!training || p == 0.0f) {
        return Variable(x.data().clone(), x.requires_grad());
    }

    Tensor<float> mask(x.data().shape());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float scale = 1.0f / (1.0f - p);

    std::vector<size_t> indices(x.data().ndim(), 0);
    for (size_t i = 0; i < x.data().numel(); ++i) {
        mask(indices) = (dist(lightwatch::detail::get_rng()) > p) ? 1.0f : 0.0f;
        for (int d = static_cast<int>(x.data().ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < x.data().shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }

    Tensor<float> result = x.data() * mask * scale;
    Variable out(result, x.requires_grad());

    if (is_grad_enabled() && out.requires_grad()) {
        auto fn = std::make_shared<DropoutBackward>();
        fn->mask = mask;
        fn->scale = scale;
        fn->inputs.push_back(x.impl());
        out.set_grad_fn(fn);
    }
    return out;
}

inline Variable layer_norm(const Variable& x, const Variable& weight, const Variable& bias, float eps) {
    int dim = static_cast<int>(x.ndim()) - 1;
    auto m = mean(x, dim, true);
    auto centered = sub(x, m);
    auto var = mean(mul(centered, centered), dim, true);

    Tensor<float> std_tensor(var.data().shape());
    std::vector<size_t> indices(var.data().ndim(), 0);
    for (size_t i = 0; i < var.data().numel(); ++i) {
        std_tensor(indices) = std::sqrt(var.data()(indices) + eps);
        for (int d = static_cast<int>(var.data().ndim()) - 1; d >= 0; --d) {
            if (++indices[static_cast<size_t>(d)] < var.data().shape()[static_cast<size_t>(d)]) break;
            indices[static_cast<size_t>(d)] = 0;
        }
    }
    Variable std_var(std_tensor, var.requires_grad());

    auto normalized = div(centered, std_var);
    auto scaled = mul(normalized, weight);
    return add(scaled, bias);
}

}  // namespace ops

// Backward implementation - propagates through the graph
inline void Variable::backward(const Tensor<float>& grad_output) {
    if (!impl_->requires_grad) return;

    // Accumulate gradient at output node
    accumulate_grad(grad_output);

    if (!impl_->grad_fn) return;

    // BFS through the graph
    std::vector<std::pair<std::shared_ptr<Function>, Tensor<float>>> work_queue;
    work_queue.emplace_back(impl_->grad_fn, grad_output);

    while (!work_queue.empty()) {
        auto item = work_queue.back();
        work_queue.pop_back();

        auto fn = item.first;
        auto grad = item.second;

        auto input_grads = fn->backward(grad);

        // Propagate to inputs
        for (size_t i = 0; i < fn->inputs.size() && i < input_grads.size(); ++i) {
            auto& inp = fn->inputs[i];
            if (inp->requires_grad) {
                // Accumulate gradient
                if (!inp->has_grad) {
                    inp->grad = Tensor<float>::zeros(inp->data.shape());
                    inp->has_grad = true;
                }
                inp->grad.add_(input_grads[i]);

                // Continue backward if not leaf
                if (inp->grad_fn) {
                    work_queue.emplace_back(inp->grad_fn, input_grads[i]);
                }
            }
        }
    }
}

}  // namespace lightwatch::autograd
