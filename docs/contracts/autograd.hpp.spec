// LightwatchAI2 API Contract: Autograd
// Defined by: Phase 05
// Consumers: 08, 11-19, 21-25, 31
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "tensor.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>

namespace lightwatch::autograd {

// Forward declarations
class Function;
class Variable;

class Variable {
public:
    Variable();
    explicit Variable(Tensor<float> data, bool requires_grad = false);

    // Access underlying tensor
    Tensor<float>& data();
    const Tensor<float>& data() const;

    // Gradient access
    Tensor<float>& grad();
    const Tensor<float>& grad() const;
    bool has_grad() const;
    bool requires_grad() const;
    void set_requires_grad(bool requires);
    void zero_grad();

    // Shape delegation
    const Shape& shape() const;
    size_t size(int dim) const;
    size_t numel() const;
    size_t ndim() const;

    // Backward pass
    void backward();
    void backward(const Tensor<float>& grad_output);

    // Computation graph
    void set_grad_fn(std::shared_ptr<Function> fn);
    std::shared_ptr<Function> grad_fn() const;

    // Detach from graph (returns new Variable with no grad_fn)
    Variable detach() const;

    // Retain gradient for non-leaf variables
    void retain_grad();

private:
    Tensor<float> data_;
    Tensor<float> grad_;
    bool requires_grad_ = false;
    bool has_grad_ = false;
    bool retain_grad_ = false;
    std::shared_ptr<Function> grad_fn_;
};

class Function {
public:
    virtual ~Function() = default;

    // Compute gradients given upstream gradient
    virtual std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) = 0;

    // Access saved tensors/variables
    const std::vector<Variable>& saved_variables() const;

protected:
    // Context for saving values needed in backward
    void save_for_backward(const Variable& v);
    void save_for_backward(const Tensor<float>& t);

    std::vector<Variable> saved_variables_;
    std::vector<Tensor<float>> saved_tensors_;
};

// Differentiable operations (return Variable with grad tracking)
namespace ops {
    // Arithmetic
    Variable add(const Variable& a, const Variable& b);
    Variable sub(const Variable& a, const Variable& b);
    Variable mul(const Variable& a, const Variable& b);  // Element-wise
    Variable div(const Variable& a, const Variable& b);
    Variable neg(const Variable& x);

    // Matrix operations
    Variable matmul(const Variable& a, const Variable& b);
    Variable transpose(const Variable& x, int dim0, int dim1);

    // Activations
    Variable relu(const Variable& x);
    Variable gelu(const Variable& x);
    Variable silu(const Variable& x);  // Swish
    Variable sigmoid(const Variable& x);
    Variable tanh(const Variable& x);
    Variable softmax(const Variable& x, int dim);
    Variable log_softmax(const Variable& x, int dim);

    // Reductions
    Variable sum(const Variable& x, int dim = -1, bool keepdim = false);
    Variable mean(const Variable& x, int dim = -1, bool keepdim = false);

    // Shape operations
    Variable reshape(const Variable& x, const Shape& new_shape);
    Variable squeeze(const Variable& x, int dim = -1);
    Variable unsqueeze(const Variable& x, int dim);

    // Indexing
    Variable slice(const Variable& x, int dim, size_t start, size_t end);
    Variable index_select(const Variable& x, int dim, const Tensor<int32_t>& indices);

    // Misc
    Variable dropout(const Variable& x, float p, bool training);
    Variable layer_norm(const Variable& x, const Variable& weight, const Variable& bias, float eps);
}

// No-grad context (RAII)
class NoGradGuard {
public:
    NoGradGuard();
    ~NoGradGuard();
private:
    static thread_local int guard_count_;
};

bool is_grad_enabled();

}  // namespace lightwatch::autograd
