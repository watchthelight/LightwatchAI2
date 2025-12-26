// Phase 12: Activation Functions
// Module wrappers for activation operations

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/autograd.hpp>

namespace lightwatch::nn {

// ReLU: max(0, x)
class ReLU : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::relu(input);
    }
};

// GELU: Gaussian Error Linear Unit
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
class GELU : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::gelu(input);
    }
};

// SiLU (Swish): x * sigmoid(x)
class SiLU : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::silu(input);
    }
};

// Sigmoid: 1 / (1 + exp(-x))
class Sigmoid : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::sigmoid(input);
    }
};

// Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
class Tanh : public Module {
public:
    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::tanh(input);
    }
};

// Softmax: exp(x - max) / sum(exp(x - max))
class Softmax : public Module {
public:
    explicit Softmax(int dim = -1) : dim_(dim) {}

    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::softmax(input, dim_);
    }

private:
    int dim_;
};

// LogSoftmax: log(softmax(x))
class LogSoftmax : public Module {
public:
    explicit LogSoftmax(int dim = -1) : dim_(dim) {}

    autograd::Variable forward(const autograd::Variable& input) override {
        return autograd::ops::log_softmax(input, dim_);
    }

private:
    int dim_;
};

}  // namespace lightwatch::nn
