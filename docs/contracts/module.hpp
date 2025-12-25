// LightwatchAI2 API Contract: Module
// Defined by: Phase 11
// Consumers: 12-19, 31
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include "autograd.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <ostream>
#include <istream>

namespace lightwatch::nn {

class Module {
public:
    virtual ~Module() = default;

    // Forward pass - derived classes implement this
    virtual autograd::Variable forward(const autograd::Variable& input) = 0;

    // Multi-input forward (for attention, etc.)
    virtual autograd::Variable forward(
        const autograd::Variable& input,
        const autograd::Variable& other) {
        (void)other;
        return forward(input);
    }

    // Parameter access
    std::vector<autograd::Variable*> parameters();
    std::vector<std::pair<std::string, autograd::Variable*>> named_parameters();
    size_t num_parameters() const;

    // Submodule access
    std::vector<Module*> modules();
    std::vector<std::pair<std::string, Module*>> named_modules();

    // Training mode
    void train(bool mode = true);
    void eval();
    bool is_training() const;

    // Gradient control
    void zero_grad();
    void requires_grad_(bool requires_grad);

    // Serialization
    virtual void save_state(std::ostream& os) const;
    virtual void load_state(std::istream& is);

    // State dict (for HuggingFace compatibility)
    std::unordered_map<std::string, Tensor<float>> state_dict() const;
    void load_state_dict(const std::unordered_map<std::string, Tensor<float>>& dict);

protected:
    bool training_ = true;

    // Registration
    void register_parameter(const std::string& name, autograd::Variable& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);
    void register_buffer(const std::string& name, Tensor<float>& buffer);

private:
    std::vector<std::pair<std::string, autograd::Variable*>> parameters_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> submodules_;
    std::vector<std::pair<std::string, Tensor<float>*>> buffers_;
};

// Common layer types (signatures only - implementations in respective phases)

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;
    autograd::Variable bias;

private:
    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
};

class LayerNorm : public Module {
public:
    LayerNorm(size_t normalized_shape, float eps = 1e-5);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;
    autograd::Variable bias;

private:
    size_t normalized_shape_;
    float eps_;
};

class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim);
    autograd::Variable forward(const Tensor<int32_t>& indices);
    autograd::Variable forward(const autograd::Variable& input) override;

    autograd::Variable weight;

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
};

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1);
    autograd::Variable forward(const autograd::Variable& input) override;

private:
    float p_;
};

}  // namespace lightwatch::nn
