// Phase 08: Embedding Layer
// Token embedding lookup with combined token and position embeddings

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/tensor.hpp>
#include <lightwatch/autograd.hpp>
#include <cstdint>
#include <stdexcept>
#include <cmath>

namespace lightwatch::nn {

// Embedding lookup backward function
class EmbeddingBackward : public autograd::Function {
public:
    std::vector<Tensor<float>> backward(const Tensor<float>& grad_output) override {
        // Scatter gradients back to weight matrix
        size_t num_embeddings = saved_tensors[0].shape()[0];
        size_t embedding_dim = saved_tensors[0].shape()[1];

        // Get indices from saved data
        const Tensor<float>& indices_float = saved_tensors[1];

        // Create gradient for weight
        Tensor<float> grad_weight = Tensor<float>::zeros({num_embeddings, embedding_dim});

        // grad_output shape: {*, embedding_dim} where * matches indices shape
        // Scatter add gradients
        size_t total_indices = indices_float.numel();

        for (size_t i = 0; i < total_indices; ++i) {
            size_t idx = static_cast<size_t>(indices_float.data()[i]);
            for (size_t j = 0; j < embedding_dim; ++j) {
                grad_weight.data()[idx * embedding_dim + j] +=
                    grad_output.data()[i * embedding_dim + j];
            }
        }

        return {grad_weight};
    }
};

// Embedding lookup layer
class Embedding : public Module {
public:
    Embedding(size_t num_embeddings, size_t embedding_dim)
        : num_embeddings_(num_embeddings)
        , embedding_dim_(embedding_dim) {

        // Initialize weight with normal distribution
        // Standard initialization: N(0, 1)
        weight = autograd::Variable(
            Tensor<float>::randn({num_embeddings, embedding_dim}),
            true);

        register_parameter("weight", weight);
    }

    // Lookup by int32 indices tensor
    autograd::Variable forward(const Tensor<int32_t>& indices) {
        // Validate indices
        for (size_t i = 0; i < indices.numel(); ++i) {
            int32_t idx = indices.data()[i];
            if (idx < 0 || static_cast<size_t>(idx) >= num_embeddings_) {
                throw std::out_of_range(
                    "Embedding index out of range: " + std::to_string(idx) +
                    " not in [0, " + std::to_string(num_embeddings_) + ")");
            }
        }

        // Compute output shape: indices_shape + {embedding_dim}
        std::vector<size_t> output_shape = indices.shape();
        output_shape.push_back(embedding_dim_);

        // Perform lookup
        Tensor<float> output(output_shape);
        size_t total_indices = indices.numel();

        for (size_t i = 0; i < total_indices; ++i) {
            size_t idx = static_cast<size_t>(indices.data()[i]);
            // Copy embedding vector
            for (size_t j = 0; j < embedding_dim_; ++j) {
                output.data()[i * embedding_dim_ + j] =
                    weight.data()({idx, j});
            }
        }

        // Create output variable with gradient tracking
        if (weight.requires_grad() && autograd::is_grad_enabled()) {
            auto result = autograd::Variable(output, true);

            // Create backward function
            auto backward_fn = std::make_shared<EmbeddingBackward>();
            backward_fn->inputs.push_back(weight.impl());
            backward_fn->save_tensor(weight.data());

            // Save indices as float tensor for backward
            Tensor<float> indices_float({total_indices});
            for (size_t i = 0; i < total_indices; ++i) {
                indices_float.data()[i] = static_cast<float>(indices.data()[i]);
            }
            backward_fn->save_tensor(indices_float);

            result.impl()->grad_fn = backward_fn;
            return result;
        }

        return autograd::Variable(output, false);
    }

    // Forward with Variable input (converts to int32 indices)
    autograd::Variable forward(const autograd::Variable& input) override {
        // Convert float tensor to int32 indices
        const Tensor<float>& input_data = input.data();
        Tensor<int32_t> indices(input_data.shape());

        for (size_t i = 0; i < input_data.numel(); ++i) {
            indices.data()[i] = static_cast<int32_t>(input_data.data()[i]);
        }

        return forward(indices);
    }

    size_t num_embeddings() const { return num_embeddings_; }
    size_t embedding_dim() const { return embedding_dim_; }

    autograd::Variable weight;

private:
    size_t num_embeddings_;
    size_t embedding_dim_;
};

// Combined token + position embedding for GPT
class GPTEmbedding : public Module {
public:
    GPTEmbedding(size_t vocab_size, size_t max_seq_len, size_t embed_dim)
        : wte_(std::make_shared<Embedding>(vocab_size, embed_dim))
        , wpe_(std::make_shared<Embedding>(max_seq_len, embed_dim))
        , max_seq_len_(max_seq_len)
        , embed_dim_(embed_dim) {

        // Initialize token embeddings with smaller std
        float std = 0.02f;
        for (size_t i = 0; i < wte_->weight.numel(); ++i) {
            wte_->weight.data().data()[i] *= std;
        }

        // Initialize position embeddings with smaller std
        for (size_t i = 0; i < wpe_->weight.numel(); ++i) {
            wpe_->weight.data().data()[i] *= std;
        }

        register_module("wte", wte_);
        register_module("wpe", wpe_);
    }

    // Forward with int32 token IDs
    autograd::Variable forward(const Tensor<int32_t>& token_ids) {
        // Get sequence length
        size_t seq_len = token_ids.shape().back();

        if (seq_len > max_seq_len_) {
            throw std::runtime_error(
                "Sequence length " + std::to_string(seq_len) +
                " exceeds max " + std::to_string(max_seq_len_));
        }

        // Get token embeddings: {batch, seq_len, embed_dim}
        auto token_embeds = wte_->forward(token_ids);

        // Create position indices: 0, 1, 2, ..., seq_len-1
        Tensor<int32_t> positions({seq_len});
        for (size_t i = 0; i < seq_len; ++i) {
            positions.data()[i] = static_cast<int32_t>(i);
        }

        // Get position embeddings: {seq_len, embed_dim}
        auto pos_embeds = wpe_->forward(positions);

        // Add position embeddings to token embeddings (broadcast)
        Tensor<float> result = token_embeds.data().clone();

        // Handle different input shapes
        std::vector<size_t> token_shape = token_ids.shape();
        if (token_shape.size() == 1) {
            // Single sequence: {seq_len}
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < embed_dim_; ++j) {
                    result.data()[i * embed_dim_ + j] +=
                        pos_embeds.data()({i, j});
                }
            }
        } else if (token_shape.size() == 2) {
            // Batch: {batch, seq_len}
            size_t batch_size = token_shape[0];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < embed_dim_; ++j) {
                        size_t idx = (b * seq_len + i) * embed_dim_ + j;
                        result.data()[idx] += pos_embeds.data()({i, j});
                    }
                }
            }
        }

        // Create result variable with gradient function
        if (token_embeds.requires_grad() && autograd::is_grad_enabled()) {
            auto output = autograd::Variable(result, true);

            // For gradient flow, we need to add both backwards
            // This is simplified - proper implementation would chain backwards
            auto add_backward = std::make_shared<autograd::AddBackward>();
            add_backward->inputs.push_back(token_embeds.impl());
            add_backward->inputs.push_back(pos_embeds.impl());
            output.impl()->grad_fn = add_backward;

            return output;
        }

        return autograd::Variable(result, false);
    }

    // Forward with Variable input
    autograd::Variable forward(const autograd::Variable& input) override {
        // Convert to int32 indices
        const Tensor<float>& input_data = input.data();
        Tensor<int32_t> token_ids(input_data.shape());

        for (size_t i = 0; i < input_data.numel(); ++i) {
            token_ids.data()[i] = static_cast<int32_t>(input_data.data()[i]);
        }

        return forward(token_ids);
    }

    // Access to embeddings
    Embedding& wte() { return *wte_; }
    Embedding& wpe() { return *wpe_; }
    const Embedding& wte() const { return *wte_; }
    const Embedding& wpe() const { return *wpe_; }

private:
    std::shared_ptr<Embedding> wte_;  // Token embeddings
    std::shared_ptr<Embedding> wpe_;  // Position embeddings
    size_t max_seq_len_;
    size_t embed_dim_;
};

}  // namespace lightwatch::nn
