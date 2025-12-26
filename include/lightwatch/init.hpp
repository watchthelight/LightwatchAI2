// Phase 33: Weight Initialization
// GPT-2 specific weight initialization scheme

#pragma once

#include <lightwatch/models/gpt.hpp>
#include <lightwatch/nn/linear.hpp>
#include <lightwatch/nn/embedding.hpp>
#include <lightwatch/nn/normalization.hpp>
#include <random>
#include <cmath>

namespace lightwatch {
namespace init {

// Initialization method enum
enum class InitMethod {
    NORMAL,      // N(0, std)
    XAVIER,      // Xavier/Glorot
    GPT2         // GPT-2 specific
};

// Initialization configuration
struct InitConfig {
    float std = 0.02f;           // Base std for embeddings
    float residual_std = 0.02f;  // For residual projections
    bool scale_by_depth = true;  // Scale residual by 1/sqrt(2*n_layers)
    unsigned int seed = 42;      // Random seed for reproducibility
};

// Random number generator with configurable seed
class RandomGen {
public:
    explicit RandomGen(unsigned int seed = 42) : gen_(seed) {}

    // Normal distribution
    void fill_normal(float* data, size_t size, float mean, float std) {
        std::normal_distribution<float> dist(mean, std);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(gen_);
        }
    }

    // Fill with constant value
    void fill_constant(float* data, size_t size, float value) {
        for (size_t i = 0; i < size; ++i) {
            data[i] = value;
        }
    }

    void reseed(unsigned int seed) {
        gen_.seed(seed);
    }

private:
    std::mt19937 gen_;
};

// Initialize a Linear layer
inline void init_linear(nn::Linear& layer, float std, RandomGen& rng) {
    // Weights: N(0, std)
    rng.fill_normal(layer.weight.data().data(), layer.weight.numel(), 0.0f, std);

    // Bias is already initialized to zeros by Linear constructor
    // No need to reinitialize
}

// Initialize a Linear layer (without RNG for convenience)
inline void init_linear(nn::Linear& layer, float std) {
    RandomGen rng;
    init_linear(layer, std, rng);
}

// Initialize an Embedding layer
inline void init_embedding(nn::Embedding& layer, float std, RandomGen& rng) {
    // Embeddings: N(0, std)
    rng.fill_normal(layer.weight.data().data(), layer.weight.numel(), 0.0f, std);
}

inline void init_embedding(nn::Embedding& layer, float std) {
    RandomGen rng;
    init_embedding(layer, std, rng);
}

// Initialize a LayerNorm layer
inline void init_layer_norm(nn::LayerNorm& layer, RandomGen& rng) {
    // Weight (gamma): 1
    rng.fill_constant(layer.weight.data().data(), layer.weight.numel(), 1.0f);

    // Bias (beta): 0
    rng.fill_constant(layer.bias.data().data(), layer.bias.numel(), 0.0f);
}

inline void init_layer_norm(nn::LayerNorm& layer) {
    RandomGen rng;
    init_layer_norm(layer, rng);
}

// Initialize all weights in GPT-2 model
inline void init_gpt2_weights(models::GPT2& model, InitConfig config = {}) {
    RandomGen rng(config.seed);

    const auto& cfg = model.config();
    float base_std = config.std;

    // Calculate residual std
    float residual_std = config.residual_std;
    if (config.scale_by_depth) {
        residual_std = config.std / std::sqrt(2.0f * static_cast<float>(cfg.num_layers));
    }

    // 1. Token embeddings: N(0, 0.02)
    rng.fill_normal(model.embedding.wte().weight.data().data(),
                    model.embedding.wte().weight.numel(),
                    0.0f, base_std);

    // 2. Position embeddings: N(0, 0.02)
    rng.fill_normal(model.embedding.wpe().weight.data().data(),
                    model.embedding.wpe().weight.numel(),
                    0.0f, base_std);

    // 3. Transformer layers
    for (auto& layer : model.layers) {
        // LayerNorm 1
        init_layer_norm(layer->ln1, rng);

        // Attention projections (Q, K, V, O)
        // Q, K, V: N(0, base_std)
        // O (output projection): N(0, residual_std) - residual connection
        init_linear(layer->attn.q_proj, base_std, rng);
        init_linear(layer->attn.k_proj, base_std, rng);
        init_linear(layer->attn.v_proj, base_std, rng);
        init_linear(layer->attn.out_proj, residual_std, rng);  // Residual projection

        // LayerNorm 2
        init_layer_norm(layer->ln2, rng);

        // FFN
        // fc1: N(0, base_std)
        // fc2: N(0, residual_std) - residual connection
        init_linear(layer->ffn.fc1, base_std, rng);
        init_linear(layer->ffn.fc2, residual_std, rng);  // Residual projection
    }

    // 4. Final LayerNorm
    init_layer_norm(model.ln_f, rng);

    // 5. LM head
    // If weight tying, copy from wte
    if (cfg.tie_weights) {
        auto& wte_weight = model.embedding.wte().weight;
        auto& lm_weight = model.lm_head.weight;
        for (size_t i = 0; i < lm_weight.numel(); ++i) {
            lm_weight.data().data()[i] = wte_weight.data().data()[i];
        }
    } else {
        init_linear(model.lm_head, base_std, rng);
    }
}

// Calculate weight statistics for verification
struct WeightStats {
    float mean;
    float std;
    size_t count;
};

inline WeightStats compute_stats(const float* data, size_t size) {
    if (size == 0) {
        return {0.0f, 0.0f, 0};
    }

    // Mean
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += data[i];
    }
    float mean = static_cast<float>(sum / static_cast<double>(size));

    // Std
    double sq_sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = data[i] - mean;
        sq_sum += diff * diff;
    }
    float std = static_cast<float>(std::sqrt(sq_sum / static_cast<double>(size)));

    return {mean, std, size};
}

}  // namespace init
}  // namespace lightwatch
