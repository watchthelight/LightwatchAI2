// Phase 34-35: Text Generation
// Greedy and sampling-based generation for GPT models

#pragma once

#include <lightwatch/models/gpt.hpp>
#include <lightwatch/autograd.hpp>
#include <vector>
#include <functional>
#include <algorithm>
#include <limits>
#include <random>
#include <cmath>
#include <numeric>

namespace lightwatch {

using TokenId = int32_t;

// Generation configuration
struct GenerateConfig {
    size_t max_new_tokens = 100;
    TokenId eos_token_id = 50256;  // GPT-2 end of text token
    bool early_stop = true;         // Stop at EOS
};

// Find argmax of a float array
inline size_t argmax(const float* data, size_t size) {
    if (size == 0) return 0;

    size_t max_idx = 0;
    float max_val = data[0];

    for (size_t i = 1; i < size; ++i) {
        if (data[i] > max_val) {
            max_val = data[i];
            max_idx = i;
        }
    }

    return max_idx;
}

// Greedy generation (argmax at each step)
inline std::vector<TokenId> generate_greedy(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    GenerateConfig config = {}) {

    if (prompt.empty()) {
        return {};
    }

    // Set model to eval mode (disables dropout)
    model.eval();

    // Copy prompt to output
    std::vector<TokenId> output = prompt;

    // Disable gradient computation for inference
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        // Check sequence length limit
        if (output.size() >= max_seq_len) {
            break;
        }

        // Create input tensor from current sequence
        Tensor<int32_t> input_ids({1, output.size()});
        for (size_t i = 0; i < output.size(); ++i) {
            input_ids.data()[i] = output[i];
        }

        // Forward pass
        auto logits = model.forward(input_ids);

        // Get logits for last position: shape is {1, seq_len, vocab_size}
        const auto& shape = logits.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        // Pointer to last position logits
        const float* last_logits = logits.data().data() + (seq_len - 1) * vocab_size;

        // Argmax to get next token
        size_t next_token_idx = argmax(last_logits, vocab_size);
        TokenId next_token = static_cast<TokenId>(next_token_idx);

        // Append token
        output.push_back(next_token);

        // Check for EOS
        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }

    return output;
}

// Streaming callback type
using TokenCallback = std::function<void(TokenId)>;

// Greedy generation with streaming callback
inline void generate_greedy_streaming(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    TokenCallback callback,
    GenerateConfig config = {}) {

    if (prompt.empty()) {
        return;
    }

    // Set model to eval mode
    model.eval();

    // Copy prompt to working sequence
    std::vector<TokenId> sequence = prompt;

    // Disable gradient computation
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        // Check sequence length limit
        if (sequence.size() >= max_seq_len) {
            break;
        }

        // Create input tensor
        Tensor<int32_t> input_ids({1, sequence.size()});
        for (size_t i = 0; i < sequence.size(); ++i) {
            input_ids.data()[i] = sequence[i];
        }

        // Forward pass
        auto logits = model.forward(input_ids);

        // Get logits for last position
        const auto& shape = logits.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];
        const float* last_logits = logits.data().data() + (seq_len - 1) * vocab_size;

        // Argmax
        size_t next_token_idx = argmax(last_logits, vocab_size);
        TokenId next_token = static_cast<TokenId>(next_token_idx);

        // Append token
        sequence.push_back(next_token);

        // Call callback
        callback(next_token);

        // Check for EOS
        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }
}

// Get just the generated tokens (excluding prompt)
inline std::vector<TokenId> get_generated_tokens(
    const std::vector<TokenId>& full_sequence,
    size_t prompt_length) {

    if (full_sequence.size() <= prompt_length) {
        return {};
    }

    return std::vector<TokenId>(
        full_sequence.begin() + static_cast<std::ptrdiff_t>(prompt_length),
        full_sequence.end()
    );
}

// ============================================================================
// Phase 35: Sampling-based generation
// ============================================================================

// Extended configuration for sampling
struct SamplingConfig : GenerateConfig {
    float temperature = 1.0f;    // Logit scaling (lower = more deterministic)
    int top_k = 0;               // 0 = disabled, keeps only top k tokens
    float top_p = 1.0f;          // 1.0 = disabled, nucleus sampling threshold
    bool do_sample = true;       // false = greedy
    unsigned int seed = 0;       // 0 = use random_device for seed
};

// Apply temperature scaling to logits
inline Tensor<float> apply_temperature(const Tensor<float>& logits, float temperature) {
    if (temperature <= 0.0f) {
        temperature = 1e-6f;  // Prevent division by zero
    }

    Tensor<float> result(logits.shape());
    const float* in = logits.data();
    float* out = result.data();

    for (size_t i = 0; i < logits.numel(); ++i) {
        out[i] = in[i] / temperature;
    }

    return result;
}

// Apply top-k filtering: keep only top k logits, set others to -inf
inline Tensor<float> apply_top_k(const Tensor<float>& logits, int k) {
    if (k <= 0) {
        return logits.clone();
    }

    size_t vocab_size = logits.numel();
    size_t k_size = static_cast<size_t>(k);
    if (k_size >= vocab_size) {
        return logits.clone();
    }

    // Create indices and copy logits
    std::vector<std::pair<float, size_t>> logit_pairs(vocab_size);
    const float* data = logits.data();
    for (size_t i = 0; i < vocab_size; ++i) {
        logit_pairs[i] = {data[i], i};
    }

    // Partial sort to find top-k threshold
    std::nth_element(
        logit_pairs.begin(),
        logit_pairs.begin() + static_cast<std::ptrdiff_t>(k_size),
        logit_pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; }
    );

    float threshold = logit_pairs[k_size - 1].first;

    // Create result with values below threshold set to -inf
    Tensor<float> result(logits.shape());
    float* out = result.data();
    for (size_t i = 0; i < vocab_size; ++i) {
        out[i] = (data[i] >= threshold) ? data[i] : -std::numeric_limits<float>::infinity();
    }

    return result;
}

// Apply top-p (nucleus) filtering: keep tokens with cumulative prob <= p
inline Tensor<float> apply_top_p(const Tensor<float>& logits, float p) {
    if (p >= 1.0f) {
        return logits.clone();
    }

    size_t vocab_size = logits.numel();
    const float* data = logits.data();

    // Compute softmax probabilities
    std::vector<float> probs(vocab_size);
    float max_logit = *std::max_element(data, data + vocab_size);

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(data[i] - max_logit);
        sum += probs[i];
    }
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Create sorted indices by probability (descending)
    std::vector<size_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&probs](size_t a, size_t b) { return probs[a] > probs[b]; });

    // Find cutoff index where cumulative prob exceeds p
    float cumsum = 0.0f;
    size_t cutoff = vocab_size;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;  // Keep at least one token beyond threshold
            break;
        }
    }

    // Create mask for tokens to keep
    std::vector<bool> keep(vocab_size, false);
    for (size_t i = 0; i < cutoff; ++i) {
        keep[indices[i]] = true;
    }

    // Create result with filtered logits
    Tensor<float> result(logits.shape());
    float* out = result.data();
    for (size_t i = 0; i < vocab_size; ++i) {
        out[i] = keep[i] ? data[i] : -std::numeric_limits<float>::infinity();
    }

    return result;
}

// Sample a token from probability distribution
inline TokenId sample_token(const Tensor<float>& logits, std::mt19937& rng) {
    size_t vocab_size = logits.numel();
    const float* data = logits.data();

    // Compute softmax probabilities
    float max_logit = *std::max_element(data, data + vocab_size);
    std::vector<float> probs(vocab_size);

    float sum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        float val = data[i] - max_logit;
        if (val > -30.0f) {  // Prevent underflow
            probs[i] = std::exp(val);
        } else {
            probs[i] = 0.0f;
        }
        sum += probs[i];
    }

    if (sum <= 0.0f) {
        // All logits are -inf, fall back to uniform random
        std::uniform_int_distribution<size_t> dist(0, vocab_size - 1);
        return static_cast<TokenId>(dist(rng));
    }

    // Normalize
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] /= sum;
    }

    // Sample using uniform random and cumulative distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return static_cast<TokenId>(i);
        }
    }

    // Edge case: return last token
    return static_cast<TokenId>(vocab_size - 1);
}

// Sample-based generation
inline std::vector<TokenId> generate_sample(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    SamplingConfig config = {}) {

    if (prompt.empty()) {
        return {};
    }

    // Initialize RNG
    std::mt19937 rng;
    if (config.seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(config.seed);
    }

    // Set model to eval mode
    model.eval();

    // Copy prompt to output
    std::vector<TokenId> output = prompt;

    // Disable gradient computation
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        // Check sequence length limit
        if (output.size() >= max_seq_len) {
            break;
        }

        // Create input tensor
        Tensor<int32_t> input_ids({1, output.size()});
        for (size_t i = 0; i < output.size(); ++i) {
            input_ids.data()[i] = output[i];
        }

        // Forward pass
        auto model_output = model.forward(input_ids);

        // Get logits for last position
        const auto& shape = model_output.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        // Extract last position logits into 1D tensor
        Tensor<float> logits({vocab_size});
        const float* last_logits = model_output.data().data() + (seq_len - 1) * vocab_size;
        std::copy(last_logits, last_logits + vocab_size, logits.data());

        TokenId next_token;

        if (!config.do_sample) {
            // Greedy decoding
            next_token = static_cast<TokenId>(argmax(logits.data(), vocab_size));
        } else {
            // Apply temperature
            if (config.temperature != 1.0f) {
                logits = apply_temperature(logits, config.temperature);
            }

            // Apply top-k
            if (config.top_k > 0) {
                logits = apply_top_k(logits, config.top_k);
            }

            // Apply top-p
            if (config.top_p < 1.0f) {
                logits = apply_top_p(logits, config.top_p);
            }

            // Sample token
            next_token = sample_token(logits, rng);
        }

        // Append token
        output.push_back(next_token);

        // Check for EOS
        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }

    return output;
}

// Streaming version of sample generation
inline void generate_sample_streaming(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    TokenCallback callback,
    SamplingConfig config = {}) {

    if (prompt.empty()) {
        return;
    }

    // Initialize RNG
    std::mt19937 rng;
    if (config.seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(config.seed);
    }

    // Set model to eval mode
    model.eval();

    std::vector<TokenId> sequence = prompt;

    // Disable gradient computation
    autograd::NoGradGuard no_grad;

    const auto& model_cfg = model.config();
    size_t max_seq_len = model_cfg.max_seq_len;

    for (size_t step = 0; step < config.max_new_tokens; ++step) {
        if (sequence.size() >= max_seq_len) {
            break;
        }

        Tensor<int32_t> input_ids({1, sequence.size()});
        for (size_t i = 0; i < sequence.size(); ++i) {
            input_ids.data()[i] = sequence[i];
        }

        auto model_output = model.forward(input_ids);
        const auto& shape = model_output.shape();
        size_t seq_len = shape[1];
        size_t vocab_size = shape[2];

        Tensor<float> logits({vocab_size});
        const float* last_logits = model_output.data().data() + (seq_len - 1) * vocab_size;
        std::copy(last_logits, last_logits + vocab_size, logits.data());

        TokenId next_token;

        if (!config.do_sample) {
            next_token = static_cast<TokenId>(argmax(logits.data(), vocab_size));
        } else {
            if (config.temperature != 1.0f) {
                logits = apply_temperature(logits, config.temperature);
            }
            if (config.top_k > 0) {
                logits = apply_top_k(logits, config.top_k);
            }
            if (config.top_p < 1.0f) {
                logits = apply_top_p(logits, config.top_p);
            }
            next_token = sample_token(logits, rng);
        }

        sequence.push_back(next_token);
        callback(next_token);

        if (config.early_stop && next_token == config.eos_token_id) {
            break;
        }
    }
}

}  // namespace lightwatch
