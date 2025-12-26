// Phase 32: Model Config
// JSON-based model configuration with presets and validation

#pragma once

#include <lightwatch/models/gpt.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <fstream>
#include <stdexcept>

namespace lightwatch {

// Config validation error
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Model configuration container
struct ModelConfig {
    std::string model_type = "gpt2";
    models::GPT2Config gpt2;

    // Parse from JSON string
    static ModelConfig from_json(const std::string& json_str) {
        ModelConfig config;

        try {
            auto j = nlohmann::json::parse(json_str);

            config.model_type = j.value("model_type", "gpt2");

            if (config.model_type == "gpt2") {
                config.gpt2.vocab_size = j.value("vocab_size", 50257u);
                config.gpt2.max_seq_len = j.value("max_seq_len", 1024u);
                config.gpt2.embed_dim = j.value("embed_dim", 768u);
                config.gpt2.num_heads = j.value("num_heads", 12u);
                config.gpt2.num_layers = j.value("num_layers", 12u);
                config.gpt2.ffn_dim = j.value("ffn_dim", 3072u);
                config.gpt2.dropout_p = j.value("dropout_p", 0.1f);
                config.gpt2.attn_dropout_p = j.value("attn_dropout_p", 0.1f);
                config.gpt2.tie_weights = j.value("tie_weights", true);
            } else {
                throw ConfigError("Unknown model type: " + config.model_type);
            }
        } catch (const nlohmann::json::exception& e) {
            throw ConfigError(std::string("JSON parse error: ") + e.what());
        }

        return config;
    }

    // Serialize to JSON string
    std::string to_json() const {
        nlohmann::json j;

        j["model_type"] = model_type;
        j["vocab_size"] = gpt2.vocab_size;
        j["max_seq_len"] = gpt2.max_seq_len;
        j["embed_dim"] = gpt2.embed_dim;
        j["num_heads"] = gpt2.num_heads;
        j["num_layers"] = gpt2.num_layers;
        j["ffn_dim"] = gpt2.ffn_dim;
        j["dropout_p"] = gpt2.dropout_p;
        j["attn_dropout_p"] = gpt2.attn_dropout_p;
        j["tie_weights"] = gpt2.tie_weights;

        return j.dump(2);  // Pretty print with 2-space indent
    }

    // Load from file
    static ModelConfig load(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw ConfigError("Cannot open config file: " + path);
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());

        return from_json(content);
    }

    // Save to file
    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            throw ConfigError("Cannot write config file: " + path);
        }

        file << to_json();
    }

    // Load a named preset
    static ModelConfig preset(const std::string& name) {
        ModelConfig config;
        config.model_type = "gpt2";

        if (name == "gpt2-small" || name == "gpt2") {
            config.gpt2 = models::GPT2Config::gpt2_small();
        } else if (name == "gpt2-medium") {
            config.gpt2 = models::GPT2Config::gpt2_medium();
        } else if (name == "gpt2-large") {
            config.gpt2 = models::GPT2Config::gpt2_large();
        } else if (name == "gpt2-xl") {
            config.gpt2 = models::GPT2Config::gpt2_xl();
        } else {
            throw ConfigError("Unknown preset: " + name);
        }

        return config;
    }
};

// Validate configuration constraints
inline void validate_config(const ModelConfig& config) {
    if (config.model_type != "gpt2") {
        throw ConfigError("Unsupported model type: " + config.model_type);
    }

    const auto& c = config.gpt2;

    // Vocabulary must be positive
    if (c.vocab_size == 0) {
        throw ConfigError("vocab_size must be positive");
    }

    // Sequence length must be positive
    if (c.max_seq_len == 0) {
        throw ConfigError("max_seq_len must be positive");
    }

    // Embedding dimension must be positive
    if (c.embed_dim == 0) {
        throw ConfigError("embed_dim must be positive");
    }

    // Number of heads must be positive
    if (c.num_heads == 0) {
        throw ConfigError("num_heads must be positive");
    }

    // embed_dim must be divisible by num_heads
    if (c.embed_dim % c.num_heads != 0) {
        throw ConfigError("embed_dim (" + std::to_string(c.embed_dim) +
                         ") must be divisible by num_heads (" +
                         std::to_string(c.num_heads) + ")");
    }

    // Number of layers must be positive
    if (c.num_layers == 0) {
        throw ConfigError("num_layers must be positive");
    }

    // FFN dimension must be positive
    if (c.ffn_dim == 0) {
        throw ConfigError("ffn_dim must be positive");
    }

    // Dropout in valid range
    if (c.dropout_p < 0.0f || c.dropout_p > 1.0f) {
        throw ConfigError("dropout_p must be in [0, 1]");
    }

    if (c.attn_dropout_p < 0.0f || c.attn_dropout_p > 1.0f) {
        throw ConfigError("attn_dropout_p must be in [0, 1]");
    }
}

// Convenience functions matching spec
inline ModelConfig load_config(const std::string& path) {
    return ModelConfig::load(path);
}

inline void save_config(const ModelConfig& config, const std::string& path) {
    config.save(path);
}

}  // namespace lightwatch
