// LightwatchAI2 API Contract: Tokenizer
// Defined by: Phases 06-07
// Consumers: 08, 27, 38
// DO NOT MODIFY without updating all consumer phases

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace lightwatch::tokenizer {

using TokenId = int32_t;

// GPT-2 special token IDs (from official tokenizer)
struct SpecialTokens {
    static constexpr TokenId PAD = 50256;  // Same as EOS for GPT-2
    static constexpr TokenId UNK = 50256;  // GPT-2 has no UNK, uses byte fallback
    static constexpr TokenId BOS = 50256;  // Not used in GPT-2
    static constexpr TokenId EOS = 50256;  // <|endoftext|>
};

class Vocabulary {
public:
    Vocabulary();

    // Token operations
    TokenId add_token(const std::string& token);
    TokenId token_to_id(const std::string& token) const;
    std::string id_to_token(TokenId id) const;

    bool contains(const std::string& token) const;
    bool contains(TokenId id) const;
    size_t size() const;

    // Special tokens
    TokenId pad_id() const;
    TokenId eos_id() const;
    bool is_special_token(TokenId id) const;

    // Serialization
    void save(const std::string& path) const;
    static Vocabulary load(const std::string& path);

    // Load from GPT-2 format
    static Vocabulary from_encoder_json(const std::string& path);

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;
};

class BPETokenizer {
public:
    BPETokenizer();

    // Encode text to token IDs
    std::vector<TokenId> encode(const std::string& text) const;

    // Decode token IDs to text
    std::string decode(const std::vector<TokenId>& tokens) const;

    // Batch operations
    std::vector<std::vector<TokenId>> encode_batch(
        const std::vector<std::string>& texts) const;
    std::vector<std::string> decode_batch(
        const std::vector<std::vector<TokenId>>& token_batches) const;

    // Vocabulary access
    const Vocabulary& vocab() const;
    size_t vocab_size() const;

    // Special tokens
    TokenId pad_id() const;
    TokenId eos_id() const;

    // Factory methods
    static BPETokenizer from_files(
        const std::string& vocab_path,    // encoder.json
        const std::string& merges_path);  // vocab.bpe

    static BPETokenizer gpt2(const std::string& vocab_dir = "data/vocab");

    // Serialization
    void save(const std::string& path) const;
    static BPETokenizer load(const std::string& path);

private:
    Vocabulary vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    // Hash function for string pairs
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;
};

}  // namespace lightwatch::tokenizer
