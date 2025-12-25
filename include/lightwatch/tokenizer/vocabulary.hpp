// Phase 07: Vocabulary
// Token-ID bidirectional mapping with special token handling

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace lightwatch::tokenizer {

using TokenId = int32_t;

// GPT-2 special token IDs
struct SpecialTokens {
    static constexpr TokenId PAD = 50256;
    static constexpr TokenId UNK = 50256;
    static constexpr TokenId BOS = 50256;
    static constexpr TokenId EOS = 50256;
};

// Vocabulary class for bidirectional token-ID mapping
class Vocabulary {
public:
    Vocabulary() = default;

    // Add a token to the vocabulary, returns its ID
    TokenId add_token(const std::string& token) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            return it->second;
        }
        TokenId id = static_cast<TokenId>(id_to_token_.size());
        token_to_id_[token] = id;
        id_to_token_.push_back(token);
        return id;
    }

    // Convert token string to ID (returns UNK if not found)
    TokenId token_to_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        if (it == token_to_id_.end()) {
            return SpecialTokens::UNK;
        }
        return it->second;
    }

    // Convert ID to token string (returns empty string if invalid)
    std::string id_to_token(TokenId id) const {
        if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) {
            return "";
        }
        return id_to_token_[static_cast<size_t>(id)];
    }

    // Check if token string exists in vocabulary
    bool contains(const std::string& token) const {
        return token_to_id_.find(token) != token_to_id_.end();
    }

    // Check if token ID exists in vocabulary
    bool contains(TokenId id) const {
        return id >= 0 && static_cast<size_t>(id) < id_to_token_.size();
    }

    // Get vocabulary size
    size_t size() const {
        return id_to_token_.size();
    }

    // Special token accessors
    TokenId pad_id() const { return SpecialTokens::PAD; }
    TokenId eos_id() const { return SpecialTokens::EOS; }
    TokenId bos_id() const { return SpecialTokens::BOS; }
    TokenId unk_id() const { return SpecialTokens::UNK; }

    // Check if a token ID is a special token
    bool is_special_token(TokenId id) const {
        return id == SpecialTokens::EOS;
    }

    // Save vocabulary to file (simple line-by-line format)
    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        for (const auto& token : id_to_token_) {
            // Escape newlines in tokens
            std::string escaped;
            for (char c : token) {
                if (c == '\n') escaped += "\\n";
                else if (c == '\r') escaped += "\\r";
                else if (c == '\\') escaped += "\\\\";
                else escaped += c;
            }
            file << escaped << "\n";
        }
    }

    // Load vocabulary from file
    static Vocabulary load(const std::string& path) {
        Vocabulary vocab;
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + path);
        }
        std::string line;
        while (std::getline(file, line)) {
            // Unescape tokens
            std::string token;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '\\' && i + 1 < line.size()) {
                    char next = line[i + 1];
                    if (next == 'n') { token += '\n'; ++i; }
                    else if (next == 'r') { token += '\r'; ++i; }
                    else if (next == '\\') { token += '\\'; ++i; }
                    else token += line[i];
                } else {
                    token += line[i];
                }
            }
            vocab.add_token(token);
        }
        return vocab;
    }

    // Load from GPT-2 encoder.json format
    static Vocabulary from_encoder_json(const std::string& path) {
        Vocabulary vocab;
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open encoder.json: " + path);
        }

        // Read entire file
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());

        // Parse JSON: {"token": id, ...}
        std::vector<std::pair<std::string, TokenId>> entries;

        size_t pos = content.find('{');
        if (pos == std::string::npos) {
            throw std::runtime_error("Invalid encoder.json format");
        }
        pos++;

        while (pos < content.size()) {
            // Skip whitespace
            while (pos < content.size() && std::isspace(content[pos])) pos++;

            if (content[pos] == '}') break;

            // Parse key (token string)
            if (content[pos] != '"') {
                throw std::runtime_error("Expected '\"' in encoder.json");
            }
            pos++;

            std::string token;
            while (pos < content.size() && content[pos] != '"') {
                if (content[pos] == '\\' && pos + 1 < content.size()) {
                    pos++;
                    switch (content[pos]) {
                        case 'n': token += '\n'; break;
                        case 'r': token += '\r'; break;
                        case 't': token += '\t'; break;
                        case '\\': token += '\\'; break;
                        case '"': token += '"'; break;
                        case 'u': {
                            // Unicode escape \uXXXX
                            if (pos + 4 < content.size()) {
                                std::string hex = content.substr(pos + 1, 4);
                                int codepoint = std::stoi(hex, nullptr, 16);
                                // Encode as UTF-8
                                if (codepoint < 0x80) {
                                    token += static_cast<char>(codepoint);
                                } else if (codepoint < 0x800) {
                                    token += static_cast<char>(0xC0 | (codepoint >> 6));
                                    token += static_cast<char>(0x80 | (codepoint & 0x3F));
                                } else {
                                    token += static_cast<char>(0xE0 | (codepoint >> 12));
                                    token += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
                                    token += static_cast<char>(0x80 | (codepoint & 0x3F));
                                }
                                pos += 4;
                            }
                            break;
                        }
                        default: token += content[pos]; break;
                    }
                } else {
                    token += content[pos];
                }
                pos++;
            }
            pos++; // Skip closing quote

            // Skip colon
            while (pos < content.size() && (std::isspace(content[pos]) || content[pos] == ':')) pos++;

            // Parse value (token id)
            std::string num_str;
            while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '-')) {
                num_str += content[pos];
                pos++;
            }
            TokenId id = static_cast<TokenId>(std::stoi(num_str));

            entries.emplace_back(token, id);

            // Skip comma and whitespace
            while (pos < content.size() && (std::isspace(content[pos]) || content[pos] == ',')) pos++;
        }

        // Sort by id and add to vocabulary
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        for (const auto& entry : entries) {
            vocab.add_token(entry.first);
        }

        return vocab;
    }

    // Access to internal storage (for BPETokenizer)
    const std::unordered_map<std::string, TokenId>& token_map() const {
        return token_to_id_;
    }

    const std::vector<std::string>& id_map() const {
        return id_to_token_;
    }

private:
    std::unordered_map<std::string, TokenId> token_to_id_;
    std::vector<std::string> id_to_token_;
};

}  // namespace lightwatch::tokenizer
