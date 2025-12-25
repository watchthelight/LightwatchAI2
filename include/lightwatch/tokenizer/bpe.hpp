// Phase 06: BPE Tokenizer
// GPT-2 compatible byte-level BPE tokenizer

#pragma once

#include <lightwatch/tokenizer/vocabulary.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <regex>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <limits>

namespace lightwatch::tokenizer {

class BPETokenizer {
public:
    BPETokenizer() {
        init_byte_encoder();
    }

    // Encode text to token IDs
    std::vector<TokenId> encode(const std::string& text) const {
        if (text.empty()) {
            return {};
        }

        std::vector<TokenId> result;

        // Split text into words using GPT-2 pattern
        std::vector<std::string> words = split_into_words(text);

        for (const auto& word : words) {
            // Convert to byte-level representation
            std::string byte_encoded = bytes_to_unicode(word);

            // Apply BPE
            std::vector<std::string> bpe_tokens = bpe(byte_encoded);

            // Convert to token IDs
            for (const auto& token : bpe_tokens) {
                TokenId id = vocab_.token_to_id(token);
                result.push_back(id);
            }
        }

        return result;
    }

    // Decode token IDs to text
    std::string decode(const std::vector<TokenId>& tokens) const {
        std::string byte_str;

        for (TokenId id : tokens) {
            std::string token = vocab_.id_to_token(id);
            byte_str += token;
        }

        // Convert byte-level representation back to bytes
        return unicode_to_bytes(byte_str);
    }

    // Batch operations
    std::vector<std::vector<TokenId>> encode_batch(
        const std::vector<std::string>& texts) const {
        std::vector<std::vector<TokenId>> results;
        results.reserve(texts.size());
        for (const auto& text : texts) {
            results.push_back(encode(text));
        }
        return results;
    }

    std::vector<std::string> decode_batch(
        const std::vector<std::vector<TokenId>>& token_batches) const {
        std::vector<std::string> results;
        results.reserve(token_batches.size());
        for (const auto& tokens : token_batches) {
            results.push_back(decode(tokens));
        }
        return results;
    }

    // Vocabulary access
    const Vocabulary& vocab() const { return vocab_; }
    size_t vocab_size() const { return vocab_.size(); }

    // Special tokens
    TokenId pad_id() const { return vocab_.pad_id(); }
    TokenId eos_id() const { return vocab_.eos_id(); }

    // Factory methods
    static BPETokenizer from_files(
        const std::string& vocab_path,
        const std::string& merges_path);

    static BPETokenizer gpt2(const std::string& vocab_dir = "data/vocab") {
        return from_files(
            vocab_dir + "/encoder.json",
            vocab_dir + "/vocab.bpe");
    }

    // Serialization
    void save(const std::string& path) const {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        // Save vocab size
        file << vocab_.size() << "\n";
        // Save tokens
        const auto& id_map = vocab_.id_map();
        for (size_t i = 0; i < id_map.size(); ++i) {
            file << id_map[i] << "\n";
        }
        // Save merges size
        file << merges_.size() << "\n";
        // Save merges
        for (const auto& merge : merges_) {
            file << merge.first << " " << merge.second << "\n";
        }
    }

    static BPETokenizer load(const std::string& path) {
        BPETokenizer tokenizer;
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + path);
        }

        size_t vocab_size;
        file >> vocab_size;
        file.ignore(); // Skip newline

        for (size_t i = 0; i < vocab_size; ++i) {
            std::string token;
            std::getline(file, token);
            tokenizer.vocab_.add_token(token);
        }

        size_t merges_size;
        file >> merges_size;
        file.ignore();

        for (size_t i = 0; i < merges_size; ++i) {
            std::string first, second;
            file >> first >> second;
            tokenizer.merges_.emplace_back(first, second);
            tokenizer.merge_ranks_[{first, second}] = static_cast<int>(i);
        }

        return tokenizer;
    }

private:
    Vocabulary vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;

    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>{}(p.first) ^
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> merge_ranks_;

    // Byte encoder/decoder for GPT-2's byte-level BPE
    std::unordered_map<uint8_t, std::string> byte_encoder_;
    std::unordered_map<std::string, uint8_t> byte_decoder_;

    void init_byte_encoder() {
        // GPT-2 maps bytes to unicode characters to avoid issues with
        // control characters and whitespace
        std::vector<int> bs;
        std::vector<int> cs;

        // Printable ASCII range (excluding space which goes to special char)
        for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
            bs.push_back(b);
            cs.push_back(b);
        }
        // Latin-1 supplement printable range
        for (int b = 0xA1; b <= 0xAC; ++b) {
            bs.push_back(b);
            cs.push_back(b);
        }
        for (int b = 0xAE; b <= 0xFF; ++b) {
            bs.push_back(b);
            cs.push_back(b);
        }

        // Map remaining bytes to unicode above 256
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool found = false;
            for (int x : bs) {
                if (x == b) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        // Create mappings
        for (size_t i = 0; i < bs.size(); ++i) {
            std::string unicode_char = encode_utf8(cs[i]);
            byte_encoder_[static_cast<uint8_t>(bs[i])] = unicode_char;
            byte_decoder_[unicode_char] = static_cast<uint8_t>(bs[i]);
        }
    }

    // Encode a unicode codepoint to UTF-8
    static std::string encode_utf8(int codepoint) {
        std::string result;
        if (codepoint < 0x80) {
            result += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            result += static_cast<char>(0xC0 | (codepoint >> 6));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            result += static_cast<char>(0xE0 | (codepoint >> 12));
            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else {
            result += static_cast<char>(0xF0 | (codepoint >> 18));
            result += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        return result;
    }

    // Convert bytes to GPT-2's unicode representation
    std::string bytes_to_unicode(const std::string& text) const {
        std::string result;
        for (unsigned char c : text) {
            auto it = byte_encoder_.find(c);
            if (it != byte_encoder_.end()) {
                result += it->second;
            }
        }
        return result;
    }

    // Convert GPT-2's unicode representation back to bytes
    std::string unicode_to_bytes(const std::string& text) const {
        std::string result;
        size_t i = 0;
        while (i < text.size()) {
            // Decode UTF-8 character
            size_t char_len = 1;
            unsigned char c = static_cast<unsigned char>(text[i]);
            if ((c & 0x80) == 0) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }

            std::string unicode_char = text.substr(i, char_len);
            auto it = byte_decoder_.find(unicode_char);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
            }
            i += char_len;
        }
        return result;
    }

    // Split text into words using GPT-2 pattern
    std::vector<std::string> split_into_words(const std::string& text) const {
        std::vector<std::string> words;

        // Simplified GPT-2 pattern: split on spaces and punctuation
        // but keep them attached appropriately
        std::string current;
        bool in_word = false;

        for (size_t i = 0; i < text.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(text[i]);

            // Handle UTF-8 multi-byte sequences
            size_t char_len = 1;
            if ((c & 0x80) == 0) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }

            std::string ch = text.substr(i, char_len);

            if (char_len == 1) {
                // ASCII character
                if (c == ' ') {
                    // Space starts a new word (space is prefix in GPT-2)
                    if (!current.empty()) {
                        words.push_back(current);
                        current.clear();
                    }
                    current = " ";
                    in_word = false;
                } else if (std::isalnum(c)) {
                    current += ch;
                    in_word = true;
                } else {
                    // Punctuation
                    if (in_word) {
                        // End current word
                        if (!current.empty()) {
                            words.push_back(current);
                            current.clear();
                        }
                    }
                    current += ch;
                    if (!current.empty()) {
                        words.push_back(current);
                        current.clear();
                    }
                    in_word = false;
                }
            } else {
                // Multi-byte UTF-8 character
                current += ch;
                in_word = true;
            }

            i += char_len - 1;  // Loop will increment by 1
        }

        if (!current.empty()) {
            words.push_back(current);
        }

        return words;
    }

    // Get all symbol pairs in a word
    std::vector<std::pair<std::string, std::string>> get_pairs(
        const std::vector<std::string>& word) const {
        std::vector<std::pair<std::string, std::string>> pairs;
        if (word.size() < 2) return pairs;

        for (size_t i = 0; i < word.size() - 1; ++i) {
            pairs.emplace_back(word[i], word[i + 1]);
        }
        return pairs;
    }

    // Apply BPE to a word
    std::vector<std::string> bpe(const std::string& token) const {
        if (token.empty()) return {};

        // Split token into individual UTF-8 characters
        std::vector<std::string> word;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = static_cast<unsigned char>(token[i]);
            size_t char_len = 1;
            if ((c & 0x80) == 0) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }
            word.push_back(token.substr(i, char_len));
            i += char_len;
        }

        if (word.size() == 1) {
            return word;
        }

        while (true) {
            auto pairs = get_pairs(word);
            if (pairs.empty()) break;

            // Find the pair with lowest merge rank
            std::pair<std::string, std::string> best_pair;
            int best_rank = std::numeric_limits<int>::max();

            for (const auto& pair : pairs) {
                auto it = merge_ranks_.find(pair);
                if (it != merge_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = pair;
                }
            }

            if (best_rank == std::numeric_limits<int>::max()) {
                // No more merges possible
                break;
            }

            // Apply the merge
            std::vector<std::string> new_word;
            size_t j = 0;
            while (j < word.size()) {
                if (j < word.size() - 1 &&
                    word[j] == best_pair.first &&
                    word[j + 1] == best_pair.second) {
                    new_word.push_back(best_pair.first + best_pair.second);
                    j += 2;
                } else {
                    new_word.push_back(word[j]);
                    j += 1;
                }
            }
            word = std::move(new_word);

            if (word.size() == 1) break;
        }

        return word;
    }
};

// Implementation of from_files
inline BPETokenizer BPETokenizer::from_files(
    const std::string& vocab_path,
    const std::string& merges_path) {

    BPETokenizer tokenizer;

    // Load vocabulary from encoder.json
    tokenizer.vocab_ = Vocabulary::from_encoder_json(vocab_path);

    // Load merges from vocab.bpe
    std::ifstream merges_file(merges_path);
    if (!merges_file) {
        throw std::runtime_error("Cannot open vocab.bpe: " + merges_path);
    }

    std::string line;
    // Skip header line
    std::getline(merges_file, line);

    int rank = 0;
    while (std::getline(merges_file, line)) {
        if (line.empty()) continue;

        // Split line into two tokens
        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) continue;

        std::string first = line.substr(0, space_pos);
        std::string second = line.substr(space_pos + 1);

        tokenizer.merges_.emplace_back(first, second);
        tokenizer.merge_ranks_[{first, second}] = rank++;
    }

    return tokenizer;
}

}  // namespace lightwatch::tokenizer
