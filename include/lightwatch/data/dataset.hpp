// Phase 27: Dataset

#pragma once

#include <lightwatch/tensor.hpp>
#include <lightwatch/tokenizer/bpe.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdint>

namespace lightwatch {
namespace data {

// A single training sample
struct Sample {
    Tensor<int32_t> input_ids;      // {seq_len}
    Tensor<int32_t> labels;          // {seq_len} - shifted by 1 for LM
    Tensor<int32_t> attention_mask;  // {seq_len} - 1 for real, 0 for padding
};

// Base class for datasets
class Dataset {
public:
    virtual ~Dataset() = default;

    // Number of samples in the dataset
    virtual size_t size() const = 0;

    // Get a single sample by index
    virtual Sample get(size_t index) const = 0;
};

// Simple in-memory dataset from a list of token sequences
class TokenDataset : public Dataset {
public:
    TokenDataset(std::vector<std::vector<tokenizer::TokenId>> sequences,
                 size_t max_length = 1024)
        : max_length_(max_length) {
        // Process sequences into fixed-length samples
        for (auto& seq : sequences) {
            if (seq.size() >= 2) {  // Need at least 2 tokens for input + label
                // Truncate if needed
                if (seq.size() > max_length_) {
                    seq.resize(max_length_);
                }
                sequences_.push_back(std::move(seq));
            }
        }
    }

    size_t size() const override {
        return sequences_.size();
    }

    Sample get(size_t index) const override {
        if (index >= sequences_.size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        const auto& seq = sequences_[index];
        size_t seq_len = seq.size();

        // Create sample tensors
        Sample sample;
        sample.input_ids = Tensor<int32_t>({seq_len});
        sample.labels = Tensor<int32_t>({seq_len});
        sample.attention_mask = Tensor<int32_t>({seq_len});

        // For language modeling: input = tokens[:-1], labels = tokens[1:]
        // But we return full sequence, shifting is done during training
        for (size_t i = 0; i < seq_len; ++i) {
            sample.input_ids.data()[i] = static_cast<int32_t>(seq[i]);
            // Labels are shifted: label[i] = input[i+1]
            if (i < seq_len - 1) {
                sample.labels.data()[i] = static_cast<int32_t>(seq[i + 1]);
            } else {
                sample.labels.data()[i] = -100;  // Ignore last label
            }
            sample.attention_mask.data()[i] = 1;
        }

        return sample;
    }

private:
    std::vector<std::vector<tokenizer::TokenId>> sequences_;
    size_t max_length_;
};

// Dataset that loads and tokenizes a text file
class TextDataset : public Dataset {
public:
    TextDataset(const std::string& path,
                const tokenizer::BPETokenizer& tokenizer,
                size_t max_length = 1024)
        : max_length_(max_length) {

        // Read file
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open text file: " + path);
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string text = buffer.str();

        // Tokenize the entire text
        auto all_tokens = tokenizer.encode(text);

        // Chunk into sequences of max_length
        for (size_t i = 0; i + max_length_ <= all_tokens.size(); i += max_length_) {
            std::vector<tokenizer::TokenId> chunk(
                all_tokens.begin() + i,
                all_tokens.begin() + i + max_length_
            );
            sequences_.push_back(std::move(chunk));
        }

        // Handle remainder if it's long enough
        size_t remainder_start = (all_tokens.size() / max_length_) * max_length_;
        if (all_tokens.size() - remainder_start >= 2) {
            std::vector<tokenizer::TokenId> remainder(
                all_tokens.begin() + remainder_start,
                all_tokens.end()
            );
            sequences_.push_back(std::move(remainder));
        }
    }

    size_t size() const override {
        return sequences_.size();
    }

    Sample get(size_t index) const override {
        if (index >= sequences_.size()) {
            throw std::out_of_range("TextDataset index out of range");
        }

        const auto& seq = sequences_[index];
        size_t seq_len = seq.size();

        Sample sample;
        sample.input_ids = Tensor<int32_t>({seq_len});
        sample.labels = Tensor<int32_t>({seq_len});
        sample.attention_mask = Tensor<int32_t>({seq_len});

        for (size_t i = 0; i < seq_len; ++i) {
            sample.input_ids.data()[i] = static_cast<int32_t>(seq[i]);
            if (i < seq_len - 1) {
                sample.labels.data()[i] = static_cast<int32_t>(seq[i + 1]);
            } else {
                sample.labels.data()[i] = -100;
            }
            sample.attention_mask.data()[i] = 1;
        }

        return sample;
    }

private:
    std::vector<std::vector<tokenizer::TokenId>> sequences_;
    size_t max_length_;
};

}  // namespace data
}  // namespace lightwatch
