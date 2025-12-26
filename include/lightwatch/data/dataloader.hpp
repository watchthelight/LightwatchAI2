// Phase 27: DataLoader

#pragma once

#include <lightwatch/data/dataset.hpp>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>

namespace lightwatch {
namespace data {

// Collated batch of samples
struct Batch {
    Tensor<int32_t> input_ids;      // {batch_size, seq_len}
    Tensor<int32_t> labels;          // {batch_size, seq_len}
    Tensor<int32_t> attention_mask;  // {batch_size, seq_len}
};

// Collate samples into a batch (with padding)
inline Batch collate(const std::vector<Sample>& samples) {
    if (samples.empty()) {
        return {};
    }

    // Find max sequence length in this batch
    size_t max_len = 0;
    for (const auto& s : samples) {
        max_len = std::max(max_len, s.input_ids.numel());
    }

    size_t batch_size = samples.size();

    Batch batch;
    batch.input_ids = Tensor<int32_t>({batch_size, max_len});
    batch.labels = Tensor<int32_t>({batch_size, max_len});
    batch.attention_mask = Tensor<int32_t>({batch_size, max_len});

    // Initialize with padding values
    batch.input_ids.fill_(0);
    batch.labels.fill_(-100);  // Ignore index
    batch.attention_mask.fill_(0);

    // Copy each sample
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& s = samples[b];
        size_t seq_len = s.input_ids.numel();

        for (size_t i = 0; i < seq_len; ++i) {
            batch.input_ids.data()[b * max_len + i] = s.input_ids.data()[i];
            batch.labels.data()[b * max_len + i] = s.labels.data()[i];
            batch.attention_mask.data()[b * max_len + i] = s.attention_mask.data()[i];
        }
    }

    return batch;
}

class DataLoader {
public:
    DataLoader(Dataset& dataset,
               size_t batch_size,
               bool shuffle = true,
               unsigned seed = 42)
        : dataset_(dataset)
        , batch_size_(batch_size)
        , shuffle_(shuffle)
        , rng_(seed) {
        reset();
    }

    // Iterator for range-based for loops
    class Iterator {
    public:
        Iterator(DataLoader* loader, size_t batch_idx)
            : loader_(loader), batch_idx_(batch_idx) {}

        std::vector<Sample> operator*() const {
            std::vector<Sample> samples;
            size_t start = batch_idx_ * loader_->batch_size_;
            size_t end = std::min(start + loader_->batch_size_,
                                   loader_->indices_.size());

            for (size_t i = start; i < end; ++i) {
                samples.push_back(loader_->dataset_.get(loader_->indices_[i]));
            }
            return samples;
        }

        Iterator& operator++() {
            ++batch_idx_;
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return batch_idx_ != other.batch_idx_;
        }

    private:
        DataLoader* loader_;
        size_t batch_idx_;
    };

    Iterator begin() {
        return Iterator(this, 0);
    }

    Iterator end() {
        size_t num_batches = (indices_.size() + batch_size_ - 1) / batch_size_;
        return Iterator(this, num_batches);
    }

    // Reset and reshuffle for new epoch
    void reset() {
        indices_.resize(dataset_.size());
        std::iota(indices_.begin(), indices_.end(), 0);

        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    // Get number of batches
    size_t num_batches() const {
        return (indices_.size() + batch_size_ - 1) / batch_size_;
    }

    // Get batch size
    size_t batch_size() const {
        return batch_size_;
    }

    // Get dataset reference
    Dataset& dataset() {
        return dataset_;
    }

private:
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    std::mt19937 rng_;
    std::vector<size_t> indices_;
};

// Convenience function to iterate and collate
inline std::vector<Batch> get_all_batches(DataLoader& loader) {
    std::vector<Batch> batches;
    for (auto samples : loader) {
        batches.push_back(collate(samples));
    }
    return batches;
}

}  // namespace data
}  // namespace lightwatch
