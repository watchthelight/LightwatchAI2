// Phase 29: Training Loop

#pragma once

#include <lightwatch/nn/module.hpp>
#include <lightwatch/nn/loss.hpp>
#include <lightwatch/optim/adam.hpp>
#include <lightwatch/optim/scheduler.hpp>
#include <lightwatch/optim/clip.hpp>
#include <lightwatch/checkpoint.hpp>
#include <lightwatch/data/dataloader.hpp>
#include <lightwatch/data/collate.hpp>

#include <functional>
#include <memory>
#include <string>
#include <iostream>
#include <filesystem>

namespace lightwatch {
namespace training {

struct TrainingConfig {
    float learning_rate = 1e-4f;
    float weight_decay = 0.01f;
    float max_grad_norm = 1.0f;
    int warmup_steps = 100;
    int total_steps = -1;      // For scheduler, -1 = auto from epochs
    int log_interval = 10;
    int eval_interval = 100;
    int save_interval = 1000;
    std::string checkpoint_dir = "checkpoints";
    bool verbose = true;
};

struct TrainingState {
    int step = 0;
    int epoch = 0;
    float loss = 0.0f;
    float lr = 0.0f;
    float grad_norm = 0.0f;
    float avg_loss = 0.0f;  // Moving average
};

class Trainer {
public:
    Trainer(nn::Module& model,
            data::DataLoader& train_loader,
            TrainingConfig config = {})
        : model_(model)
        , train_loader_(train_loader)
        , config_(config)
        , loss_fn_(0.0f, -100, true)  // label_smoothing=0, ignore_index=-100, mean reduction
    {
        // Get model parameters
        auto params = model_.parameters();

        // Create optimizer
        optim::AdamOptions adam_opts;
        adam_opts.lr = config_.learning_rate;
        adam_opts.weight_decay = config_.weight_decay;
        optimizer_ = std::make_unique<optim::AdamW>(params, adam_opts);

        // Calculate total steps if not provided
        int total = config_.total_steps;
        if (total <= 0) {
            // Will be set during training based on epochs
            total = 10000;  // Default estimate
        }

        // Create warmup + cosine scheduler
        scheduler_ = std::make_unique<optim::ChainedScheduler>(
            *optimizer_,
            config_.warmup_steps,
            total,
            0.0f,  // start_factor
            0.0f   // eta_min
        );

        state_.lr = config_.learning_rate;
    }

    // Train for specified number of epochs
    void train(int num_epochs) {
        model_.train(true);

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            state_.epoch = epoch;
            train_epoch();

            if (on_epoch_end) {
                on_epoch_end(state_);
            }

            // Reset data loader for next epoch
            train_loader_.reset();
        }
    }

    // Train for specified number of steps
    void train_steps(int num_steps) {
        model_.train(true);

        int steps_done = 0;
        while (steps_done < num_steps) {
            for (auto samples : train_loader_) {
                if (steps_done >= num_steps) break;

                auto batch = data::collate_fn(samples);
                train_step(batch);
                ++steps_done;
            }

            if (steps_done < num_steps) {
                train_loader_.reset();
                ++state_.epoch;
            }
        }
    }

    // Single training step
    void train_step(const data::BatchEx& batch) {
        ++state_.step;

        // Zero gradients
        optimizer_->zero_grad();

        // Forward pass
        // Convert int32 input_ids to float Variable for model
        Tensor<float> input_float({batch.batch_size, batch.max_seq_len});
        for (size_t i = 0; i < batch.input_ids.numel(); ++i) {
            input_float.data()[i] = static_cast<float>(batch.input_ids.data()[i]);
        }
        autograd::Variable input(input_float, false);

        auto logits = model_.forward(input);

        // Compute loss
        // Note: loss_fn expects logits {batch, seq, vocab} and labels {batch, seq}
        // For now, we'll compute a simple loss if the model output matches expected shape
        auto loss = compute_loss(logits, batch.labels);

        // Backward pass
        if (loss.requires_grad()) {
            // Create gradient for loss (scalar = 1.0)
            Tensor<float> grad_out({1});
            grad_out.fill_(1.0f);
            loss.backward(grad_out);
        }

        // Gradient clipping
        auto params = model_.parameters();
        state_.grad_norm = optim::clip_grad_norm_(params, config_.max_grad_norm);

        // Optimizer step
        optimizer_->step();

        // Scheduler step
        scheduler_->step();
        state_.lr = scheduler_->get_last_lr();

        // Update state
        state_.loss = loss.data().data()[0];

        // Moving average
        if (state_.step == 1) {
            state_.avg_loss = state_.loss;
        } else {
            state_.avg_loss = 0.99f * state_.avg_loss + 0.01f * state_.loss;
        }

        // Logging
        if (config_.verbose && state_.step % config_.log_interval == 0) {
            log_step();
        }

        // Checkpointing
        if (config_.save_interval > 0 && state_.step % config_.save_interval == 0) {
            save_checkpoint();
        }

        // Callback
        if (on_step_end) {
            on_step_end(state_);
        }
    }

    // Get current state
    TrainingState state() const { return state_; }

    // Access optimizer
    optim::AdamW& optimizer() { return *optimizer_; }

    // Callbacks
    std::function<void(const TrainingState&)> on_step_end;
    std::function<void(const TrainingState&)> on_epoch_end;

private:
    void train_epoch() {
        for (auto samples : train_loader_) {
            auto batch = data::collate_fn(samples);
            train_step(batch);
        }
    }

    autograd::Variable compute_loss(const autograd::Variable& logits,
                                     const Tensor<int32_t>& labels) {
        // Simple MSE-like loss for testing if output doesn't match expected shape
        // Real implementation would use CrossEntropyLoss

        const auto& shape = logits.shape();

        // If logits shape is {batch, seq, vocab}, use cross entropy
        if (shape.size() == 3) {
            // Convert int32 labels to TokenId
            Tensor<tokenizer::TokenId> token_labels(labels.shape());
            for (size_t i = 0; i < labels.numel(); ++i) {
                token_labels.data()[i] = static_cast<tokenizer::TokenId>(labels.data()[i]);
            }
            return loss_fn_.forward(logits, token_labels);
        }

        // Fallback: simple sum loss for testing
        auto loss_tensor = logits.data().sum();
        return autograd::Variable(loss_tensor, logits.requires_grad());
    }

    void log_step() {
        std::cout << "Step " << state_.step
                  << " | Loss: " << state_.loss
                  << " | Avg Loss: " << state_.avg_loss
                  << " | LR: " << state_.lr
                  << " | Grad Norm: " << state_.grad_norm
                  << std::endl;
    }

    void save_checkpoint() {
        // Create checkpoint directory if needed
        std::filesystem::create_directories(config_.checkpoint_dir);

        std::string path = config_.checkpoint_dir + "/checkpoint_step_" +
                           std::to_string(state_.step) + ".ckpt";

        lightwatch::save_checkpoint(
            path,
            model_,
            *optimizer_,
            state_.epoch,
            state_.step,
            state_.loss
        );

        if (config_.verbose) {
            std::cout << "Saved checkpoint: " << path << std::endl;
        }
    }

    nn::Module& model_;
    data::DataLoader& train_loader_;
    TrainingConfig config_;
    std::unique_ptr<optim::AdamW> optimizer_;
    std::unique_ptr<optim::ChainedScheduler> scheduler_;
    nn::CrossEntropyLoss loss_fn_;
    TrainingState state_;
};

}  // namespace training
}  // namespace lightwatch
