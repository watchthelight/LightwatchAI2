# Phase 29: Training Loop

## Objective
Implement the complete training loop with loss computation, backprop, optimization, and logging.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 21 | CrossEntropyLoss |
| 23 | Adam optimizer |
| 24 | LR schedulers |
| 25 | Gradient clipping |
| 26 | Checkpointing |
| 28 | Batch processing |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 21 | include/lightwatch/nn/loss.hpp | CrossEntropyLoss |
| 23 | include/lightwatch/optim/adam.hpp | AdamW |
| 24 | include/lightwatch/optim/scheduler.hpp | CosineAnnealingLR, WarmupLR |
| 25 | include/lightwatch/optim/clip.hpp | clip_grad_norm_ |
| 26 | include/lightwatch/checkpoint.hpp | save_checkpoint |
| 28 | include/lightwatch/data/collate.hpp | Batch |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/training/trainer.hpp | Trainer, TrainingConfig | Phase 30, 31 |
| src/training/trainer.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::training {

struct TrainingConfig {
    float learning_rate = 1e-4;
    float weight_decay = 0.01;
    float max_grad_norm = 1.0;
    int warmup_steps = 100;
    int max_steps = -1;      // -1 for unlimited
    int log_interval = 10;
    int eval_interval = 100;
    int save_interval = 1000;
    std::string checkpoint_dir = "checkpoints";
};

struct TrainingState {
    int step = 0;
    int epoch = 0;
    float loss = 0.0;
    float lr = 0.0;
    float grad_norm = 0.0;
};

class Trainer {
public:
    Trainer(nn::Module& model,
            data::DataLoader& train_loader,
            TrainingConfig config = {});

    void train(int num_epochs);
    void train_step(const data::Batch& batch);

    TrainingState state() const;

    // Callbacks
    std::function<void(const TrainingState&)> on_step_end;
    std::function<void(const TrainingState&)> on_epoch_end;

private:
    nn::Module& model_;
    data::DataLoader& train_loader_;
    TrainingConfig config_;
    std::unique_ptr<optim::AdamW> optimizer_;
    std::unique_ptr<optim::LRScheduler> scheduler_;
    nn::CrossEntropyLoss loss_fn_;
    TrainingState state_;
};

}  // namespace lightwatch::training
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Training step**:
   - Forward pass: logits = model(input_ids)
   - Loss: loss = loss_fn(logits, labels)
   - Backward: loss.backward()
   - Clip: clip_grad_norm_(params, max_grad_norm)
   - Update: optimizer.step(), scheduler.step()
   - Zero: optimizer.zero_grad()
2. **Epoch loop**: Iterate DataLoader, call train_step
3. **Logging**: Report loss, LR, grad_norm at intervals
4. **Checkpointing**: Save at save_interval

### Performance Constraints
- Overhead per step: < 1% of forward/backward time
- Checkpointing: Async if possible

## Required Tests
See `docs/test_specs/phase-29-training.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_29_train_step` | One batch | Loss computed, grads exist |
| `test_phase_29_overfit` | 10 samples, many steps | Loss < 0.01 |
| `test_phase_29_grad_clip` | Large gradients | Clipped to max_norm |
| `test_phase_29_lr_schedule` | After warmup | LR at peak |
| `test_phase_29_checkpoint` | After save_interval | Checkpoint file exists |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_29" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/training/trainer.hpp`
- [ ] Can overfit on small dataset

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 500-800 |
| New source files | 3 |
| New test files | 2 |
| Complexity | HIGH |

## Notes
- Overfit test is the key validation: loss -> 0 on tiny dataset
- Gradient clipping prevents explosion
- Warmup + cosine schedule is standard
- Callbacks enable custom logging/metrics
