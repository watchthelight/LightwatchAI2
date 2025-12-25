# Phase 29: Training Loop - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_training_overfit` | 10 samples, 1000 epochs | Loss < 0.01 |
| `test_training_gradient_clip` | Large gradients | Norm <= clip_value |
| `test_training_lr_schedule` | Warmup + cosine | LR follows schedule |
| `test_training_checkpoint` | Save/load mid-training | Training resumes correctly |
| `test_training_loss_decreases` | 100 steps | Loss at step 100 < loss at step 1 |
| `test_training_nan_detection` | Bad learning rate | NaN detected and reported |

## Implementation Notes

- Overfit test uses small dataset to verify learning capability
- Gradient clipping should clip by global norm, not per-parameter
- Checkpoint should save: model state, optimizer state, scheduler state, step count
- NaN detection should halt training and report which parameter caused it
