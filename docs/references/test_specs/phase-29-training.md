<!-- File: docs/references/test_specs/phase-29-training.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 29: Training Loop - Test Specifications

**Complexity:** HIGH
**Minimum Tests Required:** 6

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_29_training_overfit` | 10 samples, 1000 epochs | Loss < 0.01 (overfit to training data) |
| `test_phase_29_training_loss_decreases` | 100 steps | `loss[100] < loss[0]` |
| `test_phase_29_training_gradient_accumulation` | accum_steps=4 | Effective batch size 4x larger |
| `test_phase_29_training_checkpoint_save` | After 50 steps | Checkpoint file exists, is valid |
| `test_phase_29_training_checkpoint_resume` | Save at step 50, resume | Training continues from step 50 |
| `test_phase_29_training_lr_schedule` | Warmup 100 steps | LR starts low, increases linearly |

## Implementation Notes

- Training loop should support gradient accumulation for larger effective batches
- Checkpoint should save: model state, optimizer state, step count, RNG state
- Loss should be averaged over accumulation steps before backward
