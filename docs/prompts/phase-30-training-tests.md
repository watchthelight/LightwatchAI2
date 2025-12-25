# Phase 30: Training Tests (Checkpoint)

## Objective
Comprehensive testing of the training infrastructure (Phases 21-29) with overfit tests and checkpoint validation.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 21-29 | All training infrastructure |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 21 | include/lightwatch/nn/loss.hpp | CrossEntropyLoss |
| 23 | include/lightwatch/optim/adam.hpp | AdamW |
| 29 | include/lightwatch/training/trainer.hpp | Trainer |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| tests/integration/test_training.cpp | Training integration tests | Phase 31 |
| tests/benchmarks/bench_training.cpp | Training benchmarks | Phase 39 |

## Specification

### Data Structures
N/A (test-only phase)

### Function Signatures
N/A (test-only phase)

### Algorithmic Requirements
1. **Overfit test**: Train on 10 samples until loss < 0.01
2. **Checkpoint roundtrip**: Save, load, verify state matches
3. **LR scheduler test**: Verify warmup + cosine pattern
4. **Gradient accumulation**: Test multiple backward() before step()

### Performance Constraints
- Overfit test completes in < 60 seconds
- All tests complete in < 180 seconds

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_30_overfit_mlp` | 10 samples, simple MLP | Loss < 0.01 |
| `test_phase_30_overfit_transformer` | 10 samples, 1 layer | Loss < 0.01 |
| `test_phase_30_checkpoint_roundtrip` | Save + load | States match exactly |
| `test_phase_30_checkpoint_resume` | Resume training | Loss continues decreasing |
| `test_phase_30_lr_warmup` | Warmup phase | LR increases linearly |
| `test_phase_30_lr_cosine` | After warmup | LR follows cosine |
| `test_phase_30_grad_accumulation` | 4 micro-batches | Equivalent to 1 large batch |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_30" --output-on-failure` exits 0
- [ ] `ctest -R "phase_2[1-9]" --output-on-failure` exits 0
- [ ] Overfit test achieves loss < 0.01
- [ ] Checkpoint roundtrip preserves all state

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 4 |
| New test files | 2 |
| Complexity | MEDIUM |

## Notes
- This is CHECKPOINT 3 - do not proceed until all tests pass
- Run extended validation: `bash docs/references/scripts/checkpoint_30.sh`
- Overfit is the gold standard for training correctness
- Checkpoint test ensures training can resume after crash
- Document any training issues in DECISIONS.md
