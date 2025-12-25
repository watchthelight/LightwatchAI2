# Phase 20: Neural Core Tests (Checkpoint)

## Objective
Comprehensive integration testing of neural network layers (Phases 11-19) working together as a stack.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 11-19 | All neural network layer components |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 11 | include/lightwatch/nn/linear.hpp | Linear |
| 12 | include/lightwatch/nn/activations.hpp | GELU, Softmax |
| 13 | include/lightwatch/nn/normalization.hpp | LayerNorm |
| 14 | include/lightwatch/nn/dropout.hpp | Dropout |
| 16 | include/lightwatch/nn/attention.hpp | MultiHeadAttention |
| 17 | include/lightwatch/nn/ffn.hpp | FFN |
| 19 | include/lightwatch/nn/transformer.hpp | TransformerDecoderBlock |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| tests/integration/test_neural_core.cpp | Integration tests | Phase 31 |
| tests/benchmarks/bench_layers.cpp | Layer benchmarks | Phase 39 |

## Specification

### Data Structures
N/A (test-only phase)

### Function Signatures
N/A (test-only phase)

### Algorithmic Requirements
1. **MLP test**: Stack of Linear + GELU layers, verify forward/backward
2. **Gradient check**: Numerical gradient vs autograd gradient
3. **Attention test**: Verify causal masking prevents leakage
4. **Stack test**: Multiple decoder blocks stacked

### Performance Constraints
- All tests complete in < 120 seconds
- Memory usage stable (no leaks)
- Numerical precision: gradients match within 1e-5

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_20_mlp_stack` | 3-layer MLP forward/backward | Correct gradients |
| `test_phase_20_gradient_check_linear` | Numerical vs autograd | Match within 1e-5 |
| `test_phase_20_gradient_check_attention` | Numerical vs autograd | Match within 1e-5 |
| `test_phase_20_decoder_stack` | 2 decoder blocks | Output shape correct |
| `test_phase_20_causal_no_leak` | Attention weights | Upper triangle zero |
| `test_phase_20_dropout_training` | Training mode | Dropout active |
| `test_phase_20_dropout_eval` | Eval mode | No dropout |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_20" --output-on-failure` exits 0
- [ ] `ctest -R "phase_1[1-9]" --output-on-failure` exits 0
- [ ] Gradient check passes for all layers
- [ ] Causal attention verified no leakage

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 4 |
| New test files | 2 |
| Complexity | MEDIUM |

## Notes
- This is CHECKPOINT 2 - do not proceed until all tests pass
- Run extended validation: `bash docs/references/scripts/checkpoint_20.sh`
- Gradient check: (f(x+h) - f(x-h)) / (2h) ≈ df/dx
- Use h = 1e-5 for numerical gradients
- Document any numerical precision issues in DECISIONS.md
