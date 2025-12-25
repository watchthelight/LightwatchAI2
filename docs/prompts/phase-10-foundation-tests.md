# Phase 10: Foundation Tests (Checkpoint)

## Objective
Comprehensive integration testing of Phases 1-9: tensor operations, autograd, tokenizer, and embeddings working together.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 01-09 | All foundation components |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 03 | include/lightwatch/tensor.hpp | Tensor<T> |
| 05 | include/lightwatch/autograd.hpp | Variable, ops |
| 06-07 | include/lightwatch/tokenizer/*.hpp | BPETokenizer, Vocabulary |
| 08 | include/lightwatch/nn/embedding.hpp | Embedding, GPTEmbedding |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| tests/integration/test_foundation.cpp | Integration tests | Phase 11+ |
| tests/benchmarks/bench_tensor.cpp | Performance baseline | Phase 39 |

## Specification

### Data Structures
N/A (test-only phase)

### Function Signatures
N/A (test-only phase)

### Algorithmic Requirements
1. **Tensor-Autograd integration**: Variables wrap tensors, gradients flow correctly
2. **Tokenizer-Embedding integration**: Token IDs map to embeddings
3. **Memory baseline**: Establish RSS usage for comparison
4. **Numerical stability**: Test edge cases (very small/large values)

### Performance Constraints
- All tests complete in < 60 seconds
- Memory usage < 50 MB for test suite
- No memory leaks (valgrind clean)

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_10_tensor_autograd_integration` | Create Variable, do ops, backward | Gradients correct |
| `test_phase_10_tokenizer_embedding` | "Hello" -> embed | Valid embedding tensor |
| `test_phase_10_gpt_embedding_forward` | Token IDs -> GPTEmbedding | Shape {batch, seq, 768} |
| `test_phase_10_memory_baseline` | Allocate/free tensors | RSS < 50MB |
| `test_phase_10_numerical_stability` | Very small values (1e-30) | No NaN/Inf |
| `test_phase_10_large_tensor` | 1M element tensor | Operations succeed |
| `test_phase_10_batch_tokenize_embed` | ["a", "bb", "ccc"] | Correct padded batch |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_10" --output-on-failure` exits 0
- [ ] `ctest -R "phase_0[1-9]" --output-on-failure` exits 0 (all foundation tests)
- [ ] Memory usage verified under 50MB
- [ ] No valgrind errors (if valgrind available)

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 500-800 |
| New source files | 5 |
| New test files | 3 |
| Complexity | MEDIUM |

## Notes
- This is CHECKPOINT 1 - do not proceed until all tests pass
- Run extended validation: `bash docs/references/scripts/checkpoint_10.sh`
- Document any performance observations in DECISIONS.md
- Memory baseline establishes reference for detecting regressions
- If valgrind not available (macOS ARM), skip leak check but note in TOOLCHAIN.md
