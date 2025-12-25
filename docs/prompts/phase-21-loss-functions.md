# Phase 21: Loss Functions

## Objective
Implement cross-entropy loss and related functions for language model training.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 12 | Softmax, log_softmax |
| 13 | LayerNorm (for numerical stability patterns) |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 05 | include/lightwatch/autograd.hpp | Variable, ops |
| 12 | include/lightwatch/nn/activations.hpp | log_softmax |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/nn/loss.hpp | CrossEntropyLoss, NLLLoss | Phase 29, 31 |
| src/nn/loss.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch::nn {

class CrossEntropyLoss : public Module {
public:
    CrossEntropyLoss(
        float label_smoothing = 0.0,
        TokenId ignore_index = -100,
        bool reduction_mean = true);

    // logits: {batch, seq, vocab}, targets: {batch, seq}
    autograd::Variable forward(
        const autograd::Variable& logits,
        const Tensor<int32_t>& targets);

    autograd::Variable forward(const autograd::Variable& input) override;

private:
    float label_smoothing_;
    TokenId ignore_index_;
    bool reduction_mean_;
};

class NLLLoss : public Module {
public:
    NLLLoss(TokenId ignore_index = -100, bool reduction_mean = true);

    // log_probs: {batch, seq, vocab}, targets: {batch, seq}
    autograd::Variable forward(
        const autograd::Variable& log_probs,
        const Tensor<int32_t>& targets);

    autograd::Variable forward(const autograd::Variable& input) override;

private:
    TokenId ignore_index_;
    bool reduction_mean_;
};

}  // namespace lightwatch::nn
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **CrossEntropy**: -sum(y * log(softmax(x))) or equivalently -log_softmax(x)[target]
2. **Label smoothing**: Replace one-hot with (1-ε)*one_hot + ε/K
3. **Ignore index**: Skip loss computation for padding tokens
4. **Reduction**: Mean over non-ignored tokens

### Performance Constraints
- O(vocab_size) per token
- Numerically stable (use log_softmax, not log(softmax))

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_21_ce_basic` | logits, targets | Scalar loss |
| `test_phase_21_ce_correct_pred` | argmax matches target | Loss near 0 |
| `test_phase_21_ce_uniform` | Uniform distribution | Loss = log(vocab_size) |
| `test_phase_21_ce_ignore` | targets with -100 | Ignored in loss |
| `test_phase_21_ce_smoothing` | label_smoothing=0.1 | Smoothed targets |
| `test_phase_21_ce_backward` | Backward pass | Gradients correct |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_21" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/nn/loss.hpp`
- [ ] Cross-entropy matches reference implementation

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 300-500 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses cross-entropy loss without label smoothing
- ignore_index = -100 is PyTorch convention
- Reduction mean divides by number of non-ignored tokens
- Numerical stability: compute log_softmax in one fused op
