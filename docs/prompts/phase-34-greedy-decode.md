# Phase 34: Greedy Decode

## Objective
Implement greedy (argmax) text generation for the GPT model.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 31 | GPT2 model |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 31 | include/lightwatch/models/gpt.hpp | GPT2 |
| 06-07 | include/lightwatch/tokenizer/*.hpp | BPETokenizer |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/generate.hpp | generate_greedy, GenerateConfig | Phase 35, 38 |
| src/generate.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
namespace lightwatch {

struct GenerateConfig {
    size_t max_new_tokens = 100;
    TokenId eos_token_id = 50256;
    bool early_stop = true;  // Stop at EOS
};

// Greedy generation (argmax at each step)
std::vector<TokenId> generate_greedy(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    GenerateConfig config = {});

// Streaming callback version
using TokenCallback = std::function<void(TokenId)>;

void generate_greedy_streaming(
    models::GPT2& model,
    const std::vector<TokenId>& prompt,
    TokenCallback callback,
    GenerateConfig config = {});

}  // namespace lightwatch
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Loop**: For each position up to max_new_tokens:
   - Forward pass: logits = model(current_tokens)
   - Select: next_token = argmax(logits[-1])
   - Append: current_tokens.push_back(next_token)
   - Stop if next_token == eos_token_id
2. **No gradient**: Run in eval mode with no_grad

### Performance Constraints
- O(n²) for n tokens (due to attention)
- Memory: O(n * embed) for activations

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_34_greedy_shape` | Prompt of 5 tokens | Extended sequence |
| `test_phase_34_greedy_deterministic` | Same prompt twice | Same output |
| `test_phase_34_greedy_eos` | Generate until EOS | Stops at EOS |
| `test_phase_34_greedy_max_tokens` | max_new_tokens=10 | Exactly 10 new tokens |
| `test_phase_34_streaming` | Callback | Each token called |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_34" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/generate.hpp`
- [ ] Greedy generation produces valid tokens

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 250-400 |
| New source files | 2 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- Greedy is simplest but often repetitive
- eval mode disables dropout
- no_grad speeds up inference
- Streaming useful for interactive applications
