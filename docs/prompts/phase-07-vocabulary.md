# Phase 07: Vocabulary

## Objective
Implement the Vocabulary class for bidirectional token-ID mapping with special token handling.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 06 | include/lightwatch/tokenizer/bpe.hpp |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 06 | include/lightwatch/tokenizer/bpe.hpp | BPETokenizer internals |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/tokenizer/vocabulary.hpp | Vocabulary, SpecialTokens | Phase 08, 27, 38 |
| src/tokenizer/vocabulary.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Part of docs/contracts/tokenizer.hpp
namespace lightwatch::tokenizer {

using TokenId = int32_t;

struct SpecialTokens {
    static constexpr TokenId PAD = 50256;
    static constexpr TokenId EOS = 50256;  // <|endoftext|>
};

class Vocabulary {
public:
    Vocabulary();

    TokenId add_token(const std::string& token);
    TokenId token_to_id(const std::string& token) const;
    std::string id_to_token(TokenId id) const;

    bool contains(const std::string& token) const;
    bool contains(TokenId id) const;
    size_t size() const;

    TokenId pad_id() const;
    TokenId eos_id() const;
    bool is_special_token(TokenId id) const;

    void save(const std::string& path) const;
    static Vocabulary load(const std::string& path);
    static Vocabulary from_encoder_json(const std::string& path);
};

}  // namespace lightwatch::tokenizer
```

### Function Signatures
See `docs/contracts/tokenizer.hpp` for complete API.

### Algorithmic Requirements
1. **Token-to-ID**: O(1) lookup via hash map
2. **ID-to-token**: O(1) lookup via vector
3. **Special tokens**: EOS/PAD at index 50256 for GPT-2
4. **Serialization**: JSON format for compatibility

### Performance Constraints
- Lookup: O(1) average case
- Memory: ~50K entries * ~20 bytes average = ~1MB

## Required Tests
| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_07_vocab_size` | Load GPT-2 vocab | size() == 50257 |
| `test_phase_07_token_to_id` | "Hello" | Returns valid ID |
| `test_phase_07_id_to_token` | 50256 | "<\|endoftext\|>" |
| `test_phase_07_eos_id` | eos_id() | 50256 |
| `test_phase_07_contains` | contains("the") | true |
| `test_phase_07_roundtrip` | add->lookup | Consistent |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_07" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/tokenizer/vocabulary.hpp`
- [ ] GPT-2 vocabulary loads correctly

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 1 |
| Complexity | LOW |

## Notes
- GPT-2 has exactly 50257 tokens (50256 BPE + 1 special)
- Token strings may contain unicode/byte sequences
- encoder.json format: {"token": id, ...}
- No separate decoder.json needed (reverse of encoder)
