# Phase 06: BPE Tokenizer

## Objective
Implement Byte Pair Encoding tokenizer compatible with GPT-2's tokenization scheme.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 01 | CMakeLists.txt, build system |
| 02 | include/lightwatch/memory/* |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 02 | include/lightwatch/memory/aligned.hpp | Memory utilities |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| include/lightwatch/tokenizer/bpe.hpp | BPETokenizer | Phase 07, 08, 27, 38 |
| src/tokenizer/bpe.cpp | Implementation | N/A |

## Specification

### Data Structures
```cpp
// Part of docs/contracts/tokenizer.hpp
namespace lightwatch::tokenizer {

class BPETokenizer {
public:
    BPETokenizer();

    // Core operations
    std::vector<TokenId> encode(const std::string& text) const;
    std::string decode(const std::vector<TokenId>& tokens) const;

    // Batch operations
    std::vector<std::vector<TokenId>> encode_batch(
        const std::vector<std::string>& texts) const;

    // Factory methods
    static BPETokenizer from_files(
        const std::string& vocab_path,    // encoder.json
        const std::string& merges_path);  // vocab.bpe

    static BPETokenizer gpt2(const std::string& vocab_dir = "data/vocab");
};

}  // namespace lightwatch::tokenizer
```

### Function Signatures
See `docs/contracts/tokenizer.hpp` for complete API.

### Algorithmic Requirements
1. **Byte-level encoding**: Convert text to bytes, map to Unicode symbols
2. **BPE merge algorithm**:
   - Find most frequent adjacent pair
   - Merge according to learned merge rules
   - Repeat until no more merges possible
3. **GPT-2 specifics**:
   - Uses byte-level BPE (no OOV tokens)
   - Special regex pattern for word splitting
   - Unicode handling via byte fallback

### Performance Constraints
- Encoding: O(n * m) where n=text length, m=vocab size
- Decoding: O(n) linear in token count

## Required Tests
See `docs/test_specs/phase-06-tokenizer.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_06_encode_hello` | "Hello" | [15496] |
| `test_phase_06_encode_world` | " world" | [995] |
| `test_phase_06_encode_hello_world` | "Hello world" | [15496, 995] |
| `test_phase_06_decode_roundtrip` | encode->decode | Original text |
| `test_phase_06_unicode` | "日本語" | Valid tokens |
| `test_phase_06_special_chars` | "<|endoftext|>" | [50256] |
| `test_phase_06_empty` | "" | [] |
| `test_phase_06_whitespace` | "   " | Preserves spaces |
| `test_phase_06_newline` | "a\\nb" | Preserves newline |
| `test_phase_06_batch` | ["a", "b"] | [[64], [65]] |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_06" --output-on-failure` exits 0
- [ ] `test -f include/lightwatch/tokenizer/bpe.hpp`
- [ ] Roundtrip test passes for arbitrary text

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 600-900 |
| New source files | 3 |
| New test files | 1 |
| Complexity | MEDIUM |

## Notes
- GPT-2 uses a specific regex for splitting: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
- Byte-level means every byte maps to a vocab token (no UNK)
- vocab.bpe contains merge rules in order of priority
- encoder.json contains the final token-to-id mapping
