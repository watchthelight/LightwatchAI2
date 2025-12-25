<!-- File: docs/references/test_specs/phase-06-tokenizer.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPEC FILE TEMPLATES -->

# Phase 06: BPE Tokenizer - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 10

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_06_tokenizer_roundtrip` | `"Hello, world!"` | `decode(encode(x)) == "Hello, world!"` |
| `test_phase_06_tokenizer_special` | `tokenizer.eos_id()` | Returns `50256` |
| `test_phase_06_tokenizer_unicode` | `"日本語テスト"` | Non-empty token vector, no crash |
| `test_phase_06_tokenizer_empty` | `""` | Returns empty `vector<TokenId>{}` |
| `test_phase_06_tokenizer_vocab_size` | `tokenizer.vocab_size()` | Returns `50257` |
| `test_phase_06_tokenizer_whitespace` | `"  hello   world  "` | Roundtrip preserves exact whitespace |
| `test_phase_06_tokenizer_numbers` | `"12345 67890"` | Roundtrip exact match |
| `test_phase_06_tokenizer_long_text` | 10KB random ASCII text | No crash, all token IDs in `[0, 50256]` |
| `test_phase_06_tokenizer_emoji` | `"Hello 🌍🚀 World"` | Roundtrip preserves emojis exactly |
| `test_phase_06_tokenizer_newlines` | `"line1\nline2\r\nline3"` | Roundtrip exact match |

## Implementation Notes

- GPT-2 uses byte-level BPE (handles arbitrary UTF-8)
- Vocab files: `encoder.json` (50257 entries), `vocab.bpe` (50000 merges)
- No UNK token — unknown bytes encoded as byte tokens (e.g., `<0xFF>`)
