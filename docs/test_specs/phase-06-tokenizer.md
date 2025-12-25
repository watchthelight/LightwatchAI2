# Phase 06: BPE Tokenizer - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 10

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_tokenizer_roundtrip` | `"Hello, world!"` | `decode(encode(x)) == x` |
| `test_tokenizer_special` | EOS token | ID == 50256 |
| `test_tokenizer_unicode` | `"日本語"` | No crash, tokens produced |
| `test_tokenizer_empty` | `""` | Returns empty vector |
| `test_tokenizer_vocab_size` | Load GPT-2 vocab | `vocab_size() == 50257` |
| `test_tokenizer_whitespace` | `"  leading and trailing  "` | Roundtrip exact match |
| `test_tokenizer_numbers` | `"12345"` | Tokenizes correctly |
| `test_tokenizer_long_text` | 2000 random tokens | No crash, valid IDs |
| `test_tokenizer_emoji` | `"Hello 🌍 World"` | Roundtrip preserves emoji |
| `test_tokenizer_newlines` | `"line1\nline2\r\nline3"` | Roundtrip exact match |

## Implementation Notes

- GPT-2 uses byte-level BPE, so all UTF-8 sequences should be encodable
- Special token <|endoftext|> has ID 50256
- Whitespace handling is critical for GPT-2 compatibility
- Test with actual GPT-2 vocab files from `data/vocab/`
