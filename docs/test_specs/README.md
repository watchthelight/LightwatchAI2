# Test Specifications

This directory contains test specifications extracted from the Master Prompt to reduce context pressure during execution.

## Files

| File | Phase | Complexity | Tests |
|------|-------|------------|-------|
| `phase-03-tensor.md` | Tensor Core | HIGH | 12 |
| `phase-04-simd.md` | SIMD Operations | HIGH | 6 |
| `phase-05-autograd.md` | Autograd Engine | HIGH | 10 |
| `phase-06-tokenizer.md` | BPE Tokenizer | MEDIUM | 10 |
| `phase-15-attention.md` | Single-Head Attention | HIGH | 8 |
| `phase-16-mha.md` | Multi-Head Attention | MEDIUM | 6 |
| `phase-19-decoder.md` | Transformer Decoder | MEDIUM | 5 |
| `phase-29-training.md` | Training Loop | HIGH | 6 |
| `phase-31-gpt.md` | GPT Architecture | HIGH | 6 |
| `phase-36-kvcache.md` | KV-Cache | HIGH | 5 |
| `phase-38-cli.md` | CLI/REPL | MEDIUM | 8 |

## Usage

When implementing Phase N, read the corresponding test specification file:

```bash
cat docs/test_specs/phase-NN-*.md
```

These specifications define:
- **Required Tests**: Exact test cases with inputs and expected outputs
- **Implementation Notes**: Edge cases and validation requirements

## Minimum Test Counts

| Complexity | Minimum Tests |
|------------|---------------|
| HIGH | 6+ |
| MEDIUM | 4+ |
| LOW | 2+ |

Phases not listed here have lower complexity and should define tests in their phase prompts.
