<!-- File: docs/references/test_specs/phase-38-cli.md -->
<!-- Part of: LightwatchAI2 Master Prompt Reference Files -->
<!-- Referenced by: Master_Prompt.md > TEST SPECIFICATIONS -->

# Phase 38: CLI/REPL - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 8

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_phase_38_cli_generate` | `./lightwatch generate --prompt "Hello"` | Outputs generated text to stdout |
| `test_phase_38_cli_max_tokens` | `--max-tokens 50` | Output contains ≤50 generated tokens |
| `test_phase_38_cli_temperature` | `--temperature 0.0` | Deterministic output (greedy) |
| `test_phase_38_cli_json_output` | `--json` | Valid JSON with prompt_tokens, generated_tokens, etc. |
| `test_phase_38_cli_seed` | `--seed 42` (twice) | Identical outputs |
| `test_phase_38_cli_benchmark` | `./lightwatch benchmark` | Reports tokens_per_second in output |
| `test_phase_38_cli_help` | `./lightwatch --help` | Shows usage information, exits 0 |
| `test_phase_38_cli_invalid_arg` | `./lightwatch --invalid` | Exits non-zero, shows error message |

## Implementation Notes

- CLI uses argument parsing (getopt or similar)
- `--json` mode emits structured output for scripting
- Benchmark mode runs warmup iterations before measuring
- Error messages should go to stderr
