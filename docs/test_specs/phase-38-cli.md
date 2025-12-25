# Phase 38: CLI/REPL - Test Specifications

**Complexity:** MEDIUM
**Minimum Tests Required:** 8

## Required Tests

| Test | Input | Expected |
|------|-------|----------|
| `test_cli_generate_basic` | `--prompt "Hello"` | Non-empty output, exit 0 |
| `test_cli_generate_json` | `--prompt "Hi" --json` | Valid JSON with required fields |
| `test_cli_benchmark_json` | `benchmark --json` | Valid JSON with tokens_per_second |
| `test_cli_json_schema` | `--json` output | Contains: command, prompt_tokens, tokens_per_second |
| `test_cli_help` | `--help` | Shows usage, exit 0 |
| `test_cli_invalid_flag` | `--invalid` | Error message, exit non-zero |
| `test_cli_max_tokens` | `--max-tokens 10` | Output ≤ 10 tokens |
| `test_cli_seed` | `--seed 42` twice | Same output both times |

## Implementation Notes

- JSON output must be valid (parseable by jq)
- Exit codes: 0 = success, 1 = general error, 2 = invalid args
- Seed flag enables reproducible generation
- See `docs/contracts/cli.md` for full interface specification
