# CLI Contract

## Overview

This document specifies the command-line interface for the LightwatchAI2 `lightwatch` binary.

**Defined by:** Phase 38
**Consumers:** Phase 39 (Profiling), Phase 40 (Integration), End users

## Binary Name

```
lightwatch
```

Built to: `build/bin/lightwatch`

## Commands

### generate (default)

Generate text from a prompt.

```bash
lightwatch generate [OPTIONS] --prompt <TEXT>
lightwatch --prompt <TEXT>  # 'generate' is implicit
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt`, `-p` | string | required | Input prompt text |
| `--max-tokens`, `-n` | int | 128 | Maximum tokens to generate |
| `--temperature`, `-t` | float | 1.0 | Sampling temperature (0 = greedy) |
| `--top-k` | int | 50 | Top-k sampling (0 = disabled) |
| `--top-p` | float | 1.0 | Nucleus sampling threshold |
| `--seed`, `-s` | int | random | Random seed for reproducibility |
| `--model`, `-m` | path | `data/weights/gpt2.lwt` | Model weights path |
| `--json`, `-j` | flag | false | Output JSON instead of text |
| `--stream` | flag | false | Stream tokens as generated |
| `--no-cache` | flag | false | Disable KV-cache |

#### Output (default)

Plain text to stdout:

```
The quick brown fox jumps over the lazy dog.
```

#### Output (--json)

```json
{
    "command": "generate",
    "prompt": "The quick brown",
    "prompt_tokens": 3,
    "generated_text": " fox jumps over the lazy dog.",
    "generated_tokens": 8,
    "total_tokens": 11,
    "total_duration_ms": 156,
    "first_token_ms": 23,
    "tokens_per_second": 51.3,
    "stop_reason": "max_tokens",
    "seed": 42,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0
}
```

#### stop_reason values

| Value | Description |
|-------|-------------|
| `"max_tokens"` | Hit `--max-tokens` limit |
| `"eos"` | Generated end-of-sequence token |
| `"stop_sequence"` | Hit user-specified stop sequence (future) |

### benchmark

Run performance benchmarks.

```bash
lightwatch benchmark [OPTIONS]
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--prompt-tokens` | int | 128 | Number of prompt tokens |
| `--generate-tokens` | int | 128 | Tokens to generate per iteration |
| `--warmup` | int | 5 | Warmup iterations (discarded) |
| `--iterations` | int | 100 | Measured iterations |
| `--model`, `-m` | path | `data/weights/gpt2.lwt` | Model weights path |
| `--json`, `-j` | flag | false | Output JSON instead of text |

#### Output (default)

```
LightwatchAI2 Benchmark
=======================
Model: GPT-2 Small (124M parameters)
Prompt tokens: 128
Generated tokens: 128
Warmup: 5 iterations
Measured: 100 iterations

Results:
  Total time: 2.48s
  Tokens/second: 51.6
  Latency p50: 24.1ms
  Latency p99: 31.2ms
  Memory RSS: 612 MB
```

#### Output (--json)

```json
{
    "command": "benchmark",
    "model": "data/weights/gpt2.lwt",
    "model_params": 124443648,
    "prompt_tokens": 128,
    "generated_tokens": 128,
    "warmup_iterations": 5,
    "iterations": 100,
    "total_seconds": 2.48,
    "tokens_per_second": 51.6,
    "latency_p50_ms": 24.1,
    "latency_p99_ms": 31.2,
    "memory_rss_mb": 612,
    "hardware": {
        "arch": "x86-64",
        "simd": "AVX2",
        "threads": 1
    }
}
```

### train

Train or fine-tune the model.

```bash
lightwatch train [OPTIONS] --data <PATH>
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data`, `-d` | path | required | Training data path |
| `--model`, `-m` | path | none | Pretrained weights (optional) |
| `--output`, `-o` | path | `output/` | Output directory |
| `--epochs` | int | 1 | Training epochs |
| `--batch-size` | int | 4 | Batch size |
| `--lr` | float | 3e-4 | Learning rate |
| `--warmup-steps` | int | 100 | LR warmup steps |
| `--max-steps` | int | -1 | Max steps (-1 = unlimited) |
| `--save-every` | int | 1000 | Checkpoint frequency |
| `--eval-every` | int | 100 | Evaluation frequency |
| `--json`, `-j` | flag | false | Output JSON progress |

### info

Display model information.

```bash
lightwatch info [OPTIONS]
```

#### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model`, `-m` | path | `data/weights/gpt2.lwt` | Model weights path |
| `--json`, `-j` | flag | false | Output JSON |

#### Output (--json)

```json
{
    "command": "info",
    "model_path": "data/weights/gpt2.lwt",
    "model_type": "gpt2",
    "parameters": 124443648,
    "config": {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "vocab_size": 50257,
        "block_size": 1024
    },
    "file_size_mb": 474.5,
    "format": "lwt_v1"
}
```

### help

Display help information.

```bash
lightwatch help [COMMAND]
lightwatch --help
lightwatch -h
```

### version

Display version information.

```bash
lightwatch version
lightwatch --version
lightwatch -V
```

Output:
```
lightwatch 0.1.0 (LightwatchAI2)
Built: 2025-01-15 12:34:56
Compiler: GCC 13.2.0
Features: AVX2, KV-Cache
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | File not found |
| 4 | Model load failed |
| 5 | Out of memory |
| 6 | CUDA/GPU error (reserved for future) |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGHTWATCH_MODEL` | Default model path | `data/weights/gpt2.lwt` |
| `LIGHTWATCH_CACHE_DIR` | Cache directory | `~/.cache/lightwatch` |
| `LIGHTWATCH_LOG_LEVEL` | Log level (error/warn/info/debug) | `warn` |
| `LIGHTWATCH_THREADS` | Thread count (0 = auto) | `0` |

## Standard Streams

| Stream | Usage |
|--------|-------|
| stdout | Generated text, JSON output, normal results |
| stderr | Errors, warnings, progress (non-JSON mode) |

## Signal Handling

| Signal | Behavior |
|--------|----------|
| SIGINT (Ctrl+C) | Graceful shutdown, partial output preserved |
| SIGTERM | Graceful shutdown |
| SIGPIPE | Ignored (allows piping to head, etc.) |

## Examples

```bash
# Basic generation
lightwatch --prompt "Once upon a time"

# Reproducible generation
lightwatch --prompt "Hello" --seed 42 --temperature 0.7

# JSON output for scripting
lightwatch --prompt "Test" --json | jq '.tokens_per_second'

# Benchmark with JSON
lightwatch benchmark --iterations 50 --json > benchmark.json

# Training from scratch
lightwatch train --data data/corpus.txt --epochs 10 --output checkpoints/

# Fine-tuning
lightwatch train --data data/finetune.txt --model data/weights/gpt2.lwt
```
