# Phase 38: CLI/REPL

## Objective
Implement command-line interface for generation, benchmarking, and interactive REPL.

## Prerequisites
| Phase | Required Outputs |
|-------|------------------|
| 31 | GPT2 model |
| 35 | Sampling generation |
| 37 | Weight serialization |

## Inputs (APIs consumed from prior phases)
| Source Phase | File | Symbols Used |
|--------------|------|--------------|
| 31 | include/lightwatch/models/gpt.hpp | GPT2 |
| 35 | include/lightwatch/generate.hpp | SamplingConfig |
| 37 | include/lightwatch/serialize.hpp | load_weights |
| 06 | include/lightwatch/tokenizer/bpe.hpp | BPETokenizer |

## Outputs (Files produced for future phases)
| File | Public Symbols | Consumed By |
|------|----------------|-------------|
| src/main.cpp | CLI entry point | Phase 39, 40 |
| include/lightwatch/cli.hpp | CLI utilities | N/A |

## Specification

### Data Structures
```cpp
// CLI commands and options
// ./lightwatch generate --prompt "Hello" --max-tokens 100
// ./lightwatch benchmark --prompt-tokens 128 --generate-tokens 128
// ./lightwatch repl --model gpt2-small
// ./lightwatch info --model-path model.lwbin

namespace lightwatch::cli {

struct GenerateOptions {
    std::string prompt;
    std::string model_path;
    size_t max_tokens = 100;
    float temperature = 0.8;
    int top_k = 40;
    float top_p = 0.9;
    bool json_output = false;
    bool stream = true;
};

struct BenchmarkOptions {
    std::string model_path;
    size_t prompt_tokens = 128;
    size_t generate_tokens = 128;
    size_t warmup = 5;
    size_t iterations = 100;
    bool json_output = false;
};

int run_generate(const GenerateOptions& opts);
int run_benchmark(const BenchmarkOptions& opts);
int run_repl(const std::string& model_path);
int run_info(const std::string& model_path);

}  // namespace lightwatch::cli
```

### Function Signatures
See data structures above.

### Algorithmic Requirements
1. **Generate**: Load model, tokenize prompt, generate, print output
2. **Benchmark**: Warmup runs, timed runs, compute tok/s
3. **REPL**: Read prompt, generate, repeat
4. **JSON output**: Structured output for scripting

### Performance Constraints
- CLI overhead: < 100ms startup
- Streaming: Output tokens as generated

## Required Tests
See `docs/test_specs/phase-38-cli.md` for complete test specifications.

| Test Name | Input | Expected Output |
|-----------|-------|-----------------|
| `test_phase_38_generate` | Simple prompt | Non-empty output |
| `test_phase_38_generate_json` | --json flag | Valid JSON |
| `test_phase_38_benchmark` | benchmark command | tok/s reported |
| `test_phase_38_info` | info command | Model stats |
| `test_phase_38_temperature` | temp=0.1 | Less diverse |
| `test_phase_38_top_k` | top_k=10 | Constrained vocab |

## Acceptance Criteria
- [ ] `cmake --build build` exits 0
- [ ] `ctest -R "phase_38" --output-on-failure` exits 0
- [ ] `./build/bin/lightwatch generate --help` works
- [ ] Benchmark reports performance in tok/s

## Estimated Scope
| Metric | Value |
|--------|-------|
| Lines of code | 400-600 |
| New source files | 3 |
| New test files | 2 |
| Complexity | MEDIUM |

## Notes
- Use simple argument parsing (no external deps)
- JSON output follows schema in docs/references/schemas/
- REPL supports multi-line input (blank line to submit)
- Streaming output for better UX
