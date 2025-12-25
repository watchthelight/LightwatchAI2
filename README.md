# LightwatchAI2

A minimal-dependency C++ text generation model implementing GPT-2 Small (124M parameters) from scratch.

## Overview

This project builds a complete GPT-2 inference engine using only:
- C++17 standard library
- nlohmann/json (for configuration parsing)

**No external ML frameworks** (Eigen, OpenBLAS, PyTorch, TensorFlow, etc.) are used.

## Target Model: GPT-2 Small

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Head dimension | 64 |
| Vocabulary size | 50257 |
| Context length | 1024 |
| FFN hidden | 3072 |
| Parameters | ~124M |

## Building

### Requirements
- CMake 3.16+
- C++17 compiler (GCC 10+, Clang 12+, or MSVC 19.29+)
- Python 3.9+ (for validation scripts)

### Build Commands
```bash
cmake -B build -S .
cmake --build build
```

### Running Tests
```bash
ctest --test-dir build --output-on-failure
```

## Project Structure

```
LightwatchAI2/
├── include/lightwatch/    # Header files
├── src/                   # Implementation files
├── tests/                 # Test suites
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── smoke/            # Smoke tests
│   └── benchmarks/       # Performance benchmarks
├── docs/
│   ├── Master Prompt.md  # Build orchestration
│   ├── contracts/        # API specifications
│   ├── prompts/          # Phase implementation prompts
│   └── architecture/     # Design decisions
├── scripts/              # Build and validation scripts
├── configs/              # Model configurations
└── data/
    ├── vocab/            # Tokenizer assets
    └── weights/          # Model weights (optional)
```

## Development

This project is built in 40 phases, orchestrated by the Master Prompt. See `docs/Master Prompt.md` for details.

### Phase Progress
Check `.lightwatch_state.json` for current build status.

## License

MIT License - See LICENSE file for details.

## Author

watchthelight
