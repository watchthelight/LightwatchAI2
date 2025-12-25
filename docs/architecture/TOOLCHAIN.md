# Toolchain Documentation

This document records the toolchain versions used during development.

## Current Environment

(Run `scripts/check_toolchain.sh` to populate)

## Minimum Requirements

| Tool | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| CMake | 3.16 | 3.28+ | For FetchContent, presets |
| GCC | 10 | 13+ | C++17 constexpr support |
| Clang | 12 | 17+ | Better diagnostics |
| MSVC | 19.29 | 19.38+ | VS 2022 recommended |
| Python | 3.9 | 3.11+ | For validation scripts |
| jq | 1.6 | 1.7+ | JSON parsing in scripts |
| valgrind | 3.15 | 3.21+ | Memory leak detection (Linux) |
| curl | 7.68 | 8.0+ | Asset downloads |

## Installation

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y cmake g++ python3 python3-pip jq curl valgrind
```

### macOS (Homebrew)
```bash
brew install cmake jq curl
# valgrind not available on macOS ARM
```

### Windows (Chocolatey)
```powershell
choco install cmake jq curl python3
```

## Verification

```bash
bash scripts/check_toolchain.sh
```
