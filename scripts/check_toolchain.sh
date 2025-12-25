#!/bin/bash
# Check toolchain versions for LightwatchAI2

set -e
echo "=== Toolchain Versions ==="
cmake --version | head -1
${CXX:-g++} --version | head -1
python3 --version
jq --version
curl --version | head -1
valgrind --version 2>/dev/null || echo "valgrind: not installed (optional on macOS)"
echo "=== Check Complete ==="
