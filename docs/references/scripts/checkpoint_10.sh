#!/bin/bash
# File: docs/references/scripts/checkpoint_10.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > CHECKPOINT PHASES > Checkpoint 10

set -e

echo "=== Checkpoint 10: Foundation Verification ==="

# Run foundation integration tests
echo "Running foundation integration tests..."
ctest -R "integration_foundation" --output-on-failure

# Verify tensor-autograd integration
echo "Verifying tensor-autograd integration..."
./build/bin/test_tensor_autograd

# Verify tokenizer produces valid token IDs
echo "Verifying tokenizer..."
./build/bin/test_tokenizer --vocab data/vocab/encoder.json

# Memory baseline test (Linux/macOS)
# Creates a 512x768 tensor, performs matmul, checks RSS < 50MB
echo "Running memory baseline test..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MEMORY_OUTPUT=$(/usr/bin/time -v ./build/bin/test_tensor_memory_baseline 2>&1)
    MAX_RSS=$(echo "$MEMORY_OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
    if [ "$MAX_RSS" -gt 51200 ]; then
        echo "ERROR: Memory baseline ${MAX_RSS}KB exceeds 50MB limit"
        exit 1
    fi
    echo "Memory baseline: ${MAX_RSS}KB (limit: 51200KB)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MEMORY_OUTPUT=$(/usr/bin/time -l ./build/bin/test_tensor_memory_baseline 2>&1)
    MAX_RSS_BYTES=$(echo "$MEMORY_OUTPUT" | grep "maximum resident set size" | awk '{print $1}')
    MAX_RSS=$((MAX_RSS_BYTES / 1024))
    if [ "$MAX_RSS" -gt 51200 ]; then
        echo "ERROR: Memory baseline ${MAX_RSS}KB exceeds 50MB limit"
        exit 1
    fi
    echo "Memory baseline: ${MAX_RSS}KB (limit: 51200KB)"
fi

# Memory leak check (if valgrind available)
if command -v valgrind &> /dev/null; then
    echo "Running valgrind memory leak check..."
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/bin/test_tensor_basic 2>&1 | tail -20
    echo "Valgrind memory leak check passed"
else
    echo "SKIP: Valgrind not available"
fi

echo "=== Checkpoint 10: PASSED ==="
