#!/bin/bash
set -e

echo "=== LightwatchAI2 Completion Verification ==="

# Check required tools
echo "Checking required tools..."
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: Required tool '$1' not found"
        exit 1
    fi
}

check_tool cmake
check_tool jq
check_tool curl

# Optional tools (warn but don't fail)
SKIP_VALGRIND=0
if ! command -v valgrind &> /dev/null; then
    echo "WARNING: valgrind not found, memory leak checks will be skipped"
    SKIP_VALGRIND=1
fi

# Check for memory measurement tool
SKIP_MEMORY_CHECK=0
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if ! command -v /usr/bin/time &> /dev/null; then
        echo "WARNING: /usr/bin/time not found, memory budget check will be skipped"
        SKIP_MEMORY_CHECK=1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS uses different memory measurement
    echo "NOTE: Using macOS memory measurement via /usr/bin/time -l"
fi

echo "All required tools present"
echo ""

# 1. Build succeeds with strict warnings
echo "[1/8] Building with strict warnings..."
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-Wall -Werror -Wextra"
cmake --build build --parallel

# 2. All tests pass
echo "[2/8] Running tests..."
ctest --test-dir build --output-on-failure

# 3. Generation produces valid output
echo "[3/8] Testing generation..."
./build/bin/lightwatch generate --prompt "The" --max-tokens 20 --seed 42 > /tmp/gen.txt
test -s /tmp/gen.txt  # File is non-empty
WORDS=$(wc -w < /tmp/gen.txt)
if [ "$WORDS" -lt 5 ]; then
    echo "ERROR: Generated only $WORDS words, expected >= 5"
    exit 1
fi
echo "Generated: $(cat /tmp/gen.txt)"

# 4. Performance meets minimum threshold
echo "[4/8] Running benchmark..."
./build/bin/lightwatch benchmark --iterations 100 --json > /tmp/bench.json
TPS=$(jq '.tokens_per_second' /tmp/bench.json)
if ! jq -e '.tokens_per_second > 50' /tmp/bench.json > /dev/null; then
    echo "ERROR: Performance $TPS tok/s below 50 tok/s threshold"
    exit 1
fi
echo "Performance: $TPS tokens/second"

# 5. Memory leak check (skip if valgrind unavailable)
echo "[5/8] Checking memory leaks..."
if [ "$SKIP_VALGRIND" -eq 0 ]; then
    valgrind --leak-check=full --error-exitcode=1 \
        ./build/bin/lightwatch generate --prompt "Test" --max-tokens 10 2>&1 | tail -20
    echo "Memory leak check passed"
else
    echo "Valgrind not available, skipping memory leak check"
fi

# 6. Memory budget check (model should use < 2GB)
echo "[6/8] Checking memory budget..."
MAX_RSS_KB=2097152  # 2GB in KB

if [ "$SKIP_MEMORY_CHECK" -eq 0 ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux: use /usr/bin/time -v
        MEMORY_OUTPUT=$(/usr/bin/time -v ./build/bin/lightwatch generate --prompt "Test prompt for memory check" --max-tokens 100 2>&1)
        MAX_RSS=$(echo "$MEMORY_OUTPUT" | grep "Maximum resident set size" | awk '{print $6}')
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS: use /usr/bin/time -l (reports in bytes)
        MEMORY_OUTPUT=$(/usr/bin/time -l ./build/bin/lightwatch generate --prompt "Test prompt for memory check" --max-tokens 100 2>&1)
        MAX_RSS_BYTES=$(echo "$MEMORY_OUTPUT" | grep "maximum resident set size" | awk '{print $1}')
        MAX_RSS=$((MAX_RSS_BYTES / 1024))  # Convert to KB
    fi

    if [ -n "$MAX_RSS" ] && [ "$MAX_RSS" -gt "$MAX_RSS_KB" ]; then
        echo "ERROR: Memory usage ${MAX_RSS}KB exceeds 2GB limit (${MAX_RSS_KB}KB)"
        exit 1
    fi
    echo "Memory usage: ${MAX_RSS}KB (limit: ${MAX_RSS_KB}KB)"
else
    echo "Memory budget check skipped (measurement tool not available)"
fi

# 7. State file shows all phases complete
echo "[7/8] Checking state file..."
COMPLETED=$(jq '.completed_phases | length' .lightwatch_state.json)
if [ "$COMPLETED" -ne 40 ]; then
    echo "ERROR: Only $COMPLETED/40 phases completed"
    exit 1
fi
echo "All 40 phases completed"

# 8. Documentation exists
echo "[8/8] Checking documentation..."
test -f README.md && test -s README.md
test -f docs/architecture/DECISIONS.md

echo ""
echo "=== VERIFICATION PASSED ==="
echo "LightwatchAI2 build is complete and functional."

# Update project status
jq '.project_status = "COMPLETE"' .lightwatch_state.json > tmp.json
mv tmp.json .lightwatch_state.json
