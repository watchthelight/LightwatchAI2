#!/bin/bash
# File: docs/references/scripts/acquire_assets.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > PHASE 0.7: ACQUIRE EXTERNAL ASSETS
# Created in: Phase 0.7

set -e

echo "=== Acquiring GPT-2 Tokenizer Assets ==="

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "ERROR: curl is required but not installed"
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "ERROR: jq is required but not installed"
    exit 1
fi

mkdir -p data/vocab

try_download() {
    local name=$1
    shift
    for url in "$@"; do
        echo "Trying $url..."
        if curl -fsSL --connect-timeout 10 "$url" -o "data/vocab/$name"; then
            echo "Downloaded $name"
            return 0
        fi
        echo "Failed, trying next URL..."
    done
    echo "ERROR: Failed to download $name from all sources"
    return 1
}

try_download "vocab.bpe" \
    "https://huggingface.co/gpt2/raw/main/vocab.bpe" \
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe"

try_download "encoder.json" \
    "https://huggingface.co/gpt2/raw/main/encoder.json" \
    "https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json"

# Validate
echo ""
echo "=== Validating Assets ==="

if [ ! -s data/vocab/vocab.bpe ]; then
    echo "ERROR: vocab.bpe is empty or missing"
    exit 1
fi

VOCAB_SIZE=$(jq 'length' data/vocab/encoder.json)
if [ "$VOCAB_SIZE" -ne 50257 ]; then
    echo "ERROR: encoder.json has $VOCAB_SIZE tokens, expected 50257"
    exit 1
fi

MERGE_COUNT=$(wc -l < data/vocab/vocab.bpe | tr -d ' ')
echo "vocab.bpe: $MERGE_COUNT lines"
echo "encoder.json: $VOCAB_SIZE tokens"

echo ""
echo "=== Assets Acquired Successfully ==="
