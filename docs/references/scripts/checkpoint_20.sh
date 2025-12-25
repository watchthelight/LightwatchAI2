#!/bin/bash
# File: docs/references/scripts/checkpoint_20.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > CHECKPOINT PHASES > Checkpoint 20

set -e

echo "=== Checkpoint 20: Neural Core Verification ==="

# Build a 2-layer MLP and verify forward/backward
echo "Testing MLP integration..."
./build/bin/test_mlp_integration

# Gradient check: numerical vs analytical (tolerance 1e-5)
echo "Running gradient check..."
./build/bin/test_gradient_check --tolerance 1e-5

# Attention mask verification
echo "Verifying causal attention mask..."
./build/bin/test_attention_causal

echo "=== Checkpoint 20: PASSED ==="
