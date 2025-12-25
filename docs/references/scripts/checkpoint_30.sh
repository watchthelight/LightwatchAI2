#!/bin/bash
# File: docs/references/scripts/checkpoint_30.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > CHECKPOINT PHASES > Checkpoint 30

set -e

echo "=== Checkpoint 30: Training Verification ==="

# Overfit test: 10 samples, should reach <0.01 loss
echo "Running overfit test..."
./build/bin/test_overfit --samples 10 --max-epochs 1000 --target-loss 0.01

# Checkpoint save/load
echo "Testing checkpoint roundtrip..."
./build/bin/test_checkpoint_roundtrip

# LR scheduler verification
echo "Verifying LR scheduler..."
./build/bin/test_lr_schedule --warmup 100 --total 1000

echo "=== Checkpoint 30: PASSED ==="
