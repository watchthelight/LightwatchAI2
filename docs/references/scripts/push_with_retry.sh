#!/bin/bash
# File: docs/references/scripts/push_with_retry.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > GIT PUSH FAILURE HANDLING

push_with_retry() {
    local max_attempts=3
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        echo "Push attempt $attempt/$max_attempts..."
        if git push origin main 2>&1; then
            echo "Push succeeded"
            return 0
        fi

        echo "Push failed, waiting 60s before retry..."
        sleep 60
        attempt=$((attempt + 1))
    done

    echo "ERROR: Push failed after $max_attempts attempts"
    return 1
}

# Usage: source this file then call push_with_retry
# Or run directly to test
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    push_with_retry
fi
