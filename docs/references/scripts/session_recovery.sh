#!/bin/bash
# File: docs/references/scripts/session_recovery.sh
# Part of: LightwatchAI2 Master Prompt Reference Files
# Referenced by: Master_Prompt.md > STATE MANAGEMENT
# Run on: Every session start

# 0. Check for concurrent execution (lock file with staleness detection)
LOCK_FILE=".lightwatch.lock"
STALE_HOURS=4  # Lock files older than 4 hours are considered stale

if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)

    # Check if PID is still running
    if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "ERROR: Another session is actively running (PID $LOCK_PID)"
        exit 1
    fi

    # Check lock file age (staleness detection)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        LOCK_AGE=$(( ($(date +%s) - $(stat -f %m "$LOCK_FILE")) / 3600 ))
    else
        LOCK_AGE=$(( ($(date +%s) - $(stat -c %Y "$LOCK_FILE")) / 3600 ))
    fi

    if [ "$LOCK_AGE" -ge "$STALE_HOURS" ]; then
        echo "WARNING: Stale lock file detected (${LOCK_AGE}h old, PID $LOCK_PID not running)"
        echo "Removing stale lock and continuing..."
        rm -f "$LOCK_FILE"
    else
        echo "WARNING: Lock file exists (PID $LOCK_PID not running, ${LOCK_AGE}h old)"
        echo "If certain no other session is active: rm $LOCK_FILE"
        exit 1
    fi
fi
echo $$ > "$LOCK_FILE"  # Create lock with current PID

# 1. Check state file exists and is valid JSON
jq . .lightwatch_state.json > /dev/null || echo "ERROR: Invalid state file"

# 2. Check current branch matches state
CURRENT=$(git branch --show-current)
EXPECTED=$(jq -r '.current_branch' .lightwatch_state.json)
if [ "$CURRENT" != "$EXPECTED" ]; then
    echo "WARNING: On branch $CURRENT, state expects $EXPECTED"
fi

# 3. Check for uncommitted changes
if ! git diff --quiet; then
    echo "WARNING: Uncommitted changes detected"
    git status --short
fi

# 4. Verify last commit matches state
LAST_COMMIT=$(git log -1 --format='%H')
STATE_COMMIT=$(jq -r '.last_commit' .lightwatch_state.json)
if [ "$LAST_COMMIT" != "$STATE_COMMIT" ] && [ "$STATE_COMMIT" != "" ]; then
    echo "WARNING: HEAD ($LAST_COMMIT) differs from state ($STATE_COMMIT)"
fi

# 5. If phase was EXECUTING, verify what's done
if [ "$(jq -r '.phase_status' .lightwatch_state.json)" == "EXECUTING" ]; then
    echo "Resuming phase $(jq '.current_phase' .lightwatch_state.json)"
    echo "Completed: $(jq '.completed_deliverables' .lightwatch_state.json)"
    echo "Pending: $(jq '.pending_deliverables' .lightwatch_state.json)"
fi
