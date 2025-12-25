# Claude Code Configuration - LightwatchAI2

## CRITICAL: Read This First

**Before taking ANY action on session start, you MUST:**
1. Read the full orchestration prompt:
   ```bash
   cat "docs/Master Prompt.md"
   ```
2. Run the session recovery checklist (see below)
3. Check `.lightwatch_state.json` for current state

The Master Prompt contains all orchestration logic, phase specifications, contracts, and procedures. This CLAUDE.md file contains quick reference only.

---

## Session Recovery Checklist

Run this on EVERY session start before taking any action:

```bash
#!/bin/bash
# 1. Check state file exists and is valid JSON
jq . .lightwatch_state.json > /dev/null || { echo "ERROR: Invalid state file"; exit 1; }

# 2. Check for lock file (concurrent execution)
if [ -f ".lightwatch.lock" ]; then
    PID=$(cat .lightwatch.lock)
    if kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Another session is running (PID $PID)"
        exit 1
    fi
    rm -f .lightwatch.lock
fi

# 3. Check current branch matches state
CURRENT=$(git branch --show-current)
EXPECTED=$(jq -r '.current_branch' .lightwatch_state.json)
if [ "$CURRENT" != "$EXPECTED" ]; then
    echo "WARNING: On branch $CURRENT, state expects $EXPECTED"
fi

# 4. Check for uncommitted changes
if ! git diff --quiet; then
    echo "WARNING: Uncommitted changes detected"
    git status --short
fi

# 5. Report current state
echo "=== Current State ==="
echo "Phase: $(jq '.current_phase' .lightwatch_state.json)"
echo "Status: $(jq -r '.phase_status' .lightwatch_state.json)"
echo "Completed: $(jq '.completed_phases | length' .lightwatch_state.json)/40 phases"
```

---

## Quick Reference

### Commit Authorship (MANDATORY)
ALL commits must use:
```bash
git -c user.name="watchthelight" -c user.email="buteverythingisnormal@gmail.com" commit -m "message"
```
Claude MUST NOT be listed as commit author under any circumstances.

### Commit Message Format
```
[PHASE-XX] <type>: <subject>

<body>

Signed-off-by: watchthelight <buteverythingisnormal@gmail.com>
```

**Type values:**
- `feat` - New feature or functionality
- `fix` - Bug fix
- `test` - Adding or updating tests
- `docs` - Documentation changes
- `refactor` - Code restructuring without behavior change
- `perf` - Performance improvement
- `chore` - Build system, dependencies, tooling

**Rules:**
1. Subject line <= 72 characters
2. Body wrapped at 72 characters
3. Blank line between subject and body
4. Reference test names when fixing test failures
5. ALWAYS include Signed-off-by line

### Code Style
- C++17 standard
- 4-space indentation
- snake_case for functions/variables
- PascalCase for classes/types
- UPPER_CASE for constants

### Test Naming Convention
All test names MUST follow: `test_phase_XX_<component>_<behavior>`

### Pre-Commit Testing (MANDATORY)
Before EVERY commit:
```bash
cmake --build build
ctest --test-dir build -R "phase_$(printf '%02d' $CURRENT_PHASE)" --output-on-failure
```
If either fails, fix before committing.

### Escalation Triggers (STOP and wait for human)
- Contract change required after Phase 10
- Performance < 25 tok/s after optimization
- External dependency seems required
- Memory > 8GB RAM
- Two consecutive phase failures
- Asset download fails all fallbacks
- Same error 3 times after fix attempts
- Git push fails 3 consecutive times

### Model Target
- GPT-2 Small (124M parameters)
- 12 layers, 768 hidden, 12 heads, 64 head dim
- 50257 vocab, 1024 context, 3072 FFN

### Key Files
| Purpose | Location |
|---------|----------|
| Full orchestration | `docs/Master Prompt.md` |
| Current state | `.lightwatch_state.json` |
| API contracts | `docs/contracts/*.hpp` |
| Phase prompts | `docs/prompts/phase-XX-*.md` |
| Test specs | `docs/test_specs/phase-XX-*.md` |
| Decisions log | `docs/architecture/DECISIONS.md` |
| Escalations | `docs/architecture/ESCALATIONS.md` |

### Phase Order
```
0 -> 0.3 -> 0.7 -> 0.5 -> 0.9 -> validate -> 1..40 -> verify
```

### State Values
- `project_status`: IN_PROGRESS | COMPLETE | ESCALATED
- `phase_status`: NOT_STARTED | EXECUTING | VERIFYING | COMPLETE | FAILED | PARTIAL_FAILURE | ESCALATED
