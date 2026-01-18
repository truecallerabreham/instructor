#!/bin/bash

# Script to work through the v2 provider migration plan systematically using the Cursor agent
# Each iteration works on the next unchecked section and commits to a Graphite stack
#
# Requirements:
#   - Graphite CLI (gt) must be installed: brew install withgraphite/tap/graphite
#   - Cursor CLI must be installed: curl https://cursor.com/install -fsS | bash
#   - Permissions configured in .cursor/cli-config.json
#
# Usage:
#   ./work_through_migration.sh                              # Uses v2_provider_migration_plan.md (default)
#   ./work_through_migration.sh v2_provider_migration_plan.md # Explicit file
#   ./work_through_migration.sh custom_plan.md               # Custom plan file

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# Allow plan file to be specified as argument, default to v2_provider_migration_plan.md
PLAN_FILENAME="${1:-v2_provider_migration_plan.md}"
PLAN_FILE="refactor_plan/theme2_architecture/$PLAN_FILENAME"

MAX_ITERATIONS=50  # Safety limit to prevent infinite loops
ITERATION=0

# Change to repo root so agent works in the right context
cd "$REPO_ROOT" || exit 1

# Check if plan file exists
if [ ! -f "$REPO_ROOT/$PLAN_FILE" ]; then
    echo "Error: $PLAN_FILE not found at $REPO_ROOT/$PLAN_FILE"
    echo ""
    echo "Available plan files in refactor_plan/theme2_architecture/:"
    ls -1 "$REPO_ROOT/refactor_plan/theme2_architecture/"*.md 2>/dev/null | xargs -I{} basename {}
    exit 1
fi

echo "Using plan file: $PLAN_FILE"
echo ""

# Main loop
while [ $ITERATION -lt $MAX_ITERATIONS ]; do
    ITERATION=$((ITERATION + 1))
    echo ""
    echo "========================================="
    echo "Iteration $ITERATION"
    echo "========================================="
    echo ""
    
    # Prompt for the agent
    PROMPT="Read $PLAN_FILE and work through the v2 provider migration systematically:

1. Find the next unchecked section/item in the plan (look for sections with - [ ] checkboxes)
2. Work on completing that specific task only - do the immediate next thing
3. When you finish a task, check it off in $PLAN_FILE (change - [ ] to - [x])
4. Migration-specific guidelines:
   - Follow the patterns established in already-migrated providers (Anthropic, GenAI)
   - Read instructor/v2/providers/anthropic/ and instructor/v2/providers/genai/ as reference examples
   - Create handlers.py first (maps modes to SDK-specific format)
   - Then create client.py (factory function like from_provider())
   - Add provider to instructor/v2/__init__.py exports
   - Add test configuration to tests/v2/ or tests/llm/shared_config.py
   - Run tests after each provider migration
5. Phase notes and changelog:
   - Create notes in refactor_plan/theme2_architecture/migration_notes/{provider_name}_notes.md
   - Use provider names like 'openai', 'cohere', 'xai', 'groq', 'mistral', etc.
   - Include any important notes, decisions, blockers, or observations
   - Document any deviations from the plan or unexpected issues
   - Track test results and any failing tests
   - Create the notes file if it doesn't exist, or append to it if it does
6. Graphite stack management per provider:
   - Check current stack status with: gt log short
   - For each provider migration, create a new branch in the stack: gt branch create migration/<provider-name>
   - Examples: gt branch create migration/openai, gt branch create migration/cohere
   - Keep iterating on the same branch for that provider - make multiple commits as you work through all tasks
   - Only create a new branch when moving to a completely different provider
   - Commit your changes frequently using Graphite with clear conventional commit messages:
   - Make sure to commit the updated $PLAN_FILE when you check off tasks
   - Commit migration notes along with your work
   - After completing a provider, you can submit the stack: gt stack submit (but don't do this automatically)
7. PR submission (when provider is complete):
   - After completing a provider migration, submit the stack for review
   - Use: gt stack submit --reviewers jxnl,ivanleomk
   - Include in PR description:
     - Overview of the provider migration
     - List of modes supported
     - Test results (passing/failing)
     - Any deviations from the plan
     - Link to migration notes file
   - Don't submit PRs automatically - only when explicitly asked or when a provider is 100% complete
8. Testing priorities:
   - Prioritize providers with available API keys (P0, P1 in the plan)
   - For providers without API keys (P2, P3), create unit tests that don't require keys
   - Run existing tests to ensure backward compatibility: pytest tests/ -v
   - Run v2 tests specifically: pytest tests/v2/ -v
   - Use -k flag to run provider-specific tests: pytest -k 'openai'
9. Code quality:
   - Follow the existing code style in instructor/v2/
   - Use async over synchronous when possible
   - Add proper type hints
   - Write clear docstrings at grade 9-10 reading level
   - Each code example should be self-contained with complete imports

Important: Only work on ONE task or section per iteration. Don't jump ahead. If a task is already checked off (- [x]), move to the next unchecked one (- [ ]). Keep iterating on the same git branch per provider until that provider is fully migrated. Be thorough but focused."

    echo "Calling cursor-agent with prompt..."
    echo ""
    
    # Call cursor-agent in non-interactive mode
    # -p / --print: non-interactive mode for scripting
    # --force: actually execute commands and file modifications (not just propose)
    # --output-format text: human-readable output
    cursor-agent -p --force --output-format text "$PROMPT"
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "Agent exited with code $EXIT_CODE"
        echo "Stopping loop"
        break
    fi
    
    echo ""
    echo "Iteration $ITERATION completed"
done

echo ""
echo "Reached maximum iterations ($MAX_ITERATIONS) or stopped by user"
echo "Script completed"
