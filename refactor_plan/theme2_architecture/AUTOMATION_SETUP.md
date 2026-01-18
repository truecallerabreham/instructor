# Migration Automation Setup Complete

This directory now has automated tooling to work through the v2 provider migration plan.

## Files Created

### 1. `/work_through_migration.sh`
Main automation script that:
- Reads the migration plan file
- Finds unchecked tasks
- Calls cursor-agent to complete them
- Manages Graphite stack branches
- Creates migration notes
- Commits with conventional commit messages

### 2. `/.cursor/cli-config.json`
Permissions configuration for cursor-agent:
- Allows: git, gt, pytest, uv, python, file writes to project directories
- Denies: dangerous operations (rm -rf, .env writes, etc.)

### 3. `/MIGRATION_AUTOMATION.md`
Complete documentation on:
- Prerequisites and installation
- How to use the script
- Migration workflow
- Troubleshooting

### 4. `migration_notes/` Directory
Storage for per-provider migration notes:
- `README.md` - Explains the directory purpose
- `_template.md` - Template for creating provider notes
- Individual notes files will be created automatically by the script

## Quick Start

```bash
# Install dependencies
brew install withgraphite/tap/graphite
curl https://cursor.com/install -fsS | bash

# Run the migration
./work_through_migration.sh
```

## How It Works

1. Script reads `v2_provider_migration_plan.md`
2. Finds next unchecked task (`- [ ]`)
3. cursor-agent works on the task using patterns from Anthropic/GenAI providers
4. Creates/updates Graphite branch (`migration/<provider>`)
5. Commits changes with conventional commits
6. Checks off task (`- [x]`)
7. Creates/updates migration notes
8. Repeats for next task

## Graphite Stack Workflow

```bash
# View current stack
gt log short

# Script creates branches like:
# - migration/openai
# - migration/cohere
# - migration/xai

# When provider complete, submit stack:
gt stack submit --reviewers jxnl,ivanleomk
```

## Migration Notes

For each provider, notes are stored in:
```
migration_notes/{provider}_notes.md
```

Example: `migration_notes/openai_notes.md`

Contains:
- Implementation decisions
- Test results
- Issues encountered
- Deviations from plan
- Follow-up tasks

## Safety Features

- Max 50 iterations per run
- Permissions controlled via `.cursor/cli-config.json`
- All changes tracked in git/Graphite
- Can stop and resume anytime (idempotent)

## Monitoring Progress

```bash
# Check plan status
grep "^- \[" v2_provider_migration_plan.md | head -20

# View migration notes
ls -la migration_notes/

# View Graphite stack
gt log short
```

## Next Steps

1. Ensure you're on the right base branch (main or develop)
2. Run: `./work_through_migration.sh`
3. Monitor progress in terminal output
4. Review migration notes as providers are completed
5. Submit stacks for review when providers are done

## Additional Resources

- Full documentation: `/MIGRATION_AUTOMATION.md`
- Migration plan: `v2_provider_migration_plan.md`
- Example providers: `../../instructor/v2/providers/anthropic/`, `../../instructor/v2/providers/genai/`
