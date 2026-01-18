# Writer v2 Migration Notes

## Migration Date
2026-01-18

## Overview
Migrated Writer provider to v2 registry-based architecture.

## Files Created
- `instructor/v2/providers/writer/__init__.py` - Module exports
- `instructor/v2/providers/writer/handlers.py` - Mode handlers (TOOLS, MD_JSON)
- `instructor/v2/providers/writer/client.py` - Factory function `from_writer()`
- `tests/v2/test_writer_handlers.py` - Handler unit tests (29 tests)
- `tests/v2/test_writer_client.py` - Client unit tests (17 tests)

## Files Modified
- `instructor/v2/__init__.py` - Added `from_writer` import and export

## Modes Supported
| Core Mode | Legacy Mode | Notes |
|-----------|-------------|-------|
| `TOOLS` | `WRITER_TOOLS` | Tool calling |
| `MD_JSON` | `WRITER_JSON` | Text extraction fallback |

## Key Implementation Details

### API Differences from OpenAI
- Writer uses `client.chat.chat` instead of `client.chat.completions.create`
- Writer SDK package is `writerai` (not `writer`)
- Client classes are `Writer` and `AsyncWriter`

### Handler Implementation
- `WriterToolsHandler` - Standalone implementation (not inheriting from OpenAI)
- `WriterMDJSONHandler` - Standalone implementation with Writer-specific reask

### Reask Functions
Used existing reask functions from `instructor/providers/writer/utils.py`:
- `reask_writer_tools()` - For TOOLS mode validation errors
- `reask_writer_json()` - For MD_JSON mode validation errors

## Test Results
```
44 passed, 2 skipped in 0.11s
```

Skipped tests require Writer SDK installation:
- `test_from_writer_with_invalid_client`
- `test_from_writer_with_invalid_mode`

## Mode Normalizations
Already present in `instructor/v2/core/registry.py`:
- `Mode.WRITER_TOOLS` -> `Mode.TOOLS`
- `Mode.WRITER_JSON` -> `Mode.MD_JSON`

## Deviations from Plan
None - implementation followed the plan exactly.

## Notes
- Writer does not support JSON_SCHEMA mode (no native structured outputs)
- Writer does not support PARALLEL_TOOLS mode
- The `tool_choice` is set to "auto" for TOOLS mode (matching v1 behavior)
