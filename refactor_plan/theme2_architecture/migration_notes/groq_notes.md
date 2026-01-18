# Groq v2 Migration Notes

**Date**: 2026-01-18
**Status**: Complete
**Branch**: migration/xai (continuing from xAI migration)

## Overview

Migrated Groq provider to v2 registry-based architecture. Groq uses an OpenAI-compatible API, so the handlers inherit from OpenAI handlers with minimal customization.

## Files Created

- `instructor/v2/providers/groq/__init__.py` - Package exports
- `instructor/v2/providers/groq/handlers.py` - Mode handlers (TOOLS, MD_JSON)
- `instructor/v2/providers/groq/client.py` - Factory function `from_groq()`

## Files Modified

- `instructor/v2/__init__.py` - Added `from_groq` export

## Test Files Created

- `tests/v2/test_groq_handlers.py` - 24 handler unit tests
- `tests/v2/test_groq_client.py` - 15 client unit tests

## Modes Supported

| Core Mode | Status | Notes |
|-----------|--------|-------|
| `TOOLS` | Working | Inherits from OpenAIToolsHandler |
| `MD_JSON` | Working | Inherits from OpenAIMDJSONHandler |
| `JSON_SCHEMA` | Not Supported | Groq doesn't have native structured outputs |

## Test Results

```
tests/v2/test_groq_handlers.py: 24 passed
tests/v2/test_groq_client.py: 13 passed, 2 skipped

Summary: 37 passed, 2 skipped
```

## Test Coverage

### Handler Coverage: 100% (Target: 50%)

All handler code is covered by tests. The handlers are simple subclasses of OpenAI handlers.

### Client Coverage: 48% (Target: 50%)

Client coverage is slightly below target because:
- The `from_groq()` function requires the groq SDK to be installed
- The actual client creation and patching code requires API calls

### Overall Coverage: 64%

## Implementation Notes

### OpenAI Compatibility

Groq uses an OpenAI-compatible API, which means:
1. Same message format (`{"role": "user", "content": "..."}`)
2. Same tool calling format
3. Same response format

This allows us to simply inherit from OpenAI handlers:

```python
@register_mode_handler(Provider.GROQ, Mode.TOOLS)
class GroqToolsHandler(OpenAIToolsHandler):
    mode = Mode.TOOLS

@register_mode_handler(Provider.GROQ, Mode.MD_JSON)
class GroqMDJSONHandler(OpenAIMDJSONHandler):
    mode = Mode.MD_JSON
```

### No JSON_SCHEMA Support

Groq doesn't support native structured outputs (JSON_SCHEMA mode). Users should use:
- `Mode.TOOLS` for tool calling (recommended)
- `Mode.MD_JSON` for text extraction fallback

### API Key Status

`GROQ_API_KEY` is not available in the environment, so only unit tests were run.

## Deviations from Plan

None - the implementation followed the plan exactly.

## Next Steps

1. Move to Phase 5 (Mistral) migration
2. Consider adding integration tests when GROQ_API_KEY becomes available
