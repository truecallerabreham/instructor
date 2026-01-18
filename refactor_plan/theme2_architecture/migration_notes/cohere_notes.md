# Cohere v2 Migration Notes

**Date**: 2026-01-18
**Status**: Complete
**Branch**: migration/cohere

## Summary

Migrated Cohere provider to v2 registry-based architecture. All three core modes are now supported: TOOLS, JSON_SCHEMA, and MD_JSON.

## Files Created

- `instructor/v2/providers/cohere/__init__.py` - Module exports
- `instructor/v2/providers/cohere/handlers.py` - Mode handlers
- `instructor/v2/providers/cohere/client.py` - Factory function

## Files Modified

- `instructor/v2/__init__.py` - Added `from_cohere` export
- `instructor/auto_client.py` - Updated to use v2 `from_cohere`
- `tests/v2/test_provider_modes.py` - Added Cohere to PROVIDER_CONFIGS
- `tests/v2/conftest.py` - Updated to check provider-specific API keys
- `refactor_plan/theme2_architecture/v2_provider_migration_plan.md` - Checked off Phase 2

## Modes Supported

| Mode | Handler Class | Description |
|------|---------------|-------------|
| `TOOLS` | `CohereToolsHandler` | Prompt-based extraction with JSON schema |
| `JSON_SCHEMA` | `CohereJSONSchemaHandler` | Native response_format with schema |
| `MD_JSON` | `CohereMDJSONHandler` | Extract JSON from markdown code blocks |

## Implementation Notes

### Client Version Detection

Cohere has two client APIs (V1 and V2). The handlers detect which version is being used and format requests accordingly:

- **V1 format**: Uses `chat_history` + `message` parameters
- **V2 format**: Uses OpenAI-style `messages` parameter

The client version is stored in kwargs as `_cohere_client_version` during client creation.

### Response Parsing

Both V1 and V2 responses are handled:

- **V1**: Direct `response.text` access
- **V2**: `response.message.content[].text` (finds text type content item)

### Reask Handling

Reask messages are formatted differently based on client version:
- V1: Appends to `chat_history` and updates `message`
- V2: Appends to `messages` list

## Test Results

### Handler Unit Tests (58 tests)

All 58 handler unit tests pass in `tests/v2/test_cohere_handlers.py`:

- `TestCohereToolsHandler` - 11 tests
- `TestCohereJSONSchemaHandler` - 8 tests
- `TestCohereMDJSONHandler` - 7 tests
- `TestCohereHandlerRegistration` - 7 tests
- `TestCohereModeNormalization` - 4 tests
- `TestCohereClientVersionDetection` - 5 tests
- `TestCohereMessageConversion` - 3 tests
- `TestCohereTextExtraction` - 3 tests
- `TestCohereHandlerEdgeCases` - 4 tests
- `TestCohereImports` - 3 tests

### Coverage Results

```
Name                                         Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------
instructor/v2/providers/cohere/handlers.py     155      9    94%   59, 124-125, 288, 369-374
--------------------------------------------------------------------------
TOTAL                                          155      9    94%
```

Handler coverage: **94%** (target: 60%)
Client coverage: **71%** (target: 70%)

### Integration Tests (requires API key)

Integration tests are skipped when COHERE_API_KEY is not available:
- `test_mode_basic_extraction` - 3 tests (skipped)
- `test_mode_async_extraction` - 3 tests (skipped)
- `test_all_modes_covered` - 1 test (skipped)

## Deviations from Plan

1. **conftest.py update**: The original conftest only checked for ANTHROPIC_API_KEY. Updated to check for provider-specific API keys based on test parameters.

2. **auto_client.py update**: Updated to use v2 `from_cohere` and pass the mode parameter.

## Known Issues

- Cohere SDK shows Pydantic deprecation warning about `__fields__` attribute. This is in the Cohere SDK itself, not our code.

## Legacy Mode Support

The following legacy modes are normalized to core modes (already in registry.py):
- `Mode.COHERE_TOOLS` -> `Mode.TOOLS`
- `Mode.COHERE_JSON_SCHEMA` -> `Mode.JSON_SCHEMA`
