# Fireworks v2 Migration Notes

**Date**: 2026-01-18
**Status**: Complete (unit tests only - no API key available)

## Overview

Fireworks uses an OpenAI-compatible API, so the migration was straightforward. The handlers inherit from the OpenAI handlers with minimal customization.

## Files Created

1. `instructor/v2/providers/fireworks/__init__.py` - Module exports
2. `instructor/v2/providers/fireworks/handlers.py` - Mode handlers (inherit from OpenAI)
3. `instructor/v2/providers/fireworks/client.py` - Factory function `from_fireworks()`
4. `tests/v2/test_fireworks_handlers.py` - Handler unit tests (30 tests)
5. `tests/v2/test_fireworks_client.py` - Client factory unit tests (17 tests)

## Modes Supported

| Core Mode | Legacy Mode | Notes |
|-----------|-------------|-------|
| `TOOLS` | `FIREWORKS_TOOLS` | Tool calling (OpenAI-compatible) |
| `MD_JSON` | `FIREWORKS_JSON` | Text extraction from markdown code blocks |

## Implementation Details

### Handlers

Both handlers inherit from OpenAI handlers since Fireworks uses an OpenAI-compatible API:

- `FireworksToolsHandler` extends `OpenAIToolsHandler`
- `FireworksMDJSONHandler` extends `OpenAIMDJSONHandler`

### Client Factory

The `from_fireworks()` function follows the same pattern as other v2 providers:

1. Checks if fireworks-ai SDK is installed
2. Normalizes legacy modes to core modes
3. Validates mode is registered
4. Validates client type
5. Creates patched create function
6. Returns Instructor or AsyncInstructor

### Async Client Handling

Fireworks async client uses `acreate` method instead of `create`. The client factory wraps this appropriately:

```python
async def async_create(*args: Any, **create_kwargs: Any) -> Any:
    if create_kwargs.get("stream"):
        return client.chat.completions.acreate(*args, **create_kwargs)
    return await client.chat.completions.acreate(*args, **create_kwargs)
```

## Legacy Mode Normalization

The following legacy modes are normalized with deprecation warnings:

- `Mode.FIREWORKS_TOOLS` -> `Mode.TOOLS`
- `Mode.FIREWORKS_JSON` -> `Mode.MD_JSON`

These normalizations were already in `registry.py` (lines 52-53).

## Test Results

```
45 passed, 2 skipped in 0.09s
```

The 2 skipped tests require the Fireworks SDK to be installed:
- `test_from_fireworks_with_invalid_client`
- `test_from_fireworks_with_invalid_mode`

## Deviations from Plan

None. The implementation followed the established patterns from Groq (another OpenAI-compatible provider).

## Blockers

- No API key available (`FIREWORKS_API_KEY` - MISSING)
- Cannot run integration tests without API key

## Next Steps

1. Add to `PROVIDER_CONFIGS` when API key becomes available
2. Run integration tests with real API calls
3. Verify streaming and async functionality with live API
