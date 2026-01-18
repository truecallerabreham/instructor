# OpenAI v2 Provider Migration Notes

**Date**: 2026-01-18
**Status**: In Progress

## Summary

Migrating OpenAI provider to v2 registry-based architecture with 5 core modes.

## Modes Implemented

| Mode | Handler Class | Status | Notes |
|------|---------------|--------|-------|
| `TOOLS` | `OpenAIToolsHandler` | Done | Supports `strict=True` parameter |
| `JSON_SCHEMA` | `OpenAIJSONSchemaHandler` | Done | Native structured outputs |
| `MD_JSON` | `OpenAIMDJSONHandler` | Done | Extract JSON from markdown |
| `PARALLEL_TOOLS` | `OpenAIParallelToolsHandler` | Done | Multiple tool calls |
| `RESPONSES_TOOLS` | `OpenAIResponsesToolsHandler` | Done | Responses API |

## Files Created

- `instructor/v2/providers/openai/__init__.py`
- `instructor/v2/providers/openai/client.py`
- `instructor/v2/providers/openai/handlers.py`

## Files Modified

- `instructor/v2/__init__.py` - Added `from_openai` export
- `instructor/v2/core/registry.py` - Added OpenAI legacy mode normalizations

## Legacy Mode Normalizations

The following legacy modes are normalized to core modes:

| Legacy Mode | Core Mode | Notes |
|-------------|-----------|-------|
| `FUNCTIONS` | `TOOLS` | Deprecated OpenAI functions API |
| `TOOLS_STRICT` | `TOOLS` | Now a parameter: `strict=True` |
| `JSON_O1` | `JSON_SCHEMA` | O1 model JSON mode |
| `RESPONSES_TOOLS_WITH_INBUILT_TOOLS` | `RESPONSES_TOOLS` | Consolidated |

Note: `Mode.JSON` is NOT normalized to `JSON_SCHEMA` globally because it breaks GenAI tests.
This will be addressed in a separate deprecation phase.

## Test Results

### Integration Tests (with API key)

```
TOOLS mode: PASS
JSON_SCHEMA mode: PASS
MD_JSON mode: PASS
```

### Unit Tests

- Client creation: PASS
- Mode registration: PASS (5 modes registered)
- Handler instantiation: PASS

### Existing Tests

- `tests/llm/test_openai/test_patch.py`: 5/5 PASS
- `tests/llm/test_openai/test_hooks.py`: All PASS
- `tests/llm/test_openai/test_attr.py`: All PASS

## Decisions Made

1. **Reused existing utilities**: Used `reask_tools`, `reask_md_json`, etc. from
   `instructor/providers/openai/utils.py` instead of reimplementing.

2. **DSL support**: Implemented streaming model tracking via `WeakKeyDictionary`
   following the Anthropic pattern.

3. **Parallel tools**: Returns `ParallelModel` wrapper for proper parsing.

4. **Responses API**: Handles `max_tokens` to `max_output_tokens` conversion.

## Known Issues

1. **Registry test conflicts**: The test `test_registry_registration` tries to
   register `Provider.OPENAI, Mode.TOOLS` which is now already registered by
   the real handlers. This test needs to be updated to use a different
   provider/mode combination.

2. **GenAI module not installed**: Some v2 tests fail because `google-genai`
   is not installed in the test environment.

## Next Steps

- [ ] Add OpenAI to `PROVIDER_CONFIGS` in tests
- [ ] Run full test suite
- [ ] Test async functionality
- [ ] Test streaming functionality
- [ ] Update registry tests to avoid conflicts

## API Usage Example

```python
import openai
from instructor import Mode
from instructor.v2.providers.openai import from_openai
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Sync client
client = from_openai(openai.OpenAI(), mode=Mode.TOOLS)
user = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: John is 25 years old"}],
)

# Async client
async_client = from_openai(openai.AsyncOpenAI(), mode=Mode.JSON_SCHEMA)
user = await async_client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=User,
    messages=[{"role": "user", "content": "Extract: Jane is 30 years old"}],
)
```
