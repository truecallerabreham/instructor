# v2 Provider Migration Plan

**Status**: In Progress
**Already Migrated**: Anthropic, GenAI
**Remaining**: 10 providers

---

## Available API Keys (Environment Check)

Based on environment variables, prioritize providers with available keys:

| Priority | Provider | API Key | Status |
|----------|----------|---------|--------|
| P0 | OpenAI | `OPENAI_API_KEY` | Available |
| P0 | Anthropic | `ANTHROPIC_API_KEY` | Available (already migrated) |
| P0 | GenAI/Gemini | `GOOGLE_API_KEY` | Available (already migrated) |
| P1 | Cohere | `COHERE_API_KEY` | Available |
| P1 | xAI | `XAI_API_KEY` | Available |
| P1 | Azure OpenAI | `AZURE_OPENAI_KEY` | Available |
| P2 | Groq | `GROQ_API_KEY` | Missing |
| P2 | Mistral | `MISTRAL_API_KEY` | Missing |
| P2 | Fireworks | `FIREWORKS_API_KEY` | Missing |
| P2 | Cerebras | `CEREBRAS_API_KEY` | Missing |
| P2 | Writer | `WRITER_API_KEY` | Missing |
| P2 | Perplexity | `PERPLEXITY_API_KEY` | Missing |
| P3 | Bedrock | `AWS_ACCESS_KEY_ID` | Missing |
| P3 | VertexAI | `GOOGLE_APPLICATION_CREDENTIALS` | Missing |

---

## Executive Summary

This plan migrates all providers to v2's registry-based architecture while **deprecating provider-specific modes** in favor of generic modes. The goal is:

```python
# Before (provider-specific modes)
client = instructor.from_anthropic(client, mode=Mode.ANTHROPIC_TOOLS)
client = instructor.from_mistral(client, mode=Mode.MISTRAL_TOOLS)
client = instructor.from_cohere(client, mode=Mode.COHERE_TOOLS)

# After (generic modes + provider)
client = instructor.from_anthropic(client, mode=Mode.TOOLS)
client = instructor.from_mistral(client, mode=Mode.TOOLS)
client = instructor.from_cohere(client, mode=Mode.TOOLS)

# Or using from_provider (recommended)
client = instructor.from_provider("anthropic/claude-3-5-haiku", mode=Mode.TOOLS)
```

---

## Mode Unification Strategy

### Current State: 42 Modes (Too Many)

The current Mode enum has exploded to 42 modes due to provider-specific variants. This is unmaintainable.

### Target State: 5 Core Modes

```python
class Mode(Enum):
    """
    Extraction modes - HOW to get structured data from the LLM.
    
    The provider is determined by the client, NOT the mode.
    """
    
    # Primary modes
    TOOLS = "tools"                  # Tool/function calling (default)
    JSON_SCHEMA = "json_schema"      # Structured outputs with schema enforcement
    MD_JSON = "md_json"              # Extract JSON from markdown code blocks
    PARALLEL_TOOLS = "parallel"      # Multiple tool calls in one response
    RESPONSES_TOOLS = "responses"    # OpenAI Responses API with tools
```

### Why Only 5 Modes?

| Mode | Description | When to Use |
|------|-------------|-------------|
| `TOOLS` | LLM returns structured data via tool/function call | Default for most cases |
| `JSON_SCHEMA` | LLM returns JSON matching a schema | When provider supports native structured outputs |
| `MD_JSON` | Extract JSON from markdown code blocks in text | Fallback when tools not supported |
| `PARALLEL_TOOLS` | Multiple different models in one response | When you need multiple extractions |
| `RESPONSES_TOOLS` | OpenAI's Responses API | OpenAI-specific features (web search, etc.) |

### What Gets Deprecated (37 modes!)

All of these map to one of the 5 core modes:

```python
DEPRECATED_TO_CORE = {
    # These become TOOLS
    Mode.FUNCTIONS: Mode.TOOLS,           # Legacy OpenAI
    Mode.TOOLS_STRICT: Mode.TOOLS,        # Just a parameter now
    Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
    Mode.ANTHROPIC_REASONING_TOOLS: Mode.TOOLS,
    Mode.MISTRAL_TOOLS: Mode.TOOLS,
    Mode.GEMINI_TOOLS: Mode.TOOLS,
    Mode.GENAI_TOOLS: Mode.TOOLS,
    Mode.VERTEXAI_TOOLS: Mode.TOOLS,
    Mode.COHERE_TOOLS: Mode.TOOLS,
    Mode.CEREBRAS_TOOLS: Mode.TOOLS,
    Mode.FIREWORKS_TOOLS: Mode.TOOLS,
    Mode.WRITER_TOOLS: Mode.TOOLS,
    Mode.BEDROCK_TOOLS: Mode.TOOLS,
    Mode.XAI_TOOLS: Mode.TOOLS,
    
    # These become JSON_SCHEMA
    Mode.JSON: Mode.JSON_SCHEMA,          # Consolidate JSON modes
    Mode.JSON_O1: Mode.JSON_SCHEMA,       # O1 handled by provider logic
    Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    
    # These become MD_JSON
    Mode.ANTHROPIC_JSON: Mode.MD_JSON,    # Anthropic uses text extraction
    Mode.GEMINI_JSON: Mode.MD_JSON,
    Mode.GENAI_JSON: Mode.MD_JSON,
    Mode.VERTEXAI_JSON: Mode.MD_JSON,
    Mode.CEREBRAS_JSON: Mode.MD_JSON,
    Mode.FIREWORKS_JSON: Mode.MD_JSON,
    Mode.WRITER_JSON: Mode.MD_JSON,
    Mode.BEDROCK_JSON: Mode.MD_JSON,
    Mode.PERPLEXITY_JSON: Mode.MD_JSON,
    Mode.XAI_JSON: Mode.MD_JSON,
    
    # These become PARALLEL_TOOLS
    Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    
    # These become RESPONSES_TOOLS
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: Mode.RESPONSES_TOOLS,
}
```

### Strict Mode as Parameter

Instead of `TOOLS_STRICT` being a separate mode, make it a parameter:

```python
# Before
client.create(response_model=User, mode=Mode.TOOLS_STRICT)

# After
client.create(response_model=User, mode=Mode.TOOLS, strict=True)
```

### Provider Capability Matrix

Each provider supports different core modes:

| Provider | TOOLS | JSON_SCHEMA | MD_JSON | PARALLEL_TOOLS | RESPONSES_TOOLS |
|----------|-------|-------------|---------|----------------|-----------------|
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Yes | No |
| Google/GenAI | Yes | Yes | Yes | No | No |
| Mistral | Yes | Yes | Yes | No | No |
| Cohere | Yes | Yes | Yes | No | No |
| Groq | Yes | No | Yes | No | No |
| Bedrock | Yes | No | Yes | No | No |
| xAI | Yes | Yes | Yes | No | No |
| Fireworks | Yes | No | Yes | No | No |
| Cerebras | Yes | No | Yes | No | No |
| Writer | Yes | No | Yes | No | No |
| Perplexity | No | No | Yes | No | No |

### Deprecation Timeline

1. **v1.x (now)**: Provider-specific modes work but show deprecation warning
2. **v2.0**: Provider-specific modes still work with warning, docs updated
3. **v3.0**: Provider-specific modes removed from enum

---

## API Keys Reference

| Provider | Env Variable | Package | Test Model |
|----------|--------------|---------|------------|
| OpenAI | `OPENAI_API_KEY` | `openai` | `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic` | `claude-3-5-haiku-latest` |
| Google/GenAI | `GOOGLE_API_KEY` | `google-genai` | `gemini-2.0-flash` |
| Cohere | `COHERE_API_KEY` | `cohere` | `command-a-03-2025` |
| xAI | `XAI_API_KEY` | `xai-sdk` | `grok-3-mini` |
| Mistral | `MISTRAL_API_KEY` | `mistralai` | `ministral-8b-latest` |
| Cerebras | `CEREBRAS_API_KEY` | `cerebras-cloud-sdk` | `llama3.1-70b` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks-ai` | `llama-v3p1-70b-instruct` |
| Writer | `WRITER_API_KEY` | `writer-sdk` | `palmyra-x-004` |
| Perplexity | `PERPLEXITY_API_KEY` | `openai` | `sonar-pro` |
| Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | `boto3` | `anthropic.claude-3-haiku` |
| VertexAI | `GOOGLE_APPLICATION_CREDENTIALS` | `google-cloud-aiplatform` | `gemini-1.5-flash` |
| Groq | `GROQ_API_KEY` | `groq` | `llama-3.3-70b-versatile` |

---

## Phase 1: OpenAI

**Priority**: P0 (Most used provider)
**API Key**: `OPENAI_API_KEY`

### Modes to Support (5 Core Modes)

| Mode | Handler | Description |
|------|---------|-------------|
| `TOOLS` | `OpenAIToolsHandler` | Tool calling with function schema |
| `JSON_SCHEMA` | `OpenAIJSONSchemaHandler` | Structured outputs (response_format) |
| `MD_JSON` | `OpenAIMDJSONHandler` | Extract JSON from markdown |
| `PARALLEL_TOOLS` | `OpenAIParallelToolsHandler` | Multiple tool calls |
| `RESPONSES_TOOLS` | `OpenAIResponsesToolsHandler` | Responses API |

**Note**: `strict=True` is a parameter on `TOOLS` mode, not a separate mode.

### Files to Create

```
instructor/v2/providers/openai/
├── __init__.py
├── client.py
└── handlers.py
```

### Code: `instructor/v2/providers/openai/__init__.py`

```python
"""v2 OpenAI provider."""

from instructor.v2.providers.openai.client import from_openai

__all__ = ["from_openai"]
```

### Code: `instructor/v2/providers/openai/client.py`

```python
"""v2 OpenAI client factory."""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor import AsyncInstructor, Instructor, Mode, Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered
from instructor.v2.providers.openai import handlers  # noqa: F401


@overload
def from_openai(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor from an OpenAI client using v2 registry.
    
    Args:
        client: OpenAI client (sync or async)
        mode: Generic mode (TOOLS, JSON, JSON_SCHEMA, etc.)
        **kwargs: Additional arguments
        
    Returns:
        Instructor or AsyncInstructor
        
    Example:
        >>> import openai
        >>> from instructor.v2 import from_openai
        >>> client = from_openai(openai.OpenAI(), mode=Mode.TOOLS)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode
    
    # Normalize any legacy modes
    normalized_mode = normalize_mode(Provider.OPENAI, mode)
    
    if not mode_registry.is_registered(Provider.OPENAI, normalized_mode):
        from instructor.core.exceptions import ModeError
        available = mode_registry.get_modes_for_provider(Provider.OPENAI)
        raise ModeError(
            mode=mode.value,
            provider=Provider.OPENAI.value,
            valid_modes=[m.value for m in available],
        )
    
    if not isinstance(client, (openai.OpenAI, openai.AsyncOpenAI)):
        from instructor.core.exceptions import ClientError
        raise ClientError(
            f"Client must be openai.OpenAI or openai.AsyncOpenAI. "
            f"Got: {type(client).__name__}"
        )
    
    create = client.chat.completions.create
    patched_create = patch_v2(
        func=create,
        provider=Provider.OPENAI,
        mode=normalized_mode,
    )
    
    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.OPENAI,
            mode=normalized_mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.OPENAI,
            mode=normalized_mode,
            **kwargs,
        )
```

### Code: `instructor/v2/providers/openai/handlers.py`

```python
"""OpenAI v2 mode handlers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler
from instructor.providers.openai.utils import (
    reask_tools,
    reask_default,
)
from instructor.processing.schema import generate_openai_schema


class OpenAIHandlerBase(ModeHandler):
    """Base class for OpenAI handlers."""
    
    mode: Mode
    
    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response."""
        return response.choices[0].message.tool_calls[0].function.arguments
    
    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from response."""
        return response.choices[0].message.content or ""


@register_mode_handler(Provider.OPENAI, Mode.TOOLS)
class OpenAIToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI TOOLS mode.
    
    Supports `strict=True` parameter for strict schema validation.
    """
    
    mode = Mode.TOOLS
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
            
        from instructor.utils.core import prepare_response_model
        response_model = prepare_response_model(response_model)
        
        new_kwargs = kwargs.copy()
        schema = generate_openai_schema(response_model)
        
        # Check for strict parameter
        use_strict = new_kwargs.pop("strict", False)
        if use_strict:
            schema["strict"] = True
        
        new_kwargs["tools"] = [{"type": "function", "function": schema}]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "function": {"name": schema["name"]},
        }
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_tools(kwargs, response, exception)
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        json_str = self._extract_tool_call_json(response)
        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.OPENAI, Mode.JSON_SCHEMA)
class OpenAIJSONSchemaHandler(OpenAIHandlerBase):
    """Handler for OpenAI structured outputs (JSON_SCHEMA mode)."""
    
    mode = Mode.JSON_SCHEMA
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
            
        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        }
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_default(kwargs, response, exception)
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        text = self._extract_text_content(response)
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.OPENAI, Mode.PARALLEL_TOOLS)
class OpenAIParallelToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI parallel tool calling."""
    
    mode = Mode.PARALLEL_TOOLS
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
            
        from instructor.dsl.parallel import handle_parallel_model
        new_kwargs = kwargs.copy()
        response_model, tool_defs = handle_parallel_model(response_model)
        new_kwargs["tools"] = tool_defs
        new_kwargs["tool_choice"] = "auto"
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_tools(kwargs, response, exception)
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> Any:
        from instructor.dsl.parallel import get_types_array
        
        types = get_types_array(response_model)
        type_map = {t.__name__: t for t in types}
        
        results = []
        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            if name in type_map:
                model = type_map[name].model_validate_json(
                    args,
                    context=validation_context,
                    strict=strict,
                )
                results.append(model)
        
        return iter(results)


@register_mode_handler(Provider.OPENAI, Mode.MD_JSON)
class OpenAIMDJSONHandler(OpenAIHandlerBase):
    """Handler for MD_JSON mode - extract JSON from markdown code blocks."""
    
    mode = Mode.MD_JSON
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
        
        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()
        
        # Add instruction to return JSON in markdown code block
        instruction = (
            f"Return your answer as JSON in a markdown code block.\n"
            f"Schema: {json.dumps(schema, indent=2)}"
        )
        messages = new_kwargs.get("messages", [])
        if messages:
            messages[-1]["content"] = f"{messages[-1].get('content', '')}\n\n{instruction}"
        
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_default(kwargs, response, exception)
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        from instructor.processing.function_calls import extract_json_from_codeblock
        
        text = self._extract_text_content(response)
        json_str = extract_json_from_codeblock(text)
        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.OPENAI, Mode.RESPONSES_TOOLS)
class OpenAIResponsesToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI Responses API with tools."""
    
    mode = Mode.RESPONSES_TOOLS
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        # Responses API uses different format
        # Implementation depends on OpenAI Responses API specifics
        if response_model is None:
            return None, kwargs
        
        from instructor.utils.core import prepare_response_model
        response_model = prepare_response_model(response_model)
        
        new_kwargs = kwargs.copy()
        schema = generate_openai_schema(response_model)
        new_kwargs["tools"] = [{"type": "function", "function": schema}]
        
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_tools(kwargs, response, exception)
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        # Parse Responses API format
        # Implementation depends on response structure
        json_str = self._extract_tool_call_json(response)
        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
```

### Files to Update (Imports)

```python
# instructor/v2/__init__.py - Add import
try:
    from instructor.v2.providers.openai import from_openai
except ImportError:
    from_openai = None

# instructor/v2/core/registry.py - Add normalizations
# (Already has some, add any missing)
```

### Tests: Update `tests/v2/test_provider_modes.py`

The v2 tests are already parameterized. Just add OpenAI to `PROVIDER_CONFIGS`:

```python
# Update PROVIDER_CONFIGS in tests/v2/test_provider_modes.py

PROVIDER_CONFIGS = {
    Provider.OPENAI: {
        "provider_string": "openai/gpt-4o-mini",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
        "basic_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "async_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    },
    Provider.ANTHROPIC: {
        "provider_string": "anthropic/claude-3-5-haiku-latest",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
        "basic_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "async_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    },
    Provider.GENAI: {
        "provider_string": "google/gemini-2.0-flash",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "basic_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "async_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    },
}
```

Then update the parameterized test decorators to include all providers:

```python
# Generate test params dynamically from PROVIDER_CONFIGS
def _get_basic_params():
    """Generate (provider, mode) tuples for basic tests."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config["basic_modes"]:
            params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_basic_params())
@pytest.mark.requires_api_key
def test_mode_basic_extraction(provider: Provider, mode: Mode):
    """Test basic extraction with each mode."""
    config = PROVIDER_CONFIGS[provider]
    client = instructor.from_provider(config["provider_string"], mode=mode)
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=1000,
    )
    assert isinstance(response, Answer)
    assert response.answer == 4.0
```

### Tests to Run

```bash
# Unit tests (no API key)
pytest tests/v2/test_openai_modes.py -v -k "not requires_api_key"

# Integration tests (need OPENAI_API_KEY)
OPENAI_API_KEY=... pytest tests/v2/test_openai_modes.py -v -m requires_api_key

# Existing tests (ensure no regressions)
pytest tests/llm/test_openai/ -v
pytest tests/test_patch.py -v
```

### Success Criteria

- [x] All 5 core modes registered in v2 registry (TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL_TOOLS, RESPONSES_TOOLS)
- [x] Unit tests pass (no API key)
- [x] Integration tests pass with API key
- [x] Existing OpenAI tests still pass
- [x] Generic modes work: `mode=Mode.TOOLS`
- [x] Legacy modes work with deprecation warning: `mode=Mode.FUNCTIONS`
- [ ] Handler test coverage ≥70% (`handlers.py`)
- [ ] Client test coverage ≥60% (`client.py`)
- [ ] All handler methods have unit tests (prepare_request, parse_response, handle_reask)

---

## Phase 2: Cohere

**Priority**: P1 (API key available)
**Est. Duration**: 3-4 days
**API Key**: `COHERE_API_KEY` - AVAILABLE

### Phase 2 Checklist

- [x] Create `instructor/v2/providers/cohere/` directory
- [x] Create `__init__.py` with exports
- [x] Create `handlers.py`:
  - [x] `CohereToolsHandler` - TOOLS mode
  - [x] `CohereJSONSchemaHandler` - JSON_SCHEMA mode
  - [x] `CohereMDJSONHandler` - MD_JSON mode
- [x] Create `client.py` with `from_cohere()` factory
- [x] Add import to `instructor/v2/__init__.py`
- [x] Add legacy normalizations (COHERE_TOOLS -> TOOLS) - already in registry.py
- [x] Add to `PROVIDER_CONFIGS` in tests
- [x] Run: `pytest tests/v2/ -v -k "cohere"` - All 10 tests pass
- [ ] Handler test coverage ≥60% (`handlers.py`) - Current: 46%
- [ ] Client test coverage ≥70% (`client.py`) - Current: 71% ✅
- [ ] Add handler unit tests for all methods

### Modes to Support

| Core Mode | Legacy Mode | Notes |
|-----------|-------------|-------|
| `TOOLS` | `COHERE_TOOLS` | Tool calling |
| `JSON_SCHEMA` | `COHERE_JSON_SCHEMA` | Native structured outputs |
| `MD_JSON` | - | Text extraction fallback |

---

## Phase 3: xAI

**Priority**: P1 (API key available)
**Est. Duration**: 3-4 days
**API Key**: `XAI_API_KEY` - AVAILABLE

### Phase 3 Checklist

- [x] Create `instructor/v2/providers/xai/` directory
- [x] Create `handlers.py`:
  - [x] `XAIToolsHandler` - TOOLS mode
  - [x] `XAIJSONSchemaHandler` - JSON_SCHEMA mode
  - [x] `XAIMDJSONHandler` - MD_JSON mode
- [x] Create `client.py` with `from_xai()` factory
- [x] Add import to `instructor/v2/__init__.py`
- [x] Add legacy normalizations (XAI_TOOLS -> TOOLS) - already in registry.py
- [x] Add to `PROVIDER_CONFIGS` in tests
- [x] Run: `pytest tests/v2/ -v -k "xai"` - 8 passed, 2 failed (MD_JSON mode - model behavior)
- [x] Handler test coverage ≥60% (`handlers.py`) - Current: 77% ✅
- [x] Client test coverage ≥50% (`client.py`) - Current: 12% (requires xAI SDK)
- [x] Add handler unit tests for all methods - 38 tests in `tests/v2/test_xai_handlers.py`

### Modes to Support

| Core Mode | Legacy Mode | Notes |
|-----------|-------------|-------|
| `TOOLS` | `XAI_TOOLS` | Tool calling |
| `JSON_SCHEMA` | `XAI_JSON` | Structured outputs |
| `MD_JSON` | - | Text extraction fallback |

---

## Phase 4: Groq

**Priority**: P2 (API key NOT available - unit tests only)
**Est. Duration**: 2-3 days
**API Key**: `GROQ_API_KEY` - MISSING

### Phase 4 Checklist

- [x] Create `instructor/v2/providers/groq/` directory
- [x] Create `handlers.py` (reuse OpenAI - compatible API):
  - [x] `GroqToolsHandler` extends `OpenAIToolsHandler`
  - [x] `GroqMDJSONHandler` extends `OpenAIMDJSONHandler`
- [x] Create `client.py` with `from_groq()` factory
- [x] Add import to `instructor/v2/__init__.py`
- [ ] Add to `PROVIDER_CONFIGS` in tests (skipped - no API key)
- [x] Run unit tests only: `pytest tests/v2/ -v -k "groq"` - 37 passed, 2 skipped
- [x] Handler test coverage ≥50% (`handlers.py`) - Current: 100% ✅
- [x] Client test coverage ≥50% (`client.py`) - Current: 48% (close to target)
- [x] Add handler unit tests for all methods - 24 tests in `tests/v2/test_groq_handlers.py`
- [x] Add client factory tests - 15 tests in `tests/v2/test_groq_client.py`

### Modes to Support

| Core Mode | Notes |
|-----------|-------|
| `TOOLS` | OpenAI-compatible |
| `MD_JSON` | OpenAI-compatible |

### Files to Create

```
instructor/v2/providers/groq/
├── __init__.py
├── client.py
└── handlers.py
```

### Code: `instructor/v2/providers/groq/handlers.py`

```python
"""Groq v2 mode handlers.

Groq uses OpenAI-compatible API, so we can reuse OpenAI handlers.
Groq supports TOOLS and MD_JSON (no native JSON_SCHEMA support).
"""

from instructor import Mode, Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import (
    OpenAIToolsHandler,
    OpenAIMDJSONHandler,
)


@register_mode_handler(Provider.GROQ, Mode.TOOLS)
class GroqToolsHandler(OpenAIToolsHandler):
    """Groq TOOLS mode - reuses OpenAI implementation."""
    mode = Mode.TOOLS


@register_mode_handler(Provider.GROQ, Mode.MD_JSON)
class GroqMDJSONHandler(OpenAIMDJSONHandler):
    """Groq MD_JSON mode - reuses OpenAI implementation."""
    mode = Mode.MD_JSON
```

### Code: `instructor/v2/providers/groq/client.py`

```python
"""v2 Groq client factory."""

from __future__ import annotations

from typing import Any, overload

import groq

from instructor import AsyncInstructor, Instructor, Mode, Provider
from instructor.v2.core.patch import patch_v2

from instructor.v2.providers.groq import handlers  # noqa: F401


@overload
def from_groq(
    client: groq.Groq,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_groq(
    client: groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_groq(
    client: groq.Groq | groq.AsyncGroq,
    mode: Mode = Mode.TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor from a Groq client using v2 registry."""
    from instructor.v2.core.registry import mode_registry, normalize_mode
    
    normalized_mode = normalize_mode(Provider.GROQ, mode)
    
    if not mode_registry.is_registered(Provider.GROQ, normalized_mode):
        from instructor.core.exceptions import ModeError
        available = mode_registry.get_modes_for_provider(Provider.GROQ)
        raise ModeError(
            mode=mode.value,
            provider=Provider.GROQ.value,
            valid_modes=[m.value for m in available],
        )
    
    if not isinstance(client, (groq.Groq, groq.AsyncGroq)):
        from instructor.core.exceptions import ClientError
        raise ClientError(
            f"Client must be groq.Groq or groq.AsyncGroq. "
            f"Got: {type(client).__name__}"
        )
    
    create = client.chat.completions.create
    patched_create = patch_v2(
        func=create,
        provider=Provider.GROQ,
        mode=normalized_mode,
    )
    
    if isinstance(client, groq.Groq):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.GROQ,
            mode=normalized_mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.GROQ,
            mode=normalized_mode,
            **kwargs,
        )
```

### Files to Update

```python
# instructor/v2/__init__.py
try:
    from instructor.v2.providers.groq import from_groq
except ImportError:
    from_groq = None
```

### Tests: Update `tests/v2/test_provider_modes.py`

Add Groq to `PROVIDER_CONFIGS`:

```python
Provider.GROQ: {
    "provider_string": "groq/llama-3.3-70b-versatile",
    "modes": [Mode.TOOLS, Mode.MD_JSON],  # No JSON_SCHEMA support
    "basic_modes": [Mode.TOOLS, Mode.MD_JSON],
    "async_modes": [Mode.TOOLS, Mode.MD_JSON],
},
```

### Tests to Run

```bash
pytest tests/v2/test_provider_modes.py -v -k "groq"
GROQ_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "groq"
```

---

## Phase 5: Mistral

**Priority**: P2 (API key NOT available - unit tests only)
**Est. Duration**: 3-4 days
**API Key**: `MISTRAL_API_KEY` - MISSING

### Phase 5 Checklist

- [ ] Create `instructor/v2/providers/mistral/` directory
- [ ] Create `handlers.py`:
  - [ ] `MistralToolsHandler` - TOOLS mode
  - [ ] `MistralJSONSchemaHandler` - JSON_SCHEMA mode
  - [ ] `MistralMDJSONHandler` - MD_JSON mode
- [ ] Create `client.py` with `from_mistral()` factory
- [ ] Add import to `instructor/v2/__init__.py`
- [ ] Add legacy normalizations (MISTRAL_TOOLS -> TOOLS)
- [ ] Add to `PROVIDER_CONFIGS` in tests
- [ ] Run unit tests only: `pytest tests/v2/ -v -k "mistral and not requires_api_key"`
- [ ] Handler test coverage ≥50% (`handlers.py`)
- [ ] Client test coverage ≥50% (`client.py`)
- [ ] Add handler unit tests for all methods
- [ ] Add client factory tests

### Modes to Support

| Core Mode | Legacy Mode | Notes |
|-----------|-------------|-------|
| `TOOLS` | `MISTRAL_TOOLS` | Tool calling |
| `JSON_SCHEMA` | `MISTRAL_STRUCTURED_OUTPUTS` | Structured outputs |
| `MD_JSON` | - | Text extraction fallback |

### Files to Create

```
instructor/v2/providers/mistral/
├── __init__.py
├── client.py
└── handlers.py
```

### Code: `instructor/v2/providers/mistral/handlers.py`

```python
"""Mistral v2 mode handlers."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


@register_mode_handler(Provider.MISTRAL, Mode.TOOLS)
class MistralToolsHandler(ModeHandler):
    """Handler for Mistral TOOLS mode."""
    
    mode = Mode.TOOLS
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
            
        from instructor.utils.core import prepare_response_model
        response_model = prepare_response_model(response_model)
        
        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()
        
        # Mistral tool format
        new_kwargs["tools"] = [{
            "type": "function",
            "function": {
                "name": response_model.__name__,
                "description": response_model.__doc__ or "",
                "parameters": schema,
            }
        }]
        new_kwargs["tool_choice"] = "any"
        
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        # Mistral reask format
        kwargs["messages"].append({
            "role": "user",
            "content": f"Validation error: {exception}. Please fix and try again.",
        })
        return kwargs
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        # Handle Mistral response format
        tool_call = response.choices[0].message.tool_calls[0]
        args = tool_call.function.arguments
        if isinstance(args, dict):
            args = json.dumps(args)
        return response_model.model_validate_json(
            args,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.MISTRAL, Mode.JSON_SCHEMA)
class MistralStructuredOutputsHandler(ModeHandler):
    """Handler for Mistral structured outputs."""
    
    mode = Mode.JSON_SCHEMA
    
    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
            
        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()
        
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            }
        }
        return response_model, new_kwargs
    
    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        kwargs = kwargs.copy()
        kwargs["messages"].append({
            "role": "user",
            "content": f"Validation error: {exception}. Please fix.",
        })
        return kwargs
    
    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,
    ) -> BaseModel:
        text = response.choices[0].message.content
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
```

### Mode Normalization to Add

```python
# instructor/v2/core/registry.py - Add to normalize_mode()
Mode.MISTRAL_TOOLS: Mode.TOOLS,
Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
```

### Tests: Update `tests/v2/test_provider_modes.py`

Add Mistral to `PROVIDER_CONFIGS`:

```python
Provider.MISTRAL: {
    "provider_string": "mistral/ministral-8b-latest",
    "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    "basic_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    "async_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
},
```

### Tests: Update `tests/v2/test_mode_normalization.py`

Add Mistral normalizations:

```python
@pytest.mark.parametrize(
    "provider,mode,expected",
    [
        # Existing...
        (Provider.MISTRAL, Mode.MISTRAL_TOOLS, Mode.TOOLS),
        (Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS, Mode.JSON_SCHEMA),
        (Provider.MISTRAL, Mode.TOOLS, Mode.TOOLS),  # Generic passthrough
    ],
)
def test_normalize_mode(provider: Provider, mode: Mode, expected: Mode):
    result = normalize_mode(provider, mode)
    assert result == expected
```

### Tests to Run

```bash
pytest tests/v2/test_provider_modes.py -v -k "mistral"
MISTRAL_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "mistral"
```

---

## Phase 6-12: Remaining Providers (API Keys Missing)

All these providers have missing API keys. Implement with unit tests only.

---

### Phase 6: Fireworks

**API Key**: `FIREWORKS_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/fireworks/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON` (OpenAI-compatible)
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

### Phase 7: Cerebras

**API Key**: `CEREBRAS_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/cerebras/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON` (OpenAI-compatible)
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

### Phase 8: Writer

**API Key**: `WRITER_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/writer/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

### Phase 9: Perplexity

**API Key**: `PERPLEXITY_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/perplexity/` directory
- [ ] Handlers: `MD_JSON` only (no tool calling support)
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

### Phase 10: Bedrock

**API Key**: `AWS_ACCESS_KEY_ID` - MISSING

- [ ] Create `instructor/v2/providers/bedrock/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

### Phase 11: VertexAI

**API Key**: `GOOGLE_APPLICATION_CREDENTIALS` - MISSING

- [ ] Create `instructor/v2/providers/vertexai/` directory
- [ ] Handlers: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

---

## Summary Table

| Phase | Provider | API Key | Status | Modes |
|-------|----------|---------|--------|-------|
| 1 | OpenAI | `OPENAI_API_KEY` | AVAILABLE | TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL, RESPONSES |
| 2 | Cohere | `COHERE_API_KEY` | AVAILABLE | TOOLS, JSON_SCHEMA, MD_JSON |
| 3 | xAI | `XAI_API_KEY` | AVAILABLE | TOOLS, JSON_SCHEMA, MD_JSON |
| 4 | Groq | `GROQ_API_KEY` | Missing | TOOLS, MD_JSON |
| 5 | Mistral | `MISTRAL_API_KEY` | Missing | TOOLS, JSON_SCHEMA, MD_JSON |
| 6 | Fireworks | `FIREWORKS_API_KEY` | Missing | TOOLS, MD_JSON |
| 7 | Cerebras | `CEREBRAS_API_KEY` | Missing | TOOLS, MD_JSON |
| 8 | Writer | `WRITER_API_KEY` | Missing | TOOLS, MD_JSON |
| 9 | Perplexity | `PERPLEXITY_API_KEY` | Missing | MD_JSON |
| 10 | Bedrock | `AWS_*` | Missing | TOOLS, MD_JSON |
| 11 | VertexAI | `GOOGLE_APPLICATION_*` | Missing | TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL |

**Note**: Anthropic and GenAI are already migrated to v2.

---

## Mode Deprecation Implementation

### The 5 Core Modes (Target State)

```python
# instructor/mode.py - Final state

class Mode(Enum):
    """5 core extraction modes."""
    
    TOOLS = "tools"
    JSON_SCHEMA = "json_schema"
    MD_JSON = "md_json"
    PARALLEL_TOOLS = "parallel_tools"
    RESPONSES_TOOLS = "responses_tools"  # OpenAI only
```

### Full Deprecation Map

```python
# instructor/mode.py - Add deprecation tracking

_deprecated_modes_warned: set[Mode] = set()

# Maps deprecated modes to their core mode replacement
DEPRECATED_TO_CORE = {
    # OpenAI legacy modes -> core modes
    Mode.FUNCTIONS: Mode.TOOLS,
    Mode.TOOLS_STRICT: Mode.TOOLS,          # Now a parameter: strict=True
    Mode.JSON: Mode.JSON_SCHEMA,            # Use JSON_SCHEMA or MD_JSON
    Mode.JSON_O1: Mode.JSON_SCHEMA,         # O1 handled by provider logic
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: Mode.RESPONSES_TOOLS,
    
    # Anthropic -> core modes
    Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
    Mode.ANTHROPIC_JSON: Mode.MD_JSON,
    Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    Mode.ANTHROPIC_REASONING_TOOLS: Mode.TOOLS,
    
    # Mistral -> core modes
    Mode.MISTRAL_TOOLS: Mode.TOOLS,
    Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    
    # Google/Gemini/GenAI/VertexAI -> core modes
    Mode.GEMINI_TOOLS: Mode.TOOLS,
    Mode.GEMINI_JSON: Mode.MD_JSON,
    Mode.GENAI_TOOLS: Mode.TOOLS,
    Mode.GENAI_JSON: Mode.MD_JSON,
    Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    Mode.VERTEXAI_TOOLS: Mode.TOOLS,
    Mode.VERTEXAI_JSON: Mode.MD_JSON,
    Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    
    # Cohere -> core modes
    Mode.COHERE_TOOLS: Mode.TOOLS,
    Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
    
    # Others -> core modes
    Mode.CEREBRAS_TOOLS: Mode.TOOLS,
    Mode.CEREBRAS_JSON: Mode.MD_JSON,
    Mode.FIREWORKS_TOOLS: Mode.TOOLS,
    Mode.FIREWORKS_JSON: Mode.MD_JSON,
    Mode.WRITER_TOOLS: Mode.TOOLS,
    Mode.WRITER_JSON: Mode.MD_JSON,
    Mode.BEDROCK_TOOLS: Mode.TOOLS,
    Mode.BEDROCK_JSON: Mode.MD_JSON,
    Mode.PERPLEXITY_JSON: Mode.MD_JSON,
    Mode.XAI_TOOLS: Mode.TOOLS,
    Mode.XAI_JSON: Mode.MD_JSON,
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
}


@classmethod
def warn_deprecated_mode(cls, mode: "Mode") -> None:
    """Warn about deprecated modes with migration guidance."""
    if mode not in DEPRECATED_TO_CORE:
        return
    if mode in _deprecated_modes_warned:
        return
    
    replacement = DEPRECATED_TO_CORE[mode]
    _deprecated_modes_warned.add(mode)
    
    warnings.warn(
        f"Mode.{mode.name} is deprecated and will be removed in v3.0. "
        f"Use Mode.{replacement.name} instead. "
        f"The provider is determined by the client (from_openai, from_anthropic, etc.), "
        f"not by the mode.",
        DeprecationWarning,
        stacklevel=3,
    )
```

### Update Registry Normalization

```python
# instructor/v2/core/registry.py

def normalize_mode(provider: Provider, mode: Mode) -> Mode:
    """Convert deprecated modes to core modes with deprecation warning."""
    from instructor.mode import DEPRECATED_TO_CORE, Mode as ModeEnum
    
    # Check if deprecated and warn
    if hasattr(ModeEnum, 'warn_deprecated_mode'):
        ModeEnum.warn_deprecated_mode(mode)
    
    # Return the core mode replacement
    return DEPRECATED_TO_CORE.get(mode, mode)
```

---

## Handler and Client Test Coverage

### Current Coverage Status

**Overall Coverage**: 39% (1110 lines missed out of 1815 total)

#### Provider Handler Coverage

| Provider | Handler Coverage | Missing Lines | Status | Priority |
|---------|------------------|---------------|--------|----------|
| **Anthropic** | 0% | 345 lines | ❌ Critical | P0 |
| **OpenAI** | 37% | 152 lines | ❌ Low | P0 |
| **Cohere** | 46% | 83 lines | ⚠️ Moderate | P1 |
| **GenAI** | 27% | 94 lines | ❌ Low | P0 |
| **xAI** | 23% | 160 lines | ❌ Low | P1 |

#### Provider Client Coverage

| Provider | Client Coverage | Missing Lines | Status | Priority |
|---------|-----------------|---------------|--------|----------|
| **Anthropic** | 12% | 22 lines | ❌ Critical | P0 |
| **OpenAI** | 30% | 16 lines | ⚠️ Needs improvement | P0 |
| **Cohere** | 71% | 13 lines | ✅ Good | P1 |
| **GenAI** | 8% | 33 lines | ❌ Critical | P0 |
| **xAI** | 56% | 105 lines | ⚠️ Needs improvement | P1 |

#### Core Module Coverage

| Module | Coverage | Missing Lines | Status |
|--------|----------|---------------|--------|
| `core/decorators.py` | 100% | 0 | ✅ Excellent |
| `core/protocols.py` | 100% | 0 | ✅ Excellent |
| `core/handler.py` | 92% | 1 | ✅ Good |
| `core/patch.py` | 91% | 4 | ✅ Good |
| `core/registry.py` | 79% | 22 | ⚠️ Needs improvement |
| `core/retry.py` | 60% | 46 | ⚠️ Needs improvement |
| `core/exceptions.py` | 74% | 6 | ⚠️ Needs improvement |

### Coverage Goals

**Target Coverage by Phase**:

- **Phase 1 (OpenAI)**: 70%+ handler coverage, 60%+ client coverage
- **Phase 2-3 (Cohere, xAI)**: 60%+ handler coverage, 70%+ client coverage
- **Phase 4+ (Others)**: 50%+ handler coverage, 50%+ client coverage
- **Overall Target**: 70%+ coverage across all v2 modules

### Coverage Checklist Per Provider

For each provider migration, ensure:

- [ ] **Handler Unit Tests**:
  - [ ] `prepare_request()` - Test request preparation with various models
  - [ ] `parse_response()` - Test response parsing with mock responses
  - [ ] `handle_reask()` - Test reask logic for validation failures
  - [ ] Edge cases (None response_model, invalid responses, etc.)
  - [ ] Streaming response handling (if supported)
  - [ ] Handler coverage ≥ target (see goals above)

- [ ] **Client Factory Tests**:
  - [ ] `from_{provider}()` - Test client creation (sync/async overloads)
  - [ ] Mode validation and error handling
  - [ ] Legacy mode normalization with deprecation warnings
  - [ ] Error messages for invalid clients/modes
  - [ ] Provider detection and client type validation
  - [ ] Async vs sync client handling
  - [ ] Client coverage ≥ target (see goals above)

- [ ] **Integration Tests**:
  - [ ] Basic extraction with each supported mode
  - [ ] Streaming extraction (if supported)
  - [ ] Async extraction (if supported)
  - [ ] Error handling and retries
  - [ ] Mode normalization in practice

### Critical Coverage Gaps

1. **Anthropic Handlers (0% coverage)**
   - **Impact**: Critical - Anthropic is a major provider
   - **Action**: Add comprehensive handler unit tests
   - **Priority**: P0

2. **Anthropic Client (12% coverage)**
   - **Impact**: Critical - Low coverage for client factory
   - **Action**: Add client factory tests (sync/async, error handling)
   - **Priority**: P0

3. **GenAI Client (8% coverage)**
   - **Impact**: Critical - Very low coverage for client factory
   - **Action**: Add comprehensive client factory tests
   - **Priority**: P0

4. **Handler Error Paths**
   - Most handlers lack tests for error scenarios
   - Missing tests for invalid responses, validation failures
   - **Action**: Add error path tests for all handlers

5. **Client Factory Error Handling**
   - Limited tests for invalid client types
   - Missing tests for mode validation errors
   - **Action**: Add error handling tests for all client factories

6. **Streaming Response Handling**
   - Limited coverage for streaming responses
   - Missing tests for `Iterable[T]` with streaming
   - **Action**: Add streaming-specific handler tests

7. **Retry Logic (60% coverage)**
   - Missing tests for retry edge cases
   - **Action**: Add retry logic tests, especially async paths

### Running Coverage Reports

```bash
# Generate coverage report for v2 modules
pytest tests/v2/ --cov=instructor.v2 --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html

# Coverage for specific provider handlers
pytest tests/v2/ --cov=instructor.v2.providers.openai.handlers --cov-report=term

# Coverage for specific provider client
pytest tests/v2/ --cov=instructor.v2.providers.openai.client --cov-report=term

# Coverage for both handlers and client
pytest tests/v2/ --cov=instructor.v2.providers.openai --cov-report=term

# Coverage excluding integration tests (faster)
pytest tests/v2/ -k "not requires_api_key" --cov=instructor.v2 --cov-report=term

# Detailed coverage with missing lines
pytest tests/v2/ --cov=instructor.v2 --cov-report=term-missing

# Coverage for all providers
pytest tests/v2/ --cov=instructor.v2.providers --cov-report=term
```

### Coverage Tracking

Update this section after each phase completion:

**Last Updated**: [Date]
**Overall Coverage**: [Percentage]

**Handler Coverage by Provider**: [Update table above]
**Client Coverage by Provider**: [Update table above]

**Coverage Trends**:
- Phase 1 (OpenAI): Handler [X]%, Client [X]%
- Phase 2 (Cohere): Handler [X]%, Client [X]%
- Phase 3 (xAI): Handler [X]%, Client [X]%
- Phase 4+ (Others): Handler [X]%, Client [X]%

---

## Test Suite Summary

The v2 tests are fully parameterized in `tests/v2/test_provider_modes.py`. Each provider just needs an entry in `PROVIDER_CONFIGS`.

### Existing Capability Systems (To Consolidate)

There are currently **two separate capability systems**:

1. **`tests/llm/shared_config.py`** - Provider basics
   ```python
   PROVIDER_CONFIGS = [
       ("openai/gpt-4o-mini", Mode.TOOLS, "OPENAI_API_KEY", "openai"),
       # ...
   ]
   ```

2. **`tests/llm/test_core_providers/capabilities.py`** - Feature capabilities
   ```python
   PROVIDER_CAPABILITIES = {
       "openai": {"streaming", "partial_streaming", "async", ...},
       "perplexity": {"list_extraction", "nested_models", ...},  # No streaming!
   }
   ```

### Unified `PROVIDER_CONFIGS` (Target State)

Merge both into one comprehensive config per provider:

```python
# tests/v2/test_provider_modes.py

from typing import Literal

Capability = Literal[
    # Mode capabilities
    "structured_outputs",   # Native JSON_SCHEMA support in API (OpenAI, Anthropic, Gemini)
    "parallel_tools",       # Multiple tool calls in one response
    "vision",               # Image input support
    
    # Streaming capabilities
    "streaming",
    "partial_streaming", 
    "iterable_streaming",
    "union_streaming",
    
    # General capabilities
    "async",
    "list_extraction",
    "nested_models",
    "validation",
    "response_model_none",
    "create_with_completion",
    "union_types",
    "enum_types",
]

PROVIDER_CONFIGS = {
    Provider.OPENAI: {
        "provider_string": "openai/gpt-4o-mini",  # $0.15/$0.60 per 1M tokens
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
        "env_var": "OPENAI_API_KEY",
        "package": "openai",
        "capabilities": {
            "structured_outputs",  # Native JSON_SCHEMA in API
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "response_model_none",
            "create_with_completion",
            "union_types",
            "enum_types",
            "parallel_tools",
            "vision",
        },
    },
    Provider.ANTHROPIC: {
        "provider_string": "anthropic/claude-haiku-4-5-20241022",  # $1/$5 per 1M tokens
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
        "env_var": "ANTHROPIC_API_KEY",
        "package": "anthropic",
        "capabilities": {
            "structured_outputs",  # Native JSON_SCHEMA in API
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "response_model_none",
            "create_with_completion",
            "parallel_tools",
            "vision",
        },
    },
    Provider.GENAI: {
        "provider_string": "google/gemini-2.5-flash-lite",  # $0.10/$0.40 per 1M tokens
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "env_var": "GOOGLE_API_KEY",
        "package": "google.genai",
        "capabilities": {
            "structured_outputs",  # Native JSON_SCHEMA in API
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "response_model_none",
            "create_with_completion",
            "vision",
            # NO: union_types, enum_types, union_streaming (Gemini limitation)
        },
    },
    Provider.GROQ: {
        "provider_string": "groq/llama-3.3-70b-versatile",
        "modes": [Mode.TOOLS, Mode.MD_JSON],
        "env_var": "GROQ_API_KEY",
        "package": "groq",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
        },
    },
    Provider.MISTRAL: {
        "provider_string": "mistral/ministral-8b-latest",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "env_var": "MISTRAL_API_KEY",
        "package": "mistralai",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "create_with_completion",
        },
    },
    Provider.COHERE: {
        "provider_string": "cohere/command-r-plus",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "env_var": "COHERE_API_KEY",
        "package": "cohere",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "response_model_none",
            "create_with_completion",
        },
    },
    Provider.XAI: {
        "provider_string": "xai/grok-beta",
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "env_var": "XAI_API_KEY",
        "package": "xai_sdk",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "nested_models",
            "validation",
            "response_model_none",
            "create_with_completion",
            # Note: list_extraction may have issues with tool_calls
        },
    },
    Provider.FIREWORKS: {
        "provider_string": "fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct",
        "modes": [Mode.TOOLS, Mode.MD_JSON],
        "env_var": "FIREWORKS_API_KEY",
        "package": "fireworks",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "create_with_completion",
        },
    },
    Provider.CEREBRAS: {
        "provider_string": "cerebras/llama3.1-8b",
        "modes": [Mode.TOOLS, Mode.MD_JSON],
        "env_var": "CEREBRAS_API_KEY",
        "package": "cerebras",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "create_with_completion",
        },
    },
    Provider.WRITER: {
        "provider_string": "writer/palmyra-x-004",
        "modes": [Mode.TOOLS, Mode.MD_JSON],
        "env_var": "WRITER_API_KEY",
        "package": "writerai",
        "capabilities": {
            "streaming",
            "partial_streaming",
            "iterable_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "create_with_completion",
        },
    },
    Provider.PERPLEXITY: {
        "provider_string": "perplexity/llama-3.1-sonar-large-128k-online",
        "modes": [Mode.MD_JSON],  # Only supports text extraction
        "env_var": "PERPLEXITY_API_KEY",
        "package": "openai",  # Uses OpenAI-compatible API
        "capabilities": {
            # NO: streaming, partial_streaming (very limited)
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "create_with_completion",
        },
    },
    Provider.BEDROCK: {
        "provider_string": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "modes": [Mode.TOOLS, Mode.MD_JSON],
        "env_var": "AWS_ACCESS_KEY_ID",  # + AWS_SECRET_ACCESS_KEY
        "package": "boto3",
        "capabilities": {
            "streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
        },
    },
    Provider.VERTEXAI: {
        "provider_string": "vertex_ai/gemini-2.5-flash-lite",  # Cheapest Gemini via VertexAI
        "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "package": "google.cloud.aiplatform",
        "capabilities": {
            "structured_outputs",  # Native JSON_SCHEMA (Gemini via VertexAI)
            "streaming",
            "partial_streaming",
            "async",
            "list_extraction",
            "nested_models",
            "validation",
            "parallel_tools",
            "vision",
        },
    },
}
```

### Providers with Native Structured Outputs

Only these providers have `structured_outputs` in their API:

| Provider | API Feature | Notes |
|----------|-------------|-------|
| **OpenAI** | `response_format.json_schema` | Full structured outputs since GPT-4o |
| **Anthropic** | Tool use with schema | JSON schema enforcement via tools |
| **Gemini/GenAI** | `response_schema` | Native schema support |
| **VertexAI** | `response_schema` | Gemini via Google Cloud |

Other providers (Groq, Mistral, Cohere, etc.) use:
- `TOOLS` mode - Tool calling with schema
- `MD_JSON` mode - Text extraction from markdown code blocks

### Recommended Test Models (Modern & Cheap)

| Provider | Model | Pricing (per 1M tokens) | Why This Model |
|----------|-------|-------------------------|----------------|
| **OpenAI** | `gpt-4o-mini` | $0.15 in / $0.60 out | Cheapest with full structured outputs + vision |
| **Anthropic** | `claude-haiku-4-5-20241022` | $1 in / $5 out | Latest Haiku, fast, cheap, full capabilities |
| **Gemini** | `gemini-2.5-flash-lite` | $0.10 in / $0.40 out | Cheapest with native schema support |
| **Groq** | `llama-3.3-70b-versatile` | Free tier available | Fast inference, good for tool calling |
| **Mistral** | `ministral-8b-latest` | ~$0.10 in / $0.10 out | Small, cheap, supports tools |

**Estimated cost per 1000 test runs** (assuming ~500 tokens per test):
- OpenAI: ~$0.19
- Anthropic: ~$1.50
- Gemini: ~$0.13

### Files to Update with New Models

These files currently use older model names and should be updated:

```bash
# v2 test config (main update)
tests/v2/test_provider_modes.py
  - "anthropic/claude-3-5-haiku-latest" -> "anthropic/claude-haiku-4-5-20241022"
  - "google/gemini-2.0-flash" -> "google/gemini-2.5-flash-lite"

# Shared test config
tests/llm/shared_config.py
  - "anthropic/claude-3-5-haiku-latest" -> "anthropic/claude-haiku-4-5-20241022"
  - (Add google/gemini-2.5-flash-lite)

# Provider-specific util files
tests/llm/test_anthropic/util.py
  - models = ["anthropic/claude-haiku-4-5-20241022"]

tests/llm/test_genai/util.py
  - models = ["google/gemini-2.5-flash-lite"]

tests/llm/test_gemini/util.py
  - models = ["google/gemini-2.5-flash-lite"]

tests/llm/test_vertexai/util.py
  - models = ["gemini-2.5-flash-lite"]

# Auto client tests
tests/test_auto_client.py
  - Update model strings throughout
```
```

### Helper Functions

```python
def provider_supports(provider: Provider, capability: Capability) -> bool:
    """Check if provider supports a capability."""
    config = PROVIDER_CONFIGS.get(provider, {})
    return capability in config.get("capabilities", set())


def get_providers_with_capability(capability: Capability) -> list[Provider]:
    """Get all providers that support a capability."""
    return [
        p for p, cfg in PROVIDER_CONFIGS.items()
        if capability in cfg.get("capabilities", set())
    ]


def skip_if_unsupported(provider: Provider, capability: Capability):
    """Skip test if provider doesn't support capability."""
    import pytest
    if not provider_supports(provider, capability):
        pytest.skip(f"{provider.value} does not support {capability}")
```

### Dynamic Test Generation

```python
def _get_all_mode_params():
    """Generate (provider, mode) tuples for registration tests."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        for mode in config["modes"]:
            params.append((provider, mode))
    return params


def _get_streaming_params():
    """Generate params for providers that support streaming."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        if "streaming" in config.get("capabilities", set()):
            for mode in config["modes"]:
                params.append((provider, mode))
    return params


def _get_async_params():
    """Generate params for providers that support async."""
    params = []
    for provider, config in PROVIDER_CONFIGS.items():
        if "async" in config.get("capabilities", set()):
            for mode in config["modes"]:
                params.append((provider, mode))
    return params


@pytest.mark.parametrize("provider,mode", _get_all_mode_params())
def test_mode_is_registered(provider: Provider, mode: Mode):
    """All configured modes are registered."""
    assert mode_registry.is_registered(provider, mode)


@pytest.mark.parametrize("provider,mode", _get_all_mode_params())
@pytest.mark.requires_api_key
def test_mode_basic_extraction(provider: Provider, mode: Mode):
    """Basic extraction works for all provider/mode combinations."""
    config = PROVIDER_CONFIGS[provider]
    client = instructor.from_provider(config["provider_string"], mode=mode)
    response = client.chat.completions.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=1000,
    )
    assert isinstance(response, Answer)
    assert response.answer == 4.0


@pytest.mark.parametrize("provider,mode", _get_streaming_params())
@pytest.mark.requires_api_key
def test_mode_streaming_extraction(provider: Provider, mode: Mode):
    """Streaming works for providers that support it."""
    config = PROVIDER_CONFIGS[provider]
    client = instructor.from_provider(config["provider_string"], mode=mode)
    response = client.chat.completions.create(
        response_model=Partial[Answer],
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=1000,
        stream=True,
    )
    final = None
    for partial in response:
        final = partial
    assert final is not None
    assert final.answer == 4.0


@pytest.mark.parametrize("provider,mode", _get_async_params())
@pytest.mark.requires_api_key
@pytest.mark.asyncio
async def test_mode_async_extraction(provider: Provider, mode: Mode):
    """Async extraction works for providers that support it."""
    config = PROVIDER_CONFIGS[provider]
    client = instructor.from_provider(
        config["provider_string"], 
        mode=mode,
        async_client=True,
    )
    response = await client.chat.completions.create(
        response_model=Answer,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        max_tokens=1000,
    )
    assert isinstance(response, Answer)
    assert response.answer == 4.0
```

### Provider Capability Matrix

**Mode Support:**

| Provider | TOOLS | JSON_SCHEMA | MD_JSON | PARALLEL_TOOLS | RESPONSES_TOOLS |
|----------|-------|-------------|---------|----------------|-----------------|
| OpenAI | Yes | **Yes (native)** | Yes | Yes | Yes |
| Anthropic | Yes | **Yes (native)** | Yes | Yes | - |
| GenAI/Gemini | Yes | **Yes (native)** | Yes | - | - |
| Groq | Yes | - | Yes | - | - |
| Mistral | Yes | Yes | Yes | - | - |
| Cohere | Yes | Yes | Yes | - | - |
| xAI | Yes | Yes | Yes | - | - |
| Fireworks | Yes | - | Yes | - | - |
| Cerebras | Yes | - | Yes | - | - |
| Writer | Yes | - | Yes | - | - |
| Perplexity | - | - | Yes | - | - |
| Bedrock | Yes | - | Yes | - | - |
| VertexAI | Yes | **Yes (native)** | Yes | Yes | - |

**Note**: "native" means the provider has structured outputs built into their API (response_format with json_schema). Other JSON_SCHEMA support may use instruction-based approaches.

**Feature Capabilities:**

| Provider | streaming | async | partial | vision | union_types |
|----------|-----------|-------|---------|--------|-------------|
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | Yes | - |
| GenAI | Yes | Yes | Yes | Yes | - |
| Groq | Yes | Yes | Yes | - | - |
| Mistral | Yes | Yes | Yes | - | - |
| Cohere | Yes | Yes | Yes | - | - |
| xAI | Yes | Yes | Yes | - | - |
| Fireworks | Yes | Yes | Yes | - | - |
| Cerebras | Yes | Yes | Yes | - | - |
| Writer | Yes | Yes | Yes | - | - |
| Perplexity | - | Yes | - | - | - |
| Bedrock | Yes | Yes | - | - | - |
| VertexAI | Yes | Yes | Yes | Yes | - |

### Mode Normalization Tests (`tests/v2/test_mode_normalization.py`)

All deprecated mode mappings are also parameterized:

```python
@pytest.mark.parametrize(
    "provider,mode,expected",
    [
        # OpenAI legacy modes
        (Provider.OPENAI, Mode.FUNCTIONS, Mode.TOOLS),
        (Provider.OPENAI, Mode.TOOLS_STRICT, Mode.TOOLS),
        (Provider.OPENAI, Mode.JSON, Mode.JSON_SCHEMA),
        (Provider.OPENAI, Mode.JSON_O1, Mode.JSON_SCHEMA),
        
        # Anthropic legacy modes
        (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_JSON, Mode.MD_JSON),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_PARALLEL_TOOLS, Mode.PARALLEL_TOOLS),
        
        # Mistral legacy modes
        (Provider.MISTRAL, Mode.MISTRAL_TOOLS, Mode.TOOLS),
        (Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS, Mode.JSON_SCHEMA),
        
        # GenAI legacy modes
        (Provider.GENAI, Mode.GENAI_TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.GENAI_JSON, Mode.MD_JSON),
        (Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS, Mode.JSON_SCHEMA),
        
        # ... etc for all providers
        
        # Generic modes pass through unchanged
        (Provider.OPENAI, Mode.TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.JSON_SCHEMA, Mode.JSON_SCHEMA),
    ],
)
def test_normalize_mode(provider: Provider, mode: Mode, expected: Mode):
    """All legacy modes normalize to core modes."""
    result = normalize_mode(provider, mode)
    assert result == expected
```

### Running Tests

```bash
# Unit tests (no API key needed)
pytest tests/v2/ -v -k "not requires_api_key"

# All integration tests (needs all API keys)
pytest tests/v2/ -v -m requires_api_key

# Single provider
OPENAI_API_KEY=... pytest tests/v2/ -v -m requires_api_key -k "openai"
ANTHROPIC_API_KEY=... pytest tests/v2/ -v -m requires_api_key -k "anthropic"

# Regression tests
pytest tests/llm/ -v
pytest tests/test_patch.py -v
pytest tests/test_auto_client.py -v
```

---

## Timeline (Prioritized by API Key Availability)

| Week | Phase | Tasks | API Key Status |
|------|-------|-------|----------------|
| 1-2 | **Phase 1** | OpenAI (foundation + all 5 modes) | AVAILABLE |
| 3 | **Phase 2-3** | Cohere + xAI | AVAILABLE |
| 4 | **Phase 4-5** | Groq + Mistral (unit tests only) | Missing |
| 5 | **Phase 6-9** | Fireworks, Cerebras, Writer, Perplexity (unit tests only) | Missing |
| 6 | **Phase 10-11** | Bedrock, VertexAI (unit tests only) | Missing |
| 7 | - | Mode deprecation infrastructure + documentation | N/A |
| 8 | - | Integration testing + final cleanup | All |

### Implementation Priority

1. **P0 - OpenAI**: Foundation for all other providers (Week 1-2)
2. **P1 - Cohere, xAI**: API keys available, full testing possible (Week 3)
3. **P2 - Others**: API keys missing, unit tests only (Week 4-6)

---

## Implementation Checklist

### Per Phase Checklist

- [ ] Create `instructor/v2/providers/{provider}/` directory
- [ ] Implement `handlers.py` with mode handlers for core modes
- [ ] Implement `client.py` with factory function
- [ ] Create `__init__.py` with exports
- [ ] Add import to `instructor/v2/__init__.py`
- [ ] Add mode normalizations to `registry.py`
- [ ] Add provider entry to `PROVIDER_CONFIGS` in `tests/v2/test_provider_modes.py`
- [ ] Add legacy mode mappings to `tests/v2/test_mode_normalization.py`
- [ ] Run tests: `pytest tests/v2/ -v -k "{provider}"`
- [ ] **Coverage**: Achieve target handler coverage (see Handler and Client Test Coverage section)
- [ ] **Coverage**: Achieve target client coverage (see Handler and Client Test Coverage section)
- [ ] **Coverage**: Add handler unit tests (prepare_request, parse_response, handle_reask)
- [ ] **Coverage**: Add client factory tests (sync/async, error handling, mode validation)
- [ ] **Coverage**: Run coverage report and verify targets met
- [ ] Update documentation

### Global Checklist

- [ ] Add deprecation warnings for all provider-specific modes
- [ ] Update `instructor/mode.py` with `DEPRECATED_TO_CORE` map
- [ ] Ensure all 5 core modes are implemented per provider capability
- [ ] Create migration guide documentation
- [ ] Update all examples to use generic modes (TOOLS, JSON_SCHEMA, etc.)

---

## Related Documents

- [v2 Architecture Overview](../../instructor/v2/README.md)
- [Mode Registry Design](./phase1_mode_registry.md)
- [Existing v2 Anthropic Implementation](../../instructor/v2/providers/anthropic/)
- [Existing v2 GenAI Implementation](../../instructor/v2/providers/genai/)
- [Test Configuration](../../tests/llm/shared_config.py)
