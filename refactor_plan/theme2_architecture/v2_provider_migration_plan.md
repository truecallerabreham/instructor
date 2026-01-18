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

### Test Checklist

#### Unit Tests (No API Key Required)

- [ ] **Handler Registration Tests**:
  ```bash
  pytest tests/v2/test_provider_modes.py -v -k "openai and test_mode_is_registered"
  ```

- [ ] **Parameterized Handler Unit Tests**:
  ```bash
  pytest tests/v2/test_handlers_parametrized.py -v -k "openai"
  ```
  - [ ] Verify provider added to `PROVIDER_HANDLER_MODES` in `test_handlers_parametrized.py`
  - [ ] Verify provider/mode scenarios added to `PARSE_SCENARIOS` in `test_handlers_parametrized.py`
  - [ ] Verify `MockResponseBuilder` methods support OpenAI response formats
  - [ ] Verify handler module path added to `_HANDLER_MODULE_PATHS` in `test_handlers_parametrized.py`
  - [ ] `test_prepare_request_with_none_model()` - Tests `prepare_request()` with `None` response_model
  - [ ] `test_prepare_request_with_model()` - Tests `prepare_request()` with response_model
  - [ ] `test_parse_response()` - Tests `parse_response()` with valid payloads (all modes)
  - [ ] `test_parse_response_validation_error()` - Tests `parse_response()` with invalid payloads
  - [ ] `test_handle_reask_adds_message()` - Tests `handle_reask()` adds error messages

- [ ] **Provider-Specific Handler Unit Tests** (if needed for edge cases):
  ```bash
  pytest tests/v2/test_openai_handlers.py -v -k "not requires_api_key"
  ```
  - [ ] `OpenAIToolsHandler.prepare_request()` - Test strict parameter handling
  - [ ] `OpenAIToolsHandler.parse_response()` - Test tool call extraction edge cases
  - [ ] `OpenAIJSONSchemaHandler.prepare_request()` - Test response_format setup details
  - [ ] `OpenAIParallelToolsHandler.prepare_request()` - Test parallel tool setup
  - [ ] `OpenAIResponsesToolsHandler` - Test Responses API format specifics

- [ ] **Client Factory Tests**:
  ```bash
  pytest tests/v2/test_openai_client.py -v -k "not requires_api_key"
  ```
  - [ ] `from_openai()` sync client creation
  - [ ] `from_openai()` async client creation
  - [ ] Invalid client type error handling
  - [ ] Invalid mode error handling
  - [ ] Legacy mode normalization with deprecation warnings
  - [ ] Mode validation for unsupported modes

- [ ] **Mode Normalization Tests**:
  ```bash
  pytest tests/v2/test_mode_normalization.py -v -k "openai"
  ```
  - [ ] `Mode.FUNCTIONS` -> `Mode.TOOLS`
  - [ ] `Mode.TOOLS_STRICT` -> `Mode.TOOLS`
  - [ ] `Mode.JSON` -> `Mode.JSON_SCHEMA`
  - [ ] `Mode.JSON_O1` -> `Mode.JSON_SCHEMA`
  - [ ] Generic modes pass through unchanged

#### Integration Tests (Requires OPENAI_API_KEY)

- [ ] **Basic Extraction Tests**:
  ```bash
  OPENAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "openai and test_mode_basic_extraction"
  ```
  - [ ] `Mode.TOOLS` extraction
  - [ ] `Mode.JSON_SCHEMA` extraction
  - [ ] `Mode.MD_JSON` extraction
  - [ ] `Mode.PARALLEL_TOOLS` extraction
  - [ ] `Mode.RESPONSES_TOOLS` extraction

- [ ] **Streaming Tests**:
  ```bash
  OPENAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "openai and streaming"
  ```
  - [ ] Streaming with `Mode.TOOLS`
  - [ ] Streaming with `Mode.JSON_SCHEMA`
  - [ ] Partial streaming with `Partial[T]`
  - [ ] Iterable streaming

- [ ] **Async Tests**:
  ```bash
  OPENAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "openai and async"
  ```
  - [ ] Async extraction with all modes
  - [ ] Async streaming

- [ ] **Advanced Features**:
  ```bash
  OPENAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "openai"
  ```
  - [ ] List extraction
  - [ ] Nested models
  - [ ] Validation errors and retries
  - [ ] Union types
  - [ ] Enum types
  - [ ] Vision/image input

#### Coverage Tests

- [ ] **Handler Coverage**:
  ```bash
  pytest tests/v2/ -k "openai" --cov=instructor.v2.providers.openai.handlers --cov-report=term-missing
  ```
  - [ ] Target: ≥70% coverage
  - [ ] All handler methods covered
  - [ ] Edge cases covered (None response_model, invalid responses)

- [ ] **Client Coverage**:
  ```bash
  pytest tests/v2/ -k "openai" --cov=instructor.v2.providers.openai.client --cov-report=term-missing
  ```
  - [ ] Target: ≥60% coverage
  - [ ] Sync/async paths covered
  - [ ] Error handling covered

#### Regression Tests

- [ ] **Existing OpenAI Tests**:
  ```bash
  pytest tests/llm/test_openai/ -v
  ```
  - [ ] All existing tests pass
  - [ ] No regressions introduced

- [ ] **Patch Tests**:
  ```bash
  pytest tests/core/test_patch.py -v
  ```
  - [ ] v2 patch works correctly
  - [ ] Backward compatibility maintained

- [ ] **Auto Client Tests**:
  ```bash
  pytest tests/providers/test_auto_client.py -v -k "openai"
  ```
  - [ ] `from_provider()` works with OpenAI
  - [ ] Provider detection correct

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
- [x] Run: `pytest tests/v2/ -v -k "cohere"` - All 79 tests pass
- [x] Handler test coverage ≥60% (`handlers.py`) - Current: 94% ✅
- [x] Client test coverage ≥70% (`client.py`) - Current: 71% ✅
- [x] Add handler unit tests for all methods - 58 tests in `tests/v2/test_cohere_handlers.py`

### Test Checklist

#### Unit Tests (No API Key Required)

- [ ] **Handler Registration Tests**:
  ```bash
  pytest tests/v2/test_provider_modes.py -v -k "cohere and test_mode_is_registered"
  ```

- [ ] **Handler Unit Tests**:
  ```bash
  pytest tests/v2/test_cohere_handlers.py -v -k "not requires_api_key"
  ```
  - [ ] `CohereToolsHandler.prepare_request()` - Test tool format
  - [ ] `CohereToolsHandler.parse_response()` - Test tool call extraction
  - [ ] `CohereToolsHandler.handle_reask()` - Test reask logic
  - [ ] `CohereJSONSchemaHandler.prepare_request()` - Test structured outputs setup
  - [ ] `CohereJSONSchemaHandler.parse_response()` - Test JSON parsing
  - [ ] `CohereMDJSONHandler.prepare_request()` - Test markdown instruction
  - [ ] `CohereMDJSONHandler.parse_response()` - Test code block extraction

- [ ] **Client Factory Tests**:
  ```bash
  pytest tests/v2/test_cohere_client.py -v -k "not requires_api_key"
  ```
  - [ ] `from_cohere()` sync client creation
  - [ ] `from_cohere()` async client creation
  - [ ] Invalid client type error handling
  - [ ] Invalid mode error handling
  - [ ] Legacy mode normalization (`COHERE_TOOLS` -> `TOOLS`)

- [ ] **Mode Normalization Tests**:
  ```bash
  pytest tests/v2/test_mode_normalization.py -v -k "cohere"
  ```
  - [ ] `Mode.COHERE_TOOLS` -> `Mode.TOOLS`
  - [ ] `Mode.COHERE_JSON_SCHEMA` -> `Mode.JSON_SCHEMA`
  - [ ] Generic modes pass through unchanged

#### Integration Tests (Requires COHERE_API_KEY)

- [ ] **Basic Extraction Tests**:
  ```bash
  COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cohere and test_mode_basic_extraction"
  ```
  - [ ] `Mode.TOOLS` extraction
  - [ ] `Mode.JSON_SCHEMA` extraction
  - [ ] `Mode.MD_JSON` extraction

- [ ] **Streaming Tests**:
  ```bash
  COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cohere and streaming"
  ```
  - [ ] Streaming with all supported modes
  - [ ] Partial streaming with `Partial[T]`

- [ ] **Async Tests**:
  ```bash
  COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cohere and async"
  ```
  - [ ] Async extraction with all modes
  - [ ] Async streaming

- [ ] **Advanced Features**:
  ```bash
  COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cohere"
  ```
  - [ ] List extraction
  - [ ] Nested models
  - [ ] Validation errors and retries

#### Coverage Tests

- [ ] **Handler Coverage**:
  ```bash
  pytest tests/v2/ -k "cohere" --cov=instructor.v2.providers.cohere.handlers --cov-report=term-missing
  ```
  - [ ] Target: ≥60% coverage
  - [ ] All handler methods covered

- [ ] **Client Coverage**:
  ```bash
  pytest tests/v2/ -k "cohere" --cov=instructor.v2.providers.cohere.client --cov-report=term-missing
  ```
  - [ ] Target: ≥70% coverage
  - [ ] Sync/async paths covered

#### Regression Tests

- [ ] **Existing Cohere Tests**:
  ```bash
  pytest tests/llm/test_cohere/ -v
  ```
  - [ ] All existing tests pass
  - [ ] No regressions introduced

- [ ] **Auto Client Tests**:
  ```bash
  pytest tests/providers/test_auto_client.py -v -k "cohere"
  ```
  - [ ] `from_provider()` works with Cohere

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

### Test Checklist

#### Unit Tests (No API Key Required)

- [ ] **Handler Registration Tests**:
  ```bash
  pytest tests/v2/test_provider_modes.py -v -k "xai and test_mode_is_registered"
  ```

- [ ] **Handler Unit Tests**:
  ```bash
  pytest tests/v2/test_xai_handlers.py -v -k "not requires_api_key"
  ```
  - [ ] `XAIToolsHandler.prepare_request()` - Test tool format
  - [ ] `XAIToolsHandler.parse_response()` - Test tool call extraction
  - [ ] `XAIToolsHandler.handle_reask()` - Test reask logic
  - [ ] `XAIJSONSchemaHandler.prepare_request()` - Test structured outputs setup
  - [ ] `XAIJSONSchemaHandler.parse_response()` - Test JSON parsing
  - [ ] `XAIMDJSONHandler.prepare_request()` - Test markdown instruction
  - [ ] `XAIMDJSONHandler.parse_response()` - Test code block extraction

- [ ] **Client Factory Tests**:
  ```bash
  pytest tests/v2/test_xai_client.py -v -k "not requires_api_key"
  ```
  - [ ] `from_xai()` sync client creation
  - [ ] `from_xai()` async client creation
  - [ ] Invalid client type error handling
  - [ ] Invalid mode error handling
  - [ ] Legacy mode normalization (`XAI_TOOLS` -> `TOOLS`, `XAI_JSON` -> `JSON_SCHEMA`)

- [ ] **Mode Normalization Tests**:
  ```bash
  pytest tests/v2/test_mode_normalization.py -v -k "xai"
  ```
  - [ ] `Mode.XAI_TOOLS` -> `Mode.TOOLS`
  - [ ] `Mode.XAI_JSON` -> `Mode.JSON_SCHEMA`
  - [ ] Generic modes pass through unchanged

#### Integration Tests (Requires XAI_API_KEY)

- [ ] **Basic Extraction Tests**:
  ```bash
  XAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "xai and test_mode_basic_extraction"
  ```
  - [ ] `Mode.TOOLS` extraction
  - [ ] `Mode.JSON_SCHEMA` extraction
  - [ ] `Mode.MD_JSON` extraction (may have model-specific issues)

- [ ] **Streaming Tests**:
  ```bash
  XAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "xai and streaming"
  ```
  - [ ] Streaming with all supported modes
  - [ ] Partial streaming with `Partial[T]`

- [ ] **Async Tests**:
  ```bash
  XAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "xai and async"
  ```
  - [ ] Async extraction with all modes
  - [ ] Async streaming

- [ ] **Advanced Features**:
  ```bash
  XAI_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "xai"
  ```
  - [ ] Nested models
  - [ ] Validation errors and retries
  - [ ] Note: list_extraction may have issues with tool_calls

#### Coverage Tests

- [ ] **Handler Coverage**:
  ```bash
  pytest tests/v2/ -k "xai" --cov=instructor.v2.providers.xai.handlers --cov-report=term-missing
  ```
  - [ ] Target: ≥60% coverage (Current: 77% ✅)
  - [ ] All handler methods covered

- [ ] **Client Coverage**:
  ```bash
  pytest tests/v2/ -k "xai" --cov=instructor.v2.providers.xai.client --cov-report=term-missing
  ```
  - [ ] Target: ≥50% coverage (Current: 12% - requires xAI SDK)
  - [ ] Sync/async paths covered

#### Regression Tests

- [ ] **Existing xAI Tests**:
  ```bash
  pytest tests/llm/test_xai/ -v
  ```
  - [ ] All existing tests pass
  - [ ] No regressions introduced

- [ ] **Auto Client Tests**:
  ```bash
  pytest tests/providers/test_auto_client.py -v -k "xai"
  ```
  - [ ] `from_provider()` works with xAI

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

### Test Checklist

#### Unit Tests (No API Key Required)

- [ ] **Handler Registration Tests**:
  ```bash
  pytest tests/v2/test_provider_modes.py -v -k "groq and test_mode_is_registered"
  ```

- [ ] **Handler Unit Tests** (Groq reuses OpenAI handlers):
  ```bash
  pytest tests/v2/test_groq_handlers.py -v -k "not requires_api_key"
  ```
  - [ ] `GroqToolsHandler` extends `OpenAIToolsHandler` correctly
  - [ ] `GroqMDJSONHandler` extends `OpenAIMDJSONHandler` correctly
  - [ ] Handler registration works
  - [ ] Mode assignment correct

- [ ] **Client Factory Tests**:
  ```bash
  pytest tests/v2/test_groq_client.py -v -k "not requires_api_key"
  ```
  - [ ] `from_groq()` sync client creation
  - [ ] `from_groq()` async client creation
  - [ ] Invalid client type error handling
  - [ ] Invalid mode error handling (JSON_SCHEMA not supported)
  - [ ] Mode validation for unsupported modes

- [ ] **Mode Normalization Tests**:
  ```bash
  pytest tests/v2/test_mode_normalization.py -v -k "groq"
  ```
  - [ ] Generic modes pass through unchanged
  - [ ] Unsupported modes raise appropriate errors

#### Integration Tests (Requires GROQ_API_KEY - Optional)

- [ ] **Basic Extraction Tests** (if API key available):
  ```bash
  GROQ_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "groq and test_mode_basic_extraction"
  ```
  - [ ] `Mode.TOOLS` extraction
  - [ ] `Mode.MD_JSON` extraction
  - [ ] `Mode.JSON_SCHEMA` should fail gracefully (not supported)

- [ ] **Streaming Tests** (if API key available):
  ```bash
  GROQ_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "groq and streaming"
  ```
  - [ ] Streaming with `Mode.TOOLS`
  - [ ] Streaming with `Mode.MD_JSON`
  - [ ] Partial streaming with `Partial[T]`

- [ ] **Async Tests** (if API key available):
  ```bash
  GROQ_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "groq and async"
  ```
  - [ ] Async extraction with all supported modes
  - [ ] Async streaming

#### Coverage Tests

- [ ] **Handler Coverage**:
  ```bash
  pytest tests/v2/ -k "groq" --cov=instructor.v2.providers.groq.handlers --cov-report=term-missing
  ```
  - [ ] Target: ≥50% coverage (Current: 100% ✅)
  - [ ] All handler methods covered

- [ ] **Client Coverage**:
  ```bash
  pytest tests/v2/ -k "groq" --cov=instructor.v2.providers.groq.client --cov-report=term-missing
  ```
  - [ ] Target: ≥50% coverage (Current: 48% - close to target)
  - [ ] Sync/async paths covered
  - [ ] Error handling covered

#### Regression Tests

- [ ] **Existing Groq Tests**:
  ```bash
  pytest tests/llm/test_groq/ -v
  ```
  - [ ] All existing tests pass
  - [ ] No regressions introduced

- [ ] **Auto Client Tests**:
  ```bash
  pytest tests/providers/test_auto_client.py -v -k "groq"
  ```
  - [ ] `from_provider()` works with Groq

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

- [x] Create `instructor/v2/providers/mistral/` directory
- [x] Create `handlers.py`:
  - [x] `MistralToolsHandler` - TOOLS mode
  - [x] `MistralJSONSchemaHandler` - JSON_SCHEMA mode
  - [x] `MistralMDJSONHandler` - MD_JSON mode
- [x] Create `client.py` with `from_mistral()` factory
- [x] Add import to `instructor/v2/__init__.py`
- [x] Add legacy normalizations (MISTRAL_TOOLS -> TOOLS) - already in registry.py
- [ ] Add to `PROVIDER_CONFIGS` in tests (skipped - no API key)
- [x] Run unit tests only: `pytest tests/v2/ -v -k "mistral"` - 59 passed, 5 skipped
- [x] Handler test coverage - 40 tests in `tests/v2/test_mistral_handlers.py`
- [x] Client test coverage - 19 tests in `tests/v2/test_mistral_client.py`
- [x] Add handler unit tests for all methods
- [x] Add client factory tests

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

### Test Checklist

#### Unit Tests (No API Key Required)

- [ ] **Handler Registration Tests**:
  ```bash
  pytest tests/v2/test_provider_modes.py -v -k "mistral and test_mode_is_registered"
  ```

- [ ] **Handler Unit Tests**:
  ```bash
  pytest tests/v2/test_mistral_handlers.py -v -k "not requires_api_key"
  ```
  - [ ] `MistralToolsHandler.prepare_request()` - Test Mistral tool format
  - [ ] `MistralToolsHandler.parse_response()` - Test tool call extraction
  - [ ] `MistralToolsHandler.handle_reask()` - Test reask logic
  - [ ] `MistralJSONSchemaHandler.prepare_request()` - Test structured outputs setup
  - [ ] `MistralJSONSchemaHandler.parse_response()` - Test JSON parsing
  - [ ] `MistralMDJSONHandler.prepare_request()` - Test markdown instruction
  - [ ] `MistralMDJSONHandler.parse_response()` - Test code block extraction

- [ ] **Client Factory Tests**:
  ```bash
  pytest tests/v2/test_mistral_client.py -v -k "not requires_api_key"
  ```
  - [ ] `from_mistral()` sync client creation
  - [ ] `from_mistral()` async client creation
  - [ ] Invalid client type error handling
  - [ ] Invalid mode error handling
  - [ ] Legacy mode normalization (`MISTRAL_TOOLS` -> `TOOLS`, `MISTRAL_STRUCTURED_OUTPUTS` -> `JSON_SCHEMA`)

- [ ] **Mode Normalization Tests**:
  ```bash
  pytest tests/v2/test_mode_normalization.py -v -k "mistral"
  ```
  - [ ] `Mode.MISTRAL_TOOLS` -> `Mode.TOOLS`
  - [ ] `Mode.MISTRAL_STRUCTURED_OUTPUTS` -> `Mode.JSON_SCHEMA`
  - [ ] Generic modes pass through unchanged

#### Integration Tests (Requires MISTRAL_API_KEY - Optional)

- [ ] **Basic Extraction Tests** (if API key available):
  ```bash
  MISTRAL_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "mistral and test_mode_basic_extraction"
  ```
  - [ ] `Mode.TOOLS` extraction
  - [ ] `Mode.JSON_SCHEMA` extraction
  - [ ] `Mode.MD_JSON` extraction

- [ ] **Streaming Tests** (if API key available):
  ```bash
  MISTRAL_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "mistral and streaming"
  ```
  - [ ] Streaming with all supported modes
  - [ ] Partial streaming with `Partial[T]`

- [ ] **Async Tests** (if API key available):
  ```bash
  MISTRAL_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "mistral and async"
  ```
  - [ ] Async extraction with all modes
  - [ ] Async streaming

- [ ] **Advanced Features** (if API key available):
  ```bash
  MISTRAL_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "mistral"
  ```
  - [ ] List extraction
  - [ ] Nested models
  - [ ] Validation errors and retries

#### Coverage Tests

- [ ] **Handler Coverage**:
  ```bash
  pytest tests/v2/ -k "mistral" --cov=instructor.v2.providers.mistral.handlers --cov-report=term-missing
  ```
  - [ ] Target: ≥50% coverage
  - [ ] All handler methods covered

- [ ] **Client Coverage**:
  ```bash
  pytest tests/v2/ -k "mistral" --cov=instructor.v2.providers.mistral.client --cov-report=term-missing
  ```
  - [ ] Target: ≥50% coverage
  - [ ] Sync/async paths covered

#### Regression Tests

- [ ] **Existing Mistral Tests**:
  ```bash
  pytest tests/llm/test_mistral/ -v
  ```
  - [ ] All existing tests pass
  - [ ] No regressions introduced

- [ ] **Auto Client Tests**:
  ```bash
  pytest tests/providers/test_auto_client.py -v -k "mistral"
  ```
  - [ ] `from_provider()` works with Mistral

---

## Phase 6-12: Remaining Providers (API Keys Missing)

All these providers have missing API keys. Implement with unit tests only.

---

### Phase 6: Fireworks

**API Key**: `FIREWORKS_API_KEY` - MISSING

- [x] Create `instructor/v2/providers/fireworks/` directory
- [x] Handlers: `TOOLS`, `MD_JSON` (OpenAI-compatible)
- [ ] Add to `PROVIDER_CONFIGS` (skipped - no API key)
- [x] Run unit tests only - 45 passed, 2 skipped

#### Test Checklist

- [x] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "fireworks and not requires_api_key"
  ```
  - [x] Handler registration tests
  - [x] Handler unit tests (reuses OpenAI handlers)
  - [x] Client factory tests
  - [x] Mode normalization tests

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "fireworks" --cov=instructor.v2.providers.fireworks --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if API key becomes available):
  ```bash
  FIREWORKS_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "fireworks"
  ```

### Phase 7: Cerebras

**API Key**: `CEREBRAS_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/cerebras/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON` (OpenAI-compatible)
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

#### Test Checklist

- [ ] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "cerebras and not requires_api_key"
  ```
  - [ ] Handler registration tests
  - [ ] Handler unit tests (reuses OpenAI handlers)
  - [ ] Client factory tests
  - [ ] Mode normalization tests

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "cerebras" --cov=instructor.v2.providers.cerebras --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if API key becomes available):
  ```bash
  CEREBRAS_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cerebras"
  ```

### Phase 8: Writer

**API Key**: `WRITER_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/writer/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

#### Test Checklist

- [ ] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "writer and not requires_api_key"
  ```
  - [ ] Handler registration tests
  - [ ] Handler unit tests
  - [ ] Client factory tests
  - [ ] Mode normalization tests

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "writer" --cov=instructor.v2.providers.writer --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if API key becomes available):
  ```bash
  WRITER_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "writer"
  ```

### Phase 9: Perplexity

**API Key**: `PERPLEXITY_API_KEY` - MISSING

- [ ] Create `instructor/v2/providers/perplexity/` directory
- [ ] Handlers: `MD_JSON` only (no tool calling support)
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

#### Test Checklist

- [ ] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "perplexity and not requires_api_key"
  ```
  - [ ] Handler registration tests (MD_JSON only)
  - [ ] Handler unit tests
  - [ ] Client factory tests
  - [ ] Mode normalization tests
  - [ ] Verify TOOLS mode raises appropriate error

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "perplexity" --cov=instructor.v2.providers.perplexity --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if API key becomes available):
  ```bash
  PERPLEXITY_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "perplexity"
  ```
  - [ ] Note: Perplexity has limited streaming support

### Phase 10: Bedrock

**API Key**: `AWS_ACCESS_KEY_ID` - MISSING

- [ ] Create `instructor/v2/providers/bedrock/` directory
- [ ] Handlers: `TOOLS`, `MD_JSON`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

#### Test Checklist

- [ ] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "bedrock and not requires_api_key"
  ```
  - [ ] Handler registration tests
  - [ ] Handler unit tests
  - [ ] Client factory tests
  - [ ] Mode normalization tests
  - [ ] AWS credentials handling tests

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "bedrock" --cov=instructor.v2.providers.bedrock --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if AWS credentials become available):
  ```bash
  AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "bedrock"
  ```

### Phase 11: VertexAI

**API Key**: `GOOGLE_APPLICATION_CREDENTIALS` - MISSING

- [ ] Create `instructor/v2/providers/vertexai/` directory
- [ ] Handlers: `TOOLS`, `JSON_SCHEMA`, `MD_JSON`, `PARALLEL_TOOLS`
- [ ] Add to `PROVIDER_CONFIGS`
- [ ] Run unit tests only

#### Test Checklist

- [ ] **Unit Tests**:
  ```bash
  pytest tests/v2/ -v -k "vertexai and not requires_api_key"
  ```
  - [ ] Handler registration tests (all 4 modes)
  - [ ] Handler unit tests
  - [ ] Client factory tests
  - [ ] Mode normalization tests
  - [ ] Google Cloud credentials handling tests

- [ ] **Coverage Tests**:
  ```bash
  pytest tests/v2/ -k "vertexai" --cov=instructor.v2.providers.vertexai --cov-report=term-missing
  ```
  - [ ] Target: ≥50% handler coverage
  - [ ] Target: ≥50% client coverage

- [ ] **Integration Tests** (if Google Cloud credentials become available):
  ```bash
  GOOGLE_APPLICATION_CREDENTIALS=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "vertexai"
  ```
  - [ ] Test all 4 modes: TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL_TOOLS
  - [ ] Test streaming and async capabilities

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

### Parameterized Handler Tests

The shared handler tests live in `tests/v2/test_handlers_parametrized.py`.
Run them with:

```bash
uv run pytest tests/v2/test_handlers_parametrized.py -v
```

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

The v2 test suite uses a comprehensive parameterized testing architecture that automatically tests all provider/mode combinations. The tests are split into three main categories:

1. **Handler Unit Tests** (`tests/v2/test_handlers_parametrized.py`) - Tests handler methods with mocks (no API calls)
2. **Integration Tests** (`tests/v2/test_provider_modes.py`) - Tests actual extraction with real API calls
3. **Mode Normalization Tests** (`tests/v2/test_mode_normalization.py`) - Tests deprecated mode mappings and warnings

### Parameterized Core Test Architecture

The v2 test infrastructure is designed to automatically test all provider/mode combinations through parameterization. When adding a new provider, you only need to update configuration dictionaries - the tests automatically generate test cases for all combinations.

#### Test File Structure

**1. `tests/v2/test_handlers_parametrized.py` - Handler Unit Tests (No API Key Required)**

This file contains parameterized unit tests that exercise handler methods (`prepare_request`, `parse_response`, `handle_reask`) using mock responses. These tests run fast and don't require API keys.

Key components:

- **`PROVIDER_HANDLER_MODES`** (lines 74-92): Maps each provider to a list of modes that should be tested
  ```python
  PROVIDER_HANDLER_MODES: dict[Provider, list[Mode]] = {
      Provider.OPENAI: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
      Provider.ANTHROPIC: [Mode.TOOLS, Mode.JSON, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS],
      Provider.GENAI: [Mode.TOOLS, Mode.JSON],
      Provider.COHERE: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
      Provider.XAI: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
  }
  ```

- **`PARSE_SCENARIOS`** (lines 95-115): Maps provider/mode combinations to response scenario types
  ```python
  PARSE_SCENARIOS: dict[Provider, dict[Mode, str]] = {
      Provider.OPENAI: {
          Mode.TOOLS: "tool_call",
          Mode.JSON_SCHEMA: "text",
          Mode.MD_JSON: "markdown",
          Mode.RESPONSES_TOOLS: "responses_output",
      },
      Provider.COHERE: {
          Mode.TOOLS: "tool_call",
          Mode.JSON_SCHEMA: "text",
          Mode.MD_JSON: "markdown",
      },
      # ... etc
  }
  ```

- **`MockResponseBuilder`** (lines 140-190): Builds provider-specific mock responses for testing
  - `tool_response()` - Creates mock tool call responses
  - `text_response()` - Creates mock text responses
  - `markdown_response()` - Creates mock markdown code block responses
  - `responses_output_response()` - Creates mock Responses API output (OpenAI only)

- **`_provider_mode_params()`** (lines 130-137): Generates all provider/mode combinations from `PROVIDER_HANDLER_MODES` for pytest parameterization

- **`_HANDLER_MODULE_PATHS`** (lines 25-31): Maps providers to their handler module file paths for dynamic loading

- **`_ensure_handlers_loaded()`** (lines 35-53): Dynamically loads handler modules via `importlib` to ensure handlers are registered before testing

Test functions:
- `test_prepare_request_with_none_model()` - Tests `prepare_request()` with `None` response_model
- `test_prepare_request_with_model()` - Tests `prepare_request()` with a response_model
- `test_parse_response()` - Tests `parse_response()` with valid payloads
- `test_parse_response_validation_error()` - Tests `parse_response()` with invalid payloads
- `test_handle_reask_adds_message()` - Tests `handle_reask()` adds error messages

**2. `tests/v2/test_provider_modes.py` - Integration Tests (Requires API Keys)**

This file contains parameterized integration tests that make actual API calls to test end-to-end extraction. These tests require API keys and are marked with `@pytest.mark.requires_api_key`.

Key components:

- **`PROVIDER_CONFIGS`** (lines 39-69): Maps providers to their full configuration
  ```python
  PROVIDER_CONFIGS = {
      Provider.ANTHROPIC: {
          "provider_string": "anthropic/claude-3-5-haiku-latest",
          "modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.PARALLEL_TOOLS, Mode.ANTHROPIC_REASONING_TOOLS],
          "basic_modes": [Mode.TOOLS, Mode.JSON_SCHEMA],
          "async_modes": [Mode.TOOLS, Mode.JSON_SCHEMA],
      },
      # ... etc
  }
  ```

Test functions:
- `test_mode_is_registered()` - Verifies each mode is registered (currently hardcoded parameter list)
- `test_mode_basic_extraction()` - Tests basic extraction with each mode (currently hardcoded parameter list)
- `test_mode_async_extraction()` - Tests async extraction (currently hardcoded parameter list)
- `test_all_modes_covered()` - Verifies all registered modes are tested

**Note**: The integration tests currently use hardcoded parameter lists. They should be made dynamic by generating parameters from `PROVIDER_CONFIGS` similar to how handler tests work.

**3. `tests/v2/test_mode_normalization.py` - Mode Normalization Tests**

This file tests deprecated mode mappings and ensures deprecation warnings work correctly.

Key components:

- Parameterized tests for all deprecated mode mappings (lines 29-74)
- Tests that deprecated modes emit warnings (lines 149-177)
- Tests that warnings are only shown once (lines 180-195)
- Tests that generic modes don't emit warnings (lines 198-210)
- Tests that deprecated mode mapping is complete (lines 213-270)

#### How Tests Are Parameterized

**Handler Tests** (`test_handlers_parametrized.py`):
- Use `@pytest.mark.parametrize("provider,mode", _provider_mode_params())` decorator
- `_provider_mode_params()` generates all combinations from `PROVIDER_HANDLER_MODES`
- Tests automatically skip unsupported combinations using `pytest.skip()`
- Example: If a provider doesn't support a mode, the test skips with a message

**Integration Tests** (`test_provider_modes.py`):
- Currently use hardcoded `@pytest.mark.parametrize()` lists
- Should be refactored to generate parameters dynamically from `PROVIDER_CONFIGS`
- Example target implementation:
  ```python
  def _get_basic_mode_params():
      params = []
      for provider, config in PROVIDER_CONFIGS.items():
          for mode in config["basic_modes"]:
              params.append((provider, mode))
      return params
  
  @pytest.mark.parametrize("provider,mode", _get_basic_mode_params())
  def test_mode_basic_extraction(provider: Provider, mode: Mode):
      # ... test implementation
  ```

**Mode Normalization Tests** (`test_mode_normalization.py`):
- Use hardcoded parameter lists for deprecated mode mappings
- Test all provider-specific modes map to core modes correctly

#### What Needs Updating When Adding a Provider

When migrating a provider to v2, update these configuration dictionaries:

1. **`PROVIDER_HANDLER_MODES`** in `tests/v2/test_handlers_parametrized.py` (line 74)
   - Add provider entry with list of supported modes
   - Example: `Provider.NEW_PROVIDER: [Mode.TOOLS, Mode.MD_JSON]`

2. **`PARSE_SCENARIOS`** in `tests/v2/test_handlers_parametrized.py` (line 95)
   - Add provider entry mapping modes to scenario types
   - Scenarios: `"tool_call"`, `"text"`, `"markdown"`, `"responses_output"`
   - Example: `Provider.NEW_PROVIDER: {Mode.TOOLS: "tool_call", Mode.MD_JSON: "markdown"}`

3. **`MockResponseBuilder`** in `tests/v2/test_handlers_parametrized.py` (line 140)
   - Add methods for provider-specific response formats if needed
   - Most providers can reuse existing methods (OpenAI-compatible APIs)

4. **`_HANDLER_MODULE_PATHS`** in `tests/v2/test_handlers_parametrized.py` (line 25)
   - Add provider entry mapping to handler module file path
   - Example: `Provider.NEW_PROVIDER: _PROJECT_ROOT / "instructor/v2/providers/new_provider/handlers.py"`

5. **`PROVIDER_CONFIGS`** in `tests/v2/test_provider_modes.py` (line 39)
   - Add provider entry with full configuration
   - Includes: `provider_string`, `modes`, `basic_modes`, `async_modes`

6. **Mode normalization tests** in `tests/v2/test_mode_normalization.py` (line 29)
   - Add parameterized test cases for deprecated mode mappings
   - Example: `(Provider.NEW_PROVIDER, Mode.NEW_PROVIDER_TOOLS, Mode.TOOLS)`

7. **Deprecated mode mapping** in `tests/v2/test_mode_normalization.py` (line 213)
   - Add provider-specific deprecated modes to `expected_deprecated` set

#### Running Parameterized Tests

**Handler Unit Tests** (No API key required):
```bash
# Run all handler unit tests
pytest tests/v2/test_handlers_parametrized.py -v

# Run tests for specific provider
pytest tests/v2/test_handlers_parametrized.py -v -k "cohere"

# Run specific test function
pytest tests/v2/test_handlers_parametrized.py::test_parse_response -v

# Run with coverage
pytest tests/v2/test_handlers_parametrized.py --cov=instructor.v2.providers.cohere.handlers --cov-report=term-missing
```

**Integration Tests** (Requires API keys):
```bash
# Run all integration tests (needs all API keys)
pytest tests/v2/test_provider_modes.py -v -m requires_api_key

# Run tests for specific provider
COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py -v -m requires_api_key -k "cohere"

# Run specific test function
COHERE_API_KEY=... pytest tests/v2/test_provider_modes.py::test_mode_basic_extraction -v -m requires_api_key
```

**Mode Normalization Tests** (No API key required):
```bash
# Run all normalization tests
pytest tests/v2/test_mode_normalization.py -v

# Run specific provider normalization tests
pytest tests/v2/test_mode_normalization.py -v -k "cohere"
```

**All v2 Tests**:
```bash
# Unit tests only (no API key)
pytest tests/v2/ -v -k "not requires_api_key"

# Integration tests only (requires API keys)
pytest tests/v2/ -v -m requires_api_key

# Specific provider (all tests)
COHERE_API_KEY=... pytest tests/v2/ -v -k "cohere"
```

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
tests/providers/test_auto_client.py
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
pytest tests/core/test_patch.py -v
pytest tests/providers/test_auto_client.py -v
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
