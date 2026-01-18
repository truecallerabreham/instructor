---
title: Mode Comparison Guide
description: Compare different modes available in Instructor and understand when to use each
---

# Instructor Mode Comparison Guide

Instructor uses **core modes** that work across providers. Provider-specific
modes still work, but they are deprecated and will show warnings.

## Core Modes

| Mode | Description | Best For |
|------|-------------|----------|
| `TOOLS` | Tool or function calling | Most structured extraction |
| `JSON_SCHEMA` | Native schema support | Providers with built-in schema |
| `MD_JSON` | JSON from text or code blocks | Simple or fallback cases |
| `PARALLEL_TOOLS` | Multiple tool calls | Multi-task extraction |
| `RESPONSES_TOOLS` | OpenAI Responses API tools | OpenAI-only features |

## Legacy Modes (Deprecated)

These legacy modes map to core modes:

| Legacy Mode | Core Mode |
|------------|-----------|
| `FUNCTIONS` | `TOOLS` |
| `TOOLS_STRICT` | `TOOLS` |
| `ANTHROPIC_TOOLS` | `TOOLS` |
| `ANTHROPIC_JSON` | `MD_JSON` |
| `GENAI_TOOLS` | `TOOLS` |
| `GENAI_JSON` | `JSON` |
| `MISTRAL_TOOLS` | `TOOLS` |
| `MISTRAL_STRUCTURED_OUTPUTS` | `JSON_SCHEMA` |
| `BEDROCK_TOOLS` | `TOOLS` |
| `BEDROCK_JSON` | `MD_JSON` |

## Mode Selection Tips

- Use `TOOLS` for most structured output cases.
- Use `JSON_SCHEMA` when the provider supports native schema enforcement.
- Use `MD_JSON` if tools are not supported or outputs are simple.
- Use `PARALLEL_TOOLS` for multiple tasks in one response.

## Examples

### TOOLS Mode (Recommended)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=Mode.TOOLS,
)
```

### MD_JSON Mode (Fallback)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.MD_JSON,
)
```

### JSON_SCHEMA Mode (Native Schema)

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=Mode.JSON_SCHEMA,
)
```

See the [Mode Migration Guide](concepts/mode-migration.md) for more details.

### Google/Gemini

```python
# For complex structures (recommended)
client = instructor.from_provider(
    "google/gemini-2.5-flash",
    mode=instructor.Mode.TOOLS
)

# For structured outputs with JSON (recommended)
client = instructor.from_provider(
    "google/gemini-2.5-flash",
    mode=instructor.Mode.JSON
)
```

## Mode Compatibility Table

| Provider   | Tool-based Modes | JSON-based Modes |
|------------|------------------|------------------|
| OpenAI     | TOOLS, TOOLS_STRICT, PARALLEL_TOOLS, FUNCTIONS | JSON, MD_JSON, JSON_O1 |
| Anthropic  | ANTHROPIC_TOOLS, ANTHROPIC_PARALLEL_TOOLS | ANTHROPIC_JSON |
| Gemini     | TOOLS | JSON |
| Vertex AI  | VERTEXAI_TOOLS | VERTEXAI_JSON |
| Cohere     | COHERE_TOOLS | JSON, MD_JSON |
| Mistral    | MISTRAL_TOOLS | MISTRAL_STRUCTURED_OUTPUTS |
| Anyscale   | - | JSON, MD_JSON, JSON_SCHEMA |
| Databricks | TOOLS | JSON, MD_JSON |
| Together   | - | JSON, MD_JSON |
| Fireworks  | FIREWORKS_TOOLS | FIREWORKS_JSON |
| Cerebras   | - | CEREBRAS_JSON |
| Writer     | WRITER_TOOLS | JSON |
| Perplexity | - | PERPLEXITY_JSON |
| GenAI      | TOOLS | JSON |
| LiteLLM    | (depends on provider) | (depends on provider) |

## Best Practices

1. **Start with the recommended mode for your provider**
    - For OpenAI: `TOOLS`
    - For Anthropic: `ANTHROPIC_TOOLS` (Claude 3+) or `ANTHROPIC_JSON`
    - For Gemini: `TOOLS` or `JSON`

2. **Try JSON modes for simple structures or if you encounter issues**
   - JSON modes often work with simpler schemas
   - They may be more token-efficient
   - They work with more models

3. **Use provider-specific modes when available**
   - Provider-specific modes are optimized for that provider
   - They handle special cases and requirements

4. **Test and validate**
   - Different modes may perform differently for your specific use case
   - Always test with your actual data and models