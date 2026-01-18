---
title: Mode Migration Guide
description: Migrate from provider-specific modes to the core modes in Instructor.
---

# Mode Migration Guide

This guide helps you move from provider-specific modes to the core modes.
Core modes work across providers and are the recommended choice for new code.

## Core Modes

These are the core modes you should use:

- `TOOLS`: Tool or function calling
- `JSON_SCHEMA`: Native schema support when a provider has it
- `MD_JSON`: JSON extracted from text or code blocks
- `PARALLEL_TOOLS`: Multiple tool calls in one response
- `RESPONSES_TOOLS`: OpenAI Responses API tools

## Quick Mapping

Use this table to replace legacy modes:

| Legacy Mode | Core Mode |
|------------|-----------|
| `FUNCTIONS` | `TOOLS` |
| `TOOLS_STRICT` | `TOOLS` |
| `ANTHROPIC_TOOLS` | `TOOLS` |
| `ANTHROPIC_JSON` | `MD_JSON` |
| `COHERE_TOOLS` | `TOOLS` |
| `COHERE_JSON_SCHEMA` | `JSON_SCHEMA` |
| `XAI_TOOLS` | `TOOLS` |
| `XAI_JSON` | `MD_JSON` |
| `MISTRAL_TOOLS` | `TOOLS` |
| `MISTRAL_STRUCTURED_OUTPUTS` | `JSON_SCHEMA` |
| `BEDROCK_TOOLS` | `TOOLS` |
| `BEDROCK_JSON` | `MD_JSON` |

## Example: Anthropic

**Before:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest",
    mode=Mode.ANTHROPIC_TOOLS,
)
```

**After:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "anthropic/claude-3-5-haiku-latest",
    mode=Mode.TOOLS,
)
```

## Example: Bedrock

**Before:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.BEDROCK_TOOLS,
)
```

**After:**

```python
import instructor
from instructor import Mode

client = instructor.from_provider(
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    mode=Mode.TOOLS,
)
```

## Notes

- Legacy modes still work but show a deprecation warning.
- Use core modes for new code and docs.
- See [Mode Comparison](../modes-comparison.md) for details.
