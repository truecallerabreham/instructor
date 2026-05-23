---
authors:
  - jxnl
categories:
  - instructor
comments: true
date: 2026-05-23
description: V2 moves provider behavior beside provider clients while preserving a low-friction upgrade path.
draft: true
slug: whats-new-in-instructor-v2
tags:
  - Instructor
  - Python
  - Structured Outputs
---

# What's New in Instructor V2?

Instructor started with a useful promise: add structured outputs to the client
you already use, and make it easy to remove later. V2 keeps that promise, but
changes where the complexity lives.

Provider SDKs have diverged. Tools, JSON schemas, streaming events, media
payloads, and retry prompts no longer share one honest implementation. In V1,
some of those decisions accumulated in shared processing code and large factory
routing branches. That made a small provider change surprisingly easy to leak
into another provider.

V2 is a provider-owned architecture. Each provider package owns its request
shape, response parsing, reask behavior, media conversion, and model-string
client builder. Shared core code owns the machinery that is actually shared:
patching, retries, registry protocols, validation primitives, and lazy dispatch.

<!-- more -->

## The Upgrade Path

This is intended to be a low-friction upgrade. Existing factory imports and
patch helpers remain available as lazy compatibility facades. New code can use
the unified form:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


client = instructor.from_provider("openai/gpt-4o-mini")
user = client.create(
    response_model=User,
    messages=[{"role": "user", "content": "Jason is 25 years old"}],
)
```

Provider-specific legacy modes continue to normalize to core modes with a
deprecation warning. That gives applications time to migrate while keeping one
clear mode vocabulary for new code.

## What Changed Under the Hood

`from_provider()` is now a small lazy dispatcher. It parses a provider/model
string and imports the builder beside that provider's client only when needed.
OpenAI-compatible providers share a higher-order builder where their native
client construction is genuinely identical; endpoints, credentials, and
provider-specific defaults stay in the owning provider module.

Handlers already use decorator registration. V2 extends that idea with a
single `ProviderSpec` manifest for aliases, factories, modes, and user-visible
capabilities. Tests draw their matrices from that manifest rather than
maintaining parallel configuration dictionaries.

Provider behavior also stays local. Gemini and GenAI schema and media
conversion are handled in their provider packages. Anthropic builds Anthropic
tool schemas. OpenAI-compatible providers use their compatible wire helpers
without asking a generic response schema object to understand every SDK.

## Honest Feature Support

Uniform APIs are useful; imaginary uniform capabilities are not. A provider
may support partial streaming but not iterable streaming, image inputs but not
audio, or multiple tool calls only through a particular mode.

V2 records those differences explicitly. For example, GenAI exposes partial
and iterable streaming for its declared modes, while the legacy Gemini and
VertexAI clients currently advertise partial streaming only. xAI advertises
the streaming modes its public client implements, rather than every handler
path that happens to exist internally.

The result is a cleaner rule: documentation and conformance tests should claim
only what the provider contract declares.

## Why It Matters

Smaller dispatch code means fewer special cases during provider setup. Local
wire-format logic means a fix for one SDK is less likely to change another.
Lazy compatibility facades protect import time and memory for users who do not
install every provider SDK. Parameterized conformance tests and editor-facing
type checks make those boundaries visible before release.

V2 is not a new abstraction layered on top of every provider. It is a more
honest place to put the differences that already exist, while preserving the
simple Instructor workflow users rely on.
