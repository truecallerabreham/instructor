"""Groq v2 mode handlers.

Groq uses an OpenAI-compatible API, so we can reuse OpenAI handlers.
Groq supports TOOLS and MD_JSON modes (no native JSON_SCHEMA support).

The handlers are registered with Provider.GROQ but inherit all functionality
from the OpenAI handlers since the API format is identical.
"""

from __future__ import annotations

from instructor import Mode, Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import (
    OpenAIMDJSONHandler,
    OpenAIToolsHandler,
)


@register_mode_handler(Provider.GROQ, Mode.TOOLS)
class GroqToolsHandler(OpenAIToolsHandler):
    """Handler for Groq TOOLS mode.

    Groq uses OpenAI-compatible tool calling, so we inherit from OpenAIToolsHandler.
    All prepare_request, handle_reask, and parse_response methods work identically.
    """

    mode = Mode.TOOLS


@register_mode_handler(Provider.GROQ, Mode.MD_JSON)
class GroqMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for Groq MD_JSON mode.

    Groq uses OpenAI-compatible text responses, so we inherit from OpenAIMDJSONHandler.
    JSON is extracted from markdown code blocks in the response.
    """

    mode = Mode.MD_JSON


__all__ = [
    "GroqToolsHandler",
    "GroqMDJSONHandler",
]
