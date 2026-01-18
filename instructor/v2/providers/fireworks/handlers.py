"""Fireworks v2 mode handlers.

Fireworks uses an OpenAI-compatible API, so we can reuse OpenAI handlers.
Fireworks supports TOOLS and MD_JSON modes (no native JSON_SCHEMA support).

The handlers are registered with Provider.FIREWORKS but inherit all functionality
from the OpenAI handlers since the API format is identical.
"""

from __future__ import annotations

from instructor import Mode, Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import (
    OpenAIMDJSONHandler,
    OpenAIToolsHandler,
)


@register_mode_handler(Provider.FIREWORKS, Mode.TOOLS)
class FireworksToolsHandler(OpenAIToolsHandler):
    """Handler for Fireworks TOOLS mode.

    Fireworks uses OpenAI-compatible tool calling, so we inherit from OpenAIToolsHandler.
    All prepare_request, handle_reask, and parse_response methods work identically.
    """

    mode = Mode.TOOLS


@register_mode_handler(Provider.FIREWORKS, Mode.MD_JSON)
class FireworksMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for Fireworks MD_JSON mode.

    Fireworks uses OpenAI-compatible text responses, so we inherit from OpenAIMDJSONHandler.
    JSON is extracted from markdown code blocks in the response.
    """

    mode = Mode.MD_JSON


__all__ = [
    "FireworksToolsHandler",
    "FireworksMDJSONHandler",
]
