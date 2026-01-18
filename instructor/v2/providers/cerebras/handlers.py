"""Cerebras v2 mode handlers.

Cerebras uses an OpenAI-compatible API, so we can reuse OpenAI handlers.
Cerebras supports TOOLS and MD_JSON modes (no native JSON_SCHEMA support).

The handlers are registered with Provider.CEREBRAS but inherit all functionality
from the OpenAI handlers since the API format is identical.
"""

from __future__ import annotations

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import (
    OpenAIMDJSONHandler,
    OpenAIToolsHandler,
)


@register_mode_handler(Provider.CEREBRAS, Mode.TOOLS)
class CerebrasToolsHandler(OpenAIToolsHandler):
    """Handler for Cerebras TOOLS mode.

    Cerebras uses OpenAI-compatible tool calling, so we inherit from OpenAIToolsHandler.
    All prepare_request, handle_reask, and parse_response methods work identically.
    """

    mode = Mode.TOOLS


@register_mode_handler(Provider.CEREBRAS, Mode.MD_JSON)
class CerebrasMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for Cerebras MD_JSON mode.

    Cerebras uses OpenAI-compatible text responses, so we inherit from OpenAIMDJSONHandler.
    JSON is extracted from markdown code blocks in the response.
    """

    mode = Mode.MD_JSON


__all__ = [
    "CerebrasToolsHandler",
    "CerebrasMDJSONHandler",
]
