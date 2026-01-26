"""Perplexity v2 mode handlers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.v2.providers.perplexity.utils import (
    handle_perplexity_json,
    reask_perplexity_json,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.providers.openai.handlers import OpenAIMDJSONHandler


@register_mode_handler(Provider.PERPLEXITY, Mode.MD_JSON)
class PerplexityMDJSONHandler(OpenAIMDJSONHandler):
    """Handler for Perplexity JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        if response_model is None:
            return None, kwargs
        new_kwargs = kwargs.copy()
        return handle_perplexity_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_perplexity_json(kwargs, response, exception)


__all__ = ["PerplexityMDJSONHandler"]
