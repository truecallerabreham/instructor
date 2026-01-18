"""Bedrock v2 mode handlers."""

from __future__ import annotations

import re
from typing import Any, cast

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.core.exceptions import ConfigurationError, ResponseParsingError
from instructor.providers.bedrock.utils import (
    handle_bedrock_json,
    handle_bedrock_tools,
    reask_bedrock_json,
    reask_bedrock_tools,
)
from instructor.utils.core import prepare_response_model
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


def _extract_bedrock_text(response: Any) -> str:
    """Extract text from Bedrock response formats."""
    if isinstance(response, dict):
        content = response.get("output", {}).get("message", {}).get("content", [])
        text_block = next((block for block in content if "text" in block), None)
        if not text_block:
            raise ResponseParsingError(
                "Unexpected Bedrock response format: no text content found.",
                mode="BEDROCK_JSON",
                raw_response=response,
            )
        return text_block["text"]
    if hasattr(response, "text"):
        return response.text
    raise ResponseParsingError(
        "Unexpected Bedrock response format: no text attribute found.",
        mode="BEDROCK_JSON",
        raw_response=response,
    )


def _extract_bedrock_tool_input(
    response: Any, response_model: type[BaseModel]
) -> dict[str, Any]:
    """Extract tool input from Bedrock tool-use responses."""
    if not isinstance(response, dict):
        raise ResponseParsingError(
            "Unexpected Bedrock response format: expected dict response.",
            mode="BEDROCK_TOOLS",
            raw_response=response,
        )

    message = response.get("output", {}).get("message", {})
    content = message.get("content", [])
    for content_block in content:
        if "toolUse" in content_block:
            tool_use = content_block["toolUse"]
            if tool_use.get("name") != response_model.__name__:
                raise ResponseParsingError(
                    f"Tool name mismatch: expected {response_model.__name__}, "
                    f"got {tool_use.get('name')}",
                    mode="BEDROCK_TOOLS",
                    raw_response=response,
                )
            return tool_use.get("input", {})

    raise ResponseParsingError(
        "No tool use found in Bedrock response.",
        mode="BEDROCK_TOOLS",
        raw_response=response,
    )


@register_mode_handler(Provider.BEDROCK, Mode.TOOLS)
class BedrockToolsHandler(ModeHandler):
    """Handler for Bedrock TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return handle_bedrock_tools(None, new_kwargs)

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_tools(prepared_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_bedrock_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Bedrock in TOOLS mode."
            )
        tool_input = _extract_bedrock_tool_input(response, response_model)
        return response_model.model_validate(
            tool_input,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.BEDROCK, Mode.MD_JSON)
class BedrockMDJSONHandler(ModeHandler):
    """Handler for Bedrock MD_JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return None, new_kwargs

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))
        return handle_bedrock_json(prepared_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_bedrock_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        if stream:
            raise ConfigurationError(
                "Streaming is not supported for Bedrock in MD_JSON mode."
            )
        text = _extract_bedrock_text(response)
        match = re.search(r"```?json(.*?)```?", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        text = re.sub(r"```?json|\\n", "", text).strip()
        return response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )


__all__ = [
    "BedrockToolsHandler",
    "BedrockMDJSONHandler",
]
