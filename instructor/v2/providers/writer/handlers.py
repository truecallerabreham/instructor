"""Writer v2 mode handlers.

Writer supports TOOLS and MD_JSON modes. The API is similar to OpenAI but uses
`client.chat.chat` instead of `client.chat.completions.create`.

The handlers reuse some patterns from OpenAI but have Writer-specific
request preparation and response parsing.
"""

from __future__ import annotations

import json
from textwrap import dedent
from typing import Any

from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.core.exceptions import IncompleteOutputException
from instructor.processing.function_calls import extract_json_from_codeblock
from instructor.processing.schema import generate_openai_schema
from instructor.providers.writer.utils import reask_writer_json, reask_writer_tools
from instructor.utils.core import merge_consecutive_messages
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


@register_mode_handler(Provider.WRITER, Mode.TOOLS)
class WriterToolsHandler(ModeHandler):
    """Handler for Writer TOOLS mode.

    Writer uses OpenAI-compatible tool calling format. Tools are defined
    with function schemas and the model returns tool calls with arguments.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions for Writer."""
        if response_model is None:
            return None, kwargs

        from instructor.utils.core import prepare_response_model

        response_model = prepare_response_model(response_model)

        new_kwargs = kwargs.copy()
        schema = generate_openai_schema(response_model)

        new_kwargs["tools"] = [{"type": "function", "function": schema}]
        new_kwargs["tool_choice"] = "auto"

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Writer tools mode."""
        return reask_writer_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        """Parse tool call response from Writer."""
        # Check for truncated output
        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Extract JSON from tool call
        tool_call = response.choices[0].message.tool_calls[0]
        json_str = tool_call.function.arguments

        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


@register_mode_handler(Provider.WRITER, Mode.MD_JSON)
class WriterMDJSONHandler(ModeHandler):
    """Handler for Writer MD_JSON mode.

    Extracts JSON from markdown code blocks in the response text.
    This is a fallback mode when tool calling is not suitable.
    """

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema instruction in messages."""
        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()

        message = dedent(
            f"""
            As a genius expert, your task is to understand the content and provide
            the parsed objects in json that match the following json_schema:\n

            {json.dumps(schema, indent=2, ensure_ascii=False)}

            Make sure to return an instance of the JSON, not the schema itself
            """
        )

        # Add system message with schema
        messages = new_kwargs.get("messages", [])
        if messages and messages[0]["role"] != "system":
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": message,
                },
            )
        elif messages and isinstance(messages[0]["content"], str):
            messages[0]["content"] += f"\n\n{message}"
        elif messages and isinstance(messages[0]["content"], list):
            messages[0]["content"][0]["text"] += f"\n\n{message}"
        else:
            messages.insert(0, {"role": "system", "content": message})

        # Add user message requesting JSON in code block
        messages.append(
            {
                "role": "user",
                "content": "Return the correct JSON response within a ```json codeblock. not the JSON_SCHEMA",
            },
        )
        new_kwargs["messages"] = merge_consecutive_messages(messages)

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Writer MD_JSON mode."""
        return reask_writer_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> BaseModel:
        """Parse JSON from markdown code block in response."""
        text = response.choices[0].message.content or ""
        json_str = extract_json_from_codeblock(text)

        return response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )


__all__ = [
    "WriterToolsHandler",
    "WriterMDJSONHandler",
]
