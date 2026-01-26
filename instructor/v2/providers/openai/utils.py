"""OpenAI v2 utilities for reask and structured outputs."""

from __future__ import annotations

import json
from typing import Any

from instructor.utils.core import dump_message


def _is_stream_response(response: Any) -> bool:
    """Check if response is a Stream object rather than a ChatCompletion."""
    return response is None or not hasattr(response, "choices")


def _filter_responses_tool_calls(output_items: list[Any]) -> list[Any]:
    """Return response output items that represent tool calls."""
    tool_calls: list[Any] = []
    for item in output_items:
        item_type = getattr(item, "type", None)
        if item_type in {"function_call", "tool_call"}:
            tool_calls.append(item)
            continue
        if item_type is None and hasattr(item, "arguments"):
            tool_calls.append(item)
    return tool_calls


def _format_responses_tool_call_details(tool_call: Any) -> str:
    """Format tool call name/id details for reask messages."""
    tool_name = getattr(tool_call, "name", None)
    tool_id = (
        getattr(tool_call, "id", None)
        or getattr(tool_call, "call_id", None)
        or getattr(tool_call, "tool_call_id", None)
    )
    details: list[str] = []
    if tool_name:
        details.append(f"name={tool_name}")
    if tool_id:
        details.append(f"id={tool_id}")
    if not details:
        return ""
    return f" (tool call {', '.join(details)})"


def reask_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI tools mode when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
        return kwargs

    reask_msgs = [dump_message(response.choices[0].message)]
    for tool_call in response.choices[0].message.tool_calls:
        reask_msgs.append(
            {
                "role": "tool",  # type: ignore
                "tool_call_id": tool_call.id,
                "name": tool_call.function.name,
                "content": (
                    f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                ),
            }
        )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_responses_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI responses tools mode when validation fails."""
    kwargs = kwargs.copy()

    if response is None or not hasattr(response, "output"):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors"
                ),
            }
        )
        return kwargs

    reask_messages = []
    for tool_call in _filter_responses_tool_calls(response.output):
        details = _format_responses_tool_call_details(tool_call)
        reask_messages.append(
            {
                "role": "user",  # type: ignore
                "content": (
                    f"Validation Error found:\n{exception}\n"
                    "Recall the function correctly, fix the errors with "
                    f"{tool_call.arguments}{details}"
                ),
            }
        )

    kwargs["messages"].extend(reask_messages)
    return kwargs


def reask_md_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI JSON modes when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
            }
        )
        return kwargs

    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_default(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for OpenAI default mode when validation fails."""
    kwargs = kwargs.copy()

    if _is_stream_response(response):
        kwargs["messages"].append(
            {
                "role": "user",
                "content": (
                    f"Recall the function correctly, fix the errors, exceptions found\n{exception}"
                ),
            }
        )
        return kwargs

    reask_msgs = [dump_message(response.choices[0].message)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Recall the function correctly, fix the errors, exceptions found\n{exception}"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_openrouter_structured_outputs(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle OpenRouter structured outputs mode."""
    schema = response_model.model_json_schema()
    schema["additionalProperties"] = False
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": schema,
            "strict": True,
        },
    }
    return response_model, new_kwargs
