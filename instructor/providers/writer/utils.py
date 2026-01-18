"""Writer-specific utilities.

This module contains utilities specific to the Writer provider,
including reask functions, response handlers, and message formatting.
"""

from __future__ import annotations

from typing import Any

from ...mode import Mode
from ...processing.schema import generate_openai_schema
from ...utils.core import dump_message


def _extract_reask_message(response: Any) -> dict[str, Any]:
    """Best-effort extraction of a message dict for Writer reask flows."""
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        try:
            return dump_message(message)
        except Exception:
            return {
                "role": getattr(message, "role", "assistant"),
                "content": getattr(message, "content", ""),
            }
    if hasattr(response, "text"):
        return {"role": "assistant", "content": response.text}
    return {"role": "assistant", "content": getattr(response, "content", str(response))}


def reask_writer_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Writer tools mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user instructions to correct tool call)
    """
    kwargs = kwargs.copy()
    reask_msgs = [_extract_reask_message(response)]
    reask_msgs.append(
        {
            "role": "user",
            "content": (
                f"Validation Error found:\n{exception}\n Fix errors and fill tool call arguments/name "
                f"correctly. Just update arguments dict values or update name. Don't change the structure "
                f"of them. You have to call function by passing desired "
                f"functions name/args as part of special attribute with name tools_calls, "
                f"not as text in attribute with name content. IT'S IMPORTANT!"
            ),
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_writer_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """
    Handle reask for Writer JSON mode when validation fails.

    Kwargs modifications:
    - Adds: "messages" (user message requesting JSON correction)
    """
    kwargs = kwargs.copy()
    base_message = _extract_reask_message(response)
    reask_msgs = [base_message]
    reask_msgs.append(
        {
            "role": "user",
            "content": f"Correct your JSON response: {base_message.get('content', '')}, "
            f"based on the following errors:\n{exception}",
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def handle_writer_tools(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Writer tools mode.

    Kwargs modifications:
    - Adds: "tools" (list with function schema)
    - Sets: "tool_choice" to "auto"
    """
    new_kwargs["tools"] = [
        {
            "type": "function",
            "function": generate_openai_schema(response_model),
        }
    ]
    new_kwargs["tool_choice"] = "auto"
    return response_model, new_kwargs


def handle_writer_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """
    Handle Writer JSON mode.

    Kwargs modifications:
    - Adds: "response_format" with json_schema
    """
    new_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": response_model.model_json_schema()},
    }

    return response_model, new_kwargs


# Handler registry for Writer
WRITER_HANDLERS = {
    Mode.WRITER_TOOLS: {
        "reask": reask_writer_tools,
        "response": handle_writer_tools,
    },
    Mode.WRITER_JSON: {
        "reask": reask_writer_json,
        "response": handle_writer_json,
    },
}
