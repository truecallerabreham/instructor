"""xAI v2 utilities."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from xai_sdk import chat as xchat
else:
    try:
        from xai_sdk import chat as xchat
    except ImportError:
        xchat = None


def _convert_messages(messages: list[dict[str, Any]]):
    """Convert OpenAI-style messages to xAI format."""
    if xchat is None:
        raise ImportError("xai_sdk is required for xAI provider")

    converted = []
    for m in messages:
        role = m["role"]
        content = m.get("content", "")
        if isinstance(content, str):
            c = xchat.text(content)
        else:
            raise ValueError("Only string content supported for xAI provider")
        if role == "user":
            converted.append(xchat.user(c))
        elif role == "assistant":
            converted.append(xchat.assistant(c))
        elif role == "system":
            converted.append(xchat.system(c))
        elif role == "tool":
            converted.append(xchat.tool_result(content))
        else:
            raise ValueError(f"Unsupported role: {role}")
    return converted


def reask_xai_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for xAI JSON mode when validation fails."""
    kwargs = kwargs.copy()
    reask_msg = {
        "role": "user",
        "content": f"Validation Errors found:\n{exception}\nRecall the function correctly, fix the errors found in the following attempt:\n{response}",
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def reask_xai_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for xAI tools mode when validation fails."""
    kwargs = kwargs.copy()

    assistant_msg = {
        "role": "assistant",
        "content": str(response),
    }
    kwargs["messages"].append(assistant_msg)

    reask_msg = {
        "role": "user",
        "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors",
    }
    kwargs["messages"].append(reask_msg)
    return kwargs


def handle_xai_json(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle xAI JSON mode."""
    messages = new_kwargs.get("messages", [])
    new_kwargs["x_messages"] = _convert_messages(messages)

    new_kwargs.pop("max_retries", None)
    new_kwargs.pop("context", None)
    new_kwargs.pop("hooks", None)

    return response_model, new_kwargs


def handle_xai_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle xAI tools mode."""
    messages = new_kwargs.get("messages", [])
    new_kwargs["x_messages"] = _convert_messages(messages)

    new_kwargs.pop("max_retries", None)
    new_kwargs.pop("context", None)
    new_kwargs.pop("hooks", None)

    if response_model is not None and xchat is not None:
        new_kwargs["tool"] = xchat.tool(
            name=response_model.__name__,
            description=response_model.__doc__ or "",
            parameters=response_model.model_json_schema(),
        )

    return response_model, new_kwargs
