"""Anthropic v2 utilities for system message handling."""

from __future__ import annotations

from typing import Any, TypedDict, Union


class SystemMessage(TypedDict, total=False):
    type: str
    text: str
    cache_control: dict[str, str]


def combine_system_messages(
    existing_system: Union[str, list[SystemMessage], None],  # noqa: UP007
    new_system: Union[str, list[SystemMessage]],  # noqa: UP007
) -> Union[str, list[SystemMessage]]:  # noqa: UP007
    """
    Combine existing and new system messages.
    """
    if existing_system is None:
        return new_system

    if not isinstance(existing_system, (str, list)) or not isinstance(
        new_system, (str, list)
    ):
        raise ValueError(
            f"System messages must be strings or lists, got {type(existing_system)} and {type(new_system)}"
        )

    if isinstance(existing_system, str) and isinstance(new_system, str):
        return f"{existing_system}\n\n{new_system}"
    if isinstance(existing_system, list) and isinstance(new_system, list):
        result = list(existing_system)
        result.extend(new_system)
        return result
    if isinstance(existing_system, str) and isinstance(new_system, list):
        result = [SystemMessage(type="text", text=existing_system)]
        result.extend(new_system)
        return result
    if isinstance(existing_system, list) and isinstance(new_system, str):
        new_message = SystemMessage(type="text", text=new_system)
        result = list(existing_system)
        result.append(new_message)
        return result

    return existing_system


def extract_system_messages(messages: list[dict[str, Any]]) -> list[SystemMessage]:
    """
    Extract system messages from a list of messages.
    """
    if not messages:
        return []

    system_count = sum(1 for m in messages if m.get("role") == "system")
    if system_count == 0:
        return []

    def convert_message(content: Any) -> SystemMessage:
        if isinstance(content, str):
            return SystemMessage(type="text", text=content)
        if isinstance(content, dict):
            return SystemMessage(**content)
        raise ValueError(f"Unsupported content type: {type(content)}")

    result: list[SystemMessage] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if not content:
                continue
            if isinstance(content, list):
                for item in content:
                    if item:
                        result.append(convert_message(item))
            else:
                result.append(convert_message(content))

    return result
