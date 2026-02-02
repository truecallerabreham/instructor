"""
Regression test for OpenAI-compatible servers that reject object tool_choice.

Some OpenAI-compatible servers (for example, LM Studio) only accept string values for
`tool_choice` and reject OpenAI's newer object form with an HTTP 400 error.

This test ensures Instructor retries once with a string tool_choice when it sees the
specific "Invalid tool_choice type: 'object'" error message.
"""

from __future__ import annotations

from unittest.mock import Mock

import instructor
from instructor.mode import Mode
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


def test_tool_choice_object_retried_as_string_required() -> None:
    """Retries a 400 tool_choice object error by coercing to 'required'."""
    completion = ChatCompletion(
        id="test_id",
        created=1234567890,
        model="qwen3-vl-4b-instruct",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    refusal=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name="User",
                                arguments='{"name":"Jason","age":25}',
                            ),
                        )
                    ],
                ),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )

    state = {"calls": 0}

    def create_side_effect(*_args, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            # The OpenAI tools handler uses an object tool_choice by default.
            assert isinstance(kwargs.get("tool_choice"), dict)
            raise Exception(
                "Error code: 400 - {'error': \"Invalid tool_choice type: 'object'. Supported string values: none, auto, required\"}"
            )
        assert kwargs.get("tool_choice") == "required"
        return completion

    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock(side_effect=create_side_effect)

    client = instructor.patch(mock_client, mode=Mode.TOOLS)

    user = client.chat.completions.create(
        model="qwen3-vl-4b-instruct",
        response_model=User,
        messages=[{"role": "user", "content": "Extract: Jason is 25 years old"}],
        max_retries=2,
    )

    assert user.name == "Jason"
    assert user.age == 25
    assert state["calls"] == 2

