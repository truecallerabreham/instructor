"""Regression test for messages list mutation in reask handlers.

When a reask handler uses `kwargs = kwargs.copy()` (shallow copy), the
`messages` list inside is shared between the original and the copy.
Calling `.append()` or `.extend()` on `kwargs["messages"]` mutates the
caller's original list, corrupting the caller's state across retries.

This test proves the bug exists by checking that the original messages
list is NOT modified after a reask handler runs.
"""

import pytest
from unittest.mock import MagicMock

from instructor.v2.providers.openai.handlers import (
    reask_tools,
    reask_responses_tools,
    reask_md_json,
    reask_default,
)


def _make_openai_response(tool_calls=None, content="response text"):
    """Create a mock OpenAI ChatCompletion-like response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].message.tool_calls = tool_calls or []
    response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": content,
    }
    return response


def _make_tool_call(name="get_user", tool_call_id="call_123"):
    """Create a mock tool call."""
    tc = MagicMock()
    tc.id = tool_call_id
    tc.function.name = name
    tc.function.arguments = '{"name": "John"}'
    return tc


class TestReaskToolsMutation:
    """Test that reask_tools does not mutate the caller's messages list."""

    def test_does_not_mutate_original_messages_on_stream_response(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.choices = None  # triggers _is_stream_response

        exception = ValueError("validation error")
        result = reask_tools(kwargs, response, exception)

        # The original messages list should have exactly 1 element
        assert len(original_messages) == 1, (
            f"reask_tools mutated the caller's messages list! "
            f"Expected 1 message, got {len(original_messages)}: {original_messages}"
        )
        # The result should have 2 messages (original + error)
        assert len(result["messages"]) == 2

    def test_does_not_mutate_original_messages_on_tool_response(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        tool_call = _make_tool_call()
        response = _make_openai_response(tool_calls=[tool_call])

        exception = ValueError("validation error")
        result = reask_tools(kwargs, response, exception)

        # Original should still have 1 element
        assert len(original_messages) == 1, (
            f"reask_tools mutated the caller's messages list! "
            f"Expected 1 message, got {len(original_messages)}: {original_messages}"
        )
        # Result should have 3 messages (original + assistant + tool_result)
        assert len(result["messages"]) == 3

    def test_does_not_mutate_after_multiple_reasks(self):
        """Simulate the retry loop: reask should not accumulate messages."""
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        tool_call = _make_tool_call()
        response = _make_openai_response(tool_calls=[tool_call])
        exception = ValueError("validation error")

        # First reask
        result1 = reask_tools(kwargs, response, exception)
        assert len(original_messages) == 1

        # Second reask (simulating next retry iteration)
        result2 = reask_tools(result1, response, exception)
        # Original should STILL have 1 element
        assert len(original_messages) == 1, (
            f"Messages accumulated across retries! "
            f"Expected 1, got {len(original_messages)}"
        )
        # Each result should independently have the right count
        assert len(result2["messages"]) == 5  # 1 original + 2 from first + 2 from second


class TestReaskResponsesToolsMutation:
    """Test that reask_responses_tools does not mutate the caller's messages list."""

    def test_does_not_mutate_original_messages(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.output = []  # triggers empty output path

        exception = ValueError("validation error")
        result = reask_responses_tools(kwargs, response, exception)

        assert len(original_messages) == 1, (
            f"reask_responses_tools mutated the caller's messages list!"
        )
        assert len(result["messages"]) == 2


class TestReaskMdJsonMutation:
    """Test that reask_md_json does not mutate the caller's messages list."""

    def test_does_not_mutate_original_messages_on_stream(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.choices = None  # triggers _is_stream_response

        exception = ValueError("validation error")
        result = reask_md_json(kwargs, response, exception)

        assert len(original_messages) == 1, (
            f"reask_md_json mutated the caller's messages list!"
        )
        assert len(result["messages"]) == 2

    def test_does_not_mutate_original_messages_on_text_response(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = '{"key": "value"}'
        response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": '{"key": "value"}',
        }

        exception = ValueError("validation error")
        result = reask_md_json(kwargs, response, exception)

        assert len(original_messages) == 1, (
            f"reask_md_json mutated the caller's messages list!"
        )
        assert len(result["messages"]) == 3  # original + assistant + error


class TestReaskDefaultMutation:
    """Test that reask_default does not mutate the caller's messages list."""

    def test_does_not_mutate_original_messages_on_stream(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.choices = None

        exception = ValueError("validation error")
        result = reask_default(kwargs, response, exception)

        assert len(original_messages) == 1, (
            f"reask_default mutated the caller's messages list!"
        )
        assert len(result["messages"]) == 2

    def test_does_not_mutate_original_messages_on_text_response(self):
        original_messages = [{"role": "user", "content": "hello"}]
        kwargs = {"messages": original_messages, "model": "gpt-4"}

        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "some response"
        response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": "some response",
        }

        exception = ValueError("validation error")
        result = reask_default(kwargs, response, exception)

        assert len(original_messages) == 1, (
            f"reask_default mutated the caller's messages list!"
        )
        assert len(result["messages"]) == 3
