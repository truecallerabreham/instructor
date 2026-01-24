"""Unit tests for Writer v2 handlers.

These tests verify handler behavior without requiring API keys by using mock responses.
Writer supports TOOLS and MD_JSON modes.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider
from instructor.v2.core.registry import mode_registry


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


class User(BaseModel):
    """User model for testing."""

    name: str
    age: int


class MockToolCall:
    """Mock tool call for testing."""

    _counter = 0

    def __init__(self, name: str, arguments: dict[str, Any] | str):
        MockToolCall._counter += 1
        self.id = f"call_{MockToolCall._counter}"
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        if isinstance(arguments, dict):
            self.function.arguments = json.dumps(arguments)
        else:
            self.function.arguments = arguments


class MockMessage:
    """Mock message for testing."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MockToolCall] | None = None,
        role: str = "assistant",
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.role = role

    def model_dump(self) -> dict[str, Any]:
        """Return dict representation for compatibility."""
        result: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for i, tc in enumerate(self.tool_calls)
            ]
        return result


class MockChoice:
    """Mock choice for testing."""

    def __init__(
        self,
        message: MockMessage,
        finish_reason: str = "stop",
    ):
        self.message = message
        self.finish_reason = finish_reason


class MockResponse:
    """Mock Writer response for testing."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MockToolCall] | None = None,
        finish_reason: str = "stop",
    ):
        self.choices = [MockChoice(MockMessage(content, tool_calls), finish_reason)]


# ============================================================================
# WriterToolsHandler Tests
# ============================================================================


class TestWriterToolsHandler:
    """Tests for WriterToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS)
        return handlers

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert "messages" in result_kwargs

    def test_prepare_request_adds_tool_schema(self, handler):
        """Test prepare_request adds tool schema for response model."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs
        assert len(result_kwargs["tools"]) == 1
        assert result_kwargs["tools"][0]["type"] == "function"
        assert "tool_choice" in result_kwargs
        assert result_kwargs["tool_choice"] == "auto"

    def test_prepare_request_preserves_original_kwargs(self, handler):
        """Test prepare_request doesn't modify original kwargs."""
        original_kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 100,
        }
        kwargs_copy = original_kwargs.copy()
        handler.request_handler(Answer, original_kwargs)

        # Original should be unchanged
        assert original_kwargs == kwargs_copy

    def test_parse_response_from_tool_calls(self, handler):
        """Test parsing response from tool_calls."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": 4.0})])

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 4.0

    def test_parse_response_with_validation_context(self, handler):
        """Test parsing with validation context."""
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": 5.0})])

        result = handler.response_parser(
            response,
            Answer,
            validation_context={"test": "context"},
        )

        assert isinstance(result, Answer)
        assert result.answer == 5.0

    def test_handle_reask_adds_messages(self, handler):
        """Test handle_reask adds error message to conversation."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(tool_calls=[MockToolCall("Answer", {"answer": "bad"})])
        exception = ValueError("Validation failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added messages for reask
        assert len(result["messages"]) > 1

    def test_tools_handler_preserves_extra_kwargs(self, handler):
        """Test TOOLS handler preserves extra kwargs."""
        kwargs = {
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_kwargs["max_tokens"] == 500
        assert result_kwargs["temperature"] == 0.7


# ============================================================================
# WriterMDJSONHandler Tests
# ============================================================================


class TestWriterMDJSONHandler:
    """Tests for WriterMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.MD_JSON)
        return handlers

    def test_prepare_request_with_none_model(self, handler):
        """Test prepare_request returns unchanged kwargs when response_model is None."""
        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        result_model, result_kwargs = handler.request_handler(None, kwargs)

        assert result_model is None
        assert result_kwargs == kwargs

    def test_prepare_request_adds_system_message(self, handler):
        """Test prepare_request adds system message with schema."""
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        assert result_model is Answer
        messages = result_kwargs["messages"]

        # Should have system message at start
        assert messages[0]["role"] == "system"
        assert "json_schema" in messages[0]["content"]

    def test_prepare_request_appends_to_existing_system(self, handler):
        """Test prepare_request appends to existing system message."""
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
        }
        result_model, result_kwargs = handler.request_handler(Answer, kwargs)

        messages = result_kwargs["messages"]
        system_msg = messages[0]

        assert system_msg["role"] == "system"
        assert "You are helpful." in system_msg["content"]
        assert "json_schema" in system_msg["content"]

    def test_parse_response_from_markdown_codeblock(self, handler):
        """Test parsing JSON from markdown code block."""
        response = MockResponse(content='```json\n{"answer": 13.0}\n```')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 13.0

    def test_parse_response_from_plain_json(self, handler):
        """Test parsing plain JSON (no code block)."""
        response = MockResponse(content='{"answer": 14.0}')

        result = handler.response_parser(response, Answer)

        assert isinstance(result, Answer)
        assert result.answer == 14.0

    def test_handle_reask_adds_message(self, handler):
        """Test handle_reask adds user message with error."""
        kwargs = {"messages": [{"role": "user", "content": "Original"}]}
        response = MockResponse(content="Invalid")
        exception = ValueError("JSON extraction failed")

        result = handler.reask_handler(kwargs, response, exception)

        # Should have added messages for reask
        assert len(result["messages"]) > 1


# ============================================================================
# Handler Registration Tests
# ============================================================================


class TestWriterHandlerRegistration:
    """Tests for Writer handler registration in the v2 registry."""

    @pytest.mark.parametrize(
        "mode",
        [Mode.TOOLS, Mode.MD_JSON],
    )
    def test_mode_is_registered(self, mode: Mode):
        """Test all Writer modes are registered."""
        assert mode_registry.is_registered(Provider.WRITER, mode)

    @pytest.mark.parametrize(
        "mode",
        [Mode.TOOLS, Mode.MD_JSON],
    )
    def test_handlers_have_all_methods(self, mode: Mode):
        """Test all handlers have required methods."""
        handlers = mode_registry.get_handlers(Provider.WRITER, mode)

        assert handlers.request_handler is not None
        assert handlers.reask_handler is not None
        assert handlers.response_parser is not None

    def test_get_modes_for_provider(self):
        """Test getting all modes for Writer provider."""
        modes = mode_registry.get_modes_for_provider(Provider.WRITER)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes

    def test_json_schema_not_supported(self):
        """Test JSON_SCHEMA mode is NOT supported by Writer."""
        assert not mode_registry.is_registered(Provider.WRITER, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS mode is NOT supported by Writer."""
        assert not mode_registry.is_registered(Provider.WRITER, Mode.PARALLEL_TOOLS)


# ============================================================================
# Legacy Mode Normalization Tests
# ============================================================================


class TestWriterModeNormalization:
    """Tests for Writer mode handling in v2."""

    def test_writer_tools_normalizes_to_tools(self):
        """Test WRITER_TOOLS is not registered in v2."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.WRITER_TOOLS)
        assert result == Mode.WRITER_TOOLS
        assert not mode_registry.is_registered(Provider.WRITER, Mode.WRITER_TOOLS)

    def test_writer_json_normalizes_to_md_json(self):
        """Test WRITER_JSON is not registered in v2."""
        from instructor.v2.core.registry import mode_registry, normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.WRITER_JSON)
        assert result == Mode.WRITER_JSON
        assert not mode_registry.is_registered(Provider.WRITER, Mode.WRITER_JSON)

    def test_generic_tools_passes_through(self):
        """Test generic TOOLS mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.TOOLS)
        assert result == Mode.TOOLS

    def test_generic_md_json_passes_through(self):
        """Test generic MD_JSON mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.WRITER, Mode.MD_JSON)
        assert result == Mode.MD_JSON


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestWriterHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tools_handler_with_complex_model(self):
        """Test TOOLS handler with nested model."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS)

        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        kwargs = {"messages": [{"role": "user", "content": "Get person info"}]}
        result_model, result_kwargs = handlers.request_handler(Person, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs

    def test_md_json_handler_with_strict_validation(self):
        """Test MD_JSON handler with strict validation."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.MD_JSON)
        response = MockResponse(content='{"answer": 21.0}')

        result = handlers.response_parser(
            response,
            Answer,
            strict=True,
        )

        assert isinstance(result, Answer)
        assert result.answer == 21.0

    def test_tools_handler_with_list_model(self):
        """Test TOOLS handler with list model."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS)

        class Item(BaseModel):
            name: str
            price: float

        kwargs = {"messages": [{"role": "user", "content": "List items"}]}
        result_model, result_kwargs = handlers.request_handler(Item, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs

    def test_parse_response_with_user_model(self):
        """Test parsing response with User model."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.TOOLS)
        response = MockResponse(
            tool_calls=[MockToolCall("User", {"name": "Alice", "age": 30})]
        )

        result = handlers.response_parser(response, User)

        assert isinstance(result, User)
        assert result.name == "Alice"
        assert result.age == 30

    def test_md_json_parse_with_nested_json(self):
        """Test MD_JSON parsing with nested JSON in code block."""
        handlers = mode_registry.get_handlers(Provider.WRITER, Mode.MD_JSON)
        response = MockResponse(content='```json\n{"name": "Bob", "age": 25}\n```')

        result = handlers.response_parser(response, User)

        assert isinstance(result, User)
        assert result.name == "Bob"
        assert result.age == 25
