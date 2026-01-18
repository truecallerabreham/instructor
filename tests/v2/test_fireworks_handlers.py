"""Unit tests for Fireworks v2 handlers.

These tests verify handler behavior without requiring API keys by using mock responses.
Fireworks handlers inherit from OpenAI handlers since Fireworks uses an OpenAI-compatible API.
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
        """Return dict representation for OpenAI compatibility."""
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
    """Mock Fireworks response for testing (OpenAI-compatible format)."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list[MockToolCall] | None = None,
        finish_reason: str = "stop",
    ):
        self.choices = [MockChoice(MockMessage(content, tool_calls), finish_reason)]


# ============================================================================
# FireworksToolsHandler Tests
# ============================================================================


class TestFireworksToolsHandler:
    """Tests for FireworksToolsHandler."""

    @pytest.fixture
    def handler(self):
        """Get the TOOLS handler from registry."""
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.TOOLS)
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
# FireworksMDJSONHandler Tests
# ============================================================================


class TestFireworksMDJSONHandler:
    """Tests for FireworksMDJSONHandler."""

    @pytest.fixture
    def handler(self):
        """Get the MD_JSON handler from registry."""
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.MD_JSON)
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


class TestFireworksHandlerRegistration:
    """Tests for Fireworks handler registration in the v2 registry."""

    @pytest.mark.parametrize(
        "mode",
        [Mode.TOOLS, Mode.MD_JSON],
    )
    def test_mode_is_registered(self, mode: Mode):
        """Test all Fireworks modes are registered."""
        assert mode_registry.is_registered(Provider.FIREWORKS, mode)

    @pytest.mark.parametrize(
        "mode",
        [Mode.TOOLS, Mode.MD_JSON],
    )
    def test_handlers_have_all_methods(self, mode: Mode):
        """Test all handlers have required methods."""
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, mode)

        assert handlers.request_handler is not None
        assert handlers.reask_handler is not None
        assert handlers.response_parser is not None

    def test_get_modes_for_provider(self):
        """Test getting all modes for Fireworks provider."""
        modes = mode_registry.get_modes_for_provider(Provider.FIREWORKS)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes

    def test_json_schema_not_supported(self):
        """Test JSON_SCHEMA mode is NOT supported by Fireworks."""
        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS mode is NOT supported by Fireworks."""
        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.PARALLEL_TOOLS)


# ============================================================================
# Handler Inheritance Tests
# ============================================================================


class TestFireworksHandlerInheritance:
    """Tests verifying Fireworks handlers inherit from OpenAI handlers."""

    def test_tools_handler_inherits_from_openai(self):
        """Test FireworksToolsHandler inherits from OpenAIToolsHandler."""
        from instructor.v2.providers.fireworks.handlers import FireworksToolsHandler
        from instructor.v2.providers.openai.handlers import OpenAIToolsHandler

        assert issubclass(FireworksToolsHandler, OpenAIToolsHandler)

    def test_md_json_handler_inherits_from_openai(self):
        """Test FireworksMDJSONHandler inherits from OpenAIMDJSONHandler."""
        from instructor.v2.providers.fireworks.handlers import FireworksMDJSONHandler
        from instructor.v2.providers.openai.handlers import OpenAIMDJSONHandler

        assert issubclass(FireworksMDJSONHandler, OpenAIMDJSONHandler)


# ============================================================================
# Legacy Mode Normalization Tests
# ============================================================================


class TestFireworksModeNormalization:
    """Tests for Fireworks legacy mode normalization."""

    def test_fireworks_tools_normalizes_to_tools(self):
        """Test FIREWORKS_TOOLS normalizes to TOOLS."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="FIREWORKS_TOOLS is deprecated"):
            result = normalize_mode(Provider.FIREWORKS, Mode.FIREWORKS_TOOLS)
        assert result == Mode.TOOLS

    def test_fireworks_json_normalizes_to_md_json(self):
        """Test FIREWORKS_JSON normalizes to MD_JSON."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="FIREWORKS_JSON is deprecated"):
            result = normalize_mode(Provider.FIREWORKS, Mode.FIREWORKS_JSON)
        assert result == Mode.MD_JSON

    def test_generic_tools_passes_through(self):
        """Test generic TOOLS mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.FIREWORKS, Mode.TOOLS)
        assert result == Mode.TOOLS

    def test_generic_md_json_passes_through(self):
        """Test generic MD_JSON mode passes through unchanged."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.FIREWORKS, Mode.MD_JSON)
        assert result == Mode.MD_JSON


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestFireworksHandlerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tools_handler_with_complex_model(self):
        """Test TOOLS handler with nested model."""
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.TOOLS)

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
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.MD_JSON)
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
        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.TOOLS)

        class Item(BaseModel):
            name: str
            price: float

        kwargs = {"messages": [{"role": "user", "content": "List items"}]}
        result_model, result_kwargs = handlers.request_handler(Item, kwargs)

        assert result_model is not None
        assert "tools" in result_kwargs

    def test_incomplete_output_raises_exception(self):
        """Test that incomplete output raises IncompleteOutputException."""
        from instructor.core.exceptions import IncompleteOutputException

        handlers = mode_registry.get_handlers(Provider.FIREWORKS, Mode.TOOLS)
        response = MockResponse(
            tool_calls=[MockToolCall("Answer", {"answer": 4.0})],
            finish_reason="length",
        )

        with pytest.raises(IncompleteOutputException):
            handlers.response_parser(response, Answer)
