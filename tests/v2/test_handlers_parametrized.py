"""Parameterized handler tests for all v2 providers.

These tests exercise handler methods (prepare_request, parse_response, handle_reask)
with shared scenarios and provider-specific mock responses.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from instructor import Mode, Provider
from instructor.processing.function_calls import OpenAISchema
from instructor.v2.core.registry import mode_registry

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_HANDLER_MODULE_PATHS: dict[Provider, Path] = {
    Provider.OPENAI: _PROJECT_ROOT / "instructor/v2/providers/openai/handlers.py",
    Provider.ANTHROPIC: _PROJECT_ROOT / "instructor/v2/providers/anthropic/handlers.py",
    Provider.GENAI: _PROJECT_ROOT / "instructor/v2/providers/genai/handlers.py",
    Provider.COHERE: _PROJECT_ROOT / "instructor/v2/providers/cohere/handlers.py",
    Provider.XAI: _PROJECT_ROOT / "instructor/v2/providers/xai/handlers.py",
    Provider.GROQ: _PROJECT_ROOT / "instructor/v2/providers/groq/handlers.py",
    Provider.MISTRAL: _PROJECT_ROOT / "instructor/v2/providers/mistral/handlers.py",
    Provider.FIREWORKS: _PROJECT_ROOT / "instructor/v2/providers/fireworks/handlers.py",
}
_HANDLERS_LOADED: set[Provider] = set()


def _ensure_handlers_loaded(provider: Provider) -> None:
    if provider in _HANDLERS_LOADED:
        return
    provider_modes = PROVIDER_HANDLER_MODES.get(provider, [])
    if any(mode_registry.is_registered(provider, mode) for mode in provider_modes):
        _HANDLERS_LOADED.add(provider)
        return
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None:
        return
    spec = importlib.util.spec_from_file_location(
        f"tests.v2.handlers_{provider.value}",
        handler_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load handler module for {provider}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDLERS_LOADED.add(provider)


def _get_handlers(provider: Provider, mode: Mode):
    _ensure_handlers_loaded(provider)
    return mode_registry.get_handlers(provider, mode)


class Answer(OpenAISchema):
    """Simple answer model for handler tests."""

    answer: float


class User(OpenAISchema):
    """Simple user model for handler tests."""

    name: str
    age: int


PROVIDER_HANDLER_MODES: dict[Provider, list[Mode]] = {
    Provider.OPENAI: [
        Mode.TOOLS,
        Mode.JSON_SCHEMA,
        Mode.MD_JSON,
        Mode.PARALLEL_TOOLS,
        Mode.RESPONSES_TOOLS,
    ],
    Provider.ANTHROPIC: [
        Mode.TOOLS,
        Mode.JSON,
        Mode.JSON_SCHEMA,
        Mode.PARALLEL_TOOLS,
        Mode.ANTHROPIC_REASONING_TOOLS,
    ],
    Provider.GENAI: [Mode.TOOLS, Mode.JSON],
    Provider.COHERE: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    Provider.XAI: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    Provider.GROQ: [Mode.TOOLS, Mode.MD_JSON],
    Provider.MISTRAL: [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
    Provider.FIREWORKS: [Mode.TOOLS, Mode.MD_JSON],
}


PARSE_SCENARIOS: dict[Provider, dict[Mode, str]] = {
    Provider.OPENAI: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
        Mode.RESPONSES_TOOLS: "responses_output",
    },
    Provider.COHERE: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.XAI: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.GENAI: {
        Mode.JSON: "text",
    },
    Provider.GROQ: {
        Mode.TOOLS: "tool_call",
        Mode.MD_JSON: "markdown",
    },
    Provider.MISTRAL: {
        Mode.TOOLS: "tool_call",
        Mode.JSON_SCHEMA: "text",
        Mode.MD_JSON: "markdown",
    },
    Provider.FIREWORKS: {
        Mode.TOOLS: "tool_call",
        Mode.MD_JSON: "markdown",
    },
}


def _dependency_missing(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:
        return True


def _skip_if_missing(module: str) -> None:
    if _dependency_missing(module):
        pytest.skip(f"{module} is not installed")


def _provider_mode_params() -> Iterable[pytest.ParametrizeArg]:
    params: list[pytest.ParametrizeArg] = []
    for provider, modes in PROVIDER_HANDLER_MODES.items():
        for mode in modes:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


@dataclass(frozen=True)
class MockResponseBuilder:
    """Builds provider-specific mock responses."""

    provider: Provider

    def tool_response(self, args: dict[str, Any]) -> Any:
        if self.provider == Provider.OPENAI:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider == Provider.COHERE:
            tool_call = SimpleNamespace(parameters=args)
            return SimpleNamespace(tool_calls=[tool_call])
        if self.provider == Provider.XAI:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(arguments=json.dumps(args))
            )
            return SimpleNamespace(tool_calls=[tool_call])
        # Groq and Fireworks use OpenAI-compatible format
        if self.provider in {Provider.GROQ, Provider.FIREWORKS}:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        # Mistral uses OpenAI-compatible format but with different structure
        if self.provider == Provider.MISTRAL:
            tool_call = SimpleNamespace(
                function=SimpleNamespace(
                    name="Answer",
                    arguments=json.dumps(args),
                )
            )
            message = SimpleNamespace(content=None, tool_calls=[tool_call])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        raise NotImplementedError(f"Tool response not supported for {self.provider}")

    def text_response(self, text: str) -> Any:
        if self.provider == Provider.OPENAI:
            message = SimpleNamespace(content=text, tool_calls=[])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        if self.provider in {Provider.COHERE, Provider.XAI, Provider.GENAI}:
            return SimpleNamespace(text=text)
        # Groq, Fireworks, and Mistral use OpenAI-compatible format
        if self.provider in {Provider.GROQ, Provider.FIREWORKS, Provider.MISTRAL}:
            message = SimpleNamespace(content=text, tool_calls=[])
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        raise NotImplementedError(f"Text response not supported for {self.provider}")

    def markdown_response(self, text: str) -> Any:
        return self.text_response(f"```json\n{text}\n```")

    def responses_output_response(self, args: dict[str, Any]) -> Any:
        if self.provider != Provider.OPENAI:
            raise NotImplementedError("Responses output only applies to OpenAI")
        item = SimpleNamespace(type="function_call", arguments=json.dumps(args))
        return SimpleNamespace(output=[item])

    def reask_response(self) -> Any:
        if self.provider == Provider.ANTHROPIC:
            return SimpleNamespace(
                content=[_AnthropicContent(type="text", text="Invalid response")]
            )
        # Mistral expects OpenAI-compatible format with choices
        # For reask tests, we create a simple message without tool_calls
        # to avoid issues with dump_message expecting Pydantic models
        if self.provider == Provider.MISTRAL:
            # Create a mock that works with dump_message
            # dump_message expects a ChatCompletionMessage-like object
            class MockMessage:
                def __init__(self):
                    self.role = "assistant"
                    self.content = "Invalid response"
                    self.tool_calls = []

                def model_dump(self):
                    return {
                        "role": self.role,
                        "content": self.content,
                        "tool_calls": self.tool_calls,
                    }

            message = MockMessage()
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])
        return SimpleNamespace(text="Invalid response")


class _AnthropicContent:
    def __init__(self, type: str, text: str | None = None, id: str | None = None):
        self.type = type
        self.text = text
        self.id = id

    def model_dump(self) -> dict[str, Any]:
        return {"type": self.type, "text": self.text, "id": self.id}


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_prepare_request_with_none_model(provider: Provider, mode: Mode) -> None:
    """prepare_request should handle None response_model."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    if provider == Provider.OPENAI and mode == Mode.RESPONSES_TOOLS:
        _skip_if_missing("openai")
    if provider == Provider.MISTRAL and mode == Mode.JSON_SCHEMA:
        _skip_if_missing("mistralai")
    if mode == Mode.PARALLEL_TOOLS:
        pytest.skip("Parallel tools requires special response_model setup")
    # Anthropic JSON_SCHEMA requires a response_model
    if provider == Provider.ANTHROPIC and mode == Mode.JSON_SCHEMA:
        pytest.skip("Anthropic JSON_SCHEMA mode requires a response_model")

    handlers = _get_handlers(provider, mode)
    kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
    result_model, result_kwargs = handlers.request_handler(None, kwargs)

    assert result_model is None
    assert isinstance(result_kwargs, dict)


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_prepare_request_with_model(provider: Provider, mode: Mode) -> None:
    """prepare_request should return a model and kwargs when response_model is set."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    if provider == Provider.OPENAI and mode == Mode.RESPONSES_TOOLS:
        _skip_if_missing("openai")
    if provider == Provider.MISTRAL and mode == Mode.JSON_SCHEMA:
        _skip_if_missing("mistralai")
    if mode == Mode.PARALLEL_TOOLS:
        pytest.skip("Parallel tools requires special response_model setup")

    handlers = _get_handlers(provider, mode)
    kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}]}
    result_model, result_kwargs = handlers.request_handler(Answer, kwargs)

    assert result_model is not None
    assert isinstance(result_kwargs, dict)


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_parse_response(provider: Provider, mode: Mode) -> None:
    """parse_response should return a validated model for supported scenarios."""
    scenario = PARSE_SCENARIOS.get(provider, {}).get(mode)
    if scenario is None:
        pytest.skip("No parse_response scenario defined for this provider/mode")

    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    payload = {"answer": 4.0}

    if scenario == "tool_call":
        response = builder.tool_response(payload)
    elif scenario == "text":
        response = builder.text_response(json.dumps(payload))
    elif scenario == "markdown":
        response = builder.markdown_response(json.dumps(payload))
    elif scenario == "responses_output":
        response = builder.responses_output_response(payload)
    else:
        raise ValueError(f"Unsupported scenario {scenario}")

    result = handlers.response_parser(
        response=response,
        response_model=Answer,
        validation_context=None,
        strict=None,
        stream=False,
        is_async=False,
    )

    assert isinstance(result, Answer)
    assert result.answer == 4.0


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_parse_response_validation_error(provider: Provider, mode: Mode) -> None:
    """parse_response should raise ValidationError on invalid payloads."""
    scenario = PARSE_SCENARIOS.get(provider, {}).get(mode)
    if scenario is None:
        pytest.skip("No parse_response scenario defined for this provider/mode")

    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    invalid_payload = {"wrong": "field"}

    if scenario == "tool_call":
        response = builder.tool_response(invalid_payload)
    elif scenario == "text":
        response = builder.text_response(json.dumps(invalid_payload))
    elif scenario == "markdown":
        response = builder.markdown_response(json.dumps(invalid_payload))
    elif scenario == "responses_output":
        response = builder.responses_output_response(invalid_payload)
    else:
        raise ValueError(f"Unsupported scenario {scenario}")

    with pytest.raises(ValidationError):
        handlers.response_parser(
            response=response,
            response_model=Answer,
            validation_context=None,
            strict=None,
            stream=False,
            is_async=False,
        )


@pytest.mark.parametrize("provider,mode", _provider_mode_params())
def test_handle_reask_adds_message(provider: Provider, mode: Mode) -> None:
    """handle_reask should return kwargs with messages."""
    if provider == Provider.GENAI:
        _skip_if_missing("google.genai")
    handlers = _get_handlers(provider, mode)
    builder = MockResponseBuilder(provider)
    kwargs = {"messages": [{"role": "user", "content": "Original"}]}
    response = builder.reask_response()
    exception = ValueError("Validation failed")

    result = handlers.reask_handler(
        kwargs=kwargs,
        response=response,
        exception=exception,
    )

    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) >= 1
