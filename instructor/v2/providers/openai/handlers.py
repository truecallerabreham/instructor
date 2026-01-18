"""OpenAI v2 mode handlers with DSL-aware parsing.

This module implements mode handlers for OpenAI using the v2 registry system.
Supports TOOLS, JSON_SCHEMA, MD_JSON, PARALLEL_TOOLS, and RESPONSES_TOOLS modes.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Generator, Iterable as TypingIterable
from textwrap import dedent
from typing import TYPE_CHECKING, Any, get_origin
from weakref import WeakKeyDictionary

from pydantic import BaseModel

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai.types.chat import ChatCompletion

from instructor import Mode, Provider
from instructor.core.exceptions import ConfigurationError, IncompleteOutputException
from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import ParallelBase, ParallelModel, get_types_array
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase
from instructor.processing.function_calls import extract_json_from_codeblock
from instructor.processing.schema import generate_openai_schema
from instructor.providers.openai.utils import (
    reask_md_json,
    reask_responses_tools,
    reask_tools,
)
from instructor.utils.core import merge_consecutive_messages
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


class OpenAIHandlerBase(ModeHandler):
    """Base class for OpenAI handlers with shared utilities."""

    mode: Mode

    def __init__(self) -> None:
        self._streaming_models: WeakKeyDictionary[type[Any], None] = WeakKeyDictionary()

    def _register_streaming_from_kwargs(
        self, response_model: type[BaseModel] | None, kwargs: dict[str, Any]
    ) -> None:
        """Register model for streaming if stream=True in kwargs."""
        if response_model is None:
            return
        if kwargs.get("stream"):
            self.mark_streaming_model(response_model, True)

    def mark_streaming_model(
        self, response_model: type[BaseModel] | None, stream: bool
    ) -> None:
        """Record that the response model expects streaming output."""
        if not stream or response_model is None:
            return
        if inspect.isclass(response_model) and issubclass(
            response_model, (IterableBase, PartialBase)
        ):
            self._streaming_models[response_model] = None

    def _consume_streaming_flag(
        self, response_model: type[BaseModel] | ParallelBase | None
    ) -> bool:
        """Check and consume streaming flag for a model."""
        if response_model is None:
            return False
        if not inspect.isclass(response_model):
            return False
        if response_model in self._streaming_models:
            del self._streaming_models[response_model]
            return True
        return False

    def _parse_streaming_response(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        """Parse a streaming response using DSL methods."""
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        if inspect.isasyncgen(response):
            return response_model.from_streaming_response_async(  # type: ignore[attr-defined]
                response,
                mode=self.mode,
                **parse_kwargs,
            )

        generator = response_model.from_streaming_response(  # type: ignore[attr-defined]
            response,
            mode=self.mode,
            **parse_kwargs,
        )
        return list(generator)

    def _finalize_parsed_result(
        self,
        response_model: type[BaseModel] | ParallelBase,
        response: Any,
        parsed: Any,
    ) -> Any:
        """Finalize parsed result, handling DSL types."""
        if isinstance(parsed, IterableBase):
            return [task for task in parsed.tasks]
        if isinstance(response_model, ParallelBase):
            return parsed
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed

    def _extract_tool_call_json(self, response: Any) -> str:
        """Extract JSON from tool call response."""
        return response.choices[0].message.tool_calls[0].function.arguments

    def _extract_text_content(self, response: Any) -> str:
        """Extract text content from response."""
        return response.choices[0].message.content or ""


@register_mode_handler(Provider.OPENAI, Mode.TOOLS)
class OpenAIToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI TOOLS mode.

    Supports `strict=True` parameter for strict schema validation.
    """

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with tool definitions."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        # Detect if this is a parallel tools request (Iterable[Union[...]])
        origin = get_origin(response_model)
        is_parallel = origin is TypingIterable

        # Prepare response model: wrap simple types in ModelAdapter
        if not is_parallel:
            from instructor.utils.core import prepare_response_model

            response_model = prepare_response_model(response_model)

        new_kwargs = kwargs.copy()

        if is_parallel:
            # Handle parallel model
            from instructor.dsl.parallel import handle_parallel_model

            new_kwargs["tools"] = handle_parallel_model(response_model)
            new_kwargs["tool_choice"] = "auto"
        else:
            schema = generate_openai_schema(response_model)

            # Check for strict parameter
            use_strict = new_kwargs.pop("strict", False)
            if use_strict:
                schema["strict"] = True

            new_kwargs["tools"] = [{"type": "function", "function": schema}]
            new_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": schema["name"]},
            }

        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for tools mode."""
        return reask_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse tool call response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Handle parallel tools (Iterable[Union[...]])
        origin = get_origin(response_model)
        if origin is TypingIterable:
            the_types = get_types_array(response_model)  # type: ignore[arg-type]
            type_registry = {t.__name__: t for t in the_types}

            def parallel_generator() -> Generator[BaseModel, None, None]:
                for tool_call in response.choices[0].message.tool_calls:
                    name = tool_call.function.name
                    if name in type_registry:
                        model_class = type_registry[name]
                        yield model_class.model_validate_json(
                            tool_call.function.arguments,
                            context=validation_context,
                            strict=strict,
                        )

            return parallel_generator()

        # Standard tool call parsing
        json_str = self._extract_tool_call_json(response)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.OPENAI, Mode.JSON_SCHEMA)
class OpenAIJSONSchemaHandler(OpenAIHandlerBase):
    """Handler for OpenAI structured outputs (JSON_SCHEMA mode).

    Uses OpenAI's native structured outputs via response_format parameter.
    """

    mode = Mode.JSON_SCHEMA

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema response format."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        schema = response_model.model_json_schema()
        new_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
            },
        }
        return response_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for JSON schema mode."""
        return reask_md_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON schema response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        parsed = response_model.model_validate_json(
            text,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.OPENAI, Mode.MD_JSON)
class OpenAIMDJSONHandler(OpenAIHandlerBase):
    """Handler for MD_JSON mode - extract JSON from markdown code blocks."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request with JSON schema instruction in messages."""
        self._register_streaming_from_kwargs(response_model, kwargs)

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
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for MD_JSON mode."""
        return reask_md_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse JSON from markdown code block in response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        text = self._extract_text_content(response)
        json_str = extract_json_from_codeblock(text)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


@register_mode_handler(Provider.OPENAI, Mode.PARALLEL_TOOLS)
class OpenAIParallelToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI parallel tool calling."""

    mode = Mode.PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request for parallel tool calling."""
        if response_model is None:
            return None, kwargs

        new_kwargs = kwargs.copy()
        if new_kwargs.get("stream", False):
            raise ConfigurationError(
                "stream=True is not supported when using PARALLEL_TOOLS mode"
            )

        from instructor.dsl.parallel import handle_parallel_model

        new_kwargs["tools"] = handle_parallel_model(response_model)
        new_kwargs["tool_choice"] = "auto"

        # Wrap in ParallelModel for proper parsing
        return ParallelModel(typehint=response_model), new_kwargs  # type: ignore[return-value]

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: ChatCompletion,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for parallel tools mode."""
        return reask_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse parallel tool response."""
        # Check for incomplete output
        if hasattr(response, "choices") and response.choices:
            if response.choices[0].finish_reason == "length":
                raise IncompleteOutputException(last_completion=response)

        # Extract model types from response_model
        the_types = get_types_array(response_model)  # type: ignore[arg-type]
        type_registry = {t.__name__: t for t in the_types}

        results = []
        for tool_call in response.choices[0].message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            if name in type_registry:
                model = type_registry[name].model_validate_json(
                    args,
                    context=validation_context,
                    strict=strict,
                )
                results.append(model)

        return iter(results)


@register_mode_handler(Provider.OPENAI, Mode.RESPONSES_TOOLS)
class OpenAIResponsesToolsHandler(OpenAIHandlerBase):
    """Handler for OpenAI Responses API with tools."""

    mode = Mode.RESPONSES_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        """Prepare request for Responses API with tools."""
        self._register_streaming_from_kwargs(response_model, kwargs)

        new_kwargs = kwargs.copy()

        # Handle max_tokens to max_output_tokens conversion
        if new_kwargs.get("max_tokens") is not None:
            new_kwargs["max_output_tokens"] = new_kwargs.pop("max_tokens")

        if response_model is None:
            return None, new_kwargs

        from typing import cast
        from instructor.utils.core import prepare_response_model

        prepared_model = cast(type[BaseModel], prepare_response_model(response_model))

        from openai import pydantic_function_tool

        schema = pydantic_function_tool(prepared_model)
        del schema["function"]["strict"]

        tool_definition: dict[str, Any] = {
            "type": "function",
            "name": schema["function"]["name"],
            "parameters": schema["function"]["parameters"],
        }

        if "description" in schema["function"]:
            tool_definition["description"] = schema["function"]["description"]
        else:
            tool_definition["description"] = (
                f"Correctly extracted `{prepared_model.__name__}` with all "
                f"the required parameters with correct types"
            )

        new_kwargs["tools"] = [tool_definition]
        new_kwargs["tool_choice"] = {
            "type": "function",
            "name": schema["function"]["name"],
        }

        return prepared_model, new_kwargs

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        """Handle reask for Responses API."""
        return reask_responses_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        """Parse Responses API response."""
        # Check for streaming
        if isinstance(response_model, type) and self._consume_streaming_flag(
            response_model
        ):
            return self._parse_streaming_response(
                response_model,
                response,
                validation_context,
                strict,
            )

        # Handle Responses API format - output is a list of items
        if hasattr(response, "output"):
            for item in response.output:
                item_type = getattr(item, "type", None)
                if item_type in {"function_call", "tool_call"}:
                    args = getattr(item, "arguments", None)
                    if args:
                        parsed = response_model.model_validate_json(
                            args,
                            context=validation_context,
                            strict=strict,
                        )
                        return self._finalize_parsed_result(
                            response_model, response, parsed
                        )

        # Fallback to standard tool call parsing
        json_str = self._extract_tool_call_json(response)
        parsed = response_model.model_validate_json(
            json_str,
            context=validation_context,
            strict=strict,
        )
        return self._finalize_parsed_result(response_model, response, parsed)


__all__ = [
    "OpenAIToolsHandler",
    "OpenAIJSONSchemaHandler",
    "OpenAIMDJSONHandler",
    "OpenAIParallelToolsHandler",
    "OpenAIResponsesToolsHandler",
]
