"""VertexAI v2 mode handlers."""

from __future__ import annotations

import inspect
import json
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Generator,
    Iterable as TypingIterable,
)
from typing import Any

from pydantic import BaseModel

from instructor.mode import Mode
from instructor.utils.providers import Provider
from instructor.dsl.iterable import IterableBase
from instructor.dsl.parallel import ParallelBase
from instructor.dsl.partial import PartialBase
from instructor.dsl.simple_type import AdapterBase
from instructor.v2.providers.gemini.utils import (
    handle_vertexai_json,
    handle_vertexai_parallel_tools,
    handle_vertexai_tools,
    reask_vertexai_json,
    reask_vertexai_tools,
)
from instructor.v2.core.decorators import register_mode_handler
from instructor.v2.core.handler import ModeHandler


class VertexAIHandlerBase(ModeHandler):
    """Base handler for VertexAI modes."""

    mode: Mode

    def extract_streaming_json(
        self, completion: TypingIterable[Any]
    ) -> Generator[str, None, None]:
        """Extract JSON chunks from VertexAI streaming responses."""
        for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield chunk.candidates[0].content.parts[0].text
            except AttributeError:
                continue

    async def extract_streaming_json_async(
        self, completion: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Extract JSON chunks from VertexAI async streams."""
        async for chunk in completion:
            try:
                if self.mode == Mode.TOOLS:
                    yield json.dumps(
                        chunk.candidates[0].content.parts[0].function_call.args
                    )
                else:
                    yield chunk.candidates[0].content.parts[0].text
            except AttributeError:
                continue

    def _parse_streaming(
        self,
        response_model: type[BaseModel],
        response: Any,
        validation_context: dict[str, Any] | None,
        strict: bool | None,
    ) -> Any:
        parse_kwargs: dict[str, Any] = {}
        if validation_context is not None:
            parse_kwargs["context"] = validation_context
        if strict is not None:
            parse_kwargs["strict"] = strict

        task_parser = None
        if (
            self.mode == Mode.TOOLS
            and inspect.isclass(response_model)
            and issubclass(response_model, IterableBase)
        ):
            task_parser = response_model.tasks_from_task_list_chunks  # type: ignore[attr-defined]

        if inspect.isasyncgen(response) or isinstance(response, AsyncIterator):
            return response_model.from_streaming_response_async(  # type: ignore[attr-defined]
                response,
                stream_extractor=self.extract_streaming_json_async,
                task_parser=(
                    response_model.tasks_from_task_list_chunks_async  # type: ignore[attr-defined]
                    if task_parser is not None
                    else None
                ),
                **parse_kwargs,
            )

        generator = response_model.from_streaming_response(  # type: ignore[attr-defined]
            response,
            stream_extractor=self.extract_streaming_json,
            task_parser=task_parser,
            **parse_kwargs,
        )
        if inspect.isclass(response_model) and issubclass(response_model, IterableBase):
            return generator
        if inspect.isclass(response_model) and issubclass(response_model, PartialBase):
            return list(generator)
        return list(generator)

    def _finalize(
        self,
        response_model: type[BaseModel] | ParallelBase,  # noqa: ARG002
        response: Any,
        parsed: Any,  # noqa: ARG002
    ) -> Any:
        if isinstance(parsed, AdapterBase):
            return parsed.content
        if isinstance(parsed, BaseModel):
            parsed._raw_response = response  # type: ignore[attr-defined]
        return parsed


@register_mode_handler(Provider.VERTEXAI, Mode.TOOLS)
class VertexAIToolsHandler(VertexAIHandlerBase):
    """Handler for VertexAI TOOLS mode."""

    mode = Mode.TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_vertexai_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        if isinstance(response_model, ParallelBase):
            return response_model.from_response(  # type: ignore[attr-defined]
                response,
                mode=Mode.VERTEXAI_PARALLEL_TOOLS,
                validation_context=validation_context,
                strict=strict,
            )
        parsed = response_model.parse_vertexai_tools(  # type: ignore[attr-defined]
            response, validation_context
        )
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.VERTEXAI, Mode.MD_JSON)
class VertexAIJSONHandler(VertexAIHandlerBase):
    """Handler for VertexAI JSON mode."""

    mode = Mode.MD_JSON

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        return handle_vertexai_json(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_json(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel],
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if (
            stream
            and inspect.isclass(response_model)
            and issubclass(response_model, (IterableBase, PartialBase))
        ):
            return self._parse_streaming(
                response_model, response, validation_context, strict
            )
        parsed = response_model.parse_vertexai_json(  # type: ignore[attr-defined]
            response, validation_context, strict
        )
        return self._finalize(response_model, response, parsed)


@register_mode_handler(Provider.VERTEXAI, Mode.PARALLEL_TOOLS)
class VertexAIParallelToolsHandler(VertexAIHandlerBase):
    """Handler for VertexAI parallel tools mode."""

    mode = Mode.PARALLEL_TOOLS

    def prepare_request(
        self,
        response_model: type[BaseModel] | None,
        kwargs: dict[str, Any],
    ) -> tuple[type[BaseModel] | None, dict[str, Any]]:
        new_kwargs = kwargs.copy()
        if response_model is None:
            return None, new_kwargs
        return handle_vertexai_parallel_tools(response_model, new_kwargs)

    def handle_reask(
        self,
        kwargs: dict[str, Any],
        response: Any,
        exception: Exception,
    ) -> dict[str, Any]:
        return reask_vertexai_tools(kwargs, response, exception)

    def parse_response(
        self,
        response: Any,
        response_model: type[BaseModel] | ParallelBase,
        validation_context: dict[str, Any] | None = None,
        strict: bool | None = None,
        stream: bool = False,  # noqa: ARG002
        is_async: bool = False,  # noqa: ARG002
    ) -> Any:
        if isinstance(response_model, ParallelBase):
            return response_model.from_response(  # type: ignore[attr-defined]
                response,
                mode=Mode.VERTEXAI_PARALLEL_TOOLS,
                validation_context=validation_context,
                strict=strict,
            )
        parsed = response_model.parse_vertexai_tools(  # type: ignore[attr-defined]
            response, validation_context
        )
        return self._finalize(response_model, response, parsed)


__all__ = [
    "VertexAIToolsHandler",
    "VertexAIJSONHandler",
    "VertexAIParallelToolsHandler",
]
