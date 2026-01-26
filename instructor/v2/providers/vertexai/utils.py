"""VertexAI v2 utility helpers."""

from __future__ import annotations

from typing import Any, get_origin

import jsonref
from pydantic import BaseModel
from vertexai.preview.generative_models import ToolConfig  # type: ignore[import-not-found]
import vertexai.generative_models as gm  # type: ignore[import-not-found]

from instructor.dsl.parallel import get_types_array


def vertexai_message_parser(
    message: dict[str, str | gm.Part | list[str | gm.Part]],
) -> gm.Content:
    if isinstance(message["content"], str):
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=[gm.Part.from_text(message["content"])],
        )
    if isinstance(message["content"], list):
        parts: list[gm.Part] = []
        for item in message["content"]:
            if isinstance(item, str):
                parts.append(gm.Part.from_text(item))
            elif isinstance(item, gm.Part):
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content type in list: {type(item)}")
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=parts,
        )
    raise ValueError("Unsupported message content type")


def vertexai_message_list_parser(
    messages: list[dict[str, str | gm.Part | list[str | gm.Part]]],
) -> list[gm.Content]:
    return [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]


def vertexai_function_response_parser(
    response: gm.GenerationResponse, exception: Exception
) -> gm.Content:
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                },
            )
        ]
    )


def _create_gemini_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    if get_origin(model) is not None:
        raise TypeError(f"Expected concrete model class, got type hint {model}")

    schema = model.model_json_schema()
    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema)  # type: ignore[assignment]
    gemini_schema: dict[Any, Any] = {
        "type": schema_without_refs["type"],
        "properties": schema_without_refs["properties"],
        "required": (
            schema_without_refs["required"] if "required" in schema_without_refs else []
        ),
    }
    return gemini_schema


def _create_vertexai_tool(
    models: type[BaseModel] | list[type[BaseModel]] | Any,
) -> gm.Tool:
    """Create a tool with function declarations for model(s)."""
    if get_origin(models) is not None:
        model_list = list(get_types_array(models))
    else:
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:
        parameters = _create_gemini_json_schema(model)
        declaration = gm.FunctionDeclaration(
            name=model.__name__,
            description=model.__doc__,
            parameters=parameters,
        )
        declarations.append(declaration)

    return gm.Tool(function_declarations=declarations)


def vertexai_process_response(
    call_kwargs: dict[str, Any],
    model: type[BaseModel] | list[type[BaseModel]] | Any,
):
    messages: list[dict[str, str]] = call_kwargs.pop("messages")
    contents = vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    tool = _create_vertexai_tool(models=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(
    call_kwargs: dict[str, Any], model: type[BaseModel]
):
    messages: list[dict[str, str]] = call_kwargs.pop("messages")
    contents = vertexai_message_list_parser(messages)  # type: ignore[arg-type]

    config: dict[str, Any] | None = call_kwargs.pop("generation_config", None)
    response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config
