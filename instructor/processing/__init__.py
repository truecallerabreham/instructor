"""Processing components for request/response handling."""

import importlib
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "OpenAISchema": (".function_calls", "OpenAISchema"),
    "openai_schema": (".function_calls", "openai_schema"),
    "convert_messages": (".multimodal", "convert_messages"),
    "handle_response_model": (".response", "handle_response_model"),
    "process_response": (".response", "process_response"),
    "process_response_async": (".response", "process_response_async"),
    "handle_reask_kwargs": (".response", "handle_reask_kwargs"),
    "generate_openai_schema": (".schema", "generate_openai_schema"),
    "generate_anthropic_schema": (".schema", "generate_anthropic_schema"),
    "generate_gemini_schema": (".schema", "generate_gemini_schema"),
    "Validator": (".validators", "Validator"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
