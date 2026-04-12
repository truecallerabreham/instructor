"""Core components of the instructor package."""

import importlib
from typing import Any

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Instructor": (".client", "Instructor"),
    "AsyncInstructor": (".client", "AsyncInstructor"),
    "Response": (".client", "Response"),
    "from_openai": (".client", "from_openai"),
    "from_litellm": (".client", "from_litellm"),
    "InstructorRetryException": (".exceptions", "InstructorRetryException"),
    "InstructorError": (".exceptions", "InstructorError"),
    "ConfigurationError": (".exceptions", "ConfigurationError"),
    "IncompleteOutputException": (".exceptions", "IncompleteOutputException"),
    "ValidationError": (".exceptions", "ValidationError"),
    "ProviderError": (".exceptions", "ProviderError"),
    "ModeError": (".exceptions", "ModeError"),
    "ClientError": (".exceptions", "ClientError"),
    "AsyncValidationError": (".exceptions", "AsyncValidationError"),
    "FailedAttempt": (".exceptions", "FailedAttempt"),
    "ResponseParsingError": (".exceptions", "ResponseParsingError"),
    "MultimodalError": (".exceptions", "MultimodalError"),
    "Hooks": (".hooks", "Hooks"),
    "HookName": (".hooks", "HookName"),
    "patch": (".patch", "patch"),
    "apatch": (".patch", "apatch"),
    "retry_sync": (".retry", "retry_sync"),
    "retry_async": (".retry", "retry_async"),
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
