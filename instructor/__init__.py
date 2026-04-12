import importlib
import importlib.util
from typing import Any

__version__ = "1.15.1"

_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "Mode": (".mode", "Mode"),
    "Image": (".processing.multimodal", "Image"),
    "Audio": (".processing.multimodal", "Audio"),
    "CitationMixin": (".dsl", "CitationMixin"),
    "Maybe": (".dsl", "Maybe"),
    "Partial": (".dsl", "Partial"),
    "IterableModel": (".dsl", "IterableModel"),
    "llm_validator": (".validation", "llm_validator"),
    "openai_moderation": (".validation", "openai_moderation"),
    "OpenAISchema": (".processing.function_calls", "OpenAISchema"),
    "openai_schema": (".processing.function_calls", "openai_schema"),
    "generate_openai_schema": (".processing.schema", "generate_openai_schema"),
    "generate_anthropic_schema": (
        ".processing.schema",
        "generate_anthropic_schema",
    ),
    "generate_gemini_schema": (".processing.schema", "generate_gemini_schema"),
    "apatch": (".core.patch", "apatch"),
    "patch": (".core.patch", "patch"),
    "Instructor": (".core.client", "Instructor"),
    "AsyncInstructor": (".core.client", "AsyncInstructor"),
    "from_openai": (".core.client", "from_openai"),
    "from_litellm": (".core.client", "from_litellm"),
    "hooks": (".core.hooks", None),
    "Provider": (".utils.providers", "Provider"),
    "from_provider": (".auto_client", "from_provider"),
    "BatchProcessor": (".batch", "BatchProcessor"),
    "BatchRequest": (".batch", "BatchRequest"),
    "BatchJob": (".batch", "BatchJob"),
    "FinetuneFormat": (".distil", "FinetuneFormat"),
    "Instructions": (".distil", "Instructions"),
    "handle_response_model": (".processing.response", "handle_response_model"),
    "handle_parallel_model": (".dsl.parallel", "handle_parallel_model"),
    "client": (".client", None),
}

__all__ = [
    "Instructor",
    "Image",
    "Audio",
    "from_openai",
    "from_litellm",
    "from_provider",
    "AsyncInstructor",
    "Provider",
    "OpenAISchema",
    "CitationMixin",
    "IterableModel",
    "Maybe",
    "Partial",
    "openai_schema",
    "generate_openai_schema",
    "generate_anthropic_schema",
    "generate_gemini_schema",
    "Mode",
    "patch",
    "apatch",
    "FinetuneFormat",
    "Instructions",
    "BatchProcessor",
    "BatchRequest",
    "BatchJob",
    "llm_validator",
    "openai_moderation",
    "hooks",
    "client",
    "handle_response_model",
    "handle_parallel_model",
]


def _register_optional_attr(
    export_name: str,
    module_name: str,
    attr_name: str,
) -> None:
    _LAZY_ATTRS[export_name] = (module_name, attr_name)
    __all__.append(export_name)


if importlib.util.find_spec("anthropic") is not None:
    _register_optional_attr(
        "from_anthropic", ".providers.anthropic.client", "from_anthropic"
    )

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.generativeai") is not None
):
    _register_optional_attr("from_gemini", ".providers.gemini.client", "from_gemini")

if importlib.util.find_spec("fireworks") is not None:
    _register_optional_attr(
        "from_fireworks", ".providers.fireworks.client", "from_fireworks"
    )

if importlib.util.find_spec("cerebras") is not None:
    _register_optional_attr(
        "from_cerebras", ".providers.cerebras.client", "from_cerebras"
    )

if importlib.util.find_spec("groq") is not None:
    _register_optional_attr("from_groq", ".providers.groq.client", "from_groq")

if importlib.util.find_spec("mistralai") is not None:
    _register_optional_attr("from_mistral", ".providers.mistral.client", "from_mistral")

if importlib.util.find_spec("cohere") is not None:
    _register_optional_attr("from_cohere", ".providers.cohere.client", "from_cohere")

if all(importlib.util.find_spec(pkg) for pkg in ("vertexai", "jsonref")):
    _register_optional_attr(
        "from_vertexai", ".providers.vertexai.client", "from_vertexai"
    )

if importlib.util.find_spec("boto3") is not None:
    _register_optional_attr("from_bedrock", ".providers.bedrock.client", "from_bedrock")

if importlib.util.find_spec("writerai") is not None:
    _register_optional_attr("from_writer", ".providers.writer.client", "from_writer")

if importlib.util.find_spec("xai_sdk") is not None:
    _register_optional_attr("from_xai", ".providers.xai.client", "from_xai")

if importlib.util.find_spec("openai") is not None:
    _register_optional_attr(
        "from_perplexity", ".providers.perplexity.client", "from_perplexity"
    )

if (
    importlib.util.find_spec("google")
    and importlib.util.find_spec("google.genai") is not None
):
    _register_optional_attr("from_genai", ".providers.genai.client", "from_genai")


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name, __name__)
    value: Any

    if attr_name is None:
        value = module
    else:
        value = getattr(module, attr_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
