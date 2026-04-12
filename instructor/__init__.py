from typing import Any

from ._lazy import build_lazy_exports, register_optional_exports, resolve_lazy_attr

__version__ = "1.15.1"

_LAZY_ATTRS, __all__ = build_lazy_exports(
    ("Instructor", ".core.client"),
    ("Image", ".processing.multimodal"),
    ("Audio", ".processing.multimodal"),
    ("from_openai", ".core.client"),
    ("from_litellm", ".core.client"),
    ("from_provider", ".auto_client"),
    ("AsyncInstructor", ".core.client"),
    ("Provider", ".utils.providers"),
    ("OpenAISchema", ".processing.function_calls"),
    ("CitationMixin", ".dsl"),
    ("IterableModel", ".dsl"),
    ("Maybe", ".dsl"),
    ("Partial", ".dsl"),
    ("openai_schema", ".processing.function_calls"),
    ("generate_openai_schema", ".processing.schema"),
    ("generate_anthropic_schema", ".processing.schema"),
    ("generate_gemini_schema", ".processing.schema"),
    ("Mode", ".mode"),
    ("patch", ".core.patch"),
    ("apatch", ".core.patch"),
    ("FinetuneFormat", ".distil"),
    ("Instructions", ".distil"),
    ("BatchProcessor", ".batch"),
    ("BatchRequest", ".batch"),
    ("BatchJob", ".batch"),
    ("llm_validator", ".validation"),
    ("openai_moderation", ".validation"),
    ("hooks", ".core.hooks", None),
    ("client", ".client", None),
    ("handle_response_model", ".processing.response"),
    ("handle_parallel_model", ".dsl.parallel"),
)

register_optional_exports(
    _LAZY_ATTRS,
    __all__,
    ("from_anthropic", ("anthropic",), ".providers.anthropic.client"),
    ("from_gemini", ("google", "google.generativeai"), ".providers.gemini.client"),
    ("from_fireworks", ("fireworks",), ".providers.fireworks.client"),
    ("from_cerebras", ("cerebras",), ".providers.cerebras.client"),
    ("from_groq", ("groq",), ".providers.groq.client"),
    ("from_mistral", ("mistralai",), ".providers.mistral.client"),
    ("from_cohere", ("cohere",), ".providers.cohere.client"),
    ("from_vertexai", ("vertexai", "jsonref"), ".providers.vertexai.client"),
    ("from_bedrock", ("boto3",), ".providers.bedrock.client"),
    ("from_writer", ("writerai",), ".providers.writer.client"),
    ("from_xai", ("xai_sdk",), ".providers.xai.client"),
    ("from_perplexity", ("openai",), ".providers.perplexity.client"),
    ("from_genai", ("google", "google.genai"), ".providers.genai.client"),
)


def __getattr__(name: str) -> Any:
    return resolve_lazy_attr(__name__, globals(), _LAZY_ATTRS, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
