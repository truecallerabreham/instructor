"""Provider detection and registry utilities.

This module contains provider-related enums and detection logic.
"""

from enum import Enum

from instructor.mode import DEPRECATED_TO_CORE, Mode


class Provider(Enum):
    """Supported provider identifiers."""

    OPENAI = "openai"
    VERTEXAI = "vertexai"
    ANTHROPIC = "anthropic"
    ANYSCALE = "anyscale"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    COHERE = "cohere"
    GEMINI = "gemini"
    GENAI = "genai"
    DATABRICKS = "databricks"
    CEREBRAS = "cerebras"
    DEEPSEEK = "deepseek"
    FIREWORKS = "fireworks"
    WRITER = "writer"
    XAI = "xai"
    UNKNOWN = "unknown"
    BEDROCK = "bedrock"
    PERPLEXITY = "perplexity"
    OPENROUTER = "openrouter"


def provider_from_mode(mode: Mode, default: Provider = Provider.OPENAI) -> Provider:
    """Infer provider from a provider-specific Mode."""
    mapping = {
        Mode.ANTHROPIC_TOOLS: Provider.ANTHROPIC,
        Mode.ANTHROPIC_JSON: Provider.ANTHROPIC,
        Mode.ANTHROPIC_PARALLEL_TOOLS: Provider.ANTHROPIC,
        Mode.ANTHROPIC_REASONING_TOOLS: Provider.ANTHROPIC,
        Mode.COHERE_TOOLS: Provider.COHERE,
        Mode.COHERE_JSON_SCHEMA: Provider.COHERE,
        Mode.MISTRAL_TOOLS: Provider.MISTRAL,
        Mode.MISTRAL_STRUCTURED_OUTPUTS: Provider.MISTRAL,
        Mode.VERTEXAI_TOOLS: Provider.VERTEXAI,
        Mode.VERTEXAI_JSON: Provider.VERTEXAI,
        Mode.VERTEXAI_PARALLEL_TOOLS: Provider.VERTEXAI,
        Mode.GEMINI_TOOLS: Provider.GEMINI,
        Mode.GEMINI_JSON: Provider.GEMINI,
        Mode.GENAI_TOOLS: Provider.GENAI,
        Mode.GENAI_JSON: Provider.GENAI,
        Mode.GENAI_STRUCTURED_OUTPUTS: Provider.GENAI,
        Mode.XAI_TOOLS: Provider.XAI,
        Mode.XAI_JSON: Provider.XAI,
        Mode.CEREBRAS_TOOLS: Provider.CEREBRAS,
        Mode.CEREBRAS_JSON: Provider.CEREBRAS,
        Mode.FIREWORKS_TOOLS: Provider.FIREWORKS,
        Mode.FIREWORKS_JSON: Provider.FIREWORKS,
        Mode.WRITER_TOOLS: Provider.WRITER,
        Mode.WRITER_JSON: Provider.WRITER,
        Mode.BEDROCK_TOOLS: Provider.BEDROCK,
        Mode.BEDROCK_JSON: Provider.BEDROCK,
        Mode.PERPLEXITY_JSON: Provider.PERPLEXITY,
        Mode.OPENROUTER_STRUCTURED_OUTPUTS: Provider.OPENROUTER,
    }
    return mapping.get(mode, default)


def normalize_mode_for_provider(mode: Mode, _provider: Provider) -> Mode:
    """Apply provider-specific mode overrides before registry lookup."""
    if mode in DEPRECATED_TO_CORE:
        Mode.warn_deprecated_mode(mode)
        return DEPRECATED_TO_CORE[mode]
    return mode


def get_provider(base_url: str) -> Provider:
    """
    Detect the provider based on the base URL.

    Args:
        base_url: The base URL to analyze

    Returns:
        Provider: The detected provider enum value
    """
    if "anyscale" in str(base_url):
        return Provider.ANYSCALE
    elif "together" in str(base_url):
        return Provider.TOGETHER
    elif "anthropic" in str(base_url):
        return Provider.ANTHROPIC
    elif "cerebras" in str(base_url):
        return Provider.CEREBRAS
    elif "fireworks" in str(base_url):
        return Provider.FIREWORKS
    elif "groq" in str(base_url):
        return Provider.GROQ
    elif "openai" in str(base_url):
        return Provider.OPENAI
    elif "mistral" in str(base_url):
        return Provider.MISTRAL
    elif "cohere" in str(base_url):
        return Provider.COHERE
    elif "gemini" in str(base_url):
        return Provider.GEMINI
    elif "databricks" in str(base_url):
        return Provider.DATABRICKS
    elif "deepseek" in str(base_url):
        return Provider.DEEPSEEK
    elif "vertexai" in str(base_url):
        return Provider.VERTEXAI
    elif "writer" in str(base_url):
        return Provider.WRITER
    elif "perplexity" in str(base_url):
        return Provider.PERPLEXITY
    elif "x.ai" in str(base_url) or "xai" in str(base_url):
        return Provider.XAI
    elif "openrouter" in str(base_url):
        return Provider.OPENROUTER
    return Provider.UNKNOWN
