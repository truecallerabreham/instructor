from __future__ import annotations

import warnings

import pytest

from instructor.mode import Mode
from instructor.utils.providers import Provider

try:
    from instructor.v2.core import (
        DEPRECATED_MODE_MAPPING,
        mode_registry,
        normalize_mode,
        reset_deprecation_warnings,
    )
except ModuleNotFoundError:
    pytest.skip("v2 module not available", allow_module_level=True)


@pytest.fixture(autouse=True)
def reset_warnings():
    """Reset deprecation warnings before each test."""
    reset_deprecation_warnings()
    yield
    reset_deprecation_warnings()


@pytest.mark.parametrize(
    "provider,mode,expected",
    [
        # GenAI legacy modes
        (Provider.GENAI, Mode.GENAI_TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.GENAI_JSON, Mode.JSON),
        (Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS, Mode.JSON),
        # Anthropic legacy modes
        (Provider.ANTHROPIC, Mode.ANTHROPIC_TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_JSON, Mode.MD_JSON),
        (Provider.ANTHROPIC, Mode.ANTHROPIC_PARALLEL_TOOLS, Mode.PARALLEL_TOOLS),
        # OpenAI legacy modes
        (Provider.OPENAI, Mode.FUNCTIONS, Mode.TOOLS),
        (Provider.OPENAI, Mode.TOOLS_STRICT, Mode.TOOLS),
        (Provider.OPENAI, Mode.JSON_O1, Mode.JSON_SCHEMA),
        # Note: Mode.JSON is NOT deprecated - it's used by GenAI as a valid mode
        (
            Provider.OPENAI,
            Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
            Mode.RESPONSES_TOOLS,
        ),
        # Mistral legacy modes
        (Provider.MISTRAL, Mode.MISTRAL_TOOLS, Mode.TOOLS),
        (Provider.MISTRAL, Mode.MISTRAL_STRUCTURED_OUTPUTS, Mode.JSON_SCHEMA),
        # Cohere legacy modes
        (Provider.COHERE, Mode.COHERE_TOOLS, Mode.TOOLS),
        (Provider.COHERE, Mode.COHERE_JSON_SCHEMA, Mode.JSON_SCHEMA),
        # xAI legacy modes
        (Provider.XAI, Mode.XAI_TOOLS, Mode.TOOLS),
        (Provider.XAI, Mode.XAI_JSON, Mode.MD_JSON),
        # Generic modes should pass through unchanged
        (Provider.GENAI, Mode.TOOLS, Mode.TOOLS),
        (Provider.GENAI, Mode.JSON, Mode.JSON),
        (Provider.ANTHROPIC, Mode.TOOLS, Mode.TOOLS),
        (Provider.ANTHROPIC, Mode.JSON, Mode.JSON),
        (Provider.OPENAI, Mode.TOOLS, Mode.TOOLS),
        (Provider.OPENAI, Mode.JSON_SCHEMA, Mode.JSON_SCHEMA),
        (Provider.OPENAI, Mode.MD_JSON, Mode.MD_JSON),
        (Provider.OPENAI, Mode.PARALLEL_TOOLS, Mode.PARALLEL_TOOLS),
        (Provider.OPENAI, Mode.RESPONSES_TOOLS, Mode.RESPONSES_TOOLS),
    ],
)
def test_normalize_mode(provider: Provider, mode: Mode, expected: Mode):
    """Test that mode normalization works correctly."""
    result = normalize_mode(provider, mode)
    assert result == expected


def test_genai_handlers_registered_with_generic_modes():
    """Test that GenAI handlers are registered with generic modes."""
    assert mode_registry.is_registered(Provider.GENAI, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.JSON)


def test_genai_backwards_compatibility():
    """Test that old provider-specific modes still work."""
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_TOOLS)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_JSON)
    assert mode_registry.is_registered(Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS)

    # Should be able to get handlers with old modes
    handler1 = mode_registry.get_handler_class(Provider.GENAI, Mode.GENAI_TOOLS)
    handler2 = mode_registry.get_handler_class(Provider.GENAI, Mode.GENAI_JSON)
    handler3 = mode_registry.get_handler_class(
        Provider.GENAI, Mode.GENAI_STRUCTURED_OUTPUTS
    )

    assert handler1 is not None
    assert handler2 is not None
    assert handler3 is not None

    # All should resolve to the same handlers as generic modes
    assert handler1 == mode_registry.get_handler_class(Provider.GENAI, Mode.TOOLS)
    assert handler2 == mode_registry.get_handler_class(Provider.GENAI, Mode.JSON)
    assert handler3 == mode_registry.get_handler_class(Provider.GENAI, Mode.JSON)


def test_openai_handlers_registered_with_generic_modes():
    """Test that OpenAI handlers are registered with generic modes."""
    assert mode_registry.is_registered(Provider.OPENAI, Mode.TOOLS)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.JSON_SCHEMA)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.MD_JSON)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.PARALLEL_TOOLS)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.RESPONSES_TOOLS)


def test_openai_backwards_compatibility():
    """Test that old OpenAI-specific modes still work."""
    # Legacy modes should be registered (via normalization)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.FUNCTIONS)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.TOOLS_STRICT)
    assert mode_registry.is_registered(Provider.OPENAI, Mode.JSON_O1)
    # Note: Mode.JSON is NOT deprecated - it's used by GenAI, not registered for OpenAI
    assert mode_registry.is_registered(
        Provider.OPENAI, Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS
    )

    # Should be able to get handlers with old modes
    handler_functions = mode_registry.get_handler_class(Provider.OPENAI, Mode.FUNCTIONS)
    handler_tools_strict = mode_registry.get_handler_class(
        Provider.OPENAI, Mode.TOOLS_STRICT
    )
    handler_json_o1 = mode_registry.get_handler_class(Provider.OPENAI, Mode.JSON_O1)

    assert handler_functions is not None
    assert handler_tools_strict is not None
    assert handler_json_o1 is not None

    # All should resolve to the same handlers as generic modes
    assert handler_functions == mode_registry.get_handler_class(
        Provider.OPENAI, Mode.TOOLS
    )
    assert handler_tools_strict == mode_registry.get_handler_class(
        Provider.OPENAI, Mode.TOOLS
    )
    assert handler_json_o1 == mode_registry.get_handler_class(
        Provider.OPENAI, Mode.JSON_SCHEMA
    )


@pytest.mark.parametrize(
    "mode,expected_replacement",
    [
        (Mode.FUNCTIONS, Mode.TOOLS),
        (Mode.TOOLS_STRICT, Mode.TOOLS),
        (Mode.JSON_O1, Mode.JSON_SCHEMA),
        # Note: Mode.JSON is NOT deprecated - it's used by GenAI as a valid mode
        (Mode.ANTHROPIC_TOOLS, Mode.TOOLS),
        (Mode.GENAI_TOOLS, Mode.TOOLS),
        (Mode.MISTRAL_TOOLS, Mode.TOOLS),
        (Mode.COHERE_TOOLS, Mode.TOOLS),
        (Mode.XAI_TOOLS, Mode.TOOLS),
    ],
)
def test_deprecated_mode_emits_warning(mode: Mode, expected_replacement: Mode):
    """Test that using deprecated modes emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize_mode(Provider.OPENAI, mode)

        # Should have emitted a warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert mode.name in str(w[0].message)
        assert expected_replacement.name in str(w[0].message)
        assert "v3.0" in str(w[0].message)

        # Should still return the correct normalized mode
        assert result == expected_replacement


def test_deprecated_mode_warning_only_once():
    """Test that deprecation warning is only shown once per mode."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # First call should warn
        normalize_mode(Provider.OPENAI, Mode.FUNCTIONS)
        assert len(w) == 1

        # Second call should not warn again
        normalize_mode(Provider.OPENAI, Mode.FUNCTIONS)
        assert len(w) == 1  # Still only 1 warning

        # Different mode should warn
        normalize_mode(Provider.OPENAI, Mode.TOOLS_STRICT)
        assert len(w) == 2


def test_generic_mode_no_warning():
    """Test that using generic modes does not emit warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        normalize_mode(Provider.OPENAI, Mode.TOOLS)
        normalize_mode(Provider.OPENAI, Mode.JSON_SCHEMA)
        normalize_mode(Provider.OPENAI, Mode.MD_JSON)
        normalize_mode(Provider.OPENAI, Mode.PARALLEL_TOOLS)
        normalize_mode(Provider.OPENAI, Mode.RESPONSES_TOOLS)

        # No warnings should be emitted for generic modes
        assert len(w) == 0


def test_deprecated_mode_mapping_is_complete():
    """Test that all provider-specific modes are in the deprecated mapping."""
    # These are the provider-specific modes that should be deprecated
    # Note: Mode.JSON is NOT deprecated - it's used by GenAI as a valid mode
    expected_deprecated = {
        # OpenAI legacy
        Mode.FUNCTIONS,
        Mode.TOOLS_STRICT,
        # Mode.JSON is NOT deprecated - it's used by GenAI
        Mode.JSON_O1,
        Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS,
        # Anthropic
        Mode.ANTHROPIC_TOOLS,
        Mode.ANTHROPIC_JSON,
        Mode.ANTHROPIC_PARALLEL_TOOLS,
        # GenAI
        Mode.GENAI_TOOLS,
        Mode.GENAI_JSON,
        Mode.GENAI_STRUCTURED_OUTPUTS,
        # Mistral
        Mode.MISTRAL_TOOLS,
        Mode.MISTRAL_STRUCTURED_OUTPUTS,
        # Cohere
        Mode.COHERE_TOOLS,
        Mode.COHERE_JSON_SCHEMA,
        # xAI
        Mode.XAI_TOOLS,
        Mode.XAI_JSON,
        # Fireworks
        Mode.FIREWORKS_TOOLS,
        Mode.FIREWORKS_JSON,
        # Cerebras
        Mode.CEREBRAS_TOOLS,
        Mode.CEREBRAS_JSON,
        # Writer
        Mode.WRITER_TOOLS,
        Mode.WRITER_JSON,
        # Bedrock
        Mode.BEDROCK_TOOLS,
        Mode.BEDROCK_JSON,
        # Perplexity
        Mode.PERPLEXITY_JSON,
        # VertexAI
        Mode.VERTEXAI_TOOLS,
        Mode.VERTEXAI_JSON,
        Mode.VERTEXAI_PARALLEL_TOOLS,
        # Gemini
        Mode.GEMINI_TOOLS,
        Mode.GEMINI_JSON,
        # OpenRouter
        Mode.OPENROUTER_STRUCTURED_OUTPUTS,
    }

    # Check that all expected deprecated modes are in the mapping
    for mode in expected_deprecated:
        assert mode in DEPRECATED_MODE_MAPPING, (
            f"Mode.{mode.name} should be in DEPRECATED_MODE_MAPPING"
        )
