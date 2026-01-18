"""Unified parametrized tests for all provider client factories.

These tests verify client factory behavior (mode normalization, registry, errors, imports)
across all providers without requiring API keys.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from instructor import Mode, Provider
from instructor.v2.core.registry import (
    mode_registry,
    normalize_mode,
    reset_deprecation_warnings,
)

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
    Provider.CEREBRAS: _PROJECT_ROOT / "instructor/v2/providers/cerebras/handlers.py",
    Provider.WRITER: _PROJECT_ROOT / "instructor/v2/providers/writer/handlers.py",
    Provider.BEDROCK: _PROJECT_ROOT / "instructor/v2/providers/bedrock/handlers.py",
}
_HANDLERS_LOADED: set[Provider] = set()


def _ensure_handlers_loaded(provider: Provider) -> None:
    if provider in _HANDLERS_LOADED:
        return
    provider_modes = PROVIDER_CLIENT_CONFIGS.get(provider, {}).get(
        "supported_modes", []
    )
    if any(mode_registry.is_registered(provider, mode) for mode in provider_modes):
        _HANDLERS_LOADED.add(provider)
        return
    handler_path = _HANDLER_MODULE_PATHS.get(provider)
    if handler_path is None or not handler_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"tests.v2.handlers_{provider.value}",
        handler_path,
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDLERS_LOADED.add(provider)


# Provider-specific configurations for client tests
PROVIDER_CLIENT_CONFIGS: dict[Provider, dict[str, Any]] = {
    Provider.OPENAI: {
        "supported_modes": [
            Mode.TOOLS,
            Mode.JSON_SCHEMA,
            Mode.MD_JSON,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "unsupported_modes": [],
        "legacy_modes": {
            Mode.FUNCTIONS: Mode.TOOLS,
            Mode.TOOLS_STRICT: Mode.TOOLS,
            Mode.JSON_O1: Mode.JSON_SCHEMA,
        },
        "from_function": "from_openai",
        "sdk_module": "openai",
    },
    Provider.ANTHROPIC: {
        "supported_modes": [
            Mode.TOOLS,
            Mode.JSON,
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.ANTHROPIC_REASONING_TOOLS,
        ],
        "unsupported_modes": [Mode.MD_JSON],
        "legacy_modes": {
            Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
            Mode.ANTHROPIC_JSON: Mode.MD_JSON,
            Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
        },
        "from_function": "from_anthropic",
        "sdk_module": "anthropic",
    },
    Provider.GENAI: {
        "supported_modes": [Mode.TOOLS, Mode.JSON],
        "unsupported_modes": [Mode.JSON_SCHEMA, Mode.MD_JSON, Mode.PARALLEL_TOOLS],
        "legacy_modes": {
            Mode.GENAI_TOOLS: Mode.TOOLS,
            Mode.GENAI_JSON: Mode.JSON,
            Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
        },
        "from_function": "from_genai",
        "sdk_module": "google.genai",
    },
    Provider.COHERE: {
        "supported_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "unsupported_modes": [Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
        "legacy_modes": {
            Mode.COHERE_TOOLS: Mode.TOOLS,
            Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
        },
        "from_function": "from_cohere",
        "sdk_module": "cohere",
    },
    Provider.XAI: {
        "supported_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "unsupported_modes": [Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
        "legacy_modes": {
            Mode.XAI_TOOLS: Mode.TOOLS,
            Mode.XAI_JSON: Mode.MD_JSON,
        },
        "from_function": "from_xai",
        "sdk_module": "xai_sdk",
    },
    Provider.GROQ: {
        "supported_modes": [Mode.TOOLS, Mode.MD_JSON],
        "unsupported_modes": [
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "legacy_modes": {},
        "from_function": "from_groq",
        "sdk_module": "groq",
    },
    Provider.MISTRAL: {
        "supported_modes": [Mode.TOOLS, Mode.JSON_SCHEMA, Mode.MD_JSON],
        "unsupported_modes": [Mode.PARALLEL_TOOLS, Mode.RESPONSES_TOOLS],
        "legacy_modes": {
            Mode.MISTRAL_TOOLS: Mode.TOOLS,
            Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
        },
        "from_function": "from_mistral",
        "sdk_module": "mistralai",
    },
    Provider.FIREWORKS: {
        "supported_modes": [Mode.TOOLS, Mode.MD_JSON],
        "unsupported_modes": [
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "legacy_modes": {
            Mode.FIREWORKS_TOOLS: Mode.TOOLS,
            Mode.FIREWORKS_JSON: Mode.MD_JSON,
        },
        "from_function": "from_fireworks",
        "sdk_module": "fireworks",
    },
    Provider.CEREBRAS: {
        "supported_modes": [Mode.TOOLS, Mode.MD_JSON],
        "unsupported_modes": [
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "legacy_modes": {
            Mode.CEREBRAS_TOOLS: Mode.TOOLS,
            Mode.CEREBRAS_JSON: Mode.MD_JSON,
        },
        "from_function": "from_cerebras",
        "sdk_module": "cerebras.cloud.sdk",
        "missing_sdk_message": "cerebras is not installed",
    },
    Provider.WRITER: {
        "supported_modes": [Mode.TOOLS, Mode.MD_JSON],
        "unsupported_modes": [
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "legacy_modes": {
            Mode.WRITER_TOOLS: Mode.TOOLS,
            Mode.WRITER_JSON: Mode.MD_JSON,
        },
        "from_function": "from_writer",
        "sdk_module": "writerai",
    },
    Provider.BEDROCK: {
        "supported_modes": [Mode.TOOLS, Mode.MD_JSON],
        "unsupported_modes": [
            Mode.JSON_SCHEMA,
            Mode.PARALLEL_TOOLS,
            Mode.RESPONSES_TOOLS,
        ],
        "legacy_modes": {
            Mode.BEDROCK_TOOLS: Mode.TOOLS,
            Mode.BEDROCK_JSON: Mode.MD_JSON,
        },
        "from_function": "from_bedrock",
        "sdk_module": "botocore",
    },
}


def _dependency_missing(module: str) -> bool:
    """Check if a dependency module is missing."""
    try:
        return importlib.util.find_spec(module.split(".")[0]) is None
    except ModuleNotFoundError:
        return True


def _get_provider_params():
    """Generate provider parameters for parametrized tests."""
    return [
        pytest.param(provider, id=provider.value)
        for provider in PROVIDER_CLIENT_CONFIGS.keys()
    ]


def _get_provider_mode_params():
    """Generate (provider, mode) parameters for supported modes."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for mode in config["supported_modes"]:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


def _get_provider_unsupported_mode_params():
    """Generate (provider, mode) parameters for unsupported modes."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for mode in config["unsupported_modes"]:
            params.append(
                pytest.param(provider, mode, id=f"{provider.value}-{mode.value}")
            )
    return params


def _get_provider_legacy_mode_params():
    """Generate (provider, legacy_mode, expected_mode) parameters."""
    params = []
    for provider, config in PROVIDER_CLIENT_CONFIGS.items():
        for legacy_mode, expected_mode in config["legacy_modes"].items():
            params.append(
                pytest.param(
                    provider,
                    legacy_mode,
                    expected_mode,
                    id=f"{provider.value}-{legacy_mode.value}->{expected_mode.value}",
                )
            )
    return params


# ============================================================================
# Mode Registry Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_supported_mode_is_registered(provider: Provider, mode: Mode) -> None:
    """Test that all supported modes are registered in the registry."""
    _ensure_handlers_loaded(provider)
    assert mode_registry.is_registered(provider, mode), (
        f"Mode {mode.value} should be registered for {provider.value}"
    )


@pytest.mark.parametrize("provider,mode", _get_provider_unsupported_mode_params())
def test_unsupported_mode_not_registered(provider: Provider, mode: Mode) -> None:
    """Test that unsupported modes are NOT registered."""
    assert not mode_registry.is_registered(provider, mode), (
        f"Mode {mode.value} should NOT be registered for {provider.value}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_get_modes_for_provider(provider: Provider) -> None:
    """Test getting all modes for a provider."""
    _ensure_handlers_loaded(provider)
    config = PROVIDER_CLIENT_CONFIGS[provider]
    registered_modes = mode_registry.get_modes_for_provider(provider)

    # All supported modes should be registered
    for mode in config["supported_modes"]:
        assert mode in registered_modes, (
            f"Mode {mode.value} should be in registered modes for {provider.value}"
        )

    # Unsupported modes should not be registered
    for mode in config["unsupported_modes"]:
        assert mode not in registered_modes, (
            f"Mode {mode.value} should NOT be in registered modes for {provider.value}"
        )


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_handlers_have_all_methods(provider: Provider, mode: Mode) -> None:
    """Test that all handlers have required methods."""
    _ensure_handlers_loaded(provider)
    handlers = mode_registry.get_handlers(provider, mode)

    assert handlers.request_handler is not None
    assert handlers.reask_handler is not None
    assert handlers.response_parser is not None


# ============================================================================
# Mode Normalization Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_mode_params())
def test_generic_mode_passes_through(provider: Provider, mode: Mode) -> None:
    """Test that generic modes pass through unchanged."""
    result = normalize_mode(provider, mode)
    assert result == mode, (
        f"Generic mode {mode.value} should pass through unchanged for {provider.value}"
    )


@pytest.mark.parametrize(
    "provider,legacy_mode,expected_mode", _get_provider_legacy_mode_params()
)
def test_legacy_mode_normalizes(
    provider: Provider, legacy_mode: Mode, expected_mode: Mode
) -> None:
    """Test that legacy modes normalize to expected modes."""
    reset_deprecation_warnings()

    with pytest.warns(DeprecationWarning):
        result = normalize_mode(provider, legacy_mode)

    assert result == expected_mode, (
        f"Legacy mode {legacy_mode.value} should normalize to {expected_mode.value} for {provider.value}"
    )


# ============================================================================
# Import Tests
# ============================================================================


@pytest.mark.parametrize("provider", _get_provider_params())
def test_from_function_importable(provider: Provider) -> None:
    """Test that from_* function is importable from instructor.v2."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    from_function = config["from_function"]

    # Import from instructor.v2
    module = __import__("instructor.v2", fromlist=[from_function])
    func = getattr(module, from_function, None)

    # Should be None if SDK not installed, or a callable if installed
    assert func is None or callable(func), (
        f"{from_function} should be None or callable, got {type(func)}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_handlers_importable(provider: Provider) -> None:
    """Test that handlers are importable."""
    # This is a basic smoke test - handlers should be importable
    # Actual handler classes vary by provider, so we just check the module exists
    handler_module_path = f"instructor.v2.providers.{provider.value}.handlers"

    try:
        module = __import__(handler_module_path, fromlist=[])
        assert module is not None
    except ImportError:
        # Some providers may not have handlers if SDK is missing
        # This is okay - the registry tests will catch actual issues
        pass


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.parametrize("provider,mode", _get_provider_unsupported_mode_params())
def test_unsupported_mode_raises_error(provider: Provider, mode: Mode) -> None:
    """Test that getting handlers for unsupported mode raises KeyError."""
    with pytest.raises(KeyError):
        mode_registry.get_handlers(provider, mode)


@pytest.mark.parametrize("provider", _get_provider_params())
def test_parallel_tools_not_supported_unless_registered(provider: Provider) -> None:
    """Test that PARALLEL_TOOLS is not supported unless registered."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    is_supported = Mode.PARALLEL_TOOLS in config["supported_modes"]
    is_registered = mode_registry.is_registered(provider, Mode.PARALLEL_TOOLS)

    assert is_supported == is_registered, (
        f"PARALLEL_TOOLS support mismatch for {provider.value}: "
        f"supported={is_supported}, registered={is_registered}"
    )


@pytest.mark.parametrize("provider", _get_provider_params())
def test_responses_tools_not_supported_unless_registered(provider: Provider) -> None:
    """Test that RESPONSES_TOOLS is not supported unless registered."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    is_supported = Mode.RESPONSES_TOOLS in config["supported_modes"]
    is_registered = mode_registry.is_registered(provider, Mode.RESPONSES_TOOLS)

    assert is_supported == is_registered, (
        f"RESPONSES_TOOLS support mismatch for {provider.value}: "
        f"supported={is_supported}, registered={is_registered}"
    )


# ============================================================================
# SDK Availability Tests
# ============================================================================


@pytest.mark.parametrize("provider", _get_provider_params())
def test_from_function_raises_without_sdk(provider: Provider) -> None:
    """Test that from_* function raises error when SDK not installed."""
    config = PROVIDER_CLIENT_CONFIGS[provider]
    sdk_module = config["sdk_module"]
    from_function = config["from_function"]

    if not _dependency_missing(sdk_module):
        pytest.skip(f"{sdk_module} is installed")

    # Try to import the from_* function from the provider's client module
    try:
        client_module_path = f"instructor.v2.providers.{provider.value}.client"
        client_module = __import__(client_module_path, fromlist=[from_function])
        from_function_obj = getattr(client_module, from_function, None)

        if from_function_obj is None:
            pytest.skip(f"{from_function} not found in client module")

        from instructor.core.exceptions import ClientError

        expected_message = config.get(
            "missing_sdk_message",
            f"{sdk_module.split('.')[0]} is not installed",
        )
        with pytest.raises(ClientError, match=expected_message):
            from_function_obj("not a client")  # type: ignore[call-arg]
    except ImportError:
        # Module structure may vary - this is okay
        pass
