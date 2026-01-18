"""Unit tests for Groq v2 client factory.

These tests verify client factory behavior without requiring API keys.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from instructor import Mode, Provider


class Answer(BaseModel):
    """Simple answer model for testing."""

    answer: float


# ============================================================================
# Mode Normalization Tests
# ============================================================================


class TestGroqModeNormalization:
    """Tests for Groq mode normalization."""

    def test_mode_normalization_generic_tools(self):
        """Test generic TOOLS mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.GROQ, Mode.TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_generic_md_json(self):
        """Test generic MD_JSON mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.GROQ, Mode.MD_JSON)

        assert result == Mode.MD_JSON


# ============================================================================
# Mode Registry Tests for Groq
# ============================================================================


class TestGroqModeRegistry:
    """Tests for Groq mode registration in the v2 registry."""

    def test_tools_mode_registered(self):
        """Test TOOLS mode is registered for Groq."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.GROQ, Mode.TOOLS)

    def test_md_json_mode_registered(self):
        """Test MD_JSON mode is registered for Groq."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.GROQ, Mode.MD_JSON)

    def test_json_schema_not_registered(self):
        """Test JSON_SCHEMA mode is NOT registered for Groq."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.GROQ, Mode.JSON_SCHEMA)

    def test_get_modes_for_groq(self):
        """Test getting all modes for Groq provider."""
        from instructor.v2.core.registry import mode_registry

        modes = mode_registry.get_modes_for_provider(Provider.GROQ)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes
        assert Mode.JSON_SCHEMA not in modes

    def test_groq_in_providers_for_tools(self):
        """Test Groq is listed as provider for TOOLS mode."""
        from instructor.v2.core.registry import mode_registry

        providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

        assert Provider.GROQ in providers


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestGroqClientErrors:
    """Tests for error handling in Groq client."""

    def test_json_schema_not_supported(self):
        """Test JSON_SCHEMA mode is not supported by Groq."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.GROQ, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS is not supported by Groq."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.GROQ, Mode.PARALLEL_TOOLS)

    def test_responses_tools_not_supported(self):
        """Test RESPONSES_TOOLS is not supported by Groq."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.GROQ, Mode.RESPONSES_TOOLS)


# ============================================================================
# Import Tests
# ============================================================================


class TestGroqImports:
    """Tests for Groq module imports."""

    def test_from_groq_importable_from_v2(self):
        """Test from_groq is importable from instructor.v2."""
        from instructor.v2 import from_groq

        # Should be None if groq not installed, or a function if installed
        assert from_groq is None or callable(from_groq)

    def test_handlers_importable(self):
        """Test Groq handlers are importable."""
        from instructor.v2.providers.groq.handlers import (
            GroqMDJSONHandler,
            GroqToolsHandler,
        )

        assert GroqToolsHandler is not None
        assert GroqMDJSONHandler is not None


# ============================================================================
# Integration Tests (require Groq SDK but not API key)
# ============================================================================


class TestGroqClientWithSDK:
    """Tests that require Groq SDK but not API key."""

    @pytest.fixture
    def groq_available(self):
        """Check if groq SDK is available."""
        try:
            import groq  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_groq_raises_without_sdk(self, groq_available):
        """Test from_groq raises error when groq not installed."""
        if groq_available:
            pytest.skip("groq is installed")

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="groq is not installed"):
            from_groq("not a client")  # type: ignore[arg-type]

    def test_from_groq_with_invalid_client(self, groq_available):
        """Test from_groq raises error with invalid client."""
        if not groq_available:
            pytest.skip("groq not installed")

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_groq("not a client")  # type: ignore[arg-type]

    def test_from_groq_with_invalid_mode(self, groq_available):
        """Test from_groq raises error with invalid mode."""
        if not groq_available:
            pytest.skip("groq not installed")

        import groq

        from instructor.v2.providers.groq.client import from_groq
        from instructor.core.exceptions import ModeError

        client = groq.Groq(api_key="fake-key")

        with pytest.raises(ModeError):
            from_groq(client, mode=Mode.JSON_SCHEMA)
