"""Unit tests for Fireworks v2 client factory.

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


class TestFireworksModeNormalization:
    """Tests for Fireworks mode normalization."""

    def test_mode_normalization_generic_tools(self):
        """Test generic TOOLS mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.FIREWORKS, Mode.TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_generic_md_json(self):
        """Test generic MD_JSON mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.FIREWORKS, Mode.MD_JSON)

        assert result == Mode.MD_JSON

    def test_mode_normalization_fireworks_tools(self):
        """Test FIREWORKS_TOOLS normalizes to TOOLS with deprecation warning."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="FIREWORKS_TOOLS is deprecated"):
            result = normalize_mode(Provider.FIREWORKS, Mode.FIREWORKS_TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_fireworks_json(self):
        """Test FIREWORKS_JSON normalizes to MD_JSON with deprecation warning."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="FIREWORKS_JSON is deprecated"):
            result = normalize_mode(Provider.FIREWORKS, Mode.FIREWORKS_JSON)

        assert result == Mode.MD_JSON


# ============================================================================
# Mode Registry Tests for Fireworks
# ============================================================================


class TestFireworksModeRegistry:
    """Tests for Fireworks mode registration in the v2 registry."""

    def test_tools_mode_registered(self):
        """Test TOOLS mode is registered for Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.FIREWORKS, Mode.TOOLS)

    def test_md_json_mode_registered(self):
        """Test MD_JSON mode is registered for Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.FIREWORKS, Mode.MD_JSON)

    def test_json_schema_not_registered(self):
        """Test JSON_SCHEMA mode is NOT registered for Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.JSON_SCHEMA)

    def test_get_modes_for_fireworks(self):
        """Test getting all modes for Fireworks provider."""
        from instructor.v2.core.registry import mode_registry

        modes = mode_registry.get_modes_for_provider(Provider.FIREWORKS)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes
        assert Mode.JSON_SCHEMA not in modes

    def test_fireworks_in_providers_for_tools(self):
        """Test Fireworks is listed as provider for TOOLS mode."""
        from instructor.v2.core.registry import mode_registry

        providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

        assert Provider.FIREWORKS in providers


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestFireworksClientErrors:
    """Tests for error handling in Fireworks client."""

    def test_json_schema_not_supported(self):
        """Test JSON_SCHEMA mode is not supported by Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS is not supported by Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.PARALLEL_TOOLS)

    def test_responses_tools_not_supported(self):
        """Test RESPONSES_TOOLS is not supported by Fireworks."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.FIREWORKS, Mode.RESPONSES_TOOLS)


# ============================================================================
# Import Tests
# ============================================================================


class TestFireworksImports:
    """Tests for Fireworks module imports."""

    def test_from_fireworks_importable_from_v2(self):
        """Test from_fireworks is importable from instructor.v2."""
        from instructor.v2 import from_fireworks

        # Should be None if fireworks not installed, or a function if installed
        assert from_fireworks is None or callable(from_fireworks)

    def test_handlers_importable(self):
        """Test Fireworks handlers are importable."""
        from instructor.v2.providers.fireworks.handlers import (
            FireworksMDJSONHandler,
            FireworksToolsHandler,
        )

        assert FireworksToolsHandler is not None
        assert FireworksMDJSONHandler is not None


# ============================================================================
# Integration Tests (require Fireworks SDK but not API key)
# ============================================================================


class TestFireworksClientWithSDK:
    """Tests that require Fireworks SDK but not API key."""

    @pytest.fixture
    def fireworks_available(self):
        """Check if fireworks SDK is available."""
        try:
            from fireworks.client import Fireworks  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_fireworks_raises_without_sdk(self, fireworks_available):
        """Test from_fireworks raises error when fireworks not installed."""
        if fireworks_available:
            pytest.skip("fireworks is installed")

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="fireworks-ai is not installed"):
            from_fireworks("not a client")  # type: ignore[arg-type]

    def test_from_fireworks_with_invalid_client(self, fireworks_available):
        """Test from_fireworks raises error with invalid client."""
        if not fireworks_available:
            pytest.skip("fireworks not installed")

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_fireworks("not a client")  # type: ignore[arg-type]

    def test_from_fireworks_with_invalid_mode(self, fireworks_available):
        """Test from_fireworks raises error with invalid mode."""
        if not fireworks_available:
            pytest.skip("fireworks not installed")

        from fireworks.client import Fireworks

        from instructor.v2.providers.fireworks.client import from_fireworks
        from instructor.core.exceptions import ModeError

        client = Fireworks(api_key="fake-key")

        with pytest.raises(ModeError):
            from_fireworks(client, mode=Mode.JSON_SCHEMA)
