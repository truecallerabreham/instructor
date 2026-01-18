"""Unit tests for Cerebras v2 client factory.

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


class TestCerebrasModeNormalization:
    """Tests for Cerebras mode normalization."""

    def test_mode_normalization_generic_tools(self):
        """Test generic TOOLS mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.CEREBRAS, Mode.TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_generic_md_json(self):
        """Test generic MD_JSON mode passes through."""
        from instructor.v2.core.registry import normalize_mode

        result = normalize_mode(Provider.CEREBRAS, Mode.MD_JSON)

        assert result == Mode.MD_JSON

    def test_mode_normalization_cerebras_tools(self):
        """Test CEREBRAS_TOOLS normalizes to TOOLS with deprecation warning."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="CEREBRAS_TOOLS is deprecated"):
            result = normalize_mode(Provider.CEREBRAS, Mode.CEREBRAS_TOOLS)

        assert result == Mode.TOOLS

    def test_mode_normalization_cerebras_json(self):
        """Test CEREBRAS_JSON normalizes to MD_JSON with deprecation warning."""
        from instructor.v2.core.registry import (
            normalize_mode,
            reset_deprecation_warnings,
        )

        reset_deprecation_warnings()
        with pytest.warns(DeprecationWarning, match="CEREBRAS_JSON is deprecated"):
            result = normalize_mode(Provider.CEREBRAS, Mode.CEREBRAS_JSON)

        assert result == Mode.MD_JSON


# ============================================================================
# Mode Registry Tests for Cerebras
# ============================================================================


class TestCerebrasModeRegistry:
    """Tests for Cerebras mode registration in the v2 registry."""

    def test_tools_mode_registered(self):
        """Test TOOLS mode is registered for Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.CEREBRAS, Mode.TOOLS)

    def test_md_json_mode_registered(self):
        """Test MD_JSON mode is registered for Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert mode_registry.is_registered(Provider.CEREBRAS, Mode.MD_JSON)

    def test_json_schema_not_registered(self):
        """Test JSON_SCHEMA mode is NOT registered for Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.CEREBRAS, Mode.JSON_SCHEMA)

    def test_get_modes_for_cerebras(self):
        """Test getting all modes for Cerebras provider."""
        from instructor.v2.core.registry import mode_registry

        modes = mode_registry.get_modes_for_provider(Provider.CEREBRAS)

        assert Mode.TOOLS in modes
        assert Mode.MD_JSON in modes
        assert Mode.JSON_SCHEMA not in modes

    def test_cerebras_in_providers_for_tools(self):
        """Test Cerebras is listed as provider for TOOLS mode."""
        from instructor.v2.core.registry import mode_registry

        providers = mode_registry.get_providers_for_mode(Mode.TOOLS)

        assert Provider.CEREBRAS in providers


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCerebrasClientErrors:
    """Tests for error handling in Cerebras client."""

    def test_json_schema_not_supported(self):
        """Test JSON_SCHEMA mode is not supported by Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.CEREBRAS, Mode.JSON_SCHEMA)

    def test_parallel_tools_not_supported(self):
        """Test PARALLEL_TOOLS is not supported by Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.CEREBRAS, Mode.PARALLEL_TOOLS)

    def test_responses_tools_not_supported(self):
        """Test RESPONSES_TOOLS is not supported by Cerebras."""
        from instructor.v2.core.registry import mode_registry

        assert not mode_registry.is_registered(Provider.CEREBRAS, Mode.RESPONSES_TOOLS)


# ============================================================================
# Import Tests
# ============================================================================


class TestCerebrasImports:
    """Tests for Cerebras module imports."""

    def test_from_cerebras_importable_from_v2(self):
        """Test from_cerebras is importable from instructor.v2."""
        from instructor.v2 import from_cerebras

        # Should be None if cerebras not installed, or a function if installed
        assert from_cerebras is None or callable(from_cerebras)

    def test_handlers_importable(self):
        """Test Cerebras handlers are importable."""
        from instructor.v2.providers.cerebras.handlers import (
            CerebrasMDJSONHandler,
            CerebrasToolsHandler,
        )

        assert CerebrasToolsHandler is not None
        assert CerebrasMDJSONHandler is not None


# ============================================================================
# Integration Tests (require Cerebras SDK but not API key)
# ============================================================================


class TestCerebrasClientWithSDK:
    """Tests that require Cerebras SDK but not API key."""

    @pytest.fixture
    def cerebras_available(self):
        """Check if cerebras SDK is available."""
        try:
            from cerebras.cloud.sdk import Cerebras  # noqa: F401

            return True
        except ImportError:
            return False

    def test_from_cerebras_raises_without_sdk(self, cerebras_available):
        """Test from_cerebras raises error when cerebras not installed."""
        if cerebras_available:
            pytest.skip("cerebras is installed")

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="cerebras is not installed"):
            from_cerebras("not a client")  # type: ignore[arg-type]

    def test_from_cerebras_with_invalid_client(self, cerebras_available):
        """Test from_cerebras raises error with invalid client."""
        if not cerebras_available:
            pytest.skip("cerebras not installed")

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ClientError

        with pytest.raises(ClientError, match="must be an instance"):
            from_cerebras("not a client")  # type: ignore[arg-type]

    def test_from_cerebras_with_invalid_mode(self, cerebras_available):
        """Test from_cerebras raises error with invalid mode."""
        if not cerebras_available:
            pytest.skip("cerebras not installed")

        from cerebras.cloud.sdk import Cerebras

        from instructor.v2.providers.cerebras.client import from_cerebras
        from instructor.core.exceptions import ModeError

        client = Cerebras(api_key="fake-key")

        with pytest.raises(ModeError):
            from_cerebras(client, mode=Mode.JSON_SCHEMA)
