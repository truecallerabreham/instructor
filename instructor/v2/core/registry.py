"""Mode handler registry for v2.

Central registry mapping Mode enum values to their handler implementations.
Supports lazy loading, dynamic registration, and queryable API.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable

from instructor import Mode, Provider
from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser

# Track which deprecation warnings have been shown (to avoid spam)
_deprecated_modes_warned: set[Mode] = set()


# Mapping of deprecated modes to their core mode replacements
# This is the authoritative list of deprecated modes
#
# NOTE: Mode.JSON is NOT deprecated because it's used by GenAI as a valid mode.
# The migration plan suggested deprecating it, but GenAI already uses it.
DEPRECATED_MODE_MAPPING: dict[Mode, Mode] = {
    # OpenAI legacy modes -> core modes
    Mode.FUNCTIONS: Mode.TOOLS,
    Mode.TOOLS_STRICT: Mode.TOOLS,  # Now a parameter: strict=True
    # Mode.JSON is NOT deprecated - it's used by GenAI
    Mode.JSON_O1: Mode.JSON_SCHEMA,  # O1 handled by provider logic
    Mode.RESPONSES_TOOLS_WITH_INBUILT_TOOLS: Mode.RESPONSES_TOOLS,
    # Anthropic -> core modes
    Mode.ANTHROPIC_TOOLS: Mode.TOOLS,
    Mode.ANTHROPIC_JSON: Mode.MD_JSON,
    Mode.ANTHROPIC_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    # GenAI modes -> core modes
    Mode.GENAI_TOOLS: Mode.TOOLS,
    Mode.GENAI_JSON: Mode.JSON,
    Mode.GENAI_STRUCTURED_OUTPUTS: Mode.JSON,
    # Mistral -> core modes
    Mode.MISTRAL_TOOLS: Mode.TOOLS,
    Mode.MISTRAL_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
    # Cohere -> core modes
    Mode.COHERE_TOOLS: Mode.TOOLS,
    Mode.COHERE_JSON_SCHEMA: Mode.JSON_SCHEMA,
    # xAI -> core modes
    Mode.XAI_TOOLS: Mode.TOOLS,
    Mode.XAI_JSON: Mode.MD_JSON,
    # Groq -> core modes (OpenAI-compatible)
    # Groq doesn't have provider-specific modes, uses generic TOOLS/MD_JSON
    # Fireworks -> core modes
    Mode.FIREWORKS_TOOLS: Mode.TOOLS,
    Mode.FIREWORKS_JSON: Mode.MD_JSON,
    # Cerebras -> core modes
    Mode.CEREBRAS_TOOLS: Mode.TOOLS,
    Mode.CEREBRAS_JSON: Mode.MD_JSON,
    # Writer -> core modes
    Mode.WRITER_TOOLS: Mode.TOOLS,
    Mode.WRITER_JSON: Mode.MD_JSON,
    # Bedrock -> core modes
    Mode.BEDROCK_TOOLS: Mode.TOOLS,
    Mode.BEDROCK_JSON: Mode.MD_JSON,
    # Perplexity -> core modes
    Mode.PERPLEXITY_JSON: Mode.MD_JSON,
    # VertexAI -> core modes
    Mode.VERTEXAI_TOOLS: Mode.TOOLS,
    Mode.VERTEXAI_JSON: Mode.MD_JSON,
    Mode.VERTEXAI_PARALLEL_TOOLS: Mode.PARALLEL_TOOLS,
    # Gemini -> core modes
    Mode.GEMINI_TOOLS: Mode.TOOLS,
    Mode.GEMINI_JSON: Mode.MD_JSON,
    # OpenRouter -> core modes
    Mode.OPENROUTER_STRUCTURED_OUTPUTS: Mode.JSON_SCHEMA,
}


def _warn_deprecated_mode(mode: Mode, replacement: Mode) -> None:
    """Emit a deprecation warning for a legacy mode.

    Only warns once per mode to avoid spamming logs.

    Args:
        mode: The deprecated mode being used
        replacement: The core mode it maps to
    """
    if mode in _deprecated_modes_warned:
        return

    _deprecated_modes_warned.add(mode)

    warnings.warn(
        f"Mode.{mode.name} is deprecated and will be removed in v3.0. "
        f"Use Mode.{replacement.name} instead. "
        f"The provider is determined by the client (from_openai, from_anthropic, etc.), "
        f"not by the mode.",
        DeprecationWarning,
        stacklevel=4,  # Adjust to show caller's location
    )


def normalize_mode(_provider: Provider, mode: Mode) -> Mode:
    """Convert provider-specific modes to generic modes.

    This allows backward compatibility - users can still use provider-specific
    modes like Mode.ANTHROPIC_TOOLS, and they'll be converted to generic modes
    like Mode.TOOLS for registry lookup.

    Emits a DeprecationWarning when a legacy mode is used.

    Args:
        provider: Provider enum value (for context, though mode mapping is provider-agnostic)
        mode: Mode enum value (may be provider-specific)

    Returns:
        Generic mode enum value
    """
    # Check if this is a deprecated mode
    if mode in DEPRECATED_MODE_MAPPING:
        replacement = DEPRECATED_MODE_MAPPING[mode]
        _warn_deprecated_mode(mode, replacement)
        return replacement

    # Add ANTHROPIC_STRUCTURED_OUTPUTS if it exists in the Mode enum
    if hasattr(Mode, "ANTHROPIC_STRUCTURED_OUTPUTS"):
        if mode == Mode.ANTHROPIC_STRUCTURED_OUTPUTS:
            _warn_deprecated_mode(mode, Mode.JSON_SCHEMA)
            return Mode.JSON_SCHEMA

    # Return as-is for core modes
    return mode


def reset_deprecation_warnings() -> None:
    """Reset the deprecation warning tracker.

    Useful for testing to ensure warnings are shown again.
    """
    global _deprecated_modes_warned
    _deprecated_modes_warned = set()


@dataclass
class ModeHandlers:
    """Collection of handlers for a specific mode."""

    request_handler: RequestHandler
    reask_handler: ReaskHandler
    response_parser: ResponseParser


class ModeRegistry:
    """Central registry for mode handlers.

    Maps (Provider, Mode) tuples to their handler implementations.
    Supports lazy loading and dynamic registration.

    Example:
        >>> registry.register(
        ...     provider=Provider.ANTHROPIC,
        ...     mode=Mode.TOOLS,
        ...     request_handler=handle_request,
        ...     reask_handler=handle_reask,
        ...     response_parser=parse_response,
        ... )
        >>> # Preferred: get all handlers at once
        >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
        >>> handlers.request_handler(...)
        >>> handlers.reask_handler(...)
        >>> handlers.response_parser(...)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._handlers: dict[Mode, ModeHandlers] = {}
        self._lazy_loaders: dict[Mode, Callable[[], ModeHandlers]] = {}

    def register(
        self,
        provider: Provider,
        mode: Mode,
        request_handler: RequestHandler,
        reask_handler: ReaskHandler,
        response_parser: ResponseParser,
    ) -> None:
        """Register handlers for a mode.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            request_handler: Handler to prepare request kwargs
            reask_handler: Handler to handle validation failures
            response_parser: Handler to parse responses

        Raises:
            ConfigurationError: If mode is already registered
        """
        from instructor.core.exceptions import ConfigurationError

        mode_key = (provider, mode)
        if mode_key in self._handlers:
            raise ConfigurationError(f"Mode {mode_key} is already registered")

        self._handlers[mode_key] = ModeHandlers(
            request_handler=request_handler,
            reask_handler=reask_handler,
            response_parser=response_parser,
        )

    def register_lazy(
        self,
        provider: Provider,
        mode: Mode,
        loader: Callable[[], ModeHandlers],
    ) -> None:
        """Register a lazy loader for a mode.

        The loader will be called on first access to load handlers.

        Args:
            provider: Provider enum value
            mode: Mode enum value
            loader: Callable that returns ModeHandlers when invoked

        Raises:
            ConfigurationError: If mode is already registered
        """
        from instructor.core.exceptions import ConfigurationError

        mode_key = (provider, mode)
        if mode_key in self._handlers or mode_key in self._lazy_loaders:
            raise ConfigurationError(f"Mode {mode_key} is already registered")

        self._lazy_loaders[mode_key] = loader

    def get_handlers(self, provider: Provider, mode: Mode) -> ModeHandlers:
        """Get all handlers for a mode.

        This is the preferred method for retrieving handlers. It performs
        a single registry lookup and returns all handlers at once, which is
        more efficient than calling get_handler() multiple times.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            ModeHandlers with all handler functions (request_handler,
            reask_handler, response_parser)

        Raises:
            KeyError: If mode is not registered

        Example:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)
            >>> handlers.response_parser(...)
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)

        # Check if already loaded
        if mode_key in self._handlers:
            return self._handlers[mode_key]

        # Try lazy loading
        if mode_key in self._lazy_loaders:
            loader = self._lazy_loaders.pop(mode_key)
            handlers = loader()
            self._handlers[mode_key] = handlers
            return handlers

        from instructor.core.exceptions import ConfigurationError

        raise ConfigurationError(
            f"Mode {mode_key} is not registered. "
            f"Available modes: {list(self._handlers.keys())}"
        )

    def get_handler(
        self,
        provider: Provider,
        mode: Mode,
        handler_type: str,
    ) -> RequestHandler | ReaskHandler | ResponseParser:
        """Get a specific handler for a mode.

        This is a convenience method that internally calls get_handlers().
        For better performance when you need multiple handlers, use
        get_handlers() instead and access handlers via the returned object.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)
            handler_type: One of 'request', 'reask', 'response'

        Returns:
            The requested handler function

        Raises:
            KeyError: If mode is not registered
            ValueError: If handler_type is invalid

        Example:
            >>> # Prefer this when you need multiple handlers:
            >>> handlers = registry.get_handlers(Provider.ANTHROPIC, Mode.TOOLS)
            >>> handlers.request_handler(...)
            >>> handlers.reask_handler(...)

            >>> # Or use this convenience method for a single handler:
            >>> handler = registry.get_handler(Provider.ANTHROPIC, Mode.TOOLS, "request")
        """
        # get_handlers already handles normalization
        handlers = self.get_handlers(provider, mode)

        if handler_type == "request":
            return handlers.request_handler
        elif handler_type == "reask":
            return handlers.reask_handler
        elif handler_type == "response":
            return handlers.response_parser
        else:
            from instructor.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid handler_type: {handler_type}. "
                f"Must be 'request', 'reask', or 'response'"
            )

    def is_registered(self, provider: Provider, mode: Mode) -> bool:
        """Check if a mode is registered.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            True if mode is registered (eagerly or lazily)
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)
        return mode_key in self._handlers or mode_key in self._lazy_loaders

    def get_modes_for_provider(self, provider: Provider) -> list[Mode]:
        """Get all registered modes for a provider.

        Args:
            provider: Provider enum value

        Returns:
            List of Mode values supported by this provider
        """
        modes = []
        for p, mt in self._handlers.keys():
            if p == provider:
                modes.append(mt)
        for p, mt in self._lazy_loaders.keys():
            if p == provider:
                modes.append(mt)
        return sorted(set(modes), key=lambda m: m.value)

    def get_providers_for_mode(self, mode: Mode) -> list[Provider]:
        """Get all providers that support a mode.

        Args:
            mode: Mode enum value

        Returns:
            List of Provider values that support this mode
        """
        providers = []
        for p, mt in self._handlers.keys():
            if mt == mode:
                providers.append(p)
        for p, mt in self._lazy_loaders.keys():
            if mt == mode:
                providers.append(p)
        return sorted(set(providers), key=lambda p: p.value)

    def list_modes(self) -> list[Mode]:
        """List all registered modes.

        Returns:
            List of (Provider, Mode) tuples
        """
        all_modes = set(self._handlers.keys()) | set(self._lazy_loaders.keys())
        return sorted(all_modes, key=lambda m: (m[0].value, m[1].value))

    def get_handler_class(self, provider: Provider, mode: Mode) -> type | None:
        """Get the handler class for a mode.

        This method looks up the handler class that was registered for the given
        provider and mode. It's useful for testing and introspection.

        Args:
            provider: Provider enum value
            mode: Mode enum value (provider-specific modes will be converted)

        Returns:
            Handler class type if found, None otherwise
        """
        # Convert provider-specific modes to generic modes
        normalized_mode = normalize_mode(provider, mode)
        mode_key = (provider, normalized_mode)

        # Check if handlers are registered
        if mode_key not in self._handlers and mode_key not in self._lazy_loaders:
            return None

        # Try to get the handler class from the decorator registry
        # The decorator stores the handler class, but we need to find it
        # by looking at what was registered
        handlers = self.get_handlers(provider, mode)

        # The handlers are bound methods, so we need to get the class
        # We can inspect the handler's __self__ to get the instance, then __class__
        if hasattr(handlers.request_handler, "__self__"):
            return handlers.request_handler.__self__.__class__

        # If it's a function, we can't easily get the class
        # Return None to indicate we couldn't determine it
        return None


# Global registry instance
mode_registry = ModeRegistry()
