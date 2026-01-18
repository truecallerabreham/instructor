"""Mode handler registry for v2.

Central registry mapping Mode enum values to their handler implementations.
Supports lazy loading, dynamic registration, and queryable API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from instructor import Mode, Provider
from instructor.mode import DEPRECATED_TO_CORE, reset_deprecated_mode_warnings
from instructor.v2.core.protocols import ReaskHandler, RequestHandler, ResponseParser

DEPRECATED_MODE_MAPPING = DEPRECATED_TO_CORE


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
        Mode.warn_deprecated_mode(mode)
        return DEPRECATED_MODE_MAPPING[mode]

    # Add ANTHROPIC_STRUCTURED_OUTPUTS if it exists in the Mode enum
    structured_mode = getattr(Mode, "ANTHROPIC_STRUCTURED_OUTPUTS", None)
    if structured_mode is not None and mode == structured_mode:
        Mode.warn_deprecated_mode(mode)
        return Mode.JSON_SCHEMA

    # Return as-is for core modes
    return mode


def reset_deprecation_warnings() -> None:
    """Reset the deprecation warning tracker.

    Useful for testing to ensure warnings are shown again.
    """
    reset_deprecated_mode_warnings()


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
        self._handlers: dict[tuple[Provider, Mode], ModeHandlers] = {}
        self._lazy_loaders: dict[tuple[Provider, Mode], Callable[[], ModeHandlers]] = {}

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

        raise KeyError(
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
            raise ValueError(
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

    def list_modes(self) -> list[tuple[Provider, Mode]]:
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
        self_obj = getattr(handlers.request_handler, "__self__", None)
        if self_obj is not None:
            return self_obj.__class__

        # If it's a function, we can't easily get the class
        # Return None to indicate we couldn't determine it
        return None


# Global registry instance
mode_registry = ModeRegistry()

_DEFAULT_HANDLERS_LOADED = False
_DEFAULT_HANDLER_MODULES = (
    "instructor.v2.providers.anthropic.handlers",
    "instructor.v2.providers.genai.handlers",
    "instructor.v2.providers.openai.handlers",
    "instructor.v2.providers.cohere.handlers",
    "instructor.v2.providers.xai.handlers",
    "instructor.v2.providers.groq.handlers",
    "instructor.v2.providers.mistral.handlers",
    "instructor.v2.providers.fireworks.handlers",
    "instructor.v2.providers.cerebras.handlers",
    "instructor.v2.providers.writer.handlers",
    "instructor.v2.providers.bedrock.handlers",
)


def _load_default_handlers() -> None:
    """Load built-in handler modules to register modes."""
    global _DEFAULT_HANDLERS_LOADED
    if _DEFAULT_HANDLERS_LOADED:
        return
    for module_path in _DEFAULT_HANDLER_MODULES:
        try:
            __import__(module_path, fromlist=["__name__"])
        except Exception:
            # Handlers should be importable without SDKs.
            # Skip if an unexpected import error occurs.
            continue
    _DEFAULT_HANDLERS_LOADED = True


_load_default_handlers()
