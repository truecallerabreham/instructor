"""v2 Anthropic provider."""

try:
    from instructor.v2.providers.anthropic.client import from_anthropic
except ImportError:
    from_anthropic = None  # type: ignore
except Exception:
    # Catch other exceptions (like ConfigurationError) that might occur during import
    # This can happen if handlers are registered multiple times, but the registry
    # should now handle this idempotently. If we still get here, set to None to
    # allow the import to succeed.
    from_anthropic = None  # type: ignore

__all__ = ["from_anthropic"]
