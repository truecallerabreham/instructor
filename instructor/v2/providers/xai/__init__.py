"""v2 xAI provider.

Provides Instructor integration for xAI's Grok models using the v2 registry system.
"""

try:
    from instructor.v2.providers.xai.client import from_xai
except ImportError:
    from_xai = None  # type: ignore
except Exception:
    # Catch other exceptions (like ConfigurationError) that might occur during import
    # This can happen if handlers are registered multiple times, but the registry
    # should now handle this idempotently. If we still get here, set to None to
    # allow the import to succeed.
    from_xai = None  # type: ignore

__all__ = ["from_xai"]
