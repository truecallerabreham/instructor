"""v2 Fireworks provider.

Fireworks uses an OpenAI-compatible API, so the handlers inherit from OpenAI.
Supports TOOLS and MD_JSON modes.
"""

from instructor.v2.providers.fireworks.client import from_fireworks

__all__ = ["from_fireworks"]
