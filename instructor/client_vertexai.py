"""Backward-compatible VertexAI client module."""

from builtins import isinstance as isinstance

from .providers.vertexai.client import from_vertexai

__all__ = ["from_vertexai"]
