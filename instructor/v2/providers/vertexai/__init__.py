"""VertexAI v2 provider handlers and client."""

from .client import from_vertexai
from .handlers import (
    VertexAIJSONHandler,
    VertexAIParallelToolsHandler,
    VertexAIToolsHandler,
)

__all__ = [
    "VertexAIJSONHandler",
    "VertexAIParallelToolsHandler",
    "VertexAIToolsHandler",
    "from_vertexai",
]
