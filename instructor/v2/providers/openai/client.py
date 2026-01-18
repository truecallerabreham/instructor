"""v2 OpenAI client factory.

Creates Instructor instances using v2 hierarchical registry system.
"""

from __future__ import annotations

from typing import Any, overload

import openai

from instructor import AsyncInstructor, Instructor, Mode, Provider
from instructor.v2.core.patch import patch_v2

# Ensure handlers are registered (decorators auto-register on import)
from instructor.v2.providers.openai import handlers  # noqa: F401


@overload
def from_openai(
    client: openai.OpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_openai(
    client: openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_openai(
    client: openai.OpenAI | openai.AsyncOpenAI,
    mode: Mode = Mode.TOOLS,
    model: str | None = None,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    """Create an Instructor instance from an OpenAI client using v2 registry.

    Args:
        client: An instance of OpenAI client (sync or async)
        mode: The mode to use (defaults to Mode.TOOLS)
        model: Optional model to inject if not provided in requests
        **kwargs: Additional keyword arguments to pass to the Instructor constructor

    Returns:
        An Instructor instance (sync or async depending on the client type)

    Raises:
        ModeError: If mode is not registered for OpenAI
        ClientError: If client is not a valid OpenAI client instance

    Examples:
        >>> import openai
        >>> from instructor import Mode
        >>> from instructor.v2.providers.openai import from_openai
        >>>
        >>> client = openai.OpenAI()
        >>> instructor_client = from_openai(client, mode=Mode.TOOLS)
        >>>
        >>> # Or use JSON_SCHEMA mode for structured outputs
        >>> instructor_client = from_openai(client, mode=Mode.JSON_SCHEMA)
    """
    from instructor.v2.core.registry import mode_registry, normalize_mode

    # Normalize provider-specific modes to generic modes
    # FUNCTIONS -> TOOLS, TOOLS_STRICT -> TOOLS, etc.
    normalized_mode = normalize_mode(Provider.OPENAI, mode)

    # Validate mode is registered (use normalized mode for check)
    if not mode_registry.is_registered(Provider.OPENAI, normalized_mode):
        from instructor.core.exceptions import ModeError

        available_modes = mode_registry.get_modes_for_provider(Provider.OPENAI)
        raise ModeError(
            mode=mode.value,
            provider=Provider.OPENAI.value,
            valid_modes=[m.value for m in available_modes],
        )

    # Use normalized mode for patching
    mode = normalized_mode

    # Validate client type
    valid_client_types = (
        openai.OpenAI,
        openai.AsyncOpenAI,
    )

    if not isinstance(client, valid_client_types):
        from instructor.core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of one of: {', '.join(t.__name__ for t in valid_client_types)}. "
            f"Got: {type(client).__name__}"
        )

    # Get create function
    create = client.chat.completions.create

    # Patch using v2 registry, passing the model for injection
    patched_create = patch_v2(
        func=create,
        provider=Provider.OPENAI,
        mode=mode,
        default_model=model,
    )

    # Return sync or async instructor
    if isinstance(client, openai.OpenAI):
        return Instructor(
            client=client,
            create=patched_create,
            provider=Provider.OPENAI,
            mode=mode,
            **kwargs,
        )
    else:
        return AsyncInstructor(
            client=client,
            create=patched_create,
            provider=Provider.OPENAI,
            mode=mode,
            **kwargs,
        )
