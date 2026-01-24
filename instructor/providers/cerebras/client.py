from __future__ import annotations  # type: ignore

from typing import Any, overload
import warnings

from cerebras.cloud.sdk import Cerebras, AsyncCerebras

import instructor
from ...core.client import AsyncInstructor, Instructor


@overload
def from_cerebras(
    client: Cerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> Instructor: ...


@overload
def from_cerebras(
    client: AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> AsyncInstructor: ...


def from_cerebras(
    client: Cerebras | AsyncCerebras,
    mode: instructor.Mode = instructor.Mode.CEREBRAS_TOOLS,
    **kwargs: Any,
) -> Instructor | AsyncInstructor:
    warnings.warn(
        "from_cerebras() is deprecated and will be removed in v2.0. "
        "Use instructor.from_provider('cerebras/<model>') or "
        "instructor.v2.providers.cerebras.from_cerebras(..., mode=Mode.TOOLS or Mode.MD_JSON).",
        DeprecationWarning,
        stacklevel=2,
    )
    valid_modes = {
        instructor.Mode.CEREBRAS_TOOLS,
        instructor.Mode.CEREBRAS_JSON,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode),
            provider="Cerebras",
            valid_modes=[str(m) for m in valid_modes],
        )

    if not isinstance(client, (Cerebras, AsyncCerebras)):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of Cerebras or AsyncCerebras. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, AsyncCerebras):
        create = client.chat.completions.create
        return AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=create, mode=mode, provider=instructor.Provider.CEREBRAS
            ),
            provider=instructor.Provider.CEREBRAS,
            mode=mode,
            **kwargs,
        )

    create = client.chat.completions.create
    return Instructor(
        client=client,
        create=instructor.patch(
            create=create, mode=mode, provider=instructor.Provider.CEREBRAS
        ),
        provider=instructor.Provider.CEREBRAS,
        mode=mode,
        **kwargs,
    )
