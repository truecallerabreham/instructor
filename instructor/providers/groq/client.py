from __future__ import annotations

from typing import overload, Any
import warnings

import groq
import instructor


@overload
def from_groq(
    client: groq.Groq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor: ...


@overload
def from_groq(
    client: groq.AsyncGroq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


def from_groq(
    client: groq.Groq | groq.AsyncGroq,
    mode: instructor.Mode = instructor.Mode.TOOLS,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    warnings.warn(
        "from_groq() is deprecated and will be removed in v2.0. "
        "Use instructor.from_provider('groq/<model>') or "
        "instructor.v2.providers.groq.from_groq(..., mode=Mode.TOOLS or Mode.MD_JSON).",
        DeprecationWarning,
        stacklevel=2,
    )
    valid_modes = {
        instructor.Mode.JSON,
        instructor.Mode.TOOLS,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Groq", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, (groq.Groq, groq.AsyncGroq)):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of groq.Groq or groq.AsyncGroq. "
            f"Got: {type(client).__name__}"
        )

    if isinstance(client, groq.Groq):
        return instructor.Instructor(
            client=client,
            create=instructor.patch(
                create=client.chat.completions.create,
                mode=mode,
                provider=instructor.Provider.GROQ,
            ),
            provider=instructor.Provider.GROQ,
            mode=mode,
            **kwargs,
        )

    else:
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=client.chat.completions.create,
                mode=mode,
                provider=instructor.Provider.GROQ,
            ),
            provider=instructor.Provider.GROQ,
            mode=mode,
            **kwargs,
        )
