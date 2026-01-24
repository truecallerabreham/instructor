from __future__ import annotations

from typing import Any, Literal, overload

import google.generativeai as genai  # type: ignore[import-not-found]

import instructor


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    import warnings

    warnings.warn(
        "from_gemini() is deprecated and will be removed in v2.0. "
        "Use instructor.from_provider('google/<model>') or "
        "instructor.v2.providers.gemini.from_gemini(..., mode=Mode.TOOLS or Mode.MD_JSON).",
        DeprecationWarning,
        stacklevel=2,
    )

    valid_modes = {
        instructor.Mode.GEMINI_JSON,
        instructor.Mode.GEMINI_TOOLS,
    }

    if mode not in valid_modes:
        from ...core.exceptions import ModeError

        raise ModeError(
            mode=str(mode), provider="Gemini", valid_modes=[str(m) for m in valid_modes]
        )

    if not isinstance(client, genai.GenerativeModel):
        from ...core.exceptions import ClientError

        raise ClientError(
            f"Client must be an instance of genai.GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    if use_async:
        create = client.generate_content_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(
                create=create, mode=mode, provider=instructor.Provider.GEMINI
            ),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )

    create = client.generate_content
    return instructor.Instructor(
        client=client,
        create=instructor.patch(
            create=create, mode=mode, provider=instructor.Provider.GEMINI
        ),
        provider=instructor.Provider.GEMINI,
        mode=mode,
        **kwargs,
    )
