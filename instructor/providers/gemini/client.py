from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

import instructor


if TYPE_CHECKING:
    import google.generativeai as genai  # type: ignore[import-not-found]

    GeminiGenerativeModel = genai.GenerativeModel
else:
    GeminiGenerativeModel = Any


@overload
def from_gemini(
    client: GeminiGenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor: ...


@overload
def from_gemini(
    client: GeminiGenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor: ...


def from_gemini(
    client: GeminiGenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    import warnings

    warnings.warn(
        "from_gemini is deprecated and will be removed in a future version. "
        "Please use from_genai or from_provider instead. "
        "Install google-genai with: pip install google-genai\n"
        "Example migration:\n"
        "  # Old way\n"
        "  from instructor import from_gemini\n"
        "  import google.generativeai as genai\n"
        "  client = from_gemini(genai.GenerativeModel('gemini-3-flash'))\n\n"
        "  # New way\n"
        "  from instructor import from_genai\n"
        "  from google import genai\n"
        "  client = from_genai(genai.Client())\n"
        "  # OR use from_provider\n"
        "  client = instructor.from_provider('google/gemini-3-flash')",
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

    # We avoid importing `google.generativeai` at import-time (it emits a FutureWarning).
    # Since `from_gemini` is deprecated, a simple duck-type check is enough here.
    if not (
        hasattr(client, "generate_content")
        and hasattr(client, "generate_content_async")
    ):
        from ...core.exceptions import ClientError

        raise ClientError(
            "Client must look like a google.generativeai GenerativeModel. "
            f"Got: {type(client).__name__}"
        )

    if use_async:
        create = client.generate_content_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )

    create = client.generate_content
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.GEMINI,
        mode=mode,
        **kwargs,
    )
