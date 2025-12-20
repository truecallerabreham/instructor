"""
Debugging Example: hooks, logging, and parsing flow (no network).

This script:
- Enables DEBUG logging for instructor.*
- Patches a fake provider create function
- Attaches hook handlers to print kwargs, response types, and parse errors
- Parses a simple JSON payload into a Pydantic model
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

import instructor
from instructor.core.hooks import HookName, Hooks
from instructor.mode import Mode


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("instructor").setLevel(logging.DEBUG)


class User(BaseModel):
    name: str
    age: int


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletion:
    choices: list[FakeChoice]


def fake_create(*args: Any, **kwargs: Any) -> FakeCompletion:
    _ = args, kwargs
    payload = '{"name": "Jane Doe", "age": 41}'
    return FakeCompletion(choices=[FakeChoice(message=FakeMessage(content=payload))])


def main() -> None:
    patched_create = instructor.patch(create=fake_create, mode=Mode.JSON)

    hooks = Hooks()
    hooks.on(HookName.COMPLETION_KWARGS, lambda **kw: print("KWARGS:", kw.keys()))
    hooks.on(
        HookName.COMPLETION_RESPONSE,
        lambda resp: print("RESPONSE TYPE:", type(resp).__name__),
    )
    hooks.on(HookName.PARSE_ERROR, lambda e: print("PARSE ERROR:", e))

    result = patched_create(
        model="fake-model",
        messages=[{"role": "user", "content": "Extract Jane Doe, 41"}],
        response_model=User,
        hooks=hooks,
    )
    print("PARSED:", result)


if __name__ == "__main__":
    main()
