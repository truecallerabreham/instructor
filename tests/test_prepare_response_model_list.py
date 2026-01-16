from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel

from instructor.dsl.iterable import IterableBase
from instructor.utils.core import prepare_response_model


class User(BaseModel):
    name: str


def test_prepare_response_model_supports_list_of_basemodel() -> None:
    rm = prepare_response_model(list[User])
    assert rm is not None
    assert issubclass(rm, IterableBase)


def test_prepare_response_model_supports_iterable_of_basemodel() -> None:
    rm_iter = prepare_response_model(Iterable[User])
    assert rm_iter is not None
    assert issubclass(rm_iter, IterableBase)

