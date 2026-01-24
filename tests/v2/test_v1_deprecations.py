from __future__ import annotations

import importlib.util
import warnings

import pytest


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is not installed")


def _assert_deprecation_warns(call) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_v1_cohere_warns() -> None:
    _require_module("cohere")
    import cohere
    from instructor.providers.cohere.client import from_cohere

    client_cls = getattr(cohere, "ClientV2", cohere.Client)
    try:
        client = client_cls(api_key="test")
    except Exception as exc:
        pytest.skip(f"Failed to construct Cohere client: {exc}")

    _assert_deprecation_warns(lambda: from_cohere(client))


def test_v1_groq_warns() -> None:
    _require_module("groq")
    import groq
    from instructor.providers.groq.client import from_groq

    try:
        client = groq.Groq(api_key="test")
    except Exception as exc:
        pytest.skip(f"Failed to construct Groq client: {exc}")

    _assert_deprecation_warns(lambda: from_groq(client))


def test_v1_mistral_warns() -> None:
    _require_module("mistralai")
    from mistralai import Mistral
    from instructor.providers.mistral.client import from_mistral

    try:
        client = Mistral(api_key="test")
    except Exception as exc:
        pytest.skip(f"Failed to construct Mistral client: {exc}")

    _assert_deprecation_warns(lambda: from_mistral(client))


def test_v1_writer_warns() -> None:
    _require_module("writerai")
    from writerai import Writer
    from instructor.providers.writer.client import from_writer

    try:
        client = Writer(api_key="test")
    except Exception as exc:
        pytest.skip(f"Failed to construct Writer client: {exc}")

    _assert_deprecation_warns(lambda: from_writer(client))
