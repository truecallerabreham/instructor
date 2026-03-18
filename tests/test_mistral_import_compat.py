from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any, cast
from unittest.mock import MagicMock

import instructor

from instructor.auto_client import from_provider


def _install_fake_mistralai_modules(*, export_root_class: bool = False):
    root = ModuleType("mistralai")
    root.__path__ = []
    client_module = ModuleType("mistralai.client")

    class FakeChat:
        def stream(self, *_args, **_kwargs):  # noqa: ANN002,ANN003
            raise AssertionError("stream should not be called in this test")

        def complete(self, *_args, **_kwargs):  # noqa: ANN002,ANN003
            raise AssertionError("complete should not be called in this test")

        async def stream_async(self, *_args, **_kwargs):  # noqa: ANN002,ANN003
            raise AssertionError("stream_async should not be called in this test")

        async def complete_async(self, *_args, **_kwargs):  # noqa: ANN002,ANN003
            raise AssertionError("complete_async should not be called in this test")

    class FakeMistral:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = FakeChat()

    if export_root_class:
        cast(Any, root).Mistral = FakeMistral
    cast(Any, client_module).Mistral = FakeMistral

    return root, client_module, FakeMistral


def test_from_mistral_accepts_client_submodule_layout(monkeypatch) -> None:
    root, client_module, FakeMistral = _install_fake_mistralai_modules()
    monkeypatch.setitem(sys.modules, "mistralai", root)
    monkeypatch.setitem(sys.modules, "mistralai.client", client_module)

    import instructor.providers.mistral.client as mistral_client

    reloaded_client = cast(Any, importlib.reload(mistral_client))

    patched = reloaded_client.from_mistral(FakeMistral("test-key"))

    assert isinstance(patched, instructor.Instructor)


def test_auto_client_supports_client_submodule_layout(monkeypatch) -> None:
    root, client_module, FakeMistral = _install_fake_mistralai_modules()
    monkeypatch.setitem(sys.modules, "mistralai", root)
    monkeypatch.setitem(sys.modules, "mistralai.client", client_module)

    mock_from_mistral = MagicMock(return_value="patched-client")
    monkeypatch.setattr(instructor, "from_mistral", mock_from_mistral, raising=False)

    result = from_provider("mistral/mistral-small-latest", api_key="test-key")

    assert result == "patched-client"
    args, kwargs = mock_from_mistral.call_args
    assert isinstance(args[0], FakeMistral)
    assert kwargs["model"] == "mistral-small-latest"
