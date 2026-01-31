import builtins
import importlib
import importlib.machinery
import importlib.util
import sys


def _purge_instructor_modules() -> None:
    """Remove instructor modules so we can import it fresh."""
    for name in list(sys.modules.keys()):
        if name == "instructor" or name.startswith("instructor."):
            sys.modules.pop(name, None)


def test_import_instructor_does_not_import_google_generativeai(monkeypatch) -> None:
    """
    Regression test: `import instructor` should not import `google.generativeai`.

    The upstream `google.generativeai` package emits a FutureWarning at import time.
    We don't want that warning unless the user is actively using the legacy Gemini client.
    """

    def fake_find_spec(name: str, package: str | None = None):  # noqa: ARG001
        if name in {"google", "google.generativeai"}:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return None

    real_import = builtins.__import__

    def guard_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("google.generativeai"):
            raise AssertionError(
                "`google.generativeai` should not be imported during `import instructor`"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(builtins, "__import__", guard_import)

    _purge_instructor_modules()
    importlib.import_module("instructor")

    assert "google.generativeai" not in sys.modules
