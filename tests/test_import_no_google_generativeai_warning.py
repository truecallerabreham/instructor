from __future__ import annotations

import subprocess
import sys


def test_import_instructor_does_not_import_google_generativeai() -> None:
    """
    Regression test for #2044.

    `import instructor` should not import `google.generativeai` (deprecated) and
    should not surface its FutureWarning unless the user explicitly uses the
    legacy Gemini client.
    """

    code = r"""
import importlib.abc
import importlib.machinery
import sys
import warnings


class _GoogleGenerativeAiFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001,D401
        if fullname == "google":
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        if fullname == "google.generativeai":
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):  # noqa: ANN001
        return None

    def exec_module(self, module):  # noqa: ANN001
        name = module.__spec__.name
        if name == "google":
            module.__path__ = []
            return
        if name == "google.generativeai":
            module.__path__ = []
            warnings.warn(
                "All support for the `google.generativeai` package has ended.",
                FutureWarning,
            )
            return


# Ensure our fake modules are used (even if the real Google namespace exists).
sys.modules.pop("google", None)
sys.modules.pop("google.generativeai", None)
sys.meta_path.insert(0, _GoogleGenerativeAiFinder())

warnings.simplefilter("default")
import instructor  # noqa: F401
print("ok")
"""

    proc = subprocess.run(
        [sys.executable, "-W", "default", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "All support for the `google.generativeai` package has ended." not in proc.stderr

