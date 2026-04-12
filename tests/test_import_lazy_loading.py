from __future__ import annotations

import subprocess
import sys


def run_python(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_instructor_does_not_eagerly_load_response_or_provider_utils() -> None:
    result = run_python(
        """
import sys
import instructor

unexpected = [
    name
    for name in (
        "instructor.processing.response",
        "instructor.providers.anthropic.utils",
        "instructor.providers.bedrock.utils",
        "instructor.providers.cohere.utils",
        "instructor.providers.gemini.utils",
        "instructor.providers.mistral.utils",
        "instructor.providers.openai.utils",
    )
    if name in sys.modules
]

if unexpected:
    raise SystemExit(f"unexpected eager imports: {unexpected}")
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_lazy_top_level_exports_import_on_demand() -> None:
    result = run_python(
        """
import sys
import instructor

assert "instructor.processing.response" not in sys.modules
_ = instructor.patch
assert "instructor.processing.response" in sys.modules
"""
    )

    assert result.returncode == 0, result.stderr or result.stdout
