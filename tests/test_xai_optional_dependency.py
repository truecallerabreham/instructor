import pytest


def test_from_provider_xai_requires_optional_extra(monkeypatch: pytest.MonkeyPatch):
    import instructor
    from instructor.core.exceptions import ConfigurationError
    from instructor.providers.xai import client as xai_client

    # Force the "optional dependency missing" branch, even if xai-sdk is installed.
    monkeypatch.setattr(xai_client, "SyncClient", None, raising=False)
    monkeypatch.setattr(xai_client, "AsyncClient", None, raising=False)
    monkeypatch.setattr(xai_client, "xchat", None, raising=False)

    with pytest.raises(ConfigurationError) as excinfo:
        instructor.from_provider("xai/grok-3-mini", api_key="test-key")

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "uv pip install" in msg


def test_direct_from_xai_has_clear_error_when_sdk_missing(monkeypatch: pytest.MonkeyPatch):
    from instructor.core.exceptions import ConfigurationError
    from instructor.providers.xai import client as xai_client

    # Force the "optional dependency missing" branch, even if xai-sdk is installed.
    monkeypatch.setattr(xai_client, "SyncClient", None, raising=False)
    monkeypatch.setattr(xai_client, "AsyncClient", None, raising=False)
    monkeypatch.setattr(xai_client, "xchat", None, raising=False)

    with pytest.raises(ConfigurationError) as excinfo:
        xai_client.from_xai(object())  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert "instructor[xai]" in msg
    assert "xai-sdk" in msg

