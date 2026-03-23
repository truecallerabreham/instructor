import pytest

from instructor.cli.vestaboard import (
    FLAGSHIP_DIMENSIONS,
    NOTE_DIMENSIONS,
    VestaboardCliError,
    _enforce_dimensions,
    _normalize_base_url,
    _parse_characters_payload,
)


def test_normalize_base_url_adds_trailing_slash() -> None:
    assert (
        _normalize_base_url("https://cloud.vestaboard.com")
        == "https://cloud.vestaboard.com/"
    )


def test_parse_characters_payload_accepts_wrapped_characters() -> None:
    data = {"characters": [[0, 1], [2, 3]]}
    assert _parse_characters_payload(data) == [[0, 1], [2, 3]]


def test_parse_characters_payload_accepts_raw_matrix() -> None:
    data = [[0, 1], [2, 3]]
    assert _parse_characters_payload(data) == [[0, 1], [2, 3]]


def test_parse_characters_payload_rejects_non_rectangular() -> None:
    with pytest.raises(VestaboardCliError, match="rectangular"):
        _parse_characters_payload([[0, 1], [2]])


def test_parse_characters_payload_rejects_non_int() -> None:
    with pytest.raises(VestaboardCliError, match="Expected an integer"):
        _parse_characters_payload([[0, "x"]])  # type: ignore[list-item]


def test_enforce_dimensions_flagship() -> None:
    matrix = [[0] * FLAGSHIP_DIMENSIONS.cols for _ in range(FLAGSHIP_DIMENSIONS.rows)]
    _enforce_dimensions(matrix, expected=FLAGSHIP_DIMENSIONS, no_dimension_check=False)


def test_enforce_dimensions_note() -> None:
    matrix = [[0] * NOTE_DIMENSIONS.cols for _ in range(NOTE_DIMENSIONS.rows)]
    _enforce_dimensions(matrix, expected=NOTE_DIMENSIONS, no_dimension_check=False)


def test_enforce_dimensions_can_be_skipped() -> None:
    matrix = [[0, 0], [0, 0]]
    _enforce_dimensions(matrix, expected=FLAGSHIP_DIMENSIONS, no_dimension_check=True)
