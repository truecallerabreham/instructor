from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Final

import aiohttp
import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer(no_args_is_help=True)
console = Console()

DEFAULT_CLOUD_API_URL: Final[str] = "https://cloud.vestaboard.com/"
DEFAULT_VBML_FORMAT_URL: Final[str] = "https://vbml.vestaboard.com/format"


class VestaboardCliError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatrixDimensions:
    rows: int
    cols: int


FLAGSHIP_DIMENSIONS: Final[MatrixDimensions] = MatrixDimensions(rows=6, cols=22)
NOTE_DIMENSIONS: Final[MatrixDimensions] = MatrixDimensions(rows=3, cols=15)


def _normalize_base_url(base_url: str) -> str:
    base_url = base_url.strip()
    if not base_url:
        raise VestaboardCliError("Base URL cannot be blank.")
    if not base_url.endswith("/"):
        base_url += "/"
    return base_url


def _read_json_source(source: str) -> Any:
    if source == "-":
        raw = sys.stdin.read()
    else:
        with open(source, encoding="utf-8") as f:
            raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise VestaboardCliError(f"Invalid JSON in {source!r}: {e}") from e


def _as_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VestaboardCliError(
            f"Expected an integer at {path}, got {type(value).__name__}."
        )
    return value


def _parse_characters_payload(data: Any) -> list[list[int]]:
    """
    Accept either:
    - [[...], [...], ...] (raw array body)
    - {"characters": [[...], [...], ...]}
    """
    if isinstance(data, dict):
        if "characters" not in data:
            raise VestaboardCliError(
                "Expected an object with a 'characters' key, or a raw matrix array."
            )
        data = data["characters"]

    if not isinstance(data, list) or not data:
        raise VestaboardCliError("Expected a non-empty list of rows.")

    matrix: list[list[int]] = []
    expected_cols: int | None = None
    for r_idx, row in enumerate(data):
        if not isinstance(row, list) or not row:
            raise VestaboardCliError(f"Row {r_idx} must be a non-empty list.")
        if expected_cols is None:
            expected_cols = len(row)
        elif len(row) != expected_cols:
            raise VestaboardCliError(
                f"Matrix must be rectangular. Row {r_idx} has {len(row)} columns, "
                f"expected {expected_cols}."
            )

        parsed_row: list[int] = []
        for c_idx, value in enumerate(row):
            code = _as_int(value, path=f"characters[{r_idx}][{c_idx}]")
            if code < 0 or code > 255:
                raise VestaboardCliError(
                    f"Character code out of range at characters[{r_idx}][{c_idx}]: {code}. "
                    "Expected 0..255."
                )
            parsed_row.append(code)
        matrix.append(parsed_row)

    return matrix


def _enforce_dimensions(
    matrix: list[list[int]], *, expected: MatrixDimensions, no_dimension_check: bool
) -> None:
    if no_dimension_check:
        return
    if len(matrix) != expected.rows:
        raise VestaboardCliError(
            f"Expected {expected.rows} rows, got {len(matrix)}. "
            "If this is a different board size, pass --no-dimension-check."
        )
    cols = len(matrix[0])
    if cols != expected.cols:
        raise VestaboardCliError(
            f"Expected {expected.cols} columns, got {cols}. "
            "If this is a different board size, pass --no-dimension-check."
        )


async def _http_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: Any | None,
    timeout_s: float,
) -> tuple[int, Any]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            method, url, headers=headers, json=json_body
        ) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            text = await resp.text()

            if "application/json" in content_type:
                try:
                    return status, json.loads(text) if text else None
                except json.JSONDecodeError:
                    return status, {"raw": text}

            return status, {"raw": text}


def _resolve_token(token: str | None) -> str | None:
    return token or os.getenv("VESTABOARD_TOKEN")


def _require_token(token: str | None) -> str:
    resolved = _resolve_token(token)
    if not resolved:
        raise VestaboardCliError(
            "Missing Vestaboard token. Set VESTABOARD_TOKEN or pass --token."
        )
    return resolved


def _print_json(obj: Any) -> None:
    syntax = Syntax(json.dumps(obj, indent=2, sort_keys=True), "json", theme="monokai")
    console.print(syntax)


@app.command("send-text", help="Send a plain text message to your Vestaboard.")
def send_text(
    text: str = typer.Argument(..., help="Text to send to the board."),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Vestaboard Cloud API token. Defaults to VESTABOARD_TOKEN.",
        envvar="VESTABOARD_TOKEN",
    ),
    base_url: str = typer.Option(
        DEFAULT_CLOUD_API_URL, "--base-url", help="Vestaboard Cloud API base URL."
    ),
    forced: bool = typer.Option(
        False,
        "--forced",
        help="Override quiet hours (send even during quiet hours).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the request body and exit without calling the API.",
    ),
    timeout_s: float = typer.Option(
        15.0, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """
    Sends `{"text": "...", "forced": false}` to `https://cloud.vestaboard.com/`.
    """
    try:
        url = _normalize_base_url(base_url)
        payload: dict[str, Any] = {"text": text}
        if forced:
            payload["forced"] = True

        if dry_run:
            _print_json(
                {"url": url, "headers": {"X-Vestaboard-Token": "***"}, "body": payload}
            )
            return

        token_value = _require_token(token)
        headers = {
            "X-Vestaboard-Token": token_value,
            "Content-Type": "application/json",
        }
        status, body = asyncio.run(
            _http_json(
                method="POST",
                url=url,
                headers=headers,
                json_body=payload,
                timeout_s=timeout_s,
            )
        )
        if status >= 400:
            raise VestaboardCliError(f"HTTP {status}: {body}")
        _print_json(body)
    except VestaboardCliError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command("send-matrix", help="Send a full character matrix to your Vestaboard.")
def send_matrix(
    source: str = typer.Argument(
        ...,
        help="Path to JSON file with the matrix, or '-' to read JSON from stdin.",
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        help="Vestaboard Cloud API token. Defaults to VESTABOARD_TOKEN.",
        envvar="VESTABOARD_TOKEN",
    ),
    base_url: str = typer.Option(
        DEFAULT_CLOUD_API_URL, "--base-url", help="Vestaboard Cloud API base URL."
    ),
    note: bool = typer.Option(
        False,
        "--note",
        help="Validate against Vestaboard Note dimensions (3x15) instead of flagship (6x22).",
    ),
    no_dimension_check: bool = typer.Option(
        False,
        "--no-dimension-check",
        help="Skip dimension validation (useful for Note Arrays and custom layouts).",
    ),
    raw_array_body: bool = typer.Option(
        False,
        "--raw-array-body",
        help="Send the raw matrix as the request body, instead of wrapping in {'characters': ...}.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the request body and exit without calling the API.",
    ),
    timeout_s: float = typer.Option(
        15.0, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """
    Sends either:
    - `{"characters": [[...],[...],...]}` (default), or
    - `[[...],[...],...]` with `--raw-array-body`
    """
    try:
        url = _normalize_base_url(base_url)

        data = _read_json_source(source)
        matrix = _parse_characters_payload(data)
        expected = NOTE_DIMENSIONS if note else FLAGSHIP_DIMENSIONS
        _enforce_dimensions(
            matrix, expected=expected, no_dimension_check=no_dimension_check
        )

        body: Any = matrix if raw_array_body else {"characters": matrix}
        if dry_run:
            _print_json(
                {"url": url, "headers": {"X-Vestaboard-Token": "***"}, "body": body}
            )
            return

        token_value = _require_token(token)
        headers = {
            "X-Vestaboard-Token": token_value,
            "Content-Type": "application/json",
        }
        status, resp_body = asyncio.run(
            _http_json(
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                timeout_s=timeout_s,
            )
        )
        if status >= 400:
            raise VestaboardCliError(f"HTTP {status}: {resp_body}")
        _print_json(resp_body)
    except VestaboardCliError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e


@app.command("format", help="Format text into a character matrix using VBML.")
def format_message(
    message: str = typer.Argument(..., help="Text to format into a character matrix."),
    output: str | None = typer.Option(
        None, "--output", help="Write the resulting matrix JSON to a file."
    ),
    timeout_s: float = typer.Option(
        15.0, "--timeout", help="Request timeout in seconds."
    ),
) -> None:
    """
    Calls `https://vbml.vestaboard.com/format` and returns a matrix of character codes.
    """
    try:
        headers = {"Content-Type": "application/json"}
        status, body = asyncio.run(
            _http_json(
                method="POST",
                url=DEFAULT_VBML_FORMAT_URL,
                headers=headers,
                json_body={"message": message},
                timeout_s=timeout_s,
            )
        )
        if status >= 400:
            raise VestaboardCliError(f"HTTP {status}: {body}")
        if output:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(body, f, indent=2)
                f.write("\n")
            console.print(f"[green]Wrote matrix to[/green] {output}")
        else:
            typer.echo(json.dumps(body))
    except VestaboardCliError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e
