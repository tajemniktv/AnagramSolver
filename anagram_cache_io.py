"""Crash-safe publication of generated files and downloaded runtime data."""

from __future__ import annotations

import shutil
import time
import urllib.request
import uuid
from pathlib import Path


def publish(temporary: Path, destination: Path) -> None:
    """Retry transient sharing violations, never hide persistent I/O errors."""
    for attempt in range(6):
        try:
            temporary.replace(destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.01 * 2**attempt)


def download_atomic(
    request: urllib.request.Request, destination: Path, *, timeout: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            temporary.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
            headers = getattr(response, "headers", {})
            expected = headers.get("Content-Length")
            if expected is not None and handle.tell() != int(expected):
                raise OSError(
                    "Incomplete download: response length does not match Content-Length"
                )
        publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
