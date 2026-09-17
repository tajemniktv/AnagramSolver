"""Dependency-keyed, atomic result cache for the normal CLI."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

from anagram_cache_io import publish
from anagram_paths import NGRAM_DIR, PROJECT_DIR, WORDNET_DIR


def candidate_content_hash(candidates: Path) -> str:
    digest = hashlib.sha256()
    with candidates.open("rb") as handle:
        for line in handle:
            if not line.startswith((b"# SEARCH ", b"# SHA256 ")):
                digest.update(line)
    return digest.hexdigest()


def ranking_key(candidates: Path, options: list[str], phrase_db: Path | None) -> str:
    digest = hashlib.sha256()
    digest.update(candidate_content_hash(candidates).encode())
    digest.update(json.dumps(options).encode())
    for source in sorted(PROJECT_DIR.glob("anagram_*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    # Runtime corpora are large. Their resolved identity, size and nanosecond
    # modification time invalidate reuse without hashing gigabytes per solve.
    paths = [
        p for root in (NGRAM_DIR, WORDNET_DIR) for p in root.rglob("*") if p.is_file()
    ]
    if phrase_db is not None:
        db = phrase_db.expanduser().resolve()
        paths.extend([db, Path(str(db) + "-wal")])
    for path in sorted(paths):
        digest.update(str(path.resolve()).encode())
        try:
            stat = path.stat()
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        except FileNotFoundError:
            digest.update(b"missing")
    return digest.hexdigest()[:24]


def load_results(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != 1:
            return None
        payload = value["payload"]
        raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        if hashlib.sha256(raw).hexdigest() != value["checksum"]:
            return None
        if not isinstance(payload["summary"], dict) or not isinstance(
            payload["results"], list
        ):
            return None
        for row in payload["results"]:
            if not isinstance(row["phrase"], str) or not math.isfinite(row["score"]):
                return None
            if not all(
                type(row[key]) is int and row[key] > 0 for key in ("word_count", "rank")
            ):
                return None
        return payload
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return None


def save_results(path: Path, payload: dict) -> None:
    raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    value = {
        "schema": 1,
        "checksum": hashlib.sha256(raw).hexdigest(),
        "payload": payload,
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
        publish(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
