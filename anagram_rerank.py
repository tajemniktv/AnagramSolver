#!/usr/bin/env python3
"""Active AnagramSolver front-end with scoped top-K and safety adapters."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import multiprocessing
import sys
from pathlib import Path

import anagram_rerank_core as core

# Capture the stable core functions that facade wrappers delegate to. These must
# not be looked up dynamically after main() installs the facade overrides.
_CORE_PREPARE_ROWS = core.prepare_rows

# The top-K implementation predates this facade and historically patched core at
# import time. Capture and restore the frozen core API immediately so merely
# importing ``anagram_rerank`` is side-effect free.
_CORE_HOOK_NAMES = (
    "best_order",
    "deep_analyze",
    "apply_phrase_rescore",
    "DeepResult",
)
_core_before_impl = {name: getattr(core, name) for name in _CORE_HOOK_NAMES}
import anagram_rerank_topk_impl as impl  # noqa: E402
for _name, _value in _core_before_impl.items():
    setattr(core, _name, _value)

# Preserve the benchmark-facing API exposed by the implementation.
for _name in dir(impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(impl, _name)

ENGINE_LAYER = "top-k-order-reranking-reviewed"
PREPARED_CACHE_SCHEMA = "topk-prepared-json-gzip-1"


def _prepared_cache_key(input_path: Path, wordnet_dir: Path) -> str:
    """Fingerprint prepared caches independently of the legacy pickle schema."""
    h = hashlib.sha256()
    h.update(PREPARED_CACHE_SCHEMA.encode("ascii"))
    h.update(str(input_path.resolve()).encode("utf-8", "replace"))
    h.update(core._hash_file(input_path).encode("ascii"))
    for name in (
        "index.noun", "index.verb", "index.adj", "index.adv",
        "noun.exc", "verb.exc", "data.verb",
    ):
        path = wordnet_dir / name
        if path.exists():
            stat = path.stat()
            h.update(name.encode("ascii"))
            h.update(str(stat.st_size).encode("ascii"))
            h.update(str(stat.st_mtime_ns).encode("ascii"))
    return h.hexdigest()[:24]


def _row_to_cache_dict(row: Row) -> dict[str, object]:
    return {
        "words": list(row.words),
        "word_count": row.word_count,
        "old_rank": row.old_rank,
        "old_pre": row.old_pre,
        "lex": row.lex,
        "fam": row.fam,
        "old_pair": row.old_pair,
        "hint": row.hint,
        "zavg": row.zavg,
        "zmin": row.zmin,
        "old_pcov": row.old_pcov,
        "hints": list(row.hints),
        "wn_coverage": row.wn_coverage,
        "grammar_potential": row.grammar_potential,
        "grammar_potential_norm": row.grammar_potential_norm,
        "v13_pre": row.v13_pre,
        "family_key": list(row.family_key),
    }


_CACHE_KEYS = frozenset(_row_to_cache_dict(Row(
    words=(), word_count=0, old_rank=0, old_pre=0.0, lex=0.0, fam=0.0,
    old_pair=0.0, hint=0.0, zavg=0.0, zmin=0.0, old_pcov=0.0, hints=(),
)).keys())

# Prepared-cache values come from bounded ranking features. Enforce those
# invariants at the trust boundary so parseable NaN/Infinity or absurd finite
# values cannot leak into sorting/scoring.
_CACHE_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "old_pre": (0.0, 100.0),
    "lex": (0.0, 1.0),
    "fam": (0.0, 1.0),
    "old_pair": (0.0, 1.0),
    "hint": (0.0, 1.0),
    "zavg": (0.0, 20.0),
    "zmin": (0.0, 20.0),
    "old_pcov": (0.0, 1.0),
    "wn_coverage": (0.0, 1.0),
    "grammar_potential": (0.0, 1.0),
    "grammar_potential_norm": (0.0, 1.0),
    "v13_pre": (0.0, 100.0),
}
_MAX_CACHE_WORDS = 64
_MAX_CACHE_RANK = 1_000_000_000


def _bounded_number(item: dict[str, object], name: str) -> float | None:
    value = item.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    lo, hi = _CACHE_FLOAT_RANGES[name]
    if not math.isfinite(number) or not lo <= number <= hi:
        return None
    return number


def _row_from_cache_dict(item: object) -> Row | None:
    if not isinstance(item, dict) or set(item) != _CACHE_KEYS:
        return None

    def strings(name: str) -> tuple[str, ...] | None:
        value = item.get(name)
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            return None
        return tuple(value)

    words = strings("words")
    hints = strings("hints")
    family_key = strings("family_key")
    if words is None or hints is None or family_key is None:
        return None

    word_count = item.get("word_count")
    old_rank = item.get("old_rank")
    if (
        not isinstance(word_count, int)
        or isinstance(word_count, bool)
        or not 1 <= word_count <= _MAX_CACHE_WORDS
        or word_count != len(words)
        or not isinstance(old_rank, int)
        or isinstance(old_rank, bool)
        or not 1 <= old_rank <= _MAX_CACHE_RANK
    ):
        return None

    numeric = {name: _bounded_number(item, name) for name in _CACHE_FLOAT_RANGES}
    if any(value is None for value in numeric.values()):
        return None

    return Row(
        words=words,
        word_count=word_count,
        old_rank=old_rank,
        old_pre=numeric["old_pre"],
        lex=numeric["lex"],
        fam=numeric["fam"],
        old_pair=numeric["old_pair"],
        hint=numeric["hint"],
        zavg=numeric["zavg"],
        zmin=numeric["zmin"],
        old_pcov=numeric["old_pcov"],
        hints=hints,
        wn_coverage=numeric["wn_coverage"],
        grammar_potential=numeric["grammar_potential"],
        grammar_potential_norm=numeric["grammar_potential_norm"],
        v13_pre=numeric["v13_pre"],
        family_key=family_key,
    )


def load_prepared_cache(cache_path: Path) -> list[Row] | None:
    """Load validated primitive gzip/JSON data, never executable pickle objects."""
    try:
        with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("schema") != PREPARED_CACHE_SCHEMA:
            return None
        raw_rows = payload.get("rows")
        if not isinstance(raw_rows, list):
            return None
        rows: list[Row] = []
        for item in raw_rows:
            row = _row_from_cache_dict(item)
            if row is None:
                return None
            rows.append(row)
        core._reset_deep_fields(rows)
        return rows
    except (OSError, EOFError, json.JSONDecodeError, UnicodeError, TypeError, ValueError):
        return None


def save_prepared_cache(cache_path: Path, rows: list[Row]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(
            {
                "schema": PREPARED_CACHE_SCHEMA,
                "rows": [_row_to_cache_dict(row) for row in rows],
            },
            handle,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    tmp.replace(cache_path)


def prepare_rows(rows: list[Row], lex: WordNetLexicon) -> None:
    """Canonicalize unordered bags, then delegate to the captured core function."""
    for row in rows:
        row.words = tuple(sorted(row.words))
    rows.sort(key=lambda row: (row.word_count, row.words))
    _CORE_PREPARE_ROWS(rows, lex)


def _clear_order_side_tables() -> None:
    for name in ("_ORDER_CANDIDATES_BY_ROW_ID", "_ORDER_CANDIDATES_BY_INDEX"):
        table = getattr(impl, name, None)
        if isinstance(table, dict):
            table.clear()


def apply_phrase_rescore(
    rows: list[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
    bonus_max: float,
) -> int:
    """Canonicalize shortlist tie-breaks and release retained order state afterward."""
    for row in rows:
        row.words = tuple(sorted(row.words))
    try:
        return impl.apply_phrase_rescore(
            rows,
            collocation=collocation,
            phrase_index=phrase_index,
            top_per_group=top_per_group,
            bonus_max=bonus_max,
        )
    finally:
        _clear_order_side_tables()


def main() -> int:
    """Run the legacy core with scoped top-K/safety hooks, restoring it afterward."""
    original_argv = sys.argv[:]
    cleaned, count = impl._consume_int_flag(
        original_argv[1:], "--order-candidates", impl.DEFAULT_ORDER_CANDIDATES
    )
    if count < 1:
        raise SystemExit("--order-candidates must be >= 1")
    impl._ORDER_CANDIDATE_COUNT = count

    overrides = {
        "best_order": impl.best_order,
        "deep_analyze": impl.deep_analyze,
        "apply_phrase_rescore": apply_phrase_rescore,
        "DeepResult": impl.DeepResult,
        "prepare_rows": prepare_rows,
        "_prepared_cache_key": _prepared_cache_key,
        "load_prepared_cache": load_prepared_cache,
        "save_prepared_cache": save_prepared_cache,
    }
    originals = {name: getattr(core, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(core, name, value)
        sys.argv = [original_argv[0], *cleaned]
        return core.main()
    finally:
        sys.argv = original_argv
        for name, value in originals.items():
            setattr(core, name, value)
        _clear_order_side_tables()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
