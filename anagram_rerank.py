#!/usr/bin/env python3
"""Active AnagramSolver front-end with scoped top-K and safety adapters."""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import multiprocessing
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import anagram_rerank_core as core
from anagram_auxiliary_grammar import (
    local_grammar_raw_with_auxiliaries,
    order_local_tables_with_auxiliaries,
    phrase_structure_with_auxiliaries,
)
from anagram_order_diversity import raw_pool_size, select_diverse_orders
from anagram_performance import FastPhraseIndex, FastWordNetLexicon, performance_hooks

if TYPE_CHECKING:
    from anagram_rerank_topk_impl import OrderCandidate

# Explicit aliases keep the facade contract visible to Ruff/Pylance while using
# optimized per-instance classes without mutating the stable core on import.
Row = core.Row
WordNetLexicon = FastWordNetLexicon
PositiveBigramModel = core.PositiveBigramModel
PhraseIndex = FastPhraseIndex
ensure_wordnet = core.ensure_wordnet
DEFAULT_WORDNET_DIR = core.DEFAULT_WORDNET_DIR

# Capture the stable core function that facade preparation delegates to only for
# formula ownership tests/reference; normal optimized preparation below calls the
# same public core scoring helpers directly so their formulas stay authoritative.
_CORE_PREPARE_ROWS = core.prepare_rows

# The top-K implementation predates this facade and historically patched core at
# import time. Capture and restore the frozen core API immediately so merely
# importing ``anagram_rerank`` remains side-effect free.
_CORE_HOOK_NAMES = (
    "best_order",
    "deep_analyze",
    "apply_phrase_rescore",
    "DeepResult",
)
_core_before_impl = {name: getattr(core, name) for name in _CORE_HOOK_NAMES}
impl = importlib.import_module("anagram_rerank_topk_impl")
for _name, _value in _core_before_impl.items():
    setattr(core, _name, _value)

# Worker initializers resolve these names from the implementation module at run
# time, so point them at the optimized per-instance classes without touching core.
setattr(impl, "WordNetLexicon", FastWordNetLexicon)  # noqa: B010
setattr(impl, "PhraseIndex", FastPhraseIndex)  # noqa: B010

# Preserve the benchmark-facing API exposed by the implementation.
for _name in dir(impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(impl, _name)

# Keep the original implementation hooks stable across importlib.reload().
# The ordering-layer tests intentionally reload this facade, and spawned worker
# processes import it independently, so repeatedly wrapping an already-wrapped
# function would otherwise build a recursion matryoshka doll.
if not hasattr(impl, "_AUX_BASE_PHRASE_STRUCTURE"):
    setattr(impl, "_AUX_BASE_PHRASE_STRUCTURE", impl.phrase_structure)  # noqa: B010
if not hasattr(impl, "_AUX_BASE_ORDER_LOCAL_TABLES"):
    setattr(impl, "_AUX_BASE_ORDER_LOCAL_TABLES", impl._order_local_tables)  # noqa: B010
if not hasattr(impl, "_AUX_BASE_WORKER_INIT"):
    setattr(impl, "_AUX_BASE_WORKER_INIT", impl._worker_init)  # noqa: B010
if not hasattr(impl, "_PERF_BASE_WORKER_ANALYZE_BATCH"):
    setattr(impl, "_PERF_BASE_WORKER_ANALYZE_BATCH", impl._worker_analyze_batch)  # noqa: B010

_BASE_PHRASE_STRUCTURE = impl._AUX_BASE_PHRASE_STRUCTURE
_BASE_ORDER_LOCAL_TABLES = impl._AUX_BASE_ORDER_LOCAL_TABLES
_BASE_WORKER_INIT = impl._AUX_BASE_WORKER_INIT
_BASE_WORKER_ANALYZE_BATCH = impl._PERF_BASE_WORKER_ANALYZE_BATCH

# Keep direct references to the implementation functions before this facade
# shadows their public names with diversity-aware wrappers.
_BASE_RANK_ORDERS = impl.rank_orders
_BASE_DEEP_ANALYZE = impl.deep_analyze

ENGINE_LAYER = "diverse-top-k-order-reranking"
PREPARED_CACHE_SCHEMA = "topk-prepared-json-gzip-2"
PREPARED_CACHE_COMPRESSLEVEL = 1
DEFAULT_ORDER_CANDIDATES = 56
# Direct facade callers should see the facade default too; ``main`` may still
# override this for an explicit --order-candidates value and restores it after.
setattr(impl, "_ORDER_CANDIDATE_COUNT", DEFAULT_ORDER_CANDIDATES)  # noqa: B010

# The legacy implementation still exposes its worker width through a module
# global. Normal CLI use is one deep-analysis call per process, but serializing
# facade calls makes that compatibility bridge safe for accidental same-process
# concurrent callers until the legacy layer can accept the width explicitly.
_DEEP_ANALYZE_LOCK = threading.RLock()

# Preparation cache limits are deliberately finite. The hard 100k-bag smoke has
# only 743 surface words (at most ~551k directed pairs), so a 512k pair window
# preserves almost all useful reuse while preventing unlimited exports from
# retaining millions of tuple keys. Eviction only causes recomputation.
_PREPARE_PAIR_CACHE_LIMIT = 524_288
_PREPARE_WORD_CACHE_LIMIT = 8_192
_PREPARE_LOCK = threading.RLock()


def phrase_structure(
    words: Sequence[str],
    lex: WordNetLexicon,
) -> core.StructureResult:
    """Combine the legacy parser with explicit auxiliary-chain structures."""
    with performance_hooks():
        return phrase_structure_with_auxiliaries(words, lex, _BASE_PHRASE_STRUCTURE)


def _order_local_tables(
    words: tuple[str, ...],
    lex: WordNetLexicon,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """Add bounded local evidence for BE/HAVE morphology transitions."""
    with performance_hooks():
        return order_local_tables_with_auxiliaries(words, lex, _BASE_ORDER_LOCAL_TABLES)


def local_grammar_raw(words: Sequence[str], lex: WordNetLexicon) -> float:
    """Local grammar score using the same auxiliary-aware pair table as search."""
    with performance_hooks():
        return local_grammar_raw_with_auxiliaries(words, lex, _BASE_ORDER_LOCAL_TABLES)


def _install_auxiliary_scoring() -> None:
    setattr(impl, "phrase_structure", phrase_structure)  # noqa: B010
    setattr(impl, "_order_local_tables", _order_local_tables)  # noqa: B010
    setattr(impl, "local_grammar_raw", local_grammar_raw)  # noqa: B010


def _worker_init_with_auxiliary_scoring(
    wordnet_dir: str,
    order_mode: str,
    beam_width: int,
    exact_max_words: int,
    order_candidates: int,
) -> None:
    """Initialize a spawned worker, then install the facade's grammar hooks."""
    _BASE_WORKER_INIT(
        wordnet_dir,
        order_mode,
        beam_width,
        exact_max_words,
        order_candidates,
    )
    _install_auxiliary_scoring()


def _worker_analyze_batch_with_performance_hooks(
    batch: tuple[tuple[int, tuple[str, ...]], ...],
) -> object:
    """Give each spawned/thread worker the same scoped fast core adapters."""
    with performance_hooks():
        return _BASE_WORKER_ANALYZE_BATCH(batch)


_install_auxiliary_scoring()
setattr(impl, "_worker_init", _worker_init_with_auxiliary_scoring)  # noqa: B010
setattr(impl, "_worker_analyze_batch", _worker_analyze_batch_with_performance_hooks)  # noqa: B010


def _prepared_cache_key(input_path: Path, wordnet_dir: Path) -> str:
    """Fingerprint prepared caches by content, independent of project location."""
    h = hashlib.sha256()
    h.update(PREPARED_CACHE_SCHEMA.encode("ascii"))
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
        "pre_score": row.pre_score,
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
    "pre_score": (0.0, 100.0),
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

    numeric: dict[str, float] = {}
    for name in _CACHE_FLOAT_RANGES:
        value = _bounded_number(item, name)
        if value is None:
            return None
        numeric[name] = value

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
        pre_score=numeric["pre_score"],
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
    with gzip.open(
        tmp, "wt", encoding="utf-8", compresslevel=PREPARED_CACHE_COMPRESSLEVEL
    ) as handle:
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
    """Prepare rows with bounded memoization and core-owned scoring formulas."""
    for row in rows:
        row.words = tuple(sorted(row.words))
    rows.sort(key=lambda row: (row.word_count, row.words))

    recognized: dict[str, bool] = {}
    family_words: dict[str, str] = {}
    pair_scores: dict[tuple[str, str], float] = {}

    with _PREPARE_LOCK, performance_hooks():
        original_pair_grammar = core.pair_grammar
        owner_thread = threading.get_ident()

        def cached_pair_grammar(
            left: str,
            right: str,
            call_lex: core.WordNetLexicon,
        ) -> float:
            # The monkeypatch is process-global, so unrelated callers must retain
            # exact core behavior. Only this serialized preparation call gets the
            # local cache; everyone else delegates immediately.
            if threading.get_ident() != owner_thread or call_lex is not lex:
                return original_pair_grammar(left, right, call_lex)

            key = (left, right)
            cached = pair_scores.get(key)
            if cached is not None:
                return cached
            score = original_pair_grammar(left, right, call_lex)
            if _PREPARE_PAIR_CACHE_LIMIT > 0:
                if len(pair_scores) >= _PREPARE_PAIR_CACHE_LIMIT:
                    pair_scores.pop(next(iter(pair_scores)))
                pair_scores[key] = score
            return score

        core.pair_grammar = cached_pair_grammar
        try:
            for idx, row in enumerate(rows, 1):
                recognized_count = 0
                family: list[str] = []
                for word in row.words:
                    known = recognized.get(word)
                    if known is None:
                        known = lex.features(word).recognized
                        if _PREPARE_WORD_CACHE_LIMIT > 0:
                            if len(recognized) >= _PREPARE_WORD_CACHE_LIMIT:
                                recognized.pop(next(iter(recognized)))
                            recognized[word] = known
                    if known:
                        recognized_count += 1

                    family_word = family_words.get(word)
                    if family_word is None:
                        family_word = core.morphology_family_word(word, lex)
                        if _PREPARE_WORD_CACHE_LIMIT > 0:
                            if len(family_words) >= _PREPARE_WORD_CACHE_LIMIT:
                                family_words.pop(next(iter(family_words)))
                            family_words[word] = family_word
                    family.append(family_word)

                row.wn_coverage = (
                    recognized_count / len(row.words) if row.words else 0.0
                )
                # Core remains the single source of truth for the tunable grammar
                # aggregation; only pair_grammar underneath it is memoized.
                row.grammar_potential_norm = core.grammar_potential(row.words, lex)
                row.family_key = tuple(sorted(family))
                row.pre_score = core.score_pre(row)
                if idx % 25000 == 0:
                    print(f"  prepared {idx:,} / {len(rows):,}")
        finally:
            core.pair_grammar = original_pair_grammar


def rank_orders(
    words: Sequence[str],
    lex: WordNetLexicon,
    *,
    order_mode: str = "auto",
    beam_width: int = 128,
    exact_max_words: int = 5,
    top_k: int = DEFAULT_ORDER_CANDIDATES,
) -> tuple[tuple[OrderCandidate, ...], int]:
    """Rank a wider raw pool, then retain a quality-preserving diverse subset."""
    raw_k = raw_pool_size(top_k)
    with performance_hooks():
        candidates, evaluated = _BASE_RANK_ORDERS(
            words,
            lex,
            order_mode=order_mode,
            beam_width=beam_width,
            exact_max_words=exact_max_words,
            top_k=raw_k,
        )
    return select_diverse_orders(candidates, top_k), evaluated


def _diversify_order_side_tables(retained: int) -> None:
    """Replace worker-returned raw pools with the final diverse retained set."""
    for name in ("_ORDER_CANDIDATES_BY_ROW_ID", "_ORDER_CANDIDATES_BY_INDEX"):
        table = getattr(impl, name, None)
        if not isinstance(table, dict):
            continue
        for key, candidates in list(table.items()):
            table[key] = select_diverse_orders(candidates, retained)


def deep_analyze(
    rows: list[Row],
    selected: set[int],
    lex: WordNetLexicon,
    *,
    wordnet_dir: Path,
    backend: str,
    workers: int,
    batch_size: int,
    order_mode: str,
    beam_width: int,
    exact_max_words: int,
) -> dict[str, float]:
    """Collect a wider worker pool, then diversify it in the parent process.

    The frozen implementation communicates worker width through a module global.
    The compatibility lock makes that temporary mutation safe for concurrent
    same-process facade callers; spawned workers still receive the widened value
    through their normal initializer, and the original value is always restored.
    """
    with _DEEP_ANALYZE_LOCK, performance_hooks():
        retained = int(
            getattr(impl, "_ORDER_CANDIDATE_COUNT", DEFAULT_ORDER_CANDIDATES)
        )
        raw_k = raw_pool_size(retained)
        setattr(impl, "_ORDER_CANDIDATE_COUNT", raw_k)  # noqa: B010
        try:
            stats = _BASE_DEEP_ANALYZE(
                rows,
                selected,
                lex,
                wordnet_dir=wordnet_dir,
                backend=backend,
                workers=workers,
                batch_size=batch_size,
                order_mode=order_mode,
                beam_width=beam_width,
                exact_max_words=exact_max_words,
            )
            _diversify_order_side_tables(retained)
            return stats
        finally:
            setattr(impl, "_ORDER_CANDIDATE_COUNT", retained)  # noqa: B010


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
        with performance_hooks():
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
        original_argv[1:], "--order-candidates", DEFAULT_ORDER_CANDIDATES
    )
    if count < 1:
        raise SystemExit("--order-candidates must be >= 1")

    previous_count = int(
        getattr(impl, "_ORDER_CANDIDATE_COUNT", DEFAULT_ORDER_CANDIDATES)
    )
    # ``impl`` is intentionally loaded through importlib, so Pyright sees a
    # generic ModuleType. Keep the dynamic assignment explicit and scoped.
    setattr(impl, "_ORDER_CANDIDATE_COUNT", count)  # noqa: B010

    overrides = {
        "best_order": impl.best_order,
        "deep_analyze": deep_analyze,
        "apply_phrase_rescore": apply_phrase_rescore,
        "DeepResult": impl.DeepResult,
        "prepare_rows": prepare_rows,
        "_prepared_cache_key": _prepared_cache_key,
        "load_prepared_cache": load_prepared_cache,
        "save_prepared_cache": save_prepared_cache,
    }
    originals = {name: getattr(core, name) for name in overrides}
    try:
        with performance_hooks():
            for name, value in overrides.items():
                setattr(core, name, value)
            sys.argv = [original_argv[0], *cleaned]
            return core.main()
    finally:
        sys.argv = original_argv
        for name, value in originals.items():
            setattr(core, name, value)
        setattr(impl, "_ORDER_CANDIDATE_COUNT", previous_count)  # noqa: B010
        _clear_order_side_tables()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())