#!/usr/bin/env python3
"""Small repeatable hot-path probe for ranking-performance changes.

This is deliberately not a hard timing gate: hosted-runner noise is too large for
that. It prints stable workloads and output-sensitive digests so PRs can compare
before/after wall time without silently accepting ranking drift.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import anagram_rerank as rerank
import anagram_rerank_core as core

FRAME_WORDS = (
    "chased", "needs", "arrived", "speak", "stand", "fall", "testing",
    "tasting", "reads", "boils", "favors", "helps", "stopped", "runs",
)
FUNCTION_WORDS = (
    "the", "a", "than", "we", "they", "is", "are", "will", "have",
    "never", "of", "with", "and", "dog", "ball", "testing", "louder",
)
ORDER_BAGS = (
    ("actions", "speak", "louder", "than", "words"),
    ("united", "we", "stand", "divided", "fall"),
    ("i", "am", "testing", "anagrams"),
    ("the", "ball", "chased", "dog"),
    ("my", "phone", "needs", "charge"),
    ("the", "pot", "never", "boils"),
    ("fortune", "favors", "the", "bold"),
    ("a", "quiet", "room", "helps", "focus"),
)


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _measure(
    label: str,
    workload: Callable[[], object],
    *,
    samples: int = 3,
) -> tuple[float, str]:
    durations: list[float] = []
    digests: list[str] = []
    for _ in range(samples):
        start = time.perf_counter()
        result = workload()
        durations.append(time.perf_counter() - start)
        digests.append(_digest(result))

    if len(set(digests)) != 1:
        raise RuntimeError(f"{label} produced inconsistent sample digests: {digests}")

    best = min(durations)
    median = sorted(durations)[len(durations) // 2]
    digest = digests[0]
    print(
        f"PERF {label:<18} best={best:.6f}s median={median:.6f}s "
        f"digest={digest}"
    )
    return best, digest


def _probe_row(words: tuple[str, ...], rank: int) -> core.Row:
    return core.Row(
        words=tuple(sorted(words)),
        word_count=len(words),
        old_rank=rank,
        old_pre=50.0,
        lex=0.5,
        fam=0.5,
        old_pair=0.5,
        hint=0.0,
        zavg=0.0,
        zmin=0.0,
        old_pcov=0.5,
        hints=(),
        wn_coverage=1.0,
        grammar_potential_norm=0.5,
        pre_score=50.0,
    )


def _deep_work(
    wn_dir: Path,
    lex: core.WordNetLexicon,
    *,
    workers: int,
    batch_size: int,
) -> tuple[float, object]:
    rows = [
        _probe_row(ORDER_BAGS[i % len(ORDER_BAGS)], i + 1)
        for i in range(384)
    ]
    selected = set(range(len(rows)))
    start = time.perf_counter()
    stats = rerank.deep_analyze(
        rows,
        selected,
        lex,
        wordnet_dir=wn_dir,
        backend="process" if workers > 1 else "serial",
        workers=workers,
        batch_size=batch_size,
        order_mode="exact",
        beam_width=128,
        exact_max_words=5,
    )
    elapsed = time.perf_counter() - start
    canonical = {
        "orders": int(stats["orders"]),
        "rows": [
            {
                "order": row.best_order,
                "final": row.final,
                "grammar_raw": row.grammar_raw,
                "grammar_norm": row.grammar_norm,
                "structure": row.structure_norm,
                "valency": row.valency_norm,
                "coverage": row.syntax_coverage,
                "kind": row.phrase_kind,
            }
            for row in rows
        ],
    }
    rerank._clear_order_side_tables()
    return elapsed, canonical


def main() -> int:
    wn_dir = core.ensure_wordnet(core.DEFAULT_WORDNET_DIR)
    lex = core.WordNetLexicon.load(wn_dir)

    # Warm ordinary feature lookup so the frame probe isolates repeated verb
    # lemma/frame derivation rather than first-use POS feature construction.
    for word in set(FRAME_WORDS) | set(FUNCTION_WORDS):
        lex.features(word)

    def frame_work() -> object:
        observed: list[tuple[str, tuple[int, ...]]] = []
        for _ in range(2_000):
            observed = [
                (word, tuple(sorted(lex.frames_for(word))))
                for word in FRAME_WORDS
            ]
        return observed

    def function_work() -> object:
        observed: list[tuple[str, str | None]] = []
        for _ in range(25_000):
            observed = [
                (word, core.function_class(word))
                for word in FUNCTION_WORDS
            ]
        return observed

    def ordering_work() -> object:
        observed: list[dict[str, object]] = []
        for _ in range(3):
            for words in ORDER_BAGS:
                candidates, evaluated = rerank.rank_orders(
                    words,
                    lex,
                    order_mode="exact",
                    exact_max_words=5,
                    top_k=8,
                )
                observed.append(
                    {
                        "bag": words,
                        "evaluated": evaluated,
                        "candidates": [
                            {
                                "order": candidate.order,
                                "grammar_raw": candidate.grammar_raw,
                                "grammar_norm": candidate.grammar_norm,
                                "structure": candidate.structure_norm,
                                "valency": candidate.valency_norm,
                                "coverage": candidate.syntax_coverage,
                                "kind": candidate.phrase_kind,
                                "objective": candidate.objective,
                            }
                            for candidate in candidates
                        ],
                    }
                )
        return observed

    _measure("verb-frames", frame_work)
    _measure("function-class", function_work)
    _measure("exact-ordering", ordering_work)

    # Two workers matches GitHub's common hosted-runner CPU allocation. This is
    # a batch-overhead probe, not a claim about the best worker count on a user's
    # desktop. Keep the production worker default unchanged unless a broader
    # machine matrix justifies changing it.
    deep_digests: dict[str, str] = {}
    serial_seconds, serial_output = _deep_work(
        wn_dir, lex, workers=1, batch_size=32
    )
    deep_digests["serial"] = _digest(serial_output)
    print(
        f"PERF {'deep-serial':<18} best={serial_seconds:.6f}s "
        f"median={serial_seconds:.6f}s digest={deep_digests['serial']}"
    )
    for batch_size in (8, 32, 96):
        seconds, output = _deep_work(
            wn_dir, lex, workers=2, batch_size=batch_size
        )
        digest = _digest(output)
        deep_digests[f"p2-b{batch_size}"] = digest
        print(
            f"PERF {f'deep-p2-b{batch_size}':<18} best={seconds:.6f}s "
            f"median={seconds:.6f}s digest={digest}"
        )

    if len(set(deep_digests.values())) != 1:
        raise RuntimeError(f"deep-analysis configurations disagree: {deep_digests}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
