#!/usr/bin/env python3
"""Small repeatable hot-path probe for ranking-performance changes.

This is deliberately not a hard timing gate: hosted-runner noise is too large for
that. It prints stable workloads and checksums so PRs can compare before/after
wall time without changing ranking semantics.
"""

from __future__ import annotations

import time
from collections.abc import Callable

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


def _measure(label: str, workload: Callable[[], float], *, samples: int = 3) -> tuple[float, float]:
    durations: list[float] = []
    checksum = 0.0
    for _ in range(samples):
        start = time.perf_counter()
        checksum = workload()
        durations.append(time.perf_counter() - start)
    best = min(durations)
    median = sorted(durations)[len(durations) // 2]
    print(
        f"PERF {label:<18} best={best:.6f}s median={median:.6f}s "
        f"checksum={checksum:.6f}"
    )
    return best, checksum


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
    wn_dir,
    lex: core.WordNetLexicon,
    *,
    workers: int,
    batch_size: int,
) -> tuple[float, float]:
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
    checksum = sum(
        row.final + row.grammar_raw + row.structure_norm + row.valency_norm
        for row in rows
    ) + stats["orders"]
    rerank._clear_order_side_tables()
    return elapsed, checksum


def main() -> int:
    wn_dir = core.ensure_wordnet(core.DEFAULT_WORDNET_DIR)
    lex = core.WordNetLexicon.load(wn_dir)

    # Warm ordinary feature lookup so the frame probe isolates repeated verb
    # lemma/frame derivation rather than first-use POS feature construction.
    for word in set(FRAME_WORDS) | set(FUNCTION_WORDS):
        lex.features(word)

    def frame_work() -> float:
        total = 0
        for _ in range(2_000):
            for word in FRAME_WORDS:
                total += len(lex.frames_for(word))
        return float(total)

    def function_work() -> float:
        total = 0
        for _ in range(25_000):
            for word in FUNCTION_WORDS:
                total += len(core.function_class(word) or "")
        return float(total)

    def ordering_work() -> float:
        total = 0.0
        for _ in range(3):
            for words in ORDER_BAGS:
                candidates, evaluated = rerank.rank_orders(
                    words,
                    lex,
                    order_mode="exact",
                    exact_max_words=5,
                    top_k=8,
                )
                total += evaluated
                if candidates:
                    total += candidates[0].objective
        return total

    _measure("verb-frames", frame_work)
    _measure("function-class", function_work)
    _measure("exact-ordering", ordering_work)

    # Two workers matches GitHub's common hosted-runner CPU allocation. This is
    # a batch-overhead probe, not a claim about the best worker count on a user's
    # desktop. Keep the production worker default unchanged unless a broader
    # machine matrix justifies changing it.
    serial_seconds, serial_checksum = _deep_work(
        wn_dir, lex, workers=1, batch_size=32
    )
    print(
        f"PERF {'deep-serial':<18} best={serial_seconds:.6f}s "
        f"median={serial_seconds:.6f}s checksum={serial_checksum:.6f}"
    )
    for batch_size in (8, 32, 96):
        seconds, checksum = _deep_work(
            wn_dir, lex, workers=2, batch_size=batch_size
        )
        print(
            f"PERF {f'deep-p2-b{batch_size}':<18} best={seconds:.6f}s "
            f"median={seconds:.6f}s checksum={checksum:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
