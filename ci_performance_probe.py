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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
