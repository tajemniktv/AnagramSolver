#!/usr/bin/env python3
"""Run corpus benchmark scenarios and publish a compact GitHub Step Summary."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / ".ci_benchmark_results"


@dataclass(slots=True)
class Scenario:
    name: str
    slug: str
    phrase_db: Path | None


def run_logged(label: str, cmd: list[str], log_path: Path) -> str:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=HERE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    rc = proc.wait()
    text = "".join(lines)
    log_path.write_text(text, encoding="utf-8")
    if rc != 0:
        raise SystemExit(f"{label} failed with exit code {rc}; see {log_path}")
    return text


def section(text: str, heading: str, next_headings: tuple[str, ...]) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    ends = [text.find(h, start) for h in next_headings]
    ends = [x for x in ends if x >= 0]
    end = min(ends) if ends else len(text)
    return text[start:end]


def metric(block: str, label: str) -> float | None:
    match = re.search(
        rf"^\s*{re.escape(label)}\s+([+-]?\d+(?:\.\d+)?)\s*$",
        block,
        re.MULTILINE,
    )
    return float(match.group(1)) if match else None


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def order_metrics(text: str) -> dict[str, float | None]:
    exact = section(
        text,
        "Exact-order metrics (<=6 words):",
        ("By category", "Beam-only", "=== PHRASE-AWARE", "Suite wall time"),
    )
    retained = section(
        text,
        "Retained grammar metrics (same top-K):",
        ("Phrase-aware retained-order metrics:",),
    )
    phrase = section(
        text,
        "Phrase-aware retained-order metrics:",
        ("A/B delta on identical retained candidates:", "Suite wall time"),
    )
    delta = section(
        text,
        "A/B delta on identical retained candidates:",
        ("Suite wall time",),
    )
    return {
        "grammar_r1": metric(exact, "Recall@1"),
        "grammar_r10": metric(exact, "Recall@10"),
        "grammar_mrr": metric(exact, "MRR"),
        "retained_r1": metric(retained, "Recall@1"),
        "phrase_r1": metric(phrase, "Recall@1"),
        "phrase_r10": metric(phrase, "Recall@10"),
        "phrase_mrr": metric(phrase, "MRR"),
        "delta_r1": metric(delta, "Recall@1"),
        "delta_mrr": metric(delta, "MRR"),
    }


def full_metrics(text: str) -> dict[str, float | None]:
    bag = section(
        text,
        "Correct-word-bag ranking:",
        ("End-to-end exact phrase surfaced:",),
    )
    exact = section(
        text,
        "End-to-end exact phrase surfaced:",
        ("NOTE:",),
    )
    return {
        "bag_r10": metric(bag, "BagRecall@10"),
        "bag_mrr": metric(bag, "BagMRR"),
        "exact_r1": metric(exact, "ExactRecall@1"),
        "exact_r10": metric(exact, "ExactRecall@10"),
        "exact_mrr": metric(exact, "ExactMRR"),
    }


def _require_metrics(
    label: str,
    metrics: dict[str, float | None],
    required: tuple[str, ...],
) -> None:
    missing = [name for name in required if metrics.get(name) is None]
    if missing:
        raise RuntimeError(
            f"{label} output is missing required metrics: {', '.join(missing)}"
        )


def validate_metrics(
    scenario: Scenario,
    order: dict[str, float | None],
    full: dict[str, float | None] | None,
) -> None:
    """Treat benchmark output format drift as a CI failure, not a blank cell."""
    order_required = ("grammar_r1", "grammar_r10", "grammar_mrr")
    if scenario.phrase_db is not None:
        order_required += (
            "retained_r1",
            "phrase_r1",
            "phrase_r10",
            "phrase_mrr",
            "delta_r1",
            "delta_mrr",
        )
    _require_metrics(f"{scenario.name} ordering benchmark", order, order_required)

    if full is not None:
        _require_metrics(
            f"{scenario.name} full benchmark",
            full,
            ("bag_r10", "bag_mrr", "exact_r1", "exact_r10", "exact_mrr"),
        )


def append_summary(
    rows: list[
        tuple[Scenario, dict[str, float | None], dict[str, float | None] | None]
    ],
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    include_full = any(full is not None for _, _, full in rows)
    out = Path(summary_path)
    with out.open("a", encoding="utf-8") as f:
        f.write("## Anagram corpus benchmark matrix\n\n")
        f.write(
            "The same code and benchmark cases are evaluated with no phrase DB, "
            "Wiktionary titles, and Wiktionary+Wikipedia titles. Phrase A/B deltas "
            "compare the identical retained top-K orders. Exact <=6w R@1 is a separate "
            "exhaustive-permutation metric over only cases with at most six words.\n\n"
        )

        if include_full:
            f.write(
                "| Corpus | Exact <=6w R@1 | Retained G R@1 | Phrase R@1 | Δ R@1 | "
                "Phrase MRR | Full Bag R@10 | Full Exact R@10 | Full Exact MRR |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for scenario, order, full in rows:
                assert full is not None
                f.write(
                    f"| {scenario.name} | {fmt(order['grammar_r1'])} | "
                    f"{fmt(order['retained_r1'])} | {fmt(order['phrase_r1'])} | "
                    f"{fmt(order['delta_r1'])} | {fmt(order['phrase_mrr'])} | "
                    f"{fmt(full['bag_r10'])} | {fmt(full['exact_r10'])} | "
                    f"{fmt(full['exact_mrr'])} |\n"
                )
        else:
            f.write(
                "PR fast path: full candidate generation/reranking is intentionally "
                "omitted; main/manual runs retain the end-to-end matrix.\n\n"
            )
            f.write(
                "| Corpus | Exact <=6w R@1 | Retained G R@1 | Phrase R@1 | Δ R@1 | "
                "Phrase MRR |\n"
            )
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for scenario, order, _ in rows:
                f.write(
                    f"| {scenario.name} | {fmt(order['grammar_r1'])} | "
                    f"{fmt(order['retained_r1'])} | {fmt(order['phrase_r1'])} | "
                    f"{fmt(order['delta_r1'])} | {fmt(order['phrase_mrr'])} |\n"
                )

        f.write("\n")
        f.write(
            "This job is currently an execution/reporting check, not a phrase-quality gate. "
            "Existing ordering-regression thresholds remain the blocking quality guard.\n"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiktionary-db", type=Path, required=True)
    ap.add_argument("--wikipedia-db", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--order-only",
        action="store_true",
        help="Run only the fast ordering A/B and skip full candidate generation/reranking",
    )
    args = ap.parse_args()

    # Resolve relative paths against the caller's working directory before child
    # benchmarks switch cwd to HERE, and fail fast before running the baseline.
    wiktionary_db = args.wiktionary_db.expanduser().resolve()
    wikipedia_db = args.wikipedia_db.expanduser().resolve()
    for label, path in (
        ("Wiktionary phrase DB", wiktionary_db),
        ("Wiktionary + Wikipedia phrase DB", wikipedia_db),
    ):
        if not path.is_file():
            ap.error(f"{label} does not exist or is not a file: {path}")

    scenarios = [
        Scenario("Baseline (no phrase DB)", "baseline", None),
        Scenario("Wiktionary", "wiktionary", wiktionary_db),
        Scenario("Wiktionary + Wikipedia", "wiktionary-wikipedia", wikipedia_db),
    ]

    rows: list[
        tuple[Scenario, dict[str, float | None], dict[str, float | None] | None]
    ] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        phrase_args = (
            ["--phrase-db", str(scenario.phrase_db)] if scenario.phrase_db else []
        )
        order_cmd = [
            sys.executable,
            "anagram_benchmark.py",
            "--mode",
            "order",
            *phrase_args,
        ]

        order_text = run_logged(
            f"{scenario.name}: ordering",
            order_cmd,
            RESULTS_DIR / f"{scenario.slug}-order.txt",
        )
        order = order_metrics(order_text)

        full: dict[str, float | None] | None = None
        if not args.order_only:
            full_cmd = [
                sys.executable,
                "anagram_benchmark.py",
                "--mode",
                "full",
                "--workers",
                str(args.workers),
                *phrase_args,
            ]
            full_text = run_logged(
                f"{scenario.name}: full pipeline",
                full_cmd,
                RESULTS_DIR / f"{scenario.slug}-full.txt",
            )
            full = full_metrics(full_text)

        validate_metrics(scenario, order, full)
        rows.append((scenario, order, full))

    print("\n=== CORPUS MATRIX SUMMARY ===")
    for scenario, order, full in rows:
        line = (
            f"{scenario.name:<28} exact6wR1={fmt(order['grammar_r1'])} "
            f"phraseR1={fmt(order['phrase_r1'])} deltaR1={fmt(order['delta_r1'])}"
        )
        if full is not None:
            line += (
                f" bagR10={fmt(full['bag_r10'])} "
                f"exactR10={fmt(full['exact_r10'])}"
            )
        print(line)

    append_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())