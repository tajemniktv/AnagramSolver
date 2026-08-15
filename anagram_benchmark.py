#!/usr/bin/env python3
"""
anagram_benchmark.py

Regression/generalization harness for the Python anagram solver.

Modes
-----
order (default)
    Fast. Given each known phrase's unordered word bag, rank all word orders
    (exactly for <=6 words) using the selected reranker's grammar/structure objective.
    Reports Top-1/10/50, MRR, and the chosen order.

full
    Slow on first run, cached afterward. Uses multi_anagram_v8_analyzer.py to
    generate exact same-letter word bags for cases marked "full": true, then
    invokes the selected reranker and records PRE/FINAL ranks.

Both modes deliberately keep the answer OUT of scoring. The answer is used only
after ranking to measure where it landed.

Examples
--------
python anagram_benchmark.py --mode order
python anagram_benchmark.py --mode full --workers 8
python anagram_benchmark.py --mode order --case better_late --case shakira_control
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "anagram_benchmarks.json"
DEFAULT_RERANKER = HERE / "anagram_rerank.py"
DEFAULT_V8 = HERE / "multi_anagram_v8_analyzer.py"
DEFAULT_CACHE = HERE / ".anagram_bench_cache"


def norm_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(
        norm_token(x)
        for x in re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
        if norm_token(x)
    )


def phrase_key(text_or_words: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(text_or_words, str):
        return tokens(text_or_words)
    return tuple(norm_token(x) for x in text_or_words if norm_token(x))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(slots=True)
class OrderResult:
    case_id: str
    answer: str
    category: str
    word_count: int
    exact_rank: int | None
    total_orders: int | None
    best_order: str
    exact_best: bool
    answer_objective: float | None
    best_objective: float


def _objective(v12, order: tuple[str, ...], lex) -> float:
    local_raw = v12.local_grammar_raw(order, lex)
    local_norm = v12.grammar_normalize(local_raw)
    structure = v12.phrase_structure(order, lex)
    return (
        0.38 * local_norm
        + 0.44 * structure.norm
        + 0.12 * structure.valency
        + 0.06 * structure.coverage
    )


def run_order_case(v12, lex, case: dict) -> OrderResult:
    answer = str(case["answer"])
    bag = tokens(answer)
    acceptable = {
        phrase_key(x)
        for x in case.get("acceptable_orders", [answer])
    }

    # The reranker's own selected order.
    _, best, _, _ = v12.best_order(
        bag,
        lex,
        order_mode="exact" if len(bag) <= 6 else "beam",
        beam_width=256,
        exact_max_words=6,
    )
    best_key = phrase_key(best)
    best_obj = _objective(v12, best_key, lex)

    if len(bag) > 6:
        return OrderResult(
            case_id=str(case["id"]),
            answer=answer,
            category=str(case.get("category", "uncategorized")),
            word_count=len(bag),
            exact_rank=None,
            total_orders=None,
            best_order=" ".join(best),
            exact_best=best_key in acceptable,
            answer_objective=None,
            best_objective=best_obj,
        )

    # Exact unique phrase-order ranking. Repeated words generate duplicate
    # position permutations, so dedupe by the realized token sequence.
    pair, starts, ends = v12._order_local_tables(tuple(bag), lex)
    scored: dict[tuple[str, ...], float] = {}
    index_to_word = tuple(bag)

    for idx_order in v12._exact_index_orders(len(bag)):
        realized = tuple(index_to_word[i] for i in idx_order)
        if realized in scored:
            continue

        local_raw = v12._local_raw_indices(idx_order, pair, starts, ends)
        local_norm = v12.grammar_normalize(local_raw)
        structure = v12.phrase_structure(realized, lex)
        obj = (
            0.38 * local_norm
            + 0.44 * structure.norm
            + 0.12 * structure.valency
            + 0.06 * structure.coverage
        )
        scored[realized] = obj

    ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    ranks = [
        i
        for i, (order, _) in enumerate(ranked, 1)
        if order in acceptable
    ]
    exact_rank = min(ranks) if ranks else None
    answer_obj = max(
        (score for order, score in ranked if order in acceptable),
        default=None,
    )

    return OrderResult(
        case_id=str(case["id"]),
        answer=answer,
        category=str(case.get("category", "uncategorized")),
        word_count=len(bag),
        exact_rank=exact_rank,
        total_orders=len(ranked),
        best_order=" ".join(best),
        exact_best=best_key in acceptable,
        answer_objective=answer_obj,
        best_objective=best_obj,
    )


def print_order_summary(results: list[OrderResult]) -> None:
    exact = [r for r in results if r.exact_rank is not None]
    beam = [r for r in results if r.exact_rank is None]

    print("\n=== ORDERING REGRESSION SUITE ===")
    for r in results:
        if r.exact_rank is None:
            status = "PASS" if r.exact_best else "MISS"
            rank_text = "beam"
        else:
            status = (
                "TOP1" if r.exact_rank == 1
                else "TOP10" if r.exact_rank <= 10
                else "TOP50" if r.exact_rank <= 50
                else "MISS"
            )
            rank_text = f"{r.exact_rank}/{r.total_orders}"

        print(
            f"{status:5}  {r.case_id:<24} [{r.category:<20}] "
            f"rank={rank_text:<10} best={r.best_order}"
        )

    if exact:
        n = len(exact)
        top1 = sum(r.exact_rank <= 1 for r in exact if r.exact_rank is not None) / n
        top10 = sum(r.exact_rank <= 10 for r in exact if r.exact_rank is not None) / n
        top50 = sum(r.exact_rank <= 50 for r in exact if r.exact_rank is not None) / n
        mrr = sum(1.0 / r.exact_rank for r in exact if r.exact_rank) / n
        median = sorted(r.exact_rank for r in exact if r.exact_rank)[n // 2]

        print("\nExact-order metrics (<=6 words):")
        print(f"  cases       {n}")
        print(f"  Recall@1    {top1:.3f}")
        print(f"  Recall@10   {top10:.3f}")
        print(f"  Recall@50   {top50:.3f}")
        print(f"  MRR         {mrr:.3f}")
        print(f"  median rank {median}")

        # Category breakdown catches regressions that an aggregate can hide.
        categories = sorted({r.category for r in exact})
        if len(categories) > 1:
            print("\nBy category (exact-order cases):")
            for category in categories:
                group = [r for r in exact if r.category == category]
                g_n = len(group)
                g_top10 = sum(r.exact_rank is not None and r.exact_rank <= 10 for r in group) / g_n
                g_top50 = sum(r.exact_rank is not None and r.exact_rank <= 50 for r in group) / g_n
                g_mrr = sum(1.0 / r.exact_rank for r in group if r.exact_rank) / g_n
                print(
                    f"  {category:<24} n={g_n:<3} "
                    f"R@10={g_top10:.3f} R@50={g_top50:.3f} MRR={g_mrr:.3f}"
                )

    if beam:
        passed = sum(r.exact_best for r in beam)
        print(
            f"Beam-only (>6 words): {passed}/{len(beam)} exact expected order(s) selected."
        )


FINAL_RE = re.compile(r"V\d+ FINAL rank:\s+([\d,]+)\s+/\s+([\d,]+)")
PRE_RE = re.compile(r"V\d+ PRE rank:\s+([\d,]+)\s+/\s+([\d,]+)")
NOT_FOUND_RE = re.compile(r"NOT FOUND in input export")
BEST_ORDER_RE = re.compile(r"best order:\s+(.+?)\s*$", re.MULTILINE | re.IGNORECASE)


@dataclass(slots=True)
class FullResult:
    case_id: str
    answer: str
    generated: bool
    pre_rank: int | None
    final_rank: int | None
    final_total: int | None
    selected_order: str | None
    exact_order: bool
    seconds: float
    note: str = ""


def make_v8_command(
    case: dict,
    v8: Path,
    export: Path,
) -> list[str]:
    answer = str(case["answer"])
    words = tokens(answer)
    target = "".join(words)
    unique_words = len(set(words)) == len(words)
    min_word_len = int(
        case.get(
            "min_word_len",
            min(2 if any(len(w) <= 2 for w in words) else 3, min(map(len, words))),
        )
    )
    min_zipf = float(case.get("min_zipf", 2.7))
    max_results = int(case.get("max_results", 100000))

    cmd = [
        sys.executable,
        str(v8),
        target,
        "--min-word-len", str(min_word_len),
        "--min-words", str(len(words)),
        "--max-words", str(len(words)),
        "--min-zipf", str(min_zipf),
        "--short-word-policy", "common",
        "--show-components",
        "--export", str(export),
        "--top-per-group", "1",
    ]

    if unique_words:
        cmd.append("--no-repeat")

    hints = case.get("hints", [])
    if hints:
        cmd += ["--contains-any", ",".join(map(str, hints))]

    excludes = case.get("exclude", [])
    if excludes:
        cmd += ["--exclude", ",".join(map(str, excludes))]

    if max_results == 0:
        cmd.append("--all-results")
    else:
        cmd += ["--max-results", str(max_results)]

    return cmd


def run_full_case(
    case: dict,
    *,
    v8: Path,
    reranker: Path,
    cache_dir: Path,
    workers: int,
    rebuild: bool,
) -> FullResult:
    case_id = str(case["id"])
    answer = str(case["answer"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    export = cache_dir / f"{slug(case_id)}_v8_ALL.txt"

    t0 = time.perf_counter()
    generated = False

    if rebuild or not export.exists():
        generated = True
        cmd = make_v8_command(case, v8, export)
        print(f"\n[{case_id}] generating exact candidate bags ...")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            return FullResult(
                case_id, answer, True, None, None, None, None, False,
                time.perf_counter() - t0,
                note=f"generator failed ({proc.returncode}): {proc.stderr[-500:]}",
            )

    print(f"[{case_id}] reranking {export.name} ...")
    cmd = [
        sys.executable,
        str(reranker),
        str(export),
        "--benchmark-answer", answer,
        "--workers", str(workers),
        "--backend", "auto",
        "--deep-per-group", str(int(case.get("deep_per_group", 5000))),
        "--beam-width", "128",
        "--phrase-rescore-top", "300",
        "--top-per-group", "1",
        "--export", str(cache_dir / f"{slug(case_id)}_reranked.txt"),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    elapsed = time.perf_counter() - t0

    if proc.returncode != 0:
        return FullResult(
            case_id, answer, generated, None, None, None, None, False, elapsed,
            note=f"reranker failed ({proc.returncode}): {proc.stderr[-500:]}",
        )

    text = proc.stdout + "\n" + proc.stderr
    if NOT_FOUND_RE.search(text):
        return FullResult(
            case_id, answer, generated, None, None, None, None, False, elapsed,
            note="answer not generated / filtered out",
        )

    pre_m = PRE_RE.search(text)
    final_m = FINAL_RE.search(text)
    pre_rank = int(pre_m.group(1).replace(",", "")) if pre_m else None
    final_rank = int(final_m.group(1).replace(",", "")) if final_m else None
    final_total = int(final_m.group(2).replace(",", "")) if final_m else None

    best_m = BEST_ORDER_RE.search(text)
    selected_order = best_m.group(1).strip() if best_m else None

    def _norm_phrase(value: str) -> tuple[str, ...]:
        return tokens(value)

    exact_order = (
        selected_order is not None
        and _norm_phrase(selected_order) == _norm_phrase(answer)
    )

    return FullResult(
        case_id,
        answer,
        generated,
        pre_rank,
        final_rank,
        final_total,
        selected_order,
        exact_order,
        elapsed,
    )


def print_full_summary(results: list[FullResult]) -> None:
    print("\n=== FULL ANAGRAM REGRESSION SUITE ===")
    for r in results:
        if r.final_rank is None:
            bag_status = "MISS"
            rank_text = "-"
        else:
            bag_status = (
                "TOP1" if r.final_rank == 1
                else "TOP10" if r.final_rank <= 10
                else "TOP50" if r.final_rank <= 50
                else "TOP100" if r.final_rank <= 100
                else "MISS"
            )
            rank_text = f"{r.final_rank}/{r.final_total}"

        exact_mark = "EXACT" if r.exact_order else "ORDER!"
        cache_text = "generated" if r.generated else "cached"
        selected = r.selected_order or "-"

        print(
            f"{bag_status:6} {exact_mark:6} {r.case_id:<24} "
            f"PRE={str(r.pre_rank):<7} BAG={rank_text:<14} "
            f"{r.seconds:7.2f}s {cache_text}  selected={selected}"
            + (f"  {r.note}" if r.note else "")
        )

    found = [r for r in results if r.final_rank is not None]
    total = len(results)
    if not total:
        return

    def bag_recall(k: int) -> float:
        return sum(
            r.final_rank is not None and r.final_rank <= k
            for r in results
        ) / total

    def exact_recall(k: int) -> float:
        return sum(
            r.final_rank is not None
            and r.final_rank <= k
            and r.exact_order
            for r in results
        ) / total

    bag_mrr = sum(
        1.0 / r.final_rank for r in found if r.final_rank
    ) / total

    exact_mrr = sum(
        (1.0 / r.final_rank)
        for r in found
        if r.final_rank and r.exact_order
    ) / total

    exact_selected = sum(r.exact_order for r in results)

    print("\nFull-pipeline metrics:")
    print(f"  cases             {total}")
    print(f"  generated         {len(found)}/{total} correct word bags survived generation")
    print(f"  exact ordering    {exact_selected}/{total} target bags were ordered exactly")
    print()
    print("  Correct-word-bag ranking:")
    print(f"    BagRecall@1     {bag_recall(1):.3f}")
    print(f"    BagRecall@10    {bag_recall(10):.3f}")
    print(f"    BagRecall@50    {bag_recall(50):.3f}")
    print(f"    BagRecall@100   {bag_recall(100):.3f}")
    print(f"    BagMRR          {bag_mrr:.3f}")
    print()
    print("  End-to-end exact phrase surfaced:")
    print(f"    ExactRecall@1   {exact_recall(1):.3f}")
    print(f"    ExactRecall@10  {exact_recall(10):.3f}")
    print(f"    ExactRecall@50  {exact_recall(50):.3f}")
    print(f"    ExactRecall@100 {exact_recall(100):.3f}")
    print(f"    ExactMRR        {exact_mrr:.3f}")
    print(
        "\n  NOTE: BAG rank answers 'did we rank the correct set of words highly?'.\n"
        "        EXACT additionally requires the reranker to choose the intended word order."
    )



def load_cases(path: Path, selected_ids: set[str]) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = list(payload.get("cases", []))
    if selected_ids:
        cases = [c for c in cases if str(c["id"]) in selected_ids]
        missing = selected_ids - {str(c["id"]) for c in cases}
        if missing:
            raise SystemExit("Unknown case id(s): " + ", ".join(sorted(missing)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generalization/regression suite for the anagram solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mode", choices=("order", "full"), default="order")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--case", action="append", default=[], dest="case_ids")
    ap.add_argument(
        "--reranker",
        "--v12",
        dest="reranker",
        type=Path,
        default=DEFAULT_RERANKER,
        help="Reranker module/script to benchmark (the --v12 alias is kept for old commands)",
    )
    ap.add_argument("--v8", type=Path, default=DEFAULT_V8)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    selected = set(args.case_ids)
    cases = load_cases(args.cases, selected)

    if args.mode == "order":
        if not args.reranker.is_file():
            raise SystemExit(f"Reranker not found: {args.reranker}")
        v12 = load_module(args.reranker, "anagram_reranker_bench")
        wn_dir = v12.ensure_wordnet(v12.DEFAULT_WORDNET_DIR)
        print(f"Loading WordNet from {wn_dir} ...")
        lex = v12.WordNetLexicon.load(wn_dir)

        results = []
        t0 = time.perf_counter()
        for case in cases:
            result = run_order_case(v12, lex, case)
            results.append(result)
        print_order_summary(results)
        print(f"\nSuite wall time: {time.perf_counter() - t0:.2f}s")
        return 0

    # Full mode.
    if not args.v8.is_file():
        raise SystemExit(f"V8 generator not found: {args.v8}")
    if not args.reranker.is_file():
        raise SystemExit(f"Reranker not found: {args.reranker}")

    full_cases = [c for c in cases if bool(c.get("full", False))]
    results = [
        run_full_case(
            case,
            v8=args.v8,
            reranker=args.reranker,
            cache_dir=args.cache_dir,
            workers=args.workers,
            rebuild=args.rebuild,
        )
        for case in full_cases
    ]
    print_full_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
