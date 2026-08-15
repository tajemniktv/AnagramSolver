from __future__ import annotations

import re
from pathlib import Path

bench_path = Path("anagram_benchmark.py")
s = bench_path.read_text(encoding="utf-8")

# 1. Keep the existing PhraseOrderResult API, but add the grammar rank/order
# from the same retained top-K population so the A/B is truly comparable.
s = s.replace(
'''class PhraseOrderResult:\n    case_id: str\n    answer: str\n    category: str\n    retained_rank: int | None\n    retained_total: int\n    best_order: str\n    exact_best: bool\n    target_retained: bool\n    best_phrase_score: float\n''',
'''class PhraseOrderResult:\n    case_id: str\n    answer: str\n    category: str\n    retained_rank: int | None\n    retained_total: int\n    best_order: str\n    exact_best: bool\n    target_retained: bool\n    best_phrase_score: float\n    grammar_rank: int | None = None\n    grammar_best_order: str = ""\n''',
)

# 2. Replace the phrase A/B implementation wholesale. Both sides rank the same
# retained candidates with candidate.objective as the grammar base; phrase mode
# differs only by the positive phrase bonus.
start = s.index("def _candidate_final_order_component(")
end = s.index("\ndef print_order_summary(", start)
replacement = r'''def _rank_metrics(ranks: list[int | None]) -> dict[str, float]:
    """Metrics over a fixed case population; missing ranks count as misses."""
    if not ranks:
        return {}
    n = len(ranks)
    retained = sum(rank is not None for rank in ranks)
    return {
        "cases": float(n),
        "retained": float(retained),
        "retained_rate": retained / n,
        "recall1": sum(rank is not None and rank <= 1 for rank in ranks) / n,
        "recall10": sum(rank is not None and rank <= 10 for rank in ranks) / n,
        "recall50": sum(rank is not None and rank <= 50 for rank in ranks) / n,
        "mrr": sum((1.0 / rank) if rank else 0.0 for rank in ranks) / n,
    }


def compute_phrase_order_metrics(results: list[PhraseOrderResult]) -> dict[str, float]:
    """Phrase-aware metrics over the retained top-K population."""
    return _rank_metrics([r.retained_rank for r in results])


def compute_retained_grammar_metrics(results: list[PhraseOrderResult]) -> dict[str, float]:
    """Grammar-only metrics over exactly the same retained top-K population."""
    return _rank_metrics([r.grammar_rank for r in results])


def run_phrase_order_case(
    reranker,
    lex,
    case: dict,
    phrase_index,
    *,
    order_candidates: int,
    phrase_bonus_max: float,
) -> PhraseOrderResult:
    """Compare grammar-only and phrase-aware ranking on the same retained orders."""
    answer = str(case["answer"])
    bag = tokens(answer)
    acceptable = {
        phrase_key(x)
        for x in case.get("acceptable_orders", [answer])
    }

    candidates, _ = reranker.rank_orders(
        bag,
        lex,
        order_mode="exact" if len(bag) <= 6 else "beam",
        beam_width=256,
        exact_max_words=6,
        top_k=order_candidates,
    )

    grammar_scored: list[tuple[float, tuple[str, ...]]] = []
    phrase_scored: list[tuple[float, float, tuple[str, ...]]] = []
    for candidate in candidates:
        # rank_orders() already defines the normalized grammar/structure
        # objective. Using it here guarantees bonus=0 reproduces the retained
        # grammar ordering exactly, including syntax coverage.
        grammar_score = 100.0 * candidate.objective
        phrase_score, _ = phrase_index.score(candidate.order)
        grammar_scored.append((grammar_score, candidate.order))
        phrase_scored.append(
            (
                grammar_score + phrase_bonus_max * phrase_score,
                phrase_score,
                candidate.order,
            )
        )

    grammar_scored.sort(key=lambda item: (-item[0], item[1]))
    phrase_scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

    grammar_ranks = [
        i
        for i, (_, order) in enumerate(grammar_scored, 1)
        if phrase_key(order) in acceptable
    ]
    phrase_ranks = [
        i
        for i, (_, _, order) in enumerate(phrase_scored, 1)
        if phrase_key(order) in acceptable
    ]

    grammar_rank = min(grammar_ranks) if grammar_ranks else None
    retained_rank = min(phrase_ranks) if phrase_ranks else None
    grammar_best = grammar_scored[0][1] if grammar_scored else tuple()
    best_order = phrase_scored[0][2] if phrase_scored else tuple()
    best_phrase_score = phrase_scored[0][1] if phrase_scored else 0.0

    return PhraseOrderResult(
        case_id=str(case["id"]),
        answer=answer,
        category=str(case.get("category", "uncategorized")),
        retained_rank=retained_rank,
        retained_total=len(phrase_scored),
        best_order=" ".join(best_order),
        exact_best=phrase_key(best_order) in acceptable,
        target_retained=retained_rank is not None,
        best_phrase_score=best_phrase_score,
        grammar_rank=grammar_rank,
        grammar_best_order=" ".join(grammar_best),
    )


def print_phrase_order_summary(
    results: list[PhraseOrderResult],
    *,
    order_candidates: int,
    phrase_db: Path,
) -> None:
    print("\n=== PHRASE-AWARE FINAL ORDER A/B ===")
    print(f"Phrase DB: {phrase_db}")
    print(f"Grammar-retained orders per bag: {order_candidates}")
    if not results:
        print("No benchmark cases selected.")
        return

    for result in results:
        phrase_rank = "-" if result.retained_rank is None else str(result.retained_rank)
        grammar_rank = "-" if result.grammar_rank is None else str(result.grammar_rank)
        status = "TOP1" if result.retained_rank == 1 else (
            "TOP10" if result.retained_rank is not None and result.retained_rank <= 10
            else "TOP50" if result.retained_rank is not None and result.retained_rank <= 50
            else "DROP" if result.retained_rank is None
            else "MISS"
        )
        print(
            f"{status:5}  {result.case_id:<24} [{result.category:<20}] "
            f"G={grammar_rank:>2}/{result.retained_total:<2} "
            f"P={phrase_rank:>2}/{result.retained_total:<2} "
            f"phrase={result.best_phrase_score:5.3f} best={result.best_order}"
        )

    grammar = compute_retained_grammar_metrics(results)
    observed = compute_phrase_order_metrics(results)

    print("\nRetained grammar metrics (same top-K):")
    print(f"  cases           {int(grammar['cases'])}")
    print(
        f"  target retained {int(grammar['retained'])}/{int(grammar['cases'])} "
        f"({grammar['retained_rate']:.3f})"
    )
    print(f"  Recall@1        {grammar['recall1']:.3f}")
    print(f"  Recall@10       {grammar['recall10']:.3f}")
    print(f"  Recall@50       {grammar['recall50']:.3f}")
    print(f"  MRR             {grammar['mrr']:.3f}")

    print("\nPhrase-aware retained-order metrics:")
    print(f"  cases           {int(observed['cases'])}")
    print(
        f"  target retained {int(observed['retained'])}/{int(observed['cases'])} "
        f"({observed['retained_rate']:.3f})"
    )
    print(f"  Recall@1        {observed['recall1']:.3f}")
    print(f"  Recall@10       {observed['recall10']:.3f}")
    print(f"  Recall@50       {observed['recall50']:.3f}")
    print(f"  MRR             {observed['mrr']:.3f}")

    print("\nA/B delta on identical retained candidates:")
    print(f"  Recall@1   {observed['recall1'] - grammar['recall1']:+.3f}")
    print(f"  Recall@10  {observed['recall10'] - grammar['recall10']:+.3f}")
    print(f"  Recall@50  {observed['recall50'] - grammar['recall50']:+.3f}")
    print(f"  MRR        {observed['mrr'] - grammar['mrr']:+.3f}")
    if order_candidates < 50:
        print(
            f"  note: Recall@50 is bounded by retained top-{order_candidates}; "
            "its delta is expected to be zero unless the retained population changes."
        )


'''
s = s[:start] + replacement + s[end:]

# Remove the old duplicate compute_phrase_order_metrics definition near the dataclass.
old_metrics = re.compile(
    r"\n\ndef compute_phrase_order_metrics\(results: list\[PhraseOrderResult\]\) -> dict\[str, float\]:\n"
    r"    \"\"\"Metrics for phrase-aware ordering; dropped targets count as misses\.\"\"\"\n"
    r".*?\n    \}\n",
    re.DOTALL,
)
s, count = old_metrics.subn("", s, count=1)
if count != 1:
    raise RuntimeError(f"expected one old phrase metric block, found {count}")

# 3. Preserve the old full-mode baseline when phrase DB support is disabled.
old_cmd = '''        "--beam-width", "128",\n        "--phrase-rescore-top", "300",\n        "--phrase-bonus-max", str(phrase_bonus_max),\n        "--order-candidates", str(order_candidates),\n        "--top-per-group", "1",\n        "--export", str(output),\n    ]\n    if phrase_db is not None:\n        cmd += ["--phrase-db", str(phrase_db)]\n'''
new_cmd = '''        "--beam-width", "128",\n        "--phrase-rescore-top", "300",\n        "--top-per-group", "1",\n        "--export", str(output),\n    ]\n    if phrase_db is not None:\n        cmd += [\n            "--phrase-bonus-max", str(phrase_bonus_max),\n            "--order-candidates", str(order_candidates),\n            "--phrase-db", str(phrase_db),\n        ]\n'''
if old_cmd not in s:
    raise RuntimeError("reranker command block not found")
s = s.replace(old_cmd, new_cmd, 1)

# 4. Reject NaN / infinities.
s = s.replace(
    '''    if args.phrase_bonus_max < 0:\n        raise SystemExit("--phrase-bonus-max must be >= 0")\n''',
    '''    if not math.isfinite(args.phrase_bonus_max) or args.phrase_bonus_max < 0:\n        raise SystemExit("--phrase-bonus-max must be a finite value >= 0")\n''',
    1,
)

# 5. The phrase summary now computes both sides from the same retained results.
s = s.replace(
    '''                print_phrase_order_summary(\n                    phrase_results,\n                    grammar_results=results,\n                    order_candidates=args.order_candidates,\n                    phrase_db=phrase_db,\n                )\n''',
    '''                print_phrase_order_summary(\n                    phrase_results,\n                    order_candidates=args.order_candidates,\n                    phrase_db=phrase_db,\n                )\n''',
    1,
)

bench_path.write_text(s, encoding="utf-8")

# Replace focused tests with broader coverage for the review findings.
test_path = Path("tests/test_phrase_benchmarking.py")
test_path.write_text(r'''from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import anagram_benchmark as benchmark


class _FakeReranker:
    @staticmethod
    def rank_orders(words, _lex, **_kwargs):
        # Grammar winner first; intended phrase second.
        return (
            (
                SimpleNamespace(
                    order=("power", "is", "knowledge"),
                    objective=0.90,
                ),
                SimpleNamespace(
                    order=("knowledge", "is", "power"),
                    objective=0.88,
                ),
            ),
            6,
        )


class _NoTargetReranker:
    @staticmethod
    def rank_orders(words, _lex, **_kwargs):
        return (
            (
                SimpleNamespace(order=("power", "knowledge", "is"), objective=0.90),
                SimpleNamespace(order=("is", "power", "knowledge"), objective=0.80),
            ),
            6,
        )


class _PhraseIndex:
    def score(self, order):
        if tuple(order) == ("knowledge", "is", "power"):
            return 1.0, {}
        return 0.0, {}


class _NoEvidencePhraseIndex:
    def score(self, order):
        return 0.0, {}


class PhraseBenchmarkTests(unittest.TestCase):
    def test_phrase_evidence_can_flip_retained_order(self):
        result = benchmark.run_phrase_order_case(
            _FakeReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _PhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertTrue(result.target_retained)
        self.assertEqual(result.grammar_rank, 2)
        self.assertEqual(result.retained_rank, 1)
        self.assertEqual(result.grammar_best_order, "power is knowledge")
        self.assertEqual(result.best_order, "knowledge is power")

    def test_zero_phrase_evidence_preserves_retained_grammar_winner(self):
        result = benchmark.run_phrase_order_case(
            _FakeReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _NoEvidencePhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertEqual(result.grammar_rank, 2)
        self.assertEqual(result.retained_rank, 2)
        self.assertEqual(result.best_order, result.grammar_best_order)
        self.assertFalse(result.exact_best)

    def test_no_acceptable_order_retained_counts_as_miss(self):
        result = benchmark.run_phrase_order_case(
            _NoTargetReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _PhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertFalse(result.target_retained)
        self.assertIsNone(result.grammar_rank)
        self.assertIsNone(result.retained_rank)
        metrics = benchmark.compute_phrase_order_metrics([result])
        self.assertEqual(metrics["retained_rate"], 0.0)
        self.assertEqual(metrics["recall1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_phrase_metrics_cover_retained_higher_and_dropped_targets(self):
        rows = [
            benchmark.PhraseOrderResult(
                "id1", "a b", "test", 1, 20, "a b", True, True, 1.0, 1, "a b"
            ),
            benchmark.PhraseOrderResult(
                "id2", "c d", "test", 5, 20, "d c", False, True, 0.0, 7, "d c"
            ),
            benchmark.PhraseOrderResult(
                "id3", "e f", "test", 20, 20, "f e", False, True, 0.0, 12, "f e"
            ),
            benchmark.PhraseOrderResult(
                "id4", "g h", "test", None, 20, "h g", False, False, 0.0, None, "h g"
            ),
        ]
        metrics = benchmark.compute_phrase_order_metrics(rows)
        self.assertEqual(metrics["retained"], 3)
        self.assertAlmostEqual(metrics["retained_rate"], 0.75)
        self.assertAlmostEqual(metrics["recall1"], 0.25)
        self.assertAlmostEqual(metrics["recall10"], 0.50)
        self.assertAlmostEqual(metrics["recall50"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 0.2 + 0.05) / 4.0)

        grammar = benchmark.compute_retained_grammar_metrics(rows)
        self.assertAlmostEqual(grammar["recall1"], 0.25)
        self.assertAlmostEqual(grammar["recall10"], 0.50)
        self.assertAlmostEqual(grammar["recall50"], 0.75)

    def test_empty_phrase_summary_does_not_crash(self):
        benchmark.print_phrase_order_summary(
            [], order_candidates=16, phrase_db=Path("empty.db")
        )

    def test_full_reranker_command_forwards_phrase_options(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power", "deep_per_group": 123},
            reranker=Path("anagram_rerank.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=Path("wikimedia_phrases.db"),
            phrase_bonus_max=4.5,
            order_candidates=24,
        )
        self.assertIn("--phrase-db", cmd)
        self.assertEqual(cmd[cmd.index("--phrase-db") + 1], "wikimedia_phrases.db")
        self.assertEqual(cmd[cmd.index("--phrase-bonus-max") + 1], "4.5")
        self.assertEqual(cmd[cmd.index("--order-candidates") + 1], "24")
        self.assertEqual(cmd[cmd.index("--deep-per-group") + 1], "123")

    def test_full_reranker_command_preserves_baseline_when_phrase_db_disabled(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power"},
            reranker=Path("anagram_rerank_core.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=None,
            phrase_bonus_max=5.0,
            order_candidates=16,
        )
        self.assertNotIn("--phrase-db", cmd)
        self.assertNotIn("--phrase-bonus-max", cmd)
        self.assertNotIn("--order-candidates", cmd)
        # Existing positive-bigram rescoring remains part of the baseline.
        self.assertIn("--phrase-rescore-top", cmd)


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")
