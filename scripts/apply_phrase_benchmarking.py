from pathlib import Path

path = Path("anagram_benchmark.py")
s = path.read_text(encoding="utf-8")

s = s.replace(
'''order (default)
    Fast. Given each known phrase's unordered word bag, rank all word orders
    (exactly for <=6 words) using the selected reranker's grammar/structure objective.
    Reports Top-1/10/50, MRR, and the chosen order.
''',
'''order (default)
    Fast. Given each known phrase's unordered word bag, rank all word orders
    (exactly for <=6 words) using the selected reranker's grammar/structure objective.
    When --phrase-db is supplied, also rerank the grammar-retained top-K orders
    with positive phrase-title evidence and report an A/B comparison.
'''
)
s = s.replace(
'''python anagram_benchmark.py --mode full --workers 8
python anagram_benchmark.py --mode order --case better_late --case shakira_control
''',
'''python anagram_benchmark.py --mode full --workers 8
python anagram_benchmark.py --mode order --phrase-db wikimedia_phrases.db
python anagram_benchmark.py --mode order --case better_late --case shakira_control
'''
)

needle = '''\n\ndef compute_order_metrics(results: list[OrderResult]) -> dict[str, float]:\n'''
insert = '''\n\n@dataclass(slots=True)\nclass PhraseOrderResult:\n    case_id: str\n    answer: str\n    category: str\n    retained_rank: int | None\n    retained_total: int\n    best_order: str\n    exact_best: bool\n    target_retained: bool\n    best_phrase_score: float\n\n\ndef compute_phrase_order_metrics(results: list[PhraseOrderResult]) -> dict[str, float]:\n    \"\"\"Metrics for phrase-aware ordering; dropped targets count as misses.\"\"\"\n    if not results:\n        return {}\n    n = len(results)\n    ranks = [r.retained_rank for r in results]\n    retained = sum(rank is not None for rank in ranks)\n    return {\n        \"cases\": float(n),\n        \"retained\": float(retained),\n        \"retained_rate\": retained / n,\n        \"recall1\": sum(rank is not None and rank <= 1 for rank in ranks) / n,\n        \"recall10\": sum(rank is not None and rank <= 10 for rank in ranks) / n,\n        \"recall50\": sum(rank is not None and rank <= 50 for rank in ranks) / n,\n        \"mrr\": sum((1.0 / rank) if rank else 0.0 for rank in ranks) / n,\n    }\n\n\ndef compute_order_metrics(results: list[OrderResult]) -> dict[str, float]:\n'''
if needle not in s:
    raise SystemExit("compute_order_metrics insertion point not found")
s = s.replace(needle, insert, 1)

needle = '''\n\ndef print_order_summary(results: list[OrderResult]) -> None:\n'''
insert = '''\n\ndef _candidate_final_order_component(candidate) -> float:\n    \"\"\"Order-dependent part of FINAL; bag-level terms cancel within one bag.\"\"\"\n    return 100.0 * (\n        0.22 * candidate.grammar_norm\n        + 0.28 * candidate.structure_norm\n        + 0.08 * candidate.valency_norm\n    )\n\n\ndef run_phrase_order_case(\n    reranker,\n    lex,\n    case: dict,\n    phrase_index,\n    *,\n    order_candidates: int,\n    phrase_bonus_max: float,\n) -> PhraseOrderResult:\n    \"\"\"Rerank grammar-retained orders with positive phrase-index evidence.\"\"\"\n    answer = str(case[\"answer\"])\n    bag = tokens(answer)\n    acceptable = {\n        phrase_key(x)\n        for x in case.get(\"acceptable_orders\", [answer])\n    }\n\n    candidates, _ = reranker.rank_orders(\n        bag,\n        lex,\n        order_mode=\"exact\" if len(bag) <= 6 else \"beam\",\n        beam_width=256,\n        exact_max_words=6,\n        top_k=order_candidates,\n    )\n\n    scored: list[tuple[float, float, tuple[str, ...]]] = []\n    for candidate in candidates:\n        phrase_score, _ = phrase_index.score(candidate.order)\n        combined = (\n            _candidate_final_order_component(candidate)\n            + phrase_bonus_max * phrase_score\n        )\n        scored.append((combined, phrase_score, candidate.order))\n\n    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))\n    ranks = [\n        i\n        for i, (_, _, order) in enumerate(scored, 1)\n        if phrase_key(order) in acceptable\n    ]\n    retained_rank = min(ranks) if ranks else None\n    best_order = scored[0][2] if scored else tuple()\n    best_phrase_score = scored[0][1] if scored else 0.0\n\n    return PhraseOrderResult(\n        case_id=str(case[\"id\"]),\n        answer=answer,\n        category=str(case.get(\"category\", \"uncategorized\")),\n        retained_rank=retained_rank,\n        retained_total=len(scored),\n        best_order=\" \".join(best_order),\n        exact_best=phrase_key(best_order) in acceptable,\n        target_retained=retained_rank is not None,\n        best_phrase_score=best_phrase_score,\n    )\n\n\ndef print_phrase_order_summary(\n    results: list[PhraseOrderResult],\n    *,\n    grammar_results: list[OrderResult],\n    order_candidates: int,\n    phrase_db: Path,\n) -> None:\n    print(\"\\n=== PHRASE-AWARE FINAL ORDER A/B ===\")\n    print(f\"Phrase DB: {phrase_db}\")\n    print(f\"Grammar-retained orders per bag: {order_candidates}\")\n    for result in results:\n        if result.retained_rank is None:\n            status = \"DROP\"\n            rank_text = f\"-/{result.retained_total}\"\n        else:\n            status = (\n                \"TOP1\" if result.retained_rank == 1\n                else \"TOP10\" if result.retained_rank <= 10\n                else \"TOP50\" if result.retained_rank <= 50\n                else \"MISS\"\n            )\n            rank_text = f\"{result.retained_rank}/{result.retained_total}\"\n        print(\n            f\"{status:5}  {result.case_id:<24} [{result.category:<20}] \"\n            f\"rank={rank_text:<8} phrase={result.best_phrase_score:5.3f} \"\n            f\"best={result.best_order}\"\n        )\n\n    observed = compute_phrase_order_metrics(results)\n    print(\"\\nPhrase-aware retained-order metrics:\")\n    print(f\"  cases          {int(observed['cases'])}\")\n    print(\n        f\"  target retained {int(observed['retained'])}/{int(observed['cases'])} \"\n        f\"({observed['retained_rate']:.3f})\"\n    )\n    print(f\"  Recall@1       {observed['recall1']:.3f}\")\n    print(f\"  Recall@10      {observed['recall10']:.3f}\")\n    print(f\"  Recall@50      {observed['recall50']:.3f}\")\n    print(f\"  MRR            {observed['mrr']:.3f}\")\n\n    grammar = compute_order_metrics(grammar_results)\n    if grammar and int(grammar[\"cases\"]) == int(observed[\"cases\"]):\n        print(\"\\nA/B delta vs grammar-only exact ordering:\")\n        print(f\"  Recall@1   {observed['recall1'] - grammar['recall1']:+.3f}\")\n        print(f\"  Recall@10  {observed['recall10'] - grammar['recall10']:+.3f}\")\n        print(f\"  Recall@50  {observed['recall50'] - grammar['recall50']:+.3f}\")\n        print(f\"  MRR        {observed['mrr'] - grammar['mrr']:+.3f}\")\n\n\ndef print_order_summary(results: list[OrderResult]) -> None:\n'''
if needle not in s:
    raise SystemExit("print_order_summary insertion point not found")
s = s.replace(needle, insert, 1)

needle = '''\n\ndef run_full_case(\n'''
insert = '''\n\ndef make_reranker_command(\n    case: dict,\n    *,\n    reranker: Path,\n    export: Path,\n    output: Path,\n    workers: int,\n    phrase_db: Path | None,\n    phrase_bonus_max: float,\n    order_candidates: int,\n) -> list[str]:\n    cmd = [\n        sys.executable,\n        str(reranker),\n        str(export),\n        \"--benchmark-answer\", str(case[\"answer\"]),\n        \"--workers\", str(workers),\n        \"--backend\", \"auto\",\n        \"--deep-per-group\", str(int(case.get(\"deep_per_group\", 5000))),\n        \"--beam-width\", \"128\",\n        \"--phrase-rescore-top\", \"300\",\n        \"--phrase-bonus-max\", str(phrase_bonus_max),\n        \"--order-candidates\", str(order_candidates),\n        \"--top-per-group\", \"1\",\n        \"--export\", str(output),\n    ]\n    if phrase_db is not None:\n        cmd += [\"--phrase-db\", str(phrase_db)]\n    return cmd\n\n\ndef run_full_case(\n'''
if needle not in s:
    raise SystemExit("run_full_case insertion point not found")
s = s.replace(needle, insert, 1)

s = s.replace(
'''    cache_dir: Path,\n    workers: int,\n    rebuild: bool,\n) -> FullResult:\n''',
'''    cache_dir: Path,\n    workers: int,\n    rebuild: bool,\n    phrase_db: Path | None,\n    phrase_bonus_max: float,\n    order_candidates: int,\n) -> FullResult:\n''',
1,
)

old_cmd = '''    cmd = [\n        sys.executable,\n        str(reranker),\n        str(export),\n        \"--benchmark-answer\", answer,\n        \"--workers\", str(workers),\n        \"--backend\", \"auto\",\n        \"--deep-per-group\", str(int(case.get(\"deep_per_group\", 5000))),\n        \"--beam-width\", \"128\",\n        \"--phrase-rescore-top\", \"300\",\n        \"--top-per-group\", \"1\",\n        \"--export\", str(cache_dir / f\"{slug(case_id)}_reranked.txt\"),\n    ]\n'''
new_cmd = '''    cmd = make_reranker_command(\n        case,\n        reranker=reranker,\n        export=export,\n        output=cache_dir / f\"{slug(case_id)}_reranked.txt\",\n        workers=workers,\n        phrase_db=phrase_db,\n        phrase_bonus_max=phrase_bonus_max,\n        order_candidates=order_candidates,\n    )\n'''
if old_cmd not in s:
    raise SystemExit("reranker command block not found")
s = s.replace(old_cmd, new_cmd, 1)

needle = '''    ap.add_argument(\"--workers\", type=int, default=8)\n    ap.add_argument(\"--rebuild\", action=\"store_true\")\n    args = ap.parse_args()\n'''
replacement = '''    ap.add_argument(\"--workers\", type=int, default=8)\n    ap.add_argument(\"--rebuild\", action=\"store_true\")\n    ap.add_argument(\n        \"--phrase-db\",\n        type=Path,\n        help=(\n            \"Optional SQLite phrase index. In order mode this enables phrase-aware \"\n            \"top-K A/B metrics; in full mode it is forwarded to the reranker.\"\n        ),\n    )\n    ap.add_argument(\n        \"--phrase-bonus-max\",\n        type=float,\n        default=5.0,\n        help=\"Maximum additive phrase-evidence bonus used by the A/B and full reranker\",\n    )\n    ap.add_argument(\n        \"--order-candidates\",\n        type=int,\n        default=16,\n        help=\"Grammar-retained orders per bag available to phrase-aware selection\",\n    )\n    args = ap.parse_args()\n\n    if args.phrase_bonus_max < 0:\n        raise SystemExit(\"--phrase-bonus-max must be >= 0\")\n    if args.order_candidates < 1:\n        raise SystemExit(\"--order-candidates must be >= 1\")\n\n    phrase_db = args.phrase_db.expanduser() if args.phrase_db else None\n    if phrase_db is not None and not phrase_db.is_file():\n        raise SystemExit(f\"--phrase-db not found: {phrase_db}\")\n'''
if needle not in s:
    raise SystemExit("argparse insertion point not found")
s = s.replace(needle, replacement, 1)

old_order = '''        results = []\n        t0 = time.perf_counter()\n        for case in cases:\n            result = run_order_case(reranker, lex, case)\n            results.append(result)\n        print_order_summary(results)\n        print(f\"\\nSuite wall time: {time.perf_counter() - t0:.2f}s\")\n        return 0\n'''
new_order = '''        results = []\n        t0 = time.perf_counter()\n        for case in cases:\n            result = run_order_case(reranker, lex, case)\n            results.append(result)\n        print_order_summary(results)\n\n        if phrase_db is not None:\n            phrase_index = reranker.PhraseIndex.open(phrase_db)\n            try:\n                phrase_results = [\n                    run_phrase_order_case(\n                        reranker,\n                        lex,\n                        case,\n                        phrase_index,\n                        order_candidates=args.order_candidates,\n                        phrase_bonus_max=args.phrase_bonus_max,\n                    )\n                    for case in cases\n                ]\n                print_phrase_order_summary(\n                    phrase_results,\n                    grammar_results=results,\n                    order_candidates=args.order_candidates,\n                    phrase_db=phrase_db,\n                )\n            finally:\n                close = getattr(phrase_index, \"close\", None)\n                if callable(close):\n                    close()\n\n        print(f\"\\nSuite wall time: {time.perf_counter() - t0:.2f}s\")\n        return 0\n'''
if old_order not in s:
    raise SystemExit("order-mode block not found")
s = s.replace(old_order, new_order, 1)

old_full_args = '''            workers=args.workers,\n            rebuild=args.rebuild,\n        )\n'''
new_full_args = '''            workers=args.workers,\n            rebuild=args.rebuild,\n            phrase_db=phrase_db,\n            phrase_bonus_max=args.phrase_bonus_max,\n            order_candidates=args.order_candidates,\n        )\n'''
if old_full_args not in s:
    raise SystemExit("full-mode call block not found")
s = s.replace(old_full_args, new_full_args, 1)

path.write_text(s, encoding="utf-8")

# Focused tests use fakes so CI remains offline.
test = Path("tests/test_phrase_benchmarking.py")
test.write_text(r'''from __future__ import annotations

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
                    grammar_norm=0.90,
                    structure_norm=0.90,
                    valency_norm=1.0,
                ),
                SimpleNamespace(
                    order=("knowledge", "is", "power"),
                    grammar_norm=0.88,
                    structure_norm=0.89,
                    valency_norm=1.0,
                ),
            ),
            6,
        )


class _PhraseIndex:
    def score(self, order):
        if tuple(order) == ("knowledge", "is", "power"):
            return 1.0, {}
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
        self.assertTrue(result.exact_best)
        self.assertEqual(result.retained_rank, 1)
        self.assertEqual(result.best_order, "knowledge is power")

    def test_missing_retained_target_counts_as_miss(self):
        result = benchmark.PhraseOrderResult(
            "x", "a b", "test", None, 2, "b a", False, False, 0.0
        )
        metrics = benchmark.compute_phrase_order_metrics([result])
        self.assertEqual(metrics["retained_rate"], 0.0)
        self.assertEqual(metrics["recall1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

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

    def test_full_reranker_command_omits_phrase_db_when_disabled(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power"},
            reranker=Path("anagram_rerank.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=None,
            phrase_bonus_max=5.0,
            order_candidates=16,
        )
        self.assertNotIn("--phrase-db", cmd)


if __name__ == "__main__":
    unittest.main()
''', encoding="utf-8")
