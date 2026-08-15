from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected patch target not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# 1) Preserve rank_orders() tie order. Phrase evidence must have zero influence
# when phrase_bonus_max == 0.
benchmark = ROOT / "anagram_benchmark.py"
replace_once(
    benchmark,
    """    grammar_scored.sort(key=lambda item: (-item[0], item[1]))\n    phrase_scored.sort(key=lambda item: (-item[0], -item[1], item[2]))\n""",
    """    # Python's sort is stable, so equal objectives preserve rank_orders()'s\n    # incoming tie order. Phrase evidence may break ties only when enabled.\n    grammar_scored.sort(key=lambda item: -item[0])\n    if phrase_bonus_max > 0:\n        phrase_scored.sort(key=lambda item: (-item[0], -item[1]))\n    else:\n        phrase_scored.sort(key=lambda item: -item[0])\n""",
)

# 2) Resolve and validate DB paths before any benchmark work starts.
ci = ROOT / "ci_phrase_matrix.py"
replace_once(
    ci,
    """    args = ap.parse_args()\n\n    scenarios = [\n        Scenario(\"Baseline (no phrase DB)\", \"baseline\", None),\n        Scenario(\"Wiktionary\", \"wiktionary\", args.wiktionary_db),\n        Scenario(\"Wiktionary + Wikipedia\", \"wiktionary-wikipedia\", args.wikipedia_db),\n    ]\n""",
    """    args = ap.parse_args()\n\n    # Resolve relative paths against the caller's working directory before child\n    # benchmarks switch cwd to HERE, and fail fast before running the baseline.\n    wiktionary_db = args.wiktionary_db.expanduser().resolve()\n    wikipedia_db = args.wikipedia_db.expanduser().resolve()\n    for label, path in (\n        (\"Wiktionary phrase DB\", wiktionary_db),\n        (\"Wiktionary + Wikipedia phrase DB\", wikipedia_db),\n    ):\n        if not path.is_file():\n            ap.error(f\"{label} does not exist or is not a file: {path}\")\n\n    scenarios = [\n        Scenario(\"Baseline (no phrase DB)\", \"baseline\", None),\n        Scenario(\"Wiktionary\", \"wiktionary\", wiktionary_db),\n        Scenario(\"Wiktionary + Wikipedia\", \"wiktionary-wikipedia\", wikipedia_db),\n    ]\n""",
)

# 3) Make the summary's incomparable scopes explicit instead of inviting a bad
# side-by-side interpretation.
replace_once(
    ci,
    """        f.write(\n            \"The same code and benchmark cases are evaluated with no phrase DB, \"\n            \"Wiktionary titles, and Wiktionary+Wikipedia titles. Phrase A/B deltas \"\n            \"compare the identical retained top-K orders.\\n\\n\"\n        )\n        f.write(\n            \"| Corpus | Grammar R@1 | Retained G R@1 | Phrase R@1 | Δ R@1 | \"\n""",
    """        f.write(\n            \"The same code and benchmark cases are evaluated with no phrase DB, \"\n            \"Wiktionary titles, and Wiktionary+Wikipedia titles. Phrase A/B deltas \"\n            \"compare the identical retained top-K orders. Exact <=6w R@1 is a separate \"\n            \"exhaustive-permutation metric over only cases with at most six words.\\n\\n\"\n        )\n        f.write(\n            \"| Corpus | Exact <=6w R@1 | Retained G R@1 | Phrase R@1 | Δ R@1 | \"\n""",
)
replace_once(
    ci,
    "f\"{scenario.name:<28} grammarR1={fmt(order['grammar_r1'])} \"",
    "f\"{scenario.name:<28} exact6wR1={fmt(order['grammar_r1'])} \"",
)

# 4) Regression coverage for tied grammar objectives with phrase bonus disabled.
tests = ROOT / "tests" / "test_phrase_benchmarking.py"
replace_once(
    tests,
    """class _NoTargetReranker:\n    @staticmethod\n    def rank_orders(words, _lex, **_kwargs):\n        return (\n            (\n                SimpleNamespace(order=(\"power\", \"knowledge\", \"is\"), objective=0.90),\n                SimpleNamespace(order=(\"is\", \"power\", \"knowledge\"), objective=0.80),\n            ),\n            6,\n        )\n\n\nclass _PhraseIndex:\n""",
    """class _NoTargetReranker:\n    @staticmethod\n    def rank_orders(words, _lex, **_kwargs):\n        return (\n            (\n                SimpleNamespace(order=(\"power\", \"knowledge\", \"is\"), objective=0.90),\n                SimpleNamespace(order=(\"is\", \"power\", \"knowledge\"), objective=0.80),\n            ),\n            6,\n        )\n\n\nclass _TiedReranker:\n    @staticmethod\n    def rank_orders(words, _lex, **_kwargs):\n        # Incoming rank_orders() tie order is authoritative. The intended phrase\n        # deliberately comes second and has stronger phrase evidence.\n        return (\n            (\n                SimpleNamespace(order=(\"power\", \"is\", \"knowledge\"), objective=0.90),\n                SimpleNamespace(order=(\"knowledge\", \"is\", \"power\"), objective=0.90),\n            ),\n            6,\n        )\n\n\nclass _PhraseIndex:\n""",
)
replace_once(
    tests,
    """    def test_zero_phrase_evidence_preserves_retained_grammar_winner(self):\n        result = benchmark.run_phrase_order_case(\n            _FakeReranker,\n            object(),\n            {\"id\": \"knowledge_power\", \"answer\": \"knowledge is power\"},\n            _NoEvidencePhraseIndex(),\n            order_candidates=16,\n            phrase_bonus_max=5.0,\n        )\n        self.assertEqual(result.grammar_rank, 2)\n        self.assertEqual(result.retained_rank, 2)\n        self.assertEqual(result.best_order, result.grammar_best_order)\n        self.assertFalse(result.exact_best)\n\n""",
    """    def test_zero_phrase_evidence_preserves_retained_grammar_winner(self):\n        result = benchmark.run_phrase_order_case(\n            _FakeReranker,\n            object(),\n            {\"id\": \"knowledge_power\", \"answer\": \"knowledge is power\"},\n            _NoEvidencePhraseIndex(),\n            order_candidates=16,\n            phrase_bonus_max=5.0,\n        )\n        self.assertEqual(result.grammar_rank, 2)\n        self.assertEqual(result.retained_rank, 2)\n        self.assertEqual(result.best_order, result.grammar_best_order)\n        self.assertFalse(result.exact_best)\n\n    def test_zero_bonus_preserves_incoming_order_for_objective_ties(self):\n        result = benchmark.run_phrase_order_case(\n            _TiedReranker,\n            object(),\n            {\"id\": \"knowledge_power\", \"answer\": \"knowledge is power\"},\n            _PhraseIndex(),\n            order_candidates=16,\n            phrase_bonus_max=0.0,\n        )\n        self.assertEqual(result.grammar_best_order, \"power is knowledge\")\n        self.assertEqual(result.best_order, \"power is knowledge\")\n        self.assertEqual(result.grammar_rank, 2)\n        self.assertEqual(result.retained_rank, 2)\n        self.assertFalse(result.exact_best)\n\n""",
)
