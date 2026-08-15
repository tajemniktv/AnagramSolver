from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"anchor not found: {label}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


solver = Path("anagram_solver.py")
replace_once(
    solver,
    '        description="Solve and rank exact multi-word anagrams in one command.",\n',
    '        description=(\n'
    '            "Solve and rank exact multi-word anagrams in one command. "\n'
    '            f"The default balanced search caps candidate generation at "\n'
    '            f"{BALANCED_MAX_RESULTS:,} bags and may miss the answer; "\n'
    '            "--exhaustive removes the generation cap."\n'
    '        ),\n',
    "balanced help disclosure",
)
replace_once(
    solver,
    '        help="Enumerate every matching word bag; complete but potentially much slower",\n',
    '        help=(\n'
    '            "Enumerate every matching candidate word bag with no generation cap; "\n'
    '            "final deep reranking remains bounded and displayed results are not exhaustive"\n'
    '        ),\n',
    "exhaustive help semantics",
)

readme = Path("README.md")
replace_once(
    readme,
    "Normal use runs a balanced search capped at 100,000 generated word bags. This avoids silently turning every simple solve into an unlimited 2–6-word enumeration while still giving the reranker a large candidate population.\n",
    "Normal use runs a **balanced** search capped at 100,000 generated word bags. It is much more responsive than unlimited 2–6-word enumeration, but the cap means it **can miss the answer** if the correct bag occurs later in generation order.\n",
    "README balanced disclosure",
)
replace_once(
    readme,
    "For complete enumeration with no generation cap:\n",
    "For unlimited **candidate generation** with no generation cap:\n",
    "README exhaustive heading",
)
replace_once(
    readme,
    "`--exhaustive` is the universal completeness mode. It can become much slower when the word count is unknown or many short/common words fit the letter multiset. Supplying clues or an exact word count can reduce that search space dramatically.\n",
    "`--exhaustive` exhaustively generates matching word bags, but the user-facing reranker still deep-analyzes a bounded shortlist and only those deep-ranked rows are displayed. In other words, it removes **generation** truncation; it is not a promise that every generated bag receives full grammar/phrase analysis. It can become much slower when the word count is unknown or many short/common words fit the letter multiset. Supplying clues or an exact word count can reduce that search space dramatically.\n",
    "README exhaustive semantics",
)

tests = Path("tests/test_user_cli.py")
replace_once(
    tests,
    '        self.assertEqual(solver._generation_mode(args), "balanced")\n',
    '        self.assertEqual(solver._generation_mode(args), "balanced")\n'
    '        self.assertEqual(solver._generation_cap(args), solver.BALANCED_MAX_RESULTS)\n',
    "balanced cap test",
)
replace_once(
    tests,
    '        self.assertEqual(solver._generation_mode(args), "quick")\n',
    '        self.assertEqual(solver._generation_mode(args), "quick")\n'
    '        self.assertEqual(solver._generation_cap(args), solver.QUICK_MAX_RESULTS)\n',
    "quick cap test",
)
replace_once(
    tests,
    '    def test_hints_excludes_require_and_word_count_are_forwarded(self) -> None:\n',
    '    def test_quick_and_exhaustive_are_mutually_exclusive(self) -> None:\n'
    '        with self.assertRaises(SystemExit):\n'
    '            solver.build_parser().parse_args(["abcdef", "--quick", "--exhaustive"])\n\n'
    '    def test_hints_excludes_require_and_word_count_are_forwarded(self) -> None:\n',
    "mutual exclusion test",
)
