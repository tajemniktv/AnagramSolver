from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"patch anchor not found: {label}")
    return text.replace(old, new, 1)


generator = Path("anagram_generate.py")
text = generator.read_text(encoding="utf-8")
text = replace_once(
    text,
    "import sys\nimport unicodedata\n",
    "import sys\nimport time\nimport unicodedata\n",
    "generator time import",
)
text = replace_once(
    text,
    "from typing import Iterable, Iterator, Sequence\n",
    "from typing import Iterable, Iterator, Sequence\n\n"
    "from anagram_search_parallel import (\n"
    "    SearchCandidate,\n"
    "    resolve_worker_count,\n"
    "    solve_parallel,\n"
    ")\n",
    "generator parallel import",
)
text = replace_once(
    text,
    '    p.add_argument("--no-repeat", action="store_true")\n',
    '    p.add_argument("--no-repeat", action="store_true")\n'
    '    p.add_argument(\n'
    '        "--workers", type=int, default=0, metavar="N",\n'
    '        help="Exact-search worker processes; 0 chooses automatically",\n'
    '    )\n',
    "generator workers parser",
)
text = replace_once(
    text,
    '    if args.max_results < 0:\n        raise SystemExit("--max-results must be >= 0")\n',
    '    if args.max_results < 0:\n        raise SystemExit("--max-results must be >= 0")\n'
    '    if args.workers < 0:\n        raise SystemExit("--workers must be >= 0")\n',
    "generator workers validation",
)
old_loop = '''    solutions: list[tuple[str, ...]] = []
    generated = 0
    accepted = 0
    try:
        for solution in solve(
            remaining,
            candidates,
            args.min_words,
            args.max_words,
            args.max_results,
            allow_repeat=not args.no_repeat,
        ):
'''
new_loop = '''    solutions: list[tuple[str, ...]] = []
    generated = 0
    accepted = 0
    worker_count = resolve_worker_count(args.workers)
    search_candidates = [
        SearchCandidate(c.word, c.sig, c.length) for c in candidates
    ]
    search_started = time.perf_counter()
    try:
        for solution in solve_parallel(
            remaining,
            search_candidates,
            args.min_words,
            args.max_words,
            args.max_results,
            allow_repeat=not args.no_repeat,
            workers=worker_count,
            required_any=contains_any,
            initial_any_matched=bool(contains_any.intersection(required_words)),
        ):
'''
text = replace_once(text, old_loop, new_loop, "generator solve loop")
old_summary = '''    print(
        f"Generated {generated:,} exact word set(s); "
        f"{accepted:,} survived clue constraints.",
        file=sys.stderr,
    )
'''
new_summary = '''    search_seconds = time.perf_counter() - search_started
    print(
        f"Generated {generated:,} exact word set(s); "
        f"{accepted:,} survived clue constraints. "
        f"Exact search: {search_seconds:.2f}s with {worker_count} worker(s).",
        file=sys.stderr,
    )
'''
text = replace_once(text, old_summary, new_summary, "generator timing summary")
generator.write_text(text, encoding="utf-8")


frontend = Path("anagram_solver.py")
text = frontend.read_text(encoding="utf-8")
text = replace_once(
    text,
    "import subprocess\nimport sys\n",
    "import subprocess\nimport sys\nimport time\n",
    "frontend time import",
)
text = replace_once(
    text,
    '        "--top-per-group", "1",\n        "--export", str(output),\n',
    '        "--top-per-group", "1",\n        "--workers", str(args.workers),\n        "--export", str(output),\n',
    "frontend forwards generator workers",
)
text = text.replace(
    'help="Reranker workers; 0 chooses automatically"',
    'help="Generator/reranker workers; 0 chooses automatically"',
)
old_fn = '''def _generate_candidates(args: argparse.Namespace, candidates: Path) -> None:
    """Generate privately, publishing the shared cache only on success."""
    temporary = candidates.with_name(
        f".{candidates.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _run(build_generator_command(args, temporary), verbose=args.verbose)
        if not temporary.is_file():
            raise SystemExit(
                f"Generator completed without writing its candidate export: {temporary}"
            )
        temporary.replace(candidates)
    finally:
        temporary.unlink(missing_ok=True)
'''
new_fn = '''def _generate_candidates(args: argparse.Namespace, candidates: Path) -> float:
    """Generate privately, publishing the shared cache only on success."""
    temporary = candidates.with_name(
        f".{candidates.name}.{uuid.uuid4().hex}.tmp"
    )
    started = time.perf_counter()
    try:
        _run(build_generator_command(args, temporary), verbose=args.verbose)
        if not temporary.is_file():
            raise SystemExit(
                f"Generator completed without writing its candidate export: {temporary}"
            )
        temporary.replace(candidates)
        return time.perf_counter() - started
    finally:
        temporary.unlink(missing_ok=True)
'''
text = replace_once(text, old_fn, new_fn, "frontend generation timing")
text = replace_once(
    text,
    "        _generate_candidates(args, candidates)\n    elif not args.json:\n",
    "        generation_seconds = _generate_candidates(args, candidates)\n"
    "        if not args.json:\n"
    "            print(f\"Candidate generation finished in {generation_seconds:.2f}s.\")\n"
    "    elif not args.json:\n",
    "frontend timing output",
)
frontend.write_text(text, encoding="utf-8")


benchmark = Path("anagram_benchmark.py")
text = benchmark.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''def make_generator_command(
    case: dict,
    generator: Path,
    export: Path,
) -> list[str]:
''',
    '''def make_generator_command(
    case: dict,
    generator: Path,
    export: Path,
    workers: int = 0,
) -> list[str]:
''',
    "benchmark generator signature",
)
text = replace_once(
    text,
    '        "--top-per-group", "1",\n    ]\n',
    '        "--top-per-group", "1",\n        "--workers", str(workers),\n    ]\n',
    "benchmark forwards workers",
)
text = replace_once(
    text,
    "        cmd = make_generator_command(case, generator, export)\n",
    "        cmd = make_generator_command(case, generator, export, workers)\n",
    "benchmark generator call",
)
benchmark.write_text(text, encoding="utf-8")


cli_tests = Path("tests/test_user_cli.py")
text = cli_tests.read_text(encoding="utf-8")
anchor = '        self.assertEqual(cmd[cmd.index("--min-zipf") + 1], "2.7")\n'
text = replace_once(
    text,
    anchor,
    anchor + '        self.assertEqual(cmd[cmd.index("--workers") + 1], "0")\n',
    "CLI generator workers test",
)
cli_tests.write_text(text, encoding="utf-8")


readme = Path("README.md")
text = readme.read_text(encoding="utf-8")
needle = (
    "Normal use runs a balanced search capped at 100,000 generated word bags. "
    "This avoids silently turning every simple solve into an unlimited 2–6-word "
    "enumeration while still giving the reranker a large candidate population.\n"
)
addition = needle + (
    "\nExact candidate search uses multiple worker processes by default (up to 8, "
    "based on available CPUs). `--workers N` controls both generation and "
    "reranking. Search tasks are split at ordered word prefixes and merged in "
    "canonical DFS order, so worker count does not change the bounded no-clue "
    "candidate prefix. Hint words are enforced during search, allowing dead "
    "no-hint branches to be pruned before they become candidate bags.\n"
)
text = replace_once(text, needle, addition, "README performance paragraph")
readme.write_text(text, encoding="utf-8")
