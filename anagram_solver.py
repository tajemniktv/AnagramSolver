#!/usr/bin/env python3
"""User-facing one-command frontend for AnagramSolver.

The research tools remain split into generator and reranker modules. This file
provides the normal human path: give it letters/text, optional clues, and get a
compact ranked result list without manually managing intermediate exports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
GENERATOR = HERE / "anagram_generate.py"
RERANKER = HERE / "anagram_rerank.py"
DEFAULT_RUN_ROOT = Path.home() / ".multi_anagram" / "solver_runs"
BALANCED_MAX_RESULTS = 100_000
QUICK_MAX_RESULTS = 20_000
GENERATION_CACHE_SCHEMA = 2

_RESULT_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+FINAL=\s*(?P<score>[\d.]+).*?"
    r"PBONUS=\s*[\d.]+\s+(?P<phrase>.*?)\s+\[CANON=",
)
_SECTION_RE = re.compile(r"^===\s+(?P<words>\d+)-WORD\s+RERANKED")
_REQUIRED_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


@dataclass(slots=True, frozen=True)
class Result:
    word_count: int
    rank: int
    score: float
    phrase: str


def _csv_words(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def _normalized_target(text: str) -> str:
    """Match the generator's effective A-Z target normalization for caching."""
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return "".join(ch for ch in ascii_text.lower() if "a" <= ch <= "z")


def _normalized_words(values: Sequence[str]) -> list[str]:
    """Canonicalize set-like word constraints for cache keys."""
    return sorted(
        {
            normalized
            for word in _csv_words(values)
            if (normalized := _normalized_target(word))
        }
    )


def _normalized_required_words(values: Sequence[str]) -> list[str]:
    """Mirror generator --require tokenization without losing order or repeats."""
    out: list[str] = []
    for chunk in _csv_words(values):
        for token in _REQUIRED_TOKEN_RE.findall(chunk):
            normalized = _normalized_target(token)
            if normalized:
                out.append(normalized)
    return out


def _generation_mode(args: argparse.Namespace) -> str:
    if args.exhaustive:
        return "exhaustive"
    if args.quick:
        return "quick"
    return "balanced"


def _generation_cap(args: argparse.Namespace) -> int | None:
    if args.exhaustive:
        return None
    return QUICK_MAX_RESULTS if args.quick else BALANCED_MAX_RESULTS


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _run_key(args: argparse.Namespace) -> str:
    payload = {
        "schema": GENERATION_CACHE_SCHEMA,
        "text": _normalized_target(args.text),
        "min_word_len": args.min_word_len,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "min_zipf": args.min_zipf,
        "hints": _normalized_words(args.hint),
        "exclude": _normalized_words(args.exclude),
        "require": _normalized_required_words(args.require),
        "generation_mode": _generation_mode(args),
        "generation_cap": _generation_cap(args),
        "generator": _source_hash(GENERATOR),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def build_generator_command(args: argparse.Namespace, output: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(GENERATOR),
        args.text,
        "--min-word-len", str(args.min_word_len),
        "--min-words", str(args.min_words),
        "--max-words", str(args.max_words),
        "--min-zipf", str(args.min_zipf),
        "--short-word-policy", "common",
        "--top-per-group", "1",
        "--workers", str(args.workers),
        "--export", str(output),
    ]

    cap = _generation_cap(args)
    if cap is None:
        cmd.append("--all-results")
    else:
        cmd += ["--max-results", str(cap)]

    hints = _csv_words(args.hint)
    if hints:
        cmd += ["--contains-any", ",".join(hints)]

    excluded = _csv_words(args.exclude)
    if excluded:
        cmd += ["--exclude", ",".join(excluded)]

    for required in _csv_words(args.require):
        cmd += ["--require", required]

    return cmd


def build_reranker_command(
    args: argparse.Namespace,
    candidates: Path,
    output: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RERANKER),
        str(candidates),
        "--backend", "auto",
        "--workers", str(args.workers),
        "--deep-per-group", "2000" if args.quick else "5000",
        "--beam-width", "128",
        "--phrase-rescore-top", "300",
        "--order-candidates", "16",
        "--top-per-group", "1",
        "--export", str(output),
    ]
    if args.phrase_db is not None:
        cmd += ["--phrase-db", str(args.phrase_db.expanduser().resolve())]
    return cmd


def parse_results(path: Path, top: int) -> list[Result]:
    results: list[Result] = []
    current_word_count: int | None = None
    shown_by_count: dict[int, int] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            section = _SECTION_RE.match(line)
            if section:
                current_word_count = int(section.group("words"))
                continue
            if current_word_count is None or line.startswith("---"):
                continue
            match = _RESULT_RE.match(line)
            if not match:
                continue
            if shown_by_count.get(current_word_count, 0) >= top:
                continue
            results.append(
                Result(
                    word_count=current_word_count,
                    rank=int(match.group("rank")),
                    score=float(match.group("score")),
                    phrase=match.group("phrase").strip(),
                )
            )
            shown_by_count[current_word_count] = shown_by_count.get(current_word_count, 0) + 1

    return results


def _run(cmd: Sequence[str], *, verbose: bool) -> subprocess.CompletedProcess[str]:
    # List-form argv with shell=False (the subprocess default) keeps user text as
    # an argument rather than executable shell syntax. In verbose mode inherit
    # the terminal streams so long searches show progress immediately.
    if verbose:
        proc = subprocess.run(cmd, text=True)
    else:
        proc = subprocess.run(cmd, text=True, capture_output=True)

    if proc.returncode != 0:
        if verbose:
            detail = "See the streamed command output above."
        else:
            detail = (proc.stderr or proc.stdout or "").strip()
            if len(detail) > 2000:
                detail = detail[-2000:]
        raise SystemExit(
            f"Command failed with exit code {proc.returncode}:\n"
            f"  {' '.join(map(str, cmd[:3]))}\n\n{detail}"
        )
    return proc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve and rank exact multi-word anagrams in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python anagram_solver.py "ODITIHNSLSHEEEPT"\n'
            '  python anagram_solver.py "ODITIHNSLSHEEEPT" --hint dont --words 4\n'
            '  python anagram_solver.py "tommarvoloriddle" --hint voldemort --words 4'
        ),
    )
    parser.add_argument("text", help="Letters or text to anagram; spaces/punctuation are ignored")
    parser.add_argument("--hint", action="append", default=[], metavar="WORD[,WORD...]", help="Clue word(s); require at least one to occur")
    parser.add_argument("--exclude", action="append", default=[], metavar="WORD[,WORD...]", help="Word(s) that must not occur")
    parser.add_argument("--require", action="append", default=[], metavar="WORD[,WORD...]", help="Word(s) that must occur")

    parser.add_argument("--words", type=int, metavar="N", help="Exact number of words in the answer")
    parser.add_argument("--min-words", type=int, default=2)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--min-word-len", type=int, default=2)
    parser.add_argument(
        "--min-zipf",
        type=float,
        default=2.7,
        metavar="N",
        help="Minimum unigram frequency; lower values admit rarer words, 0 disables filtering",
    )
    parser.add_argument("--top", type=int, default=10, metavar="N", help="Results shown per word-count bucket")
    parser.add_argument("--workers", type=int, default=0, metavar="N", help="Generator/reranker workers; 0 chooses automatically")
    parser.add_argument("--phrase-db", type=Path, metavar="FILE", help="Optional Wikimedia phrase SQLite index")
    search_mode = parser.add_mutually_exclusive_group()
    search_mode.add_argument(
        "--quick",
        action="store_true",
        help=f"Fast exploratory search; cap generation at {QUICK_MAX_RESULTS:,} word bags and may miss the answer",
    )
    search_mode.add_argument(
        "--exhaustive",
        action="store_true",
        help="Enumerate every matching word bag; complete but potentially much slower",
    )
    parser.add_argument("--rebuild", action="store_true", help="Regenerate the cached candidate export")
    parser.add_argument("--verbose", action="store_true", help="Show generator/reranker diagnostic output live")
    parser.add_argument("--json", action="store_true", help="Emit final results as JSON")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        metavar="DIR",
        help="Cache directory for intermediate candidate/reranked exports",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.words is not None:
        if args.words < 1:
            raise SystemExit("--words must be >= 1")
        args.min_words = args.words
        args.max_words = args.words
    if args.min_words < 1 or args.max_words < args.min_words:
        raise SystemExit("Invalid --min-words/--max-words range")
    if args.min_word_len < 1:
        raise SystemExit("--min-word-len must be >= 1")
    if args.min_zipf < 0:
        raise SystemExit("--min-zipf must be >= 0")
    if args.top < 1:
        raise SystemExit("--top must be >= 1")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.json and args.verbose:
        raise SystemExit("--json cannot be combined with --verbose; verbose child output would corrupt JSON stdout")
    if args.phrase_db is not None and not args.phrase_db.expanduser().is_file():
        raise SystemExit(f"--phrase-db not found: {args.phrase_db}")


def _print_results(args: argparse.Namespace, results: Sequence[Result], run_dir: Path) -> None:
    if args.json:
        print(
            json.dumps(
                {
                    "target": args.text,
                    "results": [
                        {
                            "word_count": result.word_count,
                            "rank": result.rank,
                            "score": result.score,
                            "phrase": result.phrase,
                        }
                        for result in results
                    ],
                },
                indent=2,
            )
        )
        return

    print(f"\nAnagramSolver results for: {args.text}")
    if not results:
        print("  No deep-ranked results were produced.")
        return

    last_count: int | None = None
    for result in results:
        if result.word_count != last_count:
            last_count = result.word_count
            print(f"\n{last_count} words")
        print(f"  {result.rank:>2}. {result.phrase:<48} score {result.score:6.2f}")

    if args.verbose:
        print(f"\nCached run files: {run_dir}")


def _generate_candidates(args: argparse.Namespace, candidates: Path) -> float:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    run_dir = args.work_root.expanduser() / _run_key(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = run_dir / "candidates.txt"
    reranked = run_dir / "reranked.txt"

    if args.rebuild or not candidates.is_file():
        if not args.json:
            mode = _generation_mode(args)
            cap = _generation_cap(args)
            if cap is None:
                detail = "unlimited exact enumeration"
            else:
                detail = f"up to {cap:,} candidate bags"
            print(f"Generating exact candidate word bags ({mode}; {detail}) ...")
        generation_seconds = _generate_candidates(args, candidates)
        if not args.json:
            print(f"Candidate generation finished in {generation_seconds:.2f}s.")
    elif not args.json:
        print("Using cached candidate word bags ...")

    if not args.json:
        print("Ranking candidate phrases ...")
    _run(build_reranker_command(args, candidates, reranked), verbose=args.verbose)

    results = parse_results(reranked, args.top)
    _print_results(args, results, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
