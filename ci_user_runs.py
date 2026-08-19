#!/usr/bin/env python3
"""Exercise the real one-command CLI with ordinary default-user invocations."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SmokeCase:
    target: str
    expected_phrase: str | None = None


CASES = (
    SmokeCase("YOSCOZ", "so cozy"),
    SmokeCase("OEEEVHYNRI", "hi everyone"),
    # This remains a completion smoke while the research beam experiment measures
    # whether the intended bag can be retained generically without target tuning.
    SmokeCase("ODITIHNSLSHEEEPT"),
    # Proper names are intentionally still a separate lexical-source problem.
    SmokeCase("AHCWSOPSIO"),
)

_RESULT_RE = re.compile(r"^\s+\d+\.\s+.+?\s+score\s+\d", re.MULTILINE)
_RUN_DIR_RE = re.compile(r"^Cached run files:\s*(?P<path>.+)$", re.MULTILINE)
_CANDIDATE_PHRASE_RE = re.compile(
    r"PCOV=\s*[\d.]+\s{2}(?P<phrase>.*?)\s{2}\[HINT=",
)
_CANON_RE = re.compile(r"\[CANON=(?P<phrase>[^;\]]+)")
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")


def _displayed_result_lines(output: str, target: str) -> tuple[str, ...]:
    """Return only final user-facing result lines, excluding generator diagnostics."""
    marker = f"AnagramSolver results for: {target}"
    _, separator, final_output = output.partition(marker)
    if not separator:
        return ()
    return tuple(
        line
        for line in final_output.splitlines()
        if _RESULT_RE.match(line)
    )


def _normalized_bag(phrase: str) -> tuple[str, ...]:
    words = []
    for token in _TOKEN_RE.findall(phrase.lower()):
        normalized = "".join(ch for ch in token if "a" <= ch <= "z")
        if normalized:
            words.append(normalized)
    return tuple(sorted(words))


def _file_contains_bag(
    path: Path,
    expected: tuple[str, ...],
    pattern: re.Pattern[str],
) -> tuple[bool, int]:
    """Return target presence plus number of lines understood by the parser."""
    if not path.is_file():
        return False, 0
    matched_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = pattern.search(line)
            if match is None:
                continue
            matched_lines += 1
            if _normalized_bag(match.group("phrase")) == expected:
                return True, matched_lines
    return False, matched_lines


def _dropout_diagnostic(output: str, phrase: str) -> str:
    run_match = _RUN_DIR_RE.search(output)
    if run_match is None:
        return "run directory unavailable"
    run_dir = Path(run_match.group("path").strip())
    expected = _normalized_bag(phrase)
    generated, candidate_lines = _file_contains_bag(
        run_dir / "candidates.txt",
        expected,
        _CANDIDATE_PHRASE_RE,
    )
    deep_exported, reranked_lines = _file_contains_bag(
        run_dir / "reranked.txt",
        expected,
        _CANON_RE,
    )
    return (
        f"candidate_present={generated}; candidate_lines_parsed={candidate_lines}; "
        f"reranked_present={deep_exported}; reranked_lines_parsed={reranked_lines}"
    )


def run_case(case: SmokeCase) -> None:
    cmd = [sys.executable, "anagram_solver.py", case.target, "--verbose"]
    print(f"\n=== normal user run: {' '.join(cmd)} ===", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        raise SystemExit(f"Normal user run timed out after 120s: {case.target}") from exc

    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(
            f"Normal user run failed with exit code {completed.returncode}: {case.target}"
        )

    required_markers = (
        "Generating exact candidate word bags",
        "Ranking candidate phrases",
        "=== linguistic reranker TIMINGS ===",
        f"AnagramSolver results for: {case.target}",
    )
    missing = [marker for marker in required_markers if marker not in completed.stdout]
    if missing:
        raise SystemExit(
            f"Normal user run {case.target} missed output marker(s): {missing}"
        )

    result_lines = _displayed_result_lines(completed.stdout, case.target)
    if not result_lines:
        raise SystemExit(f"Normal user run {case.target} produced no displayed result")
    if case.expected_phrase is not None and not any(
        case.expected_phrase in line.lower() for line in result_lines
    ):
        diagnostic = _dropout_diagnostic(completed.stdout, case.expected_phrase)
        raise SystemExit(
            f"Normal user run {case.target} did not surface expected final phrase: "
            f"{case.expected_phrase} ({diagnostic})"
        )


def main() -> int:
    for case in CASES:
        run_case(case)
    print(f"\nNormal user CLI smoke passed for {len(CASES)} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
