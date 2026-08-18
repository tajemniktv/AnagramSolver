#!/usr/bin/env python3
"""Exercise the real one-command CLI with ordinary default-user invocations."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SmokeCase:
    target: str
    expected_phrase: str | None = None


CASES = (
    SmokeCase("YOSCOZ", "so cozy"),
    SmokeCase("OEEEVHYNRI"),
    SmokeCase("AHCWSOPSIO"),
)

_RESULT_RE = re.compile(r"^\s+\d+\.\s+.+?\s+score\s+\d", re.MULTILINE)


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
    if _RESULT_RE.search(completed.stdout) is None:
        raise SystemExit(f"Normal user run {case.target} produced no displayed result")
    if case.expected_phrase is not None and case.expected_phrase not in completed.stdout.lower():
        raise SystemExit(
            f"Normal user run {case.target} did not surface expected phrase: "
            f"{case.expected_phrase}"
        )


def main() -> int:
    for case in CASES:
        run_case(case)
    print(f"\nNormal user CLI smoke passed for {len(CASES)} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
