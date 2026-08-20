#!/usr/bin/env python3
"""Exercise registry-selected cases through the real one-command user CLI."""

from __future__ import annotations

import re
import subprocess
import sys

from anagram_suite import NormalUserCase, cases_for, normal_user_case

_RESULT_RE = re.compile(r"^\s+\d+\.\s+.+?\s+score\s+\d", re.MULTILINE)


def _displayed_result_lines(output: str, target: str) -> tuple[str, ...]:
    """Return only final user-facing result lines, excluding generator diagnostics."""
    marker = f"AnagramSolver results for: {target}"
    _, separator, final_output = output.partition(marker)
    if not separator:
        return ()
    return tuple(line for line in final_output.splitlines() if _RESULT_RE.match(line))


def run_case(case: NormalUserCase) -> None:
    cmd = [sys.executable, "anagram_solver.py", case.target, *case.solver_args]
    print(f"\n=== normal user run [{case.id}]: {' '.join(cmd)} ===", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=case.timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        print(output, end="")
        raise SystemExit(
            f"Normal user run timed out after {case.timeout_seconds}s: {case.id}"
        ) from exc

    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise SystemExit(
            f"Normal user run failed with exit code {completed.returncode}: {case.id}"
        )

    required_markers = [
        "Generating exact candidate word bags",
        "Ranking candidate phrases",
        f"AnagramSolver results for: {case.target}",
    ]
    if case.verbose:
        required_markers.append("=== linguistic reranker TIMINGS ===")
    missing = [marker for marker in required_markers if marker not in completed.stdout]
    if missing:
        raise SystemExit(f"Normal user run {case.id} missed output marker(s): {missing}")

    result_lines = _displayed_result_lines(completed.stdout, case.target)
    if not result_lines:
        raise SystemExit(f"Normal user run {case.id} produced no displayed result")
    if case.expected_phrase is not None and not any(
        case.expected_phrase.lower() in line.lower() for line in result_lines
    ):
        raise SystemExit(
            f"Normal user run {case.id} did not surface expected final phrase: "
            f"{case.expected_phrase}"
        )


def main() -> int:
    cases = [normal_user_case(case) for case in cases_for("normal_user_cli")]
    for case in cases:
        run_case(case)
    print(f"\nNormal user CLI smoke passed for {len(cases)} case(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
