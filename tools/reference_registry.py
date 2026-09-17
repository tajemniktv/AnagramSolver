"""Freeze every normal-user CLI case and the complete ordering registry."""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "archive"))
sys.path.insert(0, str(ROOT / ".codex/temp/reference-runtime"))
from capture_reference import REFERENCE
from reference_fixtures import verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_sources()
    temp_root = ROOT / ".codex/temp"
    tempfile.tempdir = str(temp_root)
    os.environ.update(TEMP=str(temp_root), TMP=str(temp_root), PYTHONIOENCODING="utf-8")
    from anagram_suite import cases_for, normal_user_case
    import anagram_benchmark as benchmark
    import anagram_rerank as ranking
    from benchmark_user_runs import measure

    lex = ranking.WordNetLexicon.load(ROOT / ".anagram_data/wordnet31/dict")
    with redirect_stdout(io.StringIO()):
        ordering = [asdict(benchmark.run_order_case(ranking, lex, case))
                    for case in cases_for("ordering")]
    records = []
    measurements = []
    for case in map(normal_user_case, cases_for("normal_user_cli")):
        # JSON replaces verbose diagnostics only; all semantic/default budgets stay.
        options = [option for option in case.solver_args if option != "--verbose"]
        with tempfile.TemporaryDirectory(dir=temp_root, prefix="registry-reference-") as temp:
            command = [sys.executable, str(ROOT / "archive/anagram_solver.py"), case.target,
                       *options, "--work-root", temp, "--json"]
            cold, elapsed, peak = measure(command, case.timeout_seconds)
            warm, warm_elapsed, warm_peak = measure(command, case.timeout_seconds)
            assert cold["results"] == warm["results"], case.id
            assert cold["results"], f"No displayed results: {case.id}"
            if case.expected_phrase:
                assert any(case.expected_phrase.lower() in row["phrase"].lower()
                           for row in cold["results"]), case.id
            assert not cold["search"]["generation_cached"], case.id
            assert warm["search"]["generation_cached"], case.id
            assert warm["search"]["ranking_cached"], case.id
            # Only cached computation time is volatile; retain semantic counts,
            # exhaustion and explicit cold/warm flags in the golden fixture.
            for value in [cold, warm]:
                value["search"].pop("seconds", None)
            records.append(dict(id=case.id, target=case.target, options=options,
                                expected_phrase=case.expected_phrase, cold=cold, warm=warm))
            measurements.append(dict(id=case.id, cold_seconds=elapsed, warm_seconds=warm_elapsed,
                                     cold_peak_tree_rss_bytes=peak, warm_peak_tree_rss_bytes=warm_peak))
            print(f"Captured default-path cold/warm registry case: {case.id}", flush=True)
    value = dict(reference_commit=REFERENCE, ordering=ordering, normal_user_cli=records)
    destination = ROOT / "tests/reference/registry.json"
    if args.check:
        assert json.loads(destination.read_text(encoding="utf-8")) == value, "Registry oracle drift"
    else:
        destination.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        (ROOT / "tests/reference/registry-measurements.json").write_text(
            json.dumps(dict(reference_commit=REFERENCE, runs=measurements), indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(ordering)} ordering and {len(records)} default CLI cases")


if __name__ == "__main__":
    main()
