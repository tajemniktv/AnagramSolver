"""Durable, sequential Python reference measurements (no native speedup claim)."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".codex/temp/reference-runtime"))
from reference_fixtures import verify_sources
from capture_reference import REFERENCE


def stages():
    import anagram_solver as solver
    runs = []
    original = solver._run
    for text in ["knowledgeispower", "testing"]:
        for strategy in ["prefix", "diverse"]:
            for sample in range(1, 4):
                with tempfile.TemporaryDirectory(dir=ROOT / ".codex/temp", prefix="reference-stage-") as temp:
                    timings = []

                    def measured(command, *, verbose):
                        start = time.perf_counter()
                        result = original(command, verbose=verbose)
                        timings.append(dict(stage="generation" if "generate" in Path(command[1]).stem else "ranking",
                                            seconds=time.perf_counter() - start,
                                            command=[str(arg).replace(temp, "<fresh-puzzle-cache>") for arg in command]))
                        return result

                    solver._run = measured
                    output = io.StringIO()
                    argv = [text, "--quick", "--max-results", "100", "--workers", "2",
                            "--search-strategy", strategy, "--work-root", temp, "--json"]
                    start = time.perf_counter()
                    try:
                        with redirect_stdout(output):
                            assert solver.main(argv) == 0
                    finally:
                        solver._run = original
                    result = json.loads(output.getvalue())
                    runs.append(dict(text=text, strategy=strategy, sample=sample, argv=argv[:-2] + ["<fresh-puzzle-cache>", "--json"],
                                     seconds=time.perf_counter() - start, stages=timings,
                                     search=result["search"], results=result["results"]))
                    print(f"Measured stages: {text}/{strategy}/{sample}", flush=True)
    return dict(reference_commit=REFERENCE, candidate_cap=100, samples=3, workers=2,
                cold_definition="fresh puzzle caches; provisioned corpora and warm OS caches",
                timing_definition="Each actual generator/reranker subprocess, including its corpus loading; parent overhead separate. No RSS attribution per stage.",
                runs=runs)


def gates():
    records = []
    # The first two are the existing quality gate and informational refinement A/B.
    # Phrase and learned checks use synthetic data, not a downloaded Wikimedia corpus.
    commands = [["-m", "unittest", "discover", "-s", "tests"],
                ["ci_ordering_gate.py"], ["ci_order_refinement.py"],
                ["tests/parity/phrase.py"], ["tests/parity/learned.py"]]
    for command in commands:
        start = time.perf_counter()
        result = subprocess.run([sys.executable, *command], cwd=ROOT, text=True, encoding="utf-8",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900,
                                env={**os.environ, "ANAGRAM_INTEGRATION": "1", "PYTHONIOENCODING": "utf-8"})
        records.append(dict(command=command, integration_enabled=True, exit_code=result.returncode,
                            seconds=time.perf_counter() - start, output=result.stdout))
        print(f"Reference gate: {' '.join(command)}: {result.returncode}", flush=True)
    return dict(reference_commit=REFERENCE, passed=all(r["exit_code"] == 0 for r in records), runs=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["stages", "gates"])
    args = parser.parse_args()
    temp = ROOT / ".codex/temp"
    temp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temp)
    os.environ.update(TEMP=str(temp), TMP=str(temp))
    verify_sources()
    result = stages() if args.mode == "stages" else gates()
    (ROOT / f"tests/reference/{args.mode}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if result.get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
