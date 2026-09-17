"""Equal-workload serial Python/native CLI time and process-tree RSS measurements."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".codex/temp"
sys.path.insert(0, str(ROOT / "tests/parity"))
from registry import native_request, pretty_phrase, verify_sources
from benchmark_native_generation import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--engine", choices=("python", "native"), action="append")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/reference/native-cli-performance.json")
    args = parser.parse_args()
    if args.samples < 1: parser.error("samples must be positive")
    verify_sources()
    TEMP.mkdir(parents=True, exist_ok=True)
    os.environ.update(TEMP=str(TEMP), TMP=str(TEMP), PYTHONIOENCODING="utf-8")
    reference = json.loads((ROOT / "tests/reference/registry.json").read_text())
    cases = reference["normal_user_cli"]
    if args.case:
        if not set(args.case) <= {case["id"] for case in cases}: parser.error("unknown case")
        cases = [case for case in cases if case["id"] in args.case]
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "anagram-cli"], cwd=ROOT, check=True)
    executable = ROOT / "target/release" / ("anagram-cli.exe" if sys.platform == "win32" else "anagram-cli")
    policy = json.loads((ROOT / ".anagram_data/dictionary/normal_user_v2.json").read_text())
    native = [str(executable), "solve", str(ROOT / ".anagram_data/dictionary/normal_user_v2.txt"),
        str(ROOT / ".anagram_data/ngrams/count_1w.txt"), str(ROOT / ".anagram_data/ngrams/count_2w.txt"),
        str(ROOT / ".anagram_data/wordnet31/dict")]
    report = dict(reference_commit=reference["reference_commit"], machine=platform.platform(),
        python=sys.version, cpu=platform.processor(), native_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        registry_sha256=hashlib.sha256((ROOT / "tests/reference/registry.json").read_bytes()).hexdigest(),
        methodology="Serial fresh processes, alternating engine order per sample; Python workers=1; cold means fresh puzzle caches, not cold OS/corpus pages. Warm native reuses completed generation/ranking after fresh exact corpus identity checks; Python reuses generation and ranking. Elapsed time includes startup, parsing and JSON transport. Sampled summed process-tree RSS may double-count shared pages. No parallel speedup claim.",
        samples=args.samples, cases=[case["id"] for case in cases], complete=False, runs=[])
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        for case in cases:
            payload = json.dumps(native_request(case, policy)).encode()
            for sample in range(args.samples):
                for engine in (("python", "native") if sample % 2 == 0 else ("native", "python")):
                    if args.engine and engine not in args.engine: continue
                    with tempfile.TemporaryDirectory(prefix="cli-benchmark-", dir=TEMP) as temp:
                        command = ([sys.executable, str(ROOT / "anagram_solver.py"), case["target"], *case["options"],
                            "--workers", "1", "--work-root", temp, "--json"] if engine == "python" else
                            [*native, "--cache", str(Path(temp) / "cache.sqlite")])
                        for mode in ("cold", "warm"):
                            values, measurement = measure(command, payload if engine == "native" else b"", process_tree=True, timeout=600, json_lines=False)
                            assert len(values) == 1
                            value = values[0]
                            if engine == "python":
                                rows = value["results"]
                                generated, deep = value["search"]["generated"], value["search"]["deep_analyzed"]
                                assert value["search"]["generation_cached"] == (mode == "warm")
                                assert value["search"]["ranking_cached"] == (mode == "warm")
                            else:
                                rows = [dict(word_count=int(wc), rank=i, score=float(f"{row['final']:.2f}"), phrase=pretty_phrase(row["best_order"]))
                                    for wc, bucket in sorted(value["buckets"].items(), key=lambda p: int(p[0])) for i, row in enumerate(bucket, 1)]
                                generated, deep = value["generated"], value["deep_analyzed"]
                                assert value["status"]["cache"]["hit"] == (mode == "warm")
                            expected = case["cold"]
                            assert rows == expected["results"], (case["id"], engine, mode, "rankings")
                            assert (generated, deep) == (expected["search"]["generated"], expected["search"]["deep_analyzed"])
                            report["runs"].append(dict(case=case["id"], sample=sample, engine=engine, mode=mode, generated=generated,
                                deep_analyzed=deep, **measurement))
                            save()
                            print(f"{case['id']} {engine} {mode} {sample+1}: {measurement['seconds']:.3f}s", flush=True)
        report["summary"] = [dict(case=case["id"], engine=engine, mode=mode,
            median_seconds=statistics.median(row["seconds"] for row in report["runs"] if (row["case"], row["engine"], row["mode"]) == (case["id"], engine, mode)),
            peak_tree_rss_bytes=max(row["sampled_peak_rss_bytes"] for row in report["runs"] if (row["case"], row["engine"], row["mode"]) == (case["id"], engine, mode)))
            for case in cases for engine in ("python", "native") if not args.engine or engine in args.engine for mode in ("cold", "warm")]
        report["complete"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
