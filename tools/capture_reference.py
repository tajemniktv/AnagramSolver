"""Capture source/corpus identities for reproducible Python/Rust comparisons.

Writes only to the requested output (default: durable tests/reference manifest).
Hashes corpus content rather than treating an mtime as a version identifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "c12e3d5519d653fd65c7ffbbb2a1597941fedd2a"


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/reference/manifest.json")
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--phrase-db", type=Path, action="append", default=[])
    parser.add_argument("--ranker-model", type=Path, action="append", default=[])
    args = parser.parse_args()
    reference=subprocess.check_output(["git","rev-parse",f"{args.reference}^{{commit}}"],cwd=ROOT,text=True).strip()
    tracked=subprocess.check_output(["git","ls-tree","--name-only",reference],cwd=ROOT,text=True).splitlines()
    source_names=sorted(name for name in tracked if name.endswith(".py") or name=="anagram_benchmarks.json")
    subprocess.run(["git","diff","--exit-code",reference,"--",*source_names],cwd=ROOT,check=True,capture_output=True)
    sources=[ROOT/name for name in source_names]
    source_files=[]
    for path in sources:
        item=identity(path)
        item["git_blob"]=subprocess.check_output(["git","rev-parse",f"{reference}:{path.name}"],cwd=ROOT,text=True).strip()
        source_files.append(item)
    corpora = []
    for folder in ("dictionary", "ngrams", "wordnet31"):
        directory = ROOT / ".anagram_data" / folder
        corpora.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    result = {
        "schema_version": 1,
        "reference_commit": reference,
        "source_files": source_files,
        "corpus_files": [identity(path) for path in corpora],
        "optional_inputs": {
            "phrase_databases": [identity(path.resolve()) for path in args.phrase_db],
            "ranker_models": [identity(path.resolve()) for path in args.ranker_model],
            "empty_list_means": "not enabled in this captured baseline; synthetic fixtures are recorded separately"
        },
        "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                    "python": platform.python_version(),"logical_cpus":os.cpu_count(),
                    "processor":os.environ.get("PROCESSOR_IDENTIFIER",platform.processor())},
        "notes": ["Snapshot of available corpora, not permission to redistribute them.",
                  "Oracle source files must match the pinned commit; capture fails on drift.",
                  "Worktree SHA-256 includes platform line endings; git_blob identifies canonical committed source.",
                  "Cache state, execution configuration, counts and timing belong to each benchmark report."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {len(sources)} source files and {len(corpora)} corpus files: {args.output}")


if __name__ == "__main__":
    main()
