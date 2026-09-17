"""Capture source/corpus identities for reproducible Python/Rust comparisons.

Writes only to the requested output (default: project-local .codex/temp).
Hashes corpus content rather than treating an mtime as a version identifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".codex/temp/reference-manifest.json")
    args = parser.parse_args()
    sources = sorted(ROOT.glob("anagram*.py"))
    sources += [ROOT / "anagram_benchmarks.json"]
    corpora = []
    for folder in ("dictionary", "ngrams", "wordnet31"):
        directory = ROOT / ".anagram_data" / folder
        corpora.extend(sorted(p for p in directory.rglob("*") if p.is_file()))
    result = {
        "schema_version": 1,
        "reference_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_files": [identity(path) for path in sources],
        "corpus_files": [identity(path) for path in corpora],
        "machine": {"platform": platform.platform(), "architecture": platform.machine(),
                    "python": platform.python_version()},
        "notes": ["Snapshot of available corpora, not permission to redistribute them.",
                  "Optional phrase databases and learned models must be recorded separately when used.",
                  "Source hashes identify dirty-worktree content independently of the reference commit."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Captured {len(sources)} source files and {len(corpora)} corpus files: {args.output}")


if __name__ == "__main__":
    main()
