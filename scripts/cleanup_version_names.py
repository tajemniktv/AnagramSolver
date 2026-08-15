#!/usr/bin/env python3
"""One-shot repository cleanup for historical solver version labels."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RENAMES = {
    "multi_anagram_v8_analyzer.py": "anagram_generate.py",
}

COMMON_REPLACEMENTS = {
    "multi_anagram_v8_analyzer.py": "anagram_generate.py",
    "anagram_rerank_v13.py": "anagram_rerank.py",
    "v8_ALL.txt": "candidates.txt",
    "_v8_ALL.txt": "_candidates.txt",
    "v13_RERANKED.txt": "reranked.txt",
    "V8_LINE_RE": "GENERATOR_LINE_RE",
    "parse_v8": "parse_candidates",
    "score_pre_v13": "score_pre",
    "score_final_v13": "score_final",
    "v13_pre": "pre_score",
    "PRE13": "PRE",
    "V13 PRE rank:": "PRE rank:",
    "V13 FINAL rank:": "FINAL rank:",
    'PREPARED_CACHE_SCHEMA = "v13-prepared-1"': 'PREPARED_CACHE_SCHEMA = "core-prepared-1"',
    'anagram-rerank-v13/1.0': 'anagram-solver/1.0',
}

CORE_DOCSTRING = '''"""
Core linguistic reranker for exact multi-word anagram candidate exports.

The reranker combines lexical/commonness evidence, morphology families, clue
informativeness, WordNet POS/morphology coverage, lightweight English grammar,
WordNet verb frames/valency, and whole-phrase structure. Short bags can be
ordered exactly; longer bags use k-best search before the more expensive global
structure scorer. Independent candidate bags can be analyzed across CPU cores.

Sparse corpus evidence is positive-only: missing n-grams are treated as unknown,
not as negative evidence. A benchmark answer, when supplied, is used only after
ranking and never contributes to scoring.

Python standard library only. WordNet is downloaded on first use and cached
under ~/.multi_anagram/wordnet31.
"""'''

GENERATOR_DOCSTRING = '''"""
Exact multi-word anagram candidate generator and lexical pre-ranker.

The generator searches exact letter decompositions, applies lexical/clue filters,
groups morphology variants, and emits a stable candidate export consumed by the
linguistic reranker. Optional Norvig unigram/bigram data is cached under
~/.multi_anagram/ngrams.

Python standard library only.
"""'''

TOPK_DOCSTRING = '''"""
Top-k word-order search and phrase-evidence reranking layer.

For each unordered word bag this layer retains several strong grammatical
orders, then allows positive phrase/collocation evidence to choose among them.
Input bags are canonicalized before search so tie-breaking is deterministic and
independent of generator emission order.
"""'''


def replace_module_docstring(text: str, replacement: str) -> str:
    start = text.find('"""')
    if start < 0:
        return text
    end = text.find('"""', start + 3)
    if end < 0:
        return text
    return text[:start] + replacement + text[end + 3 :]


def clean_benchmark(text: str) -> str:
    text = text.replace("DEFAULT_V8", "DEFAULT_GENERATOR")
    text = text.replace("make_v8_command", "make_generator_command")
    text = re.sub(r"\bv12\b", "reranker", text)
    text = re.sub(r"\bv8\b", "generator", text)
    text = text.replace('        "--v12",\n', "")
    text = text.replace('ap.add_argument("--v8", type=Path, default=DEFAULT_GENERATOR)',
                        'ap.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)')
    text = text.replace("args.v8", "args.generator")
    text = text.replace("v8=args.generator", "generator=args.generator")
    text = text.replace('FINAL_RE = re.compile(r"V\\d+ FINAL rank:',
                        'FINAL_RE = re.compile(r"FINAL rank:')
    text = text.replace('PRE_RE = re.compile(r"V\\d+ PRE rank:',
                        'PRE_RE = re.compile(r"PRE rank:')
    text = text.replace('raise SystemExit(f"V8 generator not found: {args.generator}")',
                        'raise SystemExit(f"Generator not found: {args.generator}")')
    return text


def clean_core(text: str) -> str:
    text = replace_module_docstring(text, CORE_DOCSTRING)
    text = text.replace("# V13 performance/runtime helpers", "# Performance/runtime helpers")
    text = text.replace(
        "# V10-V12 intentionally used coarse buckets. The regression suite exposed that\n"
        "# some of those buckets were semantically wrong: frames 6/7 are predicative\n",
        "# Earlier revisions used coarse frame buckets. The regression suite exposed that\n"
        "# frames 6/7 are predicative\n",
    )
    text = text.replace("V8 export", "generator export")
    text = text.replace("V8 exports", "generator exports")
    text = text.replace("V8 row", "generator row")
    text = text.replace("V13", "current reranker")
    text = text.replace("V12", "earlier reranker")
    text = text.replace("V11", "earlier reranker")
    text = text.replace("V10", "earlier reranker")
    text = text.replace("V9", "earlier reranker")
    text = text.replace("V8", "generator")
    return text


def clean_generator(text: str) -> str:
    text = replace_module_docstring(text, GENERATOR_DOCSTRING)
    text = text.replace("V8", "generator")
    text = text.replace("v8", "generator")
    return text


def clean_topk(text: str) -> str:
    text = replace_module_docstring(text, TOPK_DOCSTRING)
    text = text.replace("V13", "core reranker")
    text = text.replace("V12", "earlier reranker")
    text = text.replace("V11", "earlier reranker")
    text = text.replace("V10", "earlier reranker")
    text = text.replace("V9", "earlier reranker")
    text = text.replace("V8", "generator")
    return text


def clean_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    for old, new in COMMON_REPLACEMENTS.items():
        text = text.replace(old, new)

    if path.name == "anagram_benchmark.py":
        text = clean_benchmark(text)
    elif path.name == "anagram_rerank_core.py":
        text = clean_core(text)
    elif path.name == "anagram_generate.py":
        text = clean_generator(text)
    elif path.name == "anagram_rerank_topk_impl.py":
        text = clean_topk(text)
    else:
        text = text.replace("V13", "current reranker")
        text = text.replace("V12", "earlier reranker")
        text = text.replace("V11", "earlier reranker")
        text = text.replace("V10", "earlier reranker")
        text = text.replace("V9", "earlier reranker")
        text = text.replace("V8", "generator")

    if text != original:
        path.write_text(text, encoding="utf-8")


def main() -> int:
    for old_name, new_name in RENAMES.items():
        old = ROOT / old_name
        new = ROOT / new_name
        if old.exists() and not new.exists():
            old.rename(new)

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".git/"):
            continue
        if rel in {
            "scripts/cleanup_version_names.py",
            ".github/workflows/cleanup-version-names.yml",
        }:
            continue
        if path.suffix not in {".py", ".json", ".md", ".yml", ".yaml", ".txt"}:
            continue
        clean_file(path)

    # Catch solver-lineage labels that survived explicit cleanup. External action
    # comments such as '# v4'/'# v5' are intentionally outside this 8..13 range.
    leftover_re = re.compile(r"(?<![A-Za-z0-9])(?:V|v)(?:8|9|10|11|12|13)(?![0-9])")
    leftovers: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith(".git/") or rel in {
            "scripts/cleanup_version_names.py",
            ".github/workflows/cleanup-version-names.yml",
        }:
            continue
        if path.suffix not in {".py", ".json", ".md", ".yml", ".yaml", ".txt"}:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if leftover_re.search(line):
                leftovers.append(f"{rel}:{lineno}: {line.strip()}")

    versioned_names = [
        p.relative_to(ROOT).as_posix()
        for p in ROOT.rglob("*")
        if p.is_file()
        and re.search(r"(?:^|[_-])v(?:8|9|10|11|12|13)(?:[_-]|\.)", p.name, re.I)
        and p.name != "cleanup_version_names.py"
    ]

    if leftovers or versioned_names:
        print("Historical solver-version labels remain:")
        for item in leftovers:
            print("  ", item)
        for item in versioned_names:
            print("  filename:", item)
        return 1

    print("Historical solver-version labels removed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
