#!/usr/bin/env python3
"""
build_wikimedia_phrase_index.py

Build a local phrase-title index compatible with anagram_rerank_v13.py's
--phrase-db option, using Wikimedia's official main-namespace title dumps.

Default source:
  English Wiktionary titles (~tens of MB compressed)

Optional:
  English Wikipedia titles (~100+ MB compressed)

Why titles?
-----------
Nutrimatic's strongest feature is corpus-attested phrases. Building its full
Wikipedia occurrence index is a multi-hour / tens-of-GB job. Wikimedia title
dumps are a much cheaper middle ground:

* Wiktionary contains many idioms, proverbs and lexicalized phrases.
* Wikipedia contains song, film, book, place and other proper-name/title phrases.
* Exact title presence is POSITIVE evidence only. Absence never penalizes a
  candidate.

The resulting SQLite schema matches build_phrase_index.py:
    ngrams(text TEXT PRIMARY KEY, n INTEGER, count INTEGER)

Example:
  python build_wikimedia_phrase_index.py --output wikimedia_phrases.db

Add Wikipedia too:
  python build_wikimedia_phrase_index.py --include-wikipedia ^
      --output wikimedia_phrases.db

Then:
  python anagram_rerank_v13.py v8_ALL.txt ^
      --phrase-db wikimedia_phrases.db ...
"""

from __future__ import annotations

import argparse
import gzip
import re
import sqlite3
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path


SOURCES = {
    "enwiktionary": (
        "https://dumps.wikimedia.org/enwiktionary/latest/"
        "enwiktionary-latest-all-titles-in-ns0.gz"
    ),
    "enwiki": (
        "https://dumps.wikimedia.org/enwiki/latest/"
        "enwiki-latest-all-titles-in-ns0.gz"
    ),
}

DEFAULT_CACHE = Path.home() / ".multi_anagram" / "wikimedia_titles"


def norm_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def tokenize_title(text: str) -> list[str]:
    text = text.replace("_", " ")
    return [
        norm_token(x)
        for x in re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
        if norm_token(x)
    ]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "anagram-wikimedia-phrase-index/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last = time.perf_counter()
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.perf_counter()
            if now - last >= 1.0:
                if total:
                    print(
                        f"\r  {done/1024/1024:,.1f} / {total/1024/1024:,.1f} MiB "
                        f"({100*done/total:5.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  {done/1024/1024:,.1f} MiB",
                        end="",
                        flush=True,
                    )
                last = now
        print()


def init_db(conn: sqlite3.Connection, rebuild: bool) -> None:
    if rebuild:
        conn.execute("DROP TABLE IF EXISTS ngrams")
        conn.execute("DROP TABLE IF EXISTS meta")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ngrams(
            text TEXT PRIMARY KEY,
            n INTEGER NOT NULL,
            count INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()


def flush(conn: sqlite3.Connection, counter: Counter[tuple[str, int]]) -> int:
    if not counter:
        return 0
    rows = [(text, n, count) for (text, n), count in counter.items()]
    conn.executemany(
        """
        INSERT INTO ngrams(text,n,count)
        VALUES(?,?,?)
        ON CONFLICT(text) DO UPDATE SET
            count = count + excluded.count,
            n = MAX(n, excluded.n)
        """,
        rows,
    )
    conn.commit()
    counter.clear()
    return len(rows)


def emit_title(
    counter: Counter[tuple[str, int]],
    tokens: list[str],
    *,
    max_ngram: int,
) -> int:
    if len(tokens) < 2:
        return 0

    emitted = 0
    whole = " ".join(tokens)
    counter[(whole, len(tokens))] += 1
    emitted += 1

    upper = min(max_ngram, len(tokens))
    for n in range(2, upper + 1):
        if n == len(tokens):
            continue
        for i in range(len(tokens) - n + 1):
            counter[(" ".join(tokens[i : i + n]), n)] += 1
            emitted += 1
    return emitted


def ingest_source(
    conn: sqlite3.Connection,
    gz_path: Path,
    source_name: str,
    *,
    min_words: int,
    max_words: int,
    max_ngram: int,
    flush_unique: int,
) -> tuple[int, int]:
    counter: Counter[tuple[str, int]] = Counter()
    titles = 0
    accepted = 0
    emitted = 0
    t0 = time.perf_counter()

    print(f"Indexing {source_name}: {gz_path}")

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            titles += 1
            title = line.strip()
            if not title or title == "page_title":
                continue

            tokens = tokenize_title(title)
            if not (min_words <= len(tokens) <= max_words):
                continue

            accepted += 1
            emitted += emit_title(counter, tokens, max_ngram=max_ngram)

            if len(counter) >= flush_unique:
                flush(conn, counter)

            if titles % 500_000 == 0:
                elapsed = max(1e-9, time.perf_counter() - t0)
                print(
                    f"  titles={titles:,} accepted={accepted:,} "
                    f"({titles/elapsed:,.0f} titles/s)"
                )

    flush(conn, counter)
    elapsed = time.perf_counter() - t0
    print(
        f"  done: {titles:,} titles, {accepted:,} multiword accepted, "
        f"{emitted:,} emitted n-gram observations in {elapsed:.1f}s"
    )
    return titles, accepted


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a lightweight Wikimedia phrase-title SQLite index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--output", type=Path, default=Path("wikimedia_phrases.db"))
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--include-wikipedia", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=8)
    ap.add_argument("--max-ngram", type=int, default=5)
    ap.add_argument("--flush-unique", type=int, default=100000)
    args = ap.parse_args()

    if args.min_words < 2:
        raise SystemExit("--min-words must be >= 2")
    if args.max_words < args.min_words:
        raise SystemExit("--max-words must be >= --min-words")
    if args.max_ngram < 2:
        raise SystemExit("--max-ngram must be >= 2")

    projects = ["enwiktionary"]
    if args.include_wikipedia:
        projects.append("enwiki")

    cache = args.cache_dir.expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    source_files: list[tuple[str, Path]] = []
    for project in projects:
        url = SOURCES[project]
        filename = url.rsplit("/", 1)[-1]
        path = cache / filename
        if args.refresh and path.exists():
            path.unlink()
        if not path.exists():
            download(url, path)
        else:
            print(f"Using cached {path}")
        source_files.append((project, path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.output)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        init_db(conn, args.rebuild)

        total_titles = 0
        total_accepted = 0
        for project, path in source_files:
            titles, accepted = ingest_source(
                conn,
                path,
                project,
                min_words=args.min_words,
                max_words=args.max_words,
                max_ngram=args.max_ngram,
                flush_unique=args.flush_unique,
            )
            total_titles += titles
            total_accepted += accepted

        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('sources',?)",
            (",".join(projects),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('kind','wikimedia_titles')"
        )
        conn.commit()

        print("Creating lookup index / compacting database ...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_n ON ngrams(n)")
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")

        rows = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
        max_n = conn.execute("SELECT COALESCE(MAX(n),0) FROM ngrams").fetchone()[0]
        print(
            f"Complete: {rows:,} unique phrase/n-gram rows, max n={max_n}, "
            f"from {total_titles:,} titles ({total_accepted:,} accepted)."
        )
        print(f"Database: {args.output.resolve()}")
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
