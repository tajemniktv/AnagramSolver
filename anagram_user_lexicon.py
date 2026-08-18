"""Cached lexical policy for ordinary one-command solver runs.

The research generator intentionally exposes low-level dictionary and short-word
controls. The normal user frontend applies a safer broader policy: corpus-common
two-letter words are admitted without hard-coding individual puzzle answers,
and standard punctuationless contractions already known to the project are
added when the base dictionary omits them.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anagram_generate as generator
from anagram_paths import DICTIONARY_DIR

USER_LEXICON_SCHEMA = 2
DEFAULT_SHORT_WORD_MIN_ZIPF = 5.0
AUGMENTED_DICTIONARY = DICTIONARY_DIR / f"normal_user_v{USER_LEXICON_SCHEMA}.txt"
POLICY_CACHE = DICTIONARY_DIR / f"normal_user_v{USER_LEXICON_SCHEMA}.json"


@dataclass(frozen=True, slots=True)
class UserLexicon:
    dictionary: Path
    extra_short_words: tuple[str, ...]
    cache_token: str


def _source_stamp(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _stamp_from_json(value: Any) -> tuple[int, int] | None:
    """Validate a serialized source stamp, treating malformed cache data as stale."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return value[0], value[1]


def _policy_source_token() -> str:
    """Hash the code that determines derived user-lexicon contents."""
    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(generator.__file__).resolve()):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def _policy_token(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
    extra_short_words: tuple[str, ...],
) -> str:
    """Return a stable identity for the effective source-backed user lexicon."""
    payload = {
        "schema": USER_LEXICON_SCHEMA,
        "policy_source": _policy_source_token(),
        "dictionary_stamp": list(dictionary_stamp),
        "unigram_stamp": list(unigram_stamp),
        "extra_short_words": list(extra_short_words),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def select_corpus_short_words(
    dictionary: Path,
    unigrams: generator.UnigramModel,
    *,
    min_zipf: float = DEFAULT_SHORT_WORD_MIN_ZIPF,
) -> tuple[str, ...]:
    """Return corpus-common two-letter dictionary words in lexical order."""
    selected: set[str] = set()
    with dictionary.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            word = generator.normalize_token(line)
            if len(word) != 2:
                continue
            if unigrams.zipf(word) >= min_zipf:
                selected.add(word)
    return tuple(sorted(selected))


def build_augmented_dictionary(
    base_dictionary: Path,
    output: Path,
    supplements: set[str] | frozenset[str],
) -> Path:
    """Copy the base word list and append normalized missing supplements."""
    original = base_dictionary.read_text(encoding="utf-8", errors="ignore")
    existing = {
        word
        for line in original.splitlines()
        if (word := generator.normalize_token(line))
    }
    additions = sorted(
        {
            normalized
            for word in supplements
            if (normalized := generator.normalize_token(word))
            and normalized not in existing
        }
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(original)
            if original and not original.endswith("\n"):
                handle.write("\n")
            for word in additions:
                handle.write(word + "\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_cached_policy(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
) -> UserLexicon | None:
    if not AUGMENTED_DICTIONARY.is_file() or not POLICY_CACHE.is_file():
        return None
    try:
        payload = json.loads(POLICY_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != USER_LEXICON_SCHEMA:
        return None
    if payload.get("policy_source") != _policy_source_token():
        return None

    cached_dictionary_stamp = _stamp_from_json(payload.get("dictionary_stamp"))
    cached_unigram_stamp = _stamp_from_json(payload.get("unigram_stamp"))
    if cached_dictionary_stamp != dictionary_stamp:
        return None
    if cached_unigram_stamp != unigram_stamp:
        return None

    short_words = payload.get("extra_short_words")
    if not isinstance(short_words, list) or not all(
        isinstance(word, str) and len(word) == 2 for word in short_words
    ):
        return None
    normalized_short_words = tuple(short_words)
    return UserLexicon(
        AUGMENTED_DICTIONARY,
        normalized_short_words,
        _policy_token(dictionary_stamp, unigram_stamp, normalized_short_words),
    )


def _save_policy(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
    extra_short_words: tuple[str, ...],
) -> None:
    payload = {
        "schema": USER_LEXICON_SCHEMA,
        "policy_source": _policy_source_token(),
        "dictionary_stamp": list(dictionary_stamp),
        "unigram_stamp": list(unigram_stamp),
        "extra_short_words": list(extra_short_words),
    }
    POLICY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = POLICY_CACHE.with_name(f".{POLICY_CACHE.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, POLICY_CACHE)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_user_lexicon(
    *,
    dictionary_source: str = generator.DEFAULT_DICT_URL,
    ngram_dir: Path | str = generator.DEFAULT_NGRAM_DIR,
    refresh: bool = False,
) -> UserLexicon:
    """Provision and return the cached lexicon used by normal solver runs."""
    base_dictionary = generator.get_dictionary(dictionary_source, refresh=refresh)
    unigram_path, _ = generator.ensure_ngram_data(
        Path(ngram_dir).expanduser(),
        refresh=refresh,
        need_bigrams=False,
    )
    dictionary_stamp = _source_stamp(base_dictionary)
    unigram_stamp = _source_stamp(unigram_path)

    cached = _load_cached_policy(dictionary_stamp, unigram_stamp)
    if cached is not None:
        return cached

    unigrams = generator.load_unigram_model(unigram_path)
    extra_short_words = select_corpus_short_words(base_dictionary, unigrams)
    build_augmented_dictionary(
        base_dictionary,
        AUGMENTED_DICTIONARY,
        frozenset(generator.PRETTY_CONTRACTIONS),
    )
    _save_policy(dictionary_stamp, unigram_stamp, extra_short_words)
    return UserLexicon(
        AUGMENTED_DICTIONARY,
        extra_short_words,
        _policy_token(dictionary_stamp, unigram_stamp, extra_short_words),
    )
