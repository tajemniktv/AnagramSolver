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
DEFAULT_SHORT_WORD_MIN_ZIPF: float = 5.0
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


def _normalized_dictionary_source(source: str) -> str:
    if source.lower().startswith(("http://", "https://")):
        return source
    return str(Path(source).expanduser().resolve())


def _dictionary_source_token(dictionary_source: str) -> str:
    """Return a stable namespace for dictionary-derived artifacts."""
    normalized = _normalized_dictionary_source(dictionary_source).encode()
    return hashlib.sha256(normalized).hexdigest()[:20]


def _cache_source_token(dictionary_source: str, ngram_dir: Path | str) -> str:
    """Return a stable namespace for one dictionary/ngram policy combination."""
    payload = {
        "dictionary_source": _normalized_dictionary_source(dictionary_source),
        "ngram_dir": str(Path(ngram_dir).expanduser().resolve()),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def _derived_cache_paths(
    dictionary_source: str,
    ngram_dir: Path | str,
) -> tuple[Path, Path, str]:
    """Choose dependency-scoped derived files so custom sources cannot race."""
    dictionary_token = _dictionary_source_token(dictionary_source)
    policy_token = _cache_source_token(dictionary_source, ngram_dir)
    default_dictionary_token = _dictionary_source_token(generator.DEFAULT_DICT_URL)
    default_policy_token = _cache_source_token(
        generator.DEFAULT_DICT_URL,
        generator.DEFAULT_NGRAM_DIR,
    )

    if dictionary_token == default_dictionary_token:
        augmented_dictionary = AUGMENTED_DICTIONARY
    else:
        augmented_dictionary = DICTIONARY_DIR / (
            f"normal_user_v{USER_LEXICON_SCHEMA}_dict_{dictionary_token}.txt"
        )

    if policy_token == default_policy_token:
        policy_cache = POLICY_CACHE
    else:
        policy_cache = DICTIONARY_DIR / (
            f"normal_user_v{USER_LEXICON_SCHEMA}_policy_{policy_token}.json"
        )

    return augmented_dictionary, policy_cache, policy_token


def _policy_token(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
    extra_short_words: tuple[str, ...],
    *,
    source_token: str = "default",
) -> str:
    """Return a stable identity for the effective source-backed user lexicon."""
    payload = {
        "schema": USER_LEXICON_SCHEMA,
        "policy_source": _policy_source_token(),
        "source_token": source_token,
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

    chunks = [original]
    if original and not original.endswith("\n"):
        chunks.append("\n")
    chunks.extend(word + "\n" for word in additions)
    desired = "".join(chunks)

    if output.is_file():
        try:
            if output.read_text(encoding="utf-8", errors="ignore") == desired:
                return output
        except OSError:
            pass

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(desired, encoding="utf-8", newline="\n")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def _load_cached_policy(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
    *,
    source_token: str = "default",
    augmented_dictionary: Path | None = None,
    policy_cache: Path | None = None,
) -> UserLexicon | None:
    augmented_dictionary = augmented_dictionary or AUGMENTED_DICTIONARY
    policy_cache = policy_cache or POLICY_CACHE
    if not augmented_dictionary.is_file() or not policy_cache.is_file():
        return None
    try:
        payload = json.loads(policy_cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") != USER_LEXICON_SCHEMA:
        return None
    if payload.get("policy_source") != _policy_source_token():
        return None
    if payload.get("source_token", "default") != source_token:
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
        augmented_dictionary,
        normalized_short_words,
        _policy_token(
            dictionary_stamp,
            unigram_stamp,
            normalized_short_words,
            source_token=source_token,
        ),
    )


def _save_policy(
    dictionary_stamp: tuple[int, int],
    unigram_stamp: tuple[int, int],
    extra_short_words: tuple[str, ...],
    *,
    source_token: str = "default",
    policy_cache: Path | None = None,
) -> None:
    policy_cache = policy_cache or POLICY_CACHE
    payload = {
        "schema": USER_LEXICON_SCHEMA,
        "policy_source": _policy_source_token(),
        "source_token": source_token,
        "dictionary_stamp": list(dictionary_stamp),
        "unigram_stamp": list(unigram_stamp),
        "extra_short_words": list(extra_short_words),
    }
    policy_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = policy_cache.with_name(f".{policy_cache.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, policy_cache)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_user_lexicon(
    *,
    dictionary_source: str = generator.DEFAULT_DICT_URL,
    ngram_dir: Path | str = generator.DEFAULT_NGRAM_DIR,
    refresh: bool = False,
) -> UserLexicon:
    """Provision and return the cached lexicon used by normal solver runs."""
    resolved_ngram_dir = Path(ngram_dir).expanduser()
    augmented_dictionary, policy_cache, source_token = _derived_cache_paths(
        dictionary_source,
        resolved_ngram_dir,
    )
    base_dictionary = generator.get_dictionary(dictionary_source, refresh=refresh)
    unigram_path, _ = generator.ensure_ngram_data(
        resolved_ngram_dir,
        refresh=refresh,
        need_bigrams=False,
    )
    dictionary_stamp = _source_stamp(base_dictionary)
    unigram_stamp = _source_stamp(unigram_path)

    cached = _load_cached_policy(
        dictionary_stamp,
        unigram_stamp,
        source_token=source_token,
        augmented_dictionary=augmented_dictionary,
        policy_cache=policy_cache,
    )
    if cached is not None:
        return cached

    unigrams = generator.load_unigram_model(unigram_path)
    extra_short_words = select_corpus_short_words(
        base_dictionary,
        unigrams,
        min_zipf=DEFAULT_SHORT_WORD_MIN_ZIPF,
    )
    build_augmented_dictionary(
        base_dictionary,
        augmented_dictionary,
        frozenset(generator.PRETTY_CONTRACTIONS),
    )
    _save_policy(
        dictionary_stamp,
        unigram_stamp,
        extra_short_words,
        source_token=source_token,
        policy_cache=policy_cache,
    )
    return UserLexicon(
        augmented_dictionary,
        extra_short_words,
        _policy_token(
            dictionary_stamp,
            unigram_stamp,
            extra_short_words,
            source_token=source_token,
        ),
    )