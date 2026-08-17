"""Semantics-preserving hot-path adapters for the active reranker.

The core scorer intentionally stays simple and portable. The active facade can
therefore layer a few runtime-only optimizations over it without changing the
legacy parser's behavior or prepared-cache format.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import anagram_rerank_core as core

_ORIGINAL_NORM_TOKEN = core.norm_token
_ORIGINAL_FUNCTION_CLASS = core.function_class
_ORIGINAL_PHRASE_INDEX = core.PhraseIndex

_HOOK_LOCK = threading.RLock()
_HOOK_DEPTH = 0
_HOOK_RESTORE: tuple[
    Callable[[str], str],
    Callable[[str], str | None],
    type[core.WordNetLexicon],
    type[core.PhraseIndex],
] | None = None


def fast_norm_token(text: str) -> str:
    """Return already-normalized solver tokens without Unicode round-tripping."""
    if text and text.isascii() and text.isalpha() and text.islower():
        return text
    return _ORIGINAL_NORM_TOKEN(text)


@lru_cache(maxsize=4096)
def cached_function_class(word: str) -> str | None:
    """Cache immutable function-word metadata by exact input token."""
    return _ORIGINAL_FUNCTION_CLASS(word)


@dataclass(slots=True)
class FastWordNetLexicon(core.WordNetLexicon):
    """WordNet lexicon with bounded cached surface-form verb-frame resolution."""

    _frames_cache: OrderedDict[str, frozenset[int]] = field(
        default_factory=OrderedDict
    )
    _frames_cache_limit: int = field(default=4096, repr=False)

    def frames_for(self, raw_word: str) -> frozenset[int]:
        word = fast_norm_token(raw_word)
        if word in self._frames_cache:
            result = self._frames_cache[word]
            self._frames_cache.move_to_end(word)
            return result

        frames: set[int] = set()
        for lemma in self.verb_base_lemmas(word):
            frames.update(self.verb_frames.get(lemma, ()))
        result = frozenset(frames)
        if self._frames_cache_limit > 0:
            self._frames_cache[word] = result
            self._frames_cache.move_to_end(word)
            while len(self._frames_cache) > self._frames_cache_limit:
                self._frames_cache.popitem(last=False)
        return result


@dataclass(slots=True)
class FastPhraseIndex(core.PhraseIndex):
    """Bounded read-through cache for immutable phrase-index counts and misses."""

    # None is an explicit database-miss sentinel. Real rows may legally contain
    # zero or negative counts, and counts() preserves those baseline semantics.
    _count_cache: OrderedDict[str, int | None] = field(default_factory=OrderedDict)
    _count_cache_limit: int = field(default=32768, repr=False)

    def _remember_count(self, phrase: str, count: int | None) -> None:
        if self._count_cache_limit <= 0:
            return
        self._count_cache[phrase] = count
        self._count_cache.move_to_end(phrase)
        while len(self._count_cache) > self._count_cache_limit:
            self._count_cache.popitem(last=False)

    def counts(self, phrases: Sequence[str]) -> dict[str, int]:
        unique = tuple(dict.fromkeys(phrase for phrase in phrases if phrase))
        if not unique:
            return {}

        out: dict[str, int] = {}
        missing: list[str] = []
        for phrase in unique:
            if phrase in self._count_cache:
                count = self._count_cache[phrase]
                self._count_cache.move_to_end(phrase)
                if count is not None:
                    out[phrase] = count
            else:
                missing.append(phrase)

        for i in range(0, len(missing), 200):
            batch = missing[i : i + 200]
            placeholders = ",".join("?" for _ in batch)
            found = {
                str(phrase): int(count)
                for phrase, count in self.connection.execute(
                    f"SELECT text, count FROM ngrams WHERE text IN ({placeholders})",
                    batch,
                )
            }
            for phrase in batch:
                count = found.get(phrase)
                if count is not None:
                    out[phrase] = count
                self._remember_count(phrase, count)

        return out


def clear_performance_caches() -> None:
    """Reset process-global memoized helpers used by benchmark/test isolation."""
    cached_function_class.cache_clear()


@contextmanager
def performance_hooks() -> Iterator[None]:
    """Temporarily install fast core adapters and restore exact prior bindings.

    The depth counter keeps adapters active across nested or overlapping facade
    calls without holding the lock while user work executes. Importing the
    reranker therefore remains side-effect free, while active facade operations
    still share the optimized core functions they depend on.
    """
    global _HOOK_DEPTH, _HOOK_RESTORE

    with _HOOK_LOCK:
        if _HOOK_DEPTH == 0:
            _HOOK_RESTORE = (
                core.norm_token,
                core.function_class,
                core.WordNetLexicon,
                core.PhraseIndex,
            )
            core.norm_token = fast_norm_token
            core.function_class = cached_function_class
            core.WordNetLexicon = FastWordNetLexicon
            core.PhraseIndex = FastPhraseIndex
        _HOOK_DEPTH += 1

    try:
        yield
    finally:
        with _HOOK_LOCK:
            _HOOK_DEPTH -= 1
            if _HOOK_DEPTH == 0:
                restore = _HOOK_RESTORE
                if restore is None:
                    raise RuntimeError("performance hook restore state was lost")
                (
                    core.norm_token,
                    core.function_class,
                    core.WordNetLexicon,
                    core.PhraseIndex,
                ) = restore
                _HOOK_RESTORE = None