"""Hot-path adapters and corpus evidence for the active reranker.

The stable core scorer intentionally stays simple and portable. The active
facade layers runtime optimizations plus generic corpus-span cohesion over it
without changing the prepared-cache format.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import cast

import anagram_rerank_core as core
from anagram_corpus_cohesion import blend_phrase_cohesion, score_corpus_cohesion

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

# Cohesion changes ranking semantics, unlike the cache-only adapters. Keep that
# policy context-local so an unrelated thread touching stable core while facade
# work is active cannot accidentally inherit cohesion scoring.
_COHESION_ACTIVE: ContextVar[bool] = ContextVar(
    "anagram_cohesion_active",
    default=False,
)


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
    _frames_cache_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def load(cls, dictionary_dir: Path) -> FastWordNetLexicon:
        # The core classmethod constructs ``cls`` already; narrow its historical
        # base-class return annotation for callers of the optimized subclass.
        return cast(
            FastWordNetLexicon,
            super(FastWordNetLexicon, cls).load(dictionary_dir),
        )

    def frames_for(self, raw_word: str) -> frozenset[int]:
        word = fast_norm_token(raw_word)
        with self._frames_cache_lock:
            cached = self._frames_cache.get(word)
            if cached is not None:
                self._frames_cache.move_to_end(word)
                return cached

        # Keep lexical computation outside the cache lock. Concurrent misses for
        # the same surface form may duplicate this immutable work, but unrelated
        # threaded lookups should not serialize on WordNet traversal.
        frames: set[int] = set()
        for lemma in self.verb_base_lemmas(word):
            frames.update(self.verb_frames.get(lemma, ()))
        result = frozenset(frames)
        if self._frames_cache_limit <= 0:
            return result

        with self._frames_cache_lock:
            # Another worker may have populated this key while we were resolving
            # WordNet frames. Prefer that equivalent cached value and refresh LRU
            # order rather than needlessly replacing it.
            cached = self._frames_cache.get(word)
            if cached is not None:
                self._frames_cache.move_to_end(word)
                return cached
            self._frames_cache[word] = result
            self._frames_cache.move_to_end(word)
            while len(self._frames_cache) > self._frames_cache_limit:
                self._frames_cache.popitem(last=False)
        return result


@dataclass(slots=True)
class CachedPhraseIndex(core.PhraseIndex):
    """Bounded read-through cache with stable-core phrase scoring semantics."""

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

    def _blend_cohesion(
        self,
        words: Sequence[str],
        base_score: float,
        details: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        cohesion = score_corpus_cohesion(
            words,
            counts=self.counts,
            max_n=self.max_n,
        )
        score = blend_phrase_cohesion(base_score, cohesion)
        return score, {
            **details,
            "cohesion": cohesion.score,
            "cohesion_coverage": cohesion.coverage,
            "cohesion_longest_fraction": cohesion.longest_fraction,
            "cohesion_segments": float(cohesion.segments),
            "cohesion_splice_penalty": cohesion.splice_penalty,
            "cohesion_frequency": cohesion.frequency_strength,
        }

    def score(self, words: Sequence[str]) -> tuple[float, dict[str, float]]:
        """Keep core semantics unless the active facade context requests cohesion."""
        base_score, details = _ORIGINAL_PHRASE_INDEX.score(self, words)
        if not _COHESION_ACTIVE.get():
            return base_score, details
        return self._blend_cohesion(words, base_score, details)


class FastPhraseIndex(CachedPhraseIndex):
    """Facade phrase index with cohesion scoring enabled explicitly."""

    __slots__ = ()

    def score(self, words: Sequence[str]) -> tuple[float, dict[str, float]]:
        base_score, details = _ORIGINAL_PHRASE_INDEX.score(self, words)
        return self._blend_cohesion(words, base_score, details)


def clear_performance_caches() -> None:
    """Reset process-global memoized helpers used by benchmark/test isolation."""
    cached_function_class.cache_clear()


@contextmanager
def performance_hooks() -> Iterator[None]:
    """Temporarily install semantics-preserving fast core adapters.

    Normalization, function metadata, WordNet frames and phrase-count caching are
    safe to expose through stable core because they preserve its results. Corpus
    cohesion is different: it is a ranking policy. A ContextVar enables that
    policy only for facade execution while unrelated concurrent threads keep the
    original PhraseIndex.score semantics even if they observe CachedPhraseIndex.
    """
    global _HOOK_DEPTH, _HOOK_RESTORE

    cohesion_token = _COHESION_ACTIVE.set(True)
    try:
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
                core.PhraseIndex = CachedPhraseIndex
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
    finally:
        _COHESION_ACTIVE.reset(cohesion_token)
