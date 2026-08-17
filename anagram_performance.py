"""Semantics-preserving hot-path adapters for the active reranker.

The core scorer intentionally stays simple and portable. The active facade can
therefore layer a few runtime-only optimizations over it without changing the
legacy parser's behavior or prepared-cache format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import anagram_rerank_core as core

_ORIGINAL_NORM_TOKEN = core.norm_token
_ORIGINAL_FUNCTION_CLASS = core.function_class
_INSTALLED = False


def fast_norm_token(text: str) -> str:
    """Return already-normalized solver tokens without Unicode round-tripping."""
    if text and text.isascii() and text.isalpha() and text.islower():
        return text
    return _ORIGINAL_NORM_TOKEN(text)


@cache
def cached_function_class(word: str) -> str | None:
    """Cache immutable function-word metadata by exact input token."""
    return _ORIGINAL_FUNCTION_CLASS(word)


@dataclass(slots=True)
class FastWordNetLexicon(core.WordNetLexicon):
    """WordNet lexicon with cached surface-form verb-frame resolution."""

    _frames_cache: dict[str, frozenset[int]] = field(default_factory=dict)

    def frames_for(self, raw_word: str) -> frozenset[int]:
        word = fast_norm_token(raw_word)
        cached = self._frames_cache.get(word)
        if cached is not None:
            return cached

        frames: set[int] = set()
        for lemma in self.verb_base_lemmas(word):
            frames.update(self.verb_frames.get(lemma, ()))
        result = frozenset(frames)
        self._frames_cache[word] = result
        return result


def install_performance_hooks() -> None:
    """Install idempotent runtime adapters used by the active solver facade."""
    global _INSTALLED
    if _INSTALLED:
        return

    core.norm_token = fast_norm_token
    core.function_class = cached_function_class
    core.WordNetLexicon = FastWordNetLexicon
    _INSTALLED = True
