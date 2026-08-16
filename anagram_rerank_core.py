#!/usr/bin/env python3
"""
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
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import math
import multiprocessing
import os
import pickle
import re
import shutil
import sqlite3
import sys
import tarfile
import tempfile
import time
import unicodedata
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

WORDNET_URL = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"
DEFAULT_WORDNET_DIR = Path.home() / ".multi_anagram" / "wordnet31"

DEFAULT_PREPARED_CACHE_DIR = Path.home() / ".multi_anagram" / "prepared_cache"
DEFAULT_NGRAM_DIR = Path.home() / ".multi_anagram" / "ngrams"
NORVIG_1W_URL = "https://norvig.com/ngrams/count_1w.txt"
NORVIG_2W_URL = "https://norvig.com/ngrams/count_2w.txt"
PREPARED_CACHE_SCHEMA = "core-prepared-1"

PRETTY = {
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
}

# Function-word classes. These are grammar metadata, not answer-specific words.
DET_PL = {"these", "those", "both", "many", "few", "several"}
DET_SG = {"this", "that", "each", "every", "either", "neither", "another"}
ARTICLES = {"a", "an", "the"}
DET_ANY = {
    "some", "any", "no", "my", "your", "his", "her", "its", "our", "their",
    "whose", "what", "which",
}

NUMBER_DET = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "dozen", "hundred", "thousand",
}

COMPARATIVE_WORDS = {
    "better", "worse", "more", "less", "rather", "sooner",
}

SUBORDINATORS = {
    "after", "although", "as", "because", "before", "if", "once", "since",
    "though", "unless", "until", "when", "whenever", "where", "whereas",
    "wherever", "while",
}

PRON_1_2 = {"i", "you", "we"}
PRON_PL = {"we", "they", "you"}
PRON_SG3 = {"he", "she", "it"}
PRON_OTHER = {
    "me", "him", "her", "us", "them", "myself", "yourself", "himself",
    "herself", "itself", "ourselves", "themselves", "who", "whom",
}

# Auxiliaries/modals. "dont" is normalized punctuation-free "don't".
AUX_DONT = {"dont"}
AUX_DOESNT = {"doesnt"}
AUX_DO_BASE = {"do", "did", "didnt"}
AUX_MODAL = {
    "can", "cant", "could", "couldnt", "will", "wont", "would", "wouldnt",
    "should", "shouldnt", "may", "might", "must",
}
AUX_BE = {"am", "is", "are", "was", "were", "be", "been", "being", "isnt", "arent", "wasnt", "werent"}
AUX_HAVE = {"have", "has", "had", "havent", "hasnt", "hadnt"}

PREPOSITIONS = {
    "about", "above", "across", "after", "against", "along", "among", "around",
    "at", "before", "behind", "below", "beneath", "beside", "between", "beyond",
    "by", "despite", "down", "during", "except", "for", "from", "in", "inside",
    "into", "near", "of", "off", "on", "onto", "over", "past", "since",
    "through", "throughout", "to", "toward", "under", "until", "up", "upon",
    "with", "within", "without", "like", "than",
}
CONJUNCTIONS = {"and", "but", "or", "nor", "for", "yet", "so", "although", "because", "if", "unless", "while"}
NEG_PARTICLES = {"not", "never"}

FUNCTION_WORDS = (
    DET_PL | DET_SG | ARTICLES | DET_ANY | NUMBER_DET
    | PRON_1_2 | PRON_PL | PRON_SG3 | PRON_OTHER
    | AUX_DONT | AUX_DOESNT | AUX_DO_BASE | AUX_MODAL | AUX_BE | AUX_HAVE
    | PREPOSITIONS | CONJUNCTIONS | NEG_PARTICLES
)

GENERATOR_LINE_RE = re.compile(
    r"^\s*(?P<rank>\d+)\.\s+"
    r"PRE=\s*(?P<pre>[-+]?\d+(?:\.\d+)?)\s+"
    r"LEX=(?P<lex>\d+(?:\.\d+)?)\s+"
    r"FAM=(?P<fam>\d+(?:\.\d+)?)\s+"
    r"PAIR=(?P<pair>\d+(?:\.\d+)?)\s+"
    r"HINT=(?P<hint>\d+(?:\.\d+)?)\s+"
    r"ZAVG=(?P<zavg>[-+]?\d+(?:\.\d+)?)\s+"
    r"ZMIN=(?P<zmin>[-+]?\d+(?:\.\d+)?)\s+"
    r"PCOV=(?P<pcov>\d+(?:\.\d+)?)\s+"
    r"(?P<phrase>.*?)\s+\[HINT=(?P<hints>.*?)\]\s*$"
)

SECTION_RE = re.compile(r"^===\s+(\d+)-WORD\s+SOLUTIONS")


def norm_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def pretty_phrase(words: Sequence[str]) -> str:
    return " ".join(PRETTY.get(w, w) for w in words)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract a trusted archive while still rejecting path traversal."""
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination)
            except ValueError:
                raise RuntimeError(f"Unsafe archive path: {member.name!r}")
        tf.extractall(destination, members=members)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "anagram-solver/1.0"})
    with urllib.request.urlopen(req, timeout=90) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def find_wordnet_dict(root: Path) -> Path | None:
    required = {"index.noun", "index.verb", "index.adj", "index.adv"}
    if root.is_dir() and required.issubset({p.name for p in root.iterdir() if p.is_file()}):
        return root
    if not root.exists():
        return None
    for path in root.rglob("index.noun"):
        parent = path.parent
        names = {p.name for p in parent.iterdir() if p.is_file()}
        if required.issubset(names):
            return parent
    return None


def ensure_wordnet(root: Path, refresh: bool = False) -> Path:
    if refresh and root.exists():
        shutil.rmtree(root)

    found = find_wordnet_dict(root)
    if found is not None:
        return found

    root.mkdir(parents=True, exist_ok=True)
    archive = root / "wn3.1.dict.tar.gz"
    if not archive.exists():
        print(f"Downloading WordNet 3.1 -> {archive}")
        download(WORDNET_URL, archive)

    print(f"Extracting WordNet 3.1 under {root}")
    with tempfile.TemporaryDirectory(prefix="wn31_", dir=str(root)) as td:
        temp = Path(td)
        safe_extract_tar(archive, temp)
        extracted = find_wordnet_dict(temp)
        if extracted is None:
            raise RuntimeError("Could not locate WordNet index files after extraction.")

        dest = root / "dict"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(extracted, dest)

    found = find_wordnet_dict(root)
    if found is None:
        raise RuntimeError("WordNet extraction completed but dictionary files are missing.")
    return found


def load_index(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            # WordNet copyright/header lines begin with whitespace.
            if not line or line[0].isspace():
                continue
            lemma = line.split(" ", 1)[0].strip().lower().replace("_", " ")
            # The anagram solver works on single words. Ignore collocations here.
            if lemma and " " not in lemma and lemma.isalpha():
                out.add(lemma)
    return out


def load_exceptions(path: Path) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            parts = line.strip().lower().split()
            if len(parts) >= 2:
                out[parts[0]] = tuple(parts[1:])
    return out




# ---------------------------------------------------------------------------
# Performance/runtime helpers
# ---------------------------------------------------------------------------

_WORKER_LEX = None
_WORKER_ORDER_MODE = "auto"
_WORKER_BEAM_WIDTH = 128
_WORKER_EXACT_MAX_WORDS = 5


@dataclass(slots=True, frozen=True)
class DeepResult:
    row_index: int
    grammar_raw: float
    best_order: tuple[str, ...]
    structure_norm: float
    valency_norm: float
    syntax_coverage: float
    phrase_kind: str
    orders_evaluated: int


def _gil_enabled() -> bool:
    fn = getattr(sys, "_is_gil_enabled", None)
    if fn is None:
        return True
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        # sys._is_gil_enabled is a non-standard interpreter hook. If a build
        # exposes a broken/incompatible hook, conservatively assume the GIL.
        return True


def resolve_backend(requested: str, workers: int) -> str:
    if workers <= 1:
        return "serial"
    if requested != "auto":
        return requested
    # On a free-threaded build, threads avoid process startup/copy costs.
    # On ordinary CPython, use processes so CPU-bound Python can use all cores.
    return "thread" if not _gil_enabled() else "process"


def resolve_workers(requested: int) -> int:
    if requested > 0:
        return requested
    logical = os.process_cpu_count() or os.cpu_count() or 1
    # Conservative default for common SMT desktops. Users can request more.
    return max(1, min(8, logical))


def chunked(seq: Sequence[int], size: int) -> list[tuple[int, ...]]:
    return [tuple(seq[i : i + size]) for i in range(0, len(seq), size)]


# WordNet 3.1 generic verb-frame groups.
#
# Earlier revisions used coarse frame buckets. The regression suite exposed that
# frames 6/7 are predicative
# complements ("Something ----s Adjective/Noun"), not intransitives.
#
# Source taxonomy follows Princeton WordNet's published 35 generic frames.
FRAME_INTRANSITIVE = {1, 2, 3, 23}

FRAME_DIRECT_OBJECT = {
    5, 8, 9, 10, 11,
    14, 15, 16, 17, 18, 19, 20, 21,
    24, 25, 30, 31,
}

FRAME_PREDICATIVE = {6, 7}
FRAME_OBJECT_PREDICATIVE = {5}

FRAME_PP = {4, 12, 13, 20, 21, 22, 27, 31}

FRAME_INFINITIVE_OR_GERUND = {
    24, 25, 28, 29, 30, 32, 33, 35,
}

FRAME_CLAUSAL = {26, 29, 34}



def load_verb_frames(path: Path) -> dict[str, frozenset[int]]:
    """
    Parse WordNet data.verb.

    data.verb layout:
      offset lex_filenum v w_cnt [word lex_id] p_cnt [4-field pointers]
      f_cnt [+ f_num w_num] | gloss

    A frame with w_num=00 applies to every word in the synset; otherwise it
    applies only to the 1-based word number in that synset.
    """
    out: dict[str, set[int]] = defaultdict(set)
    if not path.exists():
        return {}

    with path.open("r", encoding="ascii", errors="ignore") as f:
        for line in f:
            if not line or line[0].isspace() or "|" not in line:
                continue

            fields = line.split("|", 1)[0].strip().split()
            if len(fields) < 5:
                continue

            try:
                pos = 0
                pos += 3  # offset, lex_filenum, ss_type
                w_cnt = int(fields[pos], 16)
                pos += 1

                words: list[str] = []
                for _ in range(w_cnt):
                    lemma = fields[pos].lower().replace("_", " ")
                    pos += 2  # word, lex_id
                    words.append(lemma)

                p_cnt = int(fields[pos])
                pos += 1 + 4 * p_cnt

                f_cnt = int(fields[pos])
                pos += 1

                frames: list[tuple[int, int]] = []
                for _ in range(f_cnt):
                    if fields[pos] != "+":
                        raise ValueError("expected + before verb frame")
                    f_num = int(fields[pos + 1])
                    w_num = int(fields[pos + 2], 16)
                    pos += 3
                    frames.append((f_num, w_num))
            except (ValueError, IndexError):
                continue

            for f_num, w_num in frames:
                if w_num == 0:
                    targets = range(len(words))
                elif 1 <= w_num <= len(words):
                    targets = (w_num - 1,)
                else:
                    continue

                for idx in targets:
                    lemma = words[idx]
                    if " " not in lemma and lemma.isalpha():
                        out[lemma].add(f_num)

    return {lemma: frozenset(values) for lemma, values in out.items()}


@dataclass(slots=True, frozen=True)
class Features:
    noun: bool = False
    verb: bool = False
    adj: bool = False
    adv: bool = False
    noun_plural: bool = False
    noun_singular: bool = False
    verb_base: bool = False
    verb_3sg: bool = False
    verb_past: bool = False
    verb_ing: bool = False
    recognized: bool = False


@dataclass(slots=True)
class WordNetLexicon:
    nouns: set[str]
    verbs: set[str]
    adjs: set[str]
    advs: set[str]
    noun_exc: dict[str, tuple[str, ...]]
    verb_exc: dict[str, tuple[str, ...]]
    verb_frames: dict[str, frozenset[int]]
    _cache: dict[str, Features] = field(default_factory=dict)

    @classmethod
    def load(cls, dictionary_dir: Path) -> WordNetLexicon:
        return cls(
            nouns=load_index(dictionary_dir / "index.noun"),
            verbs=load_index(dictionary_dir / "index.verb"),
            adjs=load_index(dictionary_dir / "index.adj"),
            advs=load_index(dictionary_dir / "index.adv"),
            noun_exc=load_exceptions(dictionary_dir / "noun.exc"),
            verb_exc=load_exceptions(dictionary_dir / "verb.exc"),
            verb_frames=load_verb_frames(dictionary_dir / "data.verb"),
        )

    def _noun_plural_bases(self, word: str) -> set[str]:
        out = set(self.noun_exc.get(word, ()))
        if len(word) > 3 and word.endswith("ies"):
            out.add(word[:-3] + "y")
        if len(word) > 3 and word.endswith(("ches", "shes", "xes", "zes", "ses", "oes")):
            out.add(word[:-2])
        if len(word) > 2 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            out.add(word[:-1])
        return {x for x in out if x in self.nouns}

    def _verb_bases(self, word: str) -> tuple[set[str], set[str], set[str]]:
        """Return candidate bases for 3sg, past, and -ing forms."""
        exc = {x for x in self.verb_exc.get(word, ()) if x in self.verbs}
        third: set[str] = set()
        past: set[str] = set(exc)
        ing: set[str] = set()

        if len(word) > 3 and word.endswith("ies"):
            third.add(word[:-3] + "y")
            # lie -> lies
            third.add(word[:-1])
        if len(word) > 3 and word.endswith(("ches", "shes", "xes", "zes", "ses", "oes")):
            third.add(word[:-2])
        if len(word) > 2 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            third.add(word[:-1])

        if len(word) > 3 and word.endswith("ied"):
            past.add(word[:-3] + "y")
            past.add(word[:-1])
        elif len(word) > 3 and word.endswith("ed"):
            past.add(word[:-2])
            past.add(word[:-1])
            # stopped -> stop, planned -> plan
            stem = word[:-2]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                past.add(stem[:-1])

        if len(word) > 4 and word.endswith("ing"):
            stem = word[:-3]
            ing.add(stem)
            ing.add(stem + "e")
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                ing.add(stem[:-1])

        return (
            {x for x in third if x in self.verbs},
            {x for x in past if x in self.verbs},
            {x for x in ing if x in self.verbs},
        )

    def verb_base_lemmas(self, raw_word: str) -> set[str]:
        word = norm_token(raw_word)
        out: set[str] = set()
        if word in self.verbs:
            out.add(word)
        v3, vpast, ving = self._verb_bases(word)
        out.update(v3)
        out.update(vpast)
        out.update(ving)
        return out

    def frames_for(self, raw_word: str) -> frozenset[int]:
        frames: set[int] = set()
        for lemma in self.verb_base_lemmas(raw_word):
            frames.update(self.verb_frames.get(lemma, ()))
        return frozenset(frames)

    def allows_object(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_DIRECT_OBJECT)

    def allows_intransitive(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_INTRANSITIVE)

    def allows_pp(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_PP)

    def allows_predicative(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_PREDICATIVE)

    def allows_object_predicative(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_OBJECT_PREDICATIVE)

    def allows_infinitive_or_gerund(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_INFINITIVE_OR_GERUND)

    def allows_clausal(self, raw_word: str) -> bool | None:
        frames = self.frames_for(raw_word)
        if not frames:
            return None
        return bool(frames & FRAME_CLAUSAL)

    def features(self, raw_word: str) -> Features:
        word = norm_token(raw_word)
        cached = self._cache.get(word)
        if cached is not None:
            return cached

        function = word in FUNCTION_WORDS
        noun_exact = word in self.nouns
        verb_exact = word in self.verbs
        adj = word in self.adjs
        adv = word in self.advs

        noun_plural_bases = self._noun_plural_bases(word)
        v3, vpast, ving = self._verb_bases(word)

        f = Features(
            noun=noun_exact or bool(noun_plural_bases),
            verb=verb_exact or bool(v3 or vpast or ving),
            adj=adj,
            adv=adv,
            noun_plural=bool(noun_plural_bases),
            noun_singular=noun_exact and not bool(noun_plural_bases),
            verb_base=verb_exact,
            verb_3sg=bool(v3),
            verb_past=bool(vpast),
            verb_ing=bool(ving),
            recognized=function or noun_exact or verb_exact or adj or adv or bool(noun_plural_bases or v3 or vpast or ving),
        )
        self._cache[word] = f
        return f


@dataclass(slots=True)
class Row:
    words: tuple[str, ...]
    word_count: int
    old_rank: int
    old_pre: float
    lex: float
    fam: float
    old_pair: float
    hint: float
    zavg: float
    zmin: float
    old_pcov: float
    hints: tuple[str, ...]

    wn_coverage: float = 0.0
    grammar_potential: float = 0.0
    grammar_potential_norm: float = 0.0
    pre_score: float = 0.0

    deep: bool = False
    best_order: tuple[str, ...] = ()
    grammar_raw: float = 0.0
    grammar_norm: float = 0.0
    structure_norm: float = 0.0
    valency_norm: float = 0.5
    syntax_coverage: float = 0.0
    phrase_kind: str = "unknown"
    base_final: float = 0.0
    colloc_norm: float = 0.0
    phrase_attest_norm: float = 0.0
    phrase_bonus: float = 0.0
    final: float = 0.0
    family_key: tuple[str, ...] = ()



def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _prepared_cache_key(input_path: Path, wordnet_dir: Path) -> str:
    h = hashlib.sha256()
    h.update(PREPARED_CACHE_SCHEMA.encode("ascii"))
    h.update(str(input_path.resolve()).encode("utf-8", "replace"))
    h.update(_hash_file(input_path).encode("ascii"))

    # WordNet dictionary version fingerprint. Hashing the huge data files every
    # run defeats the purpose; filenames, sizes and mtimes are enough to detect
    # a changed local install.
    for name in (
        "index.noun", "index.verb", "index.adj", "index.adv",
        "noun.exc", "verb.exc", "data.verb",
    ):
        p = wordnet_dir / name
        if p.exists():
            st = p.stat()
            h.update(name.encode("ascii"))
            h.update(str(st.st_size).encode("ascii"))
            h.update(str(st.st_mtime_ns).encode("ascii"))
    return h.hexdigest()[:24]


def _reset_deep_fields(rows: list[Row]) -> None:
    """
    Prepared-cache rows are saved before deep analysis, but reset defensively so
    future cache-schema changes cannot accidentally retain an old final ranking.
    """
    for row in rows:
        row.deep = False
        row.best_order = ()
        row.grammar_raw = 0.0
        row.grammar_norm = 0.0
        row.structure_norm = 0.0
        row.valency_norm = 0.5
        row.syntax_coverage = 0.0
        row.phrase_kind = "unknown"
        row.base_final = 0.0
        row.colloc_norm = 0.0
        row.phrase_attest_norm = 0.0
        row.phrase_bonus = 0.0
        row.final = 0.0


def load_prepared_cache(cache_path: Path) -> list[Row] | None:
    try:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            return None
        if payload.get("schema") != PREPARED_CACHE_SCHEMA:
            return None
        rows = payload.get("rows")
        if not isinstance(rows, list):
            return None
        _reset_deep_fields(rows)
        return rows
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None


def save_prepared_cache(cache_path: Path, rows: list[Row]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(
            {"schema": PREPARED_CACHE_SCHEMA, "rows": rows},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(cache_path)


@dataclass(slots=True)
class PositiveBigramModel:
    unigram_counts: dict[str, int]
    bigram_counts: dict[tuple[str, str], int]
    total_unigrams: int

    def score(self, words: Sequence[str]) -> tuple[float, float]:
        """
        Positive-only collocation evidence.

        Missing edges contribute zero rather than a negative score. This is
        intentionally asymmetric: corpus presence can help, corpus absence
        cannot sink a phrase.
        """
        if len(words) <= 1:
            return 0.0, 0.0

        edge_scores: list[float] = []
        seen = 0
        for left, right in itertools.pairwise(words):
            pair_count = self.bigram_counts.get((left, right), 0)
            if pair_count <= 0:
                edge_scores.append(0.0)
                continue

            seen += 1
            lc = max(1, self.unigram_counts.get(left, 1))
            rc = max(1, self.unigram_counts.get(right, 1))

            # PMI-like association plus a weak raw-frequency term. Both are
            # clipped so one internet-famous bigram cannot own the ranking.
            pmi = math.log10(
                max(1.0, (pair_count * max(1, self.total_unigrams)) / (lc * rc))
            )
            assoc = max(0.0, min(1.0, pmi / 4.0))
            freq = max(0.0, min(1.0, math.log10(pair_count + 1.0) / 7.0))
            edge_scores.append(0.78 * assoc + 0.22 * freq)

        coverage = seen / (len(words) - 1)
        mean_edge = sum(edge_scores) / (len(words) - 1)
        score = max(0.0, min(1.0, 0.72 * mean_edge + 0.28 * coverage))
        return score, coverage


def _parse_count_line(line: str) -> tuple[str, int] | None:
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        text, count_raw = line.rsplit("\t", 1)
    else:
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            return None
        text, count_raw = parts
    try:
        return text, int(count_raw)
    except ValueError:
        return None


def ensure_norvig_ngrams(ngram_dir: Path) -> tuple[Path, Path]:
    one = ngram_dir / "count_1w.txt"
    two = ngram_dir / "count_2w.txt"
    if not one.exists():
        print(f"Downloading positive unigram corpus -> {one}")
        download(NORVIG_1W_URL, one)
    if not two.exists():
        print(f"Downloading positive bigram corpus -> {two}")
        download(NORVIG_2W_URL, two)
    return one, two


def load_positive_bigram_model(
    one_path: Path,
    two_path: Path,
    vocabulary: set[str],
) -> PositiveBigramModel:
    unigram_counts: dict[str, int] = {}
    total = 0
    with one_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_count_line(line)
            if parsed is None:
                continue
            token, count = parsed
            word = norm_token(token)
            total += count
            if word in vocabulary:
                unigram_counts[word] = unigram_counts.get(word, 0) + count

    bigram_counts: dict[tuple[str, str], int] = {}
    with two_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_count_line(line)
            if parsed is None:
                continue
            text, count = parsed
            parts = text.lower().split()
            if len(parts) != 2:
                continue
            left, right = norm_token(parts[0]), norm_token(parts[1])
            if left in vocabulary and right in vocabulary:
                bigram_counts[(left, right)] = (
                    bigram_counts.get((left, right), 0) + count
                )

    return PositiveBigramModel(unigram_counts, bigram_counts, total)


@dataclass(slots=True)
class PhraseIndex:
    connection: sqlite3.Connection
    max_n: int

    @classmethod
    def open(cls, path: Path) -> PhraseIndex:
        # Read-only URI prevents a scoring run from mutating its corpus.
        uri = f"{path.resolve().as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        row = conn.execute(
            "SELECT COALESCE(MAX(n), 0) FROM ngrams"
        ).fetchone()
        return cls(conn, int(row[0] if row else 0))

    def counts(self, phrases: Sequence[str]) -> dict[str, int]:
        unique = tuple(dict.fromkeys(p for p in phrases if p))
        if not unique:
            return {}
        out: dict[str, int] = {}
        # SQLite's parameter limit varies; conservative chunks keep this dull.
        for i in range(0, len(unique), 200):
            batch = unique[i : i + 200]
            placeholders = ",".join("?" for _ in batch)
            for phrase, count in self.connection.execute(
                f"SELECT text, count FROM ngrams WHERE text IN ({placeholders})",
                batch,
            ):
                out[str(phrase)] = int(count)
        return out

    def score(self, words: Sequence[str]) -> tuple[float, dict[str, float]]:
        if not words:
            return 0.0, {}

        whole = " ".join(words)
        queries = [whole]
        grams_by_n: dict[int, list[str]] = defaultdict(list)
        max_n = min(self.max_n, len(words))
        for n in range(2, max_n + 1):
            for i in range(len(words) - n + 1):
                gram = " ".join(words[i : i + n])
                grams_by_n[n].append(gram)
                queries.append(gram)

        hit_counts = self.counts(queries)
        whole_count = hit_counts.get(whole, 0)

        exact = 0.0
        if whole_count > 0:
            # Presence itself is meaningful; frequency only adds a bounded bit.
            exact = min(1.0, 0.72 + 0.28 * math.log10(whole_count + 1.0) / 5.0)

        tri_scores: list[float] = []
        for n, grams in grams_by_n.items():
            if n < 3:
                continue
            for gram in grams:
                c = hit_counts.get(gram, 0)
                if c > 0:
                    tri_scores.append(min(1.0, 0.55 + 0.45 * math.log10(c + 1.0) / 5.0))
                else:
                    tri_scores.append(0.0)
        longer = sum(tri_scores) / len(tri_scores) if tri_scores else 0.0

        bigrams = grams_by_n.get(2, [])
        bi_hits = sum(1 for g in bigrams if hit_counts.get(g, 0) > 0)
        bi_cov = bi_hits / len(bigrams) if bigrams else 0.0

        # Nutrimatic-inspired hierarchy:
        # whole attested phrase > longer contiguous chunks > isolated bigrams.
        score = max(
            exact,
            0.76 * longer if longer else 0.0,
            0.34 * bi_cov if bi_cov else 0.0,
        )
        return min(1.0, score), {
            "whole_count": float(whole_count),
            "longer": longer,
            "bigram_coverage": bi_cov,
        }


def apply_phrase_rescore(
    rows: list[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
    bonus_max: float,
) -> int:
    """
    Rescore only already-good deep candidates.

    This is intentionally a late-stage prior, matching the architecture used
    by puzzle-oriented phrase solvers: syntax first narrows the swamp, phrase
    evidence decides among plausible survivors.
    """
    by_wc: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        if row.deep:
            by_wc[row.word_count].append(row)

    rescored = 0
    for bucket in by_wc.values():
        bucket.sort(key=lambda r: (r.final, r.pre_score), reverse=True)
        chosen = bucket[:top_per_group]
        for row in chosen:
            row.base_final = row.final

            colloc = 0.0
            if collocation is not None:
                colloc, _ = collocation.score(row.best_order)
                row.colloc_norm = colloc

            phrase = 0.0
            if phrase_index is not None:
                phrase, _ = phrase_index.score(row.best_order)
                row.phrase_attest_norm = phrase

            # Whole-phrase/longer-gram evidence dominates. Bigram evidence is
            # only a modest fallback. Absence gives exactly zero bonus.
            evidence = max(phrase, 0.55 * colloc)
            row.phrase_bonus = bonus_max * evidence
            row.final = min(100.0, row.base_final + row.phrase_bonus)
            rescored += 1

    return rescored


def parse_candidates(path: Path) -> list[Row]:
    rows: list[Row] = []
    section: int | None = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            sec = SECTION_RE.match(line)
            if sec:
                section = int(sec.group(1))
                continue
            if section is None:
                continue
            m = GENERATOR_LINE_RE.match(line.rstrip())
            if not m:
                continue

            words = tuple(norm_token(x) for x in m.group("phrase").split())
            hints_raw = m.group("hints").strip()
            hints = tuple(sorted(
                norm_token(x)
                for x in hints_raw.split(",")
                if norm_token(x) and norm_token(x) != "-"
            ))
            rows.append(
                Row(
                    words=words,
                    word_count=section,
                    old_rank=int(m.group("rank")),
                    old_pre=float(m.group("pre")),
                    lex=float(m.group("lex")),
                    fam=float(m.group("fam")),
                    old_pair=float(m.group("pair")),
                    hint=float(m.group("hint")),
                    zavg=float(m.group("zavg")),
                    zmin=float(m.group("zmin")),
                    old_pcov=float(m.group("pcov")),
                    hints=hints,
                )
            )
    return rows


def function_class(word: str) -> str | None:
    if word in DET_PL:
        return "DET_PL"
    if word in DET_SG:
        return "DET_SG"
    if word in ARTICLES:
        return "ARTICLE"
    if word in NUMBER_DET:
        return "NUM_DET"
    if word in DET_ANY:
        return "DET"
    if word in PRON_1_2:
        return "PRON_12"
    if word in PRON_PL:
        return "PRON_PL"
    if word in PRON_SG3:
        return "PRON_SG3"
    if word in PRON_OTHER:
        return "PRON"
    if word in AUX_DONT:
        return "DONT"
    if word in AUX_DOESNT:
        return "DOESNT"
    if word in AUX_DO_BASE:
        return "DO_AUX"
    if word in AUX_MODAL:
        return "MODAL"
    if word in AUX_BE:
        return "BE_AUX"
    if word in AUX_HAVE:
        return "HAVE_AUX"
    if word in PREPOSITIONS:
        return "PREP"
    if word in CONJUNCTIONS:
        return "CONJ"
    if word in NEG_PARTICLES:
        return "NEG"
    return None


def pair_grammar(left: str, right: str, lex: WordNetLexicon) -> float:
    """
    Local grammatical compatibility. Positive means "this adjacency makes
    structural sense"; negative means "this adjacency is structurally odd".

    This is deliberately conservative. Semantic plausibility is NOT attempted.
    """
    lf = lex.features(left)
    rf = lex.features(right)
    lc = function_class(left)
    rc = function_class(right)

    score = 0.0

    # Determiner -> nominal phrase agreement.
    if lc == "DET_PL":
        if rf.noun_plural:
            score += 4.0
        elif rf.adj:
            score += 0.75  # "these old ..."
        elif rf.noun_singular:
            score -= 2.5
        elif rc is not None:
            score -= 2.0

    elif lc == "DET_SG":
        if rf.noun_singular:
            score += 3.0
        elif rf.adj:
            score += 0.75
        elif rf.noun_plural:
            score -= 2.5
        elif rc is not None:
            score -= 1.5

    elif lc == "ARTICLE":
        if rf.noun:
            # "a dogs" is bad; "the dogs" is fine.
            if left in {"a", "an"} and rf.noun_plural:
                score -= 3.0
            else:
                score += 2.0
        elif rf.adj:
            score += 1.0
        elif rc is not None:
            score -= 1.5

    elif lc == "DET":
        if rf.noun:
            score += 1.8
        elif rf.adj:
            score += 0.8

    elif lc == "NUM_DET":
        if rf.noun:
            score += 1.8
        elif rf.adj:
            score += 0.4

    # Copular be strongly licenses nominal/adjectival complements.
    if lc == "BE_AUX":
        if rf.adj or rf.noun:
            score += 2.4
        elif rf.adv:
            score += 1.0

    # Subject-ish material before finite/copular be.
    if rc == "BE_AUX" and (
        lf.noun or lc in {"PRON_12", "PRON_PL", "PRON_SG3", "PRON"}
    ):
        score += 1.2

    # do-support/modal -> base verb.
    if lc == "DONT":
        if rf.verb_base:
            score += 4.5
        elif rf.verb_3sg and not rf.verb_base:
            score -= 3.5
        elif rf.verb and not rf.verb_base:
            score -= 2.0
        elif rf.recognized:
            score -= 1.5

    elif lc == "DOESNT":
        if rf.verb_base:
            score += 4.5
        elif rf.verb_3sg and not rf.verb_base:
            score -= 3.0
        elif rf.recognized:
            score -= 1.5

    elif lc == "MODAL":
        if rf.verb_base:
            score += 3.5
        elif rf.verb and not rf.verb_base:
            score -= 2.0

    elif lc == "DO_AUX":
        if right == "not":
            score += 1.5
        elif rf.verb_base:
            score += 2.5

    # Subject -> do-support agreement.
    if rc == "DONT":
        if left in {"i", "you", "we", "they"} or lf.noun_plural:
            score += 2.5
        elif left in {"he", "she", "it"}:
            score -= 2.5
        elif lf.noun_singular:
            score -= 1.5

    elif rc == "DOESNT":
        if left in {"he", "she", "it"}:
            score += 2.5
        elif lf.noun_singular:
            score += 2.0
        elif lf.noun_plural or left in {"i", "you", "we", "they"}:
            score -= 2.0

    # Infinitival "to". We only give positive evidence; "to school" remains
    # possible instead of being punished as though every "to" were infinitival.
    if left == "to" and rf.verb_base:
        score += 1.8

    # Prepositional phrase hints.
    if lc == "PREP" and lc != "to":
        if rc in {"ARTICLE", "DET", "DET_PL", "DET_SG"}:
            score += 1.0
        elif rf.noun:
            score += 0.7
        elif rc in {"PRON", "PRON_12", "PRON_PL", "PRON_SG3"}:
            score += 0.6

    # Content-word preferences. These remain weaker than explicit function-word
    # grammar, but WordNet frame evidence can make a verb-complement edge strong.
    if lf.adj and rf.noun:
        score += 0.6
    if lf.adv and rf.verb:
        score += 0.25

    # Likely subject -> finite verb adjacency.
    if lf.noun and rf.verb and rc is None:
        score += 0.75

    # Verb -> nominal object.
    if lf.verb and rf.noun and lc is None:
        allowed_obj = lex.allows_object(left)
        if allowed_obj is True:
            score += 0.9

    # Verb -> predicative adjective/adverb, e.g. "run deep", "turn red".
    if lf.verb and (rf.adj or rf.adv) and lc is None:
        allowed_pred = lex.allows_predicative(left)
        if allowed_pred is True:
            score += 2.0

    # Obvious function-word collisions.
    if lc in {"ARTICLE", "DET", "DET_PL", "DET_SG"} and rc in {"DONT", "DOESNT", "MODAL", "CONJ"}:
        score -= 1.5
    if lc == "CONJ" and rc == "CONJ":
        score -= 1.5

    return score


def start_score(word: str, lex: WordNetLexicon) -> float:
    c = function_class(word)
    f = lex.features(word)
    if c in {"CONJ", "PREP"} and word not in {"in", "on", "at", "after", "before"}:
        return -0.4
    if c in {"ARTICLE", "DET", "DET_PL", "DET_SG", "PRON_12", "PRON_PL", "PRON_SG3", "PRON"}:
        return 0.35
    if f.noun or f.adv:
        return 0.15
    return 0.0


def end_score(word: str, lex: WordNetLexicon) -> float:
    c = function_class(word)
    f = lex.features(word)
    if c in {"ARTICLE", "DET", "DET_PL", "DET_SG", "PREP", "CONJ", "DONT", "DOESNT", "MODAL", "DO_AUX"}:
        return -1.25
    if f.verb or f.noun or f.adj or f.adv:
        return 0.2
    return 0.0


def content_coverage(words: Sequence[str], lex: WordNetLexicon) -> float:
    if not words:
        return 0.0
    return sum(1.0 for w in words if lex.features(w).recognized) / len(words)


def grammar_potential(words: Sequence[str], lex: WordNetLexicon) -> float:
    """
    Cheap optimistic order-independent score used before exact ordering.
    Only positive structural edges count. Missing evidence is not a penalty.
    """
    n = len(words)
    if n <= 1:
        return 0.0

    positives: list[float] = []
    for i, left in enumerate(words):
        for j, right in enumerate(words):
            if i == j:
                continue
            s = pair_grammar(left, right, lex)
            if s > 0:
                positives.append(s)
    positives.sort(reverse=True)
    raw = sum(positives[: n - 1]) / (n - 1)

    # 0-ish grammar -> ~0.18, 1.5 -> 0.5, 3+ -> strongly positive.
    return sigmoid((raw - 1.5) * 1.35)


def morphology_family_word(word: str, lex: WordNetLexicon) -> str:
    """
    Collapse obvious inflections to a WordNet lemma when possible.
    Used only to expand deep-analysis families, never to change letters.
    """
    word = norm_token(word)
    # Preserve the feature-cache warmup side effect without an unused local.
    lex.features(word)

    noun_bases = lex._noun_plural_bases(word)
    if noun_bases:
        return min(noun_bases)

    v3, vpast, ving = lex._verb_bases(word)
    verb_bases = v3 | vpast | ving
    if verb_bases:
        return min(verb_bases)

    return word


def morphology_family(words: Sequence[str], lex: WordNetLexicon) -> tuple[str, ...]:
    return tuple(sorted(morphology_family_word(w, lex) for w in words))



def _det_class(word: str) -> str | None:
    c = function_class(word)
    return c if c in {"DET_PL", "DET_SG", "ARTICLE", "DET", "NUM_DET"} else None


def _is_subject_head(word: str, lex: WordNetLexicon) -> bool:
    c = function_class(word)
    f = lex.features(word)
    return c in {"PRON_12", "PRON_PL", "PRON_SG3", "PRON"} or f.noun


def _subject_number(word: str, lex: WordNetLexicon) -> str:
    if word in {"we", "they", "you", "i"}:
        return "non3sg"
    if word in {"he", "she", "it"}:
        return "3sg"
    f = lex.features(word)
    if f.noun_plural and not f.noun_singular:
        return "non3sg"
    if f.noun_singular and not f.noun_plural:
        return "3sg"
    return "unknown"


def _np_span_ending_at(
    words: Sequence[str],
    head_idx: int,
    lex: WordNetLexicon,
) -> tuple[int, float] | None:
    """
    Return (start_index, coherence) for a compact NP ending at head_idx.

    earlier reranker greedily allowed any noun before another noun. Because English words are
    wildly POS-ambiguous in WordNet, that let nonsense such as
    "practice perfect" become a high-confidence subject NP. linguistic reranker prefers
    adjectival modifiers and only accepts a noun modifier when it is relatively
    unambiguous (noun but not verb/adjective).
    """
    if head_idx < 0 or head_idx >= len(words):
        return None

    head = words[head_idx]
    hf = lex.features(head)
    hc = function_class(head)

    if hc in {"PRON_12", "PRON_PL", "PRON_SG3", "PRON"}:
        return head_idx, 1.0

    # Nominalized comparative/quantifier words can head short NPs:
    # "less is more", "the poor", etc. We keep this narrow.
    nominalized = head in {"more", "less", "most", "least"}
    if not hf.noun and not nominalized:
        return None

    start = head_idx
    coherence = 0.76 if hf.noun else 0.70
    i = head_idx - 1
    modifiers = 0

    while i >= 0 and modifiers < 2:
        f = lex.features(words[i])
        c = function_class(words[i])

        if c is None and f.adj:
            start = i
            coherence += 0.08
            modifiers += 1
            i -= 1
            continue

        # Relatively unambiguous noun modifier, e.g. "school bus".
        if c is None and f.noun and not f.verb and not f.adj:
            start = i
            coherence += 0.025
            modifiers += 1
            i -= 1
            continue

        break

    if i >= 0 and _det_class(words[i]) is not None:
        det = words[i]
        dc = _det_class(det)
        start = i

        if dc == "DET_PL":
            if hf.noun_plural:
                coherence += 0.24
            elif hf.noun_singular:
                coherence -= 0.35
        elif dc == "DET_SG":
            if hf.noun_singular:
                coherence += 0.20
            elif hf.noun_plural:
                coherence -= 0.35
        elif dc == "ARTICLE":
            if det in {"a", "an"} and hf.noun_plural:
                coherence -= 0.45
            else:
                coherence += 0.15
        elif dc == "NUM_DET":
            if det == "one":
                coherence += 0.12 if not hf.noun_plural else -0.20
            else:
                coherence += 0.12 if hf.noun_plural else 0.02
        else:
            coherence += 0.10

    return start, max(0.0, min(1.0, coherence))


def _np_span_starting_at(
    words: Sequence[str],
    start: int,
    lex: WordNetLexicon,
    *,
    allow_post_pp: bool = True,
) -> tuple[int, float] | None:
    """
    Return (inclusive_end, coherence) for a compact noun phrase.

    Supports one conservative postnominal PP ("mother of invention",
    "birds of a feather") and avoids greedily consuming arbitrary runs of
    POS-ambiguous nouns.
    """
    if start >= len(words):
        return None

    c0 = function_class(words[start])
    if c0 in {"PRON", "PRON_12", "PRON_PL", "PRON_SG3"}:
        return start, 0.90

    i = start
    det_word: str | None = None
    det_class: str | None = None

    if _det_class(words[i]) is not None:
        det_word = words[i]
        det_class = _det_class(words[i])
        i += 1

    # Attributive adjectives.
    adjective_count = 0
    while i < len(words):
        f = lex.features(words[i])
        if f.adj:
            adjective_count += 1
            i += 1
            continue
        break

    if i >= len(words):
        return None

    # One relatively unambiguous noun modifier before the head.
    modifier_used = False
    f0 = lex.features(words[i])
    if (
        i + 1 < len(words)
        and f0.noun
        and not f0.verb
        and not f0.adj
        and lex.features(words[i + 1]).noun
    ):
        modifier_used = True
        i += 1

    if i >= len(words) or not lex.features(words[i]).noun:
        # Narrow nominalized comparative support.
        if i < len(words) and words[i] in {"more", "less", "most", "least"}:
            head_idx = i
        else:
            return None
    else:
        head_idx = i

    hf = lex.features(words[head_idx])
    end = head_idx
    coherence = 0.72 + min(0.12, adjective_count * 0.04)
    if modifier_used:
        coherence += 0.025

    if det_class == "DET_PL":
        coherence += 0.23 if hf.noun_plural else -0.30
    elif det_class == "DET_SG":
        coherence += 0.20 if hf.noun_singular else -0.30
    elif det_class == "ARTICLE":
        if det_word in {"a", "an"} and hf.noun_plural:
            coherence -= 0.40
        else:
            coherence += 0.15
    elif det_class == "NUM_DET":
        if det_word == "one":
            coherence += 0.12 if not hf.noun_plural else -0.20
        else:
            coherence += 0.12 if hf.noun_plural else 0.02
    elif det_class == "DET":
        coherence += 0.10

    # Conservative postnominal PP. "of" is by far the most important for
    # idioms/proverbs, with "for"/"with" covering common noun phrases.
    next_i = end + 1
    if (
        allow_post_pp
        and next_i + 1 < len(words)
        and words[next_i] in {"of", "for", "with"}
    ):
        embedded = _np_span_starting_at(
            words,
            next_i + 1,
            lex,
            allow_post_pp=False,
        )
        if embedded is not None:
            embedded_end, embedded_coh = embedded
            end = embedded_end
            coherence = min(1.0, coherence + 0.08 * embedded_coh)

    return end, max(0.0, min(1.0, coherence))



def _subject_number_from_span(
    words: Sequence[str],
    start: int,
    head_idx: int,
    lex: WordNetLexicon,
) -> str:
    """
    Determine subject number using the whole NP.

    Explicit determiners beat morphologically ambiguous heads:
      this sheep  -> 3sg
      these sheep -> non3sg
    """
    if 0 <= start <= head_idx < len(words):
        for token in words[start : head_idx + 1]:
            c = _det_class(token)
            if c == "DET_PL":
                return "non3sg"
            if c == "DET_SG":
                return "3sg"
            if c == "ARTICLE" and token in {"a", "an"}:
                return "3sg"
            if c == "NUM_DET":
                return "3sg" if token == "one" else "non3sg"
    return _subject_number(words[head_idx], lex)


def _subject_agreement(
    subject_head: str,
    aux_or_verb: str,
    lex: WordNetLexicon,
    *,
    auxiliary: bool,
    number_override: str | None = None,
) -> float:
    """
    Agreement in [0,1]. 0.5 means unknown/ambiguous.
    """
    number = number_override or _subject_number(subject_head, lex)
    c = function_class(aux_or_verb)
    vf = lex.features(aux_or_verb)

    if c == "DONT":
        if number == "non3sg":
            return 1.0
        if number == "3sg":
            return 0.0
        return 0.55

    if c == "DOESNT":
        if number == "3sg":
            return 1.0
        if number == "non3sg":
            return 0.0
        return 0.55

    if c == "BE_AUX":
        # Keep this conservative; the small function-word tables do not encode
        # every tense/person combination.
        if aux_or_verb == "is":
            return 1.0 if number == "3sg" else (0.1 if number == "non3sg" else 0.55)
        if aux_or_verb == "are":
            return 1.0 if number == "non3sg" else (0.15 if number == "3sg" else 0.55)
        return 0.65

    if c in {"MODAL", "DO_AUX", "HAVE_AUX"}:
        return 0.75

    if not auxiliary:
        if vf.verb_past:
            return 0.85
        if number == "3sg":
            if vf.verb_3sg:
                return 1.0
            if vf.verb_base and not vf.verb_3sg:
                return 0.15
        if number == "non3sg":
            if vf.verb_base:
                return 1.0
            if vf.verb_3sg and not vf.verb_base:
                return 0.1
    return 0.5


def _comparative_like(word: str, lex: WordNetLexicon) -> bool:
    if word in COMPARATIVE_WORDS:
        return True
    f = lex.features(word)
    # Surface -er is noisy, so require adjective/adverb evidence too.
    return len(word) > 3 and word.endswith("er") and (f.adj or f.adv)


def _comparative_span_starting_at(
    words: Sequence[str],
    start: int,
    lex: WordNetLexicon,
) -> tuple[int, float] | None:
    """
    Parse compact comparative expressions:
      better late than never
      better than one
      louder than words
      more bark than bite
    """
    if start >= len(words):
        return None

    max_end = min(len(words), start + 5)
    for than_idx in range(start + 1, max_end):
        if words[than_idx] != "than":
            continue

        left = words[start:than_idx]
        right = words[than_idx + 1:]
        if not left or not right:
            continue

        left_ok = (
            _comparative_like(left[0], lex)
            or any(_comparative_like(w, lex) for w in left)
            or any(lex.features(w).adj or lex.features(w).adv for w in left)
        )
        if not left_ok:
            continue

        # Right side may be an NP, adjective/adverb, or a one-token elliptical
        # element such as "never".
        right_consumed = 0
        np = _np_span_starting_at(right, 0, lex)
        if np is not None:
            right_consumed = np[0] + 1
        elif right:
            rf = lex.features(right[0])
            if rf.noun or rf.adj or rf.adv or function_class(right[0]) in {
                "PRON", "PRON_12", "PRON_PL", "PRON_SG3", "NEG", "NUM_DET"
            }:
                right_consumed = 1

        if right_consumed <= 0:
            continue

        end = than_idx + right_consumed
        quality = 0.90
        if _comparative_like(left[0], lex):
            quality += 0.05
        return end, min(1.0, quality)

    return None


def _simple_clause_span_starting_at(
    words: Sequence[str],
    start: int,
    lex: WordNetLexicon,
) -> tuple[int, float] | None:
    """
    Parse a small finite clause used mainly for subordinate tails:
      you leap
      the pot boils
      the bird sings

    This is intentionally not a recursive general parser.
    """
    if start >= len(words):
        return None

    # Find a compact subject NP. Starting-NP helper returns its end.
    np = _np_span_starting_at(words, start, lex)
    if np is None:
        return None
    subj_end, subj_coh = np
    i = subj_end + 1

    # Optional adverb/negative material before the finite verb.
    while i < len(words):
        f = lex.features(words[i])
        c = function_class(words[i])
        if c == "NEG" or (f.adv and c not in {"PREP", "CONJ"}):
            i += 1
            continue
        break

    if i >= len(words):
        return None

    vf = lex.features(words[i])
    if not vf.verb:
        return None

    subject_number = _subject_number_from_span(words, start, subj_end, lex)
    agreement = _subject_agreement(
        words[subj_end],
        words[i],
        lex,
        auxiliary=False,
        number_override=subject_number,
    )

    consumed_end = i
    valency = 0.65
    if i + 1 < len(words):
        valency, tail_n = _valency_for_tail(words[i], words[i + 1:], lex)
        consumed_end = i + tail_n
    else:
        allowed = lex.allows_intransitive(words[i])
        valency = 0.95 if allowed is True else 0.65

    quality = 0.45 * subj_coh + 0.30 * agreement + 0.25 * valency
    return consumed_end, max(0.0, min(1.0, quality))


def _subordinate_span_starting_at(
    words: Sequence[str],
    start: int,
    lex: WordNetLexicon,
) -> tuple[int, float] | None:
    if start >= len(words) or words[start] not in SUBORDINATORS:
        return None
    clause = _simple_clause_span_starting_at(words, start + 1, lex)
    if clause is None:
        return None
    end, quality = clause
    return end, min(1.0, 0.85 + 0.15 * quality)


def _valency_for_tail(
    verb: str,
    tail: Sequence[str],
    lex: WordNetLexicon,
) -> tuple[float, int]:
    """
    Return (valency quality [0,1], number of tail tokens structurally consumed).

    WordNet's published frame distinctions are respected:
      - direct nominal object
      - subject predicative complement
      - object + predicative complement
      - PP
      - infinitive/gerund/clause

    Missing frame data remains neutral rather than negative.
    """
    if not tail:
        allowed = lex.allows_intransitive(verb)
        if allowed is True:
            return 1.0, 0
        if allowed is False:
            return 0.42, 0
        return 0.65, 0

    # Subordinate clause: "look before you leap".
    subordinate = _subordinate_span_starting_at(tail, 0, lex)
    if subordinate is not None:
        end, quality = subordinate
        return 0.90 * quality, end + 1

    # Comparative/result phrase: "speak louder than words".
    comparative = _comparative_span_starting_at(tail, 0, lex)
    if comparative is not None:
        end, quality = comparative
        return 0.88 * quality, end + 1

    # A prepositional complement.
    if function_class(tail[0]) == "PREP":
        allowed = lex.allows_pp(verb)
        consumed = 1
        np = _np_span_starting_at(tail, 1, lex) if len(tail) > 1 else None
        if np is not None:
            consumed = np[0] + 2
        if allowed is True:
            return 0.95, min(consumed, len(tail))
        if allowed is False:
            return 0.45, min(consumed, len(tail))
        return 0.65, min(consumed, len(tail))

    # Direct nominal object, optionally followed by a resultative complement.
    np = _np_span_starting_at(tail, 0, lex)
    if np is not None:
        np_end, _np_coh = np
        consumed = np_end + 1

        if consumed < len(tail):
            next_f = lex.features(tail[consumed])
            if next_f.adj or next_f.noun:
                allowed_result = lex.allows_object_predicative(verb)
                if allowed_result is True:
                    return 0.98, consumed + 1

        allowed = lex.allows_object(verb)
        if allowed is True:
            return 1.0, consumed
        if allowed is False:
            return 0.08, consumed
        return 0.52, consumed

    # Subject-oriented predicative complement: "run deep", "turn red".
    tf = lex.features(tail[0])
    if tf.adj or tf.adv:
        allowed = lex.allows_predicative(verb)
        if allowed is True:
            return 0.96, 1
        if allowed is False:
            return 0.38, 1
        return 0.62, 1

    # Bare infinitival / gerund.
    if tf.verb_base or tf.verb_ing:
        allowed = lex.allows_infinitive_or_gerund(verb)
        if allowed is True:
            return 0.92, 1
        if allowed is False:
            return 0.35, 1
        return 0.56, 1

    return 0.35, 0




@dataclass(slots=True, frozen=True)
class StructureResult:
    norm: float
    valency: float
    coverage: float
    agreement: float
    kind: str
    raw: float


def phrase_structure(words: Sequence[str], lex: WordNetLexicon) -> StructureResult:
    """
    Evaluate the complete order using several general English constructions.

    linguistic reranker adds:
      * real copular clauses (NP + be + NP/Adj/comparative)
      * predicative/result complements from WordNet frames
      * subordinate clauses
      * comparative/elliptical phrases
      * adverbs between subject and finite verb
      * less-greedy noun compounds
    """
    words = tuple(words)
    n = len(words)
    if n == 0:
        return StructureResult(0.0, 0.5, 0.0, 0.5, "empty", 0.0)

    det_collisions = 0
    for a, b in itertools.pairwise(words):
        if _det_class(a) is not None and _det_class(b) is not None:
            det_collisions += 1

    candidates: list[StructureResult] = []

    # ------------------------------------------------------------------
    # 0) Standalone comparative / elliptical phrase.
    # ------------------------------------------------------------------
    comp = _comparative_span_starting_at(words, 0, lex)
    if comp is not None and comp[0] == n - 1:
        _, quality = comp
        norm = min(0.97, 0.84 + 0.13 * quality)
        candidates.append(
            StructureResult(norm, 0.78, 1.0, 0.75, "comparative", 4.0 * norm)
        )

    # ------------------------------------------------------------------
    # 1) Copular BE: subject + be + NP/adjective/comparative complement.
    # This is intentionally separate from ordinary auxiliaries.
    # ------------------------------------------------------------------
    for i, token in enumerate(words):
        if function_class(token) != "BE_AUX" or i == 0 or i + 1 >= n:
            continue

        subject_span = _np_span_ending_at(words, i - 1, lex)
        if subject_span is None:
            continue

        subj_start, subj_coh = subject_span
        subject_head_idx = i - 1
        subject_number = _subject_number_from_span(
            words, subj_start, subject_head_idx, lex
        )
        agreement = _subject_agreement(
            words[subject_head_idx],
            token,
            lex,
            auxiliary=True,
            number_override=subject_number,
        )

        comp_start = i + 1
        comp_end = -1
        comp_quality = 0.0

        # Comparative complement: "are better than one".
        comparative = _comparative_span_starting_at(words, comp_start, lex)
        if comparative is not None:
            comp_end, comp_quality = comparative

        # Nominal complement: "knowledge is power",
        # "necessity is the mother of invention".
        if comp_end < 0:
            np = _np_span_starting_at(words, comp_start, lex)
            if np is not None:
                comp_end, comp_quality = np

        # Adjectival/adverbial complement.
        if comp_end < 0:
            cf = lex.features(words[comp_start])
            if cf.adj or cf.adv:
                comp_end = comp_start
                comp_quality = 0.86

        if comp_end < 0:
            continue

        consumed = (i - subj_start) + 1 + (comp_end - comp_start + 1)
        coverage = min(1.0, consumed / n)
        if subj_start > 0:
            coverage *= 0.90

        norm = (
            0.29
            + 0.22 * agreement
            + 0.16 * subj_coh
            + 0.19 * coverage
            + 0.14 * comp_quality
        )
        norm *= 0.30 + 0.70 * (coverage ** 1.5)
        if agreement <= 0.15:
            norm = min(norm, 0.42)
        norm -= 0.22 * det_collisions
        norm = max(0.0, min(1.0, norm))

        candidates.append(
            StructureResult(norm, 1.0, coverage, agreement, "copula", 4.0 * norm)
        )

    # ------------------------------------------------------------------
    # 2) Auxiliary-led verbal predicates (excluding BE, handled above).
    # ------------------------------------------------------------------
    for i, token in enumerate(words):
        c = function_class(token)
        if c not in {"DONT", "DOESNT", "MODAL", "DO_AUX", "HAVE_AUX"}:
            continue
        if i + 1 >= n:
            continue

        vf = lex.features(words[i + 1])
        complement_ok = (
            (c in {"DONT", "DOESNT", "MODAL", "DO_AUX"} and vf.verb_base)
            or (c == "HAVE_AUX" and (vf.verb_past or vf.verb_base))
        )
        if not complement_ok:
            continue

        verb_idx = i + 1
        verb = words[verb_idx]

        subject_span = _np_span_ending_at(words, i - 1, lex) if i > 0 else None
        if subject_span is not None:
            subj_start, subj_coh = subject_span
            subject_head = words[i - 1]
            explicit = True
            subject_number = _subject_number_from_span(words, subj_start, i - 1, lex)
            agreement = _subject_agreement(
                subject_head,
                token,
                lex,
                auxiliary=True,
                number_override=subject_number,
            )
            subject_tokens = i - subj_start
            clause = 1.0
        elif i == 0 and c in {"DONT", "DO_AUX", "MODAL"}:
            subj_start = 0
            subj_coh = 0.55
            explicit = False
            agreement = 0.60
            subject_tokens = 0
            clause = 0.67
        else:
            continue

        valency, tail_consumed = _valency_for_tail(
            verb, words[verb_idx + 1:], lex
        )

        consumed = subject_tokens + 2 + tail_consumed
        coverage = min(1.0, consumed / n)
        if explicit and subj_start > 0:
            coverage *= 0.90

        norm = (
            0.32 * clause
            + 0.20 * agreement
            + 0.16 * subj_coh
            + 0.18 * coverage
            + 0.14 * valency
        )
        norm *= 0.30 + 0.70 * (coverage ** 1.5)
        if explicit and agreement <= 0.15:
            norm = min(norm, 0.42)
        norm -= 0.22 * det_collisions
        norm = max(0.0, min(1.0, norm))

        candidates.append(
            StructureResult(norm, valency, coverage, agreement, "clause", 4.0 * norm)
        )

    # ------------------------------------------------------------------
    # 3) Simple lexical finite-verb clause.
    # Allows subject + adverb(s) + verb, e.g. "the pot never boils".
    # ------------------------------------------------------------------
    for i, token in enumerate(words):
        vf = lex.features(token)
        if not vf.verb:
            continue

        # Walk left across adverb/negative modifiers to locate the subject.
        subj_head_idx = i - 1
        while subj_head_idx >= 0:
            sf = lex.features(words[subj_head_idx])
            sc = function_class(words[subj_head_idx])
            if sc == "NEG" or (sf.adv and sc not in {"PREP", "CONJ"}):
                subj_head_idx -= 1
                continue
            break

        if subj_head_idx < 0:
            continue

        subject_span = _np_span_ending_at(words, subj_head_idx, lex)
        if subject_span is None:
            continue

        subj_start, subj_coh = subject_span
        subject_head = words[subj_head_idx]
        subject_number = _subject_number_from_span(
            words, subj_start, subj_head_idx, lex
        )
        agreement = _subject_agreement(
            subject_head,
            token,
            lex,
            auxiliary=False,
            number_override=subject_number,
        )

        finite_conf = 0.95 if (vf.verb_3sg or vf.verb_past) else 0.72
        if subject_number == "non3sg" and vf.verb_base:
            finite_conf = 0.95

        valency, tail_consumed = _valency_for_tail(
            token, words[i + 1:], lex
        )

        subject_tokens = subj_head_idx - subj_start + 1
        intervening = i - subj_head_idx - 1
        consumed = subject_tokens + intervening + 1 + tail_consumed
        coverage = min(1.0, consumed / n)
        if subj_start > 0:
            coverage *= 0.90

        norm = (
            0.27 * finite_conf
            + 0.24 * agreement
            + 0.17 * subj_coh
            + 0.18 * coverage
            + 0.14 * valency
        )
        norm *= 0.30 + 0.70 * (coverage ** 1.5)
        if agreement <= 0.15:
            norm = min(norm, 0.42)
        norm -= 0.22 * det_collisions
        norm = max(0.0, min(1.0, norm))

        candidates.append(
            StructureResult(norm, valency, coverage, agreement, "clause", 4.0 * norm)
        )

    # ------------------------------------------------------------------
    # 4) Coherent noun phrase.
    # ------------------------------------------------------------------
    np = _np_span_starting_at(words, 0, lex)
    if np is not None and np[0] == n - 1:
        np_coh = np[1]
        norm = max(
            0.0,
            min(0.90, 0.70 + 0.20 * np_coh - 0.22 * det_collisions),
        )
        candidates.append(
            StructureResult(norm, 0.60, 1.0, 0.70, "noun-phrase", 4.0 * norm)
        )

    # ------------------------------------------------------------------
    # 5) Imperative bare verb + complement.
    # ------------------------------------------------------------------
    if lex.features(words[0]).verb_base and function_class(words[0]) is None:
        valency, tail_consumed = _valency_for_tail(words[0], words[1:], lex)
        coverage = min(1.0, (1 + tail_consumed) / n)
        norm = 0.52 + 0.20 * valency + 0.18 * coverage
        norm *= 0.25 + 0.75 * (coverage ** 1.7)

        leftovers = words[1 + tail_consumed:]
        if any(function_class(w) is not None for w in leftovers):
            norm -= 0.25

        norm -= 0.22 * det_collisions
        norm = max(0.0, min(0.88, norm))
        candidates.append(
            StructureResult(norm, valency, coverage, 0.60, "imperative", 4.0 * norm)
        )

    if not candidates:
        recognized = content_coverage(words, lex)
        norm = max(0.05, 0.22 * recognized - 0.20 * det_collisions)
        return StructureResult(
            norm, 0.50, 0.25 * recognized, 0.50, "fragment", 4.0 * norm
        )

    return max(candidates, key=lambda r: (r.norm, r.coverage, r.valency))



def local_grammar_raw(words: Sequence[str], lex: WordNetLexicon) -> float:
    if not words:
        return 0.0
    if len(words) == 1:
        return start_score(words[0], lex) + end_score(words[0], lex)
    total = start_score(words[0], lex) + end_score(words[-1], lex)
    total += sum(pair_grammar(a, b, lex) for a, b in itertools.pairwise(words))
    return total / (len(words) - 1)



def _order_local_tables(
    words: tuple[str, ...],
    lex: WordNetLexicon,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    """
    Precompute all local ordering evidence once per word bag.

    earlier reranker recomputed pair_grammar(), function classes and feature lookups inside
    every permutation. A six-word bag has only 30 directed pairs, so caching
    this matrix turns the ordering search into mostly tuple indexing + floats.
    """
    n = len(words)
    pair_rows: list[tuple[float, ...]] = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0.0 if i == j else pair_grammar(words[i], words[j], lex))
        pair_rows.append(tuple(row))
    starts = tuple(start_score(w, lex) for w in words)
    ends = tuple(end_score(w, lex) for w in words)
    return tuple(pair_rows), starts, ends


def _local_raw_indices(
    order: tuple[int, ...],
    pair: tuple[tuple[float, ...], ...],
    starts: tuple[float, ...],
    ends: tuple[float, ...],
) -> float:
    n = len(order)
    if n == 0:
        return 0.0
    if n == 1:
        idx = order[0]
        return starts[idx] + ends[idx]
    total = starts[order[0]] + ends[order[-1]]
    total += sum(pair[a][b] for a, b in itertools.pairwise(order))
    return total / (n - 1)


def _exact_index_orders(n: int) -> Iterator[tuple[int, ...]]:
    """
    Allocation-light exact permutation generator for 1..6 unique positions.
    Avoids earlier reranker's set(permutations(...)) tuple hashing/allocation festival.
    """
    path = [0] * n

    def rec(depth: int, used: int):
        if depth == n:
            yield tuple(path)
            return
        for idx in range(n):
            bit = 1 << idx
            if used & bit:
                continue
            path[depth] = idx
            yield from rec(depth + 1, used | bit)

    yield from rec(0, 0)


def _kbest_local_orders(
    n: int,
    pair: tuple[tuple[float, ...], ...],
    starts: tuple[float, ...],
    ends: tuple[float, ...],
    max_complete: int,
) -> list[tuple[int, ...]]:
    """
    K-best dynamic-programming search over word orders.

    State is (used-mask, last-word). We retain several paths per state rather
    than only the single Viterbi path, preserving ordering diversity for the
    later non-decomposable whole-clause scorer.

    Complexity is roughly O(n * 2^n * K), tiny for n<=6, while the expensive
    global phrase parser sees at most max_complete full orders instead of n!.
    """
    if n <= 1:
        return [(0,)] if n == 1 else [()]

    # Enough paths per terminal word to supply roughly max_complete full paths.
    per_state = max(2, math.ceil(max_complete / max(1, n)))

    # key -> list[(score_sum, path)]
    states: dict[tuple[int, int], list[tuple[float, tuple[int, ...]]]] = {
        (1 << i, i): [(starts[i], (i,))] for i in range(n)
    }

    for depth in range(1, n):
        nxt: dict[tuple[int, int], list[tuple[float, tuple[int, ...]]]] = defaultdict(list)
        for (mask, last), variants in states.items():
            for score_sum, path in variants:
                for j in range(n):
                    bit = 1 << j
                    if mask & bit:
                        continue
                    key = (mask | bit, j)
                    nxt[key].append((score_sum + pair[last][j], path + (j,)))

        # Prune independently per DP state, which keeps more structural
        # diversity than a single global beam.
        pruned: dict[tuple[int, int], list[tuple[float, tuple[int, ...]]]] = {}
        for key, variants in nxt.items():
            if len(variants) > per_state:
                variants = heapq.nlargest(per_state, variants, key=lambda x: x[0])
            else:
                variants.sort(key=lambda x: x[0], reverse=True)
            pruned[key] = variants
        states = pruned

    completed: list[tuple[float, tuple[int, ...]]] = []
    full = (1 << n) - 1
    for (mask, last), variants in states.items():
        if mask != full:
            continue
        for score_sum, path in variants:
            completed.append((score_sum + ends[last], path))

    completed = heapq.nlargest(max_complete, completed, key=lambda x: x[0])
    return [path for _, path in completed]


def best_order(
    words: Sequence[str],
    lex: WordNetLexicon,
    *,
    order_mode: str = "auto",
    beam_width: int = 128,
    exact_max_words: int = 5,
) -> tuple[float, tuple[str, ...], StructureResult, int]:
    """
    Find the best ordering while minimizing expensive global structure parses.

    * exact: evaluate all n! orders
    * beam:  evaluate only k-best locally plausible full orders
    * auto:  exact through exact_max_words, k-best above it

    earlier reranker's linguistic objective is preserved. Only candidate-order generation
    changes for bags above exact_max_words.
    """
    words = tuple(words)
    n = len(words)
    if n <= 1:
        structure = phrase_structure(words, lex)
        return local_grammar_raw(words, lex), words, structure, 1

    pair, starts, ends = _order_local_tables(words, lex)

    use_exact = order_mode == "exact" or (
        order_mode == "auto" and n <= exact_max_words
    )

    if use_exact:
        order_iter: Iterable[tuple[int, ...]] = _exact_index_orders(n)
    else:
        order_iter = _kbest_local_orders(
            n,
            pair,
            starts,
            ends,
            max_complete=max(1, beam_width),
        )

    best_obj = -1e30
    best_local = -1e30
    best_ordering: tuple[str, ...] = words
    best_structure = StructureResult(0.0, 0.5, 0.0, 0.5, "fragment", 0.0)
    evaluated = 0

    for idx_order in order_iter:
        evaluated += 1
        local_raw = _local_raw_indices(idx_order, pair, starts, ends)
        local_norm = grammar_normalize(local_raw)
        word_order = tuple(words[i] for i in idx_order)
        structure = phrase_structure(word_order, lex)

        objective = (
            0.38 * local_norm
            + 0.44 * structure.norm
            + 0.12 * structure.valency
            + 0.06 * structure.coverage
        )
        if objective > best_obj:
            best_obj = objective
            best_local = local_raw
            best_ordering = word_order
            best_structure = structure

    return best_local, best_ordering, best_structure, evaluated


def grammar_normalize(raw_per_edge: float) -> float:
    return sigmoid((raw_per_edge - 0.85) * 1.25)


def score_pre(row: Row) -> float:
    """
    Shortlist score. Kept intentionally close to earlier reranker because earlier reranker already moved
    the benchmark from #1749 to #29 before deep analysis.
    """
    return 100.0 * (
        0.18 * row.lex
        + 0.28 * row.fam
        + 0.16 * row.hint
        + 0.28 * row.grammar_potential_norm
        + 0.10 * row.wn_coverage
    )


def score_final(row: Row) -> float:
    """
    linguistic reranker final score.

    Local grammar still matters, but a candidate must also form a coherent
    global phrase/clause. WordNet valency is independent evidence rather than
    another disguised lexical-frequency term.
    """
    return 100.0 * (
        0.10 * row.lex
        + 0.16 * row.fam
        + 0.12 * row.hint
        + 0.22 * row.grammar_norm
        + 0.28 * row.structure_norm
        + 0.08 * row.valency_norm
        + 0.04 * row.wn_coverage
    )


def prepare_rows(rows: list[Row], lex: WordNetLexicon) -> None:
    for idx, row in enumerate(rows, 1):
        row.wn_coverage = content_coverage(row.words, lex)
        row.grammar_potential_norm = grammar_potential(row.words, lex)
        row.family_key = morphology_family(row.words, lex)
        row.pre_score = score_pre(row)
        if idx % 25000 == 0:
            print(f"  prepared {idx:,} / {len(rows):,}")


def choose_deep(rows: list[Row], per_group: int, deep_all: bool) -> set[int]:
    """
    Choose rows by linguistic reranker PRE, then expand every selected morphology family.
    That prevents a high-frequency bad inflection from hiding its grammatically
    correct sibling.
    """
    selected_ids: set[int] = set()

    by_wc: dict[int, list[tuple[int, Row]]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_wc[row.word_count].append((i, row))

    selected_families: set[tuple[int, tuple[str, ...]]] = set()

    for wc, bucket in sorted(by_wc.items()):
        bucket.sort(key=lambda ir: (ir[1].pre_score, ir[1].fam, ir[1].lex), reverse=True)
        chosen = bucket if deep_all else bucket[:per_group]
        print(f"Deep shortlist {wc} words: {len(chosen):,} / {len(bucket):,}")
        for i, row in chosen:
            selected_ids.add(i)
            selected_families.add((wc, row.family_key))

    # Family expansion.
    before = len(selected_ids)
    for i, row in enumerate(rows):
        if (row.word_count, row.family_key) in selected_families:
            selected_ids.add(i)
    extra = len(selected_ids) - before
    if extra:
        print(f"Morphology-family expansion added {extra:,} candidate(s).")

    return selected_ids



def _worker_init(
    wordnet_dir: str,
    order_mode: str,
    beam_width: int,
    exact_max_words: int,
) -> None:
    global _WORKER_LEX, _WORKER_ORDER_MODE, _WORKER_BEAM_WIDTH, _WORKER_EXACT_MAX_WORDS
    _WORKER_LEX = WordNetLexicon.load(Path(wordnet_dir))
    _WORKER_ORDER_MODE = order_mode
    _WORKER_BEAM_WIDTH = beam_width
    _WORKER_EXACT_MAX_WORDS = exact_max_words


def _worker_analyze_batch(
    batch: tuple[tuple[int, tuple[str, ...]], ...],
) -> list[DeepResult]:
    if _WORKER_LEX is None:
        raise RuntimeError("Worker WordNet lexicon was not initialized.")

    out: list[DeepResult] = []
    for row_index, words in batch:
        raw, order, structure, evaluated = best_order(
            words,
            _WORKER_LEX,
            order_mode=_WORKER_ORDER_MODE,
            beam_width=_WORKER_BEAM_WIDTH,
            exact_max_words=_WORKER_EXACT_MAX_WORDS,
        )
        out.append(
            DeepResult(
                row_index=row_index,
                grammar_raw=raw,
                best_order=order,
                structure_norm=structure.norm,
                valency_norm=structure.valency,
                syntax_coverage=structure.coverage,
                phrase_kind=structure.kind,
                orders_evaluated=evaluated,
            )
        )
    return out


def _apply_deep_result(rows: list[Row], result: DeepResult) -> None:
    row = rows[result.row_index]
    row.deep = True
    row.grammar_raw = result.grammar_raw
    row.grammar_norm = grammar_normalize(result.grammar_raw)
    row.best_order = result.best_order
    row.structure_norm = result.structure_norm
    row.valency_norm = result.valency_norm
    row.syntax_coverage = result.syntax_coverage
    row.phrase_kind = result.phrase_kind
    row.final = score_final(row)
    row.base_final = row.final


def deep_analyze(
    rows: list[Row],
    selected: set[int],
    lex: WordNetLexicon,
    *,
    wordnet_dir: Path,
    backend: str,
    workers: int,
    batch_size: int,
    order_mode: str,
    beam_width: int,
    exact_max_words: int,
) -> dict[str, float]:
    """
    Analyze independent candidate bags in parallel.

    Ordinary CPython defaults to processes to bypass the GIL. A free-threaded
    build defaults to threads and shares the already-loaded WordNet lexicon.
    """
    selected_sorted = sorted(selected)
    total = len(selected_sorted)
    if total == 0:
        return {"seconds": 0.0, "orders": 0.0, "candidates": 0.0}

    resolved_backend = resolve_backend(backend, workers)
    print(
        f"Deep backend: {resolved_backend}; workers={workers}; "
        f"order_mode={order_mode}; exact<= {exact_max_words}; "
        f"k-best width={beam_width}; batch={batch_size}"
    )

    t0 = time.perf_counter()
    done = 0
    total_orders = 0

    def progress(increment: int, order_count: int) -> None:
        nonlocal done, total_orders
        done += increment
        total_orders += order_count
        if done == total or done % 2000 < increment:
            elapsed = max(1e-9, time.perf_counter() - t0)
            print(
                f"  deep-analyzed {done:,} / {total:,} "
                f"({done/elapsed:,.1f} candidates/s; "
                f"{total_orders/elapsed:,.0f} orders/s)"
            )

    if resolved_backend == "serial":
        for row_index in selected_sorted:
            row = rows[row_index]
            raw, order, structure, evaluated = best_order(
                row.words,
                lex,
                order_mode=order_mode,
                beam_width=beam_width,
                exact_max_words=exact_max_words,
            )
            result = DeepResult(
                row_index=row_index,
                grammar_raw=raw,
                best_order=order,
                structure_norm=structure.norm,
                valency_norm=structure.valency,
                syntax_coverage=structure.coverage,
                phrase_kind=structure.kind,
                orders_evaluated=evaluated,
            )
            _apply_deep_result(rows, result)
            progress(1, evaluated)

    elif resolved_backend == "thread":
        # Used primarily on free-threaded Python. Prime the shared globals.
        global _WORKER_LEX, _WORKER_ORDER_MODE, _WORKER_BEAM_WIDTH, _WORKER_EXACT_MAX_WORDS
        _WORKER_LEX = lex
        _WORKER_ORDER_MODE = order_mode
        _WORKER_BEAM_WIDTH = beam_width
        _WORKER_EXACT_MAX_WORDS = exact_max_words

        payloads = [
            tuple((i, rows[i].words) for i in batch)
            for batch in chunked(selected_sorted, batch_size)
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker_analyze_batch, payload) for payload in payloads]
            for fut in as_completed(futures):
                results = fut.result()
                orders = 0
                for result in results:
                    _apply_deep_result(rows, result)
                    orders += result.orders_evaluated
                progress(len(results), orders)

    elif resolved_backend == "process":
        payloads = [
            tuple((i, rows[i].words) for i in batch)
            for batch in chunked(selected_sorted, batch_size)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(
                str(wordnet_dir),
                order_mode,
                beam_width,
                exact_max_words,
            ),
        ) as pool:
            futures = [pool.submit(_worker_analyze_batch, payload) for payload in payloads]
            for fut in as_completed(futures):
                results = fut.result()
                orders = 0
                for result in results:
                    _apply_deep_result(rows, result)
                    orders += result.orders_evaluated
                progress(len(results), orders)
    else:
        raise ValueError(f"Unsupported backend: {resolved_backend}")

    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "orders": float(total_orders),
        "candidates": float(total),
    }


def format_row(rank: int, row: Row) -> str:
    if row.deep:
        phrase = pretty_phrase(row.best_order)
        return (
            f"{rank:7d}. FINAL={row.final:6.2f} PRE={row.pre_score:6.2f} "
            f"GRAM={row.grammar_norm:5.3f} STRUCT={row.structure_norm:5.3f} "
            f"VAL={row.valency_norm:4.2f} COV={row.syntax_coverage:4.2f} "
            f"KIND={row.phrase_kind:<11} WN={row.wn_coverage:4.2f} "
            f"LEX={row.lex:5.3f} FAM={row.fam:5.3f} HINT={row.hint:5.3f} "
            f"COLL={row.colloc_norm:4.2f} PHRASE={row.phrase_attest_norm:4.2f} "
            f"PBONUS={row.phrase_bonus:4.2f} "
            f"{phrase}  [CANON={' '.join(row.words)}; "
            f"OLDPRE={row.old_pre:.2f}; OLDRANK={row.old_rank}]"
        )
    return (
        f"{rank:7d}. PRE={row.pre_score:6.2f} GPOT={row.grammar_potential_norm:5.3f} "
        f"WN={row.wn_coverage:4.2f} LEX={row.lex:5.3f} FAM={row.fam:5.3f} "
        f"HINT={row.hint:5.3f} {' '.join(row.words)} "
        f"[OLDPRE={row.old_pre:.2f}; OLDRANK={row.old_rank}]"
    )


def rank_buckets(rows: list[Row]) -> dict[int, list[Row]]:
    by_wc: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        by_wc[row.word_count].append(row)

    for bucket in by_wc.values():
        bucket.sort(
            key=lambda r: (
                1 if r.deep else 0,
                r.final if r.deep else r.pre_score,
                r.pre_score,
            ),
            reverse=True,
        )
    return dict(by_wc)


def write_export(path: Path, buckets: dict[int, list[Row]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for wc in sorted(buckets):
            bucket = buckets[wc]
            deep_count = sum(1 for r in bucket if r.deep)
            f.write(
                f"=== {wc}-WORD RERANKED CANDIDATES "
                f"({len(bucket)} total; {deep_count} deep-analyzed) ===\n"
            )

            # Deep rows first, ranked by FINAL. Then non-deep by PRE.
            deep_rows = sorted(
                (r for r in bucket if r.deep),
                key=lambda r: (r.final, r.pre_score),
                reverse=True,
            )
            pre_rows = sorted(
                (r for r in bucket if not r.deep),
                key=lambda r: r.pre_score,
                reverse=True,
            )

            rank = 1
            for r in deep_rows:
                f.write(format_row(rank, r) + "\n")
                rank += 1

            if pre_rows:
                f.write("--- NOT DEEP-ANALYZED; PRE ONLY ---\n")
                for r in pre_rows:
                    f.write(format_row(rank, r) + "\n")
                    rank += 1
            f.write("\n")


def benchmark(answer: str, rows: list[Row]) -> None:
    wanted = tuple(sorted(norm_token(x) for x in answer.split() if norm_token(x)))
    matches = [r for r in rows if tuple(sorted(r.words)) == wanted]

    print()
    print(f"BENCHMARK: {answer}")
    if not matches:
        print("  NOT FOUND in input export.")
        return

    target = matches[0]
    bucket = [r for r in rows if r.word_count == target.word_count]

    old_rank = target.old_rank
    pre_sorted = sorted(bucket, key=lambda r: r.pre_score, reverse=True)
    pre_rank = pre_sorted.index(target) + 1

    print(f"  generator PRE rank: {old_rank:,} / {len(bucket):,}")
    print(f"  PRE rank: {pre_rank:,} / {len(bucket):,}  ({target.pre_score:.2f})")
    print(f"  canonical:   {' '.join(target.words)}")
    print(f"  family:      {' / '.join(target.family_key)}")
    print(f"  WN coverage: {target.wn_coverage:.2f}")
    print(f"  grammar potential: {target.grammar_potential_norm:.3f}")
    print(f"  hint:        {target.hint:.3f} ({', '.join(target.hints) or 'none'})")

    if target.deep:
        deep_bucket = [r for r in bucket if r.deep]
        final_sorted = sorted(deep_bucket, key=lambda r: (r.final, r.pre_score), reverse=True)
        final_rank = final_sorted.index(target) + 1
        print(
            f"  FINAL rank: {final_rank:,} / {len(deep_bucket):,} "
            f"({target.final:.2f})"
        )
        print(f"  best order:  {pretty_phrase(target.best_order)}")
        print(f"  grammar raw: {target.grammar_raw:.3f}")
        print(f"  grammar norm:{target.grammar_norm:.3f}")
        print(f"  structure:   {target.structure_norm:.3f} ({target.phrase_kind})")
        print(f"  valency:     {target.valency_norm:.3f}")
        print(f"  coverage:    {target.syntax_coverage:.3f}")
        print(f"  collocation: {target.colloc_norm:.3f}")
        print(f"  phrase prior:{target.phrase_attest_norm:.3f}")
        print(f"  phrase bonus:{target.phrase_bonus:.3f}")
    else:
        print("  FINAL: not deep-analyzed; increase --deep-per-group or use --deep-all")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fast WordNet + phrase-prior reranker for generator anagram exports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", type=Path, help="candidates.txt produced by anagram_generate.py")
    ap.add_argument("--wordnet-dir", type=Path, default=DEFAULT_WORDNET_DIR)
    ap.add_argument("--refresh-wordnet", action="store_true")
    ap.add_argument(
        "--deep-per-group",
        type=int,
        default=5000,
        help="Candidates per word-count bucket sent to deep analysis before morphology-family expansion",
    )
    ap.add_argument("--deep-all", action="store_true")
    ap.add_argument(
        "--backend",
        choices=("auto", "serial", "thread", "process"),
        default="auto",
        help="Deep-analysis execution backend; auto uses processes with the GIL and threads on free-threaded Python",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker count; 0 chooses a conservative automatic value",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Candidate bags sent to each parallel task",
    )
    ap.add_argument(
        "--order-mode",
        choices=("auto", "exact", "beam"),
        default="auto",
        help="Ordering search: exact all permutations, k-best search, or exact only for short bags",
    )
    ap.add_argument(
        "--beam-width",
        type=int,
        default=128,
        help="Maximum complete orders globally parsed for k-best ordering search",
    )
    ap.add_argument(
        "--exact-max-words",
        type=int,
        default=5,
        help="In auto mode, use exact ordering through this many words",
    )
    ap.add_argument("--top-per-group", type=int, default=100)
    ap.add_argument("--export", type=Path, default=Path("reranked.txt"))

    ap.add_argument(
        "--prepared-cache-dir",
        type=Path,
        default=DEFAULT_PREPARED_CACHE_DIR,
        help="Persistent cache for parsed + WordNet-prepared candidate rows",
    )
    ap.add_argument("--no-prepared-cache", action="store_true")
    ap.add_argument("--rebuild-prepared-cache", action="store_true")

    ap.add_argument(
        "--phrase-rescore-top",
        type=int,
        default=300,
        help="Top deep candidates per word-count bucket receiving phrase/collocation rescoring",
    )
    ap.add_argument(
        "--phrase-bonus-max",
        type=float,
        default=5.0,
        help="Maximum additive late-stage phrase prior bonus",
    )
    ap.add_argument(
        "--no-positive-bigrams",
        action="store_true",
        help="Disable the positive-only Norvig bigram collocation prior",
    )
    ap.add_argument(
        "--ngram-dir",
        type=Path,
        default=DEFAULT_NGRAM_DIR,
        help="Location/cache for count_1w.txt and count_2w.txt",
    )
    ap.add_argument(
        "--phrase-db",
        type=Path,
        help="Optional SQLite n-gram/phrase index built by build_phrase_index.py",
    )

    ap.add_argument("--benchmark-answer")
    args = ap.parse_args()

    if args.deep_per_group < 1:
        raise SystemExit("--deep-per-group must be >= 1")
    if args.workers < 0:
        raise SystemExit("--workers must be >= 0")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.beam_width < 1:
        raise SystemExit("--beam-width must be >= 1")
    if args.exact_max_words < 1:
        raise SystemExit("--exact-max-words must be >= 1")
    if args.top_per_group < 1:
        raise SystemExit("--top-per-group must be >= 1")
    if args.phrase_rescore_top < 1:
        raise SystemExit("--phrase-rescore-top must be >= 1")
    if args.phrase_bonus_max < 0:
        raise SystemExit("--phrase-bonus-max must be >= 0")
    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    overall_t0 = time.perf_counter()
    stage_times: dict[str, float] = {}

    _stage_t0 = time.perf_counter()
    wn_dir = ensure_wordnet(args.wordnet_dir.expanduser(), refresh=args.refresh_wordnet)
    print(f"Loading WordNet from {wn_dir} ...")
    lex = WordNetLexicon.load(wn_dir)
    stage_times["wordnet"] = time.perf_counter() - _stage_t0
    print(
        f"WordNet lemmas: nouns={len(lex.nouns):,}, verbs={len(lex.verbs):,}, "
        f"adjectives={len(lex.adjs):,}, adverbs={len(lex.advs):,}, "
        f"verbs-with-frames={len(lex.verb_frames):,}"
    )

    cache_path: Path | None = None
    rows: list[Row] | None = None
    if not args.no_prepared_cache:
        cache_key = _prepared_cache_key(args.input, wn_dir)
        prepared_cache_path = args.prepared_cache_dir.expanduser() / f"{cache_key}.pickle"
        cache_path = prepared_cache_path
        if prepared_cache_path.exists() and not args.rebuild_prepared_cache:
            print(f"Loading prepared candidate cache: {prepared_cache_path}")
            _stage_t0 = time.perf_counter()
            rows = load_prepared_cache(prepared_cache_path)
            stage_times["cache_load"] = time.perf_counter() - _stage_t0
            if rows is not None:
                print(f"Loaded {len(rows):,} prepared candidate word set(s) from cache.")

    if rows is None:
        print(f"Parsing {args.input} ...")
        _stage_t0 = time.perf_counter()
        rows = parse_candidates(args.input)
        stage_times["parse"] = time.perf_counter() - _stage_t0
        if not rows:
            raise SystemExit("No generator PRE-ranked records found. Is this a candidates.txt export?")
        print(f"Parsed {len(rows):,} candidate word set(s).")

        print("Preparing grammar/morphology features ...")
        _stage_t0 = time.perf_counter()
        prepare_rows(rows, lex)
        stage_times["prepare"] = time.perf_counter() - _stage_t0

        if cache_path is not None:
            print(f"Saving prepared candidate cache: {cache_path}")
            _stage_t0 = time.perf_counter()
            save_prepared_cache(cache_path, rows)
            stage_times["cache_save"] = time.perf_counter() - _stage_t0

    _stage_t0 = time.perf_counter()
    selected = choose_deep(rows, args.deep_per_group, args.deep_all)
    stage_times["shortlist"] = time.perf_counter() - _stage_t0

    workers = resolve_workers(args.workers)
    print(f"Deep grammar ordering for {len(selected):,} candidate(s) ...")
    deep_stats = deep_analyze(
        rows,
        selected,
        lex,
        wordnet_dir=wn_dir,
        backend=args.backend,
        workers=workers,
        batch_size=args.batch_size,
        order_mode=args.order_mode,
        beam_width=args.beam_width,
        exact_max_words=args.exact_max_words,
    )
    stage_times["deep"] = deep_stats["seconds"]

    # Late-stage phrase prior. Sparse corpus absence is neutral.
    _stage_t0 = time.perf_counter()
    collocation_model: PositiveBigramModel | None = None
    phrase_index: PhraseIndex | None = None

    if not args.no_positive_bigrams:
        deep_vocab = {
            w
            for row in rows
            if row.deep
            for w in row.best_order
        }
        one_path, two_path = ensure_norvig_ngrams(args.ngram_dir.expanduser())
        print(f"Loading positive-only bigram evidence for {len(deep_vocab):,} word(s) ...")
        collocation_model = load_positive_bigram_model(
            one_path, two_path, deep_vocab
        )
        print(
            f"Loaded {len(collocation_model.bigram_counts):,} relevant observed bigram(s)."
        )

    if args.phrase_db:
        phrase_db_path = args.phrase_db.expanduser()
        if not phrase_db_path.is_file():
            raise SystemExit(f"--phrase-db not found: {phrase_db_path}")
        phrase_index = PhraseIndex.open(phrase_db_path)
        print(
            f"Opened phrase index {phrase_db_path} "
            f"(max n-gram length {phrase_index.max_n})."
        )

    rescored = apply_phrase_rescore(
        rows,
        collocation=collocation_model,
        phrase_index=phrase_index,
        top_per_group=args.phrase_rescore_top,
        bonus_max=args.phrase_bonus_max,
    )
    stage_times["phrase"] = time.perf_counter() - _stage_t0
    print(f"Phrase/collocation rescored {rescored:,} deep candidate(s).")

    _stage_t0 = time.perf_counter()
    buckets = rank_buckets(rows)
    stage_times["rank"] = time.perf_counter() - _stage_t0

    for wc in sorted(buckets):
        bucket = buckets[wc]
        deep_rows = sorted(
            (r for r in bucket if r.deep),
            key=lambda r: (r.final, r.pre_score),
            reverse=True,
        )
        print()
        print(
            f"=== {wc}-WORD linguistic reranker FINAL "
            f"(showing {min(args.top_per_group, len(deep_rows))} "
            f"of {len(deep_rows)} deep; {len(bucket)} total) ==="
        )
        for rank, row in enumerate(deep_rows[:args.top_per_group], 1):
            print(format_row(rank, row))

    _stage_t0 = time.perf_counter()
    write_export(args.export, buckets)
    stage_times["export"] = time.perf_counter() - _stage_t0
    print(f"\nWrote {args.export}")

    if args.benchmark_answer:
        benchmark(args.benchmark_answer, rows)

    total_elapsed = time.perf_counter() - overall_t0
    print("\n=== linguistic reranker TIMINGS ===")
    for name in (
        "wordnet", "cache_load", "parse", "prepare", "cache_save",
        "shortlist", "deep", "phrase", "rank", "export"
    ):
        if name in stage_times:
            print(f"  {name:<10} {stage_times[name]:9.3f} s")
    print(f"  {'total':<10} {total_elapsed:9.3f} s")
    if deep_stats.get("seconds", 0.0) > 0:
        print(
            f"  deep work  {int(deep_stats['candidates']):,} candidates, "
            f"{int(deep_stats['orders']):,} globally parsed orders"
        )
        print(
            f"  throughput {deep_stats['candidates']/deep_stats['seconds']:,.1f} "
            f"candidates/s, {deep_stats['orders']/deep_stats['seconds']:,.0f} orders/s"
        )
    print(
        f"  runtime    backend={resolve_backend(args.backend, workers)}, "
        f"workers={workers}, GIL={'on' if _gil_enabled() else 'off'}"
    )

    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
