#!/usr/bin/env python3
"""
Exact multi-word anagram candidate generator and lexical pre-ranker.

The generator searches exact letter decompositions, applies lexical/clue filters,
groups morphology variants, and emits a stable candidate export consumed by the
linguistic reranker. Optional dictionaries and Norvig unigram/bigram data
are cached under the project-local .anagram_data directory.

Python standard library only.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from anagram_paths import DICTIONARY_DIR, NGRAM_DIR

DEFAULT_DICT_URL = "https://phillipmfeldman.org/English/large.txt"
DEFAULT_CACHE_DIR = DICTIONARY_DIR
DEFAULT_DICT_CACHE = DICTIONARY_DIR / "large.txt"

NORVIG_1W_URL = "https://norvig.com/ngrams/count_1w.txt"
NORVIG_2W_URL = "https://norvig.com/ngrams/count_2w.txt"
DEFAULT_NGRAM_DIR = NGRAM_DIR

A_ORD = ord("a")

# We deliberately distinguish "real short function words" from "whatever a
# giant word-list happens to contain". This is the default safe set.
DEFAULT_SHORT_WORDS = {
    "a", "i",
    "am", "an", "as", "at",
    "be", "by",
    "do",
    "go",
    "he",
    "if", "in", "is", "it",
    "me", "my",
    "no",
    "of", "oh", "on", "or",
    "so",
    "to",
    "up", "us",
    "we",
}

# Contraction spellings that are commonly written without punctuation in
# puzzles/chat. These are only for prettier rendering; letters stay unchanged.
PRETTY_CONTRACTIONS = {
    "dont": "don't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "couldnt": "couldn't",
    "shouldnt": "shouldn't",
    "wouldnt": "wouldn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hadnt": "hadn't",
}

# Avoid absurd stemming such as this -> thi and his -> hi.
S_STEM_EXCEPTIONS = {
    "as", "gas", "has", "his", "is", "news", "this", "thus", "us", "was", "yes",
}

IRREGULAR_ROOTS = {
    "does": "do",
    "did": "do",
    "done": "do",
    "has": "have",
    "had": "have",
    "was": "be",
    "were": "be",
    "went": "go",
    "gone": "go",
}


def normalize_letters(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in text.lower() if "a" <= ch <= "z")


def normalize_token(text: str) -> str:
    return normalize_letters(text)


def tokenize_words(text: str) -> list[str]:
    # Preserve apostrophe-ish words before normalization.
    parts = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", text)
    return [normalize_token(x) for x in parts if normalize_token(x)]


def counts(text: str) -> tuple[int, ...]:
    out = [0] * 26
    for ch in normalize_letters(text):
        out[ord(ch) - A_ORD] += 1
    return tuple(out)


def subtract_counts(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...] | None:
    result = tuple(x - y for x, y in zip(a, b))
    return result if all(x >= 0 for x in result) else None


def fits(word_counts: tuple[int, ...], remaining: tuple[int, ...]) -> bool:
    return all(w <= r for w, r in zip(word_counts, remaining))


def split_values(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def download_file(url: str, path: Path, refresh: bool = False) -> Path:
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "multi-anagram-generator/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response, path.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return path


def get_dictionary(source: str, refresh: bool = False) -> Path:
    if not re.match(r"^https?://", source, re.IGNORECASE):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dictionary not found: {path}")
        return path

    if source == DEFAULT_DICT_URL:
        path = DEFAULT_DICT_CACHE
    else:
        basename = source.rsplit("/", 1)[-1] or "dictionary.txt"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", basename)
        path = DEFAULT_CACHE_DIR / safe
    return download_file(source, path, refresh=refresh)


@dataclass(slots=True)
class UnigramModel:
    counts: dict[str, int]
    total: int

    def count(self, word: str) -> int:
        return self.counts.get(normalize_token(word), 0)

    def zipf(self, word: str) -> float:
        """
        Zipf-style log10 occurrences per billion tokens.

        This is corpus-derived and intentionally not claimed to be identical to
        the third-party wordfreq package's score.
        """
        c = self.count(word)
        if c <= 0 or self.total <= 0:
            return 0.0
        return math.log10((c / self.total) * 1_000_000_000.0)


@dataclass(slots=True)
class BigramModel:
    unigrams: UnigramModel
    counts: dict[tuple[str, str], int]

    def bigram_count(self, left: str, right: str) -> int:
        return self.counts.get((normalize_token(left), normalize_token(right)), 0)

    def edge_score(self, left: str, right: str) -> float:
        """
        Smoothed directional association / conditional-likelihood proxy.

        Higher is better. The score blends:
          * P(right | left)-like evidence, and
          * a symmetric association term that resists generic filler words.
        """
        l = normalize_token(left)
        r = normalize_token(right)
        bc = self.counts.get((l, r), 0)
        ul = max(self.unigrams.count(l), 1)
        ur = max(self.unigrams.count(r), 1)

        # A small pseudo-count prevents unseen pairs from becoming -inf.
        cond = math.log10(bc + 0.25) - math.log10(ul + 1.0)
        assoc = math.log10(bc + 0.25) - 0.5 * (
            math.log10(ul + 1.0) + math.log10(ur + 1.0)
        )
        return 0.72 * cond + 0.28 * assoc


def _parse_count_line(line: str) -> tuple[str, int] | None:
    line = line.strip()
    if not line:
        return None
    if "\t" in line:
        text, raw_count = line.rsplit("\t", 1)
    else:
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            return None
        text, raw_count = parts
    try:
        return text, int(raw_count)
    except ValueError:
        return None


def load_unigram_model(path: Path) -> UnigramModel:
    table: dict[str, int] = defaultdict(int)
    total = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_count_line(line)
            if parsed is None:
                continue
            token, c = parsed
            word = normalize_token(token)
            if not word:
                continue
            table[word] += c
            total += c
    return UnigramModel(dict(table), total)


def load_bigram_model(path: Path, unigrams: UnigramModel, vocabulary: set[str]) -> BigramModel:
    table: dict[tuple[str, str], int] = defaultdict(int)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = _parse_count_line(line)
            if parsed is None:
                continue
            phrase, c = parsed
            tokens = phrase.split()
            if len(tokens) != 2:
                continue
            left = normalize_token(tokens[0])
            right = normalize_token(tokens[1])
            if not left or not right:
                continue
            # Keep only pairs we could possibly need for this puzzle.
            if left not in vocabulary or right not in vocabulary:
                continue
            table[(left, right)] += c
    return BigramModel(unigrams, dict(table))


def ensure_ngram_data(
    ngram_dir: Path,
    refresh: bool,
    need_bigrams: bool,
) -> tuple[Path, Path | None]:
    one = download_file(NORVIG_1W_URL, ngram_dir / "count_1w.txt", refresh=refresh)
    two = None
    if need_bigrams:
        two = download_file(NORVIG_2W_URL, ngram_dir / "count_2w.txt", refresh=refresh)
    return one, two


@dataclass(slots=True, frozen=True)
class Candidate:
    word: str
    sig: tuple[int, ...]
    length: int
    zipf: float


def load_words(
    path: Path,
    target_counts: tuple[int, ...],
    min_len: int,
    max_len: int,
    excluded_words: set[str],
    exclude_regexes: list[re.Pattern[str]],
    forbid_chars: set[str],
    min_zipf: float,
    short_policy: str,
    short_whitelist: set[str],
    forced_words: set[str],
    unigrams: UnigramModel | None,
) -> list[Candidate]:
    seen: set[str] = set()
    candidates: list[Candidate] = []

    def allowed_by_length(word: str, is_forced: bool) -> bool:
        if is_forced:
            return True
        if word in short_whitelist:
            # Explicit/common short whitelist may bypass minimum length.
            return len(word) <= max_len
        if not (min_len <= len(word) <= max_len):
            return False
        if len(word) <= 2:
            if short_policy == "none":
                return False
            if short_policy == "common" and word not in short_whitelist:
                return False
        return True

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            word = normalize_token(line.strip())
            if not word or word in seen:
                continue
            seen.add(word)

            is_forced = word in forced_words
            if not allowed_by_length(word, is_forced):
                continue
            if word in excluded_words:
                continue
            if forbid_chars and any(ch in forbid_chars for ch in word):
                continue
            if any(rx.search(word) for rx in exclude_regexes):
                continue

            sig = counts(word)
            if not fits(sig, target_counts):
                continue

            z = unigrams.zipf(word) if unigrams is not None else 0.0
            if min_zipf > 0 and z < min_zipf and not is_forced:
                continue
            candidates.append(Candidate(word, sig, len(word), z))

    # Literal clue words are injected even if absent from the dictionary.
    present = {c.word for c in candidates}
    for word in sorted(forced_words):
        if not word or word in present or word in excluded_words:
            continue
        if forbid_chars and any(ch in forbid_chars for ch in word):
            continue
        if any(rx.search(word) for rx in exclude_regexes):
            continue
        sig = counts(word)
        if not fits(sig, target_counts):
            continue
        z = unigrams.zipf(word) if unigrams is not None else 0.0
        candidates.append(Candidate(word, sig, len(word), z))
        present.add(word)

    # Common words first helps DFS produce sane partial states early.
    candidates.sort(key=lambda c: (-c.zipf, -c.length, c.word))
    return candidates


def solve(
    remaining: tuple[int, ...],
    candidates: list[Candidate],
    min_words: int,
    max_words: int,
    max_results: int,
    allow_repeat: bool,
) -> Iterator[tuple[str, ...]]:
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for i, c in enumerate(candidates):
        by_signature[c.sig].append(i)

    if not candidates:
        return

    results_found = 0
    dead: set[tuple[tuple[int, ...], int, int]] = set()
    min_candidate_len = min(c.length for c in candidates)
    max_candidate_len = max(c.length for c in candidates)
    sparse_signatures = [
        tuple((letter, amount) for letter, amount in enumerate(c.sig) if amount)
        for c in candidates
    ]

    def dfs(
        rem: tuple[int, ...],
        rem_len: int,
        start: int,
        words_left: int,
        chosen: list[str],
    ) -> Iterator[tuple[str, ...]]:
        nonlocal results_found

        if max_results > 0 and results_found >= max_results:
            return

        if words_left == 0:
            if rem_len == 0:
                results_found += 1
                yield tuple(chosen)
            return
        if rem_len == 0:
            return
        if rem_len < words_left * min_candidate_len:
            return
        if rem_len > words_left * max_candidate_len:
            return

        state = (rem, start, words_left)
        if state in dead:
            return

        before = results_found

        if words_left == 1:
            for i in by_signature.get(rem, []):
                if i < start:
                    continue
                word = candidates[i].word
                results_found += 1
                yield tuple(chosen + [word])
                if max_results > 0 and results_found >= max_results:
                    return
            if results_found == before:
                dead.add(state)
            return

        min_this_len = max(
            min_candidate_len,
            rem_len - (words_left - 1) * max_candidate_len,
        )
        max_this_len = min(
            max_candidate_len,
            rem_len - (words_left - 1) * min_candidate_len,
        )

        for i in range(start, len(candidates)):
            c = candidates[i]
            if c.length < min_this_len or c.length > max_this_len:
                continue
            sparse = sparse_signatures[i]
            candidate_fits = True
            for letter, amount in sparse:
                if rem[letter] < amount:
                    candidate_fits = False
                    break
            if not candidate_fits:
                continue
            mutable_rem = list(rem)
            for letter, amount in sparse:
                mutable_rem[letter] -= amount
            new_rem = tuple(mutable_rem)
            next_start = i if allow_repeat else i + 1
            yield from dfs(
                new_rem,
                rem_len - c.length,
                next_start,
                words_left - 1,
                chosen + [c.word],
            )
            if max_results > 0 and results_found >= max_results:
                return

        if results_found == before:
            dead.add(state)

    initial_remaining_len = sum(remaining)
    for nwords in range(min_words, max_words + 1):
        yield from dfs(remaining, initial_remaining_len, 0, nwords, [])
        if max_results > 0 and results_found >= max_results:
            break


def morph_root(word: str, vocabulary: set[str], unigrams: UnigramModel | None) -> str:
    word = normalize_token(word)
    if not word:
        return word
    if word in IRREGULAR_ROOTS:
        return IRREGULAR_ROOTS[word]
    if len(word) <= 3 or word in S_STEM_EXCEPTIONS:
        return word

    variants: list[str] = []

    if word.endswith("ies") and len(word) > 4:
        variants.extend([word[:-1], word[:-3] + "y"])
    if word.endswith(("ches", "shes", "xes", "zes", "oes")) and len(word) > 4:
        variants.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss"):
        variants.append(word[:-1])

    if word.endswith("ied") and len(word) > 4:
        variants.extend([word[:-1], word[:-3] + "y"])
    elif word.endswith("ed") and len(word) > 4:
        variants.extend([word[:-1], word[:-2]])

    if word.endswith("ing") and len(word) > 5:
        variants.extend([word[:-3], word[:-3] + "e"])

    plausible = [v for v in variants if v and v in vocabulary]
    if not plausible:
        return word

    def evidence(v: str) -> tuple[int, int]:
        c = unigrams.count(v) if unigrams is not None else 0
        return (c, -len(v))

    return max(plausible, key=evidence)


def family_key(words: Sequence[str], vocabulary: set[str], unigrams: UnigramModel | None) -> tuple[str, ...]:
    return tuple(sorted(morph_root(w, vocabulary, unigrams) for w in words))


def lexical_details(words: Sequence[str], unigrams: UnigramModel | None, short_whitelist: set[str]) -> dict[str, float]:
    if unigrams is None:
        zs = [0.0 for _ in words]
    else:
        zs = [unigrams.zipf(w) for w in words]

    if not zs:
        return {"lex_raw": 0.0, "avg_zipf": 0.0, "min_zipf": 0.0, "junk_penalty": 0.0}

    avg_z = sum(zs) / len(zs)
    low_count = min(2, len(zs))
    low_tail = sum(sorted(zs)[:low_count]) / low_count

    # Soft lexical evidence. No quadratic "one uncommon word kills the phrase"
    # behavior. The correct inflection should not lose just because its plural
    # is less frequent.
    raw = 0.78 * avg_z + 0.22 * low_tail

    junk_penalty = 0.0
    for w in words:
        if len(w) <= 2 and w not in short_whitelist:
            junk_penalty += 0.75
        if unigrams is not None and unigrams.count(w) == 0:
            junk_penalty += 0.35

    duplicates = len(words) - len(set(words))
    junk_penalty += 1.25 * duplicates
    raw -= junk_penalty

    return {
        "lex_raw": raw,
        "avg_zipf": avg_z,
        "min_zipf": min(zs),
        "junk_penalty": junk_penalty,
    }


def bigram_pair_potential(words: Sequence[str], model: BigramModel) -> tuple[float, float]:
    """
    Order-independent "can these words plausibly connect?" score.

    We take the best n-1 directed pair edges. It is an optimistic upper-bound-
    like proxy, not a valid sentence probability. Cheap enough for every result.
    """
    n = len(words)
    if n <= 1:
        return 0.0, 0.0

    edges: list[tuple[float, bool]] = []
    for i, left in enumerate(words):
        for j, right in enumerate(words):
            if i == j:
                continue
            seen = model.bigram_count(left, right) > 0
            edges.append((model.edge_score(left, right), seen))

    edges.sort(key=lambda x: x[0], reverse=True)
    chosen = edges[: n - 1]
    if not chosen:
        return -99.0, 0.0
    avg = sum(x[0] for x in chosen) / len(chosen)
    coverage = sum(1 for _, seen in chosen if seen) / len(chosen)
    return avg, coverage


def best_word_order(words: Sequence[str], model: BigramModel) -> tuple[float, tuple[str, ...], float]:
    """
    Exact Held-Karp-style dynamic programming for the best directed word path.

    Complexity is O(n^2 2^n), tiny for n <= 6 compared with enumerating all
    n! permutations for every candidate.
    """
    words = tuple(words)
    n = len(words)
    if n <= 1:
        return 0.0, words, 0.0

    edge = [[0.0] * n for _ in range(n)]
    seen = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge[i][j] = model.edge_score(words[i], words[j])
            seen[i][j] = model.bigram_count(words[i], words[j]) > 0

    # (mask, last) -> (score_sum, seen_edges, path_tuple_indices)
    dp: dict[tuple[int, int], tuple[float, int, tuple[int, ...]]] = {}
    for i in range(n):
        dp[(1 << i, i)] = (0.0, 0, (i,))

    full = (1 << n) - 1
    for mask in range(1, full + 1):
        for last in range(n):
            state = dp.get((mask, last))
            if state is None:
                continue
            score_sum, seen_edges, path = state
            for nxt in range(n):
                bit = 1 << nxt
                if mask & bit:
                    continue
                new_mask = mask | bit
                candidate = (
                    score_sum + edge[last][nxt],
                    seen_edges + int(seen[last][nxt]),
                    path + (nxt,),
                )
                old = dp.get((new_mask, nxt))
                # Primary: language score. Secondary: number of observed edges.
                if old is None or (candidate[0], candidate[1]) > (old[0], old[1]):
                    dp[(new_mask, nxt)] = candidate

    best = max(
        (dp[(full, last)] for last in range(n) if (full, last) in dp),
        key=lambda x: (x[0], x[1]),
    )
    score_sum, seen_edges, path = best
    avg = score_sum / (n - 1)
    coverage = seen_edges / (n - 1)
    return avg, tuple(words[i] for i in path), coverage


def percentile_map(values: Sequence[float]) -> list[float]:
    """
    Tie-aware percentile ranks in [0, 1].
    """
    n = len(values)
    if n <= 1:
        return [1.0] * n

    order = sorted(range(n), key=lambda i: values[i])
    out = [0.0] * n
    pos = 0
    while pos < n:
        end = pos + 1
        v = values[order[pos]]
        while end < n and values[order[end]] == v:
            end += 1
        mid_rank = (pos + end - 1) / 2
        pct = mid_rank / (n - 1)
        for k in range(pos, end):
            out[order[k]] = pct
        pos = end
    return out


@dataclass(slots=True)
class Record:
    words: tuple[str, ...]
    word_count: int
    matched_hints: tuple[str, ...]
    lex_raw: float
    avg_zipf: float
    min_zipf: float
    junk_penalty: float
    family: tuple[str, ...]
    family_best_lex: float = 0.0
    family_size: int = 1
    hint_info: float = 0.0
    pair_raw: float = 0.0
    pair_coverage: float = 0.0
    lex_pct: float = 0.0
    family_pct: float = 0.0
    pair_pct: float = 0.0
    pre_score: float = 0.0
    deep: bool = False
    order_raw: float = 0.0
    order_coverage: float = 0.0
    best_order: tuple[str, ...] = ()
    order_pct: float = 0.0
    final_score: float = 0.0


def build_records(
    solutions: Sequence[tuple[str, ...]],
    required_words: Sequence[str],
    contains_any: set[str],
    hint_mode: str,
    unigrams: UnigramModel | None,
    bigrams: BigramModel | None,
    vocabulary: set[str],
    short_whitelist: set[str],
) -> list[Record]:
    records: list[Record] = []

    for solution in solutions:
        words = (*required_words, *solution)
        matched = tuple(sorted(contains_any.intersection(words)))

        if contains_any:
            if hint_mode == "any" and not matched:
                continue
            if hint_mode == "exactly-one" and len(matched) != 1:
                continue

        lex = lexical_details(words, unigrams, short_whitelist)
        fam = family_key(words, vocabulary, unigrams)

        if bigrams is not None:
            pair_raw, pair_cov = bigram_pair_potential(words, bigrams)
        else:
            pair_raw, pair_cov = 0.0, 0.0

        records.append(
            Record(
                words=words,
                word_count=len(words),
                matched_hints=matched,
                lex_raw=lex["lex_raw"],
                avg_zipf=lex["avg_zipf"],
                min_zipf=lex["min_zipf"],
                junk_penalty=lex["junk_penalty"],
                family=fam,
                pair_raw=pair_raw,
                pair_coverage=pair_cov,
            )
        )

    # Morphology-family inheritance.
    family_members: dict[tuple[str, ...], list[Record]] = defaultdict(list)
    for r in records:
        family_members[r.family].append(r)
    for members in family_members.values():
        best = max(r.lex_raw for r in members)
        size = len(members)
        for r in members:
            r.family_best_lex = best
            r.family_size = size

    # Hint informativeness is derived from how much each hint actually narrows
    # the candidate set, not hard-coded words.
    if contains_any and records:
        hint_counts = Counter()
        for r in records:
            hint_counts.update(set(r.matched_hints))
        total = len(records)
        raw_info = {
            hint: math.log((total + 1.0) / (hint_counts.get(hint, 0) + 1.0))
            for hint in contains_any
        }
        max_info = max(raw_info.values(), default=1.0) or 1.0
        for r in records:
            if r.matched_hints:
                best = max(raw_info[h] for h in r.matched_hints)
                # Small multi-hint bonus, capped.
                multi = min(0.15, 0.05 * (len(r.matched_hints) - 1))
                r.hint_info = min(1.0, best / max_info + multi)

    # Percentiles and pre-score are computed separately for each word count.
    by_wc: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        by_wc[r.word_count].append(r)

    for bucket in by_wc.values():
        lex_pcts = percentile_map([r.lex_raw for r in bucket])
        fam_pcts = percentile_map([r.family_best_lex for r in bucket])
        pair_pcts = percentile_map([r.pair_raw for r in bucket])

        for r, lp, fp, pp in zip(bucket, lex_pcts, fam_pcts, pair_pcts):
            r.lex_pct = lp
            r.family_pct = fp
            r.pair_pct = pp

            # Pair compatibility is the largest pre-analysis signal.
            # Lexical frequency is intentionally soft.
            r.pre_score = 100.0 * (
                0.26 * lp
                + 0.12 * fp
                + 0.46 * pp
                + 0.16 * r.hint_info
            )

    return records


def deep_analyze(
    records: Sequence[Record],
    bigrams: BigramModel,
    deep_per_group: int,
    deep_all: bool,
) -> None:
    by_wc: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        by_wc[r.word_count].append(r)

    for wc, bucket in by_wc.items():
        ordered = sorted(bucket, key=lambda r: (r.pre_score, r.lex_raw), reverse=True)
        chosen = ordered if deep_all else ordered[:deep_per_group]

        print(
            f"Deep phrase-order analysis: {len(chosen)} / {len(bucket)} "
            f"candidate(s) in {wc}-word bucket",
            file=sys.stderr,
        )

        for r in chosen:
            order_raw, best_order, coverage = best_word_order(r.words, bigrams)
            r.deep = True
            r.order_raw = order_raw
            r.best_order = best_order
            r.order_coverage = coverage

        if not chosen:
            continue

        pcts = percentile_map([r.order_raw for r in chosen])
        for r, op in zip(chosen, pcts):
            r.order_pct = op
            # Exact phrase order dominates once we have computed it.
            coverage_bonus = 5.0 * r.order_coverage
            r.final_score = min(
                100.0,
                0.36 * r.pre_score + 64.0 * op + coverage_bonus,
            )


def pretty_word(word: str) -> str:
    return PRETTY_CONTRACTIONS.get(word, word)


def pretty_phrase(words: Sequence[str]) -> str:
    return " ".join(pretty_word(w) for w in words)


def canonical_phrase(words: Sequence[str]) -> str:
    return " ".join(words)


def format_pre_record(rank: int, r: Record, show_components: bool) -> str:
    phrase = canonical_phrase(r.words)
    hints = ",".join(r.matched_hints) if r.matched_hints else "-"
    if show_components:
        return (
            f"{rank:7d}. PRE={r.pre_score:6.2f} "
            f"LEX={r.lex_pct:5.3f} FAM={r.family_pct:5.3f} "
            f"PAIR={r.pair_pct:5.3f} HINT={r.hint_info:5.3f} "
            f"ZAVG={r.avg_zipf:4.2f} ZMIN={r.min_zipf:4.2f} "
            f"PCOV={r.pair_coverage:4.2f}  {phrase}  [HINT={hints}]"
        )
    return f"{rank:7d}. PRE={r.pre_score:6.2f}  {phrase}"


def format_deep_record(rank: int, r: Record, show_components: bool) -> str:
    best = r.best_order if r.best_order else r.words
    phrase = pretty_phrase(best)
    hints = ",".join(r.matched_hints) if r.matched_hints else "-"
    if show_components:
        return (
            f"{rank:6d}. FINAL={r.final_score:6.2f} PRE={r.pre_score:6.2f} "
            f"ORDER={r.order_pct:5.3f} OCOV={r.order_coverage:4.2f} "
            f"LEX={r.lex_pct:5.3f} PAIR={r.pair_pct:5.3f} "
            f"HINT={r.hint_info:5.3f}  {phrase}  "
            f"[CANON={' '.join(r.words)}; HINT={hints}]"
        )
    return f"{rank:6d}. FINAL={r.final_score:6.2f}  {phrase}"


def write_full_export(path: Path, records: Sequence[Record], show_components: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_wc: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        by_wc[r.word_count].append(r)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for wc in sorted(by_wc):
            bucket = sorted(
                by_wc[wc],
                key=lambda r: (r.pre_score, r.lex_raw, canonical_phrase(r.words)),
                reverse=True,
            )
            f.write(f"=== {wc}-WORD SOLUTIONS (ALL {len(bucket)}; PRE-RANKED) ===\n")
            for rank, r in enumerate(bucket, 1):
                f.write(format_pre_record(rank, r, show_components) + "\n")
            f.write("\n")


def write_deep_export(path: Path, records: Sequence[Record], show_components: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_wc: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        if r.deep:
            by_wc[r.word_count].append(r)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for wc in sorted(by_wc):
            bucket = sorted(
                by_wc[wc],
                key=lambda r: (r.final_score, r.pre_score),
                reverse=True,
            )
            f.write(f"=== {wc}-WORD DEEP-RANKED SOLUTIONS ({len(bucket)}) ===\n")
            for rank, r in enumerate(bucket, 1):
                f.write(format_deep_record(rank, r, show_components) + "\n")
            f.write("\n")


def answer_signature(text: str) -> tuple[str, ...]:
    return tuple(sorted(tokenize_words(text)))


def report_benchmark(answer: str, records: Sequence[Record]) -> None:
    sig = answer_signature(answer)
    matches = [r for r in records if tuple(sorted(r.words)) == sig]

    print(file=sys.stderr)
    print(f"BENCHMARK ANSWER: {answer}", file=sys.stderr)
    if not matches:
        print("  NOT FOUND in generated candidates.", file=sys.stderr)
        return

    # There should normally be one canonical record.
    target = matches[0]
    same_wc = [r for r in records if r.word_count == target.word_count]

    pre_sorted = sorted(same_wc, key=lambda r: (r.pre_score, r.lex_raw), reverse=True)
    pre_rank = pre_sorted.index(target) + 1
    print(
        f"  PRE rank:   {pre_rank} / {len(pre_sorted)} "
        f"(score {target.pre_score:.2f})",
        file=sys.stderr,
    )
    print(
        f"  canonical:  {' '.join(target.words)}",
        file=sys.stderr,
    )
    print(
        f"  family:     {' / '.join(target.family)} "
        f"(family size {target.family_size})",
        file=sys.stderr,
    )
    print(
        f"  hint info:  {target.hint_info:.3f} "
        f"({', '.join(target.matched_hints) or 'none'})",
        file=sys.stderr,
    )
    if target.deep:
        deep_bucket = [r for r in same_wc if r.deep]
        deep_sorted = sorted(
            deep_bucket,
            key=lambda r: (r.final_score, r.pre_score),
            reverse=True,
        )
        deep_rank = deep_sorted.index(target) + 1
        print(
            f"  FINAL rank: {deep_rank} / {len(deep_sorted)} "
            f"(score {target.final_score:.2f})",
            file=sys.stderr,
        )
        print(
            f"  best order: {pretty_phrase(target.best_order)}",
            file=sys.stderr,
        )
        print(
            f"  order seen-edge coverage: {target.order_coverage:.2f}",
            file=sys.stderr,
        )
    else:
        print(
            "  FINAL rank: not deep-analyzed; raise --deep-per-group or use --deep-all",
            file=sys.stderr,
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Exact multi-word anagram solver with non-LLM phrase analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("text", help="Text whose letters should be anagrammed")
    p.add_argument("--dict", dest="dictionary", default=DEFAULT_DICT_URL, help="Dictionary path or URL")
    p.add_argument("--refresh", action="store_true", help="Refresh cached dictionary/ngram downloads")

    p.add_argument("--min-word-len", type=int, default=2)
    p.add_argument("--max-word-len", type=int, default=99)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=int, default=6)
    p.add_argument(
        "--max-results", type=int, default=1000, metavar="N",
        help="Stop after N unique generated word sets; 0 means unlimited",
    )
    p.add_argument(
        "--all-results", action="store_true",
        help="Exhaustive generation; shorthand for --max-results 0",
    )

    p.add_argument("--exclude", action="append", default=[], metavar="WORD[,WORD...]")
    p.add_argument("--exclude-file", action="append", default=[], metavar="FILE")
    p.add_argument("--exclude-regex", action="append", default=[], metavar="REGEX")
    p.add_argument("--forbid-chars", default="", metavar="LETTERS")
    p.add_argument("--subtract", action="append", default=[], metavar="TEXT")
    p.add_argument("--require", action="append", default=[], metavar="TEXT")

    p.add_argument(
        "--contains-any", action="append", default=[], metavar="WORD[,WORD...]",
        help="Require at least one of these clue words; clues bypass lexical filtering",
    )
    p.add_argument(
        "--hint-mode", choices=("any", "exactly-one"), default="any",
        help="Interpret --contains-any as at-least-one or exactly-one matched clue word",
    )
    p.add_argument("--hint-stats", action="store_true")

    p.add_argument(
        "--min-zipf", type=float, default=0.0, metavar="N",
        help=(
            "Minimum corpus-derived Zipf-style unigram frequency. "
            "Try 3.0 for broad normal-English search; 0 disables filtering."
        ),
    )

    p.add_argument(
        "--short-word-policy",
        choices=("common", "all", "none"),
        default="common",
        help=(
            "Treatment of 1-2 letter dictionary entries. 'common' keeps a sane "
            "English whitelist; 'all' is truly raw; 'none' rejects them except "
            "explicit clue/extra-short words."
        ),
    )
    p.add_argument(
        "--extra-short-words", action="append", default=[], metavar="WORD[,WORD...]",
        help="Additional 1-2 letter words allowed by the safe short-word policy",
    )
    p.add_argument("--no-repeat", action="store_true")

    p.add_argument(
        "--analyze", action="store_true",
        help="Enable bigram/morphology/hint-aware ranking and best-order analysis",
    )
    p.add_argument(
        "--ngram-dir", default=str(DEFAULT_NGRAM_DIR), metavar="DIR",
        help="Cache/location for Norvig count_1w.txt and count_2w.txt",
    )
    p.add_argument(
        "--deep-per-group", type=int, default=10000, metavar="N",
        help="Exact best-order analysis for the top N PRE-ranked candidates per word-count bucket",
    )
    p.add_argument(
        "--deep-all", action="store_true",
        help="Exact best-order analysis for every generated candidate; can be expensive",
    )

    p.add_argument(
        "--top-per-group", type=int, default=100, metavar="N",
        help="Number of FINAL/PRE results shown on the console per word-count bucket",
    )
    p.add_argument(
        "--show-components", action="store_true",
        help="Show component diagnostics for each ranked result",
    )
    p.add_argument(
        "--export", metavar="FILE",
        help="Write ALL generated solutions, PRE-ranked, to this file",
    )
    p.add_argument(
        "--deep-export", metavar="FILE",
        help="Write the deeply analyzed shortlist/full deep set to this file",
    )
    p.add_argument(
        "--stream-export", metavar="FILE",
        help="Stream every accepted canonical solution to disk while generating it",
    )
    p.add_argument(
        "--benchmark-answer", metavar="PHRASE",
        help="Known answer used ONLY to report ranks and validate the analyzer; never affects scoring",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.min_word_len < 1:
        raise SystemExit("--min-word-len must be >= 1")
    if args.max_word_len < args.min_word_len:
        raise SystemExit("--max-word-len must be >= --min-word-len")
    if args.min_words < 1 or args.max_words < args.min_words:
        raise SystemExit("Invalid --min-words/--max-words")
    if args.max_results < 0:
        raise SystemExit("--max-results must be >= 0")
    if args.all_results:
        args.max_results = 0
    if args.deep_per_group < 1:
        raise SystemExit("--deep-per-group must be >= 1")
    if args.top_per_group < 1:
        raise SystemExit("--top-per-group must be >= 1")
    if args.min_zipf < 0:
        raise SystemExit("--min-zipf must be >= 0")

    target = counts(args.text)
    if sum(target) == 0:
        raise SystemExit("Target contains no A-Z letters.")

    required_chunks = split_values(args.require)
    required_words: list[str] = []
    for chunk in required_chunks:
        required_words.extend(tokenize_words(chunk))

    remaining = target
    operations = (
        [("--require", x) for x in required_chunks]
        + [("--subtract", x) for x in split_values(args.subtract)]
    )
    for label, chunk in operations:
        new_remaining = subtract_counts(remaining, counts(chunk))
        if new_remaining is None:
            raise SystemExit(
                f"{label} value {chunk!r} uses unavailable/already-subtracted letters."
            )
        remaining = new_remaining

    excluded_words = {
        normalize_token(x)
        for x in split_values(args.exclude)
        if normalize_token(x)
    }
    for filename in args.exclude_file:
        path = Path(filename).expanduser()
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = normalize_token(line)
                if w:
                    excluded_words.add(w)

    try:
        exclude_regexes = [re.compile(x, re.IGNORECASE) for x in args.exclude_regex]
    except re.error as exc:
        raise SystemExit(f"Bad --exclude-regex: {exc}")

    forbid_chars = set(normalize_letters(args.forbid_chars))

    contains_any = {
        normalize_token(x)
        for x in split_values(args.contains_any)
        if normalize_token(x)
    }
    contains_any.difference_update(excluded_words)

    impossible_hints = sorted(
        h for h in contains_any if not fits(counts(h), remaining)
    )
    if impossible_hints:
        print(
            "Ignoring impossible clue word(s): " + ", ".join(impossible_hints),
            file=sys.stderr,
        )
        contains_any.difference_update(impossible_hints)

    extra_short = {
        normalize_token(x)
        for x in split_values(args.extra_short_words)
        if normalize_token(x)
    }
    short_whitelist = set(DEFAULT_SHORT_WORDS)
    short_whitelist.update(extra_short)
    short_whitelist.update(h for h in contains_any if len(h) <= 2)
    short_whitelist.update(w for w in required_words if len(w) <= 2)
    short_whitelist.difference_update(excluded_words)

    if sum(remaining) == 0:
        print(" ".join(required_words))
        return 0

    # We need unigram data for --min-zipf and for meaningful lexical analysis.
    need_unigrams = args.min_zipf > 0 or args.analyze
    unigrams: UnigramModel | None = None
    bigrams_path: Path | None = None

    if need_unigrams:
        ngram_dir = Path(args.ngram_dir).expanduser()
        one_path, bigrams_path = ensure_ngram_data(
            ngram_dir,
            refresh=args.refresh,
            need_bigrams=args.analyze,
        )
        print("Loading unigram frequency data...", file=sys.stderr)
        unigrams = load_unigram_model(one_path)

    dictionary_path = get_dictionary(args.dictionary, refresh=args.refresh)

    candidates = load_words(
        dictionary_path,
        remaining,
        args.min_word_len,
        min(args.max_word_len, sum(remaining)),
        excluded_words,
        exclude_regexes,
        forbid_chars,
        args.min_zipf,
        args.short_word_policy,
        short_whitelist,
        contains_any,
        unigrams,
    )

    if not candidates:
        print("No candidate dictionary words fit the remaining letters.", file=sys.stderr)
        return 1

    vocabulary = {c.word for c in candidates}
    vocabulary.update(required_words)
    vocabulary.update(contains_any)

    if args.benchmark_answer:
        benchmark_words = tokenize_words(args.benchmark_answer)
        missing_benchmark = [w for w in benchmark_words if w not in vocabulary]
        if missing_benchmark:
            print(
                "Benchmark warning: these answer words are absent from the "
                "candidate vocabulary:",
                file=sys.stderr,
            )
            for w in missing_benchmark:
                z = unigrams.zipf(w) if unigrams is not None else 0.0
                reason = []
                if w in excluded_words:
                    reason.append("excluded")
                if len(w) < args.min_word_len and w not in short_whitelist:
                    reason.append("below min length")
                if (
                    len(w) <= 2
                    and args.short_word_policy == "common"
                    and w not in short_whitelist
                ):
                    reason.append("short-word policy")
                if args.min_zipf > 0 and z < args.min_zipf:
                    reason.append(f"below min Zipf-style {args.min_zipf}")
                why = ", ".join(reason) if reason else "dictionary/filter mismatch"
                print(
                    f"  {w:<16} zipf-style={z:4.2f}  reason={why}",
                    file=sys.stderr,
                )

    bigrams: BigramModel | None = None
    if args.analyze:
        assert unigrams is not None and bigrams_path is not None
        print("Loading relevant word bigrams...", file=sys.stderr)
        bigrams = load_bigram_model(bigrams_path, unigrams, vocabulary)
        print(
            f"Loaded {len(bigrams.counts):,} relevant directed bigram(s).",
            file=sys.stderr,
        )

    remaining_letters = "".join(
        chr(A_ORD + i) * n for i, n in enumerate(remaining)
    )
    print(
        f"Target letters:      {normalize_letters(args.text)}\n"
        f"Remaining letters:   {remaining_letters}\n"
        f"Candidate words:     {len(candidates)}\n"
        f"Min Zipf-style:      {args.min_zipf if args.min_zipf > 0 else 'disabled'}\n"
        f"Short-word policy:   {args.short_word_policy}\n"
        f"Contains-any:        {', '.join(sorted(contains_any)) if contains_any else 'disabled'}\n"
        f"Hint mode:           {args.hint_mode if contains_any else 'n/a'}",
        file=sys.stderr,
    )

    stream = None
    if args.stream_export:
        stream_path = Path(args.stream_export).expanduser()
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream = stream_path.open("w", encoding="utf-8", newline="\n")

    solutions: list[tuple[str, ...]] = []
    generated = 0
    accepted = 0
    search_started = time.perf_counter()
    try:
        for solution in solve(
            remaining,
            candidates,
            args.min_words,
            args.max_words,
            args.max_results,
            allow_repeat=not args.no_repeat,
        ):
            generated += 1
            all_words = (*required_words, *solution)
            matched = contains_any.intersection(all_words)

            if contains_any:
                if args.hint_mode == "any" and not matched:
                    continue
                if args.hint_mode == "exactly-one" and len(matched) != 1:
                    continue

            accepted += 1
            solutions.append(solution)
            if stream is not None:
                stream.write(" ".join(all_words) + "\n")
    finally:
        if stream is not None:
            stream.close()

    search_seconds = time.perf_counter() - search_started
    print(
        f"Generated {generated:,} exact word set(s); "
        f"{accepted:,} survived clue constraints. "
        f"Exact search: {search_seconds:.2f}s.",
        file=sys.stderr,
    )

    if not solutions:
        return 1

    records = build_records(
        solutions=solutions,
        required_words=required_words,
        contains_any=contains_any,
        hint_mode=args.hint_mode,
        unigrams=unigrams,
        bigrams=bigrams,
        vocabulary=vocabulary,
        short_whitelist=short_whitelist,
    )

    if contains_any and args.hint_stats:
        stats = Counter()
        for r in records:
            stats.update(set(r.matched_hints))
        print("Hint match counts:", file=sys.stderr)
        for h in sorted(contains_any):
            print(f"  {h:<16} {stats[h]:,}", file=sys.stderr)

    if args.analyze:
        assert bigrams is not None
        deep_analyze(
            records,
            bigrams=bigrams,
            deep_per_group=args.deep_per_group,
            deep_all=args.deep_all,
        )

    # Console output grouped by word count.
    by_wc: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        by_wc[r.word_count].append(r)

    for wc in sorted(by_wc):
        bucket = by_wc[wc]
        deep_bucket = [r for r in bucket if r.deep]

        if deep_bucket:
            ranked = sorted(
                deep_bucket,
                key=lambda r: (r.final_score, r.pre_score),
                reverse=True,
            )
            shown = min(args.top_per_group, len(ranked))
            print(
                f"=== {wc}-WORD FINAL RANKING "
                f"(showing {shown} of {len(ranked)} deep-analyzed; "
                f"{len(bucket)} total) ==="
            )
            for rank, r in enumerate(ranked[:shown], 1):
                print(format_deep_record(rank, r, args.show_components))
        else:
            ranked = sorted(
                bucket,
                key=lambda r: (r.pre_score, r.lex_raw),
                reverse=True,
            )
            shown = min(args.top_per_group, len(ranked))
            print(
                f"=== {wc}-WORD PRE-RANKING "
                f"(showing {shown} of {len(ranked)}) ==="
            )
            for rank, r in enumerate(ranked[:shown], 1):
                print(format_pre_record(rank, r, args.show_components))
        print()

    if args.export:
        export_path = Path(args.export).expanduser()
        write_full_export(export_path, records, args.show_components)
        print(
            f"Full exhaustive PRE-ranked export: {export_path} "
            f"({len(records):,} records)",
            file=sys.stderr,
        )

    if args.deep_export:
        deep_path = Path(args.deep_export).expanduser()
        write_deep_export(deep_path, records, args.show_components)
        deep_count = sum(1 for r in records if r.deep)
        print(
            f"Deep-ranked export: {deep_path} ({deep_count:,} records)",
            file=sys.stderr,
        )

    if args.benchmark_answer:
        report_benchmark(args.benchmark_answer, records)

    if args.max_results == 0:
        print(
            f"Exhaustive search completed: {len(records):,} accepted result set(s).",
            file=sys.stderr,
        )
    elif generated >= args.max_results:
        print(
            f"Search stopped at --max-results {args.max_results:,}.",
            file=sys.stderr,
        )
    else:
        print(
            f"Search exhausted naturally: {len(records):,} accepted result set(s).",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
