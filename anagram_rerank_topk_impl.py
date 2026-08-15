#!/usr/bin/env python3
"""
Top-k word-order search and phrase-evidence reranking layer.

For each unordered word bag this layer retains several strong grammatical
orders, then allows positive phrase/collocation evidence to choose among them.
Input bags are canonicalized before search so tie-breaking is deterministic and
independent of generator emission order.
"""

from __future__ import annotations

import math
import multiprocessing
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import anagram_rerank_core as core

# Re-export the existing public and benchmark-facing surface, including the
# underscore-prefixed ordering helpers used by anagram_benchmark.py.
for _name in dir(core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(core, _name)

ENGINE_LAYER = "top-k-order-reranking"
DEFAULT_ORDER_CANDIDATES = 16
_ORDER_CANDIDATE_COUNT = DEFAULT_ORDER_CANDIDATES

# Main-process side table. Row uses slots, so keeping the alternatives outside
# Row lets us add the feature without invalidating prepared-cache pickles.
_ORDER_CANDIDATES_BY_ROW_ID: dict[int, tuple["OrderCandidate", ...]] = {}

# Worker-local settings.
_WORKER_LEX = None
_WORKER_ORDER_MODE = "auto"
_WORKER_BEAM_WIDTH = 128
_WORKER_EXACT_MAX_WORDS = 5
_WORKER_ORDER_CANDIDATES = DEFAULT_ORDER_CANDIDATES


@dataclass(slots=True, frozen=True)
class OrderCandidate:
    order: tuple[str, ...]
    grammar_raw: float
    grammar_norm: float
    structure_norm: float
    valency_norm: float
    syntax_coverage: float
    phrase_kind: str
    objective: float


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
    order_candidates: tuple[OrderCandidate, ...] = ()


def _candidate_sort_key(candidate: OrderCandidate) -> tuple:
    """Best-first key with lexical order as the final deterministic tie break."""
    return (
        -candidate.objective,
        -candidate.structure_norm,
        -candidate.grammar_norm,
        -candidate.valency_norm,
        -candidate.syntax_coverage,
        candidate.order,
    )


def rank_orders(
    words: Sequence[str],
    lex: WordNetLexicon,
    *,
    order_mode: str = "auto",
    beam_width: int = 128,
    exact_max_words: int = 5,
    top_k: int = DEFAULT_ORDER_CANDIDATES,
) -> tuple[tuple[OrderCandidate, ...], int]:
    """
    Return several strong complete orders for one unordered word bag.

    The input bag is canonicalized before any search, so equal scores cannot be
    resolved differently merely because the generator happened to emit
    ``power knowledge is`` instead of ``is knowledge power``.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    canonical_words = tuple(sorted(words))
    n = len(canonical_words)
    if n == 0:
        return (), 0

    if n == 1:
        structure = phrase_structure(canonical_words, lex)
        raw = local_grammar_raw(canonical_words, lex)
        candidate = OrderCandidate(
            order=canonical_words,
            grammar_raw=raw,
            grammar_norm=grammar_normalize(raw),
            structure_norm=structure.norm,
            valency_norm=structure.valency,
            syntax_coverage=structure.coverage,
            phrase_kind=structure.kind,
            objective=(
                0.38 * grammar_normalize(raw)
                + 0.44 * structure.norm
                + 0.12 * structure.valency
                + 0.06 * structure.coverage
            ),
        )
        return (candidate,), 1

    pair, starts, ends = _order_local_tables(canonical_words, lex)
    use_exact = order_mode == "exact" or (
        order_mode == "auto" and n <= exact_max_words
    )

    if use_exact:
        order_iter: Iterable[tuple[int, ...]] = _exact_index_orders(n)
    else:
        # Retain a larger locally-plausible pool than the final top-K because
        # whole-clause structure is deliberately non-decomposable.
        search_width = max(beam_width, top_k * 8)
        order_iter = _kbest_local_orders(
            n,
            pair,
            starts,
            ends,
            max_complete=search_width,
        )

    by_order: dict[tuple[str, ...], OrderCandidate] = {}
    evaluated = 0

    for idx_order in order_iter:
        word_order = tuple(canonical_words[i] for i in idx_order)
        # Repeated words can realize the same phrase through several position
        # permutations. Score each realized sequence once.
        if word_order in by_order:
            continue

        evaluated += 1
        raw = _local_raw_indices(idx_order, pair, starts, ends)
        grammar_norm = grammar_normalize(raw)
        structure = phrase_structure(word_order, lex)
        objective = (
            0.38 * grammar_norm
            + 0.44 * structure.norm
            + 0.12 * structure.valency
            + 0.06 * structure.coverage
        )
        by_order[word_order] = OrderCandidate(
            order=word_order,
            grammar_raw=raw,
            grammar_norm=grammar_norm,
            structure_norm=structure.norm,
            valency_norm=structure.valency,
            syntax_coverage=structure.coverage,
            phrase_kind=structure.kind,
            objective=objective,
        )

    ranked = tuple(sorted(by_order.values(), key=_candidate_sort_key)[:top_k])
    return ranked, evaluated


def best_order(
    words: Sequence[str],
    lex: WordNetLexicon,
    *,
    order_mode: str = "auto",
    beam_width: int = 128,
    exact_max_words: int = 5,
) -> tuple[float, tuple[str, ...], StructureResult, int]:
    """Compatibility wrapper returning the strongest grammar-only order."""
    candidates, evaluated = rank_orders(
        words,
        lex,
        order_mode=order_mode,
        beam_width=beam_width,
        exact_max_words=exact_max_words,
        top_k=1,
    )
    if not candidates:
        structure = phrase_structure(tuple(words), lex)
        return local_grammar_raw(words, lex), tuple(words), structure, evaluated

    winner = candidates[0]
    # OrderCandidate deliberately stores only the structure fields needed by the
    # ranker. Public callers of best_order historically receive the complete
    # StructureResult, so recompute it for the single winning order rather than
    # fabricating the omitted complexity/raw fields.
    structure = phrase_structure(winner.order, lex)
    return winner.grammar_raw, winner.order, structure, evaluated


def _worker_init(
    wordnet_dir: str,
    order_mode: str,
    beam_width: int,
    exact_max_words: int,
    order_candidates: int,
) -> None:
    global _WORKER_LEX, _WORKER_ORDER_MODE, _WORKER_BEAM_WIDTH
    global _WORKER_EXACT_MAX_WORDS, _WORKER_ORDER_CANDIDATES
    _WORKER_LEX = WordNetLexicon.load(Path(wordnet_dir))
    _WORKER_ORDER_MODE = order_mode
    _WORKER_BEAM_WIDTH = beam_width
    _WORKER_EXACT_MAX_WORDS = exact_max_words
    _WORKER_ORDER_CANDIDATES = order_candidates


def _worker_analyze_batch(
    batch: tuple[tuple[int, tuple[str, ...]], ...],
) -> list[DeepResult]:
    if _WORKER_LEX is None:
        raise RuntimeError("Worker WordNet lexicon was not initialized.")

    out: list[DeepResult] = []
    for row_index, words in batch:
        candidates, evaluated = rank_orders(
            words,
            _WORKER_LEX,
            order_mode=_WORKER_ORDER_MODE,
            beam_width=_WORKER_BEAM_WIDTH,
            exact_max_words=_WORKER_EXACT_MAX_WORDS,
            top_k=_WORKER_ORDER_CANDIDATES,
        )
        if not candidates:
            continue
        winner = candidates[0]
        out.append(
            DeepResult(
                row_index=row_index,
                grammar_raw=winner.grammar_raw,
                best_order=winner.order,
                structure_norm=winner.structure_norm,
                valency_norm=winner.valency_norm,
                syntax_coverage=winner.syntax_coverage,
                phrase_kind=winner.phrase_kind,
                orders_evaluated=evaluated,
                order_candidates=candidates,
            )
        )
    return out


def _order_base_final(row: Row, candidate: OrderCandidate) -> float:
    return 100.0 * (
        0.10 * row.lex
        + 0.16 * row.fam
        + 0.12 * row.hint
        + 0.22 * candidate.grammar_norm
        + 0.28 * candidate.structure_norm
        + 0.08 * candidate.valency_norm
        + 0.04 * row.wn_coverage
    )


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
    row.final = core.score_final(row)
    row.base_final = row.final
    _ORDER_CANDIDATES_BY_ROW_ID[id(row)] = result.order_candidates


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
    """Multicore deep analysis that also returns top-K alternative orders."""
    selected_sorted = sorted(selected)
    total = len(selected_sorted)
    if total == 0:
        return {"seconds": 0.0, "orders": 0.0, "candidates": 0.0}

    _ORDER_CANDIDATES_BY_ROW_ID.clear()
    resolved_backend = resolve_backend(backend, workers)
    print(
        f"Deep backend: {resolved_backend}; workers={workers}; "
        f"order_mode={order_mode}; exact<= {exact_max_words}; "
        f"k-best width={beam_width}; retained-orders={_ORDER_CANDIDATE_COUNT}; "
        f"batch={batch_size}"
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
            candidates, evaluated = rank_orders(
                rows[row_index].words,
                lex,
                order_mode=order_mode,
                beam_width=beam_width,
                exact_max_words=exact_max_words,
                top_k=_ORDER_CANDIDATE_COUNT,
            )
            if not candidates:
                continue
            winner = candidates[0]
            result = DeepResult(
                row_index=row_index,
                grammar_raw=winner.grammar_raw,
                best_order=winner.order,
                structure_norm=winner.structure_norm,
                valency_norm=winner.valency_norm,
                syntax_coverage=winner.syntax_coverage,
                phrase_kind=winner.phrase_kind,
                orders_evaluated=evaluated,
                order_candidates=candidates,
            )
            _apply_deep_result(rows, result)
            progress(1, evaluated)

    else:
        payloads = [
            tuple((i, rows[i].words) for i in batch)
            for batch in chunked(selected_sorted, batch_size)
        ]

        if resolved_backend == "thread":
            global _WORKER_LEX, _WORKER_ORDER_MODE, _WORKER_BEAM_WIDTH
            global _WORKER_EXACT_MAX_WORDS, _WORKER_ORDER_CANDIDATES
            _WORKER_LEX = lex
            _WORKER_ORDER_MODE = order_mode
            _WORKER_BEAM_WIDTH = beam_width
            _WORKER_EXACT_MAX_WORDS = exact_max_words
            _WORKER_ORDER_CANDIDATES = _ORDER_CANDIDATE_COUNT
            pool_type = ThreadPoolExecutor
            pool_kwargs = {"max_workers": workers}
        elif resolved_backend == "process":
            pool_type = ProcessPoolExecutor
            pool_kwargs = {
                "max_workers": workers,
                "initializer": _worker_init,
                "initargs": (
                    str(wordnet_dir),
                    order_mode,
                    beam_width,
                    exact_max_words,
                    _ORDER_CANDIDATE_COUNT,
                ),
            }
        else:
            raise ValueError(f"Unsupported backend: {resolved_backend}")

        with pool_type(**pool_kwargs) as pool:
            futures = [pool.submit(_worker_analyze_batch, payload) for payload in payloads]
            for fut in as_completed(futures):
                results = fut.result()
                order_count = 0
                for result in results:
                    _apply_deep_result(rows, result)
                    order_count += result.orders_evaluated
                progress(len(results), order_count)

    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "orders": float(total_orders),
        "candidates": float(total),
    }


def _row_phrase_candidates(row: Row) -> tuple[OrderCandidate, ...]:
    candidates = _ORDER_CANDIDATES_BY_ROW_ID.get(id(row))
    if candidates:
        return candidates

    structure = StructureResult(
        row.structure_norm,
        row.valency_norm,
        row.syntax_coverage,
        0.5,
        row.phrase_kind,
        4.0 * row.structure_norm,
    )
    objective = (
        0.38 * row.grammar_norm
        + 0.44 * structure.norm
        + 0.12 * structure.valency
        + 0.06 * structure.coverage
    )
    return (
        OrderCandidate(
            order=row.best_order,
            grammar_raw=row.grammar_raw,
            grammar_norm=row.grammar_norm,
            structure_norm=row.structure_norm,
            valency_norm=row.valency_norm,
            syntax_coverage=row.syntax_coverage,
            phrase_kind=row.phrase_kind,
            objective=objective,
        ),
    )



def _corpus_probe_scores(
    bucket: Sequence[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
) -> dict[int, float]:
    """Cheap positive-only corpus evidence for phrase-rescore admission.

    Full phrase scoring performs several n-gram lookups per retained order. Doing
    that for every deep row would erase most of the late-stage shortlist's cost
    advantage. This probe instead batches only whole-order phrase lookups and
    combines them with the already in-memory positive bigram model.

    Missing corpus evidence remains exactly neutral: a zero score never removes
    a row selected by PRE or FINAL.
    """
    scores = {id(row): 0.0 for row in bucket}
    phrase_owners: dict[str, set[int]] = defaultdict(set)

    for row in bucket:
        row_id = id(row)
        for candidate in _row_phrase_candidates(row):
            if collocation is not None:
                colloc, _ = collocation.score(candidate.order)
                scores[row_id] = max(scores[row_id], 0.55 * colloc)

            if phrase_index is not None:
                phrase_owners[" ".join(candidate.order)].add(row_id)

    if phrase_index is not None and phrase_owners:
        # PhraseIndex.counts() batches its SQLite queries internally. Any whole
        # retained order attested by the corpus is stronger admission evidence
        # than isolated bigrams, matching PhraseIndex.score()'s hierarchy.
        for phrase, count in phrase_index.counts(tuple(phrase_owners)).items():
            if count <= 0:
                continue
            exact = min(
                1.0,
                0.72 + 0.28 * math.log10(count + 1.0) / 5.0,
            )
            for row_id in phrase_owners.get(phrase, ()):
                scores[row_id] = max(scores[row_id], exact)

    return scores


def _select_phrase_rescore_rows(
    bucket: Sequence[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
) -> tuple[list[Row], int]:
    """Diversified late-stage shortlist without sacrificing existing winners.

    PRE and FINAL keep their historical full quotas. A third bounded channel
    admits rows whose retained grammatical orders have positive corpus evidence.
    This improves recall for word bags that look mediocre under order-agnostic
    lexical scoring but form a strongly attested phrase in one retained order.
    """
    by_final = sorted(
        bucket,
        key=lambda r: (-r.final, -r.pre_score, r.words),
    )[:top_per_group]
    by_pre = sorted(
        bucket,
        key=lambda r: (-r.pre_score, -r.final, r.words),
    )[:top_per_group]

    baseline_by_id = {id(row): row for row in (*by_final, *by_pre)}
    probe_scores = _corpus_probe_scores(
        bucket,
        collocation=collocation,
        phrase_index=phrase_index,
    )
    by_corpus = [
        row
        for row in sorted(
            bucket,
            key=lambda r: (
                -probe_scores.get(id(r), 0.0),
                -max(r.final, r.pre_score),
                r.words,
            ),
        )
        if probe_scores.get(id(row), 0.0) > 0.0
    ][:top_per_group]

    corpus_added = sum(1 for row in by_corpus if id(row) not in baseline_by_id)
    chosen_by_id = {
        id(row): row for row in (*by_final, *by_pre, *by_corpus)
    }
    chosen = sorted(
        chosen_by_id.values(),
        key=lambda r: (-max(r.final, r.pre_score), r.words),
    )
    return chosen, corpus_added

def apply_phrase_rescore(
    rows: list[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
    bonus_max: float,
) -> int:
    """
    Rescore a union of strong PRE and strong grammar candidates, then score
    every retained order for each selected bag.

    Corpus absence remains neutral. Phrase/collocation evidence can therefore
    rescue a known expression without making unseen but grammatical phrases bad.
    """
    by_wc: dict[int, list[Row]] = defaultdict(list)
    for row in rows:
        if row.deep:
            by_wc[row.word_count].append(row)

    rescored = 0
    for word_count, bucket in sorted(by_wc.items()):
        chosen, corpus_added = _select_phrase_rescore_rows(
            bucket,
            collocation=collocation,
            phrase_index=phrase_index,
            top_per_group=top_per_group,
        )
        if corpus_added:
            print(
                f"Corpus-probe shortlist {word_count} words: "
                f"added {corpus_added:,} candidate(s) beyond PRE/FINAL."
            )

        for row in chosen:
            current_order = row.best_order
            order_results: list[tuple[float, float, float, OrderCandidate]] = []

            for candidate in _row_phrase_candidates(row):
                colloc = 0.0
                if collocation is not None:
                    colloc, _ = collocation.score(candidate.order)

                phrase = 0.0
                if phrase_index is not None:
                    phrase, _ = phrase_index.score(candidate.order)

                evidence = max(phrase, 0.55 * colloc)
                base = _order_base_final(row, candidate)
                combined = min(100.0, base + bonus_max * evidence)
                order_results.append((combined, phrase, colloc, candidate))

            if not order_results:
                continue

            if not any(phrase > 0.0 or colloc > 0.0 for _, phrase, colloc, _ in order_results):
                row.base_final = row.final
                row.colloc_norm = 0.0
                row.phrase_attest_norm = 0.0
                row.phrase_bonus = 0.0
                rescored += 1
                continue

            order_results.sort(
                key=lambda item: (
                    -item[0],
                    -item[1],
                    -item[2],
                    0 if item[3].order == current_order else 1,
                    item[3].order,
                )
            )
            combined, phrase, colloc, winner = order_results[0]

            row.best_order = winner.order
            row.grammar_raw = winner.grammar_raw
            row.grammar_norm = winner.grammar_norm
            row.structure_norm = winner.structure_norm
            row.valency_norm = winner.valency_norm
            row.syntax_coverage = winner.syntax_coverage
            row.phrase_kind = winner.phrase_kind
            row.base_final = _order_base_final(row, winner)
            row.colloc_norm = colloc
            row.phrase_attest_norm = phrase
            row.phrase_bonus = combined - row.base_final
            row.final = combined
            rescored += 1

    return rescored


def _consume_int_flag(argv: list[str], flag: str, default: int) -> tuple[list[str], int]:
    """Consume ``--flag N`` or ``--flag=N`` before core argparse sees it."""
    out: list[str] = []
    value = default
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == flag:
            if i + 1 >= len(argv):
                raise SystemExit(f"{flag} requires an integer")
            try:
                value = int(argv[i + 1])
            except ValueError as exc:
                raise SystemExit(f"{flag} requires an integer") from exc
            i += 2
            continue
        if arg.startswith(flag + "="):
            try:
                value = int(arg.split("=", 1)[1])
            except ValueError as exc:
                raise SystemExit(f"{flag} requires an integer") from exc
            i += 1
            continue
        out.append(arg)
        i += 1
    return out, value


def _install_overrides() -> None:
    core.best_order = best_order
    core.deep_analyze = deep_analyze
    core.apply_phrase_rescore = apply_phrase_rescore
    core.DeepResult = DeepResult


_install_overrides()


def main() -> int:
    global _ORDER_CANDIDATE_COUNT
    cleaned, count = _consume_int_flag(
        sys.argv[1:], "--order-candidates", DEFAULT_ORDER_CANDIDATES
    )
    if count < 1:
        raise SystemExit("--order-candidates must be >= 1")
    _ORDER_CANDIDATE_COUNT = count
    sys.argv = [sys.argv[0], *cleaned]
    return core.main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
