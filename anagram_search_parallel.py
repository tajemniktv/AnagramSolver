from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

Signature = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    word: str
    sig: Signature
    length: int


@dataclass(frozen=True, slots=True)
class SearchTask:
    remaining: Signature
    start: int
    words_left: int
    prefix: tuple[str, ...]
    hint_matched: bool


_WORKER_CANDIDATES: tuple[SearchCandidate, ...] = ()
_WORKER_BY_SIGNATURE: dict[Signature, tuple[int, ...]] = {}
_WORKER_ALLOW_REPEAT = True
_WORKER_REQUIRED_ANY: frozenset[str] = frozenset()
_WORKER_HINT_INDICES: tuple[int, ...] = ()
_WORKER_MIN_LEN = 0
_WORKER_MAX_LEN = 0
_WORKER_STOP = None
_WORKER_TASK_LIMIT = 0


def resolve_worker_count(requested: int) -> int:
    """Resolve 0=auto to a conservative process count."""
    if requested > 0:
        return requested
    return max(1, min(8, os.cpu_count() or 1))


def fits(word_counts: Signature, remaining: Signature) -> bool:
    return all(w <= r for w, r in zip(word_counts, remaining))


def _subtract(a: Signature, b: Signature) -> Signature:
    return tuple(x - y for x, y in zip(a, b))


def _hint_possible(
    remaining: Signature,
    start: int,
    candidates: Sequence[SearchCandidate],
    hint_indices: Sequence[int],
) -> bool:
    return any(
        i >= start and fits(candidates[i].sig, remaining)
        for i in hint_indices
    )


def _state_possible(
    rem: Signature,
    start: int,
    words_left: int,
    *,
    min_len: int,
    max_len: int,
    candidates: Sequence[SearchCandidate],
    required_any: frozenset[str],
    hint_indices: Sequence[int],
    hint_matched: bool,
) -> bool:
    rem_len = sum(rem)
    if words_left == 0:
        return rem_len == 0 and (not required_any or hint_matched)
    if rem_len == 0:
        return False
    if rem_len < words_left * min_len or rem_len > words_left * max_len:
        return False
    if required_any and not hint_matched:
        if not _hint_possible(rem, start, candidates, hint_indices):
            return False
    return True


def _build_tasks(
    remaining: Signature,
    candidates: Sequence[SearchCandidate],
    min_words: int,
    max_words: int,
    allow_repeat: bool,
    required_any: frozenset[str],
    initial_any_matched: bool,
) -> Iterator[SearchTask]:
    if not candidates:
        return
    min_len = min(c.length for c in candidates)
    max_len = max(c.length for c in candidates)
    hint_indices = tuple(
        i for i, c in enumerate(candidates) if c.word in required_any
    )

    for nwords in range(min_words, max_words + 1):
        # Two chosen words are enough to produce a broad task frontier for the
        # large 4-6 word spaces, while avoiding giant per-task result lists.
        split_depth = min(2, max(1, nwords - 1))

        def descend(
            rem: Signature,
            start: int,
            depth_left: int,
            words_left: int,
            prefix: tuple[str, ...],
            hint_matched: bool,
        ) -> Iterator[SearchTask]:
            if not _state_possible(
                rem,
                start,
                words_left,
                min_len=min_len,
                max_len=max_len,
                candidates=candidates,
                required_any=required_any,
                hint_indices=hint_indices,
                hint_matched=hint_matched,
            ):
                return
            if depth_left == 0 or words_left <= 1:
                yield SearchTask(rem, start, words_left, prefix, hint_matched)
                return

            rem_len = sum(rem)
            min_this_len = max(
                min_len,
                rem_len - (words_left - 1) * max_len,
            )
            max_this_len = min(
                max_len,
                rem_len - (words_left - 1) * min_len,
            )

            for i in range(start, len(candidates)):
                candidate = candidates[i]
                if (
                    candidate.length < min_this_len
                    or candidate.length > max_this_len
                ):
                    continue
                if not fits(candidate.sig, rem):
                    continue
                yield from descend(
                    _subtract(rem, candidate.sig),
                    i if allow_repeat else i + 1,
                    depth_left - 1,
                    words_left - 1,
                    prefix + (candidate.word,),
                    hint_matched or candidate.word in required_any,
                )

        yield from descend(
            remaining,
            0,
            split_depth,
            nwords,
            (),
            initial_any_matched,
        )


def _init_worker(
    candidates: tuple[SearchCandidate, ...],
    allow_repeat: bool,
    required_any: frozenset[str],
    stop_event,
    task_limit: int,
) -> None:
    global _WORKER_CANDIDATES
    global _WORKER_BY_SIGNATURE
    global _WORKER_ALLOW_REPEAT
    global _WORKER_REQUIRED_ANY
    global _WORKER_HINT_INDICES
    global _WORKER_MIN_LEN
    global _WORKER_MAX_LEN
    global _WORKER_STOP
    global _WORKER_TASK_LIMIT

    _WORKER_CANDIDATES = candidates
    grouped: dict[Signature, list[int]] = defaultdict(list)
    for i, candidate in enumerate(candidates):
        grouped[candidate.sig].append(i)
    _WORKER_BY_SIGNATURE = {
        signature: tuple(indices) for signature, indices in grouped.items()
    }
    _WORKER_ALLOW_REPEAT = allow_repeat
    _WORKER_REQUIRED_ANY = required_any
    _WORKER_HINT_INDICES = tuple(
        i for i, candidate in enumerate(candidates)
        if candidate.word in required_any
    )
    _WORKER_MIN_LEN = min(c.length for c in candidates)
    _WORKER_MAX_LEN = max(c.length for c in candidates)
    _WORKER_STOP = stop_event
    _WORKER_TASK_LIMIT = task_limit


def _should_stop() -> bool:
    return _WORKER_STOP is not None and _WORKER_STOP.is_set()


def _solve_task(task: SearchTask) -> list[tuple[str, ...]]:
    output: list[tuple[str, ...]] = []
    dead: set[tuple[Signature, int, int, bool]] = set()

    def dfs(
        rem: Signature,
        start: int,
        words_left: int,
        chosen: tuple[str, ...],
        hint_matched: bool,
    ) -> None:
        if _should_stop():
            return
        if _WORKER_TASK_LIMIT > 0 and len(output) >= _WORKER_TASK_LIMIT:
            return

        if not _state_possible(
            rem,
            start,
            words_left,
            min_len=_WORKER_MIN_LEN,
            max_len=_WORKER_MAX_LEN,
            candidates=_WORKER_CANDIDATES,
            required_any=_WORKER_REQUIRED_ANY,
            hint_indices=_WORKER_HINT_INDICES,
            hint_matched=hint_matched,
        ):
            return

        if words_left == 0:
            output.append(chosen)
            return

        state = (rem, start, words_left, hint_matched)
        if state in dead:
            return
        before = len(output)

        if words_left == 1:
            for i in _WORKER_BY_SIGNATURE.get(rem, ()):
                if i < start:
                    continue
                word = _WORKER_CANDIDATES[i].word
                if _WORKER_REQUIRED_ANY and not (
                    hint_matched or word in _WORKER_REQUIRED_ANY
                ):
                    continue
                output.append(chosen + (word,))
                if (
                    _WORKER_TASK_LIMIT > 0
                    and len(output) >= _WORKER_TASK_LIMIT
                ):
                    return
            if len(output) == before:
                dead.add(state)
            return

        rem_len = sum(rem)
        min_this_len = max(
            _WORKER_MIN_LEN,
            rem_len - (words_left - 1) * _WORKER_MAX_LEN,
        )
        max_this_len = min(
            _WORKER_MAX_LEN,
            rem_len - (words_left - 1) * _WORKER_MIN_LEN,
        )
        for i in range(start, len(_WORKER_CANDIDATES)):
            if _should_stop():
                return
            candidate = _WORKER_CANDIDATES[i]
            if (
                candidate.length < min_this_len
                or candidate.length > max_this_len
            ):
                continue
            if not fits(candidate.sig, rem):
                continue
            dfs(
                _subtract(rem, candidate.sig),
                i if _WORKER_ALLOW_REPEAT else i + 1,
                words_left - 1,
                chosen + (candidate.word,),
                hint_matched or candidate.word in _WORKER_REQUIRED_ANY,
            )
            if (
                _WORKER_TASK_LIMIT > 0
                and len(output) >= _WORKER_TASK_LIMIT
            ):
                return

        if len(output) == before:
            dead.add(state)

    dfs(
        task.remaining,
        task.start,
        task.words_left,
        task.prefix,
        task.hint_matched,
    )
    return output


def _solve_in_process(
    remaining: Signature,
    candidates: tuple[SearchCandidate, ...],
    min_words: int,
    max_words: int,
    max_results: int,
    allow_repeat: bool,
    required_any: frozenset[str],
    initial_any_matched: bool,
) -> Iterator[tuple[str, ...]]:
    class NeverStop:
        @staticmethod
        def is_set() -> bool:
            return False

    _init_worker(
        candidates,
        allow_repeat,
        required_any,
        NeverStop(),
        max_results,
    )
    found = 0
    for task in _build_tasks(
        remaining,
        candidates,
        min_words,
        max_words,
        allow_repeat,
        required_any,
        initial_any_matched,
    ):
        for solution in _solve_task(task):
            yield solution
            found += 1
            if max_results > 0 and found >= max_results:
                return


def solve_parallel(
    remaining: Signature,
    candidates: Sequence[SearchCandidate],
    min_words: int,
    max_words: int,
    max_results: int,
    allow_repeat: bool,
    *,
    workers: int,
    required_any: Iterable[str] = (),
    initial_any_matched: bool = False,
) -> Iterator[tuple[str, ...]]:
    """Enumerate canonical exact bags with ordered process parallelism.

    Tasks are cut at ordered two-word prefixes and consumed in the same order as
    the historical DFS. Therefore a bounded no-clue search returns the same
    candidate prefix regardless of worker count. When clue words are supplied,
    branches that can no longer contain any clue are pruned during search and
    the cap applies to clue-satisfying candidates rather than discarded bags.
    """
    search_candidates = tuple(candidates)
    if not search_candidates:
        return

    worker_count = resolve_worker_count(workers)
    required = frozenset(required_any)
    if worker_count <= 1 or len(search_candidates) < 64:
        yield from _solve_in_process(
            remaining,
            search_candidates,
            min_words,
            max_words,
            max_results,
            allow_repeat,
            required,
            initial_any_matched,
        )
        return

    context = multiprocessing.get_context(
        "spawn" if os.name == "nt" else "fork"
    )
    stop_event = context.Event()
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            search_candidates,
            allow_repeat,
            required,
            stop_event,
            max_results,
        ),
    )
    found = 0
    pending: deque[
        concurrent.futures.Future[list[tuple[str, ...]]]
    ] = deque()
    task_iter = iter(
        _build_tasks(
            remaining,
            search_candidates,
            min_words,
            max_words,
            allow_repeat,
            required,
            initial_any_matched,
        )
    )
    exhausted = False
    window = max(worker_count * 3, 4)

    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < window:
                try:
                    task = next(task_iter)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(executor.submit(_solve_task, task))

            if not pending:
                break

            branch_results = pending.popleft().result()
            for solution in branch_results:
                yield solution
                found += 1
                if max_results > 0 and found >= max_results:
                    stop_event.set()
                    return
    finally:
        stop_event.set()
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
