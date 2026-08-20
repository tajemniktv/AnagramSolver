"""Typed access to the canonical AnagramSolver scenario catalog.

All shared real-world benchmark/CI data lives in ``anagram_benchmarks.json``.
Small synthetic fixtures that exist only to exercise one function stay beside
that subsystem's unit tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "anagram_benchmarks.json"
REGISTRY_SCHEMA = 2


@dataclass(frozen=True, slots=True)
class SmokeCase:
    id: str
    target: str
    expected_phrase: str | None = None


@dataclass(frozen=True, slots=True)
class OrderingGateConfig:
    min_recall_1: float
    min_recall_10: float
    min_recall_50: float
    min_mrr: float
    min_cross_bag_margin: float
    cross_bag_case_id: str
    malformed_bags: tuple[tuple[str, ...], ...]
    target_max_ranks: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PerformanceProbeConfig:
    frame_words: tuple[str, ...]
    function_words: tuple[str, ...]
    order_bags: tuple[tuple[str, ...], ...]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) for key in value
    ):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _load_document(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, "scenario catalog")


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list of strings")
    return tuple(item for item in value if isinstance(item, str))


def _word_bags(value: object, name: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list of word lists")
    return tuple(_string_tuple(raw, name) for raw in value)


def _number(mapping: dict[str, object], key: str, name: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name}.{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}.{key} must be finite")
    return number


def _letters(text: str) -> tuple[str, ...]:
    return tuple(sorted(ch.lower() for ch in text if ch.isalpha()))


def _parse_profiles(
    document: dict[str, object],
) -> tuple[
    tuple[SmokeCase, ...],
    OrderingGateConfig,
    PerformanceProbeConfig,
]:
    profiles = _mapping(document.get("profiles"), "profiles")

    raw_smoke = profiles.get("normal_user_cli")
    if not isinstance(raw_smoke, list):
        raise TypeError("profiles.normal_user_cli must be a list")
    smoke_cases: list[SmokeCase] = []
    for raw in raw_smoke:
        item = _mapping(raw, "normal_user_cli case")
        case_id = item.get("id")
        target = item.get("target")
        expected = item.get("expected_phrase")
        if not isinstance(case_id, str) or not isinstance(target, str):
            raise TypeError("normal_user_cli cases require string id and target")
        if expected is not None and not isinstance(expected, str):
            raise TypeError("normal_user_cli expected_phrase must be a string or null")
        smoke_cases.append(SmokeCase(case_id, target, expected))

    raw_ordering = _mapping(profiles.get("ordering_gate"), "profiles.ordering_gate")
    raw_ranks = _mapping(
        raw_ordering.get("target_max_ranks"),
        "profiles.ordering_gate.target_max_ranks",
    )
    target_max_ranks: list[tuple[str, int]] = []
    for case_id, maximum in raw_ranks.items():
        if not isinstance(maximum, int) or isinstance(maximum, bool):
            raise TypeError("ordering target max ranks must be integers")
        target_max_ranks.append((case_id, maximum))

    cross_bag_case_id = raw_ordering.get("cross_bag_case_id")
    if not isinstance(cross_bag_case_id, str):
        raise TypeError("ordering cross_bag_case_id must be a string")
    ordering = OrderingGateConfig(
        min_recall_1=_number(raw_ordering, "min_recall_1", "ordering_gate"),
        min_recall_10=_number(raw_ordering, "min_recall_10", "ordering_gate"),
        min_recall_50=_number(raw_ordering, "min_recall_50", "ordering_gate"),
        min_mrr=_number(raw_ordering, "min_mrr", "ordering_gate"),
        min_cross_bag_margin=_number(
            raw_ordering,
            "min_cross_bag_margin",
            "ordering_gate",
        ),
        cross_bag_case_id=cross_bag_case_id,
        malformed_bags=_word_bags(
            raw_ordering.get("malformed_bags"),
            "profiles.ordering_gate.malformed_bags",
        ),
        target_max_ranks=tuple(target_max_ranks),
    )

    raw_perf = _mapping(
        profiles.get("performance_probe"),
        "profiles.performance_probe",
    )
    performance = PerformanceProbeConfig(
        frame_words=_string_tuple(
            raw_perf.get("frame_words"),
            "profiles.performance_probe.frame_words",
        ),
        function_words=_string_tuple(
            raw_perf.get("function_words"),
            "profiles.performance_probe.function_words",
        ),
        order_bags=_word_bags(
            raw_perf.get("order_bags"),
            "profiles.performance_probe.order_bags",
        ),
    )
    return tuple(smoke_cases), ordering, performance


def _load_profiles(
    path: Path = DEFAULT_CASES,
) -> tuple[
    tuple[SmokeCase, ...],
    OrderingGateConfig,
    PerformanceProbeConfig,
]:
    document = _load_document(path)
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"scenario catalog schema must be {REGISTRY_SCHEMA}, "
            f"got {document.get('schema')!r}"
        )
    return _parse_profiles(document)


SMOKE_CASES, ORDERING_GATE, PERFORMANCE_PROBE = _load_profiles()


def load_cases(
    path: Path = DEFAULT_CASES,
    selected_ids: set[str] | None = None,
    *,
    require_ids: bool = True,
) -> list[dict[str, object]]:
    """Load benchmark cases, optionally permitting answer-only custom training data.

    Canonical and ID-addressable consumers keep ``require_ids=True``. The feature
    ranker's historical ``--cases`` contract accepts answer-only objects and uses
    ``require_ids=False``; it never performs stable-ID selection on those files.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    raw_cases: object = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise TypeError("benchmark case file must contain a cases list")

    selected = selected_ids or set()
    if selected and not require_ids:
        raise ValueError("selected_ids requires stable case IDs")

    cases: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise TypeError("benchmark cases must be JSON objects with string keys")
        if "answer" not in raw:
            raise ValueError("benchmark cases require answer")

        raw_id = raw.get("id")
        if require_ids:
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError("benchmark case id must be a non-empty string")
            if raw_id in seen:
                raise ValueError(f"duplicate benchmark case id: {raw_id}")
            seen.add(raw_id)

        cases.append(
            {key: value for key, value in raw.items() if isinstance(key, str)}
        )

    if selected:
        by_id = {str(case["id"]): case for case in cases}
        missing = selected - set(by_id)
        if missing:
            raise KeyError("Unknown case id(s): " + ", ".join(sorted(missing)))
        cases = [case for case in cases if str(case["id"]) in selected]
    return cases


def case_by_id(
    case_id: str,
    *,
    path: Path = DEFAULT_CASES,
) -> dict[str, object]:
    """Return one canonical benchmark case by stable id."""
    try:
        return next(case for case in load_cases(path) if case["id"] == case_id)
    except StopIteration as exc:
        raise KeyError(f"Unknown case id: {case_id}") from exc


def validate_registry(path: Path = DEFAULT_CASES) -> tuple[str, ...]:
    """Return cross-profile integrity errors for a complete scenario catalog."""
    errors: list[str] = []
    document = _load_document(path)
    if document.get("schema") != REGISTRY_SCHEMA:
        return (
            f"scenario catalog schema must be {REGISTRY_SCHEMA}, "
            f"got {document.get('schema')!r}",
        )

    smoke_cases, ordering_gate, performance_probe = _parse_profiles(document)
    cases = load_cases(path)
    cases_by_id = {str(case["id"]): case for case in cases}
    case_ids = set(cases_by_id)

    smoke_ids: set[str] = set()
    smoke_targets: set[str] = set()
    for smoke in smoke_cases:
        if smoke.id in smoke_ids:
            errors.append(f"duplicate smoke id: {smoke.id}")
        smoke_ids.add(smoke.id)
        if smoke.target in smoke_targets:
            errors.append(f"duplicate smoke target: {smoke.target}")
        smoke_targets.add(smoke.target)
        if smoke.expected_phrase is not None and _letters(smoke.target) != _letters(
            smoke.expected_phrase
        ):
            errors.append(f"smoke expected phrase is not an anagram: {smoke.id}")

        linked = cases_by_id.get(smoke.id)
        if linked is not None:
            answer = linked.get("answer")
            if not isinstance(answer, str) or _letters(smoke.target) != _letters(answer):
                errors.append(
                    f"smoke target disagrees with linked benchmark case: {smoke.id}"
                )

    shared_ids = {ordering_gate.cross_bag_case_id}
    shared_ids.update(case_id for case_id, _ in ordering_gate.target_max_ranks)
    missing_shared = shared_ids - case_ids
    for case_id in sorted(missing_shared):
        errors.append(f"CI profile references unknown benchmark case: {case_id}")

    if ordering_gate.min_cross_bag_margin < 0:
        errors.append("cross-bag margin must be non-negative")
    for name, value in (
        ("recall1", ordering_gate.min_recall_1),
        ("recall10", ordering_gate.min_recall_10),
        ("recall50", ordering_gate.min_recall_50),
        ("mrr", ordering_gate.min_mrr),
    ):
        if not 0.0 <= value <= 1.0:
            errors.append(f"ordering threshold {name} must be in [0, 1]")

    if (
        not performance_probe.frame_words
        or not performance_probe.function_words
        or not performance_probe.order_bags
    ):
        errors.append("performance probe workloads must be non-empty")
    elif any(not bag for bag in performance_probe.order_bags):
        errors.append("performance probe ordering bags must be non-empty")
    return tuple(errors)
