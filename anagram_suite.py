"""Typed access and management for the canonical AnagramSolver test registry.

All shared real-world benchmark/CI scenarios live in ``anagram_benchmarks.json``.
Each case declares which CI suites consume it. Small synthetic fixtures that only
exercise one function stay beside that subsystem's unit tests.
"""

from __future__ import annotations

import argparse
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "anagram_benchmarks.json"
REGISTRY_SCHEMA = 3

CASE_SUITES = (
    "ordering",
    "phrase_ordering",
    "normal_user_cli",
    "full",
    "performance",
    "refinement",
    "feature_ranker",
)
_SUITE_SET = frozenset(CASE_SUITES)
_SUITE_GROUPS = {
    "core": frozenset(("ordering", "phrase_ordering", "refinement", "feature_ranker")),
    "all": _SUITE_SET,
}
_ANSWER_REQUIRED_SUITES = _SUITE_SET - {"normal_user_cli"}


@dataclass(frozen=True, slots=True)
class NormalUserDefaults:
    timeout_seconds: int
    verbose: bool
    expect_answer: bool


@dataclass(frozen=True, slots=True)
class NormalUserCase:
    id: str
    target: str
    solver_args: tuple[str, ...]
    timeout_seconds: int
    verbose: bool
    expected_phrase: str | None


@dataclass(frozen=True, slots=True)
class OrderingGateConfig:
    min_recall_1: float
    min_recall_10: float
    min_recall_50: float
    min_mrr: float
    min_cross_bag_margin: float
    malformed_bags: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class PhraseOrderingConfig:
    order_candidates: int


@dataclass(frozen=True, slots=True)
class PerformanceProbeConfig:
    frame_words: tuple[str, ...]
    function_words: tuple[str, ...]


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a JSON object with string keys")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _load_document(path: Path) -> dict[str, object]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(payload, "test registry")


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


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _arg_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _ascii_letters(text: str) -> tuple[str, ...]:
    normalized = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return tuple(ch for ch in normalized if "a" <= ch <= "z")


def _letters(text: str) -> tuple[str, ...]:
    return tuple(sorted(_ascii_letters(text)))


def _has_benchmark_token(text: str) -> bool:
    return bool(_ascii_letters(text))


def _expand_suites(value: object, name: str) -> frozenset[str]:
    if isinstance(value, str):
        raw = (value,)
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        raw = tuple(item for item in value if isinstance(item, str))
    else:
        raise TypeError(f"{name} must be a string or list of strings")

    suites: set[str] = set()
    for token in raw:
        if token in _SUITE_GROUPS:
            suites.update(_SUITE_GROUPS[token])
        elif token in _SUITE_SET:
            suites.add(token)
        else:
            raise ValueError(f"{name} contains unknown suite/group: {token}")
    return frozenset(suites)


def _parse_defaults(
    document: dict[str, object],
) -> tuple[frozenset[str], NormalUserDefaults]:
    defaults = _mapping(document.get("defaults"), "defaults")
    default_suites = _expand_suites(defaults.get("suites", ["all"]), "defaults.suites")
    raw_cli = _mapping(defaults.get("normal_user_cli", {}), "defaults.normal_user_cli")
    timeout = _positive_int(
        raw_cli.get("timeout_seconds", 120),
        "normal_user_cli.timeout_seconds",
    )
    verbose = raw_cli.get("verbose", True)
    expect_answer = raw_cli.get("expect_answer", True)
    if not isinstance(verbose, bool) or not isinstance(expect_answer, bool):
        raise TypeError("normal_user_cli verbose/expect_answer defaults must be booleans")
    return default_suites, NormalUserDefaults(timeout, verbose, expect_answer)


def _parse_profiles(
    document: dict[str, object],
) -> tuple[OrderingGateConfig, PhraseOrderingConfig, PerformanceProbeConfig]:
    profiles = _mapping(document.get("profiles"), "profiles")

    raw_ordering = _mapping(profiles.get("ordering_gate"), "profiles.ordering_gate")
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
        malformed_bags=_word_bags(
            raw_ordering.get("malformed_bags"),
            "profiles.ordering_gate.malformed_bags",
        ),
    )

    raw_phrase = _mapping(
        profiles.get("phrase_ordering"),
        "profiles.phrase_ordering",
    )
    phrase = PhraseOrderingConfig(
        _positive_int(
            raw_phrase.get("order_candidates"),
            "phrase_ordering.order_candidates",
        )
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
    )
    return ordering, phrase, performance


def _load_registry_config(
    path: Path = DEFAULT_CASES,
) -> tuple[
    frozenset[str],
    NormalUserDefaults,
    OrderingGateConfig,
    PhraseOrderingConfig,
    PerformanceProbeConfig,
]:
    document = _load_document(path)
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"test registry schema must be {REGISTRY_SCHEMA}, "
            f"got {document.get('schema')!r}"
        )
    default_suites, normal_user_defaults = _parse_defaults(document)
    ordering, phrase, performance = _parse_profiles(document)
    return default_suites, normal_user_defaults, ordering, phrase, performance


(
    DEFAULT_SUITES,
    NORMAL_USER_DEFAULTS,
    ORDERING_GATE,
    PHRASE_ORDERING,
    PERFORMANCE_PROBE,
) = _load_registry_config()


def load_cases(
    path: Path = DEFAULT_CASES,
    selected_ids: set[str] | None = None,
    *,
    require_ids: bool = True,
    require_answer: bool = False,
) -> list[dict[str, object]]:
    """Load raw case objects from a registry or legacy case file.

    Raw registry cases may be target-only because the normal-user suite supports
    control puzzles without a known answer. Consumers that require labels, such
    as custom feature-ranker training, opt into ``require_answer=True``.
    """
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    raw_cases: object = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list):
        raise TypeError("case file must contain a cases list")

    selected = selected_ids or set()
    if selected and not require_ids:
        raise ValueError("selected_ids requires stable case IDs")

    cases: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise TypeError("cases must be JSON objects with string keys")

        raw_id = raw.get("id")
        if require_ids:
            if not isinstance(raw_id, str) or not raw_id:
                raise ValueError("case id must be a non-empty string")
            if raw_id in seen:
                raise ValueError(f"duplicate case id: {raw_id}")
            seen.add(raw_id)

        answer = raw.get("answer")
        if require_answer and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("cases require a non-empty answer")

        cases.append({key: value for key, value in raw.items() if isinstance(key, str)})

    if selected:
        by_id = {str(case["id"]): case for case in cases}
        missing = selected - set(by_id)
        if missing:
            raise KeyError("Unknown case id(s): " + ", ".join(sorted(missing)))
        cases = [case for case in cases if str(case["id"]) in selected]
    return cases


def _case_suites(
    case: dict[str, object],
    default_suites: frozenset[str],
) -> frozenset[str]:
    return _expand_suites(
        case.get("suites", list(default_suites)),
        f"case {case.get('id')}.suites",
    )


def cases_for(
    suite: str,
    *,
    path: Path = DEFAULT_CASES,
    selected_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Return enabled registry cases selected for one named CI/test suite."""
    if suite not in _SUITE_SET:
        raise ValueError(f"Unknown case suite: {suite}")
    document = _load_document(path)
    if document.get("schema") != REGISTRY_SCHEMA:
        raise ValueError(
            f"test registry schema must be {REGISTRY_SCHEMA}, "
            f"got {document.get('schema')!r}"
        )
    default_suites, _ = _parse_defaults(document)
    cases = load_cases(path)
    selected: list[dict[str, object]] = []
    for case in cases:
        enabled = case.get("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError(f"case {case.get('id')} enabled must be boolean")
        if not enabled or suite not in _case_suites(case, default_suites):
            continue
        if suite in _ANSWER_REQUIRED_SUITES:
            answer = case.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(
                    f"case {case.get('id')} requires an answer for suite {suite}"
                )
        selected.append(case)

    wanted = selected_ids or set()
    if wanted:
        by_id = {str(case["id"]): case for case in selected}
        missing = wanted - set(by_id)
        if missing:
            raise KeyError(
                f"Case id(s) not enabled for {suite}: " + ", ".join(sorted(missing))
            )
        selected = [case for case in selected if str(case["id"]) in wanted]
    return selected


def case_by_id(
    case_id: str,
    *,
    path: Path = DEFAULT_CASES,
) -> dict[str, object]:
    """Return one canonical registry case by stable id, including target-only cases."""
    try:
        return next(case for case in load_cases(path) if case["id"] == case_id)
    except StopIteration as exc:
        raise KeyError(f"Unknown case id: {case_id}") from exc


def case_options(case: dict[str, object], suite: str) -> dict[str, object]:
    """Merge common solver options with one suite's per-case overrides."""
    merged: dict[str, object] = {}
    raw_solver = case.get("solver")
    if raw_solver is not None:
        merged.update(_mapping(raw_solver, f"case {case.get('id')}.solver"))
    raw_suite = case.get(suite)
    if raw_suite is not None:
        merged.update(_mapping(raw_suite, f"case {case.get('id')}.{suite}"))
    return merged


def target_for_case(case: dict[str, object]) -> str:
    """Return explicit input text or derive canonical letters from source/answer."""
    target = case.get("target")
    if isinstance(target, str) and target.strip():
        return target
    for key in ("source", "answer"):
        value = case.get(key)
        if isinstance(value, str) and value.strip():
            derived = "".join(ch for ch in value if ch.isalpha()).upper()
            if derived:
                return derived
    raise ValueError(f"case {case.get('id')} needs target, source, or answer")


def _option_strings(options: dict[str, object], key: str) -> tuple[str, ...]:
    value = options.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} must be a list of strings")
    return tuple(item for item in value if isinstance(item, str))


def normal_user_case(
    case: dict[str, object],
    *,
    defaults: NormalUserDefaults = NORMAL_USER_DEFAULTS,
) -> NormalUserCase:
    """Resolve one registry case into an ordinary user-facing solver invocation."""
    options = case_options(case, "normal_user_cli")
    target = target_for_case(case)

    timeout_value = options.get("timeout_seconds", defaults.timeout_seconds)
    timeout = _positive_int(
        timeout_value,
        f"case {case.get('id')}.timeout_seconds",
    )
    verbose = options.get("verbose", defaults.verbose)
    expect_answer = options.get("expect_answer", defaults.expect_answer)
    if not isinstance(verbose, bool) or not isinstance(expect_answer, bool):
        raise TypeError("normal_user_cli verbose/expect_answer must be booleans")

    explicit_expected = options.get("expected_phrase")
    if explicit_expected is not None and not isinstance(explicit_expected, str):
        raise TypeError("normal_user_cli expected_phrase must be a string or null")
    answer = case.get("answer")
    expected_phrase = (
        explicit_expected
        if isinstance(explicit_expected, str)
        else str(answer)
        if expect_answer and isinstance(answer, str) and answer.strip()
        else None
    )

    args: list[str] = []
    for key, flag in (
        ("hints", "--hint"),
        ("exclude", "--exclude"),
        ("require", "--require"),
    ):
        for value in _option_strings(options, key):
            args += [flag, value]

    for key, flag in (
        ("words", "--words"),
        ("min_words", "--min-words"),
        ("max_words", "--max-words"),
        ("min_word_len", "--min-word-len"),
        ("top", "--top"),
        ("workers", "--workers"),
        ("order_candidates", "--order-candidates"),
    ):
        value = options.get(key)
        if value is not None:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"normal_user_cli {key} must be an integer")
            args += [flag, str(value)]

    min_zipf_value = options.get("min_zipf")
    if min_zipf_value is not None:
        if not isinstance(min_zipf_value, (int, float)) or isinstance(
            min_zipf_value,
            bool,
        ):
            raise TypeError("normal_user_cli min_zipf must be numeric")
        min_zipf = float(min_zipf_value)
        if not math.isfinite(min_zipf):
            raise ValueError("normal_user_cli min_zipf must be finite")
        args += ["--min-zipf", str(min_zipf)]

    mode = options.get("mode", "balanced")
    if mode not in ("balanced", "quick", "exhaustive"):
        raise ValueError("normal_user_cli mode must be balanced, quick, or exhaustive")
    if mode == "quick":
        args.append("--quick")
    elif mode == "exhaustive":
        args.append("--exhaustive")

    if options.get("rebuild", False):
        args.append("--rebuild")
    if verbose:
        args.append("--verbose")

    # Reuse the real frontend's validation so registry-backed CI cannot accept
    # argument ranges or require/word-count combinations the solver itself rejects.
    from anagram_solver import _validate_args as validate_solver_args
    from anagram_solver import build_parser as build_solver_parser

    try:
        parsed = build_solver_parser().parse_args([target, *args])
        validate_solver_args(parsed)
    except SystemExit as exc:
        raise ValueError(f"solver arguments invalid: {exc}") from exc

    return NormalUserCase(
        id=str(case["id"]),
        target=target,
        solver_args=tuple(args),
        timeout_seconds=timeout,
        verbose=verbose,
        expected_phrase=expected_phrase,
    )


def validate_registry(path: Path = DEFAULT_CASES) -> tuple[str, ...]:
    """Return integrity errors for the complete case-centric registry."""
    errors: list[str] = []
    try:
        document = _load_document(path)
        if document.get("schema") != REGISTRY_SCHEMA:
            return (
                f"test registry schema must be {REGISTRY_SCHEMA}, "
                f"got {document.get('schema')!r}",
            )
        default_suites, cli_defaults = _parse_defaults(document)
        ordering_gate, phrase_ordering, performance_probe = _parse_profiles(document)
        cases = load_cases(path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        return (str(exc),)

    cross_bag_references: list[str] = []
    for case in cases:
        case_id = str(case["id"])
        enabled = case.get("enabled", True)
        if not isinstance(enabled, bool):
            errors.append(f"case {case_id} enabled must be boolean")
            continue
        try:
            suites = _case_suites(case, default_suites)
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if enabled and not suites:
            errors.append(f"case {case_id} is enabled but has no suites")

        answer = case.get("answer")
        for suite in suites & _ANSWER_REQUIRED_SUITES:
            if not isinstance(answer, str) or not answer.strip():
                errors.append(f"case {case_id} requires an answer for suite {suite}")
        if (
            enabled
            and "performance" in suites
            and isinstance(answer, str)
            and answer.strip()
            and not _has_benchmark_token(answer)
        ):
            errors.append(f"case {case_id} performance answer has no benchmark tokens")

        explicit_target = case.get("target")
        has_explicit_target = isinstance(explicit_target, str) and bool(
            explicit_target.strip()
        )
        if (
            isinstance(explicit_target, str)
            and explicit_target.strip()
            and isinstance(answer, str)
            and _letters(explicit_target) != _letters(answer)
        ):
            errors.append(f"case {case_id} target is not an anagram of answer")

        source = case.get("source")
        if (
            not has_explicit_target
            and isinstance(source, str)
            and source.strip()
            and isinstance(answer, str)
            and _letters(source) != _letters(answer)
        ):
            errors.append(f"case {case_id} source is not an anagram of answer")

        if "normal_user_cli" in suites:
            try:
                normal_user_case(case, defaults=cli_defaults)
            except (TypeError, ValueError) as exc:
                errors.append(f"case {case_id} normal_user_cli: {exc}")

        ordering_options = case.get("ordering")
        if ordering_options is not None:
            try:
                options = _mapping(ordering_options, f"case {case_id}.ordering")
            except TypeError as exc:
                errors.append(str(exc))
                options = {}
            max_rank = options.get("max_rank")
            if max_rank is not None and (
                not isinstance(max_rank, int)
                or isinstance(max_rank, bool)
                or max_rank < 1
            ):
                errors.append(
                    f"case {case_id} ordering.max_rank must be a positive integer"
                )
            cross_ref = options.get("cross_bag_reference", False)
            if not isinstance(cross_ref, bool):
                errors.append(
                    f"case {case_id} ordering.cross_bag_reference must be boolean"
                )
            elif cross_ref and enabled:
                if "ordering" not in suites:
                    errors.append(f"case {case_id} cross-bag reference must run ordering")
                else:
                    cross_bag_references.append(case_id)

    if len(cross_bag_references) != 1:
        errors.append(
            "ordering gate requires exactly one enabled cross_bag_reference case; got "
            + repr(cross_bag_references)
        )

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

    if phrase_ordering.order_candidates < 1:
        errors.append("phrase ordering candidate count must be positive")
    if not performance_probe.frame_words or not performance_probe.function_words:
        errors.append("performance probe word workloads must be non-empty")

    for suite in CASE_SUITES:
        try:
            if not cases_for(suite, path=path):
                errors.append(f"suite {suite} has no enabled cases")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
    return tuple(dict.fromkeys(errors))


def _write_registry(path: Path, document: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    errors = validate_registry(tmp)
    if errors:
        tmp.unlink(missing_ok=True)
        raise SystemExit("Registry update rejected:\n  - " + "\n  - ".join(errors))
    tmp.replace(path)


def _management_main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or edit the AnagramSolver test registry"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_CASES)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="validate the registry")
    list_parser = sub.add_parser("list", help="list registry cases")
    list_parser.add_argument("--suite", choices=CASE_SUITES)

    add = sub.add_parser("add", help="add one case; omitted --suite means all suites")
    add.add_argument("id")
    add.add_argument("--answer")
    add.add_argument("--target")
    add.add_argument("--category", default="custom")
    add.add_argument(
        "--suite",
        action="append",
        choices=(*CASE_SUITES, "core", "all"),
    )
    add.add_argument("--hint", action="append", default=[])
    add.add_argument("--words", type=_arg_positive_int)
    verbosity = add.add_mutually_exclusive_group()
    verbosity.add_argument("--verbose", dest="verbose", action="store_true")
    verbosity.add_argument("--quiet", dest="verbose", action="store_false")
    add.set_defaults(verbose=None)

    for command in ("remove", "enable", "disable"):
        action = sub.add_parser(command, help=f"{command} one registry case")
        action.add_argument("id")

    args = parser.parse_args()
    path = args.registry.expanduser()
    if args.command == "validate":
        errors = validate_registry(path)
        if errors:
            print("Registry INVALID:")
            for error in errors:
                print(f"  - {error}")
            return 1
        print("Registry valid.")
        return 0

    document = _load_document(path)
    cases = load_cases(path)
    if args.command == "list":
        selected = cases_for(args.suite, path=path) if args.suite else cases
        default_suites, _ = _parse_defaults(document)
        for case in selected:
            enabled = case.get("enabled", True)
            suites = ",".join(sorted(_case_suites(case, default_suites)))
            answer = case.get("answer", "?")
            print(
                f"{case['id']:<34} enabled={enabled!s:<5} "
                f"suites={suites:<70} {answer}"
            )
        return 0

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise SystemExit("Registry has no cases list")
    index = next(
        (
            idx
            for idx, case in enumerate(raw_cases)
            if isinstance(case, dict) and case.get("id") == args.id
        ),
        None,
    )

    if args.command == "add":
        if index is not None:
            raise SystemExit(f"Case already exists: {args.id}")
        if not args.answer and not args.target:
            raise SystemExit("add requires --answer and/or --target")
        new_case: dict[str, object] = {"id": args.id, "category": args.category}
        if args.answer:
            new_case["answer"] = args.answer
        if args.target:
            new_case["target"] = args.target
        if args.suite:
            new_case["suites"] = args.suite
        solver: dict[str, object] = {}
        if args.hint:
            solver["hints"] = args.hint
        if args.words is not None:
            solver["words"] = args.words
        if solver:
            new_case["solver"] = solver
        if args.verbose is not None:
            new_case["normal_user_cli"] = {"verbose": args.verbose}
        raw_cases.append(new_case)
        _write_registry(path, document)
        print(f"Added {args.id}.")
        return 0

    if index is None:
        raise SystemExit(f"Unknown case id: {args.id}")
    if args.command == "remove":
        del raw_cases[index]
    else:
        case = raw_cases[index]
        assert isinstance(case, dict)
        case["enabled"] = args.command == "enable"
    _write_registry(path, document)
    print(f"{args.command.title()}d {args.id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_management_main())