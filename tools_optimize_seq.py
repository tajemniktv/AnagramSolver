from __future__ import annotations

from pathlib import Path


def find(lines: list[str], value: str, start: int = 0) -> int:
    try:
        return lines.index(value, start)
    except ValueError as exc:
        raise SystemExit(f"line anchor not found: {value!r}") from exc


path = Path("anagram_generate.py")
text = path.read_text(encoding="utf-8")
text = text.replace("import sys\nimport unicodedata\n", "import sys\nimport time\nimport unicodedata\n", 1)
lines = text.splitlines()

# Precompute sparse candidate signatures once per solve call.
i = find(lines, "    max_candidate_len = max(c.length for c in candidates)")
lines[i + 1:i + 1] = [
    "    sparse_signatures = [",
    "        tuple((letter, amount) for letter, amount in enumerate(c.sig) if amount)",
    "        for c in candidates",
    "    ]",
]

# Carry remaining length through recursion instead of summing 26 counters at every node.
i = find(lines, "    def dfs(")
rem_line = find(lines, "        rem: tuple[int, ...],", i)
lines.insert(rem_line + 1, "        rem_len: int,")
i = find(lines, "        rem_len = sum(rem)", rem_line)
del lines[i]

# Replace 26-wide fit/subtract operations with sparse letter operations.
i = find(lines, "            if not fits(c.sig, rem):")
assert lines[i + 1] == "                continue"
assert lines[i + 2] == "            new_rem = tuple(r - w for r, w in zip(rem, c.sig))"
assert lines[i + 3] == "            next_start = i if allow_repeat else i + 1"
assert lines[i + 4] == "            yield from dfs(new_rem, next_start, words_left - 1, chosen + [c.word])"
lines[i:i + 5] = [
    "            sparse = sparse_signatures[i]",
    "            candidate_fits = True",
    "            for letter, amount in sparse:",
    "                if rem[letter] < amount:",
    "                    candidate_fits = False",
    "                    break",
    "            if not candidate_fits:",
    "                continue",
    "            mutable_rem = list(rem)",
    "            for letter, amount in sparse:",
    "                mutable_rem[letter] -= amount",
    "            new_rem = tuple(mutable_rem)",
    "            next_start = i if allow_repeat else i + 1",
    "            yield from dfs(",
    "                new_rem,",
    "                rem_len - c.length,",
    "                next_start,",
    "                words_left - 1,",
    "                chosen + [c.word],",
    "            )",
]

# Update the root DFS call for the carried remaining length.
i = find(lines, "    for nwords in range(min_words, max_words + 1):")
lines.insert(i, "    initial_remaining_len = sum(remaining)")
i = find(lines, "        yield from dfs(remaining, 0, nwords, [])", i)
lines[i] = "        yield from dfs(remaining, initial_remaining_len, 0, nwords, [])"

# Time only the exact-cover search, separately from frequency loading and export/ranking.
i = find(lines, "    accepted = 0", find(lines, "def main() -> int:"))
lines.insert(i + 1, "    search_started = time.perf_counter()")
i = find(lines, '        f"Generated {generated:,} exact word set(s); "', i)
print_start = i - 1
assert lines[print_start] == "    print("
assert lines[i + 1] == '        f"{accepted:,} survived clue constraints.",'
lines[print_start:i + 2] = [
    "    search_seconds = time.perf_counter() - search_started",
    "    print(",
    '        f"Generated {generated:,} exact word set(s); "',
    '        f"{accepted:,} survived clue constraints. "',
    '        f"Exact search: {search_seconds:.2f}s.",',
]

path.write_text("\n".join(lines) + "\n", encoding="utf-8")

Path("tests/test_generator_search_optimization.py").write_text(
    '''from __future__ import annotations

import unittest

import anagram_generate as generator


def sig(a=0, b=0, c=0):
    return (a, b, c) + (0,) * 23


class SearchOptimizationTests(unittest.TestCase):
    def test_historical_exhaustive_order_is_preserved(self) -> None:
        candidates = [
            generator.Candidate("ab", sig(1, 1), 2, 5.0),
            generator.Candidate("ac", sig(1, 0, 1), 2, 4.9),
            generator.Candidate("bc", sig(0, 1, 1), 2, 4.8),
            generator.Candidate("abc", sig(1, 1, 1), 3, 4.7),
        ]
        remaining = sig(2, 2, 2)
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 3, 0, True)),
            [("abc", "abc"), ("ab", "ac", "bc")],
        )
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 3, 0, False)),
            [("ab", "ac", "bc")],
        )

    def test_historical_bounded_prefix_is_preserved(self) -> None:
        candidates = [
            generator.Candidate("a", sig(1), 1, 5.0),
            generator.Candidate("b", sig(0, 1), 1, 4.9),
            generator.Candidate("ab", sig(1, 1), 2, 4.8),
        ]
        remaining = sig(2, 2)
        expected = [("ab", "ab"), ("a", "b", "ab"), ("a", "a", "b", "b")]
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 4, 0, True)),
            expected,
        )
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                self.assertEqual(
                    list(generator.solve(remaining, candidates, 2, 4, limit, True)),
                    expected[:limit],
                )
''',
    encoding="utf-8",
)
