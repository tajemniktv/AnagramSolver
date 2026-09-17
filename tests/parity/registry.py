"""Native ranked CLI at all five frozen normal-user workloads (no reduced caps)."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive"))
sys.path.insert(0, str(ROOT / "tools"))
from reference_fixtures import verify_sources
import anagram_solver as oracle
from anagram_rerank_core import pretty_phrase


def native_request(case, policy):
    options = oracle.build_parser().parse_args([case["target"], *case["options"]])
    return dict(generation=dict(schema_version=1, text=case["target"], required=[], hints=[], excluded=[],
        extra_short_words=policy["extra_short_words"], min_words=options.words or options.min_words,
        max_words=options.words or options.max_words, min_word_length=options.min_word_len, max_word_length=99,
        min_zipf=options.min_zipf, candidate_budget=case["cold"]["search"]["cap"], allow_repeat=True,
        strategy=options.search_strategy, hint_mode="any"), deep_per_group=5000, deep_all=False,
        order_mode="auto", beam_width=128, exact_max_words=5, retained_orders=options.order_candidates,
        phrase_rescore_top=300, phrase_bonus_max=5.0, positive_bigrams=True, result_limit_per_group=options.top)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    verify_sources()
    cases = json.loads((ROOT / "tests/reference/registry.json").read_text())["normal_user_cli"]
    if args.case:
        if not set(args.case) <= {case["id"] for case in cases}:
            parser.error("Unknown registry case")
        cases = [case for case in cases if case["id"] in args.case]
    subprocess.run(["cargo", "build", "--release", "--locked", "-p", "anagram-cli"], cwd=ROOT, check=True)
    executable = ROOT / "target/release" / ("anagram-cli.exe" if sys.platform == "win32" else "anagram-cli")
    policy = json.loads((ROOT / ".anagram_data/dictionary/normal_user_v2.json").read_text())
    command = [str(executable), "solve", str(ROOT / ".anagram_data/dictionary/normal_user_v2.txt"),
               str(ROOT / ".anagram_data/ngrams/count_1w.txt"), str(ROOT / ".anagram_data/ngrams/count_2w.txt"),
               str(ROOT / ".anagram_data/wordnet31/dict")]
    for case in cases:
        request = native_request(case, policy)
        with tempfile.TemporaryDirectory(prefix="native-registry-", dir=ROOT / ".codex/temp") as temporary:
            cold = None
            for mode in ("cold", "warm"):
                completed = subprocess.run([*command, "--cache", str(Path(temporary) / "cache.sqlite")],
                    input=json.dumps(request), text=True, capture_output=True, timeout=600, check=True)
                result = json.loads(completed.stdout)
                displayed = [dict(word_count=int(wc), rank=index, score=float(f"{row['final']:.2f}"),
                                  phrase=pretty_phrase(row["best_order"]))
                             for wc, rows in sorted(result["buckets"].items(), key=lambda pair: int(pair[0]))
                             for index, row in enumerate(rows, 1)]
                expected = case["cold"]
                observed = dict(results=displayed, generated=result["generated"], deep_analyzed=result["deep_analyzed"],
                                truncated=result["generation_stop"] == "candidate_cap")
                wanted = dict(results=expected["results"], generated=expected["search"]["generated"],
                              deep_analyzed=expected["search"]["deep_analyzed"], truncated=expected["search"]["truncated"])
                if observed != wanted:
                    (ROOT / ".codex/temp/native-registry-mismatch.json").write_text(json.dumps(
                        dict(case=case["id"], mode=mode, request=request, observed=observed, expected=wanted), indent=2))
                    raise AssertionError(f"Registry mismatch: {case['id']} / {mode}")
                assert result["status"]["cache"]["hit"] == (mode == "warm")
                if cold is None:
                    cold = result["buckets"]
                else:
                    if cold != result["buckets"]:
                        (ROOT / ".codex/temp/native-cache-roundtrip.json").write_text(json.dumps(dict(cold=cold, warm=result["buckets"]), indent=2))
                    assert cold == result["buckets"], "Cache changed exact row values"
                print(f"Native registry passed: {case['id']} / {mode} / generated={result['generated']}", flush=True)
    print(f"Native normal-user registry passed: {len(cases)} workloads, cold and warm")


if __name__ == "__main__":
    main()
