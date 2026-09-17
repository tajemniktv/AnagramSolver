"""Serial synthetic generation parity/time/RSS baseline; not a full migration gate."""
import hashlib
import itertools
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".codex/temp"
sys.path.insert(0, str(ROOT / "archive"))


def reference_worker():
    import anagram_generate as ref
    for line in sys.stdin:
        case = json.loads(line)
        candidates = [ref.Candidate(w, ref.counts(w), len(w), 0) for w in case["candidates"]]
        stats = ref.SearchStats()
        bags = list(ref.search_solutions(ref.counts(case["text"]), candidates,
            case["min_words"], case["max_words"], case["cap"], case["repeat"],
            clue_words=set(case["clues"]), initial_clue_words=set(case["initial"]),
            hint_mode=case["mode"], strategy=case["strategy"], stats=stats))
        print(json.dumps(dict(bags=bags, truncated=stats.truncated)))


def measure(command, payload, *, process_tree=False, timeout=120, json_lines=True):
    sys.path.insert(0, str(TEMP / "reference-runtime"))
    import psutil
    with tempfile.TemporaryFile(dir=TEMP) as source, tempfile.TemporaryFile(dir=TEMP) as output, tempfile.TemporaryFile(dir=TEMP) as errors:
        source.write(payload)
        source.seek(0)
        start = time.perf_counter()
        process = subprocess.Popen(command, stdin=source, stdout=output, stderr=errors, cwd=ROOT)
        root = psutil.Process(process.pid)
        peak = 0
        try:
            while process.poll() is None:
                try:
                    members = [root, *root.children(recursive=True)] if process_tree else [root]
                    rss = 0
                    for member in members:
                        try: rss += member.memory_info().rss
                        except (psutil.NoSuchProcess, psutil.AccessDenied): pass
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    pass
                if time.perf_counter() - start > timeout:
                    raise TimeoutError(f"process exceeded {timeout} seconds")
                time.sleep(0.002)
        finally:
            if process.poll() is None:
                if process_tree:
                    try: children = root.children(recursive=True)
                    except psutil.NoSuchProcess: children = []
                    for child in reversed(children):
                        try: child.kill()
                        except psutil.NoSuchProcess: pass
                process.kill()
                process.wait()
        elapsed = time.perf_counter() - start
        errors.seek(0)
        if process.returncode:
            raise RuntimeError(errors.read().decode(errors="replace"))
        output.seek(0)
        result = [json.loads(line) for line in output] if json_lines else [json.load(output)]
        return result, dict(seconds=elapsed, sampled_peak_rss_bytes=peak)


def main():
    TEMP.mkdir(parents=True, exist_ok=True)
    real = "--real-corpus" in sys.argv
    corpus_identities = {}
    words = ["".join(w) for n in range(1, 4) for w in itertools.product("abc", repeat=n)]
    cases = [dict(text="aaabbbccc", candidates=words, min_words=2, max_words=6,
                  cap=2000, repeat=True, clues=clues, initial=[], mode=mode, strategy=strategy)
             for strategy in ("prefix", "diverse")
             for clues, mode in (([], "any"), (["abc", "cba"], "exactly-one"))]
    if real:
        import anagram_generate as ref
        dictionary = ROOT / ".anagram_data/dictionary/large.txt"
        frequency = ROOT / ".anagram_data/ngrams/count_1w.txt"
        for role, path in (("dictionary", dictionary), ("unigrams", frequency)):
            corpus_identities[role] = hashlib.sha256(path.read_bytes()).hexdigest()
        unigrams = ref.load_unigram_model(frequency)
        cases = []
        for text in ("knowledgeispower", "thesehipsdontlie", "testing", "ateate"):
            admitted = ref.load_words(dictionary, ref.counts(text), 3, 30, set(), [],
                set(), 2.7, "common", ref.DEFAULT_SHORT_WORDS, set(), unigrams)
            for strategy in ("prefix", "diverse"):
                cases.append(dict(text=text, candidates=[c.word for c in admitted],
                    min_words=1, max_words=4, cap=200, repeat=True, clues=[],
                    initial=[], mode="any", strategy=strategy))
    payload = ("\n".join(map(json.dumps, cases)) + "\n").encode()
    subprocess.run(["cargo", "build", "--release", "--locked", "--example", "generation_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/release/examples" / ("generation_probe.exe" if sys.platform == "win32" else "generation_probe")
    commands = {"python": [sys.executable, str(Path(__file__).resolve()), "--worker"], "rust": [str(executable)]}
    runs = []
    expected = None
    for sample in range(3):
        for engine in (("python", "rust") if sample % 2 == 0 else ("rust", "python")):
            result, metrics = measure(commands[engine], payload)
            if expected is None:
                expected = result
            assert result == expected, f"ordered parity failed: {engine}, sample {sample}"
            runs.append(dict(engine=engine, sample=sample, **metrics))
    report_cases = cases
    if real:
        # Corpus redistribution is not approved; record identities, not word lists.
        report_cases = [{**{k: v for k, v in case.items() if k != "candidates"},
                         "candidate_count": len(case["candidates"]),
                         "candidates_sha256": hashlib.sha256(json.dumps(case["candidates"]).encode()).hexdigest()}
                        for case in cases]
    report = dict(scope=f"{len(cases)} {'real-corpus-admitted' if real else 'synthetic'} generation cases per fresh process; three serial samples per engine",
        limitations="includes startup, Python imports, JSON I/O and candidate setup; 2ms sampled root RSS can miss peaks; OS caches not flushed; no corpus, ranking or application performance claim",
        platform=platform.platform(), python=sys.version, input_sha256=hashlib.sha256(payload).hexdigest(),
        rust_binary_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(), cases=report_cases,
        corpus_identities=corpus_identities,
        reference_source_sha256=hashlib.sha256((ROOT / "archive/anagram_generate.py").read_bytes()).hexdigest(),
        output_counts=[len(r["bags"]) for r in expected], runs=runs)
    path = ROOT / "tests/reference" / ("native-generation-real.json" if real else "native-generation-synthetic.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(report=str(path), runs=runs, output_counts=report["output_counts"]), indent=2))


if __name__ == "__main__":
    reference_worker() if "--worker" in sys.argv else main()
