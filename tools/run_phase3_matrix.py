"""Run the unchanged reference corpus matrix with all scratch work project-local."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / ".codex/temp/phase3-corpora"
sys.path.insert(0, str(ROOT))


def main():
    if "--benchmark-worker" in sys.argv:
        sys.argv.remove("--benchmark-worker")
        import anagram_benchmark as benchmark
        original = benchmark.make_reranker_command

        def reranker(*args, **kwargs):
            return original(*args, **kwargs) + ["--prepared-cache-dir", str(TEMP / "prepared")]

        benchmark.make_reranker_command = reranker
        return benchmark.main()

    import ci_phrase_matrix as matrix
    matrix.RESULTS_DIR = TEMP / "matrix-results"
    original = matrix.run_logged

    def logged(label, command, log_path):
        assert Path(command[1]).name == "anagram_benchmark.py"
        command = [command[0], str(Path(__file__).resolve()), "--benchmark-worker", *command[2:],
                   "--cache-dir", str(TEMP / "benchmark-cache")]
        return original(label, command, log_path)

    matrix.run_logged = logged
    sys.argv = [sys.argv[0], "--wiktionary-db", str(TEMP / "wiktionary.db"),
                "--wikipedia-db", str(TEMP / "wiktionary-wikipedia.db"), "--workers", "1", *sys.argv[1:]]
    return matrix.main()


if __name__ == "__main__":
    raise SystemExit(main())
