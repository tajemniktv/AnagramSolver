"""Differential k-opt budgets, duplicate seeds, ties and multi-round endpoints."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from anagram_order_refinement import refine_order, refine_seed_pool, augment_seed_pool
from scoring import close


def main():
    rng = random.Random(917)
    cases, expected = [], []
    for index in range(160):
        n = index % 8
        seeds = [[str(rng.randrange(4)) for _ in range(n)] for _ in range(4)]
        seeds.append(seeds[0])
        scale = (1.0, 1e-12, 0.0)[index % 3]
        weights = [[rng.randrange(-10, 11)*scale for _ in range(4)] for _ in range(n)]
        budget = (1, 2, 12, 128, 512)[index % 5]
        rounds, seed_limit = index % 4, 1 + index % 6
        # Give both black-box refiners identical sequential floating arithmetic;
        # Python's built-in sum uses compensation on recent interpreters.
        def scorer(order):
            score = 0.0
            for i,w in enumerate(order): score += weights[i][int(w)]
            return score
        options = dict(max_rounds=rounds, max_evaluations_per_seed=budget, seed_limit=seed_limit)
        expected.append(dict(single=asdict(refine_order(seeds[0],scorer,max_rounds=rounds,max_evaluations=budget)),
                             pool=[asdict(r) for r in refine_seed_pool(seeds,scorer,**options)],
                             augmented=asdict(augment_seed_pool(seeds,scorer,**options))))
        cases.append(dict(seeds=seeds,weights=weights,budget=budget,rounds=rounds,seed_limit=seed_limit))
    subprocess.run(["cargo","build","--locked","--example","refinement_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("refinement_probe.exe" if sys.platform=="win32" else "refinement_probe")
    output=subprocess.run([str(executable)],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=120)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for case,got,want in zip(cases,actual,expected):
        try: close(got,want)
        except AssertionError as error: raise AssertionError((case,error)) from error
    print(f"Refinement parity passed: {len(cases)} bounded single/refined/augmented pools")


if __name__ == "__main__": main()
