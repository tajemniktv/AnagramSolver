"""Exact and k-best native order-pool parity before diversity/refinement."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_rerank_core as core
import anagram_rerank_topk_impl as reference
import anagram_auxiliary_grammar as auxiliary
from scoring import close


def main():
    lex=core.WordNetLexicon.load(ROOT/".anagram_data/wordnet31/dict")
    reference.phrase_structure=lambda words,lex: auxiliary.phrase_structure_with_auxiliaries(words,lex,core.phrase_structure)
    reference.local_grammar_raw=lambda words,lex: auxiliary.local_grammar_raw_with_auxiliaries(words,lex,core._order_local_tables)
    reference._order_local_tables=lambda words,lex: auxiliary.order_local_tables_with_auxiliaries(words,lex,core._order_local_tables)
    phrases=["", "dog", "knowledge is power", "these hips dont lie", "the birds are flying",
             "the engine is repaired", "they have been running", "actions speak louder than words",
             "a a dog", "united we stand divided we fall", "the cat is being tested", "xxyy zzxx aabb"]
    cases=[dict(words=p.split(),exact=exact) for p in phrases for exact in (True,False)]
    expected=[]
    for case in cases:
        orders,evaluated=reference.rank_orders(case["words"],lex,order_mode="exact" if case["exact"] else "beam",beam_width=32,top_k=8)
        expected.append(dict(orders=[asdict(order) for order in orders],evaluated=evaluated))
    subprocess.run(["cargo","build","--locked","--example","ordering_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("ordering_probe.exe" if sys.platform=="win32" else "ordering_probe")
    output=subprocess.run([str(executable),str(ROOT/".anagram_data/wordnet31/dict")],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=120)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for case,got,want in zip(cases,actual,expected):
        try: close(got,want)
        except AssertionError as error: raise AssertionError((case,error)) from error
    print(f"Retained-order pool parity passed: {len(cases)} exact/beam cases")


if __name__=="__main__":main()
