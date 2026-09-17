"""Exact and k-best native order-pool parity before diversity/refinement."""
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "archive"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_rerank_core as core
import anagram_rerank_topk_impl as reference
import anagram_auxiliary_grammar as auxiliary
from anagram_order_diversity import raw_pool_size, select_diverse_orders
from scoring import close
from anagram_suite import cases_for
from anagram_benchmark import tokens


def main():
    lex=core.WordNetLexicon.load(ROOT/".anagram_data/wordnet31/dict")
    reference.phrase_structure=lambda words,lex: auxiliary.phrase_structure_with_auxiliaries(words,lex,core.phrase_structure)
    reference.local_grammar_raw=lambda words,lex: auxiliary.local_grammar_raw_with_auxiliaries(words,lex,core._order_local_tables)
    reference._order_local_tables=lambda words,lex: auxiliary.order_local_tables_with_auxiliaries(words,lex,core._order_local_tables)
    phrases=["", "dog", "knowledge is power", "these hips dont lie", "the birds are flying",
             "the engine is repaired", "they have been running", "actions speak louder than words",
             "a a dog", "united we stand divided we fall", "the cat is being tested", "xxyy zzxx aabb"]
    cases=[dict(words=p.split(),exact=exact) for p in phrases for exact in (True,False)]
    cases += [dict(words=p.split(),exact=exact,top_k=k,diverse=True)
              for p in phrases if len(p.split()) >= 4
              for exact in (True,False) for k in (48,64,72)]
    registry = cases_for("ordering")
    cases += [dict(words=tokens(str(case["answer"])),
                   exact=len(tokens(str(case["answer"]))) <= 6,
                   top_k=50, beam_width=256, diverse=True, registry_id=case["id"])
              for case in registry]
    expected=[]
    for case in cases:
        k=case.get("top_k",8)
        orders,evaluated=reference.rank_orders(case["words"],lex,order_mode="exact" if case["exact"] else "beam",beam_width=case.get("beam_width",32),top_k=raw_pool_size(k) if case.get("diverse") else k)
        if case.get("diverse"): orders=select_diverse_orders(orders,k)
        expected.append(dict(orders=[asdict(order) for order in orders],evaluated=evaluated))
    subprocess.run(["cargo","build","--locked","--example","ordering_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("ordering_probe.exe" if sys.platform=="win32" else "ordering_probe")
    output=subprocess.run([str(executable),str(ROOT/".anagram_data/wordnet31/dict")],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=120)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for case,got,want in zip(cases,actual,expected):
        try: close(got,want)
        except AssertionError as error:
            destination=ROOT/".codex/temp/ordering-mismatch.json"
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(json.dumps(dict(case=case,actual=got,expected=want),indent=2),encoding="utf-8")
            raise AssertionError((case,error,str(destination))) from error
    print(f"Retained-order pool parity passed: {len(cases)} exact/beam cases")


if __name__=="__main__":main()
