"""Explicit learned-feature schema, model validation, scores and deterministic ties."""
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from anagram_feature_ranker import FEATURE_NAMES,LinearRankModel,RankItem,explicit_order_features,_rank
from scoring import close

def main():
    temp=tempfile.TemporaryDirectory(prefix="learned-parity-",dir=ROOT/".codex/temp")
    rng=random.Random(917)
    cases,expected=[],[]
    for i in range(240):
        candidate=dict(order=[],grammar_raw=0.0,phrase_kind="unknown",**{key:rng.uniform(-0.5,1.5) for key in ("grammar_norm","structure_norm","valency_norm","syntax_coverage","objective")})
        details={key:rng.uniform(-1,2) for key in ("whole_count","longer","bigram_coverage","cohesion","cohesion_coverage","cohesion_longest_fraction","cohesion_frequency","cohesion_splice_penalty")}
        details["cohesion_segments"]=rng.randrange(5)
        if i%5==0:details={}
        phrase=rng.random();wc=i%12
        model=LinearRankModel(tuple(rng.uniform(-10,10) if i%4 else 0.0 for _ in FEATURE_NAMES))
        features=explicit_order_features(**{k:candidate[k] for k in ("grammar_norm","structure_norm","valency_norm","syntax_coverage","objective")},phrase_score=phrase,phrase_details=details,word_count=wc)
        items=[RankItem(str(j),tuple(rng.randrange(3)/2 for _ in FEATURE_NAMES),False,rng.randrange(3)/2) for j in range(15)]
        cases.append(dict(candidate=candidate,details=details,phrase=phrase,word_count=wc,model=model.to_dict(),items=[dict(key=item.key,features=item.features,baseline_score=item.baseline_score) for item in items]))
        expected.append(dict(features=features,score=model.score(features),rank=[int(item.key) for item in _rank(items,model)],baseline=[int(item.key) for item in _rank(items,None)],model=model.to_dict()))
    valid=cases[0]["model"]
    for model in (None,{},dict(valid,schema="future"),dict(valid,features=list(reversed(FEATURE_NAMES))),dict(valid,weights=[0]),dict(valid,weights=[True]*18),dict(valid,weights=["0"]*18)):
        cases.append(dict(model=model));expected.append(dict(error=True))
    for raw in ("{broken",json.dumps(dict(valid,weights=[float("nan")]*18)),json.dumps(dict(valid,weights=[float("inf")]*18))):
        cases.append(dict(model_text=raw));expected.append(dict(error=True))
    missing=Path(temp.name)/"missing.json"
    oversized=Path(temp.name)/"oversized.json"
    oversized.write_bytes(b" "*65537)
    for path in (missing,oversized):
        cases.append(dict(model_path=str(path)));expected.append(dict(error=True))
    model_path=Path(temp.name)/"model.json"
    model_path.write_text(json.dumps(cases[1]["model"]),encoding="utf-8")
    cases.append(dict(cases[1],model_path=str(model_path)));expected.append(expected[1])
    subprocess.run(["cargo","build","--locked","--example","learned_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("learned_probe.exe" if sys.platform=="win32" else "learned_probe")
    output=subprocess.run([str(executable)],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=30)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for case,got,want in zip(cases,actual,expected):
        try:close(got,want)
        except AssertionError as error:raise AssertionError((case,error)) from error
    print(f"Learned ranker parity passed: {len(cases)} feature/model/ranking cases")
    temp.cleanup()
if __name__=="__main__":main()
