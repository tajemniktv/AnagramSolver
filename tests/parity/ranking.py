"""Preparation, family expansion, deep components and stable final bucket parity."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_rerank as reference
from scoring import close


def main():
    lex=reference.WordNetLexicon.load(ROOT/".anagram_data/wordnet31/dict")
    rng=random.Random(917)
    phrases=["", "dog", "dogs", "knowledge is power", "these hips dont lie", "these hips dont lies",
             "the bird is flying", "the birds are flying", "the engine is repaired", "they have been running",
             "actions speak louder than words", "a a dog", "united we stand divided we fall", "xxyy zzxx aabb"]
    cases, expected=[],[]
    for mode in ("auto","exact","beam"):
        for per_group,deep_all in ((0,False),(1,False),(3,False),(1,True)):
            rows=[]
            for i,p in enumerate(phrases):
                words=p.split()
                rng.shuffle(words)
                rows.append(reference.Row(tuple(words),len(words),i+1,50.0,rng.randrange(5)/4,rng.randrange(5)/4,0.25,0.5,3.0,2.0,0.0,()))
            rng.shuffle(rows)
            cases.append(dict(rows=[{k:v for k,v in asdict(row).items() if k in ("words","word_count","old_rank","old_pre","lex","fam","old_pair","hint","zavg","zmin","old_pcov","hints")} for row in rows],mode=mode,per_group=per_group,deep_all=deep_all))
            with redirect_stdout(io.StringIO()):
                reference.prepare_rows(rows,lex)
                prepared=[asdict(row) for row in rows]
                selected=reference.choose_deep(rows,per_group,deep_all)
                stats=reference.deep_analyze(rows,selected,lex,wordnet_dir=ROOT/".anagram_data/wordnet31/dict",backend="serial",workers=1,batch_size=64,order_mode=mode,beam_width=32,exact_max_words=5)
            ids={id(row):i for i,row in enumerate(rows)}
            alternatives=[[asdict(candidate) for candidate in reference.impl._ORDER_CANDIDATES_BY_ROW_ID.get(id(row),())] for row in rows]
            expected.append(dict(prepared=prepared,selected=sorted(selected),rows=[asdict(row) for row in rows],evaluated=int(stats["orders"]),buckets={str(k):[ids[id(row)] for row in bucket] for k,bucket in reference.rank_buckets(rows).items()},alternatives=alternatives))
            reference._clear_order_side_tables()
    subprocess.run(["cargo","build","--locked","--example","ranking_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("ranking_probe.exe" if sys.platform=="win32" else "ranking_probe")
    output=subprocess.run([str(executable),str(ROOT/".anagram_data/wordnet31/dict")],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=120)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for i,(got,want) in enumerate(zip(actual,expected)):
        try: close(got,want)
        except AssertionError as error:
            (ROOT/".codex/temp/ranking-mismatch.json").write_text(json.dumps(dict(case=cases[i],got=got,want=want),indent=2),encoding="utf-8")
            raise AssertionError((i,error)) from error
    print(f"Final base-ranking parity passed: {len(cases)} complete preparation/shortlist/deep/bucket cases")


if __name__=="__main__":main()
