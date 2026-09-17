"""Preparation, family expansion, deep components and stable final bucket parity."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import subprocess
import sys
import sqlite3
import tempfile
import hashlib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_rerank as reference
from scoring import close


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase-db",type=Path)
    args=parser.parse_args()
    def digest(path):
        with path.open("rb") as stream: return hashlib.file_digest(stream,"sha256").hexdigest()
    real_database=args.phrase_db.resolve() if args.phrase_db else None
    real_digest=digest(real_database) if real_database else None
    temp=tempfile.TemporaryDirectory(prefix="ranking-parity-",dir=ROOT/".codex/temp")
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
            counts={p:10**rng.randrange(1,6) for p in phrases if p and rng.random()<0.7}
            for p in phrases:
                words=p.split()
                for n in (2,3):
                    for start in range(len(words)-n+1):
                        if rng.random()<0.6: counts[" ".join(words[start:start+n])]=rng.choice((0,1,100,100000))
            unigrams={w:rng.randrange(1,100000) for p in phrases for w in p.split()}
            bigrams={(a,b):rng.randrange(1,10000) for p in phrases for a,b in zip(p.split(),p.split()[1:])}
            total=10**9
            number=len(cases)-1
            use_collocation=number%4 in (1,3)
            use_phrase=number%4 in (2,3)
            top=(0,1,2,10)[number%4]
            bonus=(0.0,18.0,100.0)[number%3]
            model=reference.PositiveBigramModel(unigrams,bigrams,total) if use_collocation else None
            database=real_database or Path(temp.name)/f"phrase {number} # corpus.sqlite"
            if real_database is None:
                with sqlite3.connect(database) as connection:
                    connection.execute("CREATE TABLE ngrams (text TEXT PRIMARY KEY, n INTEGER, count INTEGER)")
                    connection.executemany("INSERT INTO ngrams VALUES (?, ?, ?)",[(text,len(text.split()),count) for text,count in counts.items()])
                    # Ensure corpus maximum order agrees with component fixtures.
                    connection.execute("INSERT INTO ngrams VALUES ('corpus maximum order sentinel never queried',6,1)")
                connection.close()
            before=real_digest or digest(database)
            index=reference.PhraseIndex.open(database) if use_phrase else None
            admission={}
            for wc in sorted({r.word_count for r in rows if r.deep}):
                chosen,added=reference.impl._select_phrase_rescore_rows([r for r in rows if r.deep and r.word_count==wc],collocation=model,phrase_index=index,top_per_group=top)
                admission[str(wc)]=([ids[id(r)] for r in chosen],added)
            rescored=reference.apply_phrase_rescore(rows,collocation=model,phrase_index=index,top_per_group=top,bonus_max=bonus)
            expected[-1]["corpus"]=dict(rows=[asdict(r) for r in rows],rescored=rescored,admission=admission,buckets={str(k):[ids[id(r)] for r in bucket] for k,bucket in reference.rank_buckets(rows).items()})
            cases[-1]["corpus"]=dict(counts=counts,unigrams=unigrams,bigrams=[(a,b,c) for (a,b),c in bigrams.items()],total=total,use_collocation=use_collocation,use_phrase=use_phrase,top=top,bonus=bonus)
            if index is not None: index.connection.close()
            cases[-1]["corpus"].update(path=str(database),sha256=before)
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
    for path, expected_digest in {(case["corpus"]["path"],case["corpus"]["sha256"]) for case in cases}:
        assert digest(Path(path))==expected_digest
    temp.cleanup()
    print(f"Ranking and corpus-rescoring parity passed: {len(cases)} complete pipeline cases")


if __name__=="__main__":main()
