"""Real-corpus native ranked CLI versus the frozen Python export pipeline."""
from contextlib import closing, redirect_stdout
from dataclasses import asdict
import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import sqlite3
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "archive"));sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_generate as generator
import anagram_rerank as ranking
from scoring import close

def main():
    dictionary=ROOT/".anagram_data/dictionary/large.txt"
    one=ROOT/".anagram_data/ngrams/count_1w.txt";two=one.with_name("count_2w.txt")
    wordnet=ROOT/".anagram_data/wordnet31/dict"
    identity_paths={"dictionary":dictionary,"unigrams":one,"bigrams":two}
    identity_paths.update({f"wordnet/{name}":wordnet/name for name in ("data.verb","index.noun","index.verb","index.adj","index.adv","noun.exc","verb.exc")})
    identities={role:dict(role=role,representation="file_bytes",present=path.exists(),sha256=hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest(),bytes=path.stat().st_size if path.exists() else 0) for role,path in identity_paths.items()}
    unigrams=generator.load_unigram_model(one)
    lex=ranking.WordNetLexicon.load(wordnet)
    subprocess.run(["cargo","build","--locked","-p","anagram-cli"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug"/("anagram-cli.exe" if sys.platform=="win32" else "anagram-cli")
    with tempfile.TemporaryDirectory(prefix="solve-parity-",dir=ROOT/".codex/temp") as temp:
        phrase_path=Path(temp)/"phrase.sqlite"
        with closing(sqlite3.connect(phrase_path)) as connection, connection:
            connection.execute("CREATE TABLE ngrams(text TEXT PRIMARY KEY,n INTEGER,count INTEGER)")
            connection.executemany("INSERT INTO ngrams VALUES (?,?,?)",[("eat tea",2,10000),("eat ate",2,10),("eat eat",2,1)])
            schema=connection.execute("SELECT sql FROM sqlite_schema WHERE name='ngrams'").fetchone()[0]
            canonical="".join(json.dumps(row,separators=(",",":"))+"\n" for row in [("schema",schema),*connection.execute("SELECT text,n,count FROM ngrams ORDER BY text COLLATE BINARY,n,count")]).encode()
        phrase_identity=dict(role="phrase_index",representation="phrase_rows_v1",present=True,bytes=len(canonical),sha256=hashlib.sha256(canonical).hexdigest())
        checked_hard_deep_limit=False
        for text,required,hints,with_phrase in (("knowledgeispower",[],[],False),("thesehipsdontlie",[],["hips"],False),("ateate",["eat"],[],False),("testing",[],[],False),("ateate",["eat"],[],True)):
            if "--phrase-only" in sys.argv and not with_phrase: continue
            for strategy in ("prefix","diverse"):
                g=dict(schema_version=1,text=text,required=required,hints=hints,excluded=[],min_words=1,max_words=4,min_word_length=3,max_word_length=30,min_zipf=2.7,candidate_budget=20,allow_repeat=True,strategy=strategy,hint_mode="any")
                request=dict(generation=g,deep_per_group=3,deep_all=False,order_mode="auto",beam_width=32,exact_max_words=5,retained_orders=56,phrase_rescore_top=2,phrase_bonus_max=5.0,positive_bigrams=True,result_limit_per_group=5)
                remaining=generator.counts(text)
                for word in required: remaining=generator.subtract_counts(remaining,generator.counts(word))
                short=set(generator.DEFAULT_SHORT_WORDS)|{w for w in required+hints if len(w)<=2}
                candidates=generator.load_words(dictionary,remaining,3,30,set(),[],set(),2.7,"common",short,set(hints),unigrams)
                vocabulary={c.word for c in candidates}|set(required)|set(hints)
                bigrams=generator.load_bigram_model(two,unigrams,vocabulary)
                stats=generator.SearchStats()
                solutions=list(generator.search_solutions(remaining,candidates,max(1,1-len(required)),4-len(required),20,True,clue_words=set(hints),initial_clue_words=set(required)&set(hints),hint_mode="any",strategy=strategy,stats=stats))
                records=generator.build_records(solutions,required,set(hints),"any",unigrams,bigrams,vocabulary,short)
                export=Path(temp)/"candidates.txt"
                generator.write_full_export(export,records,True)
                rows=ranking.parse_candidates(export)
                with redirect_stdout(io.StringIO()):
                    ranking.prepare_rows(rows,lex)
                    selected=ranking.choose_deep(rows,3,False)
                    deep=ranking.deep_analyze(rows,selected,lex,wordnet_dir=wordnet,backend="serial",workers=1,batch_size=64,order_mode="auto",beam_width=32,exact_max_words=5)
                    colloc=ranking.load_positive_bigram_model(one,two,vocabulary)
                    phrase_index=ranking.PhraseIndex.open(phrase_path) if with_phrase else None
                    try:
                        rescored=ranking.apply_phrase_rescore(rows,collocation=colloc,phrase_index=phrase_index,top_per_group=2,bonus_max=5.0)
                    finally:
                        if phrase_index is not None: phrase_index.connection.close()
                buckets={str(wc):[asdict(row) for row in bucket[:5]] for wc,bucket in ranking.rank_buckets(rows).items()}
                command=[str(executable),"solve",str(dictionary),str(one),str(two),str(wordnet)]
                if with_phrase: command.append(str(phrase_path))
                result=subprocess.run([*command,"--progress"],input=json.dumps(request),text=True,capture_output=True,check=True,timeout=120)
                actual=json.loads(result.stdout)
                status=actual.pop("status")
                events=[json.loads(line) for line in result.stderr.splitlines()]
                assert events[0]["state"]=="queued" and events[-1]==status
                assert status["state"]=="succeeded" and status["stage"]=="complete"
                assert status["versions"]["data_complete"]
                expected_identities={**identities,**({"phrase_index":phrase_identity} if with_phrase else {})}
                assert {item["role"]:item for item in status["versions"]["data"]}==expected_identities
                for key in ("generated","deep_analyzed","shown","orders_evaluated","corpus_rescored"):
                    assert status["counts"][key]==actual[key],key
                assert status["counts"]["deep_selected"]==len(selected)
                assert status["budgets"]["candidate_limit"]==20
                assert status["exhaustion"]==("truncated" if stats.truncated else "exhausted")
                expected=dict(schema_version=1,kind="ranked",engine_version="0.1.0",normalized_input=text,generated=len(records),deep_analyzed=sum(r.deep for r in rows),shown=sum(map(len,buckets.values())),orders_evaluated=int(deep["orders"]),corpus_rescored=rescored,generation_stop="candidate_cap" if stats.truncated else "exhausted",buckets=buckets)
                try:close(actual,expected)
                except AssertionError:
                    (ROOT/".codex/temp/solve-mismatch.json").write_text(json.dumps(dict(request=request,actual=actual,expected=expected),indent=2),encoding="utf-8")
                    raise
                print(f"Ranked CLI parity passed: {text} / {strategy} / phrase={with_phrase}")
                if with_phrase:
                    cached_command=[*command,"--cache",str(Path(temp)/f"{strategy}-cache.sqlite")]
                    original_ranking_ms=None
                    for cache_mode in ("cold","warm","rebuild"):
                        cached_run=subprocess.run([*cached_command, *(["--rebuild-cache"] if cache_mode=="rebuild" else [])],input=json.dumps(request),text=True,capture_output=True,check=True,timeout=120)
                        cached=json.loads(cached_run.stdout)
                        cache_status=cached.pop("status")
                        timings=cached.pop("timings")
                        assert cache_status["cache"]["hit"]==(cache_mode=="warm")
                        assert cache_status["cache"]["rebuilt"]==(cache_mode=="rebuild")
                        assert isinstance(timings["execution_ms"],int) and timings["execution_ms"]>=0
                        if cache_mode=="cold": original_ranking_ms=timings["ranking_computation_ms"]
                        if cache_mode=="warm": assert timings["ranking_computation_ms"]==original_ranking_ms
                        close(cached,expected)
                    print(f"Ranked CLI cache parity passed: {text} / {strategy} / cold,warm,rebuild")
                if not checked_hard_deep_limit and len(selected)>1:
                    policy=Path(temp)/"policy.json"
                    policy.write_text(json.dumps(dict(max_deep_analyzed=1)),encoding="utf-8")
                    denied=subprocess.run([str(executable),"solve",str(dictionary),str(one),str(two),str(wordnet),"--limits",str(policy)],input=json.dumps(request),text=True,capture_output=True,timeout=120)
                    assert denied.returncode==2,denied.stdout
                    assert json.loads(denied.stdout)["error"]["code"]=="deployment_limit_exceeded",denied.stdout
                    checked_hard_deep_limit=True
        assert checked_hard_deep_limit
    print(f"Real-corpus ranked CLI parity passed: {2 if '--phrase-only' in sys.argv else 10} cases including optional SQLite")
if __name__=="__main__":main()
