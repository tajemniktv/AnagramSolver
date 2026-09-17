"""Real-corpus native ranked CLI versus the frozen Python export pipeline."""
from contextlib import redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
import anagram_generate as generator
import anagram_rerank as ranking
from scoring import close

def main():
    dictionary=ROOT/".anagram_data/dictionary/large.txt"
    one=ROOT/".anagram_data/ngrams/count_1w.txt";two=one.with_name("count_2w.txt")
    wordnet=ROOT/".anagram_data/wordnet31/dict"
    unigrams=generator.load_unigram_model(one)
    lex=ranking.WordNetLexicon.load(wordnet)
    subprocess.run(["cargo","build","--locked","-p","anagram-cli"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug"/("anagram-cli.exe" if sys.platform=="win32" else "anagram-cli")
    with tempfile.TemporaryDirectory(prefix="solve-parity-",dir=ROOT/".codex/temp") as temp:
        for text,required,hints in (("knowledgeispower",[],[]),("thesehipsdontlie",[],["hips"]),("ateate",["eat"],[]),("testing",[],[])):
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
                    rescored=ranking.apply_phrase_rescore(rows,collocation=colloc,phrase_index=None,top_per_group=2,bonus_max=5.0)
                buckets={str(wc):[asdict(row) for row in bucket[:5]] for wc,bucket in ranking.rank_buckets(rows).items()}
                result=subprocess.run([str(executable),"solve",str(dictionary),str(one),str(two),str(wordnet)],input=json.dumps(request),text=True,capture_output=True,check=True,timeout=120)
                actual=json.loads(result.stdout)
                expected=dict(schema_version=1,kind="ranked",engine_version="0.1.0",normalized_input=text,generated=len(records),deep_analyzed=sum(r.deep for r in rows),shown=sum(map(len,buckets.values())),orders_evaluated=int(deep["orders"]),corpus_rescored=rescored,generation_stop="candidate_cap" if stats.truncated else "exhausted",buckets=buckets)
                try:close(actual,expected)
                except AssertionError:
                    (ROOT/".codex/temp/solve-mismatch.json").write_text(json.dumps(dict(request=request,actual=actual,expected=expected),indent=2),encoding="utf-8")
                    raise
                print(f"Ranked CLI parity passed: {text} / {strategy}")
    print("Real-corpus ranked CLI parity passed: 8 cases")
if __name__=="__main__":main()
