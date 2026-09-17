"""Phrase hierarchy and non-overlapping cohesion parity including missing data."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "archive"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from anagram_corpus_cohesion import score_corpus_cohesion, blend_phrase_cohesion
from anagram_rerank_core import PhraseIndex
from scoring import close

class Index:
    def __init__(self, counts,max_n): self.data,self.max_n=counts,max_n
    def counts(self,queries): return {q:self.data[q] for q in queries if q in self.data}

def main():
    rng=random.Random(917)
    cases,expected=[],[]
    for i in range(400):
        words=[rng.choice(("a","b","c","d","")) for _ in range(i%9)]
        counts={" ".join(words[start:start+n]):rng.choice((-1,0,1,10,1000,1000000)) for n in range(1,len(words)+1) for start in range(len(words)-n+1) if rng.random()<0.6}
        max_n=i%10
        index=Index(counts,max_n)
        cohesion=score_corpus_cohesion(words,counts=index.counts,max_n=max_n)
        phrase=PhraseIndex.score(index,words)
        details=dict(phrase[1],cohesion=cohesion.score,cohesion_coverage=cohesion.coverage,
                     cohesion_longest_fraction=cohesion.longest_fraction,cohesion_segments=float(cohesion.segments),
                     cohesion_splice_penalty=cohesion.splice_penalty,cohesion_frequency=cohesion.frequency_strength)
        expected.append(dict(cohesion=asdict(cohesion),phrase=phrase,blended=(blend_phrase_cohesion(phrase[0],cohesion),details)))
        cases.append(dict(words=words,counts=counts,max_n=max_n))
    subprocess.run(["cargo","build","--locked","--example","phrase_probe"],cwd=ROOT,check=True)
    executable=ROOT/"target/debug/examples"/("phrase_probe.exe" if sys.platform=="win32" else "phrase_probe")
    output=subprocess.run([str(executable)],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=120)
    actual=[json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual)==len(expected)
    for case,got,want in zip(cases,actual,expected):
        try: close(got,want)
        except AssertionError as error: raise AssertionError((case,error)) from error
    print(f"Phrase/cohesion parity passed: {len(cases)} seeded corpus cases")

if __name__=="__main__":main()
