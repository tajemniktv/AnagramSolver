"""Read-only SQLite batching/errors and distinct positive-corpus loading semantics."""
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / "archive"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from anagram_rerank_core import PhraseIndex,load_positive_bigram_model
from scoring import close

def main():
    with tempfile.TemporaryDirectory(prefix="corpus-storage-",dir=ROOT/".codex/temp") as temp:
        directory=Path(temp)
        database=directory/"corpus # ü.sqlite"
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE ngrams (text TEXT PRIMARY KEY, n INTEGER, count INTEGER)")
            connection.executemany("INSERT INTO ngrams VALUES (?,2,?)",[(f"phrase {i}",i) for i in range(501)])
        connection.close()
        digest=hashlib.sha256(database.read_bytes()).hexdigest()
        queries=[f"phrase {i}" for i in range(600)]+["","phrase 1","not present"]
        index=PhraseIndex.open(database)
        cases=[dict(path=str(database),queries=queries)]
        expected=[dict(max_n=index.max_n,counts=index.counts(queries))]
        index.connection.close()
        missing=directory/"missing.sqlite"
        corrupt=directory/"corrupt.sqlite"
        corrupt.write_bytes(b"not a database")
        no_table=directory/"no-table.sqlite"
        with sqlite3.connect(no_table) as connection: connection.execute("CREATE TABLE unrelated (id INTEGER)")
        connection.close()
        for path in (missing,corrupt,no_table):
            cases.append(dict(path=str(path),queries=queries));expected.append(dict(error=True))
        one="a\t100\nb\t200\nc\t300\nA\t20\n!!!\t90\nunknown\t1000\nbroken\n"
        two="a b\t10\nb c\t20\nA b\t5\nc a\t-1\na b c\t500\n"
        (directory/"one.txt").write_text(one,encoding="utf-8")
        (directory/"two.txt").write_text(two,encoding="utf-8")
        vocabulary={"a","b","c"}
        orders=[[],["a"],["a","b","c"],["c","a","b"],["a","a"],["b","c","a","b"]]
        model=load_positive_bigram_model(directory/"one.txt",directory/"two.txt",vocabulary)
        cases.append(dict(one=one,two=two,vocabulary=sorted(vocabulary),orders=orders))
        expected.append(dict(total=model.total_unigrams,unigrams=model.unigram_counts,scores=[model.score(order) for order in orders]))
        subprocess.run(["cargo","build","--locked","--example","corpus_probe"],cwd=ROOT,check=True)
        executable=ROOT/"target/debug/examples"/("corpus_probe.exe" if sys.platform=="win32" else "corpus_probe")
        output=subprocess.run([str(executable)],input="\n".join(map(json.dumps,cases)),text=True,capture_output=True,check=True,timeout=30)
        actual=[json.loads(line) for line in output.stdout.splitlines()]
        close(actual,expected)
        assert not missing.exists()
        assert hashlib.sha256(database.read_bytes()).hexdigest()==digest
    print("Corpus storage parity passed: batched SQLite counts, missing/corrupt/schema failures, immutable database, positive counts")

if __name__=="__main__":main()
