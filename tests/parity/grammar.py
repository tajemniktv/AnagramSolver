"""Local grammar parity against Python's actual WordNet-backed scorer."""
import json
import inspect
from dataclasses import asdict
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anagram_rerank_core as reference
import anagram_auxiliary_grammar as auxiliary
import anagram_comparative_grammar as comparative
import anagram_clause_validity as validity
from scoring import close


def main():
    # Preserve the historical oracle on disk. These three reviewed bug fixes
    # deliberately improve native behavior; adapt only those reference helpers
    # for this differential gate, and include explicit regression phrases below.
    reference._comparative_like = lambda word, lex: comparative.comparative_evidence(word, lex).confidence >= 0.9
    edits = {
        "_comparative_span_starting_at": ("or any(lex.features(w).adj or lex.features(w).adv for w in left)", ""),
        "_np_span_starting_at": ("and words[i].endswith((\"ed\", \"en\")) and following_is_nominal", "and function_class(words[i]) is None and words[i].endswith((\"ed\", \"en\")) and following_is_nominal"),
    }
    for name, (before, after) in edits.items():
        source = inspect.getsource(getattr(reference, name))
        if before not in source:
            raise RuntimeError(f"Reference correction no longer applies: {name}")
        exec(compile(source.replace(before, after), f"<reviewed-reference-{name}>", "exec"), reference.__dict__)
    directory = ROOT / ".anagram_data/wordnet31/dict"
    lex = reference.WordNetLexicon.load(directory)
    rng = random.Random(917)
    vocabulary = sorted(reference.FUNCTION_WORDS | {"dog", "dogs", "cat", "cats", "run", "runs", "deep", "red", "turn", "give", "knowledge", "power", "quickly", "xyznonword"})
    cases = [rng.choices(vocabulary, k=rng.randint(0, 8)) for _ in range(250)]
    # Every function-word pair exercises class priority and collision rules.
    cases.append(vocabulary)
    cases.extend(phrase.split() for phrase in ["the young", "a professional", "the light faded",
                 "the repaired engine", "the mechanic repaired engines", "birds of a feather",
                 "mother of invention", "these old sheep", "a dogs", "school bus", "less is more",
                 "better late than never", "louder than words", "before you leap", "the bird sings",
                 "the pot boils", "turn red", "give him a book", "paint the wall red"])
    cases.extend(phrase.split() for phrase in ["closer than one", "number than ice", "happier than before",
                 "they are being tested", "he has been running", "they will have been being tested",
                 "we will not be tested", "she does not run", "he has not been running",
                 "they being tested", "he is runs", "the larger dog"])
    cases.extend(phrase.split() for phrase in ["a am sitting managers", "an game starting aims",
                 "one am here", "one is here", "a hour", "an university", "do run", "dont run",
                 "the cat is was running", "the bird singing", "a honest person"])
    cases.extend(phrase.split() for phrase in ["united we stand divided we fall", "actions speak louder than words",
                 "the engine is repaired by the mechanic quickly", "the engine is repaired by the mechanic dog",
                 "they have been being tested", "the birds are flying", "the bird are flying"])
    cases.extend(phrase.split() for phrase in ["old than words", "silver than words", "older than words", "the been dog", "the broken window"])
    subprocess.run(["cargo", "build", "--locked", "--example", "grammar_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/debug/examples" / ("grammar_probe.exe" if sys.platform == "win32" else "grammar_probe")
    result = subprocess.run([str(executable), str(directory)], input="\n".join(map(json.dumps, cases)),
                            text=True, capture_output=True, check=True, timeout=60)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(cases)
    for words, got in zip(cases, actual):
        expected = dict(classes=[reference.function_class(w) for w in words],
                        pairs=[[reference.pair_grammar(a,b,lex) for b in words] for a in words],
                        start=[reference.start_score(w,lex) for w in words],
                        end=[reference.end_score(w,lex) for w in words],
                        local=reference.local_grammar_raw(words,lex),
                        potential=reference.grammar_potential(words,lex), coverage=reference.content_coverage(words,lex),
                        np_start=[reference._np_span_starting_at(words,i,lex) for i in range(len(words))],
                        np_end=[reference._np_span_ending_at(words,i,lex) for i in range(len(words))],
                        comparative=[reference._comparative_span_starting_at(words,i,lex) for i in range(len(words))],
                        simple=[reference._simple_clause_span_starting_at(words,i,lex) for i in range(len(words))],
                        subordinate=[reference._subordinate_span_starting_at(words,i,lex) for i in range(len(words))],
                        numbers=[reference._subject_number(w,lex) for w in words],
                        agreement=[[reference._subject_agreement(w,v,lex,auxiliary=False) for v in
                                    ["is","are","dont","doesnt","runs","run","stopped","have"]] for w in words],
                        tails=[reference._valency_for_tail(v,words,lex) for v in ["run","give","speak","look","turn"]],
                        structure=asdict(reference.phrase_structure(words,lex)),
                        extended_structure=asdict(auxiliary.phrase_structure_with_auxiliaries(words,lex,reference.phrase_structure)),
                        extended_local=auxiliary.local_grammar_raw_with_auxiliaries(words,lex,reference._order_local_tables),
                        aux_structure=asdict(s) if (s:=auxiliary.auxiliary_structure(words,lex)) else None,
                        comparative_structure=asdict(s) if (s:=auxiliary.comparative_clause_structure(words,lex)) else None,
                        parallel_structure=asdict(s) if (s:=auxiliary.parallel_clause_structure(words,lex)) else None,
                        passive_tail=auxiliary._passive_tail(words,lex),
                        valid_subject=[validity.valid_subject_head(w,lex) for w in words],
                        finite_lexical=[validity.lexical_finite_form(w,lex) for w in words],
                        valid_pairs=[[validity.pair_validity_adjustment(a,b,lex) for b in words] for a in words],
                        valid_coverage=validity.best_valid_lexical_clause_coverage(words,lex),
                        adjusted=asdict(validity.adjust_base_clause_structure(words,lex,reference.phrase_structure(words,lex))),
                        surface=asdict(validity.apply_surface_structure_penalties(words,lex,reference.phrase_structure(words,lex))),
                        chains=[asdict(chain) if (chain := auxiliary.parse_auxiliary_chain(words,i,lex)) else None for i in range(len(words))],
                        comparative_evidence=[asdict(comparative.comparative_evidence(w,lex)) for w in words],
                        comparative_bases=[comparative.comparative_base_candidates(w) for w in words],
                        graded_comparative=[comparative.comparative_span_starting_at(words,i,lex) for i in range(len(words))],
                        aux_agreement=[[auxiliary.auxiliary_agreement(w,reference._subject_number(w,lex),v,lex) for v in
                                        ["am","is","are","was","were","has","have","had","hasnt","dont","can"]] for w in words])
        try:
            close(got, expected)
        except (AssertionError, TypeError) as error:
            raise AssertionError((words, error)) from error
    print(f"Local grammar parity passed: {len(cases)} cases including all function-word pairs")


if __name__ == "__main__":
    main()
