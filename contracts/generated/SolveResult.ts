// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type Row = {
  "base_final": number;
  "best_order": Array<string>;
  "colloc_norm": number;
  "deep": boolean;
  "fam": number;
  "family_key": Array<string>;
  "final": number;
  "grammar_norm": number;
  "grammar_potential": number;
  "grammar_potential_norm": number;
  "grammar_raw": number;
  "hint": number;
  "hints": Array<string>;
  "lex": number;
  "old_pair": number;
  "old_pcov": number;
  "old_pre": number;
  "old_rank": number;
  "phrase_attest_norm": number;
  "phrase_bonus": number;
  "phrase_kind": string;
  "pre_score": number;
  "structure_norm": number;
  "syntax_coverage": number;
  "valency_norm": number;
  "wn_coverage": number;
  "word_count": number;
  "words": Array<string>;
  "zavg": number;
  "zmin": number;
  [key: string]: unknown;
};

export type Stop = "exhausted" | "candidate_cap" | "cancelled" | "timed_out";

export type SolveResult = {
  "buckets": {

};
  "corpus_rescored": number;
  "deep_analyzed": number;
  "engine_version": string;
  "generated": number;
  "generation_stop": Stop;
  "kind": string;
  "normalized_input": string;
  "orders_evaluated": number;
  "schema_version": number;
  "shown": number;
  [key: string]: unknown;
};
