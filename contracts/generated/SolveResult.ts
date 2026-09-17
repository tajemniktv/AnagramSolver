// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type CacheFlags = {
  "corrupt_entry_ignored": boolean;
  "hit": boolean;
  "rebuilt": boolean;
};

export type Counts = {
  "corpus_rescored": number;
  "deep_analyzed": number;
  "deep_selected": number;
  "generated": number;
  "orders_evaluated": number;
  "shown": number;
};

export type DataIdentity = {
  "bytes": number;
  "role": string;
  "sha256": string;
};

export type EffectiveBudgets = {
  "candidate_limit"?: number | null;
  "deadline_ms"?: number | null;
  "deep_limit"?: number | null;
  "deep_per_group": number;
  "result_limit_per_group": number;
};

export type Exhaustion = "exhausted" | "truncated" | "unknown";

export type Failure = {
  "code": string;
  "message": string;
};

export type JobState = "queued" | "running" | "succeeded" | "cancelled" | "timed_out" | "failed";

export type JobStatus = {
  "budgets": EffectiveBudgets;
  "cache": CacheFlags;
  "counts": Counts;
  "error"?: (Failure) | (null);
  "exhaustion": Exhaustion;
  "job_id": string;
  "schema_version": number;
  "stage": Stage;
  "state": JobState;
  "versions": Versions;
};

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

export type Stage = "queued" | "loading_corpora" | "generating" | "preparing" | "deep_ranking" | "corpus_ranking" | "finalizing" | "complete";

export type Stop = "exhausted" | "candidate_cap" | "cancelled" | "timed_out";

export type Versions = {
  "data": Array<DataIdentity>;
  "engine": string;
  "ranking": string;
};

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
  "status": JobStatus;
  [key: string]: unknown;
};
