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

export type Stage = "queued" | "loading_corpora" | "generating" | "preparing" | "deep_ranking" | "corpus_ranking" | "finalizing" | "complete";

export type Versions = {
  "data": Array<DataIdentity>;
  "engine": string;
  "ranking": string;
};

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
