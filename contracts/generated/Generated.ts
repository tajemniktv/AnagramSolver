// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type Stop = "exhausted" | "candidate_cap" | "cancelled" | "timed_out";

export type Generated = {
  "bags": Array<Array<string>>;
  "candidate_budget": number;
  "engine_version": string;
  "generated": number;
  "kind": string;
  "normalized_input": string;
  "schema_version": number;
  "stop": Stop;
  "vocabulary_size": number;
  [key: string]: unknown;
};
