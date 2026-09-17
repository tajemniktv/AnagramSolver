// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type GenerateRequest = {
  "allow_repeat": boolean;
  "candidate_budget": number;
  "excluded"?: Array<string>;
  "hint_mode": HintMode;
  "hints"?: Array<string>;
  "max_word_length": number;
  "max_words": number;
  "min_word_length": number;
  "min_words": number;
  "min_zipf": number;
  "required"?: Array<string>;
  "schema_version": number;
  "strategy": Strategy;
  "text": string;
};

export type HintMode = "any" | "exactly_one";

export type OrderMode = "auto" | "exact" | "beam";

export type Strategy = "prefix" | "diverse";

export type SolveRequest = {
  "beam_width": number;
  "deep_all": boolean;
  "deep_per_group": number;
  "exact_max_words": number;
  "generation": GenerateRequest;
  "order_mode": OrderMode;
  "phrase_bonus_max": number;
  "phrase_rescore_top": number;
  "positive_bigrams": boolean;
  "result_limit_per_group": number;
  "retained_orders": number;
};
