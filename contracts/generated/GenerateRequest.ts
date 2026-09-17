// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type HintMode = "any" | "exactly_one";

export type ShortPolicy = "none" | "common" | "all";

export type Strategy = "prefix" | "diverse";

export type GenerateRequest = {
  "allow_repeat": boolean;
  "candidate_budget": number;
  "exclude_regex"?: Array<string>;
  "excluded"?: Array<string>;
  "extra_short_words"?: Array<string>;
  "forbid_chars"?: string;
  "hint_mode": HintMode;
  "hints"?: Array<string>;
  "max_word_length": number;
  "max_words": number;
  "min_word_length": number;
  "min_words": number;
  "min_zipf": number;
  "required"?: Array<string>;
  "schema_version": number;
  "short_policy"?: ShortPolicy;
  "strategy": Strategy;
  "text": string;
};
