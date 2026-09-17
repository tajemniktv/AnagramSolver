// Generated from Rust by tools/generate_contracts.py. Do not edit.

export type HintMode = "any" | "exactly_one";

export type Strategy = "prefix" | "diverse";

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
