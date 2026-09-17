import type { GenerateRequest } from './generated/GenerateRequest.js';
import type { SolveRequest } from './generated/SolveRequest.js';
import type { JobStatus } from './generated/JobStatus.js';

const generation = {
  schema_version: 1, text: 'ate', min_words: 1, max_words: 3,
  min_word_length: 1, max_word_length: 20, min_zipf: 0,
  candidate_budget: 20, allow_repeat: true, strategy: 'prefix', hint_mode: 'any'
} satisfies GenerateRequest;

const solve = {
  generation, deep_per_group: 100, deep_all: false, order_mode: 'auto',
  beam_width: 128, exact_max_words: 5, retained_orders: 56,
  phrase_rescore_top: 300, phrase_bonus_max: 5, positive_bigrams: true,
  result_limit_per_group: 20
} satisfies SolveRequest;

// @ts-expect-error Unknown strategy must not widen to arbitrary string.
const invalid: GenerateRequest = { ...generation, strategy: 'random' };
// @ts-expect-error Explicit ranking budgets are required.
const missing: SolveRequest = { generation };
// @ts-expect-error Terminal states use the shared exact spelling.
const invalidState: JobStatus['state'] = 'timeout';
void [solve, invalid, missing, invalidState];
