import { invoke } from '@tauri-apps/api/core';
import type { SolveRequest } from '../../../contracts/generated/SolveRequest';
import type { SolveResult as RankedResult } from '../../../contracts/generated/SolveResult';
import type { Generated } from '../../../contracts/generated/Generated';
export type GeneratedResult = Generated & {status: JobStatus};
export type SolveResult = RankedResult | GeneratedResult;
export type { RankedResult };
import type { JobStatus } from '../../../contracts/generated/JobStatus';
import type { DeploymentLimits } from '../../../contracts/generated/DeploymentLimits';
export type { SolveRequest, JobStatus };
export type CorpusKey = 'dictionary' | 'unigrams' | 'bigrams' | 'wordnet' | 'phrase';
export interface Settings { schema_version: number; theme: 'dark' | 'light' | 'system'; corpora: Record<CorpusKey, string>; model: string; request: SolveRequest; runtime: {cache_path: string; cache_entries: number; cache_bytes: number; custom_limits: boolean; limits: DeploymentLimits} }
export interface Bootstrap { settings: Settings; notice: string | null; limits: DeploymentLimits; corpus_error: string | null; data_directory: string; engine_version: string; active_jobs: number; queue_capacity: number }
export interface Job { id: number; active: boolean; status: JobStatus | null; error: string | null }
export interface TrainingStatus { active: boolean; error: string | null; report: { model_path: string; baseline: { groups: number; recall1: number; mrr: number }; held_out: { groups: number; recall1: number; mrr: number } } | null }
// All desktop transport lives here. Browser development never fabricates a solver.
export const bridge = {
  manage: (action: unknown) => invoke<void>('manage', { action }),
  diagnostics: () => invoke<Record<string, unknown>>('diagnostics'),
  readReport: (path: string) => invoke<unknown>('read_report', { path }),
  maintenanceStatus: () => invoke<{active: boolean; error: string | null; report: Record<string, unknown> | null}>('training_status'),
  bootstrap: () => invoke<Bootstrap>('bootstrap'),
  save: (settings: Settings) => invoke<void>('save_settings', { settings }),
  start: (request: SolveRequest, rebuild = false, extended = false, generationOnly = false) => invoke<number>('start_job', { request, rebuild, extended, generationOnly }),
  clearCache: () => invoke<void>('clear_cache'),
  status: () => invoke<Job>('job_status'),
  result: (id: number) => invoke<SolveResult | null>('job_result', { id }),
  cancel: (id: number) => invoke<void>('cancel_job', { id }),
  choose: (kind: CorpusKey | 'model' | 'training' | 'source' | 'titles' | 'cache') => invoke<string | null>('choose_path', { kind }),
  train: (dataset: string, epochs: number, folds: number, learningRate = 0.08, l2 = 0.002) => invoke<void>('train_model', { dataset, epochs, folds, learningRate, l2 }),
  trainingStatus: () => invoke<TrainingStatus>('training_status'),
  cancelTraining: () => invoke<void>('cancel_training'),
  copy: (id: number) => invoke<void>('copy_results', { id }),
  export: (id: number, format: 'txt' | 'json') => invoke<string | null>('export_results', { id, format }),
};
export const message = (error: unknown) => error instanceof Error ? error.message : String(error);
