//! Serial ranked vertical slice. Corpus paths are adapter-owned, not wire data.
use crate::control::Control;
use crate::{
    contracts::{Exhaustion, JobStatus, Stage},
    progress::Progress,
};
use crate::{
    corpus_ranking::{self, Collocation},
    lexicon::Unigrams,
    normalize_letters,
    phrase_index::PhraseIndex,
    provenance::Snapshot,
    ranking::{self, OrderMode},
    request::{self, Error, GenerateRequest},
    scoring::{self, Bigrams},
    wordnet::WordNet,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, io::Cursor, path::Path};

#[derive(Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// One worker by default; explicit parallelism never consumes every core implicitly.
    #[serde(default = "one_worker")]
    #[schemars(range(min = 0, max = 32))]
    pub workers: usize,
    #[serde(default)]
    pub refine: bool,
    pub generation: GenerateRequest,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub deep_per_group: usize,
    pub deep_all: bool,
    pub order_mode: OrderMode,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub beam_width: usize,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub exact_max_words: usize,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub retained_orders: usize,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub phrase_rescore_top: usize,
    #[schemars(range(min = 0))]
    pub phrase_bonus_max: f64,
    pub positive_bigrams: bool,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub result_limit_per_group: usize,
}
#[derive(Serialize, schemars::JsonSchema)]
pub struct Timings {
    /// Current solve invocation, excluding adapter JSON input/output transport.
    #[schemars(range(max = 9007199254740991_u64))]
    pub execution_ms: u64,
    /// Deep/corpus ranking and final bucket assembly, excluding cache writes.
    /// On a cache hit this belongs to the original computation.
    #[schemars(range(max = 9007199254740991_u64))]
    pub ranking_computation_ms: u64,
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Result {
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    pub kind: &'static str,
    pub engine_version: &'static str,
    pub normalized_input: String,
    pub generated: usize,
    pub deep_analyzed: usize,
    pub shown: usize,
    pub orders_evaluated: usize,
    pub corpus_rescored: usize,
    pub generation_stop: crate::generation::Stop,
    pub buckets: BTreeMap<usize, Vec<ranking::Row>>,
    pub status: JobStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Timings>,
}
pub struct Paths<'a> {
    pub model: Option<&'a Path>,
    pub dictionary: &'a Path,
    pub unigrams: &'a Path,
    pub bigrams: &'a Path,
    pub wordnet: &'a Path,
    pub phrase: Option<&'a Path>,
}
fn one_worker() -> usize {
    1
}
fn corpus(e: std::io::Error) -> Error {
    Error::new("corpus_error", e.to_string())
}

pub fn solve(request: &Request, paths: Paths<'_>) -> std::result::Result<Result, Error> {
    solve_controlled(request, paths, &Control::default())
}
pub fn solve_controlled(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
) -> std::result::Result<Result, Error> {
    solve_with_limits(
        request,
        paths,
        control,
        &crate::policy::DeploymentLimits::default(),
    )
}

pub fn solve_with_limits(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
    limits: &crate::policy::DeploymentLimits,
) -> std::result::Result<Result, Error> {
    solve_observed(request, paths, control, limits, "local", &mut |_| {})
}

/// Only admitted requests become executions; validation failures have no job.
pub fn solve_observed(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
    limits: &crate::policy::DeploymentLimits,
    job_id: &str,
    observer: &mut dyn FnMut(&JobStatus),
) -> std::result::Result<Result, Error> {
    solve_cached_observed(request, paths, control, limits, (job_id, observer), None)
}

/// Optional native result cache. Exact corpus identity and current admission
/// remain fresh; completed generation and ranking are reused on a valid hit.
pub fn solve_cached_observed(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
    limits: &crate::policy::DeploymentLimits,
    job: (&str, &mut dyn FnMut(&JobStatus)),
    cache: Option<&crate::cache::Config<'_>>,
) -> std::result::Result<Result, Error> {
    let (job_id, observer) = job;
    let execution_started = std::time::Instant::now();
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    validate(request)?;
    limits.admit(request)?;
    if job_id.is_empty() {
        return Err(Error::new("invalid_job_id", "Job ID must not be empty"));
    }
    let control = limits.control(control)?;
    if control.deadline().is_some_and(|d| {
        d.saturating_duration_since(std::time::Instant::now())
            .as_millis()
            > crate::contracts::MAX_WIRE_INTEGER as u128
    }) {
        return Err(Error::new(
            "invalid_timeout",
            "Deadline exceeds exact JSON integer range",
        ));
    }
    let mut progress = Progress::new(request, limits, &control, job_id, observer);
    let mut result = match control.check() {
        Ok(()) => {
            progress.start();
            run(request, paths, &control, limits, &mut progress, cache)
        }
        Err(reason) => Err(Error::new(reason, reason)),
    };
    if let Err(reason) = control.check() {
        result = Err(Error::new(reason, reason));
    }
    if result.is_err() {
        // No result rows are published for a failed or interrupted execution.
        progress.status.counts.shown = 0;
    }
    progress.finish(result.as_ref().err());
    if let Ok(value) = &mut result {
        value.status = progress.status;
        if let Some(timings) = &mut value.timings {
            timings.execution_ms = elapsed_ms(execution_started);
        }
    }
    result
}
/// Request semantics only; adapters can reject invalid requests before any I/O.
pub fn validate(request: &Request) -> std::result::Result<(), Error> {
    request::validate(&request.generation)?;
    if request.workers > 32 {
        return Err(Error::new(
            "invalid_workers",
            "Workers must be 0 (automatic) or between 1 and 32",
        ));
    }
    if request.generation.max_words > 10
        || request.deep_per_group == 0
        || request.beam_width == 0
        || request.exact_max_words == 0
        || request.retained_orders == 0
        || request.phrase_rescore_top == 0
        || request.result_limit_per_group == 0
        || !request.phrase_bonus_max.is_finite()
        || request.phrase_bonus_max < 0.0
        || [
            request.deep_per_group,
            request.beam_width,
            request.exact_max_words,
            request.retained_orders,
            request.phrase_rescore_top,
            request.result_limit_per_group,
        ]
        .iter()
        .any(|&n| n as u128 > 9_007_199_254_740_991)
    {
        return Err(Error::new(
            "invalid_ranking_limits",
            "Invalid ranking budgets; ordering supports at most 10 words",
        ));
    }
    Ok(())
}

fn run(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
    limits: &crate::policy::DeploymentLimits,
    progress: &mut Progress<'_>,
    cache: Option<&crate::cache::Config<'_>>,
) -> std::result::Result<Result, Error> {
    validate(request)?;
    let model = paths
        .model
        .map(|p| crate::learned::Model::load_controlled(p, control))
        .transpose()
        .map_err(corpus)?;
    if let Some(identity) = model.as_ref().and_then(crate::learned::Model::identity) {
        progress.status.versions.data.push(identity.clone());
    }
    let unigram_source = Snapshot::load(paths.unigrams, "unigrams", control).map_err(corpus)?;
    progress
        .status
        .versions
        .data
        .push(unigram_source.identity.clone());
    let dictionary = Snapshot::load(paths.dictionary, "dictionary", control).map_err(corpus)?;
    progress
        .status
        .versions
        .data
        .push(dictionary.identity.clone());
    let bigram_source = Snapshot::load(paths.bigrams, "bigrams", control).map_err(corpus)?;
    progress
        .status
        .versions
        .data
        .push(bigram_source.identity.clone());
    let (mut lex, identities) = WordNet::load_identified(paths.wordnet, control).map_err(corpus)?;
    progress.status.versions.data.extend(identities);
    let phrase = paths
        .phrase
        .map(|path| PhraseIndex::open_controlled(path, control))
        .transpose()
        .map_err(corpus)?;
    if let Some(index) = &phrase {
        progress
            .status
            .versions
            .data
            .push(index.identity().map_err(corpus)?);
    }
    progress.status.versions.data_complete = true;
    let mut cache_store = cache
        .and_then(|config| crate::cache::Store::open(config.path, config.limits, control).ok());
    let cache_key = if cache_store.is_some() {
        crate::cache::ranked_key(request, &progress.status.versions)?
    } else {
        String::new()
    };
    if let Some(store) = &mut cache_store {
        progress.status.cache.rebuilt = cache.is_some_and(|c| c.rebuild);
        match store.lookup(&cache_key, progress.status.cache.rebuilt) {
            Ok(crate::cache::Lookup::Hit(stored)) => {
                match serde_json::from_slice::<crate::cache::RankedPayload>(&stored.payload) {
                    Ok(hit)
                        if hit.generated as u128 <= crate::contracts::MAX_WIRE_INTEGER as u128
                            && (request.generation.candidate_budget == 0
                                || hit.generated <= request.generation.candidate_budget)
                            && match hit.generation_stop {
                                crate::generation::Stop::Exhausted => true,
                                crate::generation::Stop::CandidateCap => {
                                    request.generation.candidate_budget > 0
                                        && hit.generated == request.generation.candidate_budget
                                }
                                _ => false,
                            }
                            && stored.computation_ms <= crate::contracts::MAX_WIRE_INTEGER
                            && hit.deep_selected <= hit.generated
                            && hit.deep_analyzed == hit.deep_selected
                            && hit.orders_evaluated as u128
                                <= crate::contracts::MAX_WIRE_INTEGER as u128
                            && hit.corpus_rescored <= hit.generated
                            && hit.buckets.values().map(Vec::len).sum::<usize>()
                                <= hit.generated
                            && hit
                                .buckets
                                .values()
                                .all(|rows| rows.len() <= request.result_limit_per_group) =>
                    {
                        control.check().map_err(|e| Error::new(e, e))?;
                        // The key covers the exact request and consumed corpora.
                        // Apply current deployment policy to the saved shortlist,
                        // never to the number of displayed rows.
                        progress.stage(Stage::Preparing);
                        control.check().map_err(|e| Error::new(e, e))?;
                        progress.status.cache.hit = true;
                        progress.status.counts.generated = hit.generated;
                        progress.status.exhaustion = match hit.generation_stop {
                            crate::generation::Stop::Exhausted => Exhaustion::Exhausted,
                            _ => Exhaustion::Truncated,
                        };
                        progress.status.counts.deep_selected = hit.deep_selected;
                        progress.emit();
                        limits.admit_deep(hit.deep_selected)?;
                        progress.stage(Stage::Finalizing);
                        control.check().map_err(|e| Error::new(e, e))?;
                        let shown = hit.buckets.values().map(Vec::len).sum();
                        progress.status.counts.deep_analyzed = hit.deep_analyzed;
                        progress.status.counts.orders_evaluated = hit.orders_evaluated;
                        progress.status.counts.corpus_rescored = hit.corpus_rescored;
                        progress.status.counts.shown = shown;
                        return Ok(Result {
                            schema_version: 1,
                            kind: "ranked",
                            engine_version: env!("CARGO_PKG_VERSION"),
                            normalized_input: normalize_letters(&request.generation.text),
                            generated: hit.generated,
                            deep_analyzed: hit.deep_analyzed,
                            shown,
                            orders_evaluated: hit.orders_evaluated,
                            corpus_rescored: hit.corpus_rescored,
                            generation_stop: hit.generation_stop,
                            buckets: hit.buckets,
                            status: progress.status.clone(),
                            timings: Some(Timings {
                                execution_ms: 0,
                                ranking_computation_ms: stored.computation_ms,
                            }),
                        });
                    }
                    _ => progress.status.cache.corrupt_entry_ignored = true,
                }
            }
            Ok(crate::cache::Lookup::Corrupt) => progress.status.cache.corrupt_entry_ignored = true,
            _ => {} // Busy/unavailable storage is not a solver failure.
        }
    }
    control.check().map_err(|e| Error::new(e, e))?;
    let unigrams =
        Unigrams::load(control.reader(Cursor::new(&unigram_source.bytes))).map_err(corpus)?;
    progress.stage(Stage::Generating);
    let generated = request::generate_controlled(
        &request.generation,
        control.reader(Cursor::new(&dictionary.bytes)),
        Some(&unigrams),
        control,
    )?;
    drop(dictionary);
    progress.status.counts.generated = generated.generated;
    progress.status.exhaustion = match generated.stop {
        crate::generation::Stop::Exhausted => Exhaustion::Exhausted,
        crate::generation::Stop::CandidateCap => Exhaustion::Truncated,
        _ => Exhaustion::Unknown,
    };
    progress.emit();
    control.check().map_err(|e| Error::new(e, e))?;
    progress.stage(Stage::Preparing);
    lex.prepare_words(generated.vocabulary.iter().map(String::as_str), control)
        .map_err(|e| Error::new(e, e))?;
    let bigrams = Bigrams::load(
        control.reader(Cursor::new(&bigram_source.bytes)),
        &unigrams,
        &generated.vocabulary,
    )
    .map_err(corpus)?;
    let hints = request
        .generation
        .hints
        .iter()
        .map(|w| normalize_letters(w))
        .collect();
    let records = scoring::pre_rank_controlled(
        &generated.bags,
        &hints,
        Some(&unigrams),
        Some(&bigrams),
        &generated.vocabulary,
        &generated.short_whitelist,
        control,
    )
    .map_err(|e| Error::new(e, e))?;
    let inputs =
        ranking::from_records_controlled(&records, control).map_err(|e| Error::new(e, e))?;
    let mut rows =
        ranking::prepare_controlled(inputs, &lex, control).map_err(|e| Error::new(e, e))?;
    let selected =
        ranking::choose_deep_controlled(&rows, request.deep_per_group, request.deep_all, control)
            .map_err(|e| Error::new(e, e))?;
    progress.status.counts.deep_selected = selected.len();
    progress.emit();
    limits.admit_deep(selected.len())?;
    let ranking_started = std::time::Instant::now();
    let options = ranking::Options {
        mode: request.order_mode,
        beam_width: request.beam_width,
        exact_max_words: request.exact_max_words,
        retained_orders: request.retained_orders,
    };
    progress.stage(Stage::DeepRanking);
    let mut orders_evaluated = ranking::deep_analyze_parallel(
        &mut rows,
        &selected,
        &lex,
        &options,
        request.workers,
        control,
        &mut |completed, orders| {
            progress.status.counts.deep_analyzed = completed;
            progress.status.counts.orders_evaluated = orders;
            progress.emit();
        },
    )
    .map_err(|e| Error::new("ranking_error", e))?;
    if request.refine {
        orders_evaluated += ranking::refine_rows(&mut rows, &lex, control)?;
        progress.status.counts.orders_evaluated = orders_evaluated;
        progress.emit();
    }
    progress.stage(Stage::CorpusRanking);
    let positive = if request.positive_bigrams {
        Some(
            Collocation::load(
                control.reader(Cursor::new(&unigram_source.bytes)),
                control.reader(Cursor::new(&bigram_source.bytes)),
                &generated.vocabulary,
            )
            .map_err(corpus)?,
        )
    } else {
        None
    };
    drop(unigram_source);
    drop(bigram_source);
    progress.emit();
    let corpus_rescored = corpus_ranking::rescore_with_model(
        &mut rows,
        positive.as_ref(),
        phrase
            .as_ref()
            .map(|p| p as &dyn corpus_ranking::PhraseCorpus),
        request.phrase_rescore_top,
        (request.phrase_bonus_max, model.as_ref()),
        control,
        &mut |completed| {
            progress.status.counts.corpus_rescored = completed;
            progress.emit();
        },
    )
    .map_err(corpus)?;
    let deep_analyzed = rows.iter().filter(|r| r.deep).count();
    progress.status.counts.corpus_rescored = corpus_rescored;
    progress.stage(Stage::Finalizing);
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    let ranked = ranking::rank_buckets_controlled(&rows, control).map_err(|e| Error::new(e, e))?;
    let mut buckets: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for (wc, indices) in ranked {
        let mut bucket = Vec::new();
        for i in indices.into_iter().take(request.result_limit_per_group) {
            control.check().map_err(|e| Error::new(e, e))?;
            let mut row = rows[i].clone();
            row.display_phrase = crate::format_phrase(&row.best_order);
            bucket.push(row);
        }
        buckets.insert(wc, bucket);
    }
    control.check().map_err(|e| Error::new(e, e))?;
    let shown = buckets.values().map(Vec::len).sum();
    let ranking_computation_ms = elapsed_ms(ranking_started);
    if let Some(store) = &mut cache_store {
        let payload = crate::cache::RankedPayload {
            generated: generated.generated,
            generation_stop: generated.stop,
            deep_selected: selected.len(),
            deep_analyzed,
            orders_evaluated,
            corpus_rescored,
            buckets: buckets.clone(),
        };
        if let Ok(bytes) = serde_json::to_vec(&payload) {
            let _ = store.put(&cache_key, &bytes, ranking_computation_ms);
        }
    }
    control.check().map_err(|e| Error::new(e, e))?;
    progress.status.counts.shown = shown;
    Ok(Result {
        schema_version: 1,
        kind: "ranked",
        engine_version: env!("CARGO_PKG_VERSION"),
        normalized_input: generated.normalized_input,
        generated: generated.generated,
        deep_analyzed,
        shown,
        orders_evaluated,
        corpus_rescored,
        generation_stop: generated.stop,
        buckets,
        status: progress.status.clone(),
        timings: cache.map(|_| Timings {
            execution_ms: 0,
            ranking_computation_ms,
        }),
    })
}

fn elapsed_ms(start: std::time::Instant) -> u64 {
    start
        .elapsed()
        .as_millis()
        .min(crate::contracts::MAX_WIRE_INTEGER as u128) as u64
}
