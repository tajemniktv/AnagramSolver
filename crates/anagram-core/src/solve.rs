//! Serial ranked vertical slice. Corpus paths are adapter-owned, not wire data.
use crate::control::Control;
use crate::{
    corpus_ranking::{self, Collocation},
    lexicon::Unigrams,
    normalize_letters,
    phrase_index::PhraseIndex,
    ranking::{self, OrderMode},
    request::{self, Error, GenerateRequest},
    scoring::{self, Bigrams},
    wordnet::WordNet,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs::File, io::BufReader, path::Path};

#[derive(Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub generation: GenerateRequest,
    pub deep_per_group: usize,
    pub deep_all: bool,
    pub order_mode: OrderMode,
    pub beam_width: usize,
    pub exact_max_words: usize,
    pub retained_orders: usize,
    pub phrase_rescore_top: usize,
    pub phrase_bonus_max: f64,
    pub positive_bigrams: bool,
    pub result_limit_per_group: usize,
}
#[derive(Serialize, schemars::JsonSchema)]
pub struct Result {
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
}
pub struct Paths<'a> {
    pub dictionary: &'a Path,
    pub unigrams: &'a Path,
    pub bigrams: &'a Path,
    pub wordnet: &'a Path,
    pub phrase: Option<&'a Path>,
}
fn corpus(e: std::io::Error) -> Error {
    Error::new("corpus_error", e.to_string())
}
fn open(path: &Path) -> std::result::Result<BufReader<File>, Error> {
    File::open(path).map(BufReader::new).map_err(corpus)
}

pub fn solve(request: &Request, paths: Paths<'_>) -> std::result::Result<Result, Error> {
    solve_controlled(request, paths, &Control::default())
}
pub fn solve_controlled(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
) -> std::result::Result<Result, Error> {
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    let result = run(request, paths, control);
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    result
}
fn run(
    request: &Request,
    paths: Paths<'_>,
    control: &Control,
) -> std::result::Result<Result, Error> {
    if request.generation.max_words > 10
        || request.deep_per_group == 0
        || request.beam_width == 0
        || request.exact_max_words == 0
        || request.retained_orders == 0
        || request.phrase_rescore_top == 0
        || request.result_limit_per_group == 0
        || !request.phrase_bonus_max.is_finite()
        || request.phrase_bonus_max < 0.0
    {
        return Err(Error::new(
            "invalid_ranking_limits",
            "Invalid ranking budgets; ordering supports at most 10 words",
        ));
    }
    let required: String = request
        .generation
        .required
        .iter()
        .map(|w| normalize_letters(w))
        .collect();
    let mut required_letters: Vec<_> = required.chars().collect();
    required_letters.sort();
    let mut target: Vec<_> = normalize_letters(&request.generation.text)
        .chars()
        .collect();
    target.sort();
    if !target.is_empty() && target == required_letters {
        return Err(Error::new(
            "zero_residual_ranked_input",
            "Required words consume the entire target; the reference ranked solver does not support zero-residual answers",
        ));
    }
    let unigrams = Unigrams::load(control.reader(open(paths.unigrams)?)).map_err(corpus)?;
    let generated = request::generate(
        &request.generation,
        control.reader(open(paths.dictionary)?),
        Some(&unigrams),
        control.flag(),
        control.deadline(),
    )?;
    let bigrams = Bigrams::load(
        control.reader(open(paths.bigrams)?),
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
    let lex = WordNet::load_controlled(paths.wordnet, control).map_err(corpus)?;
    let mut rows = ranking::prepare_controlled(ranking::from_records(&records), &lex, control)
        .map_err(|e| Error::new(e, e))?;
    let selected = ranking::choose_deep(&rows, request.deep_per_group, request.deep_all);
    let options = ranking::Options {
        mode: request.order_mode,
        beam_width: request.beam_width,
        exact_max_words: request.exact_max_words,
        retained_orders: request.retained_orders,
    };
    let orders_evaluated =
        ranking::deep_analyze_controlled(&mut rows, &selected, &lex, &options, control)
            .map_err(|e| Error::new("ranking_error", e))?;
    let positive = if request.positive_bigrams {
        Some(
            Collocation::load(
                control.reader(open(paths.unigrams)?),
                control.reader(open(paths.bigrams)?),
                &generated.vocabulary,
            )
            .map_err(corpus)?,
        )
    } else {
        None
    };
    let phrase = paths
        .phrase
        .map(|path| PhraseIndex::open_controlled(path, control))
        .transpose()
        .map_err(corpus)?;
    let corpus_rescored = corpus_ranking::rescore_controlled(
        &mut rows,
        positive.as_ref(),
        phrase
            .as_ref()
            .map(|p| p as &dyn corpus_ranking::PhraseCorpus),
        request.phrase_rescore_top,
        request.phrase_bonus_max,
        control,
    )
    .map_err(corpus)?;
    let deep_analyzed = rows.iter().filter(|r| r.deep).count();
    let buckets: BTreeMap<_, Vec<_>> = ranking::rank_buckets(&rows)
        .into_iter()
        .map(|(wc, indices)| {
            (
                wc,
                indices
                    .into_iter()
                    .take(request.result_limit_per_group)
                    .map(|i| rows[i].clone())
                    .collect(),
            )
        })
        .collect();
    let shown = buckets.values().map(Vec::len).sum();
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
    })
}
