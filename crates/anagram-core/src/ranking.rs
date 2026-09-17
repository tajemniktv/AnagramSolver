//! Prepared rows, family-expanded shortlists and serial deep ranking.
//! Retained alternatives belong to each row, never a global identity side table.
use crate::{diversity, grammar, ordering, wordnet::WordNet};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Input {
    pub words: Vec<String>,
    pub word_count: usize,
    pub old_rank: usize,
    pub old_pre: f64,
    pub lex: f64,
    pub fam: f64,
    pub old_pair: f64,
    pub hint: f64,
    pub zavg: f64,
    pub zmin: f64,
    pub old_pcov: f64,
    pub hints: Vec<String>,
}

/// Preserve the frozen generator-to-reranker text boundary, including decimal
/// quantization. Removing it would change rankings, not merely serialization.
pub fn from_records(records: &[crate::scoring::PreRecord]) -> Vec<Input> {
    from_records_controlled(records, &crate::control::Control::default())
        .expect("unlimited control")
}
pub fn from_records_controlled(
    records: &[crate::scoring::PreRecord],
    control: &crate::control::Control,
) -> Result<Vec<Input>, &'static str> {
    control.check()?;
    fn rounded(value: f64, precision: usize) -> f64 {
        format!("{value:.precision$}")
            .parse()
            .expect("finite score formatting")
    }
    let mut buckets: BTreeMap<usize, Vec<&crate::scoring::PreRecord>> = BTreeMap::new();
    for record in records {
        control.check()?;
        buckets.entry(record.words.len()).or_default().push(record);
    }
    let mut inputs = Vec::with_capacity(records.len());
    for bucket in buckets.values_mut() {
        control.sort_by(bucket, |a, b| {
            b.pre_score
                .total_cmp(&a.pre_score)
                .then_with(|| b.lexical.lex_raw.total_cmp(&a.lexical.lex_raw))
                .then_with(|| b.words.join(" ").cmp(&a.words.join(" ")))
        })?;
        for (i, record) in bucket.iter().enumerate() {
            control.check()?;
            inputs.push(Input {
                words: record.words.clone(),
                word_count: record.words.len(),
                old_rank: i + 1,
                old_pre: rounded(record.pre_score, 2),
                lex: rounded(record.lex_pct, 3),
                fam: rounded(record.family_pct, 3),
                old_pair: rounded(record.pair_pct, 3),
                hint: rounded(record.hint_info, 3),
                zavg: rounded(record.lexical.avg_zipf, 2),
                zmin: rounded(record.lexical.min_zipf, 2),
                old_pcov: rounded(record.pair_coverage, 2),
                hints: record.matched_hints.clone(),
            });
        }
    }
    Ok(inputs)
}

#[derive(Clone, Debug, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Row {
    #[serde(default)]
    pub display_phrase: String,
    #[serde(flatten)]
    pub input: Input,
    pub wn_coverage: f64,
    pub grammar_potential: f64,
    pub grammar_potential_norm: f64,
    pub pre_score: f64,
    pub deep: bool,
    pub best_order: Vec<String>,
    pub grammar_raw: f64,
    pub grammar_norm: f64,
    pub structure_norm: f64,
    pub valency_norm: f64,
    pub syntax_coverage: f64,
    pub phrase_kind: String,
    pub base_final: f64,
    pub colloc_norm: f64,
    pub phrase_attest_norm: f64,
    pub phrase_bonus: f64,
    #[serde(rename = "final")]
    pub final_score: f64,
    pub family_key: Vec<String>,
    #[serde(skip)]
    pub alternatives: Vec<ordering::Candidate>,
}

pub fn score_pre(row: &Row) -> f64 {
    100.0
        * (0.18 * row.input.lex
            + 0.28 * row.input.fam
            + 0.16 * row.input.hint
            + 0.28 * row.grammar_potential_norm
            + 0.10 * row.wn_coverage)
}
pub fn order_base_final(row: &Row, candidate: &ordering::Candidate) -> f64 {
    100.0
        * (0.10 * row.input.lex
            + 0.16 * row.input.fam
            + 0.12 * row.input.hint
            + 0.22 * candidate.grammar_norm
            + 0.28 * candidate.structure_norm
            + 0.08 * candidate.valency_norm
            + 0.04 * row.wn_coverage)
}
pub fn score_final(row: &Row) -> f64 {
    100.0
        * (0.10 * row.input.lex
            + 0.16 * row.input.fam
            + 0.12 * row.input.hint
            + 0.22 * row.grammar_norm
            + 0.28 * row.structure_norm
            + 0.08 * row.valency_norm
            + 0.04 * row.wn_coverage)
}

pub fn prepare(inputs: Vec<Input>, lex: &WordNet) -> Vec<Row> {
    prepare_controlled(inputs, lex, &crate::control::Control::default()).expect("unlimited control")
}
pub fn prepare_controlled(
    inputs: Vec<Input>,
    lex: &WordNet,
    control: &crate::control::Control,
) -> Result<Vec<Row>, &'static str> {
    control.check()?;
    let mut rows: Vec<_> = inputs
        .into_iter()
        .map(|mut input| {
            control.check()?;
            input.words.sort();
            let mut family_key: Vec<_> = input
                .words
                .iter()
                .map(|w| lex.morphology_family_word(w))
                .collect();
            family_key.sort();
            let mut row = Row {
                display_phrase: String::new(),
                wn_coverage: grammar::coverage(&input.words, lex),
                grammar_potential_norm: grammar::potential(&input.words, lex),
                input,
                family_key,
                grammar_potential: 0.0,
                pre_score: 0.0,
                deep: false,
                best_order: Vec::new(),
                grammar_raw: 0.0,
                grammar_norm: 0.0,
                structure_norm: 0.0,
                valency_norm: 0.5,
                syntax_coverage: 0.0,
                phrase_kind: "unknown".into(),
                base_final: 0.0,
                colloc_norm: 0.0,
                phrase_attest_norm: 0.0,
                phrase_bonus: 0.0,
                final_score: 0.0,
                alternatives: Vec::new(),
            };
            row.pre_score = score_pre(&row);
            Ok(row)
        })
        .collect::<Result<_, &'static str>>()?;
    control.sort_by(&mut rows, |a, b| {
        a.input
            .word_count
            .cmp(&b.input.word_count)
            .then_with(|| a.input.words.cmp(&b.input.words))
    })?;
    control.check()?;
    Ok(rows)
}

/// `per_group` is the initial shortlist size, not a hard deployment work cap:
/// family expansion intentionally includes every selected inflection sibling.
pub fn choose_deep(rows: &[Row], per_group: usize, deep_all: bool) -> BTreeSet<usize> {
    choose_deep_controlled(
        rows,
        per_group,
        deep_all,
        &crate::control::Control::default(),
    )
    .expect("unlimited control")
}

pub fn choose_deep_controlled(
    rows: &[Row],
    per_group: usize,
    deep_all: bool,
    control: &crate::control::Control,
) -> Result<BTreeSet<usize>, &'static str> {
    control.check()?;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        control.check()?;
        groups.entry(row.input.word_count).or_default().push(i);
    }
    let mut families = BTreeSet::new();
    for bucket in groups.values_mut() {
        control.sort_by(bucket, |&i, &j| {
            rows[j]
                .pre_score
                .total_cmp(&rows[i].pre_score)
                .then_with(|| rows[j].input.fam.total_cmp(&rows[i].input.fam))
                .then_with(|| rows[j].input.lex.total_cmp(&rows[i].input.lex))
        })?;
        for &i in bucket
            .iter()
            .take(if deep_all { bucket.len() } else { per_group })
        {
            control.check()?;
            families.insert((rows[i].input.word_count, rows[i].family_key.clone()));
        }
    }
    let mut selected = BTreeSet::new();
    for (i, row) in rows.iter().enumerate() {
        control.check()?;
        if families.contains(&(row.input.word_count, row.family_key.clone())) {
            selected.insert(i);
        }
    }
    Ok(selected)
}

#[derive(Clone, Copy, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OrderMode {
    Auto,
    Exact,
    Beam,
}
pub struct Options {
    pub mode: OrderMode,
    pub beam_width: usize,
    pub exact_max_words: usize,
    pub retained_orders: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            mode: OrderMode::Auto,
            beam_width: 128,
            exact_max_words: 5,
            retained_orders: 56,
        }
    }
}

pub fn deep_analyze(
    rows: &mut [Row],
    selected: &BTreeSet<usize>,
    lex: &WordNet,
    options: &Options,
) -> Result<usize, &'static str> {
    deep_analyze_controlled(
        rows,
        selected,
        lex,
        options,
        &crate::control::Control::default(),
    )
}
pub fn deep_analyze_controlled(
    rows: &mut [Row],
    selected: &BTreeSet<usize>,
    lex: &WordNet,
    options: &Options,
    control: &crate::control::Control,
) -> Result<usize, &'static str> {
    deep_analyze_observed(rows, selected, lex, options, control, &mut |_, _| {})
}

pub fn deep_analyze_observed(
    rows: &mut [Row],
    selected: &BTreeSet<usize>,
    lex: &WordNet,
    options: &Options,
    control: &crate::control::Control,
    observer: &mut dyn FnMut(usize, usize),
) -> Result<usize, &'static str> {
    control.check()?;
    let raw_k = diversity::raw_pool_size(options.retained_orders)?;
    if options.beam_width == 0 || selected.iter().any(|i| *i >= rows.len()) {
        return Err("invalid deep-analysis options or row index");
    }
    let mut evaluated = 0;
    let mut completed = 0;
    for &i in selected {
        control.check()?;
        let row = &mut rows[i];
        let exact = matches!(options.mode, OrderMode::Exact)
            || (matches!(options.mode, OrderMode::Auto)
                && row.input.words.len() <= options.exact_max_words);
        let mut in_flight = 0;
        let outcome = ordering::rank_observed(
            &row.input.words,
            lex,
            exact,
            options.beam_width,
            raw_k,
            control,
            &mut |count| {
                in_flight = count;
                if count % 256 == 0 {
                    observer(completed, evaluated + count);
                }
            },
        );
        let (orders, count) = match outcome {
            Ok(result) => result,
            Err(reason) => {
                observer(completed, evaluated + in_flight);
                return Err(reason);
            }
        };
        let Some(winner) = orders.first() else {
            continue;
        };
        evaluated += count;
        let alternatives = match diversity::select_controlled(
            &orders,
            options.retained_orders,
            diversity::QUALITY_CORE,
            0.12,
            control,
        ) {
            Ok(alternatives) => alternatives,
            Err(reason) => {
                observer(completed, evaluated);
                return Err(reason);
            }
        };
        row.deep = true;
        row.grammar_raw = winner.grammar_raw;
        row.grammar_norm = ordering::normalize(winner.grammar_raw);
        row.best_order = winner.order.clone();
        row.structure_norm = winner.structure_norm;
        row.valency_norm = winner.valency_norm;
        row.syntax_coverage = winner.syntax_coverage;
        row.phrase_kind = winner.phrase_kind.clone();
        row.final_score = score_final(row);
        row.base_final = row.final_score;
        row.alternatives = alternatives;
        completed += 1;
        observer(completed, evaluated);
    }
    Ok(evaluated)
}

pub fn rank_buckets(rows: &[Row]) -> BTreeMap<usize, Vec<usize>> {
    rank_buckets_controlled(rows, &crate::control::Control::default()).expect("unlimited control")
}

/// Workers own independent rows; only this coordinator mutates the result slice
/// and emits aggregate progress. Scoped joins prevent cancellation from orphaning work.
pub fn deep_analyze_parallel(
    rows: &mut [Row],
    selected: &BTreeSet<usize>,
    lex: &WordNet,
    options: &Options,
    workers: usize,
    control: &crate::control::Control,
    observer: &mut dyn FnMut(usize, usize),
) -> Result<usize, &'static str> {
    if workers > 32 || selected.iter().any(|i| *i >= rows.len()) {
        return Err("invalid parallel ranking options");
    }
    let workers = if workers == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(32))
    } else {
        workers
    };
    if workers == 1 || selected.len() < 2 {
        return deep_analyze_observed(rows, selected, lex, options, control, observer);
    }
    control.check()?;
    let tasks: Vec<_> = selected.iter().map(|&i| (i, rows[i].clone())).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let (tx, rx) = std::sync::mpsc::channel();
    let mut counts = vec![(0, 0); workers];
    let mut failure = None;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..workers.min(tasks.len()) {
            let tx = tx.clone();
            let tasks = &tasks;
            let next = &next;
            handles.push(scope.spawn(move || {
                let mut completed = 0;
                let mut evaluated = 0;
                loop {
                    control.check()?;
                    let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((index, source)) = tasks.get(n) else {
                        break;
                    };
                    let mut row = [source.clone()];
                    let result = deep_analyze_observed(
                        &mut row,
                        &BTreeSet::from([0]),
                        lex,
                        options,
                        control,
                        &mut |c, e| {
                            let _ = tx.send((worker, completed + c, evaluated + e, None));
                        },
                    );
                    evaluated += result?;
                    completed += usize::from(row[0].deep);
                    let _ = tx.send((worker, completed, evaluated, Some((*index, row[0].clone()))));
                }
                Ok::<_, &'static str>(())
            }));
        }
        drop(tx);
        for (worker, completed, evaluated, row) in rx {
            counts[worker] = (completed, evaluated);
            if let Some((index, row)) = row {
                rows[index] = row;
            }
            observer(
                counts.iter().map(|c| c.0).sum(),
                counts.iter().map(|c| c.1).sum(),
            );
        }
        for handle in handles {
            if let Err(e) = handle.join().unwrap_or(Err("ranking worker panicked")) {
                failure = Some(e);
            }
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    control.check()?;
    Ok(counts.iter().map(|c| c.1).sum())
}

/// Refine each deep row's best seed, retaining the original alternatives for
/// subsequent corpus/model selection. The per-row search is strictly bounded.
pub fn refine_rows(
    rows: &mut [Row],
    lex: &WordNet,
    control: &crate::control::Control,
) -> Result<usize, crate::request::Error> {
    let mut evaluated = 0;
    for row in rows.iter_mut().filter(|r| r.deep) {
        let mut scorer = |words: &[String]| {
            let raw = crate::structure::local_raw(words, lex);
            ordering::score(words.to_vec(), raw, lex).objective
        };
        let refined = crate::refinement::refine_controlled(
            &row.best_order,
            &mut scorer,
            crate::refinement::Options::default(),
            None,
            control,
        )
        .map_err(|e| crate::request::Error::new("refinement_error", e))?;
        evaluated += refined.evaluated;
        if refined.improved {
            let raw = crate::structure::local_raw(&refined.order, lex);
            let winner = ordering::score(refined.order, raw, lex);
            row.best_order = winner.order.clone();
            row.grammar_raw = winner.grammar_raw;
            row.grammar_norm = winner.grammar_norm;
            row.structure_norm = winner.structure_norm;
            row.valency_norm = winner.valency_norm;
            row.syntax_coverage = winner.syntax_coverage;
            row.phrase_kind = winner.phrase_kind.clone();
            row.final_score = score_final(row);
            row.base_final = row.final_score;
            row.alternatives.push(winner);
        }
    }
    Ok(evaluated)
}

pub fn rank_buckets_controlled(
    rows: &[Row],
    control: &crate::control::Control,
) -> Result<BTreeMap<usize, Vec<usize>>, &'static str> {
    control.check()?;
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        control.check()?;
        buckets.entry(row.input.word_count).or_default().push(i);
    }
    for bucket in buckets.values_mut() {
        control.sort_by(bucket, |&i, &j| {
            let a = &rows[i];
            let b = &rows[j];
            b.deep
                .cmp(&a.deep)
                .then_with(|| {
                    (if b.deep { b.final_score } else { b.pre_score })
                        .total_cmp(&(if a.deep { a.final_score } else { a.pre_score }))
                })
                .then_with(|| b.pre_score.total_cmp(&a.pre_score))
        })?;
    }
    Ok(buckets)
}
