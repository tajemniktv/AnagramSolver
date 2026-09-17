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
    fn rounded(value: f64, precision: usize) -> f64 {
        format!("{value:.precision$}")
            .parse()
            .expect("finite score formatting")
    }
    let mut buckets: BTreeMap<usize, Vec<&crate::scoring::PreRecord>> = BTreeMap::new();
    for record in records {
        buckets.entry(record.words.len()).or_default().push(record);
    }
    let mut inputs = Vec::with_capacity(records.len());
    for bucket in buckets.values_mut() {
        bucket.sort_by(|a, b| {
            b.pre_score
                .total_cmp(&a.pre_score)
                .then_with(|| b.lexical.lex_raw.total_cmp(&a.lexical.lex_raw))
                .then_with(|| b.words.join(" ").cmp(&a.words.join(" ")))
        });
        for (i, record) in bucket.iter().enumerate() {
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
    inputs
}

#[derive(Clone, Debug, Serialize, schemars::JsonSchema)]
pub struct Row {
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
    rows.sort_by(|a, b| {
        a.input
            .word_count
            .cmp(&b.input.word_count)
            .then_with(|| a.input.words.cmp(&b.input.words))
    });
    control.check()?;
    Ok(rows)
}

/// `per_group` is the initial shortlist size, not a hard deployment work cap:
/// family expansion intentionally includes every selected inflection sibling.
pub fn choose_deep(rows: &[Row], per_group: usize, deep_all: bool) -> BTreeSet<usize> {
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        groups.entry(row.input.word_count).or_default().push(i);
    }
    let mut families = BTreeSet::new();
    for bucket in groups.values_mut() {
        bucket.sort_by(|&i, &j| {
            rows[j]
                .pre_score
                .total_cmp(&rows[i].pre_score)
                .then_with(|| rows[j].input.fam.total_cmp(&rows[i].input.fam))
                .then_with(|| rows[j].input.lex.total_cmp(&rows[i].input.lex))
        });
        for &i in bucket
            .iter()
            .take(if deep_all { bucket.len() } else { per_group })
        {
            families.insert((rows[i].input.word_count, rows[i].family_key.clone()));
        }
    }
    rows.iter()
        .enumerate()
        .filter_map(|(i, row)| {
            families
                .contains(&(row.input.word_count, row.family_key.clone()))
                .then_some(i)
        })
        .collect()
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
        let (orders, count) = ordering::rank_controlled(
            &row.input.words,
            lex,
            exact,
            options.beam_width,
            raw_k,
            control,
        )?;
        let Some(winner) = orders.first() else {
            continue;
        };
        evaluated += count;
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
        row.alternatives = diversity::select(
            &orders,
            options.retained_orders,
            diversity::QUALITY_CORE,
            0.12,
        )?;
        completed += 1;
        observer(completed, evaluated);
    }
    Ok(evaluated)
}

pub fn rank_buckets(rows: &[Row]) -> BTreeMap<usize, Vec<usize>> {
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter().enumerate() {
        buckets.entry(row.input.word_count).or_default().push(i);
    }
    for bucket in buckets.values_mut() {
        bucket.sort_by(|&i, &j| {
            let a = &rows[i];
            let b = &rows[j];
            b.deep
                .cmp(&a.deep)
                .then_with(|| {
                    (if b.deep { b.final_score } else { b.pre_score })
                        .total_cmp(&(if a.deep { a.final_score } else { a.pre_score }))
                })
                .then_with(|| b.pre_score.total_cmp(&a.pre_score))
        });
    }
    buckets
}
