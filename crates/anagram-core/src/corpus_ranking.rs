//! Positive corpus evidence and three-channel admission for alternative orders.
use crate::{
    lexicon::decoded_lines,
    normalize_letters,
    ordering::Candidate,
    ranking::{self, Row},
};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    io::{self, BufRead},
};

#[derive(Default)]
pub struct Collocation {
    pub unigrams: HashMap<String, i64>,
    pub bigrams: HashMap<(String, String), i64>,
    pub total: i64,
}
fn parse(line: &str) -> Option<(&str, i64)> {
    let line = line.trim();
    let (token, count) = line.rsplit_once('\t').or_else(|| {
        line.rfind(char::is_whitespace)
            .map(|i| (&line[..i], line[i..].trim()))
    })?;
    Some((token, count.trim().parse().ok()?))
}
fn add(value: &mut i64, count: i64) -> io::Result<()> {
    *value = value
        .checked_add(count)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "corpus count overflow"))?;
    Ok(())
}
impl Collocation {
    /// Unlike generation frequency, total includes every parsed corpus token.
    pub fn load(
        one: impl BufRead,
        two: impl BufRead,
        vocabulary: &BTreeSet<String>,
    ) -> io::Result<Self> {
        let mut model = Self::default();
        for line in decoded_lines(one) {
            let line = line?;
            let Some((token, count)) = parse(&line) else {
                continue;
            };
            add(&mut model.total, count)?;
            let word = normalize_letters(token);
            if vocabulary.contains(&word) {
                add(model.unigrams.entry(word).or_default(), count)?;
            }
        }
        for line in decoded_lines(two) {
            let line = line?;
            let Some((text, count)) = parse(&line) else {
                continue;
            };
            let parts: Vec<_> = text.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }
            let left = normalize_letters(&parts[0].to_lowercase());
            let right = normalize_letters(&parts[1].to_lowercase());
            if vocabulary.contains(&left) && vocabulary.contains(&right) {
                add(model.bigrams.entry((left, right)).or_default(), count)?;
            }
        }
        Ok(model)
    }
    pub fn score(&self, words: &[String]) -> (f64, f64) {
        if words.len() <= 1 {
            return (0.0, 0.0);
        }
        let mut seen = 0;
        let mut sum = 0.0;
        for edge in words.windows(2) {
            let count = *self
                .bigrams
                .get(&(edge[0].clone(), edge[1].clone()))
                .unwrap_or(&0);
            if count <= 0 {
                continue;
            }
            seen += 1;
            let lc = (*self.unigrams.get(&edge[0]).unwrap_or(&1)).max(1) as f64;
            let rc = (*self.unigrams.get(&edge[1]).unwrap_or(&1)).max(1) as f64;
            let pmi = ((count as f64 * self.total.max(1) as f64) / (lc * rc))
                .max(1.0)
                .log10();
            let assoc = (pmi / 4.0).clamp(0.0, 1.0);
            let freq = ((count as f64 + 1.0).log10() / 7.0).clamp(0.0, 1.0);
            sum += 0.78 * assoc + 0.22 * freq;
        }
        let coverage = seen as f64 / (words.len() - 1) as f64;
        let mean = sum / (words.len() - 1) as f64;
        ((0.72 * mean + 0.28 * coverage).clamp(0.0, 1.0), coverage)
    }
}

/// Storage owns batching, read-only access and errors; policy owns scoring.
pub trait PhraseCorpus {
    fn counts(&self, phrases: &[String]) -> io::Result<HashMap<String, i64>>;
    fn max_n(&self) -> usize;
    fn score(&self, words: &[String]) -> io::Result<f64> {
        let queries = crate::phrase_evidence::queries(words, self.max_n());
        Ok(crate::phrase_evidence::score(words, &self.counts(&queries)?, self.max_n(), true).0)
    }
}

fn candidates(row: &Row) -> Vec<Candidate> {
    if !row.alternatives.is_empty() {
        return row.alternatives.clone();
    }
    vec![Candidate {
        order: row.best_order.clone(),
        grammar_raw: row.grammar_raw,
        grammar_norm: row.grammar_norm,
        structure_norm: row.structure_norm,
        valency_norm: row.valency_norm,
        syntax_coverage: row.syntax_coverage,
        phrase_kind: row.phrase_kind.clone(),
        objective: 0.38 * row.grammar_norm
            + 0.44 * row.structure_norm
            + 0.12 * row.valency_norm
            + 0.06 * row.syntax_coverage,
    }]
}

pub fn select(
    rows: &[Row],
    bucket: &[usize],
    collocation: Option<&Collocation>,
    phrase: Option<&dyn PhraseCorpus>,
    top: usize,
) -> io::Result<(Vec<usize>, usize)> {
    select_controlled(
        rows,
        bucket,
        collocation,
        phrase,
        top,
        &crate::control::Control::default(),
    )
}
fn select_controlled(
    rows: &[Row],
    bucket: &[usize],
    collocation: Option<&Collocation>,
    phrase: Option<&dyn PhraseCorpus>,
    top: usize,
    control: &crate::control::Control,
) -> io::Result<(Vec<usize>, usize)> {
    control.check_io()?;
    let mut by_final = bucket.to_vec();
    by_final.sort_by(|&i, &j| {
        rows[j]
            .final_score
            .total_cmp(&rows[i].final_score)
            .then_with(|| rows[j].pre_score.total_cmp(&rows[i].pre_score))
            .then_with(|| rows[i].input.words.cmp(&rows[j].input.words))
    });
    let mut by_pre = bucket.to_vec();
    by_pre.sort_by(|&i, &j| {
        rows[j]
            .pre_score
            .total_cmp(&rows[i].pre_score)
            .then_with(|| rows[j].final_score.total_cmp(&rows[i].final_score))
            .then_with(|| rows[i].input.words.cmp(&rows[j].input.words))
    });
    let mut chosen: Vec<_> = by_final.into_iter().take(top).collect();
    for i in by_pre.into_iter().take(top) {
        if !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    let mut pool: Vec<_> = bucket
        .iter()
        .copied()
        .filter(|i| !chosen.contains(i))
        .collect();
    let mut scores: HashMap<usize, f64> = pool.iter().map(|&i| (i, 0.0)).collect();
    let mut owners: HashMap<String, Vec<usize>> = HashMap::new();
    for &i in &pool {
        control.check_io()?;
        let orders = candidates(&rows[i]);
        if let Some(model) = collocation {
            scores.insert(i, 0.55 * model.score(&orders[0].order).0);
        }
        if phrase.is_some() {
            for order in orders {
                owners.entry(order.order.join(" ")).or_default().push(i);
            }
        }
    }
    if let Some(index) = phrase {
        for (text, count) in index.counts(&owners.keys().cloned().collect::<Vec<_>>())? {
            if count <= 0 {
                continue;
            }
            let exact = (0.72 + 0.28 * (count as f64 + 1.0).log10() / 5.0).min(1.0);
            if let Some(ids) = owners.get(&text) {
                for i in ids {
                    scores.entry(*i).and_modify(|s| *s = s.max(exact));
                }
            }
        }
    }
    pool.sort_by(|&i, &j| {
        scores[&j]
            .total_cmp(&scores[&i])
            .then_with(|| {
                rows[j]
                    .final_score
                    .max(rows[j].pre_score)
                    .total_cmp(&rows[i].final_score.max(rows[i].pre_score))
            })
            .then_with(|| rows[i].input.words.cmp(&rows[j].input.words))
    });
    let added: Vec<_> = pool
        .into_iter()
        .filter(|i| scores[i] > 0.0)
        .take(top)
        .collect();
    let corpus_added = added.len();
    chosen.extend(added);
    chosen.sort_by(|&i, &j| {
        rows[j]
            .final_score
            .max(rows[j].pre_score)
            .total_cmp(&rows[i].final_score.max(rows[i].pre_score))
            .then_with(|| rows[i].input.words.cmp(&rows[j].input.words))
    });
    Ok((chosen, corpus_added))
}

pub fn rescore(
    rows: &mut [Row],
    collocation: Option<&Collocation>,
    phrase: Option<&dyn PhraseCorpus>,
    top: usize,
    bonus_max: f64,
) -> io::Result<usize> {
    rescore_controlled(
        rows,
        collocation,
        phrase,
        top,
        bonus_max,
        &crate::control::Control::default(),
    )
}
pub fn rescore_controlled(
    rows: &mut [Row],
    collocation: Option<&Collocation>,
    phrase: Option<&dyn PhraseCorpus>,
    top: usize,
    bonus_max: f64,
    control: &crate::control::Control,
) -> io::Result<usize> {
    if !bonus_max.is_finite() || bonus_max < 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "phrase bonus must be finite and nonnegative",
        ));
    }
    let result = rescore_inner(rows, collocation, phrase, top, bonus_max, control);
    // Retained-order memory is released even on corpus errors, like the facade.
    for row in rows {
        row.alternatives.clear();
    }
    result
}
fn rescore_inner(
    rows: &mut [Row],
    collocation: Option<&Collocation>,
    phrase: Option<&dyn PhraseCorpus>,
    top: usize,
    bonus_max: f64,
    control: &crate::control::Control,
) -> io::Result<usize> {
    control.check_io()?;
    let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, row) in rows.iter_mut().enumerate() {
        control.check_io()?;
        row.input.words.sort();
        if row.deep {
            buckets.entry(row.input.word_count).or_default().push(i);
        }
    }
    let mut rescored = 0;
    for bucket in buckets.values() {
        let (chosen, _) = select_controlled(rows, bucket, collocation, phrase, top, control)?;
        for i in chosen {
            let row = &mut rows[i];
            let mut results = Vec::new();
            for candidate in candidates(row) {
                control.check_io()?;
                let colloc = collocation.map_or(0.0, |m| m.score(&candidate.order).0);
                let attest = match phrase {
                    Some(p) => p.score(&candidate.order)?,
                    None => 0.0,
                };
                let combined = (ranking::order_base_final(row, &candidate)
                    + bonus_max * attest.max(0.55 * colloc))
                .min(100.0);
                results.push((combined, attest, colloc, candidate));
            }
            if !results.iter().any(|r| r.1 > 0.0 || r.2 > 0.0) {
                row.base_final = row.final_score;
                row.colloc_norm = 0.0;
                row.phrase_attest_norm = 0.0;
                row.phrase_bonus = 0.0;
                rescored += 1;
                continue;
            }
            results.sort_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| b.1.total_cmp(&a.1))
                    .then_with(|| b.2.total_cmp(&a.2))
                    .then_with(|| (b.3.order == row.best_order).cmp(&(a.3.order == row.best_order)))
                    .then_with(|| a.3.order.cmp(&b.3.order))
            });
            let (combined, attest, colloc, winner) = results.remove(0);
            row.base_final = ranking::order_base_final(row, &winner);
            row.best_order = winner.order;
            row.grammar_raw = winner.grammar_raw;
            row.grammar_norm = winner.grammar_norm;
            row.structure_norm = winner.structure_norm;
            row.valency_norm = winner.valency_norm;
            row.syntax_coverage = winner.syntax_coverage;
            row.phrase_kind = winner.phrase_kind;
            row.colloc_norm = colloc;
            row.phrase_attest_norm = attest;
            row.phrase_bonus = combined - row.base_final;
            row.final_score = combined;
            rescored += 1;
        }
    }
    Ok(rescored)
}
