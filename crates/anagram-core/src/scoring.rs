//! Corpus evidence primitives; these do not substitute for grammar ranking.
use crate::{
    lexicon::{Unigrams, decoded_lines},
    normalize_letters,
};
use serde::Serialize;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    io::{self, BufRead},
};

#[derive(Debug, Serialize)]
pub struct Lexical {
    pub lex_raw: f64,
    pub avg_zipf: f64,
    pub min_zipf: f64,
    pub junk_penalty: f64,
}

pub fn lexical(words: &[String], unigrams: Option<&Unigrams>, short: &BTreeSet<String>) -> Lexical {
    if words.is_empty() {
        return Lexical {
            lex_raw: 0.0,
            avg_zipf: 0.0,
            min_zipf: 0.0,
            junk_penalty: 0.0,
        };
    }
    let mut scores: Vec<_> = words
        .iter()
        .map(|w| unigrams.map_or(0.0, |u| u.zipf(w)))
        .collect();
    let avg_zipf = scores.iter().sum::<f64>() / scores.len() as f64;
    scores.sort_by(f64::total_cmp);
    let low_count = scores.len().min(2);
    let low_tail = scores[..low_count].iter().sum::<f64>() / low_count as f64;
    let mut junk_penalty = 0.0;
    for word in words {
        if word.len() <= 2 && !short.contains(word) {
            junk_penalty += 0.75;
        }
        if unigrams.is_some_and(|u| u.count(word) == 0) {
            junk_penalty += 0.35;
        }
    }
    let unique: HashSet<_> = words.iter().collect();
    junk_penalty += 1.25 * (words.len() - unique.len()) as f64;
    Lexical {
        lex_raw: 0.78 * avg_zipf + 0.22 * low_tail - junk_penalty,
        avg_zipf,
        min_zipf: scores[0],
        junk_penalty,
    }
}

pub struct Bigrams<'a> {
    pub unigrams: &'a Unigrams,
    counts: HashMap<(String, String), i64>,
}

impl<'a> Bigrams<'a> {
    pub fn load(
        reader: impl BufRead,
        unigrams: &'a Unigrams,
        vocabulary: &BTreeSet<String>,
    ) -> io::Result<Self> {
        let mut counts: HashMap<(String, String), i64> = HashMap::new();
        for line in decoded_lines(reader) {
            let line = line?;
            let line = line.trim();
            let pair = line.rsplit_once('\t').or_else(|| {
                line.rfind(char::is_whitespace)
                    .map(|i| (&line[..i], line[i..].trim()))
            });
            let Some((phrase, count)) = pair else {
                continue;
            };
            let Ok(count) = count.trim().parse::<i64>() else {
                continue;
            };
            let words: Vec<_> = phrase.split_whitespace().collect();
            if words.len() != 2 {
                continue;
            }
            let left = normalize_letters(words[0]);
            let right = normalize_letters(words[1]);
            if left.is_empty()
                || right.is_empty()
                || !vocabulary.contains(&left)
                || !vocabulary.contains(&right)
            {
                continue;
            }
            let value = counts.entry((left, right)).or_default();
            *value = value.checked_add(count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "bigram count overflow")
            })?;
        }
        if counts.values().any(|count| *count < 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "negative bigram count",
            ));
        }
        Ok(Self { unigrams, counts })
    }

    pub fn count(&self, left: &str, right: &str) -> i64 {
        self.counts
            .get(&(normalize_letters(left), normalize_letters(right)))
            .copied()
            .unwrap_or(0)
    }

    pub fn edge(&self, left: &str, right: &str) -> f64 {
        let count = self.count(left, right) as f64;
        let left = self.unigrams.count(left).max(1) as f64;
        let right = self.unigrams.count(right).max(1) as f64;
        let conditional = (count + 0.25).log10() - (left + 1.0).log10();
        let association =
            (count + 0.25).log10() - 0.5 * ((left + 1.0).log10() + (right + 1.0).log10());
        0.72 * conditional + 0.28 * association
    }

    pub fn pair_potential(&self, words: &[String]) -> (f64, f64) {
        if words.len() <= 1 {
            return (0.0, 0.0);
        }
        let mut edges = Vec::new();
        for (i, left) in words.iter().enumerate() {
            for (j, right) in words.iter().enumerate() {
                if i != j {
                    edges.push((self.edge(left, right), self.count(left, right) > 0));
                }
            }
        }
        // Stable ordering preserves coverage selection when scores tie.
        edges.sort_by(|a, b| b.0.total_cmp(&a.0));
        let chosen = &edges[..words.len() - 1];
        (
            chosen.iter().map(|e| e.0).sum::<f64>() / chosen.len() as f64,
            chosen.iter().filter(|e| e.1).count() as f64 / chosen.len() as f64,
        )
    }

    /// Exact path DP with reference traversal/tie behavior. Reject oversized
    /// requests rather than allocating an exponential table without a bound.
    pub fn best_order(&self, words: &[String]) -> Result<(f64, Vec<String>, f64), &'static str> {
        let n = words.len();
        if n <= 1 {
            return Ok((0.0, words.to_vec(), 0.0));
        }
        if n > 16 {
            return Err("exact bigram ordering supports at most 16 words");
        }
        #[derive(Clone)]
        struct Path {
            score: f64,
            seen: usize,
            indices: Vec<usize>,
        }
        let mut dp: Vec<Option<Path>> = vec![None; (1 << n) * n];
        for i in 0..n {
            dp[(1 << i) * n + i] = Some(Path {
                score: 0.0,
                seen: 0,
                indices: vec![i],
            });
        }
        let full = (1 << n) - 1;
        let mut edges = vec![vec![0.0; n]; n];
        let mut observed = vec![vec![false; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    edges[i][j] = self.edge(&words[i], &words[j]);
                    observed[i][j] = self.count(&words[i], &words[j]) > 0;
                }
            }
        }
        for mask in 1..=full {
            for last in 0..n {
                let Some(path) = dp[mask * n + last].clone() else {
                    continue;
                };
                for next in 0..n {
                    if mask & (1 << next) != 0 {
                        continue;
                    }
                    let score = path.score + edges[last][next];
                    let seen = path.seen + usize::from(observed[last][next]);
                    let slot = &mut dp[(mask | (1 << next)) * n + next];
                    if slot
                        .as_ref()
                        .is_none_or(|old| (score, seen) > (old.score, old.seen))
                    {
                        let mut indices = path.indices.clone();
                        indices.push(next);
                        *slot = Some(Path {
                            score,
                            seen,
                            indices,
                        });
                    }
                }
            }
        }
        let mut best: Option<Path> = None;
        for last in 0..n {
            let path = dp[full * n + last].take().unwrap();
            if best
                .as_ref()
                .is_none_or(|old| (path.score, path.seen) > (old.score, old.seen))
            {
                best = Some(path);
            }
        }
        let best = best.unwrap();
        Ok((
            best.score / (n - 1) as f64,
            best.indices.iter().map(|i| words[*i].clone()).collect(),
            best.seen as f64 / (n - 1) as f64,
        ))
    }
}

pub fn percentiles(values: &[f64]) -> Vec<f64> {
    if values.len() <= 1 {
        return vec![1.0; values.len()];
    }
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
    let mut result = vec![0.0; values.len()];
    let mut pos = 0;
    while pos < values.len() {
        let mut end = pos + 1;
        while end < values.len() && values[indices[end]] == values[indices[pos]] {
            end += 1;
        }
        let rank = (pos + end - 1) as f64 / 2.0 / (values.len() - 1) as f64;
        for index in &indices[pos..end] {
            result[*index] = rank;
        }
        pos = end;
    }
    result
}

pub fn morph_root(
    word: &str,
    vocabulary: &BTreeSet<String>,
    unigrams: Option<&Unigrams>,
) -> String {
    let word = normalize_letters(word);
    let irregular = match word.as_str() {
        "does" | "did" | "done" => Some("do"),
        "has" | "had" => Some("have"),
        "was" | "were" => Some("be"),
        "went" | "gone" => Some("go"),
        _ => None,
    };
    if let Some(root) = irregular {
        return root.into();
    }
    if word.len() <= 3
        || [
            "as", "gas", "has", "his", "is", "news", "this", "thus", "us", "was", "yes",
        ]
        .contains(&word.as_str())
    {
        return word;
    }
    let mut variants = Vec::new();
    let n = word.len();
    if word.ends_with("ies") && n > 4 {
        variants.extend([word[..n - 1].to_owned(), format!("{}y", &word[..n - 3])]);
    }
    if ["ches", "shes", "xes", "zes", "oes"]
        .iter()
        .any(|suffix| word.ends_with(suffix))
        && n > 4
    {
        variants.push(word[..n - 2].to_owned());
    }
    if word.ends_with('s') && !word.ends_with("ss") {
        variants.push(word[..n - 1].to_owned());
    }
    if word.ends_with("ied") && n > 4 {
        variants.extend([word[..n - 1].to_owned(), format!("{}y", &word[..n - 3])]);
    } else if word.ends_with("ed") && n > 4 {
        variants.extend([word[..n - 1].to_owned(), word[..n - 2].to_owned()]);
    }
    if word.ends_with("ing") && n > 5 {
        variants.extend([word[..n - 3].to_owned(), format!("{}e", &word[..n - 3])]);
    }
    let mut best: Option<String> = None;
    for variant in variants.into_iter().filter(|v| vocabulary.contains(v)) {
        let evidence = |v: &str| {
            (
                unigrams.map_or(0, |u| u.count(v)),
                std::cmp::Reverse(v.len()),
            )
        };
        if best
            .as_ref()
            .is_none_or(|old| evidence(&variant) > evidence(old))
        {
            best = Some(variant);
        }
    }
    best.unwrap_or(word)
}

#[derive(Serialize)]
pub struct PreRecord {
    pub words: Vec<String>,
    pub matched_hints: Vec<String>,
    pub lexical: Lexical,
    pub family: Vec<String>,
    pub family_best_lex: f64,
    pub family_size: usize,
    pub hint_info: f64,
    pub pair_raw: f64,
    pub pair_coverage: f64,
    pub lex_pct: f64,
    pub family_pct: f64,
    pub pair_pct: f64,
    pub pre_score: f64,
}

/// Pre-analysis scores are normalized within word-count buckets, never across
/// unlike lengths. This stage is not the grammar-aware final ranking.
pub fn pre_rank(
    bags: &[Vec<String>],
    hints: &BTreeSet<String>,
    unigrams: Option<&Unigrams>,
    bigrams: Option<&Bigrams<'_>>,
    vocabulary: &BTreeSet<String>,
    short: &BTreeSet<String>,
) -> Vec<PreRecord> {
    pre_rank_controlled(
        bags,
        hints,
        unigrams,
        bigrams,
        vocabulary,
        short,
        &crate::control::Control::default(),
    )
    .expect("unlimited control")
}
pub fn pre_rank_controlled(
    bags: &[Vec<String>],
    hints: &BTreeSet<String>,
    unigrams: Option<&Unigrams>,
    bigrams: Option<&Bigrams<'_>>,
    vocabulary: &BTreeSet<String>,
    short: &BTreeSet<String>,
    control: &crate::control::Control,
) -> Result<Vec<PreRecord>, &'static str> {
    control.check()?;
    let mut records: Vec<_> = bags
        .iter()
        .map(|words| {
            control.check()?;
            let mut family: Vec<_> = words
                .iter()
                .map(|w| morph_root(w, vocabulary, unigrams))
                .collect();
            family.sort();
            let (pair_raw, pair_coverage) = bigrams.map_or((0.0, 0.0), |b| b.pair_potential(words));
            Ok(PreRecord {
                words: words.clone(),
                matched_hints: hints
                    .iter()
                    .filter(|h| words.contains(h))
                    .cloned()
                    .collect(),
                lexical: lexical(words, unigrams, short),
                family,
                family_best_lex: 0.0,
                family_size: 1,
                hint_info: 0.0,
                pair_raw,
                pair_coverage,
                lex_pct: 0.0,
                family_pct: 0.0,
                pair_pct: 0.0,
                pre_score: 0.0,
            })
        })
        .collect::<Result<_, &'static str>>()?;
    let mut families: HashMap<Vec<String>, (f64, usize)> = HashMap::new();
    let mut hint_counts: HashMap<&str, usize> = HashMap::new();
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, record) in records.iter().enumerate() {
        control.check()?;
        let family = families
            .entry(record.family.clone())
            .or_insert((record.lexical.lex_raw, 0));
        family.0 = family.0.max(record.lexical.lex_raw);
        family.1 += 1;
        for hint in &record.matched_hints {
            *hint_counts.entry(hint).or_default() += 1;
        }
        buckets.entry(record.words.len()).or_default().push(index);
    }
    let raw_info: HashMap<String, f64> = hints
        .iter()
        .map(|h| {
            (
                h.clone(),
                ((records.len() as f64 + 1.0)
                    / (*hint_counts.get(h.as_str()).unwrap_or(&0) as f64 + 1.0))
                    .ln(),
            )
        })
        .collect();
    let max_info = raw_info
        .values()
        .copied()
        .reduce(f64::max)
        .filter(|v| *v != 0.0)
        .unwrap_or(1.0);
    for record in &mut records {
        control.check()?;
        (record.family_best_lex, record.family_size) = families[&record.family];
        if !record.matched_hints.is_empty() {
            let best = record
                .matched_hints
                .iter()
                .map(|h| raw_info[h])
                .reduce(f64::max)
                .unwrap();
            let multi = (0.05 * (record.matched_hints.len() - 1) as f64).min(0.15);
            record.hint_info = (best / max_info + multi).min(1.0);
        }
    }
    for bucket in buckets.values() {
        control.check()?;
        let lex = percentiles(
            &bucket
                .iter()
                .map(|i| records[*i].lexical.lex_raw)
                .collect::<Vec<_>>(),
        );
        let fam = percentiles(
            &bucket
                .iter()
                .map(|i| records[*i].family_best_lex)
                .collect::<Vec<_>>(),
        );
        let pair = percentiles(
            &bucket
                .iter()
                .map(|i| records[*i].pair_raw)
                .collect::<Vec<_>>(),
        );
        for (position, index) in bucket.iter().enumerate() {
            control.check()?;
            let record = &mut records[*index];
            record.lex_pct = lex[position];
            record.family_pct = fam[position];
            record.pair_pct = pair[position];
            record.pre_score = 100.0
                * (0.26 * record.lex_pct
                    + 0.12 * record.family_pct
                    + 0.46 * record.pair_pct
                    + 0.16 * record.hint_info);
        }
    }
    control.check()?;
    Ok(records)
}
