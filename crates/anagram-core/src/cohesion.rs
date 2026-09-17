//! Positive-only non-overlapping corpus-span segmentation.
use serde::Serialize;
use std::{cmp::Ordering, collections::HashMap};

#[derive(Clone, Debug, Serialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub count: i64,
    pub strength: f64,
}
#[derive(Debug, Default, Serialize)]
pub struct Evidence {
    pub score: f64,
    pub coverage: f64,
    pub longest_fraction: f64,
    pub segments: usize,
    pub splice_penalty: f64,
    pub frequency_strength: f64,
    pub spans: Vec<Span>,
}
#[derive(Clone, Default)]
struct Path {
    utility: f64,
    covered: usize,
    frequency_sum: f64,
    spans: Vec<Span>,
}
impl Path {
    fn longest(&self) -> usize {
        self.spans
            .iter()
            .map(|s| s.end - s.start)
            .max()
            .unwrap_or(0)
    }
    fn compare(&self, other: &Self) -> Ordering {
        self.utility
            .total_cmp(&other.utility)
            .then_with(|| self.covered.cmp(&other.covered))
            .then_with(|| other.spans.len().cmp(&self.spans.len()))
            .then_with(|| self.longest().cmp(&other.longest()))
            .then_with(|| self.frequency_sum.total_cmp(&other.frequency_sum))
            .then_with(|| {
                self.spans
                    .iter()
                    .map(|s| &s.text)
                    .cmp(other.spans.iter().map(|s| &s.text))
            })
    }
}

pub fn score(words: &[String], counts: &HashMap<String, i64>, max_n: usize) -> Evidence {
    let n = words.len();
    if n < 2 || max_n < 2 || words.iter().any(String::is_empty) {
        return Evidence::default();
    }
    let mut starts: Vec<Vec<Span>> = vec![Vec::new(); n];
    for length in 2..=max_n.min(n) {
        for start in 0..=n - length {
            let text = words[start..start + length].join(" ");
            let count = *counts.get(&text).unwrap_or(&0);
            if count > 0 {
                starts[start].push(Span {
                    start,
                    end: start + length,
                    text,
                    count,
                    strength: ((count as f64 + 1.0).log10() / 5.0).min(1.0),
                });
            }
        }
    }
    let mut dp: Vec<Option<Path>> = vec![None; n + 1];
    dp[0] = Some(Path::default());
    for position in 0..n {
        let Some(current) = dp[position].clone() else {
            continue;
        };
        if dp[position + 1]
            .as_ref()
            .is_none_or(|old| current.compare(old).is_gt())
        {
            dp[position + 1] = Some(current.clone());
        }
        for span in &starts[position] {
            let mut candidate = current.clone();
            let length = span.end - span.start;
            candidate.utility = current.utility + length as f64 - 0.55 + 0.12 * span.strength;
            candidate.covered += length;
            candidate.frequency_sum += span.strength;
            candidate.spans.push(span.clone());
            if dp[span.end]
                .as_ref()
                .is_none_or(|old| candidate.compare(old).is_gt())
            {
                dp[span.end] = Some(candidate);
            }
        }
    }
    let Some(best) = dp[n].take().filter(|p| !p.spans.is_empty()) else {
        return Evidence::default();
    };
    let coverage = best.covered as f64 / n as f64;
    let longest_fraction = best.longest() as f64 / n as f64;
    let segments = best.spans.len();
    let frequency_strength = best.frequency_sum / segments as f64;
    let splice_penalty = (segments - 1) as f64 / (n - 1) as f64;
    let shape = 0.52 + 0.30 * longest_fraction + 0.18 * frequency_strength;
    Evidence {
        score: (coverage * shape * (1.0 - 0.22 * splice_penalty)).clamp(0.0, 1.0),
        coverage,
        longest_fraction,
        segments,
        splice_penalty,
        frequency_strength,
        spans: best.spans,
    }
}

pub fn blend(base: f64, evidence: &Evidence) -> f64 {
    base.max(0.92 * evidence.score).min(1.0)
}
