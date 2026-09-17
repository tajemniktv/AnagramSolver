//! Phrase attestation from corpus counts, independent of storage ownership.
use crate::cohesion;
use std::collections::HashMap;

pub fn queries(words: &[String], max_n: usize) -> Vec<String> {
    if words.is_empty() {
        return Vec::new();
    }
    let mut queries = vec![words.join(" ")];
    for n in 2..=max_n.min(words.len()) {
        queries.extend(words.windows(n).map(|window| window.join(" ")));
    }
    queries
}

pub fn score(
    words: &[String],
    counts: &HashMap<String, i64>,
    max_n: usize,
    with_cohesion: bool,
) -> (f64, HashMap<String, f64>) {
    if words.is_empty() {
        return (0.0, HashMap::new());
    }
    let whole_count = *counts.get(&words.join(" ")).unwrap_or(&0);
    let exact = if whole_count > 0 {
        (0.72 + 0.28 * (whole_count as f64 + 1.0).log10() / 5.0).min(1.0)
    } else {
        0.0
    };
    let mut longer_sum = 0.0;
    let mut longer_count = 0;
    let mut bi_hits = 0;
    let mut bi_count = 0;
    for n in 2..=max_n.min(words.len()) {
        for window in words.windows(n) {
            let count = *counts.get(&window.join(" ")).unwrap_or(&0);
            if n == 2 {
                bi_count += 1;
                bi_hits += usize::from(count > 0);
            } else {
                longer_count += 1;
                if count > 0 {
                    longer_sum += (0.55 + 0.45 * (count as f64 + 1.0).log10() / 5.0).min(1.0);
                }
            }
        }
    }
    let longer = if longer_count > 0 {
        longer_sum / longer_count as f64
    } else {
        0.0
    };
    let bi_cov = if bi_count > 0 {
        bi_hits as f64 / bi_count as f64
    } else {
        0.0
    };
    let base = exact.max(0.76 * longer).max(0.34 * bi_cov).min(1.0);
    let score = if with_cohesion {
        cohesion::blend(base, &cohesion::score(words, counts, max_n))
    } else {
        base
    };
    (
        score,
        HashMap::from([
            ("whole_count".into(), whole_count as f64),
            ("longer".into(), longer),
            ("bigram_coverage".into(), bi_cov),
        ]),
    )
}
