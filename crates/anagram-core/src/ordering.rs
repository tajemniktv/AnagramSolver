//! Deterministic retained-order search before diversity/refinement extensions.
use crate::{grammar, structure, wordnet::WordNet};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub order: Vec<String>,
    pub grammar_raw: f64,
    pub grammar_norm: f64,
    pub structure_norm: f64,
    pub valency_norm: f64,
    pub syntax_coverage: f64,
    pub phrase_kind: String,
    pub objective: f64,
}
pub fn normalize(raw: f64) -> f64 {
    1.0 / (1.0 + (-((raw - 0.85) * 1.25)).exp())
}
pub fn compare(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.objective
        .total_cmp(&a.objective)
        .then_with(|| b.structure_norm.total_cmp(&a.structure_norm))
        .then_with(|| b.grammar_norm.total_cmp(&a.grammar_norm))
        .then_with(|| b.valency_norm.total_cmp(&a.valency_norm))
        .then_with(|| b.syntax_coverage.total_cmp(&a.syntax_coverage))
        .then_with(|| a.order.cmp(&b.order))
}
pub fn score(order: Vec<String>, raw: f64, lex: &WordNet) -> Candidate {
    let s = structure::evaluate(&order, lex);
    let grammar_norm = normalize(raw);
    Candidate {
        order,
        grammar_raw: raw,
        grammar_norm,
        structure_norm: s.norm,
        valency_norm: s.valency,
        syntax_coverage: s.coverage,
        phrase_kind: s.kind,
        objective: 0.38 * grammar_norm + 0.44 * s.norm + 0.12 * s.valency + 0.06 * s.coverage,
    }
}

#[derive(Clone)]
struct Path {
    score: f64,
    indices: Vec<usize>,
}
struct State {
    mask: usize,
    last: usize,
    paths: Vec<Path>,
}

fn kbest(
    pair: &[Vec<f64>],
    starts: &[f64],
    ends: &[f64],
    max_complete: usize,
    control: &crate::control::Control,
) -> Result<Vec<Vec<usize>>, &'static str> {
    let n = starts.len();
    let per_state = 2.max(max_complete.div_ceil(n));
    let mut states: Vec<_> = (0..n)
        .map(|i| State {
            mask: 1 << i,
            last: i,
            paths: vec![Path {
                score: starts[i],
                indices: vec![i],
            }],
        })
        .collect();
    for _ in 1..n {
        // Preserve Python dict insertion order for exact ties in per-state pruning.
        let mut next: Vec<State> = Vec::new();
        let mut lookup = HashMap::new();
        for state in states {
            control.check()?;
            for path in state.paths {
                control.check()?;
                for (j, edge) in pair[state.last].iter().enumerate() {
                    if state.mask & (1 << j) != 0 {
                        continue;
                    }
                    let key = (state.mask | (1 << j), j);
                    let index = *lookup.entry(key).or_insert_with(|| {
                        next.push(State {
                            mask: key.0,
                            last: j,
                            paths: Vec::new(),
                        });
                        next.len() - 1
                    });
                    let mut indices = path.indices.clone();
                    indices.push(j);
                    next[index].paths.push(Path {
                        score: path.score + edge,
                        indices,
                    });
                }
            }
        }
        for state in &mut next {
            control.sort_by(&mut state.paths, |a, b| b.score.total_cmp(&a.score))?;
            state.paths.truncate(per_state);
        }
        states = next;
    }
    let mut complete = Vec::new();
    for state in states {
        for mut path in state.paths {
            control.check()?;
            path.score += ends[state.last];
            complete.push(path);
        }
    }
    control.sort_by(&mut complete, |a, b| b.score.total_cmp(&a.score))?;
    complete.truncate(max_complete);
    control.check()?;
    Ok(complete.into_iter().map(|p| p.indices).collect())
}

fn permutations(
    n: usize,
    path: &mut Vec<usize>,
    used: usize,
    visit: &mut impl FnMut(&[usize]) -> Result<(), &'static str>,
) -> Result<(), &'static str> {
    if path.len() == n {
        return visit(path);
    }
    for i in 0..n {
        if used & (1 << i) == 0 {
            path.push(i);
            permutations(n, path, used | (1 << i), visit)?;
            path.pop();
        }
    }
    Ok(())
}

/// The initial native pool implementation. Final public ranking additionally
/// needs order-diversity/refinement and corpus evidence before cutover.
pub fn rank(
    words: &[String],
    lex: &WordNet,
    exact: bool,
    beam_width: usize,
    top_k: usize,
) -> Result<(Vec<Candidate>, usize), &'static str> {
    rank_controlled(
        words,
        lex,
        exact,
        beam_width,
        top_k,
        &crate::control::Control::default(),
    )
}

pub fn rank_controlled(
    words: &[String],
    lex: &WordNet,
    exact: bool,
    beam_width: usize,
    top_k: usize,
    control: &crate::control::Control,
) -> Result<(Vec<Candidate>, usize), &'static str> {
    rank_observed(words, lex, exact, beam_width, top_k, control, &mut |_| {})
}

pub fn rank_observed(
    words: &[String],
    lex: &WordNet,
    exact: bool,
    beam_width: usize,
    top_k: usize,
    control: &crate::control::Control,
    observer: &mut dyn FnMut(usize),
) -> Result<(Vec<Candidate>, usize), &'static str> {
    control.check()?;
    if top_k == 0 || beam_width == 0 {
        return Err("ordering budgets must be positive");
    }
    if words.len() > 10 {
        return Err("native order pool currently supports at most ten words");
    }
    let mut words = words.to_vec();
    words.sort();
    let n = words.len();
    if n == 0 {
        return Ok((Vec::new(), 0));
    }
    if n == 1 {
        let raw = structure::local_raw(&words, lex);
        let candidate = score(words, raw, lex);
        observer(1);
        control.check()?;
        return Ok((vec![candidate], 1));
    }
    let starts: Vec<_> = words.iter().map(|w| grammar::start(w, lex)).collect();
    let ends: Vec<_> = words.iter().map(|w| grammar::end(w, lex)).collect();
    let pair: Vec<Vec<_>> = words
        .iter()
        .enumerate()
        .map(|(i, a)| {
            words
                .iter()
                .enumerate()
                .map(|(j, b)| {
                    if i == j {
                        0.0
                    } else {
                        structure::pair(a, b, lex)
                    }
                })
                .collect()
        })
        .collect();
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    let mut visit = |indices: &[usize]| {
        control.check()?;
        let order: Vec<_> = indices.iter().map(|i| words[*i].clone()).collect();
        if !seen.insert(order.clone()) {
            return Ok(());
        }
        // Modern Python sum compensates edge accumulation. One-ULP differences
        // here can change grammar tie-breaks even when objectives compare equal.
        let edges = crate::compensated_sum(indices.windows(2).map(|edge| pair[edge[0]][edge[1]]));
        let raw = (starts[indices[0]] + ends[*indices.last().unwrap()] + edges) / (n - 1) as f64;
        candidates.push(score(order, raw, lex));
        observer(candidates.len());
        control.check()?;
        Ok(())
    };
    if exact {
        permutations(n, &mut Vec::new(), 0, &mut visit)?;
    } else {
        let width = beam_width.max(top_k.checked_mul(8).ok_or("ordering budget overflow")?);
        for order in kbest(&pair, &starts, &ends, width, control)? {
            visit(&order)?;
        }
    }
    let evaluated = candidates.len();
    control.sort_by(&mut candidates, compare)?;
    candidates.truncate(top_k);
    control.check()?;
    Ok((candidates, evaluated))
}
