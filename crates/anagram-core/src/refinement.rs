//! Bounded deterministic window-permutation refinement; opt-in, not default ranking.
use crate::control::Control;
use serde::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy)]
pub struct Options {
    pub min_window: usize,
    pub max_window: usize,
    pub max_rounds: usize,
    pub max_evaluations: usize,
    pub epsilon: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            min_window: 3,
            max_window: 5,
            max_rounds: 3,
            max_evaluations: 512,
            epsilon: 1e-12,
        }
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct Result {
    pub order: Vec<String>,
    pub score: f64,
    pub evaluated: usize,
    pub rounds: usize,
    pub improved: bool,
}
#[derive(Debug, Serialize)]
pub struct Pool {
    pub candidates: Vec<Result>,
    pub evaluated: usize,
    pub improved_seeds: usize,
}

// Index-order recursion matches itertools.permutations, including repeated tokens.
// Returning false aborts immediately instead of materializing factorial neighbors.
fn permutations(
    control: &Control,
    window: &[String],
    used: &mut [bool],
    path: &mut Vec<String>,
    visit: &mut impl FnMut(&[String]) -> bool,
) -> bool {
    if control.check().is_err() {
        return false;
    }
    if path.len() == window.len() {
        return visit(path);
    }
    for i in 0..window.len() {
        if used[i] {
            continue;
        }
        used[i] = true;
        path.push(window[i].clone());
        let keep_going = permutations(control, window, used, path, visit);
        path.pop();
        used[i] = false;
        if !keep_going {
            return false;
        }
    }
    true
}

pub fn refine(
    seed: &[String],
    scorer: &mut impl FnMut(&[String]) -> f64,
    options: Options,
    initial_score: Option<f64>,
) -> std::result::Result<Result, &'static str> {
    refine_controlled(seed, scorer, options, initial_score, &Control::default())
}

/// Cooperative variant; scorers must bound their own individual calls.
pub fn refine_controlled(
    seed: &[String],
    scorer: &mut impl FnMut(&[String]) -> f64,
    options: Options,
    initial_score: Option<f64>,
    control: &Control,
) -> std::result::Result<Result, &'static str> {
    control.check()?;
    let o = options;
    if o.min_window < 2
        || o.max_window < o.min_window
        || o.max_evaluations == 0
        || !o.epsilon.is_finite()
        || o.epsilon < 0.0
    {
        return Err("invalid refinement options");
    }
    let mut result = Result {
        order: seed.to_vec(),
        score: initial_score.unwrap_or_else(|| scorer(seed)),
        evaluated: usize::from(initial_score.is_none()),
        rounds: 0,
        improved: false,
    };
    if !result.score.is_finite() {
        return Err("seed score must be finite");
    }
    let mut cache = HashMap::from([(result.order.clone(), result.score)]);
    control.check()?;
    while result.rounds < o.max_rounds && result.evaluated < o.max_evaluations {
        let mut best = result.order.clone();
        let mut best_score = result.score;
        let mut seen = HashSet::new();
        let mut invalid = false;
        'windows: for width in o.min_window..=o.max_window.min(result.order.len()) {
            for start in 0..=result.order.len() - width {
                let source = &result.order;
                let mut visit = |replacement: &[String]| {
                    if control.check().is_err() {
                        return false;
                    }
                    if replacement == &source[start..start + width] {
                        return true;
                    }
                    let mut candidate = source.clone();
                    candidate[start..start + width].clone_from_slice(replacement);
                    if !seen.insert(candidate.clone()) {
                        return true;
                    }
                    let score = if let Some(score) = cache.get(&candidate) {
                        *score
                    } else {
                        if result.evaluated >= o.max_evaluations {
                            return false;
                        }
                        let score = scorer(&candidate);
                        if control.check().is_err() {
                            return false;
                        }
                        if !score.is_finite() {
                            invalid = true;
                            return false;
                        }
                        cache.insert(candidate.clone(), score);
                        result.evaluated += 1;
                        score
                    };
                    if score > result.score + o.epsilon
                        && (best == *source
                            || score > best_score + o.epsilon
                            || ((score - best_score).abs() <= o.epsilon && candidate < best))
                    {
                        best = candidate;
                        best_score = score;
                    }
                    true
                };
                if !permutations(
                    control,
                    &source[start..start + width],
                    &mut vec![false; width],
                    &mut Vec::with_capacity(width),
                    &mut visit,
                ) {
                    break 'windows;
                }
            }
        }
        control.check()?;
        if invalid {
            return Err("candidate score must be finite");
        }
        if best == result.order {
            break;
        }
        result.order = best;
        result.score = best_score;
        result.rounds += 1;
        result.improved = true;
    }
    Ok(result)
}

fn sorted(
    values: HashMap<Vec<String>, Result>,
    control: &Control,
) -> std::result::Result<Vec<Result>, &'static str> {
    let mut results = Vec::with_capacity(values.len());
    for value in values.into_values() {
        control.check()?;
        results.push(value);
    }
    control.sort_by(&mut results, |a, b| {
        b.score
            .partial_cmp(&a.score)
            .expect("scores validated finite")
            .then_with(|| a.order.cmp(&b.order))
    })?;
    Ok(results)
}
fn merge(values: &mut HashMap<Vec<String>, Result>, result: Result) {
    if values
        .get(&result.order)
        .is_none_or(|previous| result.score > previous.score)
    {
        values.insert(result.order.clone(), result);
    }
}

pub fn refine_pool(
    seeds: &[Vec<String>],
    scorer: &mut impl FnMut(&[String]) -> f64,
    seed_limit: usize,
    options: Options,
) -> std::result::Result<Vec<Result>, &'static str> {
    refine_pool_controlled(seeds, scorer, seed_limit, options, &Control::default())
}

pub fn refine_pool_controlled(
    seeds: &[Vec<String>],
    scorer: &mut impl FnMut(&[String]) -> f64,
    seed_limit: usize,
    options: Options,
    control: &Control,
) -> std::result::Result<Vec<Result>, &'static str> {
    control.check()?;
    if seed_limit == 0 || options.max_evaluations == 0 {
        return Err("seed limit and evaluation budget must be positive");
    }
    let mut values = HashMap::new();
    for seed in seeds.iter().take(seed_limit) {
        merge(
            &mut values,
            refine_controlled(seed, scorer, options, None, control)?,
        );
    }
    control.check()?;
    let results = sorted(values, control)?;
    control.check()?;
    Ok(results)
}

/// Keep all original seeds; the per-seed budget includes its original score.
pub fn augment_pool(
    seeds: &[Vec<String>],
    scorer: &mut impl FnMut(&[String]) -> f64,
    seed_limit: usize,
    options: Options,
) -> std::result::Result<Pool, &'static str> {
    augment_pool_controlled(seeds, scorer, seed_limit, options, &Control::default())
}

pub fn augment_pool_controlled(
    seeds: &[Vec<String>],
    scorer: &mut impl FnMut(&[String]) -> f64,
    seed_limit: usize,
    options: Options,
    control: &Control,
) -> std::result::Result<Pool, &'static str> {
    control.check()?;
    if seed_limit == 0 || options.max_evaluations == 0 {
        return Err("seed limit and evaluation budget must be positive");
    }
    let mut values = HashMap::new();
    let mut unique = Vec::new();
    for seed in seeds {
        control.check()?;
        if values.contains_key(seed) {
            continue;
        }
        let score = scorer(seed);
        control.check()?;
        if !score.is_finite() {
            return Err("seed score must be finite");
        }
        unique.push((seed, score));
        values.insert(
            seed.clone(),
            Result {
                order: seed.clone(),
                score,
                evaluated: 0,
                rounds: 0,
                improved: false,
            },
        );
    }
    let mut evaluated = unique.len();
    let mut improved_seeds = 0;
    if options.max_evaluations > 1 && options.max_rounds > 0 {
        for (seed, score) in unique.into_iter().take(seed_limit) {
            let endpoint = refine_controlled(
                seed,
                scorer,
                Options {
                    max_evaluations: options.max_evaluations - 1,
                    ..options
                },
                Some(score),
                control,
            )?;
            evaluated += endpoint.evaluated;
            improved_seeds += usize::from(endpoint.improved);
            merge(&mut values, endpoint);
        }
    }
    control.check()?;
    let candidates = sorted(values, control)?;
    control.check()?;
    Ok(Pool {
        candidates,
        evaluated,
        improved_seeds,
    })
}
