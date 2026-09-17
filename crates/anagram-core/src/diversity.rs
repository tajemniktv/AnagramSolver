//! Preserve the quality head and diversify only the retained runner-up tail.
use crate::ordering::Candidate;
use std::collections::HashMap;

pub const QUALITY_CORE: usize = 48;

pub fn raw_pool_size(retained: usize) -> Result<usize, &'static str> {
    if retained == 0 {
        return Err("retained must be >= 1");
    }
    Ok(if retained <= QUALITY_CORE {
        retained
    } else {
        retained.max(retained.saturating_add(8).min(128))
    })
}

struct Fingerprint<'a> {
    candidate: &'a Candidate,
    adjacency: HashMap<(&'a str, &'a str), usize>,
}
impl<'a> Fingerprint<'a> {
    fn new(candidate: &'a Candidate) -> Self {
        let mut adjacency = HashMap::new();
        for edge in candidate.order.windows(2) {
            *adjacency
                .entry((edge[0].as_str(), edge[1].as_str()))
                .or_default() += 1;
        }
        Self {
            candidate,
            adjacency,
        }
    }

    fn similarity(&self, other: &Self) -> f64 {
        let a = &self.candidate.order;
        let b = &other.candidate.order;
        if a == b {
            return 1.0;
        }
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }
        let adjacency = if a.len() <= 1 || b.len() <= 1 {
            0.0
        } else {
            let overlap: usize = self
                .adjacency
                .iter()
                .map(|(edge, count)| (*count).min(*other.adjacency.get(edge).unwrap_or(&0)))
                .sum();
            overlap as f64 / (a.len().max(b.len()) - 1) as f64
        };
        let position =
            a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len().max(b.len()) as f64;
        let endpoints =
            0.5 * f64::from(a.first() == b.first()) + 0.5 * f64::from(a.last() == b.last());
        let kind = f64::from(self.candidate.phrase_kind == other.candidate.phrase_kind);
        (0.55 * adjacency + 0.30 * position + 0.10 * endpoints + 0.05 * kind).clamp(0.0, 1.0)
    }
}

pub fn similarity(a: &Candidate, b: &Candidate) -> f64 {
    Fingerprint::new(a).similarity(&Fingerprint::new(b))
}

/// Input is already score-ranked. Output preserves that order, not greedy order.
pub fn select(
    candidates: &[Candidate],
    top_k: usize,
    quality_core: usize,
    strength: f64,
) -> Result<Vec<Candidate>, &'static str> {
    select_controlled(
        candidates,
        top_k,
        quality_core,
        strength,
        &crate::control::Control::default(),
    )
}

pub fn select_controlled(
    candidates: &[Candidate],
    top_k: usize,
    quality_core: usize,
    strength: f64,
    control: &crate::control::Control,
) -> Result<Vec<Candidate>, &'static str> {
    control.check()?;
    if top_k == 0 || quality_core == 0 {
        return Err("retention and quality core must be positive");
    }
    if !(0.0..=1.0).contains(&strength) {
        return Err("diversity strength must be between 0 and 1");
    }
    let limit = top_k.min(candidates.len());
    let core = quality_core.min(limit);
    if core >= limit || candidates.len() <= limit {
        return candidates[..limit]
            .iter()
            .map(|candidate| {
                control.check()?;
                Ok(candidate.clone())
            })
            .collect();
    }
    let fingerprints: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            control.check()?;
            Ok(Fingerprint::new(candidate))
        })
        .collect::<Result<_, &'static str>>()?;
    let mut selected = vec![false; candidates.len()];
    selected[..core].fill(true);
    let mut maximum = vec![0.0_f64; candidates.len()];
    for i in core..candidates.len() {
        for j in 0..core {
            control.check()?;
            maximum[i] = maximum[i].max(fingerprints[i].similarity(&fingerprints[j]));
        }
    }
    for _ in core..limit {
        let mut winner: Option<usize> = None;
        for i in core..candidates.len() {
            control.check()?;
            if selected[i] {
                continue;
            }
            let better = winner.is_none_or(|j| {
                let a = &candidates[i];
                let b = &candidates[j];
                let au = a.objective - strength * maximum[i];
                let bu = b.objective - strength * maximum[j];
                bu.total_cmp(&au)
                    .then_with(|| b.objective.total_cmp(&a.objective))
                    .then_with(|| a.order.cmp(&b.order))
                    .is_lt()
            });
            if better {
                winner = Some(i);
            }
        }
        let winner = winner.expect("unselected candidate exists until limit");
        selected[winner] = true;
        for i in core..candidates.len() {
            control.check()?;
            if !selected[i] {
                maximum[i] = maximum[i].max(fingerprints[i].similarity(&fingerprints[winner]));
            }
        }
    }
    let mut result = Vec::with_capacity(limit);
    for (candidate, keep) in candidates.iter().zip(selected) {
        control.check()?;
        if keep {
            result.push(candidate.clone());
        }
    }
    Ok(result)
}
