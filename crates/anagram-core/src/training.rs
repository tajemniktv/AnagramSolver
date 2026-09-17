//! Deterministic pairwise fitting with word-bag-isolated held-out evaluation.
use crate::{
    control::Control,
    learned::{Features, Item, Model},
    request::Error,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Deserialize, Serialize)]
pub struct Sample {
    pub key: String,
    pub features: Features,
    pub baseline_score: f64,
    pub positive: bool,
}
#[derive(Deserialize, Serialize)]
pub struct Group {
    pub key: String,
    pub items: Vec<Sample>,
}
#[derive(Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub folds: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub l2: f64,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            folds: 5,
            epochs: 80,
            learning_rate: 0.08,
            l2: 0.002,
        }
    }
}
#[derive(Default, Serialize)]
pub struct Metrics {
    pub groups: usize,
    pub recall1: f64,
    pub mrr: f64,
}
#[derive(Serialize)]
pub struct Report {
    pub baseline: Metrics,
    pub held_out: Metrics,
    pub model: Model,
}
fn error(message: &str) -> Error {
    Error::new("training_error", message)
}
fn check(control: &Control) -> Result<(), Error> {
    control.check().map_err(error)
}
fn fold(key: &str, folds: usize) -> usize {
    let digest = Sha256::digest(key.as_bytes());
    (u64::from_be_bytes(digest[..8].try_into().unwrap()) % folds as u64) as usize
}
pub fn train(groups: &[Group], options: &Options, control: &Control) -> Result<Report, Error> {
    if !(2..=100).contains(&options.folds)
        || !(1..=10000).contains(&options.epochs)
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || !options.l2.is_finite()
        || options.l2 < 0.0
    {
        return Err(error("Invalid training options"));
    }
    let mut keys = BTreeSet::new();
    let mut folds = BTreeSet::new();
    for group in groups {
        check(control)?;
        if group.items.len() < 2
            || !keys.insert(group.key.clone())
            || !group.items.iter().any(|i| i.positive)
            || !group.items.iter().any(|i| !i.positive)
        {
            return Err(error(
                "Each unique word bag needs positive and negative orders",
            ));
        }
        folds.insert(fold(&group.key, options.folds));
        let mut orders = BTreeSet::new();
        for item in &group.items {
            let mut words: Vec<_> = item
                .key
                .split_whitespace()
                .map(crate::normalize_letters)
                .collect();
            if words.join(" ") != item.key || words.iter().any(String::is_empty) {
                return Err(error(
                    "Training order keys must be normalized lowercase words",
                ));
            }
            words.sort();
            if words.join(" ") != group.key
                || !orders.insert(&item.key)
                || !item.baseline_score.is_finite()
                || item.features.iter().any(|v| !v.is_finite())
            {
                return Err(error(
                    "Invalid features, duplicate order, or inconsistent word-bag identity",
                ));
            }
        }
    }
    if folds.len() < 2 {
        return Err(error(
            "Need word bags in at least two hash folds for honest held-out evaluation",
        ));
    }
    // Each pair participates in K-1 held-out fits and the final all-data fit.
    let updates = groups
        .iter()
        .try_fold(0usize, |total, group| {
            let positives = group.items.iter().filter(|item| item.positive).count();
            positives
                .checked_mul(group.items.len() - positives)
                .and_then(|pairs| total.checked_add(pairs))
        })
        .and_then(|pairs| pairs.checked_mul(options.epochs))
        .and_then(|updates| updates.checked_mul(folds.len()));
    if updates.is_none_or(|updates| updates > 50_000_000) {
        return Err(error("Training exceeds the 50-million pair-update budget"));
    }
    let mut baseline = Metrics::default();
    let mut held_out = Metrics::default();
    for f in folds {
        let training: Vec<_> = groups
            .iter()
            .filter(|g| fold(&g.key, options.folds) != f)
            .collect();
        let model = fit(&training, options, control)?;
        for group in groups.iter().filter(|g| fold(&g.key, options.folds) == f) {
            check(control)?;
            let items: Vec<_> = group
                .items
                .iter()
                .map(|i| Item {
                    key: i.key.clone(),
                    features: i.features,
                    baseline_score: i.baseline_score,
                })
                .collect();
            for (metrics, ranker) in [(&mut baseline, None), (&mut held_out, Some(&model))] {
                let indices = crate::learned::rank_controlled(&items, ranker, control)
                    .map_err(|e| error(&e.to_string()))?;
                let rank = indices
                    .iter()
                    .position(|&i| group.items[i].positive)
                    .unwrap()
                    + 1;
                metrics.groups += 1;
                metrics.recall1 += f64::from(rank == 1);
                metrics.mrr += 1.0 / rank as f64;
            }
        }
    }
    for metrics in [&mut baseline, &mut held_out] {
        metrics.recall1 /= metrics.groups as f64;
        metrics.mrr /= metrics.groups as f64;
    }
    let model = fit(&groups.iter().collect::<Vec<_>>(), options, control)?;
    Ok(Report {
        baseline,
        held_out,
        model,
    })
}
fn fit(groups: &[&Group], options: &Options, control: &Control) -> Result<Model, Error> {
    let mut groups = groups.to_vec();
    groups.sort_by(|a, b| a.key.cmp(&b.key));
    let mut weights = [0.0; 18];
    let mut step = 0_u64;
    for _ in 0..options.epochs {
        for group in &groups {
            for p in group.items.iter().filter(|i| i.positive) {
                for n in group.items.iter().filter(|i| !i.positive) {
                    check(control)?;
                    let diff: Features = std::array::from_fn(|i| p.features[i] - n.features[i]);
                    let margin =
                        crate::compensated_sum(weights.iter().zip(diff).map(|(w, d)| w * d));
                    let probability = if margin >= 0.0 {
                        let x = (-margin.min(700.0)).exp();
                        x / (1.0 + x)
                    } else {
                        1.0 / (1.0 + margin.max(-700.0).exp())
                    };
                    step += 1;
                    let rate = options.learning_rate / (1.0 + 0.0005 * step as f64).sqrt();
                    for i in 0..18 {
                        weights[i] += rate * (probability * diff[i] - options.l2 * weights[i]);
                    }
                    if weights.iter().any(|w| !w.is_finite()) {
                        return Err(error("Training overflow"));
                    }
                }
            }
        }
    }
    Model::new(weights).map_err(|e| error(&e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learns_positive_orders_without_leaking_bags_and_rejects_bad_data() {
        let groups: Vec<_> = (0..20)
            .map(|n| {
                let a = format!("a{}", char::from(b'a' + n));
                let b = format!("b{}", char::from(b'a' + n));
                Group {
                    key: format!("{a} {b}"),
                    items: vec![
                        Sample {
                            key: format!("{a} {b}"),
                            features: [1.0; 18],
                            baseline_score: 0.0,
                            positive: true,
                        },
                        Sample {
                            key: format!("{b} {a}"),
                            features: [0.0; 18],
                            baseline_score: 1.0,
                            positive: false,
                        },
                    ],
                }
            })
            .collect();
        let report = train(&groups, &Options::default(), &Control::default()).unwrap();
        assert_eq!(report.baseline.recall1, 0.0);
        assert_eq!(report.held_out.recall1, 1.0);
        assert_eq!(report.held_out.groups, groups.len());
        assert!(train(&groups[..1], &Options::default(), &Control::default()).is_err());
        let cancelled = Control::default();
        cancelled.cancel();
        assert!(train(&groups, &Options::default(), &cancelled).is_err());
        let mut invalid = groups;
        invalid[0].items[0].features[0] = f64::NAN;
        assert!(train(&invalid, &Options::default(), &Control::default()).is_err());
    }
}
