//! Optional explicit-feature linear ranker. Never enabled implicitly.
use crate::contracts::{DataIdentity, DataRepresentation};
use crate::ordering::Candidate;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    io::{self, Read},
    path::Path,
};

pub const SCHEMA: &str = "anagram-explicit-ranker-1";
pub const FEATURE_NAMES: [&str; 18] = [
    "grammar_norm",
    "structure_norm",
    "valency_norm",
    "syntax_coverage",
    "objective",
    "phrase_score",
    "phrase_exact",
    "phrase_longer",
    "phrase_bigram_coverage",
    "cohesion",
    "cohesion_coverage",
    "cohesion_longest",
    "cohesion_compactness",
    "cohesion_frequency",
    "cohesion_splice_clean",
    "grammar_x_phrase",
    "structure_x_cohesion",
    "word_count_scaled",
];
pub type Features = [f64; 18];
fn bounded(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}
pub fn features(
    candidate: &Candidate,
    phrase_score: f64,
    details: &HashMap<String, f64>,
    word_count: usize,
) -> Features {
    let get = |key: &str| *details.get(key).unwrap_or(&0.0);
    let grammar = bounded(candidate.grammar_norm);
    let structure = bounded(candidate.structure_norm);
    let phrase = bounded(phrase_score);
    let cohesion = bounded(get("cohesion"));
    let segments = get("cohesion_segments");
    [
        grammar,
        structure,
        bounded(candidate.valency_norm),
        bounded(candidate.syntax_coverage),
        bounded(candidate.objective),
        phrase,
        f64::from(get("whole_count") > 0.0),
        bounded(get("longer")),
        bounded(get("bigram_coverage")),
        cohesion,
        bounded(get("cohesion_coverage")),
        bounded(get("cohesion_longest_fraction")),
        if segments > 0.0 { 1.0 / segments } else { 0.0 },
        bounded(get("cohesion_frequency")),
        if cohesion > 0.0 {
            1.0 - bounded(get("cohesion_splice_penalty"))
        } else {
            0.0
        },
        grammar * phrase,
        structure * cohesion,
        (word_count as f64 / 8.0).clamp(0.0, 1.0),
    ]
}

#[derive(Debug, Serialize)]
pub struct Model {
    schema: String,
    features: Vec<String>,
    weights: Features,
    #[serde(skip)]
    identity: Option<DataIdentity>,
}
#[derive(Deserialize)]
struct Payload {
    schema: String,
    features: Vec<String>,
    weights: Vec<f64>,
}
impl Model {
    pub fn new(weights: Features) -> io::Result<Self> {
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ranker weights must be finite",
            ));
        }
        Ok(Self {
            schema: SCHEMA.into(),
            features: FEATURE_NAMES.iter().map(|s| (*s).into()).collect(),
            weights,
            identity: None,
        })
    }
    pub fn from_json(bytes: &[u8]) -> io::Result<Self> {
        let payload: Payload = serde_json::from_slice(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        if payload.schema != SCHEMA {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported explicit-ranker model schema",
            ));
        }
        if payload
            .features
            .iter()
            .map(String::as_str)
            .ne(FEATURE_NAMES)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ranker feature schema does not match this solver",
            ));
        }
        let mut model = Self::new(payload.weights.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "ranker weight count does not match feature schema",
            )
        })?)?;
        model.identity = Some(DataIdentity {
            role: "learned_model".into(),
            representation: DataRepresentation::FileBytes,
            present: true,
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(bytes)),
        });
        Ok(model)
    }
    /// Exact loaded bytes; programmatically constructed weights have no file identity.
    pub fn identity(&self) -> Option<&DataIdentity> {
        self.identity.as_ref()
    }
    pub fn load(path: &Path) -> io::Result<Self> {
        Self::load_controlled(path, &crate::control::Control::default())
    }
    pub fn load_controlled(path: &Path, control: &crate::control::Control) -> io::Result<Self> {
        control.check_io()?;
        // A model has 18 weights; reject oversized input before unbounded parsing.
        let mut bytes = Vec::new();
        control
            .reader(std::fs::File::open(path)?)
            .take(65537)
            .read_to_end(&mut bytes)?;
        if bytes.len() > 65536 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ranker model exceeds 64 KiB",
            ));
        }
        control.check_io()?;
        let result = Self::from_json(&bytes);
        control.check_io()?;
        result
    }
    pub fn score(&self, features: &Features) -> io::Result<f64> {
        if features.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ranker features must be finite",
            ));
        }
        // Neumaier accumulation matches modern Python sum's compensated float
        // behavior, important when learned weights have opposite signs.
        let mut sum = 0.0_f64;
        let mut correction = 0.0;
        for (w, f) in self.weights.iter().zip(features) {
            let value = w * f;
            let next = sum + value;
            correction += if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
            sum = next;
        }
        let result = sum + correction;
        if !result.is_finite() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ranker score overflow",
            ));
        }
        Ok(result)
    }
}
#[derive(Deserialize)]
pub struct Item {
    pub key: String,
    pub features: Features,
    pub baseline_score: f64,
}
pub fn rank(items: &[Item], model: Option<&Model>) -> io::Result<Vec<usize>> {
    rank_controlled(items, model, &crate::control::Control::default())
}

#[derive(Debug, Serialize)]
pub struct RankingResult {
    pub schema_version: u32,
    pub indices: Vec<usize>,
    pub model_schema: Option<&'static str>,
    pub model_identity: Option<DataIdentity>,
}

/// Explicit offline result; no model is loaded or enabled implicitly.
pub fn rank_result(
    items: &[Item],
    model: Option<&Model>,
    control: &crate::control::Control,
) -> io::Result<RankingResult> {
    Ok(RankingResult {
        schema_version: 1,
        indices: rank_controlled(items, model, control)?,
        model_schema: model.map(|_| SCHEMA),
        model_identity: model.and_then(Model::identity).cloned(),
    })
}

pub fn rank_controlled(
    items: &[Item],
    model: Option<&Model>,
    control: &crate::control::Control,
) -> io::Result<Vec<usize>> {
    control.check_io()?;
    let mut scores = Vec::with_capacity(items.len());
    for item in items {
        control.check_io()?;
        if !item.baseline_score.is_finite() || item.features.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rank item values must be finite",
            ));
        }
        scores.push(match model {
            Some(m) => m.score(&item.features)?,
            None => item.baseline_score,
        });
    }
    let mut indices: Vec<_> = (0..items.len()).collect();
    control
        .sort_by(&mut indices, |&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap()
                .then_with(|| {
                    items[b]
                        .baseline_score
                        .partial_cmp(&items[a].baseline_score)
                        .unwrap()
                })
                .then_with(|| items[a].key.cmp(&items[b].key))
        })
        .map_err(io::Error::other)?;
    Ok(indices)
}
