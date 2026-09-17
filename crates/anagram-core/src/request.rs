//! Versioned native generation contract. Ranking is not yet part of this endpoint.
use crate::{
    Inventory,
    generation::{self, HintMode, Options, Stop, Strategy},
    lexicon::{self, Admission, ShortPolicy, Unigrams},
    normalize_letters,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, io::BufRead, sync::atomic::AtomicBool, time::Instant};

#[derive(Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    pub schema_version: u32,
    pub text: String,
    #[serde(default)]
    pub required: Vec<String>,
    #[serde(default)]
    pub hints: Vec<String>,
    #[serde(default)]
    pub excluded: Vec<String>,
    pub min_words: usize,
    pub max_words: usize,
    pub min_word_length: usize,
    pub max_word_length: usize,
    pub min_zipf: f64,
    pub candidate_budget: usize,
    pub allow_repeat: bool,
    pub strategy: Strategy,
    pub hint_mode: HintMode,
}

#[derive(Debug, Serialize, schemars::JsonSchema)]
pub struct Error {
    pub code: &'static str,
    pub message: String,
}

impl Error {
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Serialize, schemars::JsonSchema)]
pub struct Generated {
    pub schema_version: u32,
    pub kind: &'static str,
    pub engine_version: &'static str,
    pub normalized_input: String,
    pub bags: Vec<Vec<String>>,
    pub generated: usize,
    pub vocabulary_size: usize,
    pub candidate_budget: usize,
    pub stop: Stop,
    #[serde(skip)]
    pub vocabulary: BTreeSet<String>,
    #[serde(skip)]
    pub short_whitelist: BTreeSet<String>,
}

fn words(values: &[String]) -> Result<Vec<String>, Error> {
    values
        .iter()
        .map(|s| {
            // Contract arrays contain individual words, never implicit phrases.
            if s.trim().is_empty() || s.trim().chars().any(char::is_whitespace) {
                return Err(Error::new(
                    "invalid_word",
                    "Each constraint must contain one word",
                ));
            }
            let word = normalize_letters(s);
            if word.is_empty() {
                Err(Error::new(
                    "invalid_word",
                    "Constraint contains no A-Z letters",
                ))
            } else {
                Ok(word)
            }
        })
        .collect()
}

pub fn generate(
    request: &GenerateRequest,
    dictionary: impl BufRead,
    unigrams: Option<&Unigrams>,
    cancel: &AtomicBool,
    deadline: Option<Instant>,
) -> Result<Generated, Error> {
    if request.schema_version != 1 {
        return Err(Error::new(
            "unsupported_version",
            "Expected schema_version 1",
        ));
    }
    if request.min_words == 0
        || request.max_words < request.min_words
        || request.min_word_length == 0
        || request.max_word_length < request.min_word_length
        || !request.min_zipf.is_finite()
        || request.min_zipf < 0.0
    {
        return Err(Error::new(
            "invalid_limits",
            "Invalid word-count, length or frequency limits",
        ));
    }
    let normalized_input = normalize_letters(&request.text);
    let mut remaining = Inventory::from_text(&normalized_input);
    if remaining.is_empty() {
        return Err(Error::new("empty_input", "Input contains no A-Z letters"));
    }
    let required = words(&request.required)?;
    let mut hints: BTreeSet<_> = words(&request.hints)?.into_iter().collect();
    let excluded: BTreeSet<_> = words(&request.excluded)?.into_iter().collect();
    for word in &required {
        if excluded.contains(word) {
            return Err(Error::new(
                "conflicting_constraints",
                "Required word is excluded",
            ));
        }
        remaining = remaining
            .subtract(Inventory::from_text(word))
            .ok_or_else(|| {
                Error::new(
                    "unavailable_letters",
                    "Required words exceed available letters",
                )
            })?;
    }
    let required_set: BTreeSet<_> = required.iter().cloned().collect();
    hints.retain(|word| {
        !excluded.contains(word)
            && (required_set.contains(word)
                || remaining.subtract(Inventory::from_text(word)).is_some())
    });
    if !request.hints.is_empty() && hints.is_empty() {
        return Err(Error::new(
            "impossible_hints",
            "No supplied hint can satisfy the constraints",
        ));
    }
    if request.min_zipf > 0.0 && unigrams.is_none() {
        return Err(Error::new(
            "missing_frequency_data",
            "Frequency filtering requires a unigram corpus",
        ));
    }
    let initial_clues: BTreeSet<_> = required_set.intersection(&hints).cloned().collect();
    let mut vocabulary_size = 0;
    let mut vocabulary: BTreeSet<String> = required_set.union(&hints).cloned().collect();
    let mut short_whitelist: BTreeSet<String> =
        "a i am an as at be by do go he if in is it me my no of oh on or so to up us we"
            .split_whitespace()
            .map(str::to_owned)
            .collect();
    short_whitelist.extend(vocabulary.iter().filter(|w| w.len() <= 2).cloned());
    short_whitelist.retain(|w| !excluded.contains(w));
    let result = if remaining.is_empty() {
        let valid = required.len() >= request.min_words
            && required.len() <= request.max_words
            && (hints.is_empty()
                || match request.hint_mode {
                    HintMode::Any => !initial_clues.is_empty(),
                    HintMode::ExactlyOne => initial_clues.len() == 1,
                });
        generation::ResultSet {
            bags: if valid { vec![Vec::new()] } else { vec![] },
            stop: Stop::Exhausted,
        }
    } else if required.len() >= request.max_words {
        generation::ResultSet {
            bags: vec![],
            stop: Stop::Exhausted,
        }
    } else {
        let policy = Admission {
            min_length: request.min_word_length,
            max_length: request.max_word_length,
            min_zipf: request.min_zipf,
            short_policy: ShortPolicy::Common,
            short_whitelist: short_whitelist.clone(),
            forced: hints.clone(),
            excluded,
            forbidden: BTreeSet::new(),
        };
        let candidates = lexicon::admit(dictionary, remaining, &policy, unigrams, |_| false)
            .map_err(|e| Error::new("corpus_error", e.to_string()))?;
        vocabulary_size = candidates.len();
        vocabulary.extend(candidates.iter().map(|c| c.word.clone()));
        let options = Options {
            min_words: request.min_words.saturating_sub(required.len()).max(1),
            max_words: request.max_words - required.len(),
            max_results: request.candidate_budget,
            allow_repeat: request.allow_repeat,
            clues: hints,
            initial_clues,
            hint_mode: request.hint_mode,
            strategy: request.strategy,
        };
        generation::search(remaining, &candidates, &options, cancel, deadline)
            .map_err(|e| Error::new("invalid_search", e))?
    };
    let bags: Vec<_> = result
        .bags
        .into_iter()
        .map(|bag| required.iter().cloned().chain(bag).collect())
        .collect();
    Ok(Generated {
        schema_version: 1,
        kind: "generation_only",
        engine_version: env!("CARGO_PKG_VERSION"),
        normalized_input,
        generated: bags.len(),
        bags,
        vocabulary_size,
        candidate_budget: request.candidate_budget,
        stop: result.stop,
        vocabulary,
        short_whitelist,
    })
}
