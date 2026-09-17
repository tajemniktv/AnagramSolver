//! Lazy exact enumeration. Candidate order is supplied by lexical admission;
//! this module must not reorder it because bounded prefixes are observable.
use crate::{Inventory, normalize_letters};
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub word: String,
    inventory: Inventory,
}

impl Candidate {
    pub fn new(word: &str) -> Option<Self> {
        let word = normalize_letters(word);
        let inventory = Inventory::from_text(&word);
        (!inventory.is_empty()).then_some(Self { word, inventory })
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum HintMode {
    Any,
    ExactlyOne,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    Prefix,
    Diverse,
}

pub struct Options {
    pub min_words: usize,
    pub max_words: usize,
    /// Zero means unlimited generation, not unlimited ranking.
    pub max_results: usize,
    pub allow_repeat: bool,
    pub clues: BTreeSet<String>,
    pub initial_clues: BTreeSet<String>,
    pub hint_mode: HintMode,
    pub strategy: Strategy,
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Stop {
    Exhausted,
    CandidateCap,
    Cancelled,
    TimedOut,
}

pub struct ResultSet {
    pub bags: Vec<Vec<String>>,
    pub stop: Stop,
}

struct Frame {
    remaining: Inventory,
    left: usize,
    next: usize,
    chosen: Vec<usize>,
    matched: BTreeSet<String>,
}

struct Stream<'a> {
    candidates: &'a [Candidate],
    target: Inventory,
    options: &'a Options,
    clues: BTreeSet<String>,
    next_count: usize,
    max_count: usize,
    frames: Vec<Frame>,
    min_length: usize,
    max_length: usize,
}

impl<'a> Stream<'a> {
    fn new(
        candidates: &'a [Candidate],
        target: Inventory,
        options: &'a Options,
        clues: BTreeSet<String>,
        low: usize,
        high: usize,
    ) -> Self {
        Self {
            candidates,
            target,
            options,
            clues,
            next_count: low,
            max_count: high,
            frames: Vec::new(),
            min_length: candidates
                .iter()
                .map(|c| c.inventory.len())
                .min()
                .unwrap_or(1),
            max_length: candidates
                .iter()
                .map(|c| c.inventory.len())
                .max()
                .unwrap_or(1),
        }
    }

    fn next(
        &mut self,
        cancel: &AtomicBool,
        deadline: Option<Instant>,
    ) -> Result<Option<Vec<String>>, Stop> {
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(Stop::Cancelled);
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                return Err(Stop::TimedOut);
            }
            if self.frames.is_empty() {
                if self.next_count > self.max_count {
                    return Ok(None);
                }
                self.frames.push(Frame {
                    remaining: self.target,
                    left: self.next_count,
                    next: 0,
                    chosen: Vec::new(),
                    matched: self
                        .options
                        .initial_clues
                        .intersection(&self.clues)
                        .cloned()
                        .collect(),
                });
                self.next_count += 1;
            }
            let frame = self.frames.last_mut().unwrap();
            if frame.left == 0 {
                let valid = frame.remaining.is_empty()
                    && (self.clues.is_empty()
                        || match self.options.hint_mode {
                            HintMode::Any => !frame.matched.is_empty(),
                            HintMode::ExactlyOne => frame.matched.len() == 1,
                        });
                let frame = self.frames.pop().unwrap();
                if valid {
                    return Ok(Some(
                        frame
                            .chosen
                            .iter()
                            .map(|i| self.candidates[*i].word.clone())
                            .collect(),
                    ));
                }
                continue;
            }
            let length = frame.remaining.len();
            if frame.next >= self.candidates.len()
                || length == 0
                || length < frame.left.saturating_mul(self.min_length)
                || length > frame.left.saturating_mul(self.max_length)
                || (self.options.hint_mode == HintMode::ExactlyOne && frame.matched.len() > 1)
            {
                self.frames.pop();
                continue;
            }
            let index = frame.next;
            frame.next += 1;
            let candidate = &self.candidates[index];
            let Some(remaining) = frame.remaining.subtract(candidate.inventory) else {
                continue;
            };
            let mut chosen = frame.chosen.clone();
            chosen.push(index);
            let mut matched = frame.matched.clone();
            if self.clues.contains(&candidate.word) {
                matched.insert(candidate.word.clone());
            }
            let left = frame.left - 1;
            self.frames.push(Frame {
                remaining,
                left,
                next: if self.options.allow_repeat {
                    index
                } else {
                    index + 1
                },
                chosen,
                matched,
            });
        }
    }
}

/// Enumerate bounded results with a one-extra-unique-bag probe. Deadline and
/// cancellation outcomes deliberately do not claim that the search exhausted.
pub fn search(
    target: Inventory,
    candidates: &[Candidate],
    options: &Options,
    cancel: &AtomicBool,
    deadline: Option<Instant>,
) -> Result<ResultSet, &'static str> {
    if options.min_words > options.max_words {
        return Err("invalid word-count range");
    }
    let mut words = HashSet::new();
    if candidates.iter().any(|c| !words.insert(&c.word)) {
        return Err("duplicate candidate");
    }
    let mut streams = VecDeque::new();
    let low = options.min_words;
    // Every candidate has at least one letter. Clamp before constructing strata.
    let high = options.max_words.min(target.len());
    if options.strategy == Strategy::Prefix || options.max_results == 0 {
        streams.push_back(Stream::new(
            candidates,
            target,
            options,
            options.clues.clone(),
            low,
            high,
        ));
    } else {
        let groups = if !options.clues.is_empty()
            && options.initial_clues.is_empty()
            && options.hint_mode == HintMode::Any
        {
            options
                .clues
                .iter()
                .map(|c| BTreeSet::from([c.clone()]))
                .collect::<Vec<_>>()
        } else {
            vec![options.clues.clone()]
        };
        for count in low..=high {
            for group in &groups {
                streams.push_back(Stream::new(
                    candidates,
                    target,
                    options,
                    group.clone(),
                    count,
                    count,
                ));
            }
        }
    }
    let mut seen = HashSet::new();
    let mut bags = Vec::new();
    while let Some(mut stream) = streams.pop_front() {
        loop {
            match stream.next(cancel, deadline) {
                Err(stop) => return Ok(ResultSet { bags, stop }),
                Ok(None) => break,
                Ok(Some(bag)) => {
                    if !seen.insert(bag.clone()) {
                        continue;
                    }
                    if options.max_results > 0 && bags.len() == options.max_results {
                        return Ok(ResultSet {
                            bags,
                            stop: Stop::CandidateCap,
                        });
                    }
                    bags.push(bag);
                    streams.push_back(stream);
                    break;
                }
            }
        }
    }
    Ok(ResultSet {
        bags,
        stop: Stop::Exhausted,
    })
}
