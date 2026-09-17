//! Lazy exact enumeration. Candidate order is supplied by lexical admission;
//! this module must not reorder it because bounded prefixes are observable.
use crate::{Inventory, normalize_letters};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::{cell::OnceCell, rc::Rc};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub word: String,
    inventory: Inventory,
    sparse: Vec<(usize, usize)>,
}

impl Candidate {
    pub fn new(word: &str) -> Option<Self> {
        let word = normalize_letters(word);
        let inventory = Inventory::from_text(&word);
        let sparse = inventory
            .0
            .iter()
            .enumerate()
            .filter_map(|(letter, &amount)| (amount != 0).then_some((letter, amount)))
            .collect();
        (!inventory.is_empty()).then_some(Self {
            word,
            inventory,
            sparse,
        })
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
    start: usize,
    produced_at_entry: usize,
    chosen: Vec<usize>,
    matched: BTreeSet<String>,
}

type DeadState = (Inventory, usize, usize, BTreeSet<String>);
const MAX_DEAD_STATES: usize = 4096;

struct Stream<'a> {
    candidates: &'a [Candidate],
    target: Inventory,
    options: &'a Options,
    clues: BTreeSet<String>,
    clue_candidates: Vec<usize>,
    by_signature: Rc<OnceCell<HashMap<Inventory, Vec<usize>>>>,
    next_count: usize,
    max_count: usize,
    frames: Vec<Frame>,
    min_length: usize,
    max_length: usize,
    dead: HashSet<DeadState>,
    produced: usize,
}

impl<'a> Stream<'a> {
    fn new(
        candidates: &'a [Candidate],
        target: Inventory,
        options: &'a Options,
        clues: BTreeSet<String>,
        low: usize,
        high: usize,
        mut check: impl FnMut() -> Result<(), Stop>,
    ) -> Result<Self, Stop> {
        check()?;
        let mut clue_candidates = Vec::new();
        let mut min_length = usize::MAX;
        let mut max_length = 1;
        for (index, candidate) in candidates.iter().enumerate() {
            check()?;
            if clues.contains(&candidate.word) {
                clue_candidates.push(index);
            }
            min_length = min_length.min(candidate.inventory.len());
            max_length = max_length.max(candidate.inventory.len());
        }
        Ok(Self {
            candidates,
            target,
            options,
            clues,
            clue_candidates,
            by_signature: Rc::new(OnceCell::new()),
            next_count: low,
            max_count: high,
            frames: Vec::new(),
            dead: HashSet::new(),
            produced: 0,
            min_length: if candidates.is_empty() { 1 } else { min_length },
            max_length,
        })
    }

    fn retire_dead_branch(&mut self) {
        let frame = self.frames.pop().unwrap();
        if frame.produced_at_entry == self.produced && self.dead.len() < MAX_DEAD_STATES {
            self.dead
                .insert((frame.remaining, frame.start, frame.left, frame.matched));
        }
    }

    fn next(
        &mut self,
        cancel: &AtomicBool,
        deadline: Option<Instant>,
    ) -> Result<Option<Vec<String>>, Stop> {
        if self.by_signature.get().is_none() {
            let mut signatures: HashMap<Inventory, Vec<usize>> = HashMap::new();
            for (index, candidate) in self.candidates.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return Err(Stop::Cancelled);
                }
                if deadline.is_some_and(|d| Instant::now() >= d) {
                    return Err(Stop::TimedOut);
                }
                signatures
                    .entry(candidate.inventory)
                    .or_default()
                    .push(index);
            }
            self.by_signature
                .set(signatures)
                .expect("serial index initialization");
        }
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
                    start: 0,
                    produced_at_entry: self.produced,
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
            if frame.next == frame.start
                && self.dead.contains(&(
                    frame.remaining,
                    frame.start,
                    frame.left,
                    frame.matched.clone(),
                ))
            {
                self.frames.pop();
                continue;
            }
            if frame.left == 0 {
                let valid = frame.remaining.is_empty()
                    && (self.clues.is_empty()
                        || match self.options.hint_mode {
                            HintMode::Any => !frame.matched.is_empty(),
                            HintMode::ExactlyOne => frame.matched.len() == 1,
                        });
                let frame = self.frames.pop().unwrap();
                if valid {
                    self.produced += 1;
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
            // Monotone candidate indices mean a clue behind `next`, or one
            // that cannot fit the residual, can never rescue this branch.
            if !self.clues.is_empty() && frame.matched.is_empty() {
                let mut possible = false;
                for &index in &self.clue_candidates {
                    if cancel.load(Ordering::Relaxed) {
                        return Err(Stop::Cancelled);
                    }
                    if deadline.is_some_and(|d| Instant::now() >= d) {
                        return Err(Stop::TimedOut);
                    }
                    if index >= frame.next
                        && frame
                            .remaining
                            .subtract(self.candidates[index].inventory)
                            .is_some()
                    {
                        possible = true;
                        break;
                    }
                }
                if !possible {
                    self.retire_dead_branch();
                    continue;
                }
            }
            if frame.next >= self.candidates.len()
                || length == 0
                || length < frame.left.saturating_mul(self.min_length)
                || length > frame.left.saturating_mul(self.max_length)
                || (self.options.hint_mode == HintMode::ExactlyOne && frame.matched.len() > 1)
            {
                self.retire_dead_branch();
                continue;
            }
            let index = if frame.left == 1 {
                // Lists retain admission order; binary search skips only candidates
                // before the monotone cursor, never changing bounded prefixes.
                let matching = self.by_signature.get().unwrap().get(&frame.remaining);
                let found = matching.and_then(|indices| {
                    indices
                        .get(indices.partition_point(|&i| i < frame.next))
                        .copied()
                });
                let Some(index) = found else {
                    self.retire_dead_branch();
                    continue;
                };
                index
            } else {
                frame.next
            };
            frame.next = index + 1;
            let candidate = &self.candidates[index];
            let slots_after = frame.left - 1;
            let min_this = self
                .min_length
                .max(length.saturating_sub(slots_after.saturating_mul(self.max_length)));
            let max_this = self
                .max_length
                .min(length.saturating_sub(slots_after.saturating_mul(self.min_length)));
            if candidate.word.len() < min_this || candidate.word.len() > max_this {
                continue;
            }
            if candidate
                .sparse
                .iter()
                .any(|&(letter, amount)| frame.remaining.0[letter] < amount)
            {
                continue;
            }
            if self.options.hint_mode == HintMode::ExactlyOne
                && !frame.matched.is_empty()
                && self.clues.contains(&candidate.word)
                && !frame.matched.contains(&candidate.word)
            {
                continue;
            }
            let mut remaining = frame.remaining;
            for &(letter, amount) in &candidate.sparse {
                remaining.0[letter] -= amount;
            }
            let mut chosen = frame.chosen.clone();
            chosen.push(index);
            let mut matched = frame.matched.clone();
            if self.clues.contains(&candidate.word) {
                matched.insert(candidate.word.clone());
            }
            let left = frame.left - 1;
            let start = if self.options.allow_repeat {
                index
            } else {
                index + 1
            };
            self.frames.push(Frame {
                remaining,
                left,
                next: start,
                start,
                produced_at_entry: self.produced,
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
    let interrupted = || {
        let stop = if cancel.load(Ordering::Relaxed) {
            Some(Stop::Cancelled)
        } else if deadline.is_some_and(|d| Instant::now() >= d) {
            Some(Stop::TimedOut)
        } else {
            None
        };
        stop.map(|stop| ResultSet {
            bags: Vec::new(),
            stop,
        })
    };
    if let Some(result) = interrupted() {
        return Ok(result);
    }
    let mut words = HashSet::new();
    for candidate in candidates {
        if let Some(result) = interrupted() {
            return Ok(result);
        }
        if !words.insert(&candidate.word) {
            return Err("duplicate candidate");
        }
    }
    let mut streams = VecDeque::new();
    let low = options.min_words;
    // Every candidate has at least one letter. Clamp before constructing strata.
    let high = options.max_words.min(target.len());
    if options.strategy == Strategy::Prefix || options.max_results == 0 {
        let stream = Stream::new(
            candidates,
            target,
            options,
            options.clues.clone(),
            low,
            high,
            || interrupted().map_or(Ok(()), |result| Err(result.stop)),
        );
        match stream {
            Ok(stream) => streams.push_back(stream),
            Err(stop) => {
                return Ok(ResultSet {
                    bags: Vec::new(),
                    stop,
                });
            }
        }
    } else {
        let groups = if !options.clues.is_empty()
            && options.initial_clues.is_empty()
            && options.hint_mode == HintMode::Any
        {
            let mut groups = Vec::new();
            for clue in &options.clues {
                if let Some(result) = interrupted() {
                    return Ok(result);
                }
                groups.push(BTreeSet::from([clue.clone()]));
            }
            groups
        } else {
            vec![options.clues.clone()]
        };
        for count in low..=high {
            for group in &groups {
                if let Some(result) = interrupted() {
                    return Ok(result);
                }
                let stream = Stream::new(
                    candidates,
                    target,
                    options,
                    group.clone(),
                    count,
                    count,
                    || interrupted().map_or(Ok(()), |result| Err(result.stop)),
                );
                match stream {
                    Ok(stream) => streams.push_back(stream),
                    Err(stop) => {
                        return Ok(ResultSet {
                            bags: Vec::new(),
                            stop,
                        });
                    }
                }
            }
        }
    }
    // Strata differ in clue/count constraints, not candidate signatures. Keep
    // one lazily initialized index per serial search instead of one per stream.
    let signatures = Rc::new(OnceCell::new());
    for stream in &mut streams {
        if let Some(result) = interrupted() {
            return Ok(result);
        }
        stream.by_signature = Rc::clone(&signatures);
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

#[cfg(test)]
mod probe_tests {
    use super::*;
    #[test]
    fn stream_setup_stops_mid_vocabulary_scan() {
        let candidates: Vec<_> = ["ate", "eat", "tea"]
            .into_iter()
            .map(|w| Candidate::new(w).unwrap())
            .collect();
        let options = Options {
            min_words: 1,
            max_words: 1,
            max_results: 1,
            allow_repeat: true,
            clues: BTreeSet::new(),
            initial_clues: BTreeSet::new(),
            hint_mode: HintMode::Any,
            strategy: Strategy::Prefix,
        };
        for reason in [Stop::Cancelled, Stop::TimedOut] {
            let mut checked = 0;
            let result = Stream::new(
                &candidates,
                Inventory::from_text("ate"),
                &options,
                BTreeSet::new(),
                1,
                1,
                || {
                    checked += 1;
                    if checked == 3 { Err(reason) } else { Ok(()) }
                },
            );
            assert!(matches!(result, Err(stop) if stop == reason));
            assert_eq!(checked, 3);
        }
    }
    #[test]
    fn expiry_between_cap_and_extra_candidate_never_proves_exhaustion_or_truncation() {
        for words in [vec!["ate"], vec!["ate", "eat"]] {
            let candidates: Vec<_> = words
                .into_iter()
                .map(|w| Candidate::new(w).unwrap())
                .collect();
            let options = Options {
                min_words: 1,
                max_words: 1,
                max_results: 1,
                allow_repeat: true,
                clues: BTreeSet::new(),
                initial_clues: BTreeSet::new(),
                hint_mode: HintMode::Any,
                strategy: Strategy::Prefix,
            };
            let cancel = AtomicBool::new(false);
            let mut stream = Stream::new(
                &candidates,
                Inventory::from_text("ate"),
                &options,
                BTreeSet::new(),
                1,
                1,
                || Ok(()),
            )
            .unwrap();
            assert_eq!(
                stream.next(&cancel, None).unwrap(),
                Some(vec!["ate".to_owned()])
            );
            assert_eq!(
                stream.next(&cancel, Some(Instant::now())),
                Err(Stop::TimedOut)
            );
        }
    }
}
