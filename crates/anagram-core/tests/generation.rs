use anagram_core::{Inventory, generation::*};
use std::{collections::BTreeSet, sync::atomic::AtomicBool, time::Instant};

fn options() -> Options {
    Options {
        min_words: 1,
        max_words: 4,
        max_results: 1,
        allow_repeat: true,
        clues: BTreeSet::new(),
        initial_clues: BTreeSet::new(),
        hint_mode: HintMode::Any,
        strategy: Strategy::Prefix,
    }
}

#[test]
fn reaching_cap_is_not_proof_of_truncation() {
    let target = Inventory::from_text("aa");
    let cancel = AtomicBool::new(false);
    let candidates = [Candidate::new("a").unwrap()];
    let result = search(target, &candidates, &options(), &cancel, None).unwrap();
    assert_eq!(result.bags, vec![vec!["a", "a"]]);
    assert_eq!(result.stop, Stop::Exhausted);
    let candidates = [Candidate::new("aa").unwrap(), Candidate::new("a").unwrap()];
    let result = search(target, &candidates, &options(), &cancel, None).unwrap();
    assert_eq!(result.bags, vec![vec!["aa"]]);
    assert_eq!(result.stop, Stop::CandidateCap);
}

#[test]
fn interruption_does_not_claim_exhaustion() {
    let candidates = [Candidate::new("a").unwrap()];
    let target = Inventory::from_text("aa");
    let result = search(
        target,
        &candidates,
        &options(),
        &AtomicBool::new(true),
        None,
    )
    .unwrap();
    assert_eq!(result.stop, Stop::Cancelled);
    assert!(result.bags.is_empty());
    let result = search(
        target,
        &candidates,
        &options(),
        &AtomicBool::new(false),
        Some(Instant::now()),
    )
    .unwrap();
    assert_eq!(result.stop, Stop::TimedOut);
}

#[test]
fn repeated_hint_counts_as_one_distinct_clue() {
    let mut options = options();
    options.clues.insert("a".into());
    options.hint_mode = HintMode::ExactlyOne;
    let candidates = [Candidate::new("a").unwrap()];
    let result = search(
        Inventory::from_text("aa"),
        &candidates,
        &options,
        &AtomicBool::new(false),
        None,
    )
    .unwrap();
    assert_eq!(result.bags, vec![vec!["a", "a"]]);
    options.allow_repeat = false;
    let result = search(
        Inventory::from_text("aa"),
        &candidates,
        &options,
        &AtomicBool::new(false),
        None,
    )
    .unwrap();
    assert!(result.bags.is_empty());
}
