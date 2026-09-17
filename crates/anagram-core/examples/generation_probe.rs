//! Test-only JSON-lines adapter for differential checks against Python.
use anagram_core::{Inventory, generation::*};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{self, BufRead},
    sync::atomic::AtomicBool,
};

fn set(value: &Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect()
}

fn main() {
    for line in io::stdin().lock().lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let candidates: Vec<_> = value["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Candidate::new(v.as_str().unwrap()).unwrap())
            .collect();
        let options = Options {
            min_words: value["min_words"].as_u64().unwrap() as usize,
            max_words: value["max_words"].as_u64().unwrap() as usize,
            max_results: value["cap"].as_u64().unwrap() as usize,
            allow_repeat: value["repeat"].as_bool().unwrap(),
            clues: set(&value["clues"]),
            initial_clues: set(&value["initial"]),
            hint_mode: if value["mode"] == "any" {
                HintMode::Any
            } else {
                HintMode::ExactlyOne
            },
            strategy: if value["strategy"] == "prefix" {
                Strategy::Prefix
            } else {
                Strategy::Diverse
            },
        };
        let result = search(
            Inventory::from_text(value["text"].as_str().unwrap()),
            &candidates,
            &options,
            &AtomicBool::new(false),
            None,
        )
        .unwrap();
        println!(
            "{}",
            json!({"bags": result.bags, "truncated": result.stop == Stop::CandidateCap})
        );
    }
}
