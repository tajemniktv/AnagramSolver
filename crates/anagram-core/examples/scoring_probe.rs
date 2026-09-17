//! Test-only adapter for Python/native component parity.
use anagram_core::{
    lexicon::Unigrams,
    scoring::{self, Bigrams},
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{self, BufRead},
};

fn main() {
    for line in io::stdin().lock().lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let words: Vec<String> = serde_json::from_value(value["words"].clone()).unwrap();
        let short: BTreeSet<String> = serde_json::from_value(value["short"].clone()).unwrap();
        let vocabulary: BTreeSet<String> =
            serde_json::from_value(value["vocabulary"].clone()).unwrap();
        let unigrams = Unigrams::load(value["unigrams"].as_str().unwrap().as_bytes()).unwrap();
        let bigrams = Bigrams::load(
            value["bigrams"].as_str().unwrap().as_bytes(),
            &unigrams,
            &vocabulary,
        )
        .unwrap();
        let values: Vec<f64> = serde_json::from_value(value["values"].clone()).unwrap();
        let bags: Vec<Vec<String>> = serde_json::from_value(value["bags"].clone()).unwrap();
        let hints: BTreeSet<String> = serde_json::from_value(value["hints"].clone()).unwrap();
        println!(
            "{}",
            json!({"lexical": scoring::lexical(&words, Some(&unigrams), &short),
            "pair": bigrams.pair_potential(&words), "order": bigrams.best_order(&words).unwrap(),
            "percentiles": scoring::percentiles(&values),
            "records": scoring::pre_rank(&bags, &hints, Some(&unigrams), Some(&bigrams), &vocabulary, &short)})
        );
    }
}
