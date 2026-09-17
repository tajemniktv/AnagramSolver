use anagram_core::{
    corpus_ranking::{Collocation, PhraseCorpus},
    phrase_index::PhraseIndex,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    io::{self, BufRead, Cursor},
    path::Path,
};
fn run(v: &Value) -> io::Result<Value> {
    if let Some(path) = v["path"].as_str() {
        let index = PhraseIndex::open(Path::new(path))?;
        let queries: Vec<String> = serde_json::from_value(v["queries"].clone()).unwrap();
        return Ok(json!({"max_n":index.max_n(),"counts":index.counts(&queries)?}));
    }
    let vocabulary: BTreeSet<String> = serde_json::from_value(v["vocabulary"].clone()).unwrap();
    let model = Collocation::load(
        Cursor::new(v["one"].as_str().unwrap()),
        Cursor::new(v["two"].as_str().unwrap()),
        &vocabulary,
    )?;
    let orders: Vec<Vec<String>> = serde_json::from_value(v["orders"].clone()).unwrap();
    Ok(
        json!({"total":model.total,"unigrams":model.unigrams,"scores":orders.iter().map(|order| model.score(order)).collect::<Vec<_>>()}),
    )
}
fn main() {
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        println!(
            "{}",
            match run(&v) {
                Ok(result) => result,
                Err(_) => json!({"error":true}),
            }
        );
    }
}
