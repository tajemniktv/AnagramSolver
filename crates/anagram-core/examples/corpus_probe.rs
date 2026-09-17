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
        let queries: Vec<String> =
            serde_json::from_value(v["queries"].clone()).map_err(io::Error::other)?;
        return Ok(json!({"max_n":index.max_n(),"counts":index.counts(&queries)?}));
    }
    let vocabulary: BTreeSet<String> =
        serde_json::from_value(v["vocabulary"].clone()).map_err(io::Error::other)?;
    let model = Collocation::load(
        Cursor::new(
            v["one"]
                .as_str()
                .ok_or_else(|| io::Error::other("Missing one"))?,
        ),
        Cursor::new(
            v["two"]
                .as_str()
                .ok_or_else(|| io::Error::other("Missing two"))?,
        ),
        &vocabulary,
    )?;
    let orders: Vec<Vec<String>> =
        serde_json::from_value(v["orders"].clone()).map_err(io::Error::other)?;
    Ok(
        json!({"total":model.total,"unigrams":model.unigrams,"scores":orders.iter().map(|order| model.score(order)).collect::<Vec<_>>()}),
    )
}
fn main() {
    for line in io::stdin().lock().lines() {
        println!(
            "{}",
            match line
                .and_then(|line| serde_json::from_str(&line).map_err(io::Error::other))
                .and_then(|v| run(&v))
            {
                Ok(result) => result,
                Err(_) => json!({"error":true}),
            }
        );
    }
}
