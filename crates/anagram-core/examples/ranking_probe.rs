use anagram_core::{ranking, wordnet::WordNet};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead},
    path::Path,
};
fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let lex = WordNet::load(Path::new(&path)).unwrap();
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let inputs = serde_json::from_value(v["rows"].clone()).unwrap();
        let mut rows = ranking::prepare(inputs, &lex);
        let prepared = serde_json::to_value(&rows).unwrap();
        let selected = ranking::choose_deep(
            &rows,
            v["per_group"].as_u64().unwrap() as usize,
            v["deep_all"].as_bool().unwrap(),
        );
        let options = ranking::Options {
            mode: serde_json::from_value(v["mode"].clone()).unwrap(),
            beam_width: 32,
            retained_orders: 56,
            ..Default::default()
        };
        let evaluated = ranking::deep_analyze(&mut rows, &selected, &lex, &options).unwrap();
        let alternatives: Vec<_> = rows.iter().map(|row| &row.alternatives).collect();
        println!(
            "{}",
            json!({"prepared":prepared,"selected":selected,"rows":rows,"evaluated":evaluated,"buckets":ranking::rank_buckets(&rows),"alternatives":alternatives})
        );
    }
}
