use anagram_core::{diversity, ordering, wordnet::WordNet};
use serde_json::{Value, json};
use std::{
    io::{self, BufRead},
    path::Path,
};
fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let lex = WordNet::load(Path::new(&path)).unwrap();
    for line in io::stdin().lock().lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let words: Vec<String> = serde_json::from_value(value["words"].clone()).unwrap();
        let top_k = value["top_k"].as_u64().unwrap_or(8) as usize;
        let diverse = value["diverse"].as_bool().unwrap_or(false);
        let raw_k = if diverse {
            diversity::raw_pool_size(top_k).unwrap()
        } else {
            top_k
        };
        let (orders, evaluated) =
            ordering::rank(&words, &lex, value["exact"].as_bool().unwrap(), 32, raw_k).unwrap();
        let orders = if diverse {
            diversity::select(&orders, top_k, diversity::QUALITY_CORE, 0.12).unwrap()
        } else {
            orders
        };
        println!("{}", json!({"orders":orders,"evaluated":evaluated}));
    }
}
