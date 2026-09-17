use anagram_core::{ordering, wordnet::WordNet};
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
        let (orders, evaluated) =
            ordering::rank(&words, &lex, value["exact"].as_bool().unwrap(), 32, 8).unwrap();
        println!("{}", json!({"orders":orders,"evaluated":evaluated}));
    }
}
