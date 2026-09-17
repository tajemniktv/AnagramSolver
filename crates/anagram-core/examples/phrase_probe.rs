use anagram_core::{cohesion, phrase_evidence};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, BufRead},
};
fn main() {
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let words: Vec<String> = serde_json::from_value(v["words"].clone()).unwrap();
        let counts: HashMap<String, i64> = serde_json::from_value(v["counts"].clone()).unwrap();
        let max_n = v["max_n"].as_u64().unwrap() as usize;
        let evidence = cohesion::score(&words, &counts, max_n);
        println!(
            "{}",
            json!({"cohesion":evidence,"phrase":phrase_evidence::score(&words,&counts,max_n,false),"blended":phrase_evidence::score(&words,&counts,max_n,true)})
        );
    }
}
