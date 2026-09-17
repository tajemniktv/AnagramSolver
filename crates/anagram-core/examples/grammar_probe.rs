use anagram_core::{grammar, phrase, wordnet::WordNet};
use serde_json::json;
use std::{
    io::{self, BufRead},
    path::Path,
};

fn main() {
    let path = std::env::args_os().nth(1).unwrap();
    let lex = WordNet::load(Path::new(&path)).unwrap();
    for line in io::stdin().lock().lines() {
        let words: Vec<String> = serde_json::from_str(&line.unwrap()).unwrap();
        let pairs: Vec<Vec<f64>> = words
            .iter()
            .map(|a| words.iter().map(|b| grammar::pair(a, b, &lex)).collect())
            .collect();
        println!(
            "{}",
            json!({"classes": words.iter().map(|w| grammar::function_class(w)).collect::<Vec<_>>(),
            "pairs": pairs, "start": words.iter().map(|w| grammar::start(w,&lex)).collect::<Vec<_>>(),
            "end": words.iter().map(|w| grammar::end(w,&lex)).collect::<Vec<_>>(),
            "local": grammar::local_raw(&words,&lex), "potential": grammar::potential(&words,&lex),
            "coverage": grammar::coverage(&words,&lex),
            "np_start": (0..words.len()).map(|i| phrase::starting_at(&words,i,&lex,true)).collect::<Vec<_>>(),
            "np_end": (0..words.len()).map(|i| phrase::ending_at(&words,i,&lex)).collect::<Vec<_>>()})
        );
    }
}
