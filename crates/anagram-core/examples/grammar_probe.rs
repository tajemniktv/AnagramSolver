use anagram_core::{clause, grammar, phrase, wordnet::WordNet};
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
            "np_end": (0..words.len()).map(|i| phrase::ending_at(&words,i,&lex)).collect::<Vec<_>>(),
            "comparative": (0..words.len()).map(|i| clause::comparative(&words,i,&lex)).collect::<Vec<_>>(),
            "simple": (0..words.len()).map(|i| clause::simple(&words,i,&lex)).collect::<Vec<_>>(),
            "subordinate": (0..words.len()).map(|i| clause::subordinate(&words,i,&lex)).collect::<Vec<_>>(),
            "numbers": words.iter().map(|w| clause::subject_number(w,&lex)).collect::<Vec<_>>(),
            "agreement": words.iter().map(|w| ["is","are","dont","doesnt","runs","run","stopped","have"].map(|v| clause::agreement(w,v,&lex,false,None))).collect::<Vec<_>>(),
            "tails": (["run","give","speak","look","turn"].map(|v| clause::tail(v,&words,&lex))),
            "structure": clause::base_structure(&words,&lex)})
        );
    }
}
