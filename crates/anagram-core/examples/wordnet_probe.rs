//! Test-only adapter for real WordNet feature/frame parity.
use anagram_core::wordnet::*;
use serde_json::json;
use std::{
    io::{self, BufRead},
    path::Path,
};

fn main() {
    let directory = std::env::args_os().nth(1).expect("WordNet directory");
    let lexicon = WordNet::load(Path::new(&directory)).unwrap();
    for line in io::stdin().lock().lines() {
        let word = line.unwrap();
        println!(
            "{}",
            json!({"features": lexicon.features(&word),
            "bases": lexicon.verb_base_lemmas(&word), "frames": lexicon.frames_for(&word),
            "allows": ([FRAME_DIRECT_OBJECT, FRAME_INTRANSITIVE, FRAME_PP, FRAME_PREDICATIVE,
                FRAME_OBJECT_PREDICATIVE, FRAME_INFINITIVE_OR_GERUND, FRAME_CLAUSAL]
                .map(|category| lexicon.allows(&word, category)))})
        );
    }
}
