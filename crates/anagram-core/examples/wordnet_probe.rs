//! Test-only adapter for real WordNet feature/frame parity.
use anagram_core::wordnet::*;
use serde_json::json;
use std::{
    io::{self, BufRead},
    path::Path,
};

fn main() {
    let directory = std::env::args_os().nth(1).expect("WordNet directory");
    let mut lexicon = WordNet::load(Path::new(&directory)).unwrap();
    let words: Vec<_> = io::stdin().lock().lines().map(Result::unwrap).collect();
    let expected: Vec<_> = words
        .iter()
        .map(|word| {
            (
                lexicon.features(word),
                lexicon.frames_for(word),
                lexicon.morphology_family_word(word),
            )
        })
        .collect();
    lexicon
        .prepare_words(
            words.iter().map(String::as_str),
            &anagram_core::control::Control::default(),
        )
        .unwrap();
    for (word, expected) in words.iter().zip(expected) {
        assert_eq!(
            (
                lexicon.features(word),
                lexicon.frames_for(word),
                lexicon.morphology_family_word(word)
            ),
            expected
        );
        println!(
            "{}",
            json!({"features": lexicon.features(word),
            "bases": lexicon.verb_base_lemmas(word), "frames": lexicon.frames_for(word),
            "allows": ([FRAME_DIRECT_OBJECT, FRAME_INTRANSITIVE, FRAME_PP, FRAME_PREDICATIVE,
                FRAME_OBJECT_PREDICATIVE, FRAME_INFINITIVE_OR_GERUND, FRAME_CLAUSAL]
                .map(|category| lexicon.allows(word, category)))})
        );
    }
}
