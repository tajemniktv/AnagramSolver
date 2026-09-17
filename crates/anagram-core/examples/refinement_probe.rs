use anagram_core::refinement::{self, Options};
use serde_json::{Value, json};
use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let seeds: Vec<Vec<String>> = serde_json::from_value(value["seeds"].clone()).unwrap();
        let weights: Vec<Vec<f64>> = serde_json::from_value(value["weights"].clone()).unwrap();
        let mut scorer = |order: &[String]| {
            order
                .iter()
                .enumerate()
                .map(|(i, w)| weights[i][w.parse::<usize>().unwrap()])
                .fold(0.0, |sum, value| sum + value)
        };
        let options = Options {
            max_evaluations: value["budget"].as_u64().unwrap() as usize,
            max_rounds: value["rounds"].as_u64().unwrap() as usize,
            ..Options::default()
        };
        let seed_limit = value["seed_limit"].as_u64().unwrap() as usize;
        let single = refinement::refine(&seeds[0], &mut scorer, options, None).unwrap();
        let pool = refinement::refine_pool(&seeds, &mut scorer, seed_limit, options).unwrap();
        let augmented = refinement::augment_pool(&seeds, &mut scorer, seed_limit, options).unwrap();
        println!(
            "{}",
            json!({"single":single,"pool":pool,"augmented":augmented})
        );
    }
}
