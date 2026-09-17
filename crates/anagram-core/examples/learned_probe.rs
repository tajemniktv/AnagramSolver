use anagram_core::{
    learned::{self, Model},
    ordering::Candidate,
};
use serde_json::{Value, json};
use std::io::{self, BufRead};
fn run(v: &Value) -> io::Result<Value> {
    let model = if let Some(path) = v["model_path"].as_str() {
        Model::load(std::path::Path::new(path))?
    } else if let Some(raw) = v["model_text"].as_str() {
        Model::from_json(raw.as_bytes())?
    } else {
        Model::from_json(&serde_json::to_vec(&v["model"])?)?
    };
    let candidate: Candidate = serde_json::from_value(v["candidate"].clone())?;
    let details = serde_json::from_value(v["details"].clone())?;
    let features = learned::features(
        &candidate,
        v["phrase"]
            .as_f64()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing numeric phrase"))?,
        &details,
        v["word_count"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid word_count"))?,
    );
    let items = serde_json::from_value::<Vec<learned::Item>>(v["items"].clone())?;
    Ok(
        json!({"features":features,"score":model.score(&features)?,"rank":learned::rank(&items,Some(&model))?,"baseline":learned::rank(&items,None)?,"model":model}),
    )
}
fn main() {
    for line in io::stdin().lock().lines() {
        let result = line
            .and_then(|line| serde_json::from_str::<Value>(&line).map_err(io::Error::from))
            .and_then(|v| run(&v));
        println!(
            "{}",
            match result {
                Ok(result) => result,
                Err(_) => json!({"error":true}),
            }
        );
    }
}
