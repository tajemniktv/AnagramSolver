use anagram_core::{
    corpus_ranking::{self, Collocation, PhraseCorpus},
    ranking,
    wordnet::WordNet,
};
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    io::{self, BufRead},
    path::Path,
};
struct Index {
    counts: HashMap<String, i64>,
}
impl PhraseCorpus for Index {
    fn counts(&self, phrases: &[String]) -> io::Result<HashMap<String, i64>> {
        Ok(phrases
            .iter()
            .filter_map(|p| self.counts.get(p).map(|c| (p.clone(), *c)))
            .collect())
    }
    fn max_n(&self) -> usize {
        6
    }
}
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
        let mut result = json!({"prepared":prepared,"selected":selected,"rows":rows,"evaluated":evaluated,"buckets":ranking::rank_buckets(&rows),"alternatives":alternatives});
        if !v["corpus"].is_null() {
            let c = &v["corpus"];
            let model = Collocation {
                unigrams: serde_json::from_value(c["unigrams"].clone()).unwrap(),
                bigrams: c["bigrams"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|e| {
                        (
                            (e[0].as_str().unwrap().into(), e[1].as_str().unwrap().into()),
                            e[2].as_i64().unwrap(),
                        )
                    })
                    .collect(),
                total: c["total"].as_i64().unwrap(),
            };
            let index = Index {
                counts: serde_json::from_value(c["counts"].clone()).unwrap(),
            };
            let collocation = c["use_collocation"].as_bool().unwrap().then_some(&model);
            let database = c["path"]
                .as_str()
                .map(|p| anagram_core::phrase_index::PhraseIndex::open(Path::new(p)).unwrap());
            let phrase = c["use_phrase"].as_bool().unwrap().then_some(
                database
                    .as_ref()
                    .map_or(&index as &dyn PhraseCorpus, |db| db as &dyn PhraseCorpus),
            );
            let top = c["top"].as_u64().unwrap() as usize;
            let mut admission = std::collections::BTreeMap::new();
            for (wc, bucket) in ranking::rank_buckets(&rows) {
                // Reference admission receives original prepared row order.
                let mut bucket: Vec<_> = bucket.into_iter().filter(|&i| rows[i].deep).collect();
                bucket.sort();
                if !bucket.is_empty() {
                    admission.insert(
                        wc,
                        corpus_ranking::select(&rows, &bucket, collocation, phrase, top).unwrap(),
                    );
                }
            }
            let rescored = corpus_ranking::rescore(
                &mut rows,
                collocation,
                phrase,
                top,
                c["bonus"].as_f64().unwrap(),
            )
            .unwrap();
            result["corpus"] = json!({"rows":rows,"rescored":rescored,"admission":admission,"buckets":ranking::rank_buckets(&rows)});
        }
        println!("{result}");
    }
}
