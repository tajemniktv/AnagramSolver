//! Graded comparative morphology; noun/verb homographs are weak, not vetoed.
use crate::{
    grammar::{Class, function_class},
    normalize_letters, phrase,
    wordnet::WordNet,
};
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Serialize)]
pub struct Evidence {
    pub confidence: f64,
    pub base: Option<String>,
    pub source: &'static str,
}

pub fn bases(raw: &str) -> BTreeSet<String> {
    let word = normalize_letters(raw);
    if word.len() <= 3 || !word.ends_with("er") {
        return BTreeSet::new();
    }
    let stem = &word[..word.len() - 2];
    let mut candidates = BTreeSet::from([stem.to_owned(), word[..word.len() - 1].to_owned()]);
    if stem.ends_with('i') && stem.len() > 1 {
        candidates.insert(format!("{}y", &stem[..stem.len() - 1]));
    }
    if stem.len() >= 2 && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2] {
        candidates.insert(stem[..stem.len() - 1].into());
    }
    candidates
}

pub fn evidence(raw: &str, lex: &WordNet) -> Evidence {
    let word = normalize_letters(raw);
    if ["better", "worse", "more", "less", "rather", "sooner"].contains(&word.as_str()) {
        return Evidence {
            confidence: 1.0,
            base: None,
            source: "lexical",
        };
    }
    if ["elder", "farther", "further"].contains(&word.as_str()) {
        return Evidence {
            confidence: 0.98,
            base: None,
            source: "irregular",
        };
    }
    let base = bases(&word)
        .into_iter()
        .filter_map(|base| {
            let f = lex.features(&base);
            (f.adj || f.adv).then_some((usize::from(f.adj) + usize::from(f.adv), base))
        })
        .max()
        .map(|(_, base)| base);
    let Some(base) = base else {
        return Evidence {
            confidence: 0.0,
            base: None,
            source: "none",
        };
    };
    let surface = lex.features(&word);
    let (confidence, source) = if surface.adj || surface.adv {
        (0.96, "regular-surface")
    } else if surface.recognized {
        (0.38, "regular-ambiguous")
    } else {
        (0.90, "regular-recovered")
    };
    Evidence {
        confidence,
        base: Some(base),
        source,
    }
}

pub fn span(words: &[String], start: usize, lex: &WordNet) -> Option<(usize, f64)> {
    if start >= words.len() {
        return None;
    }
    for than in start + 1..words.len().min(start.saturating_add(5)) {
        if words[than] != "than" || than + 1 >= words.len() {
            continue;
        }
        let left = &words[start..than];
        let right = &words[than + 1..];
        let best = left
            .iter()
            .map(|w| evidence(w, lex).confidence)
            .reduce(f64::max)
            .unwrap();
        if best <= 0.0 {
            continue;
        }
        let consumed = if let Some((end, _)) = phrase::starting_at(right, 0, lex, true) {
            end + 1
        } else {
            let f = lex.features(&right[0]);
            if f.noun
                || f.adj
                || f.adv
                || matches!(
                    function_class(&right[0]),
                    Some(
                        Class::Pron
                            | Class::Pron12
                            | Class::PronPl
                            | Class::PronSg3
                            | Class::Neg
                            | Class::NumDet
                    )
                )
            {
                1
            } else {
                0
            }
        };
        if consumed == 0 {
            continue;
        }
        let mut quality = 0.82 + 0.16 * best;
        if evidence(&left[0], lex).confidence > 0.0 {
            quality += 0.02;
        }
        return Some((than + consumed, quality.min(0.99)));
    }
    None
}
