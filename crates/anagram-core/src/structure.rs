//! Combined constructions and validity scorer; no mutable hooks or global state.
use crate::{
    auxiliary,
    clause::{self, Structure},
    comparative,
    grammar::{self, Class, function_class},
    phrase, validity,
    wordnet::{FRAME_INTRANSITIVE, WordNet},
};

fn result(
    norm: f64,
    valency: f64,
    coverage: f64,
    agreement: f64,
    kind: impl Into<String>,
) -> Structure {
    Structure {
        norm,
        valency,
        coverage,
        agreement,
        kind: kind.into(),
        raw: 4.0 * norm,
    }
}
fn choose(candidates: impl IntoIterator<Item = Structure>) -> Option<Structure> {
    let mut best: Option<Structure> = None;
    for candidate in candidates {
        if best.as_ref().is_none_or(|old| {
            (candidate.norm, candidate.coverage, candidate.valency)
                > (old.norm, old.coverage, old.valency)
        }) {
            best = Some(candidate);
        }
    }
    best
}
fn pronoun(word: &str) -> bool {
    matches!(
        function_class(word),
        Some(Class::Pron | Class::Pron12 | Class::PronPl | Class::PronSg3)
    )
}
fn passive_adverbs(words: &[String], mut start: usize, lex: &WordNet) -> usize {
    while let Some(word) = words.get(start) {
        let f = lex.features(word);
        let c = function_class(word);
        if c == Some(Class::Neg) || (f.adv && !matches!(c, Some(Class::Prep | Class::Conj))) {
            start += 1;
        } else {
            break;
        }
    }
    start
}
pub fn passive_tail(words: &[String], lex: &WordNet) -> (f64, usize) {
    if words.is_empty() {
        return (0.96, 0);
    }
    if words[0] == "by" && words.len() > 1 {
        if let Some((end, _)) = phrase::starting_at(words, 1, lex, true) {
            return (0.99, passive_adverbs(words, end + 1, lex));
        }
    }
    let consumed = passive_adverbs(words, 0, lex);
    if consumed > 0 {
        (0.92, consumed)
    } else {
        (0.90, 0)
    }
}
pub fn auxiliary_structure(words: &[String], lex: &WordNet) -> Option<Structure> {
    let n = words.len();
    if n < 3 {
        return None;
    }
    let collisions = words
        .windows(2)
        .filter(|w| phrase::determiner(&w[0]).is_some() && phrase::determiner(&w[1]).is_some())
        .count() as f64;
    let mut candidates = Vec::new();
    for (i, word) in words.iter().enumerate().skip(1) {
        let Some(chain) = auxiliary::parse(words, i, lex) else {
            continue;
        };
        if !validity::valid_subject(&words[i - 1], lex) {
            continue;
        }
        let Some((start, coherence)) = phrase::ending_at(words, i - 1, lex) else {
            continue;
        };
        let number = clause::span_number(words, start, i - 1, lex);
        let agreement = auxiliary::agreement(&words[i - 1], number, word, lex);
        let tail = &words[chain.main_idx + 1..];
        let (valency, consumed) = if chain.passive {
            passive_tail(tail, lex)
        } else {
            clause::tail(&words[chain.main_idx], tail, lex)
        };
        let mut coverage = ((chain.main_idx - start + 1 + consumed) as f64 / n as f64).min(1.0);
        if start > 0 {
            coverage *= 0.90;
        }
        let mut norm = 0.18 * coherence
            + 0.22 * agreement
            + 0.20 * chain.quality
            + 0.18 * valency
            + 0.22 * coverage;
        norm *= 0.42 + 0.58 * coverage.powf(1.3);
        if agreement <= 0.20 {
            norm = norm.min(0.42);
        }
        norm = (norm - 0.22 * collisions).clamp(0.0, 0.98);
        candidates.push(result(
            norm,
            valency,
            coverage,
            agreement,
            format!("aux-{}", chain.kind),
        ));
    }
    choose(candidates)
}
pub fn comparative_structure(words: &[String], lex: &WordNet) -> Option<Structure> {
    let n = words.len();
    if n < 4 {
        return None;
    }
    let mut candidates = Vec::new();
    for i in 1..n - 2 {
        if !validity::finite_lexical(&words[i], lex) {
            continue;
        }
        let Some((start, coherence)) = phrase::ending_at(words, i - 1, lex) else {
            continue;
        };
        let Some((end, quality)) = comparative::span(words, i + 1, lex) else {
            continue;
        };
        if end != n - 1 {
            continue;
        }
        let agreement = clause::agreement(
            &words[i - 1],
            &words[i],
            lex,
            false,
            Some(clause::span_number(words, start, i - 1, lex)),
        );
        if agreement <= 0.15 {
            continue;
        }
        let valency = 0.88 * quality;
        let mut coverage = ((n - start) as f64 / n as f64).min(1.0);
        if start > 0 {
            coverage *= 0.90;
        }
        let mut norm = 0.22
            + 0.14 * coherence
            + 0.20 * agreement
            + 0.18 * valency
            + 0.16 * quality
            + 0.10 * coverage;
        norm *= 0.55 + 0.45 * coverage.powf(1.25);
        norm = norm.clamp(0.0, 0.98);
        candidates.push(result(
            norm,
            valency,
            coverage,
            agreement,
            "comparative-clause",
        ));
    }
    choose(candidates)
}
fn parallel_half(
    modifier: &str,
    subject: &str,
    verb: &str,
    lex: &WordNet,
) -> Option<(f64, f64, f64)> {
    if function_class(modifier).is_some() {
        return None;
    }
    let f = lex.features(modifier);
    let quality = if f.verb_past {
        0.98
    } else if f.adj && !f.noun {
        0.82
    } else {
        return None;
    };
    if !pronoun(subject) || !validity::finite_lexical(verb, lex) {
        return None;
    }
    let agreement = clause::agreement(
        subject,
        verb,
        lex,
        false,
        Some(clause::subject_number(subject, lex)),
    );
    if agreement <= 0.15 {
        return None;
    }
    let valency = match lex.allows(verb, FRAME_INTRANSITIVE) {
        Some(true) => 0.98,
        Some(false) => 0.35,
        None => 0.65,
    };
    Some((quality, agreement, valency))
}
pub fn parallel_structure(words: &[String], lex: &WordNet) -> Option<Structure> {
    if words.len() != 6 || words[1] != words[4] {
        return None;
    }
    let left = parallel_half(&words[0], &words[1], &words[2], lex)?;
    let right = parallel_half(&words[3], &words[4], &words[5], lex)?;
    let modifier = 0.5 * (left.0 + right.0);
    let agreement = 0.5 * (left.1 + right.1);
    let valency = 0.5 * (left.2 + right.2);
    let norm = (0.32 + 0.20 * modifier + 0.20 * agreement + 0.16 * valency + 0.12).clamp(0.0, 0.98);
    Some(result(norm, valency, 1.0, agreement, "parallel-clause"))
}
pub fn evaluate(words: &[String], lex: &WordNet) -> Structure {
    let base = validity::adjust(words, lex, clause::base_structure(words, lex));
    let candidates = std::iter::once(base).chain(
        [
            auxiliary_structure(words, lex),
            comparative_structure(words, lex),
            parallel_structure(words, lex),
        ]
        .into_iter()
        .flatten(),
    );
    validity::surface(words, choose(candidates).unwrap())
}
pub fn auxiliary_bonus(left: &str, right: &str, lex: &WordNet) -> f64 {
    let lc = function_class(left);
    let rc = function_class(right);
    let f = lex.features(right);
    if lc == Some(Class::BeAux) {
        if right == "being" && rc == Some(Class::BeAux) {
            return 1.10;
        }
        if rc.is_none() && f.verb_ing {
            return 1.20;
        }
        if rc.is_none() && f.verb_past {
            return 1.00;
        }
    }
    if lc == Some(Class::HaveAux) {
        if right == "been" && rc == Some(Class::BeAux) {
            return 1.05;
        }
        if rc.is_none() && f.verb_past {
            return 1.10;
        }
    }
    0.0
}
pub fn construction_bonus(left: &str, right: &str, lex: &WordNet) -> f64 {
    let mut score = 0.0;
    if right == "than" {
        score += 1.90 * comparative::evidence(left, lex).confidence;
    }
    if left == "than" {
        let f = lex.features(right);
        if f.noun
            || f.adj
            || f.adv
            || pronoun(right)
            || matches!(function_class(right), Some(Class::Neg | Class::NumDet))
            || phrase::determiner(right).is_some()
        {
            score += 0.65;
        }
    }
    if function_class(left).is_none() && pronoun(right) {
        let f = lex.features(left);
        if f.verb_past && f.adj {
            score += 0.85;
        }
    }
    score
}
pub fn pair(left: &str, right: &str, lex: &WordNet) -> f64 {
    grammar::pair(left, right, lex)
        + validity::pair(left, right)
        + construction_bonus(left, right, lex)
        + auxiliary_bonus(left, right, lex)
}
pub fn local_raw(words: &[String], lex: &WordNet) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let total = grammar::start(&words[0], lex) + grammar::end(words.last().unwrap(), lex);
    if words.len() == 1 {
        return total;
    }
    // Preserve the reference addition order for realized adjacencies.
    let edges = words
        .windows(2)
        .map(|w| {
            grammar::pair(&w[0], &w[1], lex)
                + auxiliary_bonus(&w[0], &w[1], lex)
                + construction_bonus(&w[0], &w[1], lex)
                + validity::pair(&w[0], &w[1])
        })
        .sum::<f64>();
    (total + edges) / (words.len() - 1) as f64
}
