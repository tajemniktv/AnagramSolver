//! Subject agreement and compact comparative/subordinate/valency parsing.
use crate::{
    grammar::{Class, function_class},
    phrase,
    wordnet::*,
};
use serde::Serialize;

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Number {
    #[serde(rename = "non3sg")]
    NonThird,
    #[serde(rename = "3sg")]
    Third,
    #[serde(rename = "unknown")]
    Unknown,
}

pub fn subject_number(word: &str, lex: &WordNet) -> Number {
    if ["we", "they", "you", "i"].contains(&word) {
        return Number::NonThird;
    }
    if ["he", "she", "it"].contains(&word) {
        return Number::Third;
    }
    let f = lex.features(word);
    if f.noun_plural && !f.noun_singular {
        Number::NonThird
    } else if f.noun_singular && !f.noun_plural {
        Number::Third
    } else {
        Number::Unknown
    }
}

pub fn span_number(words: &[String], start: usize, head: usize, lex: &WordNet) -> Number {
    if start <= head && head < words.len() {
        for word in &words[start..=head] {
            match phrase::determiner(word) {
                Some(Class::DetPl) => return Number::NonThird,
                Some(Class::DetSg) => return Number::Third,
                Some(Class::Article) if ["a", "an"].contains(&word.as_str()) => {
                    return Number::Third;
                }
                Some(Class::NumDet) => {
                    return if word == "one" {
                        Number::Third
                    } else {
                        Number::NonThird
                    };
                }
                _ => {}
            }
        }
    }
    words
        .get(head)
        .map_or(Number::Unknown, |w| subject_number(w, lex))
}

pub fn agreement(
    subject: &str,
    verb: &str,
    lex: &WordNet,
    auxiliary: bool,
    number: Option<Number>,
) -> f64 {
    use Number::*;
    let number = number.unwrap_or_else(|| subject_number(subject, lex));
    let class = function_class(verb);
    let vf = lex.features(verb);
    match class {
        Some(Class::Dont) => {
            return match number {
                NonThird => 1.0,
                Third => 0.0,
                Unknown => 0.55,
            };
        }
        Some(Class::Doesnt) => {
            return match number {
                Third => 1.0,
                NonThird => 0.0,
                Unknown => 0.55,
            };
        }
        Some(Class::BeAux) => {
            return match verb {
                "is" => match number {
                    Third => 1.0,
                    NonThird => 0.1,
                    Unknown => 0.55,
                },
                "are" => match number {
                    NonThird => 1.0,
                    Third => 0.15,
                    Unknown => 0.55,
                },
                _ => 0.65,
            };
        }
        Some(Class::Modal | Class::DoAux | Class::HaveAux) => return 0.75,
        _ => {}
    }
    if !auxiliary {
        if vf.verb_past {
            return 0.85;
        }
        if number == Third {
            if vf.verb_3sg {
                return 1.0;
            }
            if vf.verb_base {
                return 0.15;
            }
        }
        if number == NonThird {
            if vf.verb_base {
                return 1.0;
            }
            if vf.verb_3sg {
                return 0.1;
            }
        }
    }
    0.5
}

pub fn comparative_like(word: &str, lex: &WordNet) -> bool {
    if ["better", "worse", "more", "less", "rather", "sooner"].contains(&word) {
        return true;
    }
    let f = lex.features(word);
    word.len() > 3 && word.ends_with("er") && (f.adj || f.adv)
}

pub fn comparative(words: &[String], start: usize, lex: &WordNet) -> Option<(usize, f64)> {
    if start >= words.len() {
        return None;
    }
    for than in start + 1..words.len().min(start.saturating_add(5)) {
        if words[than] != "than" || than + 1 >= words.len() {
            continue;
        }
        let left = &words[start..than];
        let right = &words[than + 1..];
        if !left.iter().any(|w| {
            let f = lex.features(w);
            comparative_like(w, lex) || f.adj || f.adv
        }) {
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
        if consumed > 0 {
            return Some((
                than + consumed,
                if comparative_like(&left[0], lex) {
                    0.90 + 0.05
                } else {
                    0.90
                },
            ));
        }
    }
    None
}

pub fn simple(words: &[String], start: usize, lex: &WordNet) -> Option<(usize, f64)> {
    let (subject_end, coherence) = phrase::starting_at(words, start, lex, true)?;
    let mut i = subject_end + 1;
    while let Some(word) = words.get(i) {
        let f = lex.features(word);
        let c = function_class(word);
        if c == Some(Class::Neg) || (f.adv && !matches!(c, Some(Class::Prep | Class::Conj))) {
            i += 1;
        } else {
            break;
        }
    }
    let verb = words.get(i)?;
    if !lex.features(verb).verb {
        return None;
    }
    let agreement = agreement(
        &words[subject_end],
        verb,
        lex,
        false,
        Some(span_number(words, start, subject_end, lex)),
    );
    let (valency, consumed) = if i + 1 < words.len() {
        tail(verb, &words[i + 1..], lex)
    } else {
        (
            if lex.allows(verb, FRAME_INTRANSITIVE) == Some(true) {
                0.95
            } else {
                0.65
            },
            0,
        )
    };
    Some((
        i + consumed,
        (0.45 * coherence + 0.30 * agreement + 0.25 * valency).clamp(0.0, 1.0),
    ))
}

pub fn subordinate(words: &[String], start: usize, lex: &WordNet) -> Option<(usize, f64)> {
    let word = words.get(start)?;
    if !"after although as because before if once since though unless until when whenever where whereas wherever while".split_whitespace().any(|w| w==word) {return None;}
    let (end, quality) = simple(words, start + 1, lex)?;
    Some((end, (0.85 + 0.15 * quality).min(1.0)))
}

fn evidence(value: Option<bool>, yes: f64, no: f64, unknown: f64) -> f64 {
    match value {
        Some(true) => yes,
        Some(false) => no,
        None => unknown,
    }
}

pub fn tail(verb: &str, words: &[String], lex: &WordNet) -> (f64, usize) {
    if words.is_empty() {
        return (
            evidence(lex.allows(verb, FRAME_INTRANSITIVE), 1.0, 0.42, 0.65),
            0,
        );
    }
    if let Some((end, quality)) = subordinate(words, 0, lex) {
        return (0.90 * quality, end + 1);
    }
    if let Some((end, quality)) = comparative(words, 0, lex) {
        return (0.88 * quality, end + 1);
    }
    if function_class(&words[0]) == Some(Class::Prep) {
        // Preserve the reference's inclusive-index consumption convention.
        let consumed = phrase::starting_at(words, 1, lex, true)
            .map_or(1, |(end, _)| end + 2)
            .min(words.len());
        return (
            evidence(lex.allows(verb, FRAME_PP), 0.95, 0.45, 0.65),
            consumed,
        );
    }
    if let Some((end, _)) = phrase::starting_at(words, 0, lex, true) {
        let consumed = end + 1;
        if let Some(next) = words.get(consumed) {
            let f = lex.features(next);
            if (f.adj || f.noun) && lex.allows(verb, FRAME_OBJECT_PREDICATIVE) == Some(true) {
                return (0.98, consumed + 1);
            }
        }
        return (
            evidence(lex.allows(verb, FRAME_DIRECT_OBJECT), 1.0, 0.08, 0.52),
            consumed,
        );
    }
    let f = lex.features(&words[0]);
    if f.adj || f.adv {
        return (
            evidence(lex.allows(verb, FRAME_PREDICATIVE), 0.96, 0.38, 0.62),
            1,
        );
    }
    if f.verb_base || f.verb_ing {
        return (
            evidence(
                lex.allows(verb, FRAME_INFINITIVE_OR_GERUND),
                0.92,
                0.35,
                0.56,
            ),
            1,
        );
    }
    (0.35, 0)
}

#[derive(Clone, Debug, Serialize)]
pub struct Structure {
    pub norm: f64,
    pub valency: f64,
    pub coverage: f64,
    pub agreement: f64,
    pub kind: &'static str,
    pub raw: f64,
}

fn structure(
    norm: f64,
    valency: f64,
    coverage: f64,
    agreement: f64,
    kind: &'static str,
) -> Structure {
    Structure {
        norm,
        valency,
        coverage,
        agreement,
        kind,
        raw: 4.0 * norm,
    }
}

/// Base reference constructions. Auxiliary-chain/validity extensions are applied
/// separately; this function must remain independently differential-testable.
pub fn base_structure(words: &[String], lex: &WordNet) -> Structure {
    let n = words.len();
    if n == 0 {
        return structure(0.0, 0.5, 0.0, 0.5, "empty");
    }
    let collisions = words
        .windows(2)
        .filter(|w| phrase::determiner(&w[0]).is_some() && phrase::determiner(&w[1]).is_some())
        .count() as f64;
    let mut candidates = Vec::new();
    if let Some((end, quality)) = comparative(words, 0, lex) {
        if end == n - 1 {
            let norm = (0.84 + 0.13 * quality).min(0.97);
            candidates.push(structure(norm, 0.78, 1.0, 0.75, "comparative"));
        }
    }
    for (i, token) in words.iter().enumerate() {
        if function_class(token) != Some(Class::BeAux) || i == 0 || i + 1 >= n {
            continue;
        }
        let Some((start, coherence)) = phrase::ending_at(words, i - 1, lex) else {
            continue;
        };
        let agree = agreement(
            &words[i - 1],
            token,
            lex,
            true,
            Some(span_number(words, start, i - 1, lex)),
        );
        let complement = comparative(words, i + 1, lex)
            .or_else(|| phrase::starting_at(words, i + 1, lex, true))
            .or_else(|| {
                let f = lex.features(&words[i + 1]);
                (f.adj || f.adv).then_some((i + 1, 0.86))
            });
        let Some((end, quality)) = complement else {
            continue;
        };
        let mut coverage = ((end - start + 1) as f64 / n as f64).min(1.0);
        if start > 0 {
            coverage *= 0.90;
        }
        let mut norm = 0.29 + 0.22 * agree + 0.16 * coherence + 0.19 * coverage + 0.14 * quality;
        norm *= 0.30 + 0.70 * coverage.powf(1.5);
        if agree <= 0.15 {
            norm = norm.min(0.42);
        }
        norm = (norm - 0.22 * collisions).clamp(0.0, 1.0);
        candidates.push(structure(norm, 1.0, coverage, agree, "copula"));
    }
    for (i, token) in words.iter().enumerate() {
        let c = function_class(token);
        if !matches!(
            c,
            Some(Class::Dont | Class::Doesnt | Class::Modal | Class::DoAux | Class::HaveAux)
        ) || i + 1 >= n
        {
            continue;
        }
        let f = lex.features(&words[i + 1]);
        let complement_ok = if c == Some(Class::HaveAux) {
            f.verb_past || f.verb_base
        } else {
            f.verb_base
        };
        if !complement_ok {
            continue;
        }
        let subject = i
            .checked_sub(1)
            .and_then(|head| phrase::ending_at(words, head, lex));
        let (start, coherence, explicit, agree, subject_tokens, clause) =
            if let Some((start, coherence)) = subject {
                (
                    start,
                    coherence,
                    true,
                    agreement(
                        &words[i - 1],
                        token,
                        lex,
                        true,
                        Some(span_number(words, start, i - 1, lex)),
                    ),
                    i - start,
                    1.0,
                )
            } else if i == 0 && matches!(c, Some(Class::Dont | Class::DoAux | Class::Modal)) {
                (0, 0.55, false, 0.60, 0, 0.67)
            } else {
                continue;
            };
        let (valency, consumed) = tail(&words[i + 1], &words[i + 2..], lex);
        let mut coverage = ((subject_tokens + 2 + consumed) as f64 / n as f64).min(1.0);
        if explicit && start > 0 {
            coverage *= 0.90;
        }
        let mut norm =
            0.32 * clause + 0.20 * agree + 0.16 * coherence + 0.18 * coverage + 0.14 * valency;
        norm *= 0.30 + 0.70 * coverage.powf(1.5);
        if explicit && agree <= 0.15 {
            norm = norm.min(0.42);
        }
        norm = (norm - 0.22 * collisions).clamp(0.0, 1.0);
        candidates.push(structure(norm, valency, coverage, agree, "clause"));
    }
    for (i, token) in words.iter().enumerate() {
        let vf = lex.features(token);
        if !vf.verb {
            continue;
        }
        let mut head = i.checked_sub(1);
        while let Some(index) = head {
            let f = lex.features(&words[index]);
            let c = function_class(&words[index]);
            if c == Some(Class::Neg) || (f.adv && !matches!(c, Some(Class::Prep | Class::Conj))) {
                head = index.checked_sub(1);
            } else {
                break;
            }
        }
        let Some(head) = head else {
            continue;
        };
        let Some((start, coherence)) = phrase::ending_at(words, head, lex) else {
            continue;
        };
        let number = span_number(words, start, head, lex);
        let agree = agreement(&words[head], token, lex, false, Some(number));
        let finite = if vf.verb_3sg || vf.verb_past || (number == Number::NonThird && vf.verb_base)
        {
            0.95
        } else {
            0.72
        };
        let (valency, consumed) = tail(token, &words[i + 1..], lex);
        let mut coverage = ((i - start + 1 + consumed) as f64 / n as f64).min(1.0);
        if start > 0 {
            coverage *= 0.90;
        }
        let mut norm =
            0.27 * finite + 0.24 * agree + 0.17 * coherence + 0.18 * coverage + 0.14 * valency;
        norm *= 0.30 + 0.70 * coverage.powf(1.5);
        if agree <= 0.15 {
            norm = norm.min(0.42);
        }
        norm = (norm - 0.22 * collisions).clamp(0.0, 1.0);
        candidates.push(structure(norm, valency, coverage, agree, "clause"));
    }
    if let Some((end, coherence)) = phrase::starting_at(words, 0, lex, true) {
        if end == n - 1 {
            let norm = (0.70 + 0.20 * coherence - 0.22 * collisions).clamp(0.0, 0.90);
            candidates.push(structure(norm, 0.60, 1.0, 0.70, "noun-phrase"));
        }
    }
    if lex.features(&words[0]).verb_base && function_class(&words[0]).is_none() {
        let (valency, consumed) = tail(&words[0], &words[1..], lex);
        let coverage = ((1 + consumed) as f64 / n as f64).min(1.0);
        let mut norm =
            (0.52 + 0.20 * valency + 0.18 * coverage) * (0.25 + 0.75 * coverage.powf(1.7));
        if words[1 + consumed..]
            .iter()
            .any(|w| function_class(w).is_some())
        {
            norm -= 0.25;
        }
        norm = (norm - 0.22 * collisions).clamp(0.0, 0.88);
        candidates.push(structure(norm, valency, coverage, 0.60, "imperative"));
    }
    let mut best: Option<Structure> = None;
    for candidate in candidates {
        if best.as_ref().is_none_or(|old| {
            (candidate.norm, candidate.coverage, candidate.valency)
                > (old.norm, old.coverage, old.valency)
        }) {
            best = Some(candidate);
        }
    }
    best.unwrap_or_else(|| {
        let recognized = crate::grammar::coverage(words, lex);
        let norm = (0.22 * recognized - 0.20 * collisions).max(0.05);
        structure(norm, 0.50, 0.25 * recognized, 0.50, "fragment")
    })
}
