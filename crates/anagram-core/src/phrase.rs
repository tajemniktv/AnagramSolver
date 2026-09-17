//! Compact noun-phrase parsing used by clause and valency scoring.
use crate::{
    grammar::{Class, function_class},
    wordnet::WordNet,
};

pub fn determiner(word: &str) -> Option<Class> {
    function_class(word).filter(|c| {
        matches!(
            c,
            Class::DetPl | Class::DetSg | Class::Article | Class::Det | Class::NumDet
        )
    })
}

fn pronoun(class: Option<Class>) -> bool {
    matches!(
        class,
        Some(Class::Pron12 | Class::PronPl | Class::PronSg3 | Class::Pron)
    )
}
fn nominalized(word: &str) -> bool {
    ["more", "less", "most", "least"].contains(&word)
}

pub fn ending_at(words: &[String], head: usize, lex: &WordNet) -> Option<(usize, f64)> {
    let word = words.get(head)?;
    let hf = lex.features(word);
    if pronoun(function_class(word)) {
        return Some((head, 1.0));
    }
    if !hf.noun && !nominalized(word) {
        return None;
    }
    let mut start = head;
    let mut coherence: f64 = if hf.noun { 0.76 } else { 0.70 };
    let mut previous = head.checked_sub(1);
    let mut modifiers = 0;
    while let Some(i) = previous {
        if modifiers == 2 {
            break;
        }
        let f = lex.features(&words[i]);
        let c = function_class(&words[i]);
        let increment = if c.is_none() && f.adj {
            0.08
        } else if c.is_none()
            && f.verb_past
            && (words[i].ends_with("ed") || words[i].ends_with("en"))
            && i > 0
            && determiner(&words[i - 1]).is_some()
        {
            0.06
        } else if c.is_none() && f.noun && !f.verb && !f.adj {
            0.025
        } else {
            break;
        };
        start = i;
        coherence += increment;
        modifiers += 1;
        previous = i.checked_sub(1);
    }
    if let Some(i) = previous {
        if let Some(class) = determiner(&words[i]) {
            start = i;
            coherence += match class {
                Class::DetPl => {
                    if hf.noun_plural {
                        0.24
                    } else if hf.noun_singular {
                        -0.35
                    } else {
                        0.0
                    }
                }
                Class::DetSg => {
                    if hf.noun_singular {
                        0.20
                    } else if hf.noun_plural {
                        -0.35
                    } else {
                        0.0
                    }
                }
                Class::Article => {
                    if ["a", "an"].contains(&words[i].as_str()) && hf.noun_plural {
                        -0.45
                    } else {
                        0.15
                    }
                }
                Class::NumDet => {
                    if words[i] == "one" {
                        if hf.noun_plural { -0.20 } else { 0.12 }
                    } else if hf.noun_plural {
                        0.12
                    } else {
                        0.02
                    }
                }
                _ => 0.10,
            };
        }
    }
    Some((start, coherence.clamp(0.0, 1.0)))
}

pub fn starting_at(
    words: &[String],
    start: usize,
    lex: &WordNet,
    allow_post_pp: bool,
) -> Option<(usize, f64)> {
    if pronoun(function_class(words.get(start)?)) {
        return Some((start, 0.90));
    }
    let mut i = start;
    let det = determiner(&words[i]);
    let det_word = det.map(|_| words[i].as_str());
    if det.is_some() {
        i += 1;
    }
    let mut adjective_count = 0;
    while i < words.len() {
        let f = lex.features(&words[i]);
        let following_nominal = words.get(i + 1).is_some_and(|w| {
            let f = lex.features(w);
            f.noun || f.adj
        });
        if (f.adj && (!f.noun || following_nominal))
            || (i == start + 1
                && det.is_some()
                && f.verb_past
                && (words[i].ends_with("ed") || words[i].ends_with("en"))
                && following_nominal)
        {
            adjective_count += 1;
            i += 1;
        } else {
            break;
        }
    }
    let f = lex.features(words.get(i)?);
    let modifier =
        f.noun && !f.verb && !f.adj && words.get(i + 1).is_some_and(|w| lex.features(w).noun);
    if modifier {
        i += 1;
    }
    let head = words.get(i)?;
    let hf = lex.features(head);
    if !hf.noun && !nominalized(head) {
        return None;
    }
    let mut end = i;
    let mut coherence = 0.72 + (adjective_count as f64 * 0.04).min(0.12);
    if modifier {
        coherence += 0.025;
    }
    coherence += match det {
        Some(Class::DetPl) => {
            if hf.noun_plural {
                0.23
            } else {
                -0.30
            }
        }
        Some(Class::DetSg) => {
            if hf.noun_singular {
                0.20
            } else {
                -0.30
            }
        }
        Some(Class::Article) => {
            if matches!(det_word, Some("a" | "an")) && hf.noun_plural {
                -0.40
            } else {
                0.15
            }
        }
        Some(Class::NumDet) => {
            if det_word == Some("one") {
                if hf.noun_plural { -0.20 } else { 0.12 }
            } else if hf.noun_plural {
                0.12
            } else {
                0.02
            }
        }
        Some(Class::Det) => 0.10,
        _ => 0.0,
    };
    if allow_post_pp
        && words
            .get(end + 1)
            .is_some_and(|w| ["of", "for", "with"].contains(&w.as_str()))
    {
        if let Some((embedded_end, embedded_coherence)) = starting_at(words, end + 2, lex, false) {
            end = embedded_end;
            coherence = (coherence + 0.08 * embedded_coherence).min(1.0);
        }
    }
    Some((end, coherence.clamp(0.0, 1.0)))
}
