//! Finite auxiliary chains, preserving lexical morphology and negation evidence.
use crate::{
    clause::{self, Number},
    grammar::{Class, function_class},
    wordnet::WordNet,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Chain {
    pub main_idx: usize,
    pub kind: String,
    pub quality: f64,
    pub passive: bool,
}

fn chain(main_idx: usize, kind: &str, quality: f64, passive: bool) -> Chain {
    Chain {
        main_idx,
        kind: kind.into(),
        quality,
        passive,
    }
}

fn skip_negation(words: &[String], index: usize) -> (usize, f64) {
    if words
        .get(index)
        .is_some_and(|w| function_class(w) == Some(Class::Neg))
    {
        (index + 1, 0.97)
    } else {
        (index, 1.0)
    }
}

enum Form {
    Base,
    Past,
    Ing,
}
fn lexical(words: &[String], index: usize, lex: &WordNet, form: Form) -> bool {
    let Some(word) = words.get(index) else {
        return false;
    };
    if function_class(word).is_some() {
        return false;
    }
    let f = lex.features(word);
    match form {
        Form::Base => f.verb_base,
        Form::Past => f.verb_past,
        Form::Ing => f.verb_ing,
    }
}

fn be(words: &[String], index: usize, lex: &WordNet) -> Option<Chain> {
    let (index, neg) = skip_negation(words, index + 1);
    if words.get(index)? == "being" {
        let (main, inner) = skip_negation(words, index + 1);
        return lexical(words, main, lex, Form::Past)
            .then(|| chain(main, "progressive-passive", 0.97 * neg * inner, true));
    }
    if lexical(words, index, lex, Form::Ing) {
        return Some(chain(index, "progressive", 0.99 * neg, false));
    }
    lexical(words, index, lex, Form::Past).then(|| chain(index, "passive", 0.96 * neg, true))
}

fn have(words: &[String], index: usize, lex: &WordNet) -> Option<Chain> {
    let (index, neg) = skip_negation(words, index + 1);
    if words.get(index)? == "been" {
        let nested = be(words, index, lex)?;
        return Some(chain(
            nested.main_idx,
            &format!("perfect-{}", nested.kind),
            0.98 * neg * nested.quality,
            nested.passive,
        ));
    }
    lexical(words, index, lex, Form::Past).then(|| chain(index, "perfect", 0.98 * neg, false))
}

pub fn parse(words: &[String], index: usize, lex: &WordNet) -> Option<Chain> {
    let word = words.get(index)?;
    match function_class(word) {
        Some(Class::BeAux) => {
            if ![
                "am", "is", "are", "was", "were", "isnt", "arent", "wasnt", "werent",
            ]
            .contains(&word.as_str())
            {
                return None;
            }
            be(words, index, lex)
        }
        Some(Class::HaveAux) => have(words, index, lex),
        Some(Class::Modal) => {
            let (index, neg) = skip_negation(words, index + 1);
            let word = words.get(index)?;
            let nested = match word.as_str() {
                "be" => be(words, index, lex),
                "have" => have(words, index, lex),
                _ => None,
            };
            if let Some(nested) = nested {
                return Some(chain(
                    nested.main_idx,
                    &format!("modal-{}", nested.kind),
                    0.98 * neg * nested.quality,
                    nested.passive,
                ));
            }
            lexical(words, index, lex, Form::Base).then(|| chain(index, "modal", 0.97 * neg, false))
        }
        Some(Class::Dont | Class::Doesnt | Class::DoAux) => {
            let (index, neg) = skip_negation(words, index + 1);
            lexical(words, index, lex, Form::Base)
                .then(|| chain(index, "do-support", 0.96 * neg, false))
        }
        _ => None,
    }
}

pub fn agreement(subject: &str, number: Number, auxiliary: &str, lex: &WordNet) -> f64 {
    use Number::*;
    match function_class(auxiliary) {
        Some(Class::Modal) => 0.98,
        Some(Class::BeAux) => match auxiliary {
            "am" => {
                if subject == "i" {
                    1.0
                } else {
                    0.05
                }
            }
            "is" | "isnt" => match number {
                Third => 1.0,
                NonThird => 0.10,
                Unknown => 0.55,
            },
            "are" | "arent" => {
                if subject == "i" {
                    0.05
                } else {
                    match number {
                        NonThird => 1.0,
                        Third => 0.15,
                        Unknown => 0.55,
                    }
                }
            }
            "was" | "wasnt" => {
                if subject == "i" || number == Third {
                    1.0
                } else if number == NonThird {
                    0.15
                } else {
                    0.55
                }
            }
            "were" | "werent" => {
                if subject == "i" || number == Third {
                    0.15
                } else if number == NonThird {
                    1.0
                } else {
                    0.55
                }
            }
            _ => 0.55,
        },
        Some(Class::HaveAux) => match auxiliary {
            "had" | "hadnt" => 0.96,
            "has" | "hasnt" => match number {
                Third => 1.0,
                NonThird => 0.10,
                Unknown => 0.55,
            },
            "have" | "havent" => match number {
                NonThird => 1.0,
                Third => 0.12,
                Unknown => 0.55,
            },
            _ => 0.55,
        },
        _ => clause::agreement(subject, auxiliary, lex, true, Some(number)),
    }
}
