//! Conservative local grammatical evidence; phrase-structure scoring is separate.
use crate::wordnet::{FRAME_DIRECT_OBJECT, FRAME_PREDICATIVE, WordNet};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Class {
    DetPl,
    DetSg,
    Article,
    NumDet,
    Det,
    #[serde(rename = "PRON_12")]
    Pron12,
    PronPl,
    PronSg3,
    Pron,
    Dont,
    Doesnt,
    DoAux,
    Modal,
    BeAux,
    HaveAux,
    Prep,
    Conj,
    Neg,
}

fn member(word: &str, list: &str) -> bool {
    list.split_whitespace().any(|w| w == word)
}

pub fn function_class(word: &str) -> Option<Class> {
    use Class::*;
    // Order matters for ambiguous words: e.g. "her", "for", "you".
    for (class, words) in [
        (DetPl, "these those both many few several"),
        (DetSg, "this that each every either neither another"),
        (Article, "a an the"),
        (
            NumDet,
            "one two three four five six seven eight nine ten eleven twelve dozen hundred thousand",
        ),
        (
            Det,
            "some any no my your his her its our their whose what which",
        ),
        (Pron12, "i you we"),
        (PronPl, "we they you"),
        (PronSg3, "he she it"),
        (
            Pron,
            "me him her us them myself yourself himself herself itself ourselves themselves who whom",
        ),
        (Dont, "dont"),
        (Doesnt, "doesnt"),
        (DoAux, "do did didnt"),
        (
            Modal,
            "can cant could couldnt will wont would wouldnt should shouldnt may might must",
        ),
        (
            BeAux,
            "am is are was were be been being isnt arent wasnt werent",
        ),
        (HaveAux, "have has had havent hasnt hadnt"),
        (
            Prep,
            "about above across after against along among around at before behind below beneath beside between beyond by despite down during except for from in inside into near of off on onto over past since through throughout to toward under until up upon with within without like than",
        ),
        (
            Conj,
            "and but or nor for yet so although because if unless while",
        ),
        (Neg, "not never"),
    ] {
        if member(word, words) {
            return Some(class);
        }
    }
    None
}

pub fn pair(left: &str, right: &str, lex: &WordNet) -> f64 {
    use Class::*;
    let lf = lex.features(left);
    let rf = lex.features(right);
    let lc = function_class(left);
    let rc = function_class(right);
    let mut score = 0.0;
    score += match lc {
        Some(DetPl) => {
            if rf.noun_plural {
                4.0
            } else if rf.adj {
                0.75
            } else if rf.noun_singular {
                -2.5
            } else if rc.is_some() {
                -2.0
            } else {
                0.0
            }
        }
        Some(DetSg) => {
            if rf.noun_singular {
                3.0
            } else if rf.adj {
                0.75
            } else if rf.noun_plural {
                -2.5
            } else if rc.is_some() {
                -1.5
            } else {
                0.0
            }
        }
        Some(Article) => {
            if rf.noun {
                if member(left, "a an") && rf.noun_plural {
                    -3.0
                } else {
                    2.0
                }
            } else if rf.adj {
                1.0
            } else if rc.is_some() {
                -1.5
            } else {
                0.0
            }
        }
        Some(Det) => {
            if rf.noun {
                1.8
            } else if rf.adj {
                0.8
            } else {
                0.0
            }
        }
        Some(NumDet) => {
            if rf.noun {
                1.8
            } else if rf.adj {
                0.4
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    if lc == Some(BeAux) {
        if rf.adj || rf.noun {
            score += 2.4;
        } else if rf.adv {
            score += 1.0;
        }
    }
    if rc == Some(BeAux) && (lf.noun || matches!(lc, Some(Pron12 | PronPl | PronSg3 | Pron))) {
        score += 1.2;
    }
    score += match lc {
        Some(Dont) => {
            if rf.verb_base {
                4.5
            } else if rf.verb_3sg {
                -3.5
            } else if rf.verb {
                -2.0
            } else if rf.recognized {
                -1.5
            } else {
                0.0
            }
        }
        Some(Doesnt) => {
            if rf.verb_base {
                4.5
            } else if rf.verb_3sg {
                -3.0
            } else if rf.recognized {
                -1.5
            } else {
                0.0
            }
        }
        Some(Modal) => {
            if rf.verb_base {
                3.5
            } else if rf.verb {
                -2.0
            } else {
                0.0
            }
        }
        Some(DoAux) => {
            if right == "not" {
                1.5
            } else if rf.verb_base {
                2.5
            } else {
                0.0
            }
        }
        _ => 0.0,
    };
    if rc == Some(Dont) {
        if member(left, "i you we they") || lf.noun_plural {
            score += 2.5;
        } else if member(left, "he she it") {
            score -= 2.5;
        } else if lf.noun_singular {
            score -= 1.5;
        }
    } else if rc == Some(Doesnt) {
        if member(left, "he she it") {
            score += 2.5;
        } else if lf.noun_singular {
            score += 2.0;
        } else if lf.noun_plural || member(left, "i you we they") {
            score -= 2.0;
        }
    }
    if left == "to" && rf.verb_base {
        score += 1.8;
    }
    // Reference checks the class against "to", so PREP includes "to" here.
    if lc == Some(Prep) {
        if matches!(rc, Some(Article | Det | DetPl | DetSg)) {
            score += 1.0;
        } else if rf.noun {
            score += 0.7;
        } else if matches!(rc, Some(Pron | Pron12 | PronPl | PronSg3)) {
            score += 0.6;
        }
    }
    if lf.adj && rf.noun {
        score += 0.6;
    }
    if lf.adv && rf.verb {
        score += 0.25;
    }
    if lf.noun && rf.verb && rc.is_none() {
        score += 0.75;
    }
    if lf.verb && rf.noun && lc.is_none() && lex.allows(left, FRAME_DIRECT_OBJECT) == Some(true) {
        score += 0.9;
    }
    if lf.verb
        && (rf.adj || rf.adv)
        && lc.is_none()
        && lex.allows(left, FRAME_PREDICATIVE) == Some(true)
    {
        score += 2.0;
    }
    if matches!(lc, Some(Article | Det | DetPl | DetSg))
        && matches!(rc, Some(Dont | Doesnt | Modal | Conj))
    {
        score -= 1.5;
    }
    if lc == Some(Conj) && rc == Some(Conj) {
        score -= 1.5;
    }
    score
}

pub fn start(word: &str, lex: &WordNet) -> f64 {
    use Class::*;
    let class = function_class(word);
    let f = lex.features(word);
    if matches!(class, Some(Conj | Prep)) && !member(word, "in on at after before") {
        -0.4
    } else if matches!(
        class,
        Some(Article | Det | DetPl | DetSg | Pron12 | PronPl | PronSg3 | Pron)
    ) {
        0.35
    } else if f.noun || f.adv {
        0.15
    } else {
        0.0
    }
}

pub fn end(word: &str, lex: &WordNet) -> f64 {
    use Class::*;
    let class = function_class(word);
    let f = lex.features(word);
    if matches!(
        class,
        Some(Article | Det | DetPl | DetSg | Prep | Conj | Dont | Doesnt | Modal | DoAux)
    ) {
        -1.25
    } else if f.verb || f.noun || f.adj || f.adv {
        0.2
    } else {
        0.0
    }
}

pub fn local_raw(words: &[String], lex: &WordNet) -> f64 {
    if words.is_empty() {
        return 0.0;
    }
    let total = start(&words[0], lex) + end(words.last().unwrap(), lex);
    if words.len() == 1 {
        return total;
    }
    (total
        + words
            .windows(2)
            .map(|w| pair(&w[0], &w[1], lex))
            .sum::<f64>())
        / (words.len() - 1) as f64
}

pub fn potential(words: &[String], lex: &WordNet) -> f64 {
    if words.len() <= 1 {
        return 0.0;
    }
    let mut positives = Vec::new();
    for (i, left) in words.iter().enumerate() {
        for (j, right) in words.iter().enumerate() {
            if i != j {
                let value = pair(left, right, lex);
                if value > 0.0 {
                    positives.push(value);
                }
            }
        }
    }
    positives.sort_by(|a, b| b.total_cmp(a));
    let raw = positives.iter().take(words.len() - 1).sum::<f64>() / (words.len() - 1) as f64;
    1.0 / (1.0 + (-((raw - 1.5) * 1.35)).exp())
}

pub fn coverage(words: &[String], lex: &WordNet) -> f64 {
    if words.is_empty() {
        0.0
    } else {
        words.iter().filter(|w| lex.features(w).recognized).count() as f64 / words.len() as f64
    }
}
