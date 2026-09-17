//! Bounded clause demotion for high-confidence POS ambiguity and surface errors.
use crate::{
    clause::{self, Structure},
    grammar::{Class, function_class},
    normalize_letters, phrase,
    wordnet::WordNet,
};

pub fn finite_be(word: &str) -> bool {
    [
        "am", "is", "are", "was", "were", "isnt", "arent", "wasnt", "werent",
    ]
    .contains(&word)
}
fn nominal(word: &str) -> bool {
    ["one", "more", "less", "most", "least"].contains(&word)
}
fn finite_aux(word: &str) -> bool {
    matches!(
        function_class(word),
        Some(
            Class::BeAux
                | Class::HaveAux
                | Class::Modal
                | Class::Dont
                | Class::Doesnt
                | Class::DoAux
        )
    )
}
fn nominal_compatible(subject: &str, aux: &str) -> bool {
    nominal(subject)
        && (subject != "one"
            || !["am", "are", "arent", "have", "havent", "do", "dont"].contains(&aux))
}

pub fn valid_subject(word: &str, lex: &WordNet) -> bool {
    let class = function_class(word);
    matches!(
        class,
        Some(Class::Pron | Class::Pron12 | Class::PronPl | Class::PronSg3)
    ) || nominal(word)
        || (class.is_none() && lex.features(word).noun)
}

pub fn finite_lexical(word: &str, lex: &WordNet) -> bool {
    if function_class(word).is_some() {
        return false;
    }
    let f = lex.features(word);
    f.verb_base || f.verb_3sg || f.verb_past
}

pub fn article_mismatch(article: &str, following: &str) -> bool {
    if !["a", "an"].contains(&article) {
        return false;
    }
    let word = normalize_letters(following);
    if word.is_empty() {
        return false;
    }
    let expected = if ["heir", "honest", "honor", "hour"]
        .iter()
        .any(|p| word.starts_with(p))
    {
        "an"
    } else if ["euro", "one", "once", "uni", "use", "user", "usual"]
        .iter()
        .any(|p| word.starts_with(p))
    {
        "a"
    } else if "aeiou".contains(word.as_bytes()[0] as char) {
        "an"
    } else {
        "a"
    };
    article != expected
}

pub fn subject_before(words: &[String], verb: usize, lex: &WordNet) -> Option<(usize, usize, f64)> {
    let mut head = verb.checked_sub(1)?;
    loop {
        let word = words.get(head)?;
        let f = lex.features(word);
        let c = function_class(word);
        if c == Some(Class::Neg) || (f.adv && !matches!(c, Some(Class::Prep | Class::Conj))) {
            head = head.checked_sub(1)?;
        } else {
            break;
        }
    }
    if !valid_subject(&words[head], lex) {
        return None;
    }
    let (start, coherence) = phrase::ending_at(words, head, lex)?;
    Some((start, head, coherence))
}

fn explicit_aux(words: &[String], lex: &WordNet) -> (bool, bool) {
    let mut found = false;
    for (i, token) in words.iter().enumerate() {
        if i == 0
            || !finite_aux(token)
            || (function_class(token) == Some(Class::BeAux) && !finite_be(token))
        {
            continue;
        }
        found = true;
        let subject = &words[i - 1];
        if !valid_subject(subject, lex) {
            continue;
        }
        if nominal(subject) {
            if nominal_compatible(subject, token) {
                return (true, true);
            }
            continue;
        }
        if phrase::ending_at(words, i - 1, lex).is_some() {
            return (true, true);
        }
    }
    (found, false)
}

pub fn lexical_coverage(words: &[String], lex: &WordNet) -> f64 {
    let mut best: f64 = 0.0;
    if words.len() < 2 {
        return best;
    }
    for (i, word) in words.iter().enumerate() {
        if !finite_lexical(word, lex) {
            continue;
        }
        let Some((start, _, _)) = subject_before(words, i, lex) else {
            continue;
        };
        let (_, tail) = clause::tail(word, &words[i + 1..], lex);
        let mut coverage = ((i - start + 1 + tail) as f64 / words.len() as f64).min(1.0);
        if start > 0 {
            coverage *= 0.90;
        }
        best = best.max(coverage);
    }
    best
}

fn demote(mut result: Structure, coverage: f64, hard: bool) -> Structure {
    let coverage = coverage.min(result.coverage).max(0.0);
    if hard || coverage <= 0.0 {
        result.norm = result.norm.min(0.25);
        result.coverage = result.coverage.min(0.25);
        result.valency = result.valency.min(0.50);
        result.agreement = result.agreement.min(0.50);
        result.kind = "fragment".into();
    } else {
        let ratio = (coverage / result.coverage).clamp(0.0, 1.0);
        result.norm = result
            .norm
            .min(result.norm * (0.25 + 0.75 * ratio.powf(1.4)));
        result.coverage = coverage;
        result.valency = result.valency.min(0.65);
        result.agreement = result.agreement.min(0.65);
        if coverage < 0.60 {
            result.kind = "fragment".into();
        }
    }
    result.norm = result.norm.max(0.0);
    result.raw = 4.0 * result.norm;
    result
}

pub fn adjust(words: &[String], lex: &WordNet, result: Structure) -> Structure {
    if !["clause", "copula"].contains(&result.kind.as_str()) || result.coverage <= 0.0 {
        return result;
    }
    let (has_aux, valid) = explicit_aux(words, lex);
    if has_aux {
        return if valid {
            result
        } else {
            demote(result, 0.25, true)
        };
    }
    if result.kind == "clause"
        && words.len() >= 2
        && ["do", "dont"].contains(&words[0].as_str())
        && lex.features(&words[1]).verb_base
    {
        return result;
    }
    if result.kind == "copula" {
        return result;
    }
    let coverage = lexical_coverage(words, lex);
    if coverage + 0.05 >= result.coverage {
        result
    } else {
        demote(result, coverage, coverage <= 0.0)
    }
}

fn det_aux(left: &str, right: &str) -> bool {
    !nominal_compatible(left, right) && phrase::determiner(left).is_some() && finite_aux(right)
}

pub fn pair(left: &str, right: &str) -> f64 {
    let mut score = 0.0;
    if finite_be(left) && finite_be(right) {
        score -= 1.0;
    }
    if det_aux(left, right) {
        score -= 1.75;
    }
    if article_mismatch(left, right) {
        score -= 1.60;
    }
    score
}

pub fn surface(words: &[String], mut result: Structure) -> Structure {
    let article = words
        .windows(2)
        .filter(|w| article_mismatch(&w[0], &w[1]))
        .count();
    let determiner = words.windows(2).filter(|w| det_aux(&w[0], &w[1])).count();
    let be = words
        .windows(2)
        .filter(|w| finite_be(&w[0]) && finite_be(&w[1]))
        .count();
    let penalty = 0.12 * article as f64 + 0.18 * determiner as f64 + 0.30 * be as f64;
    if penalty > 0.0 {
        result.norm = (result.norm - penalty).max(0.0);
        result.raw = 4.0 * result.norm;
    }
    result
}
