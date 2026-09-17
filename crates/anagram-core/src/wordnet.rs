//! Immutable WordNet tables and morphological features. Request owners may cache
//! feature results without putting mutable state in the shared corpus.
use crate::normalize_letters;
use serde::Serialize;
use std::{
    collections::{BTreeSet, HashMap},
    fs, io,
    path::Path,
};

fn ascii(path: &Path) -> io::Result<String> {
    Ok(fs::read(path)?
        .into_iter()
        .filter(u8::is_ascii)
        .map(char::from)
        .collect())
}

fn index(path: &Path) -> io::Result<BTreeSet<String>> {
    Ok(ascii(path)?
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with(char::is_whitespace))
        .map(|line| {
            line.split(' ')
                .next()
                .unwrap()
                .trim()
                .to_ascii_lowercase()
                .replace('_', " ")
        })
        .filter(|word| !word.is_empty() && word.bytes().all(|b| b.is_ascii_alphabetic()))
        .collect())
}

fn exceptions(path: &Path) -> io::Result<HashMap<String, BTreeSet<String>>> {
    let text = match ascii(path) {
        Ok(text) => text,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(HashMap::new()),
        Err(e) => return Err(e),
    };
    Ok(text
        .lines()
        .filter_map(|line| {
            let words: Vec<_> = line
                .split_whitespace()
                .map(str::to_ascii_lowercase)
                .collect();
            (words.len() >= 2).then(|| (words[0].clone(), words[1..].iter().cloned().collect()))
        })
        .collect())
}

fn parse_frames(line: &str) -> Option<Vec<(String, u32)>> {
    if line.starts_with(char::is_whitespace) {
        return None;
    }
    let fields: Vec<_> = line.split_once('|')?.0.split_whitespace().collect();
    let count = usize::from_str_radix(fields.get(3)?, 16).ok()?;
    let mut pos = 4;
    let mut words = Vec::new();
    for _ in 0..count {
        words.push(fields.get(pos)?.to_ascii_lowercase().replace('_', " "));
        pos += 2;
    }
    let pointers: usize = fields.get(pos)?.parse().ok()?;
    pos = pos.checked_add(1 + pointers.checked_mul(4)?)?;
    let count: usize = fields.get(pos)?.parse().ok()?;
    pos += 1;
    let mut output = Vec::new();
    for _ in 0..count {
        if *fields.get(pos)? != "+" {
            return None;
        }
        let frame: u32 = fields.get(pos + 1)?.parse().ok()?;
        let target = usize::from_str_radix(fields.get(pos + 2)?, 16).ok()?;
        pos += 3;
        for (i, word) in words.iter().enumerate() {
            if (target == 0 || target == i + 1)
                && !word.is_empty()
                && word.bytes().all(|b| b.is_ascii_alphabetic())
            {
                output.push((word.clone(), frame));
            }
        }
    }
    Some(output)
}

#[derive(Clone, Copy, Debug, Default, Serialize, PartialEq, Eq)]
pub struct Features {
    pub noun: bool,
    pub verb: bool,
    pub adj: bool,
    pub adv: bool,
    pub noun_plural: bool,
    pub noun_singular: bool,
    pub verb_base: bool,
    pub verb_3sg: bool,
    pub verb_past: bool,
    pub verb_ing: bool,
    pub recognized: bool,
}

pub struct WordNet {
    nouns: BTreeSet<String>,
    verbs: BTreeSet<String>,
    adjs: BTreeSet<String>,
    advs: BTreeSet<String>,
    noun_exc: HashMap<String, BTreeSet<String>>,
    verb_exc: HashMap<String, BTreeSet<String>>,
    frames: HashMap<String, BTreeSet<u32>>,
}

impl WordNet {
    pub fn load(directory: &Path) -> io::Result<Self> {
        let mut frames: HashMap<String, BTreeSet<u32>> = HashMap::new();
        let data = match ascii(&directory.join("data.verb")) {
            Ok(data) => data,
            Err(e) if e.kind() == io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e),
        };
        for line in data.lines() {
            if let Some(entries) = parse_frames(line) {
                for (word, frame) in entries {
                    frames.entry(word).or_default().insert(frame);
                }
            }
        }
        Ok(Self {
            nouns: index(&directory.join("index.noun"))?,
            verbs: index(&directory.join("index.verb"))?,
            adjs: index(&directory.join("index.adj"))?,
            advs: index(&directory.join("index.adv"))?,
            noun_exc: exceptions(&directory.join("noun.exc"))?,
            verb_exc: exceptions(&directory.join("verb.exc"))?,
            frames,
        })
    }

    fn plural_bases(&self, word: &str) -> BTreeSet<String> {
        let mut out = self.noun_exc.get(word).cloned().unwrap_or_default();
        let n = word.len();
        if n > 3 && word.ends_with("ies") {
            out.insert(format!("{}y", &word[..n - 3]));
        }
        if n > 3
            && ["ches", "shes", "xes", "zes", "ses", "oes"]
                .iter()
                .any(|s| word.ends_with(s))
        {
            out.insert(word[..n - 2].into());
        }
        if n > 2 && word.ends_with('s') && !["ss", "us", "is"].iter().any(|s| word.ends_with(s)) {
            out.insert(word[..n - 1].into());
        }
        out.retain(|w| self.nouns.contains(w));
        out
    }

    fn verb_bases(&self, word: &str) -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
        let mut third = BTreeSet::new();
        let mut past = self.verb_exc.get(word).cloned().unwrap_or_default();
        let mut ing = BTreeSet::new();
        let n = word.len();
        if n > 3 && word.ends_with("ies") {
            third.insert(format!("{}y", &word[..n - 3]));
            third.insert(word[..n - 1].into());
        }
        if n > 3
            && ["ches", "shes", "xes", "zes", "ses", "oes"]
                .iter()
                .any(|s| word.ends_with(s))
        {
            third.insert(word[..n - 2].into());
        }
        if n > 2 && word.ends_with('s') && !["ss", "us", "is"].iter().any(|s| word.ends_with(s)) {
            third.insert(word[..n - 1].into());
        }
        if n > 3 && word.ends_with("ied") {
            past.insert(format!("{}y", &word[..n - 3]));
            past.insert(word[..n - 1].into());
        } else if n > 3 && word.ends_with("ed") {
            past.insert(word[..n - 2].into());
            past.insert(word[..n - 1].into());
            let stem = &word[..n - 2];
            if stem.len() >= 2 && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2]
            {
                past.insert(stem[..stem.len() - 1].into());
            }
        }
        if n > 4 && word.ends_with("ing") {
            let stem = &word[..n - 3];
            ing.insert(stem.into());
            ing.insert(format!("{stem}e"));
            if stem.len() >= 2 && stem.as_bytes()[stem.len() - 1] == stem.as_bytes()[stem.len() - 2]
            {
                ing.insert(stem[..stem.len() - 1].into());
            }
        }
        third.retain(|w| self.verbs.contains(w));
        past.retain(|w| self.verbs.contains(w));
        ing.retain(|w| self.verbs.contains(w));
        (third, past, ing)
    }

    pub fn morphology_family_word(&self, raw: &str) -> String {
        let word = normalize_letters(raw);
        if let Some(base) = self.plural_bases(&word).into_iter().next() {
            return base;
        }
        let (third, past, ing) = self.verb_bases(&word);
        third
            .into_iter()
            .chain(past)
            .chain(ing)
            .min()
            .unwrap_or(word)
    }

    pub fn verb_base_lemmas(&self, raw: &str) -> BTreeSet<String> {
        let word = normalize_letters(raw);
        let (mut out, past, ing) = self.verb_bases(&word);
        out.extend(past);
        out.extend(ing);
        if self.verbs.contains(&word) {
            out.insert(word);
        }
        out
    }

    pub fn frames_for(&self, raw: &str) -> BTreeSet<u32> {
        self.verb_base_lemmas(raw)
            .iter()
            .filter_map(|w| self.frames.get(w))
            .flatten()
            .copied()
            .collect()
    }

    pub fn allows(&self, raw: &str, category: &[u32]) -> Option<bool> {
        let frames = self.frames_for(raw);
        (!frames.is_empty()).then(|| category.iter().any(|frame| frames.contains(frame)))
    }

    pub fn features(&self, raw: &str) -> Features {
        let word = normalize_letters(raw);
        let noun_exact = self.nouns.contains(&word);
        let verb_exact = self.verbs.contains(&word);
        let adj = self.adjs.contains(&word);
        let adv = self.advs.contains(&word);
        let plural = !self.plural_bases(&word).is_empty();
        let (third, past, ing) = self.verb_bases(&word);
        let (third, past, ing) = (!third.is_empty(), !past.is_empty(), !ing.is_empty());
        Features {
            noun: noun_exact || plural,
            verb: verb_exact || third || past || ing,
            adj,
            adv,
            noun_plural: plural,
            noun_singular: noun_exact && !plural,
            verb_base: verb_exact,
            verb_3sg: third,
            verb_past: past,
            verb_ing: ing,
            recognized: is_function_word(&word)
                || noun_exact
                || verb_exact
                || adj
                || adv
                || plural
                || third
                || past
                || ing,
        }
    }
}

pub const FRAME_INTRANSITIVE: &[u32] = &[1, 2, 3, 23];
pub const FRAME_DIRECT_OBJECT: &[u32] = &[
    5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 30, 31,
];
pub const FRAME_PREDICATIVE: &[u32] = &[6, 7];
pub const FRAME_OBJECT_PREDICATIVE: &[u32] = &[5];
pub const FRAME_PP: &[u32] = &[4, 12, 13, 20, 21, 22, 27, 31];
pub const FRAME_INFINITIVE_OR_GERUND: &[u32] = &[24, 25, 28, 29, 30, 32, 33, 35];
pub const FRAME_CLAUSAL: &[u32] = &[26, 29, 34];

pub fn is_function_word(word: &str) -> bool {
    const WORDS: &str = "these those both many few several this that each every either neither another a an the some any no my your his her its our their whose what which one two three four five six seven eight nine ten eleven twelve dozen hundred thousand i you we they he she it me him us them myself yourself himself herself itself ourselves themselves who whom dont doesnt do did didnt can cant could couldnt will wont would wouldnt should shouldnt may might must am is are was were be been being isnt arent wasnt werent have has had havent hasnt hadnt about above across after against along among around at before behind below beneath beside between beyond by despite down during except for from in inside into near of off on onto over past since through throughout to toward under until up upon with within without like than and but or nor yet so although because if unless while not never";
    WORDS.split_whitespace().any(|w| w == word)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frames_apply_to_all_or_one_lemma_and_skip_malformed_records() {
        let result = parse_frames("000 00 v 02 give 0 offer 0 0 2 + 8 00 + 9 02 | gloss").unwrap();
        assert_eq!(
            result,
            vec![("give".into(), 8), ("offer".into(), 8), ("offer".into(), 9)]
        );
        assert!(parse_frames("000 00 v 02 give 0 | broken").is_none());
        assert!(parse_frames(" header | text").is_none());
    }
}
