//! Corpus readers and request-local vocabulary admission. No network or globals.
use crate::{Inventory, generation::Candidate, normalize_letters};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{self, BufRead};

// Python's corpus readers use UTF-8 errors="ignore". Preserve valid text while
// dropping only malformed byte sequences, including errors inside count fields.
pub(crate) fn decoded_lines(reader: impl BufRead) -> impl Iterator<Item = io::Result<String>> {
    reader.split(b'\n').map(|bytes| {
        let bytes = bytes?;
        let mut remaining = bytes.as_slice();
        let mut text = String::new();
        while !remaining.is_empty() {
            match std::str::from_utf8(remaining) {
                Ok(valid) => {
                    text.push_str(valid);
                    break;
                }
                Err(error) => {
                    text.push_str(std::str::from_utf8(&remaining[..error.valid_up_to()]).unwrap());
                    let Some(length) = error.error_len() else {
                        break;
                    };
                    remaining = &remaining[error.valid_up_to() + length..];
                }
            }
        }
        Ok(text)
    })
}

#[derive(Default)]
pub struct Unigrams {
    counts: HashMap<String, i64>,
    total: i64,
}

impl Unigrams {
    pub fn load(reader: impl BufRead) -> io::Result<Self> {
        let mut model = Self::default();
        for line in decoded_lines(reader) {
            let line = line?;
            let line = line.trim();
            let pair = line.rsplit_once('\t').or_else(|| {
                line.rfind(char::is_whitespace)
                    .map(|i| (&line[..i], line[i..].trim()))
            });
            let Some((token, count)) = pair else {
                continue;
            };
            let Ok(count) = count.trim().parse::<i64>() else {
                continue;
            };
            let word = normalize_letters(token);
            if word.is_empty() {
                continue;
            }
            let entry = model.counts.entry(word).or_default();
            *entry = entry.checked_add(count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unigram count overflow")
            })?;
            model.total = model.total.checked_add(count).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "unigram total overflow")
            })?;
        }
        Ok(model)
    }

    pub fn count(&self, word: &str) -> i64 {
        self.counts
            .get(&normalize_letters(word))
            .copied()
            .unwrap_or(0)
    }

    pub fn zipf(&self, word: &str) -> f64 {
        let count = self.count(word);
        if count <= 0 || self.total <= 0 {
            0.0
        } else {
            ((count as f64 / self.total as f64) * 1_000_000_000.0).log10()
        }
    }
}

#[derive(Clone, Copy, Default, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShortPolicy {
    None,
    #[default]
    Common,
    All,
}

pub struct Admission {
    pub min_length: usize,
    pub max_length: usize,
    pub min_zipf: f64,
    pub short_policy: ShortPolicy,
    pub short_whitelist: BTreeSet<String>,
    pub forced: BTreeSet<String>,
    pub excluded: BTreeSet<String>,
    pub forbidden: BTreeSet<char>,
}

/// Exclusion predicate is supplied by the caller, keeping regex policy separate
/// from lexical ownership. Forced words bypass length/frequency, not exclusions.
pub fn admit(
    reader: impl BufRead,
    target: Inventory,
    policy: &Admission,
    unigrams: Option<&Unigrams>,
    excludes: impl Fn(&str) -> bool,
) -> io::Result<Vec<Candidate>> {
    admit_controlled(
        reader,
        target,
        policy,
        unigrams,
        excludes,
        &crate::control::Control::default(),
    )
}

pub fn admit_controlled(
    reader: impl BufRead,
    target: Inventory,
    policy: &Admission,
    unigrams: Option<&Unigrams>,
    excludes: impl Fn(&str) -> bool,
    control: &crate::control::Control,
) -> io::Result<Vec<Candidate>> {
    admit_checked(reader, target, policy, unigrams, excludes, || {
        control.check()
    })
}

pub(crate) fn admit_checked(
    reader: impl BufRead,
    target: Inventory,
    policy: &Admission,
    unigrams: Option<&Unigrams>,
    excludes: impl Fn(&str) -> bool,
    check: impl Fn() -> Result<(), &'static str>,
) -> io::Result<Vec<Candidate>> {
    check().map_err(io::Error::other)?;
    if policy.min_length == 0
        || policy.max_length < policy.min_length
        || !policy.min_zipf.is_finite()
        || policy.min_zipf < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid lexical limits",
        ));
    }
    let mut seen = HashSet::new();
    let mut admitted = Vec::new();
    let mut add = |word: String| {
        if word.is_empty() || !seen.insert(word.clone()) {
            return;
        }
        let forced = policy.forced.contains(&word);
        let length = word.len();
        let allowed_length = forced
            || if policy.short_whitelist.contains(&word) {
                length <= policy.max_length
            } else {
                length >= policy.min_length
                    && length <= policy.max_length
                    && (length > 2 || matches!(policy.short_policy, ShortPolicy::All))
            };
        if !allowed_length
            || policy.excluded.contains(&word)
            || excludes(&word)
            || word.chars().any(|c| policy.forbidden.contains(&c))
            || target.subtract(Inventory::from_text(&word)).is_none()
        {
            return;
        }
        let zipf = unigrams.map_or(0.0, |u| u.zipf(&word));
        if policy.min_zipf > 0.0 && zipf < policy.min_zipf && !forced {
            return;
        }
        admitted.push((Candidate::new(&word).unwrap(), zipf));
    };
    for line in decoded_lines(reader) {
        check().map_err(io::Error::other)?;
        add(normalize_letters(&line?));
    }
    for word in &policy.forced {
        check().map_err(io::Error::other)?;
        add(word.clone());
    }
    crate::control::Control::sort_checked(
        &mut admitted,
        |(a, az), (b, bz)| {
            bz.total_cmp(az)
                .then_with(|| b.word.len().cmp(&a.word.len()))
                .then_with(|| a.word.cmp(&b.word))
        },
        &check,
    )
    .map_err(io::Error::other)?;
    Ok(admitted
        .into_iter()
        .map(|(candidate, _)| candidate)
        .collect())
}
