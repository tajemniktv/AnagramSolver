//! Native solver primitives. The Python solver remains the reference until the
//! full generation and ranking parity gates pass.

use unicode_normalization::UnicodeNormalization;

pub mod auxiliary;
pub mod clause;
pub mod cohesion;
pub mod comparative;
pub mod contracts;
pub mod corpus_ranking;
pub mod diversity;
pub mod generation;
pub mod grammar;
pub mod learned;
pub mod lexicon;
pub mod ordering;
pub mod phrase;
pub mod phrase_evidence;
pub mod phrase_index;
pub mod ranking;
pub mod refinement;
pub mod request;
pub mod scoring;
pub mod solve;
pub mod structure;
pub mod validity;
pub mod wordnet;

/// Match Python's NFKD -> ASCII(ignore) -> lowercase -> [a-z] pipeline.
/// In particular, do not transliterate letters such as ł or ß.
pub fn normalize_letters(text: &str) -> String {
    text.nfkd()
        .filter(char::is_ascii)
        .map(|ch| ch.to_ascii_lowercase())
        .filter(char::is_ascii_lowercase)
        .collect()
}

/// Exact letter inventory; counts are wide enough for any in-memory input.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Inventory(pub [usize; 26]);

impl Inventory {
    pub fn from_text(text: &str) -> Self {
        let mut result = Self::default();
        for byte in normalize_letters(text).bytes() {
            result.0[usize::from(byte - b'a')] += 1;
        }
        result
    }

    pub fn subtract(self, other: Self) -> Option<Self> {
        let mut result = Self::default();
        for (index, amount) in self.0.iter().enumerate() {
            result.0[index] = amount.checked_sub(other.0[index])?;
        }
        Some(result)
    }

    pub fn len(self) -> usize {
        self.0.iter().sum()
    }

    pub fn is_empty(self) -> bool {
        self.0.iter().all(|amount| *amount == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_decomposition_without_transliteration() {
        assert_eq!(
            normalize_letters("École ﬁ Ａ ① Łódź Straße!"),
            "ecolefiaodzstrae"
        );
        assert_eq!(normalize_letters("DON’T stop—42"), "dontstop");
        assert_eq!(normalize_letters("中文🦀"), "");
    }

    #[test]
    fn inventory_preserves_multiplicity_and_rejects_underflow() {
        let inventory = Inventory::from_text("Açaí");
        assert_eq!(inventory.len(), 4);
        assert_eq!(
            inventory.subtract(Inventory::from_text("aa")),
            Some(Inventory::from_text("ci"))
        );
        assert_eq!(inventory.subtract(Inventory::from_text("aaa")), None);
        assert!(inventory.subtract(inventory).unwrap().is_empty());
    }
}
