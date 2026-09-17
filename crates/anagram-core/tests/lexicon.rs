use anagram_core::{Inventory, lexicon::*};
use std::collections::BTreeSet;

#[test]
fn negative_counts_and_nonletter_forced_entries_are_safe() {
    assert!(Unigrams::load(b"cat 5\ndog -1\n".as_slice()).is_err());
    assert!(
        anagram_core::corpus_ranking::Collocation::load(
            b"cat 5\n".as_slice(),
            b"cat dog -1\n".as_slice(),
            &BTreeSet::new()
        )
        .is_err()
    );
    let policy = Admission {
        min_length: 1,
        max_length: 5,
        min_zipf: 0.0,
        short_policy: ShortPolicy::All,
        short_whitelist: BTreeSet::new(),
        forced: BTreeSet::from(["123".into()]),
        excluded: BTreeSet::new(),
        forbidden: BTreeSet::new(),
    };
    assert!(
        admit(
            b"".as_slice(),
            Inventory::from_text("cat"),
            &policy,
            None,
            |_| false
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn short_policy_keeps_explicit_whitelist_but_all_admits_other_short_words() {
    for (short_policy, expected) in [
        (ShortPolicy::None, vec!["at"]),
        (ShortPolicy::Common, vec!["at"]),
        (ShortPolicy::All, vec!["at", "ta"]),
    ] {
        let policy = Admission {
            min_length: 1,
            max_length: 2,
            min_zipf: 0.0,
            short_policy,
            short_whitelist: BTreeSet::from(["at".into()]),
            forced: BTreeSet::new(),
            excluded: BTreeSet::new(),
            forbidden: BTreeSet::new(),
        };
        let actual = admit(
            "at\nta\n".as_bytes(),
            Inventory::from_text("at"),
            &policy,
            None,
            |_| false,
        )
        .unwrap();
        assert_eq!(
            actual.iter().map(|c| c.word.as_str()).collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn frequency_normalization_and_duplicate_rows() {
    let model = Unigrams::load("Café\t10\ncafe 5\ntea\t5\ninvalid\n".as_bytes()).unwrap();
    assert_eq!(model.count("CAFÉ"), 15);
    assert!((model.zipf("cafe") - 750_000_000_f64.log10()).abs() < 1e-12);
    assert_eq!(model.zipf("absent"), 0.0);
}

#[test]
fn malformed_utf8_is_ignored_like_reference() {
    let model = Unigrams::load(&b"ca\xfffe\t1\xff0\ntea\t5\n"[..]).unwrap();
    assert_eq!(model.count("cafe"), 10);
    assert_eq!(model.count("tea"), 5);
}

#[test]
fn forced_words_do_not_bypass_exclusion_and_order_is_stable() {
    let policy = Admission {
        min_length: 3,
        max_length: 4,
        min_zipf: 0.0,
        short_policy: ShortPolicy::None,
        short_whitelist: BTreeSet::from(["a".into()]),
        forced: BTreeSet::from(["at".into(), "eat".into(), "tea".into()]),
        excluded: BTreeSet::from(["eat".into()]),
        forbidden: BTreeSet::new(),
    };
    let words = admit(
        "a\nat\ntea\neat\nate\nATE\nx\n".as_bytes(),
        Inventory::from_text("ate"),
        &policy,
        None,
        |_| false,
    )
    .unwrap();
    assert_eq!(
        words.iter().map(|c| c.word.as_str()).collect::<Vec<_>>(),
        ["ate", "tea", "at", "a"]
    );
}
