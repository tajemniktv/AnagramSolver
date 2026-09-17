use anagram_core::{
    control::Control, lexicon::Unigrams, ordering, phrase_index::PhraseIndex, wordnet::WordNet,
};
use std::{
    fs,
    io::{self, BufRead, Cursor, Read},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
fn exact_bigram_dynamic_program_honors_deadline() {
    let unigrams = Unigrams::load("".as_bytes()).unwrap();
    let words: Vec<_> = (b'a'..=b'p').map(|c| (c as char).to_string()).collect();
    let model = anagram_core::scoring::Bigrams::load(
        "".as_bytes(),
        &unigrams,
        &words.iter().cloned().collect(),
    )
    .unwrap();
    let start = Instant::now();
    let control = Control::with_deadline(start + Duration::from_millis(20));
    assert_eq!(
        model.best_order_controlled(&words, &control).unwrap_err(),
        "timed_out"
    );
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn controlled_sort_is_stable_and_preserves_payloads_on_interruption() {
    for n in 0..150 {
        let mut actual: Vec<_> = (0..n).map(|i| ((i * 37) % 11, i)).collect();
        let mut expected = actual.clone();
        expected.sort_by_key(|v| v.0);
        Control::default()
            .sort_by(&mut actual, |a, b| a.0.cmp(&b.0))
            .unwrap();
        assert_eq!(actual, expected);
    }
    let control = Control::default();
    let mut values: Vec<_> = (0..1000).rev().collect();
    let original = values.clone();
    let mut comparisons = 0;
    let outcome = control.sort_by(&mut values, |a, b| {
        comparisons += 1;
        if comparisons == 50 {
            control.cancel();
        }
        a.cmp(b)
    });
    assert_eq!(outcome, Err("cancelled"));
    assert_eq!(comparisons, 50);
    assert_eq!(values, original);
}

#[test]
fn refinement_acknowledges_cancellation_before_another_score() {
    use anagram_core::refinement::{Options, augment_pool_controlled, refine_controlled};
    let seed: Vec<String> = ["a", "b", "c", "d"].map(String::from).to_vec();
    let control = Control::default();
    let mut calls = 0;
    let result = refine_controlled(
        &seed,
        &mut |_| {
            calls += 1;
            if calls == 2 {
                control.cancel();
            }
            0.0
        },
        Options::default(),
        None,
        &control,
    );
    assert_eq!(result.unwrap_err(), "cancelled");
    assert_eq!(calls, 2);
    let expired = Control::with_deadline(Instant::now());
    let result = augment_pool_controlled(
        &[seed],
        &mut |_| panic!("expired work scored"),
        1,
        Options::default(),
        &expired,
    );
    assert_eq!(result.unwrap_err(), "timed_out");
}
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.codex/temp")
            .join(format!(
                "control-{}-{}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
struct CancelOnRead {
    data: Cursor<Vec<u8>>,
    control: Control,
    reads: usize,
}
impl Read for CancelOnRead {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        self.data.read(b)
    }
}
impl BufRead for CancelOnRead {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.reads += 1;
        if self.reads == 3 {
            self.control.cancel();
        }
        self.data.fill_buf()
    }
    fn consume(&mut self, n: usize) {
        self.data.consume(n)
    }
}

#[test]
fn corpus_cancellation_is_not_retried_as_interrupted_io() {
    let control = Control::default();
    let reader = CancelOnRead {
        data: Cursor::new(b"a\t100\nb\t200\nc\t300\nd\t400\n".to_vec()),
        control: control.clone(),
        reads: 0,
    };
    let error = Unigrams::load(control.reader(reader)).err().unwrap();
    assert_eq!(error.to_string(), "cancelled");
    assert_ne!(error.kind(), io::ErrorKind::Interrupted);
}

#[test]
fn generation_admission_reports_cancellation_not_corpus_failure() {
    use anagram_core::request::{GenerateRequest, generate_controlled};
    let request: GenerateRequest = serde_json::from_value(serde_json::json!({
        "schema_version": 1, "text": "ateate", "required": [], "hints": [],
        "excluded": [], "min_words": 1, "max_words": 4, "min_word_length": 1,
        "max_word_length": 6, "min_zipf": 0.0, "allow_repeat": true,
        "hint_mode": "any", "strategy": "prefix", "candidate_budget": 10
    }))
    .unwrap();
    let control = Control::default();
    let reader = CancelOnRead {
        data: Cursor::new(b"ate\neat\ntea\na\n".to_vec()),
        control: control.clone(),
        reads: 0,
    };
    let error = generate_controlled(&request, reader, None, &control)
        .err()
        .unwrap();
    assert_eq!(error.code, "cancelled");
    let expired = Control::with_deadline(Instant::now());
    let error = generate_controlled(&request, Cursor::new(b"ate"), None, &expired)
        .err()
        .unwrap();
    assert_eq!(error.code, "timed_out");
    let legacy = Control::default();
    let reader = CancelOnRead {
        data: Cursor::new(b"ate\neat\ntea\na\n".to_vec()),
        control: legacy.clone(),
        reads: 0,
    };
    let error = anagram_core::request::generate(&request, reader, None, legacy.flag(), None)
        .err()
        .unwrap();
    assert_eq!(error.code, "cancelled");
    let uncancelled = Control::default();
    let error = anagram_core::request::generate(
        &request,
        Cursor::new(b"ate"),
        None,
        uncancelled.flag(),
        Some(Instant::now()),
    )
    .err()
    .unwrap();
    assert_eq!(error.code, "timed_out");
}

#[test]
fn exact_ordering_stops_inside_permutation_search() {
    let scratch = Scratch::new();
    for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
        fs::write(scratch.0.join(name), "dog n\ncat n\n").unwrap();
    }
    let lex = WordNet::load(&scratch.0).unwrap();
    let words = "the small dog and a black cat will run today"
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let start = Instant::now();
    let control = Control::with_deadline(start + Duration::from_millis(20));
    let error = ordering::rank_controlled(&words, &lex, true, 128, 56, &control).unwrap_err();
    assert_eq!(error, "timed_out");
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn order_progress_counts_completed_scores_before_cancellation() {
    let scratch = Scratch::new();
    for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
        fs::write(scratch.0.join(name), "").unwrap();
    }
    let lex = WordNet::load(&scratch.0).unwrap();
    let words = ["a", "dog", "runs"].map(str::to_owned);
    let control = Control::default();
    let mut counts = Vec::new();
    let result = ordering::rank_observed(&words, &lex, true, 128, 6, &control, &mut |count| {
        counts.push(count);
        if count == 3 {
            control.cancel();
        }
    });
    assert_eq!(result.unwrap_err(), "cancelled");
    assert_eq!(counts, vec![1, 2, 3]);
}

#[test]
fn sqlite_deadline_interrupts_query_vm_not_just_next_batch() {
    let scratch = Scratch::new();
    let path = scratch.0.join("slow.sqlite");
    let connection = rusqlite::Connection::open(&path).unwrap();
    connection.execute_batch("CREATE VIEW ngrams AS WITH RECURSIVE seq(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM seq WHERE x<100000000) SELECT 'a b' AS text, x AS n, 1 AS count FROM seq;").unwrap();
    drop(connection);
    let start = Instant::now();
    let control = Control::with_deadline(start + Duration::from_millis(20));
    assert!(PhraseIndex::open_controlled(&path, &control).is_err());
    assert_eq!(control.check(), Err("timed_out"));
    assert!(start.elapsed() < Duration::from_secs(5));
}
