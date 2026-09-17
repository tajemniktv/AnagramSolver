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
