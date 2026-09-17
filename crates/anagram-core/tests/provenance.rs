use anagram_core::{
    contracts::DataRepresentation, control::Control, corpus_ranking::PhraseCorpus,
    phrase_index::PhraseIndex, provenance::Snapshot,
};
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};

#[test]
fn learned_model_identity_tracks_loaded_bytes_without_changing_model_format() {
    use anagram_core::learned::Model;
    let model = Model::new([0.25; 18]).unwrap();
    assert!(model.identity().is_none());
    let bytes = serde_json::to_vec(&model).unwrap();
    let scratch = Scratch::new("model");
    let path = scratch.0.join("model.json");
    fs::write(&path, &bytes).unwrap();
    let loaded = Model::load(&path).unwrap();
    fs::write(&path, b"invalid replacement").unwrap();
    let identity = loaded.identity().unwrap();
    assert_eq!(identity.sha256, format!("{:x}", Sha256::digest(&bytes)));
    assert_eq!(identity.bytes, bytes.len() as u64);
    assert_eq!(identity.role, "learned_model");
    assert_eq!(serde_json::to_vec(&loaded).unwrap(), bytes);
    assert_eq!(loaded.score(&[1.0; 18]).unwrap(), 4.5);
    let items = vec![anagram_core::learned::Item {
        key: "first".into(),
        features: [1.0; 18],
        baseline_score: 2.0,
    }];
    let report =
        anagram_core::learned::rank_result(&items, Some(&loaded), &Control::default()).unwrap();
    assert_eq!(report.model_identity.as_ref(), Some(identity));
    assert_eq!(report.indices, vec![0]);
    assert_eq!(report.model_schema, Some(anagram_core::learned::SCHEMA));
    let baseline = anagram_core::learned::rank_result(&items, None, &Control::default()).unwrap();
    assert!(baseline.model_identity.is_none() && baseline.model_schema.is_none());
    let cancelled = Control::default();
    cancelled.cancel();
    let error = Model::load_controlled(&scratch.0.join("missing"), &cancelled).unwrap_err();
    assert_eq!(error.to_string(), "cancelled");
    fs::write(&path, vec![b' '; 65537]).unwrap();
    let error = Model::load_controlled(&path, &Control::default()).unwrap_err();
    assert_eq!(error.to_string(), "ranker model exceeds 64 KiB");
    assert!(anagram_core::learned::rank_result(&items, Some(&loaded), &cancelled).is_err());
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(name: &str) -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.codex/temp");
        fs::create_dir_all(&root).unwrap();
        let path = root
            .canonicalize()
            .unwrap()
            .join(format!("provenance-{}-{name}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn file_identity_hashes_the_bytes_retained_for_parsing() {
    let scratch = Scratch::new("file");
    let path = scratch.0.join("data");
    fs::write(&path, b"abc").unwrap();
    let snapshot = Snapshot::load(&path, "dictionary", &Control::default()).unwrap();
    fs::write(&path, b"changed").unwrap();
    assert_eq!(snapshot.bytes(), b"abc");
    assert_eq!(
        snapshot.identity().sha256,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(snapshot.identity().bytes, 3);
    assert!(snapshot.identity().present);
}

#[test]
fn phrase_lookup_rejects_duplicates_and_accepts_identifier_case() {
    let scratch = Scratch::new("case-duplicates");
    let path = scratch.0.join("phrases.sqlite");
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer.execute_batch("CREATE TABLE NGrams(text TEXT,n INTEGER,count INTEGER); INSERT INTO NGrams VALUES('a b',2,1);").unwrap();
    {
        let index = PhraseIndex::open(&path).unwrap();
        assert!(index.identity().is_ok());
    }
    writer
        .execute_batch("INSERT INTO NGrams VALUES('a b',2,99)")
        .unwrap();
    let index = PhraseIndex::open(&path).unwrap();
    assert!(index.counts(&["a b".into()]).is_err());
}

#[test]
fn sqlite_identity_and_queries_share_a_snapshot_during_wal_writes() {
    let scratch = Scratch::new("sqlite");
    let path = scratch.0.join("phrases.sqlite");
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE ngrams(text TEXT PRIMARY KEY,n INTEGER,count INTEGER); INSERT INTO ngrams VALUES ('a b',2,10);").unwrap();
    let index = PhraseIndex::open(&path).unwrap();
    let identity = index.identity().unwrap();
    assert_eq!(identity.representation, DataRepresentation::PhraseRowsV1);
    let mut canonical = serde_json::to_vec(&(
        "schema",
        "CREATE TABLE ngrams(text TEXT PRIMARY KEY,n INTEGER,count INTEGER)",
    ))
    .unwrap();
    canonical.extend_from_slice(b"\n[\"a b\",2,10]\n");
    assert_eq!(identity.sha256, format!("{:x}", Sha256::digest(&canonical)));
    assert_eq!(identity.bytes, canonical.len() as u64);
    writer.execute("UPDATE ngrams SET count=99", []).unwrap();
    assert_eq!(index.counts(&["a b".to_owned()]).unwrap()["a b"], 10);
    assert_eq!(index.identity().unwrap(), identity);
    let fresh = PhraseIndex::open(&path).unwrap();
    assert_eq!(fresh.counts(&["a b".to_owned()]).unwrap()["a b"], 99);
    assert_ne!(fresh.identity().unwrap(), identity);
}
