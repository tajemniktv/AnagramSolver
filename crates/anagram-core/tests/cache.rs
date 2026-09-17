use anagram_core::{cache::ranked_key, contracts::Versions, provenance::Snapshot, solve};

#[test]
fn cached_score_json_preserves_exact_float_bits() {
    for value in [
        0.9511422213778251_f64,
        0.9445381143324907,
        0.20584512177393266,
        1.8166666666666667,
    ] {
        let encoded = serde_json::to_vec(&value).unwrap();
        let decoded: f64 = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(value.to_bits(), decoded.to_bits());
    }
}
use anagram_core::{
    cache::{Lookup, StorageLimits, Store},
    control::Control,
};

struct Scratch(std::path::PathBuf);
#[test]
fn reopening_with_lower_limits_prunes_without_waiting_for_a_write() {
    let scratch = Scratch::new();
    let control = Control::default();
    let mut store = Store::open(
        &scratch.database(),
        StorageLimits {
            max_entries: 4,
            max_payload_bytes: 100,
        },
        &control,
    )
    .unwrap();
    for key in ["a", "b", "c"] {
        store.put(key, b"1234", 1).unwrap();
    }
    drop(store);
    let mut store = Store::open(
        &scratch.database(),
        StorageLimits {
            max_entries: 2,
            max_payload_bytes: 5,
        },
        &control,
    )
    .unwrap();
    assert!(matches!(store.lookup("a", false).unwrap(), Lookup::Miss));
    assert!(matches!(store.lookup("b", false).unwrap(), Lookup::Miss));
    assert!(matches!(store.lookup("c", false).unwrap(), Lookup::Hit(_)));
    let reader = rusqlite::Connection::open(scratch.database()).unwrap();
    let sizes: (i64, i64) = reader
        .query_row(
            "SELECT count(*),sum(length(payload)) FROM native_cache_v1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(sizes, (1, 4));
}
impl Scratch {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.codex/temp")
            .join(format!(
                "cache-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn database(&self) -> std::path::PathBuf {
        self.0.join("cache.sqlite")
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn persistent_cache_bounds_rebuild_and_corruption_recovery() {
    let scratch = Scratch::new();
    let control = Control::default();
    let limits = StorageLimits {
        max_entries: 2,
        max_payload_bytes: 8,
    };
    let mut store = Store::open(&scratch.database(), limits, &control).unwrap();
    assert!(store.put("a", b"1234", 37).unwrap());
    assert!(store.put("b", b"5678", 48).unwrap());
    assert!(matches!(store.lookup("a", true).unwrap(), Lookup::Miss));
    drop(store);
    let mut store = Store::open(&scratch.database(), limits, &control).unwrap();
    let Lookup::Hit(hit) = store.lookup("a", false).unwrap() else {
        panic!("missing persistent entry")
    };
    assert_eq!(hit.payload, b"1234");
    assert_eq!(hit.computation_ms, 37);
    assert!(!store.put("a", b"oversized", 0).unwrap());
    assert!(store.put("c", b"90", 0).unwrap());
    assert!(matches!(store.lookup("a", false).unwrap(), Lookup::Miss));
    let writer = rusqlite::Connection::open(scratch.database()).unwrap();
    writer
        .execute("UPDATE native_cache_v1 SET payload=x'00' WHERE key='b'", [])
        .unwrap();
    assert!(matches!(store.lookup("b", false).unwrap(), Lookup::Corrupt));
    assert!(matches!(store.lookup("b", false).unwrap(), Lookup::Miss));
    assert!(store.put("b", b"new", 5).unwrap());
    assert!(matches!(store.lookup("b", false).unwrap(), Lookup::Hit(_)));
}

#[test]
fn concurrent_writer_failure_never_publishes_or_evicts_partial_data() {
    let scratch = Scratch::new();
    let control = Control::default();
    let limits = StorageLimits {
        max_entries: 1,
        max_payload_bytes: 20,
    };
    let mut first = Store::open(&scratch.database(), limits, &control).unwrap();
    let mut second = Store::open(&scratch.database(), limits, &control).unwrap();
    first.put("old", b"complete", 1).unwrap();
    let mut writer = rusqlite::Connection::open(scratch.database()).unwrap();
    let lock = writer
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .unwrap();
    assert!(second.put("new", b"replacement", 2).is_err());
    drop(lock);
    assert!(matches!(
        first.lookup("old", false).unwrap(),
        Lookup::Hit(_)
    ));
    assert!(matches!(first.lookup("new", false).unwrap(), Lookup::Miss));
    second.put("new", b"replacement", 2).unwrap();
    assert!(matches!(first.lookup("old", false).unwrap(), Lookup::Miss));
    assert!(matches!(
        first.lookup("new", false).unwrap(),
        Lookup::Hit(_)
    ));
    control.cancel();
    assert_eq!(
        second.put("cancelled", b"none", 0).unwrap_err().code,
        "cancelled"
    );
}

#[test]
fn ranked_identity_preserves_result_affecting_inputs() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let value = fixtures
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["schema"] == "SolveRequest" && v["valid"] == true)
        .unwrap()["value"]
        .clone();
    let mut request: solve::Request = serde_json::from_value(value).unwrap();
    let mut versions = Versions {
        engine: "engine-1".into(),
        ranking: "ranking-1".into(),
        data_complete: true,
        data: vec![Snapshot::absent("dictionary"), Snapshot::absent("bigrams")],
    };
    let initial = ranked_key(&request, &versions).unwrap();
    request.generation.text = format!("{}!", request.generation.text.to_uppercase());
    versions.data.reverse();
    assert_eq!(initial, ranked_key(&request, &versions).unwrap());
    request.generation.candidate_budget += 1;
    assert_ne!(initial, ranked_key(&request, &versions).unwrap());
    request.generation.candidate_budget -= 1;
    request.beam_width += 1;
    assert_ne!(initial, ranked_key(&request, &versions).unwrap());
    request.beam_width -= 1;
    request.generation.exclude_regex.push("^zzz".into());
    assert_ne!(initial, ranked_key(&request, &versions).unwrap());
    request.generation.exclude_regex.clear();
    versions.ranking.push('2');
    assert_ne!(initial, ranked_key(&request, &versions).unwrap());
    versions.ranking.pop();
    versions.data[0].present = true;
    assert_ne!(initial, ranked_key(&request, &versions).unwrap());
    versions.data_complete = false;
    assert!(ranked_key(&request, &versions).is_err());
    versions.data_complete = true;
    versions.data.push(versions.data[0].clone());
    assert!(ranked_key(&request, &versions).is_err());
}
