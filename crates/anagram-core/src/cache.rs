//! Native cache identity. Persistent storage must use identities from the same
//! immutable input snapshots as computation, never pathname or mtime guesses.
use crate::control::Control;
use crate::{contracts::Versions, normalize_letters, request::Error, solve};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Bounds cover retained payload bytes and entry count, not SQLite page overhead.
#[derive(Clone, Copy)]
pub struct StorageLimits {
    pub max_entries: u32,
    pub max_payload_bytes: u32,
}

pub struct Config<'a> {
    pub path: &'a Path,
    pub limits: StorageLimits,
    pub rebuild: bool,
}

/// Final ranked payload only. Job identity, policy and lifecycle are never replayed.
#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RankedPayload {
    pub generated: usize,
    pub generation_stop: crate::generation::Stop,
    pub deep_selected: usize,
    pub deep_analyzed: usize,
    pub orders_evaluated: usize,
    pub corpus_rescored: usize,
    pub buckets: std::collections::BTreeMap<usize, Vec<crate::ranking::Row>>,
}

pub struct Stored {
    pub payload: Vec<u8>,
    /// Original computation duration; never the current request's latency.
    pub computation_ms: u64,
}

pub enum Lookup {
    Miss,
    Corrupt,
    Hit(Stored),
}

/// A dedicated native SQLite cache; never point this at a corpus database.
/// IMMEDIATE transactions serialize pruning and publication across processes.
/// Database-level errors are returned to the adapter for fail-open handling;
/// an unreadable database is never destructively replaced by this layer.
pub struct Store {
    connection: Connection,
    limits: StorageLimits,
    control: Control,
}

impl Store {
    /// Clear only our cache table, never delete or replace the database file.
    pub fn clear(&mut self) -> Result<(), Error> {
        check(&self.control)?;
        self.connection
            .execute("DELETE FROM native_cache_v1", [])
            .map_err(storage_error)?;
        Ok(())
    }
    pub fn open(path: &Path, limits: StorageLimits, control: &Control) -> Result<Self, Error> {
        check(control)?;
        if limits.max_entries == 0 || limits.max_payload_bytes == 0 {
            return Err(Error::new(
                "invalid_cache_limits",
                "Cache bounds must be positive",
            ));
        }
        let mut connection = Connection::open(path).map_err(storage_error)?;
        // Do not wait behind another process past this request's deadline.
        // Busy is an ordinary fail-open cache error, not a solver failure.
        connection
            .busy_timeout(std::time::Duration::ZERO)
            .map_err(storage_error)?;
        let callback = control.clone();
        connection
            .progress_handler(1000, Some(move || callback.check().is_err()))
            .map_err(storage_error)?;
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS native_cache_v1 (
            key TEXT PRIMARY KEY, payload BLOB NOT NULL, digest TEXT NOT NULL,
            computation_ms INTEGER NOT NULL, sequence INTEGER NOT NULL)",
            )
            .map_err(storage_error)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        prune(&transaction, limits, 0, 0, control)?;
        transaction.commit().map_err(storage_error)?;
        Ok(Self {
            connection,
            limits,
            control: control.clone(),
        })
    }

    pub fn lookup(&mut self, key: &str, rebuild: bool) -> Result<Lookup, Error> {
        check(&self.control)?;
        if rebuild {
            return Ok(Lookup::Miss);
        }
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        // Test length in SQL before transferring potentially corrupt/oversized blobs.
        let row = transaction.query_row(
            "SELECT CASE WHEN length(payload)<=?2 THEN payload ELSE NULL END, digest, computation_ms FROM native_cache_v1 WHERE key=?1",
            params![key, self.limits.max_payload_bytes],
            |row| Ok((row.get::<_, Option<Vec<u8>>>(0)?, row.get::<_, String>(1)?, row.get::<_, i64>(2)?))
        ).optional().map_err(storage_error)?;
        let Some((payload, digest, computation_ms)) = row else {
            return Ok(Lookup::Miss);
        };
        let valid = match &payload {
            Some(bytes) => payload_digest(bytes, &self.control)? == digest && computation_ms >= 0,
            None => false,
        };
        if !valid {
            transaction
                .execute("DELETE FROM native_cache_v1 WHERE key=?1", [key])
                .map_err(storage_error)?;
            transaction.commit().map_err(storage_error)?;
            return Ok(Lookup::Corrupt);
        }
        check(&self.control)?;
        transaction.commit().map_err(storage_error)?;
        Ok(Lookup::Hit(Stored {
            payload: payload.unwrap(),
            computation_ms: computation_ms as u64,
        }))
    }

    /// Oversized results are not cached. Replacement and FIFO eviction commit
    /// together, so readers never observe a partially written result.
    pub fn put(&mut self, key: &str, payload: &[u8], computation_ms: u64) -> Result<bool, Error> {
        check(&self.control)?;
        if key.is_empty() || key.len() > 64 {
            return Err(Error::new(
                "invalid_cache_key",
                "Cache keys must contain 1 to 64 bytes",
            ));
        }
        if payload.len() > self.limits.max_payload_bytes as usize {
            return Ok(false);
        }
        let computation_ms = i64::try_from(computation_ms).map_err(|_| {
            Error::new(
                "invalid_cache_timing",
                "Computation duration exceeds storage range",
            )
        })?;
        let digest = payload_digest(payload, &self.control)?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(storage_error)?;
        transaction
            .execute("DELETE FROM native_cache_v1 WHERE key=?1", [key])
            .map_err(storage_error)?;
        prune(
            &transaction,
            self.limits,
            1,
            payload.len() as i64,
            &self.control,
        )?;
        transaction.execute("INSERT INTO native_cache_v1 VALUES (?1,?2,?3,?4,(SELECT coalesce(max(sequence),0)+1 FROM native_cache_v1))", params![key,payload,digest,computation_ms]).map_err(storage_error)?;
        check(&self.control)?;
        transaction.commit().map_err(storage_error)?;
        Ok(true)
    }
}

fn prune(
    transaction: &rusqlite::Transaction<'_>,
    limits: StorageLimits,
    reserve_entries: i64,
    reserve_bytes: i64,
    control: &Control,
) -> Result<(), Error> {
    loop {
        check(control)?;
        let (count, bytes): (i64, i64) = transaction
            .query_row(
                "SELECT count(*), coalesce(sum(length(payload)),0) FROM native_cache_v1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(storage_error)?;
        if count + reserve_entries <= i64::from(limits.max_entries)
            && bytes + reserve_bytes <= i64::from(limits.max_payload_bytes)
        {
            return Ok(());
        }
        transaction.execute("DELETE FROM native_cache_v1 WHERE key=(SELECT key FROM native_cache_v1 ORDER BY sequence,key LIMIT 1)", []).map_err(storage_error)?;
    }
}

fn check(control: &Control) -> Result<(), Error> {
    control.check().map_err(|code| Error::new(code, code))
}

fn payload_digest(bytes: &[u8], control: &Control) -> Result<String, Error> {
    let mut digest = Sha256::new();
    for chunk in bytes.chunks(16 * 1024) {
        check(control)?;
        digest.update(chunk);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn storage_error(error: rusqlite::Error) -> Error {
    Error::new("cache_error", error.to_string())
}

/// Version the cache format independently of the public transport schema.
pub const FORMAT_VERSION: u32 = 2;

/// Canonical ranked-request key. Array order and repeated required words are
/// retained conservatively: forced-word order can affect intermediate ties.
/// Deployment deadlines and observer/job identity are deliberately not semantic
/// inputs; only successful completed computations may be stored under this key.
pub fn ranked_key(request: &solve::Request, versions: &Versions) -> Result<String, Error> {
    solve::validate(request)?;
    if !versions.data_complete || versions.engine.is_empty() || versions.ranking.is_empty() {
        return Err(Error::new(
            "invalid_cache_identity",
            "Complete input and engine identities are required",
        ));
    }
    let mut data = versions.data.clone();
    data.sort_by(|a, b| a.role.cmp(&b.role));
    if data.is_empty()
        || data.windows(2).any(|pair| pair[0].role == pair[1].role)
        || data.iter().any(|item| {
            item.role.is_empty()
                || item.sha256.len() != 64
                || !item
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    {
        return Err(Error::new(
            "invalid_cache_identity",
            "Input identities must have unique roles and canonical SHA-256 hashes",
        ));
    }
    let encode = |error: serde_json::Error| Error::new("serialization_error", error.to_string());
    let mut semantic = serde_json::to_value(request).map_err(encode)?;
    let generation = &mut semantic["generation"];
    generation["text"] = normalize_letters(&request.generation.text).into();
    generation["forbid_chars"] = normalize_letters(&request.generation.forbid_chars).into();
    for name in ["required", "hints", "excluded", "extra_short_words"] {
        if let Some(words) = generation[name].as_array_mut() {
            for word in words {
                *word =
                    normalize_letters(word.as_str().expect("serialized string constraint")).into();
            }
        }
    }
    let bytes = serde_json::to_vec(&(
        FORMAT_VERSION,
        &versions.engine,
        &versions.ranking,
        data,
        semantic,
    ))
    .map_err(encode)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
