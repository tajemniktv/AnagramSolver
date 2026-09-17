//! Read-only SQLite phrase corpus. Each worker owns its connection.
use crate::contracts::{DataIdentity, DataRepresentation};
use crate::corpus_ranking::PhraseCorpus;
use rusqlite::{Connection, OpenFlags, params_from_iter};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io,
    path::Path,
};

pub struct PhraseIndex {
    connection: Connection,
    max_n: usize,
    control: crate::control::Control,
}
fn error(e: rusqlite::Error) -> io::Error {
    io::Error::other(e)
}
impl PhraseIndex {
    pub fn open(path: &Path) -> io::Result<Self> {
        Self::open_controlled(path, &crate::control::Control::default())
    }
    pub fn open_controlled(path: &Path, control: &crate::control::Control) -> io::Result<Self> {
        control.check_io()?;
        // Deliberately omit URI parsing: a user-supplied filename is a path.
        let connection = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(error)?;
        let callback_control = control.clone();
        connection
            .progress_handler(1000, Some(move || callback_control.check().is_err()))
            .map_err(error)?;
        // MAX(n), provenance and every later query must see one read snapshot.
        connection.execute_batch("BEGIN").map_err(error)?;
        let max_n: i64 = connection
            .query_row("SELECT COALESCE(MAX(n), 0) FROM ngrams", [], |row| {
                row.get(0)
            })
            .map_err(error)?;
        let max_n = usize::try_from(max_n)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "negative phrase order"))?;
        Ok(Self {
            connection,
            max_n,
            control: control.clone(),
        })
    }

    /// Canonical logical identity, not a hash of SQLite pages or WAL files.
    pub fn identity(&self) -> io::Result<DataIdentity> {
        self.control.check_io()?;
        let (kind, schema): (String, String) = self
            .connection
            .query_row(
                "SELECT type, sql FROM sqlite_schema WHERE name='ngrams' COLLATE NOCASE",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .map_err(error)?;
        if kind != "table" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Phrase provenance requires a concrete ngrams table, not a dynamic view",
            ));
        }
        let mut statement = self
            .connection
            .prepare("SELECT text, n, count FROM ngrams ORDER BY text COLLATE BINARY, n, count")
            .map_err(error)?;
        let mut rows = statement.query([]).map_err(error)?;
        let mut digest = Sha256::new();
        // Collations can affect lookup semantics even for identical visible rows.
        let mut header = serde_json::to_vec(&("schema", schema)).map_err(io::Error::other)?;
        header.push(b'\n');
        let mut bytes = header.len() as u64;
        digest.update(header);
        let mut previous = None;
        while let Some(row) = rows.next().map_err(error)? {
            self.control.check_io()?;
            let record = (
                row.get::<_, Option<String>>(0).map_err(error)?,
                row.get::<_, Option<i64>>(1).map_err(error)?,
                row.get::<_, Option<i64>>(2).map_err(error)?,
            );
            if previous.as_ref().is_some_and(|text| text == &record.0) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Duplicate phrase keys have no deterministic lookup identity",
                ));
            }
            previous = Some(record.0.clone());
            let mut encoded = serde_json::to_vec(&record).map_err(io::Error::other)?;
            encoded.push(b'\n');
            bytes = bytes
                .checked_add(encoded.len() as u64)
                .ok_or_else(|| io::Error::other("Phrase identity size overflow"))?;
            digest.update(encoded);
        }
        Ok(DataIdentity {
            role: "phrase_index".to_owned(),
            representation: DataRepresentation::PhraseRowsV1,
            present: true,
            sha256: format!("{:x}", digest.finalize()),
            bytes,
        })
    }
}
impl PhraseCorpus for PhraseIndex {
    fn max_n(&self) -> usize {
        self.max_n
    }
    fn counts(&self, phrases: &[String]) -> io::Result<HashMap<String, i64>> {
        self.control.check_io()?;
        let mut seen = HashSet::new();
        let unique: Vec<_> = phrases
            .iter()
            .filter(|p| !p.is_empty() && seen.insert(p.as_str()))
            .collect();
        let mut counts = HashMap::new();
        for batch in unique.chunks(200) {
            self.control.check_io()?;
            let placeholders = vec!["?"; batch.len()].join(",");
            let mut statement = self
                .connection
                .prepare(&format!(
                    "SELECT text, count FROM ngrams WHERE text IN ({placeholders})"
                ))
                .map_err(error)?;
            let matches = statement
                .query_map(params_from_iter(batch), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
                })
                .map_err(error)?;
            for item in matches {
                let (phrase, count) = item.map_err(error)?;
                if counts.insert(phrase, count).is_some() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Duplicate phrase keys",
                    ));
                }
            }
        }
        Ok(counts)
    }
}
