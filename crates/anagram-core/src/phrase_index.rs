//! Read-only SQLite phrase corpus. Each worker owns its connection.
use crate::corpus_ranking::PhraseCorpus;
use rusqlite::{Connection, OpenFlags, params_from_iter};
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
                counts.insert(phrase, count);
            }
        }
        Ok(counts)
    }
}
