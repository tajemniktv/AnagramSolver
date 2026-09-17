//! Bounded parallel title parsing with one SQLite writer and atomic publication.
use anagram_core::{control::Control, request::Error};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
};

fn err(e: impl std::fmt::Display) -> Error {
    Error::new("tool_error", e.to_string())
}
pub fn default_workers() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get().saturating_sub(1).clamp(1, 8))
}
type Counts = HashMap<String, i64>;
fn parse(
    lines: &[Vec<u8>],
    token: &regex::Regex,
    control: &Control,
) -> Result<(u64, Counts), Error> {
    let mut counts = Counts::new();
    let mut accepted = 0;
    for line in lines {
        control.check().map_err(err)?;
        let text = String::from_utf8_lossy(line).replace('_', " ");
        // A ninth token already makes this title ineligible. Avoid tokenizing the rest.
        let words: Vec<_> = token
            .find_iter(&text)
            .take(9)
            .map(|m| anagram_core::normalize_letters(m.as_str()))
            .collect();
        if !(2..=8).contains(&words.len()) || text.trim() == "page title" {
            continue;
        }
        accepted += 1;
        *counts.entry(words.join(" ")).or_default() += 1;
        for n in 2..=5.min(words.len().saturating_sub(1)) {
            for part in words.windows(n) {
                *counts.entry(part.join(" ")).or_default() += 1;
            }
        }
    }
    Ok((accepted, counts))
}

pub fn build(
    output: &Path,
    sources: &[PathBuf],
    workers: usize,
    control: &Control,
) -> Result<Value, Error> {
    if !(1..=8).contains(&workers) {
        return Err(err("Phrase workers must be between 1 and 8"));
    }
    control.check().map_err(err)?;
    if output.exists() {
        return Err(err(
            "Output already exists; choose a new filename to preserve existing data",
        ));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let file = tempfile::NamedTempFile::new_in(parent).map_err(err)?;
    let mut db = rusqlite::Connection::open(file.path()).map_err(err)?;
    // Keep SQLite journaling/durability defaults. A larger bounded page cache and
    // a single primary-key B-tree avoid the old table + duplicate text index I/O.
    db.execute_batch("PRAGMA cache_size=-65536; CREATE TABLE ngrams(text TEXT PRIMARY KEY,n INTEGER NOT NULL,count INTEGER NOT NULL) WITHOUT ROWID;").map_err(err)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|i| format!("phrase-parser-{i}"))
        .build()
        .map_err(err)?;
    let token = regex::Regex::new(r"[A-Za-z]+(?:['’][A-Za-z]+)?").unwrap();
    let transaction = db.transaction().map_err(err)?;
    let mut accepted = 0_u64;
    let mut updates = 0_u64;
    {
        let mut insert = transaction.prepare("INSERT INTO ngrams(text,n,count) VALUES(?1,?2,?3) ON CONFLICT(text) DO UPDATE SET count=count+excluded.count").map_err(err)?;
        for source in sources {
            let input = File::open(source).map_err(err)?;
            let reader: Box<dyn Read> = if source.extension().is_some_and(|e| e == "gz") {
                Box::new(flate2::read::MultiGzDecoder::new(input))
            } else {
                Box::new(input)
            };
            let mut reader = control.reader(BufReader::with_capacity(256 * 1024, reader));
            loop {
                // Bound both line count and decompressed bytes; never retain a whole dump.
                let mut lines = Vec::with_capacity(8192);
                let mut bytes = 0;
                while lines.len() < 8192 && bytes < 4 * 1024 * 1024 {
                    let mut line = Vec::new();
                    let n = reader
                        .by_ref()
                        .take(65537)
                        .read_until(b'\n', &mut line)
                        .map_err(err)?;
                    if n == 0 {
                        break;
                    }
                    if n > 65536 {
                        return Err(err("Title exceeds 64 KiB"));
                    }
                    bytes += n;
                    lines.push(line);
                }
                if lines.is_empty() {
                    break;
                }
                let parts: Vec<Result<(u64, Counts), Error>> = pool.install(|| {
                    lines
                        .par_chunks(256)
                        .map_init(
                            || token.clone(),
                            |regex, chunk| parse(chunk, regex, control),
                        )
                        .collect()
                });
                // install joins all work before any error is propagated or staging removed.
                let mut merged = Counts::new();
                for part in parts {
                    let (titles, counts) = part?;
                    accepted += titles;
                    for (text, count) in counts {
                        *merged.entry(text).or_default() += count;
                    }
                }
                let mut rows: Vec<_> = merged.into_iter().collect();
                rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
                for (text, count) in rows {
                    control.check().map_err(err)?;
                    insert
                        .execute(rusqlite::params![
                            text,
                            text.split(' ').count() as i64,
                            count
                        ])
                        .map_err(err)?;
                    updates += 1;
                }
            }
        }
    }
    control.check().map_err(err)?;
    transaction.commit().map_err(err)?;
    let rows: i64 = db
        .query_row("SELECT COUNT(*) FROM ngrams", [], |r| r.get(0))
        .map_err(err)?;
    drop(db);
    file.as_file().sync_all().map_err(err)?;
    control.check().map_err(err)?;
    file.persist_noclobber(output).map_err(err)?;
    Ok(
        json!({"database":output,"accepted_titles":accepted,"rows":rows,"workers":workers,"database_updates":updates}),
    )
}
