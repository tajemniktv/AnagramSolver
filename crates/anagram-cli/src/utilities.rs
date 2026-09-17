//! Offline native tooling. Writes are staged next to explicit outputs and never
//! overwrite an existing model/corpus. No solver subprocess or Python dependency.
use anagram_core::{control::Control, request::Error};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashSet},
    ffi::OsString,
    fs::File,
    io::{self, BufReader, Read, Write},
    path::Path,
};
fn err(e: impl std::fmt::Display) -> Error {
    Error::new("tool_error", e.to_string())
}
pub fn is_command(name: &str) -> bool {
    matches!(
        name,
        "ranker-build" | "ranker-train" | "ranker-rank" | "prepare-dictionary" | "build-phrases"
    )
}
fn input<T: serde::de::DeserializeOwned>(
    control: &Control,
    source: &mut dyn Read,
) -> Result<T, Error> {
    let mut bytes = Vec::new();
    control
        .reader(source)
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(err)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err(err("Tool input exceeds 16 MiB"));
    }
    serde_json::from_slice(&bytes).map_err(err)
}
fn stage(path: &Path) -> Result<tempfile::NamedTempFile, Error> {
    if path.exists() {
        return Err(err(
            "Output already exists; choose a new filename to preserve existing data",
        ));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    tempfile::NamedTempFile::new_in(parent).map_err(err)
}
pub fn run(args: &[OsString], control: &Control) -> Result<Value, Error> {
    if args
        .first()
        .is_some_and(|arg| arg.to_string_lossy().starts_with("ranker-"))
    {
        let bytes = read_stdin(control, 16 * 1024 * 1024)?;
        run_with_input(args, control, &mut io::Cursor::new(bytes))
    } else {
        run_with_input(args, control, &mut io::empty())
    }
}

/// CLI-only input owner: a deadline must also work while a pipe writer stays open.
/// On timeout the CLI exits; it never joins a thread blocked in an OS stdin read.
pub fn read_stdin(control: &Control, limit: usize) -> Result<Vec<u8>, Error> {
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = io::stdin()
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map(|_| bytes);
        let _ = send.send(result);
    });
    loop {
        control
            .check()
            .map_err(|reason| Error::new(reason, reason))?;
        match receive.recv_timeout(std::time::Duration::from_millis(10)) {
            Ok(result) => {
                let bytes = result.map_err(err)?;
                if bytes.len() > limit {
                    return Err(Error::new("request_too_large", "Input exceeds byte limit"));
                }
                return Ok(bytes);
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(error) => return Err(err(error)),
        }
    }
}
pub fn run_with_input(
    args: &[OsString],
    control: &Control,
    source: &mut dyn Read,
) -> Result<Value, Error> {
    match args.first().and_then(|a| a.to_str()).unwrap_or("") {
        "ranker-build" if (2..=3).contains(&args.len()) => build_groups(args, control, source),
        "ranker-train" if args.len() == 2 => {
            #[derive(Deserialize)]
            struct Training {
                groups: Vec<anagram_core::training::Group>,
                #[serde(default)]
                options: anagram_core::training::Options,
            }
            let data: Training = input(control, source)?;
            let report = anagram_core::training::train(&data.groups, &data.options, control)?;
            let mut file = stage(Path::new(&args[1]))?;
            serde_json::to_writer_pretty(&mut file, &report.model).map_err(err)?;
            file.as_file().sync_all().map_err(err)?;
            control.check().map_err(err)?;
            file.persist_noclobber(&args[1]).map_err(err)?;
            Ok(
                json!({"baseline":report.baseline,"held_out":report.held_out,"model_path":args[1].to_string_lossy(),"note":"Saved all-group fit; only held_out metrics estimate generalization"}),
            )
        }
        "ranker-rank" if args.len() == 2 => {
            let model = anagram_core::learned::Model::load_controlled(Path::new(&args[1]), control)
                .map_err(err)?;
            let items: Vec<anagram_core::learned::Item> = input(control, source)?;
            serde_json::to_value(
                anagram_core::learned::rank_result(&items, Some(&model), control).map_err(err)?,
            )
            .map_err(err)
        }
        "prepare-dictionary" if args.len() == 4 => {
            let unigrams = anagram_core::lexicon::Unigrams::load(
                control.reader(BufReader::new(File::open(&args[2]).map_err(err)?)),
            )
            .map_err(err)?;
            let mut words = BTreeSet::new();
            for line in anagram_core::lexicon::decoded_lines(
                control.reader(BufReader::new(File::open(&args[1]).map_err(err)?)),
            ) {
                let word = anagram_core::normalize_letters(&line.map_err(err)?);
                if !word.is_empty() {
                    words.insert(word);
                }
            }
            let extra: Vec<_> = words
                .iter()
                .filter(|w| w.len() == 2 && unigrams.zipf(w) >= 5.0)
                .cloned()
                .collect();
            words.extend(
                [
                    "dont", "cant", "wont", "isnt", "arent", "wasnt", "werent", "didnt", "doesnt",
                    "couldnt", "shouldnt", "wouldnt", "hasnt", "havent", "hadnt",
                ]
                .map(str::to_owned),
            );
            let mut file = stage(Path::new(&args[3]))?;
            for word in &words {
                control.check().map_err(err)?;
                writeln!(file, "{word}").map_err(err)?;
            }
            file.as_file().sync_all().map_err(err)?;
            file.persist_noclobber(&args[3]).map_err(err)?;
            Ok(
                json!({"dictionary":args[3].to_string_lossy(),"words":words.len(),"extra_short_words":extra}),
            )
        }
        "build-phrases" if args.len() >= 3 => build_phrases(args, control),
        _ => Err(err(
            "Usage: ranker-build WORDNET [PHRASE_DB] < cases.json | ranker-train NEW_MODEL < groups.json | ranker-rank MODEL < items.json | prepare-dictionary BASE UNIGRAMS NEW_DICTIONARY | build-phrases NEW_DB TITLE_FILE[.gz] ...",
        )),
    }
}

fn build_groups(
    args: &[OsString],
    control: &Control,
    source: &mut dyn Read,
) -> Result<Value, Error> {
    use anagram_core::corpus_ranking::PhraseCorpus;
    #[derive(Deserialize)]
    struct Case {
        answer: String,
        acceptable_orders: Option<Vec<String>>,
    }
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Cases {
        List(Vec<Case>),
        Configured {
            cases: Vec<Case>,
            #[serde(default)]
            options: BuildOptions,
        },
    }
    #[derive(Deserialize, serde::Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct BuildOptions {
        retained_orders: usize,
        phrase_bonus_max: f64,
    }
    impl Default for BuildOptions {
        fn default() -> Self {
            Self {
                retained_orders: 56,
                phrase_bonus_max: 10.0,
            }
        }
    }
    let (cases, options) = match input(control, source)? {
        Cases::List(cases) => (cases, BuildOptions::default()),
        Cases::Configured { cases, options } => (cases, options),
    };
    if options.retained_orders < 2
        || !options.phrase_bonus_max.is_finite()
        || options.phrase_bonus_max < 0.0
    {
        return Err(err(
            "Training pool requires at least two retained orders and a finite nonnegative phrase bonus",
        ));
    }
    let lex = anagram_core::wordnet::WordNet::load_identified(Path::new(&args[1]), control)
        .map_err(err)?
        .0;
    let phrase = args
        .get(2)
        .map(|p| anagram_core::phrase_index::PhraseIndex::open_controlled(Path::new(p), control))
        .transpose()
        .map_err(err)?;
    let token = regex::Regex::new(r"[A-Za-z]+(?:['’][A-Za-z]+)?").unwrap();
    let normalize = |s: &str| {
        token
            .find_iter(s)
            .map(|m| anagram_core::normalize_letters(m.as_str()))
            .collect::<Vec<_>>()
    };
    let mut groups = Vec::new();
    let mut skipped = Vec::new();
    let mut bags = HashSet::new();
    for case in cases {
        control.check().map_err(err)?;
        let words = normalize(&case.answer);
        if !(2..=6).contains(&words.len()) {
            skipped.push(format!("{}: outside exact training range 2–6", case.answer));
            continue;
        }
        let mut bag = words.clone();
        bag.sort();
        let key = bag.join(" ");
        if !bags.insert(key.clone()) {
            return Err(err(
                "Duplicate word bag; combine its acceptable_orders in one case",
            ));
        }
        let acceptable: HashSet<_> = case
            .acceptable_orders
            .unwrap_or_else(|| vec![case.answer.clone()])
            .iter()
            .map(|s| normalize(s).join(" "))
            .collect();
        let (candidates, _) = anagram_core::ordering::rank_controlled(
            &words,
            &lex,
            true,
            256,
            options.retained_orders,
            control,
        )
        .map_err(err)?;
        let mut items = Vec::new();
        for c in candidates {
            let (score, details) = if let Some(p) = &phrase {
                anagram_core::phrase_evidence::score(
                    &c.order,
                    &p.counts(&anagram_core::phrase_evidence::queries(&c.order, p.max_n()))
                        .map_err(err)?,
                    p.max_n(),
                    true,
                )
            } else {
                (0.0, Default::default())
            };
            let order = c.order.join(" ");
            items.push(anagram_core::training::Sample {
                positive: acceptable.contains(&order),
                key: order,
                features: anagram_core::learned::features(&c, score, &details, words.len()),
                baseline_score: c.objective + (options.phrase_bonus_max / 100.0) * score,
            });
        }
        if !items.iter().any(|i| i.positive) || !items.iter().any(|i| !i.positive) {
            skipped.push(format!(
                "{}: retained pool needs both positive and negative orders",
                case.answer
            ));
            continue;
        }
        groups.push(anagram_core::training::Group { key, items });
    }
    Ok(json!({"groups":groups,"skipped":skipped,"build_options":options}))
}

fn build_phrases(args: &[OsString], control: &Control) -> Result<Value, Error> {
    crate::phrase_builder::build(
        Path::new(&args[1]),
        &args[2..]
            .iter()
            .map(std::path::PathBuf::from)
            .collect::<Vec<_>>(),
        crate::phrase_builder::default_workers(),
        control,
    )
}
