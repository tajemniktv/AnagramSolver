//! Desktop maintenance uses the same native operations as the CLI, never a subprocess.
use crate::{
    jobs::Desktop,
    settings::{Corpora, Settings},
};
use anagram_core::{control::Control, solve};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    ffi::OsString,
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    time::Duration,
};

fn error(e: impl std::fmt::Display) -> String {
    e.to_string()
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    Validate {
        corpora: Corpora,
        model: String,
    },
    BuildTraining {
        cases: String,
        corpora: Corpora,
        #[serde(default)]
        options: Option<Value>,
    },
    RankModel {
        model: String,
        items: String,
    },
    Dictionary {
        base: String,
        unigrams: String,
    },
    Phrases {
        sources: Vec<String>,
    },
    Prepare {
        source: String,
        wiktionary: bool,
        wikipedia: bool,
    },
}
impl Desktop {
    pub fn manage(&self, action: Action) -> Result<(), String> {
        let data = self.data.clone();
        self.operation(3600, move |control| match action {
            Action::Validate { corpora, model } => validate(&corpora, &model, control),
            Action::BuildTraining { cases, corpora, options } => {
                let mut args = vec![OsString::from("ranker-build"), corpora.wordnet.into()];
                if !corpora.phrase.is_empty() { args.push(corpora.phrase.into()); }
                let mut bytes = Vec::new();
                control.reader(fs::File::open(cases).map_err(error)?).take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes).map_err(error)?;
                if bytes.len() > 16 * 1024 * 1024 { return Err("Cases exceed 16 MiB".into()); }
                let mut input: Value = serde_json::from_slice(&bytes).map_err(error)?;
                if let Some(options) = options { if !input.is_array() { return Err("With desktop build controls, select a JSON array of cases".into()); } input = json!({"cases":input,"options":options}); }
                let bytes = serde_json::to_vec(&input).map_err(error)?;
                let result = anagram_cli::utilities::run_with_input(&args, control, &mut bytes.as_slice()).map_err(|e| e.message)?;
                if result["groups"].as_array().is_none_or(|g| g.is_empty()) { return Err(format!("No usable training groups: {}", result["skipped"])); }
                let bytes = serde_json::to_vec_pretty(&result).map_err(error)?;
                if bytes.len() > 4 * 1024 * 1024 || result["groups"].as_array().is_some_and(|g| g.len() > 1000) {
                    return Err("Built dataset exceeds desktop training limits (4 MiB / 1,000 groups). Split cases into smaller files or use the native CLI.".into());
                }
                let directory = data.join("training");
                fs::create_dir_all(&directory).map_err(error)?;
                let mut file = tempfile::Builder::new().prefix("groups-").suffix(".json").tempfile_in(directory).map_err(error)?;
                file.write_all(&bytes).map_err(error)?;
                file.as_file().sync_all().map_err(error)?;
                control.check().map_err(error)?;
                let (_, path) = file.keep().map_err(error)?;
                Ok(json!({"dataset_path":path,"groups":result["groups"].as_array().map(Vec::len),"skipped":result["skipped"]}))
            }
            Action::RankModel {model, items} => {
                let mut input = fs::File::open(items).map_err(error)?;
                anagram_cli::utilities::run_with_input(&["ranker-rank".into(),model.into()],control,&mut input).map_err(|e| e.message)
            }
            Action::Dictionary {base,unigrams} => tool_output(&data,"dictionary.txt",vec!["prepare-dictionary".into(),base.into(),unigrams.into()],false,control),
            Action::Phrases {sources} => {
                if sources.is_empty() { return Err("Choose at least one title file".into()); }
                let mut args = vec!["build-phrases".into()]; args.extend(sources.into_iter().map(OsString::from));
                tool_output(&data,"phrases.db",args,true,control)
            }
            Action::Prepare { source, wiktionary, wikipedia } => prepare(&data, &source, wiktionary, wikipedia, control),
        })
    }
}

fn tool_output(
    data: &Path,
    filename: &str,
    mut args: Vec<OsString>,
    output_first: bool,
    control: &Control,
) -> Result<Value, String> {
    let root = data.join("prepared");
    fs::create_dir_all(&root).map_err(error)?;
    let stage = tempfile::Builder::new()
        .prefix("tool-")
        .tempdir_in(root)
        .map_err(error)?;
    let output = stage.path().join(filename).into_os_string();
    if output_first {
        args.insert(1, output);
    } else {
        args.push(output);
    }
    let result = anagram_cli::utilities::run_with_input(&args, control, &mut std::io::empty())
        .map_err(|e| e.message)?;
    control.check().map_err(error)?;
    let _ = stage.keep();
    Ok(result)
}

pub fn validate(c: &Corpora, model: &str, control: &Control) -> Result<Value, String> {
    c.validate()?;
    let mut text_stats = serde_json::Map::new();
    for (role, path, tokens) in [
        ("dictionary", &c.dictionary, 0),
        ("unigrams", &c.unigrams, 1),
        ("bigrams", &c.bigrams, 2),
    ] {
        let mut valid = 0_u64;
        let mut ignored = 0_u64;
        for line in control
            .reader(BufReader::new(fs::File::open(path).map_err(error)?))
            .lines()
        {
            let line = line.map_err(error)?;
            let fields: Vec<_> = line.split_whitespace().collect();
            let accepted = if tokens == 0 {
                fields.len() == 1
                    && fields[0]
                        .chars()
                        .all(|c| c.is_alphabetic() || "-'’".contains(c))
                    && !anagram_core::normalize_letters(fields[0]).is_empty()
            } else {
                fields.len() == tokens + 1
                    && fields[tokens].parse::<i64>().is_ok_and(|n| n > 0)
                    && fields[..tokens]
                        .iter()
                        .all(|word| !anagram_core::normalize_letters(word).is_empty())
            };
            if accepted {
                valid += 1;
            } else {
                ignored += 1;
            }
        }
        if valid == 0 {
            return Err(format!("{role}: no usable dictionary/count rows"));
        }
        text_stats.insert(
            role.into(),
            json!({"usable_rows":valid,"other_rows":ignored}),
        );
    }
    // Exercise the real loaders and their exact-byte/row provenance, without using the result cache.
    let mut request: solve::Request =
        serde_json::from_value(Settings::default().request).map_err(error)?;
    request.generation.text = "a".into();
    request.generation.candidate_budget = 1;
    request.deep_per_group = 1;
    let result = solve::solve_controlled(
        &request,
        solve::Paths {
            dictionary: Path::new(&c.dictionary),
            unigrams: Path::new(&c.unigrams),
            bigrams: Path::new(&c.bigrams),
            wordnet: Path::new(&c.wordnet),
            phrase: (!c.phrase.is_empty()).then(|| Path::new(&c.phrase)),
            model: (!model.is_empty()).then(|| Path::new(model)),
        },
        control,
    )
    .map_err(|e| e.message)?;
    if result
        .status
        .versions
        .data
        .iter()
        .any(|d| d.present && d.bytes == 0)
    {
        return Err("A selected corpus is empty".into());
    }
    Ok(
        json!({"validation":"Text sanity checks, native loaders and a one-candidate smoke solve succeeded; not a linguistic quality assessment.","text_rows":text_stats,"versions":result.status.versions,"timings":result.timings}),
    )
}

fn fetch(
    url: &str,
    local: &Path,
    name: &str,
    output: &Path,
    control: &Control,
) -> Result<(), String> {
    control.check().map_err(error)?;
    if local.as_os_str().is_empty() {
        return download(url, output, control);
    }
    let mut input = fs::File::open(local.join(name)).map_err(error)?;
    let mut file = fs::File::create_new(output).map_err(error)?;
    let mut total = 0_u64;
    let mut buffer = [0; 64 * 1024];
    loop {
        control.check().map_err(error)?;
        let n = input.read(&mut buffer).map_err(error)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        if total > 2 * 1024 * 1024 * 1024 {
            return Err("Source exceeds 2 GiB; use offline preparation for larger sources".into());
        }
        file.write_all(&buffer[..n]).map_err(error)?;
    }
    if total == 0 {
        return Err(format!("Empty download: {name}"));
    }
    file.sync_all().map_err(error)
}

fn download(url: &str, output: &Path, control: &Control) -> Result<(), String> {
    // The network future lives on the existing joined maintenance worker. Dropping
    // it cancels a stalled request; no detached download survives cancellation.
    tokio::runtime::Builder::new_current_thread().enable_all().build().map_err(error)?.block_on(async {
        let work = async {
            let client = reqwest::Client::builder().connect_timeout(Duration::from_secs(15))
                .user_agent(concat!("TajsAnagrams/", env!("CARGO_PKG_VERSION"), " (https://github.com/tajemniktv/AnagramSolver)"))
                .read_timeout(Duration::from_secs(30)).timeout(Duration::from_secs(3600)).build().map_err(error)?;
            let mut response = client.get(url).send().await.map_err(error)?.error_for_status().map_err(error)?;
            let mut file = fs::File::create_new(output).map_err(error)?;
            let mut total = 0_u64;
            while let Some(bytes) = response.chunk().await.map_err(error)? {
                control.check().map_err(error)?;
                total += bytes.len() as u64;
                if total > 2 * 1024 * 1024 * 1024 { return Err("Source exceeds 2 GiB; use offline preparation for larger sources".into()); }
                file.write_all(&bytes).map_err(error)?;
            }
            if total == 0 { return Err("Empty download".into()); }
            file.sync_all().map_err(error)
        };
        tokio::pin!(work);
        loop {
            tokio::select! {
                result = &mut work => return result,
                _ = tokio::time::sleep(Duration::from_millis(100)) => control.check().map_err(error)?,
            }
        }
    })
}

fn prepare(
    data: &Path,
    source: &str,
    wiktionary: bool,
    wikipedia: bool,
    control: &Control,
) -> Result<Value, String> {
    let sets = data.join("corpus-sets");
    fs::create_dir_all(&sets).map_err(error)?;
    let stage = tempfile::Builder::new()
        .prefix("set-")
        .tempdir_in(&sets)
        .map_err(error)?;
    let root = stage.path();
    let local = Path::new(source);
    for (url, name) in [
        ("https://phillipmfeldman.org/English/large.txt", "base.txt"),
        ("https://norvig.com/ngrams/count_1w.txt", "count_1w.txt"),
        ("https://norvig.com/ngrams/count_2w.txt", "count_2w.txt"),
        (
            "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz",
            "wordnet.tar.gz",
        ),
    ] {
        fetch(url, local, name, &root.join(name), control)?;
    }
    let dictionary = root.join("dictionary.txt");
    let policy = anagram_cli::utilities::run_with_input(
        &[
            "prepare-dictionary".into(),
            root.join("base.txt").into_os_string(),
            root.join("count_1w.txt").into_os_string(),
            dictionary.clone().into_os_string(),
        ],
        control,
        &mut std::io::empty(),
    )
    .map_err(|e| e.message)?;
    let wordnet = root.join("wordnet");
    fs::create_dir(&wordnet).map_err(error)?;
    let archive = fs::File::open(root.join("wordnet.tar.gz")).map_err(error)?;
    let mut archive = tar::Archive::new(control.reader(flate2::read::GzDecoder::new(archive)));
    let required = [
        "index.noun",
        "index.verb",
        "index.adj",
        "index.adv",
        "data.verb",
        "noun.exc",
        "verb.exc",
    ];
    let mut found = std::collections::HashSet::new();
    for entry in archive.entries().map_err(error)? {
        control.check().map_err(error)?;
        let mut entry = entry.map_err(error)?;
        let path = entry.path().map_err(error)?.into_owned();
        for name in required {
            if path == Path::new(&format!("dict/{name}"))
                || path == Path::new(&format!("./dict/{name}"))
            {
                if !entry.header().entry_type().is_file()
                    || !found.insert(name)
                    || entry.size() > 128 * 1024 * 1024
                {
                    return Err("Invalid or duplicate WordNet member".into());
                }
                let mut output = fs::File::create_new(wordnet.join(name)).map_err(error)?;
                std::io::copy(&mut control.reader(&mut entry), &mut output).map_err(error)?;
            }
        }
    }
    if found.len() != required.len() {
        return Err("WordNet archive is missing required files".into());
    }
    let mut titles = vec![];
    for (enabled, name) in [(wiktionary, "enwiktionary"), (wikipedia, "enwiki")] {
        if enabled {
            let file = root.join(format!("{name}.gz"));
            fetch(
                &format!(
                    "https://dumps.wikimedia.org/{name}/latest/{name}-latest-all-titles-in-ns0.gz"
                ),
                local,
                &format!("{name}.gz"),
                &file,
                control,
            )?;
            titles.push(file.into_os_string());
        }
    }
    let mut phrase_build = None;
    let phrase = if titles.is_empty() {
        String::new()
    } else {
        let path = root.join("phrases.db");
        let mut args = vec!["build-phrases".into(), path.clone().into_os_string()];
        args.extend(titles);
        phrase_build = Some(
            anagram_cli::utilities::run_with_input(&args, control, &mut std::io::empty())
                .map_err(|e| e.message)?,
        );
        path.to_string_lossy().into_owned()
    };
    let corpora = Corpora {
        dictionary: dictionary.to_string_lossy().into_owned(),
        unigrams: root.join("count_1w.txt").to_string_lossy().into_owned(),
        bigrams: root.join("count_2w.txt").to_string_lossy().into_owned(),
        wordnet: wordnet.to_string_lossy().into_owned(),
        phrase,
    };
    let report = validate(&corpora, "", control)?;
    let result = json!({"corpora":corpora,"validation":report,"dictionary_policy":policy,"phrase_build":phrase_build,"note":"Prepared separately. Select this set and save settings to activate; previous corpora are preserved."});
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&result).map_err(error)?,
    )
    .map_err(error)?;
    control.check().map_err(error)?;
    let _ = stage.keep();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scratch() -> tempfile::TempDir {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.codex/temp");
        fs::create_dir_all(&root).unwrap();
        tempfile::tempdir_in(root).unwrap()
    }
    fn sources(path: &Path) {
        fs::create_dir_all(path).unwrap();
        for (name, text) in [
            ("base.txt", "a\nate\ntea\n"),
            ("count_1w.txt", "a\t100\nate\t50\n"),
            ("count_2w.txt", "a tea\t10\n"),
        ] {
            fs::write(path.join(name), text).unwrap();
        }
        let gzip = flate2::write::GzEncoder::new(
            fs::File::create(path.join("wordnet.tar.gz")).unwrap(),
            flate2::Compression::default(),
        );
        let mut archive = tar::Builder::new(gzip);
        for name in [
            "index.noun",
            "index.verb",
            "index.adj",
            "index.adv",
            "data.verb",
            "noun.exc",
            "verb.exc",
        ] {
            let content = b"  fixture header\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            archive
                .append_data(&mut header, format!("dict/{name}"), &content[..])
                .unwrap();
        }
        archive.into_inner().unwrap().finish().unwrap();
        let mut gzip = flate2::write::GzEncoder::new(
            fs::File::create(path.join("enwiktionary.gz")).unwrap(),
            flate2::Compression::default(),
        );
        gzip.write_all(b"we_are_home\n").unwrap();
        gzip.finish().unwrap();
    }
    #[test]
    fn staged_corpora_validate_preserve_previous_and_remove_failed_sets() {
        let temp = scratch();
        let input = temp.path().join("source");
        sources(&input);
        let control = Control::default();
        let first = prepare(temp.path(), input.to_str().unwrap(), true, false, &control).unwrap();
        let c: Corpora = serde_json::from_value(first["corpora"].clone()).unwrap();
        assert!(Path::new(&c.phrase).is_file());
        let before = fs::read(&c.dictionary).unwrap();
        let second = prepare(temp.path(), input.to_str().unwrap(), false, false, &control).unwrap();
        assert_ne!(
            first["corpora"]["dictionary"],
            second["corpora"]["dictionary"]
        );
        assert_eq!(fs::read(&c.dictionary).unwrap(), before);
        fs::write(input.join("wordnet.tar.gz"), b"broken archive").unwrap();
        assert!(prepare(temp.path(), input.to_str().unwrap(), false, false, &control).is_err());
        assert_eq!(
            fs::read_dir(temp.path().join("corpus-sets"))
                .unwrap()
                .count(),
            2
        );
        fs::write(&c.dictionary, "").unwrap();
        assert!(validate(&c, "", &control).is_err());
    }
    #[test]
    fn http_fetch_checks_status_and_cancel_before_network() {
        let temp = scratch();
        for (status, ok) in [("200 OK", true), ("404 Not Found", false)] {
            let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = server.local_addr().unwrap();
            let worker = std::thread::spawn(move || {
                let (mut stream, _) = server.accept().unwrap();
                let mut buffer = [0; 4096];
                let mut request = Vec::new();
                while !request.ends_with(b"\r\n\r\n") {
                    let n = stream.read(&mut buffer).unwrap();
                    assert!(n > 0);
                    request.extend_from_slice(&buffer[..n]);
                    assert!(request.len() <= 16384);
                }
                let request = String::from_utf8(request).unwrap().to_ascii_lowercase();
                assert!(request.contains(concat!(
                    "user-agent: tajsanagrams/",
                    env!("CARGO_PKG_VERSION"),
                    " (https://github.com/tajemniktv/anagramsolver)"
                )));
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Length: 5\r\nConnection: close\r\n\r\nhello"
                )
                .unwrap();
            });
            let path = temp.path().join(if ok { "success" } else { "failure" });
            assert_eq!(
                fetch(
                    &format!("http://{address}"),
                    Path::new(""),
                    "fixture",
                    &path,
                    &Control::default()
                )
                .is_ok(),
                ok
            );
            worker.join().unwrap();
            if ok {
                assert_eq!(fs::read(path).unwrap(), b"hello");
            }
        }
        let cancelled = Control::default();
        cancelled.cancel();
        assert!(
            fetch(
                "http://127.0.0.1:1",
                Path::new(""),
                "cancelled",
                &temp.path().join("cancelled"),
                &cancelled
            )
            .is_err()
        );
        assert!(!temp.path().join("cancelled").exists());
    }
    #[test]
    fn cancellation_interrupts_a_stalled_http_response() {
        let temp = scratch();
        let server = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = server.local_addr().unwrap();
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = server.accept().unwrap();
            let mut buffer = [0; 4096];
            assert!(stream.read(&mut buffer).unwrap() > 0);
            entered_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        });
        let control = Control::default();
        let cancel = control.clone();
        let canceller = std::thread::spawn(move || {
            entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            cancel.cancel();
        });
        let started = std::time::Instant::now();
        let result = fetch(
            &format!("http://{address}"),
            Path::new(""),
            "stalled",
            &temp.path().join("stalled"),
            &control,
        );
        release_tx.send(()).unwrap();
        server.join().unwrap();
        canceller.join().unwrap();
        assert!(result.unwrap_err().contains("cancelled"));
        assert!(started.elapsed() < Duration::from_secs(3));
    }
    #[test]
    fn desktop_builds_reusable_training_dataset_without_changing_settings() {
        let temp = scratch();
        let input = temp.path().join("source");
        sources(&input);
        let report = prepare(
            temp.path(),
            input.to_str().unwrap(),
            false,
            false,
            &Control::default(),
        )
        .unwrap();
        let corpora: Corpora = serde_json::from_value(report["corpora"].clone()).unwrap();
        let cases = temp.path().join("cases.json");
        fs::write(&cases,r#"[{"answer":"we are home"},{"answer":"you are here"},{"answer":"they can go"},{"answer":"i will stay"},{"answer":"he is happy"},{"answer":"she was ready"}]"#).unwrap();
        let app = Desktop::new(temp.path().join("app-data")).unwrap();
        app.manage(Action::BuildTraining {
            cases: cases.to_string_lossy().into_owned(),
            corpora,
            options: None,
        })
        .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while app.inner.lock().unwrap().experiment.active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        let inner = app.inner.lock().unwrap();
        assert!(
            inner.experiment.error.is_none(),
            "{:?}",
            inner.experiment.error
        );
        let report = inner.experiment.report.as_ref().unwrap();
        assert_eq!(report["groups"], 6);
        let dataset = report["dataset_path"].as_str().unwrap().to_owned();
        assert!(Path::new(&dataset).starts_with(app.data.join("training")));
        assert!(inner.settings.model.is_empty());
        drop(inner);
        let items = temp.path().join("items.json");
        let groups: Value = serde_json::from_slice(&fs::read(&dataset).unwrap()).unwrap();
        fs::write(
            &items,
            serde_json::to_vec(&groups["groups"][0]["items"]).unwrap(),
        )
        .unwrap();
        app.train_model(dataset, 5, 5).unwrap();
        while app.inner.lock().unwrap().experiment.active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(app.inner.lock().unwrap().experiment.error.is_none());
        let model = app
            .inner
            .lock()
            .unwrap()
            .experiment
            .report
            .as_ref()
            .unwrap()["model_path"]
            .as_str()
            .unwrap()
            .to_owned();
        app.manage(Action::RankModel {
            model,
            items: items.to_string_lossy().into_owned(),
        })
        .unwrap();
        while app.inner.lock().unwrap().experiment.active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            app.inner
                .lock()
                .unwrap()
                .experiment
                .report
                .as_ref()
                .unwrap()["indices"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        app.manage(Action::Dictionary {
            base: input.join("base.txt").to_string_lossy().into_owned(),
            unigrams: input.join("count_1w.txt").to_string_lossy().into_owned(),
        })
        .unwrap();
        while app.inner.lock().unwrap().experiment.active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            Path::new(
                app.inner
                    .lock()
                    .unwrap()
                    .experiment
                    .report
                    .as_ref()
                    .unwrap()["dictionary"]
                    .as_str()
                    .unwrap()
            )
            .is_file()
        );
        app.manage(Action::Phrases {
            sources: vec![input.join("enwiktionary.gz").to_string_lossy().into_owned()],
        })
        .unwrap();
        while app.inner.lock().unwrap().experiment.active {
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(
            Path::new(
                app.inner
                    .lock()
                    .unwrap()
                    .experiment
                    .report
                    .as_ref()
                    .unwrap()["database"]
                    .as_str()
                    .unwrap()
            )
            .is_file()
        );
        app.shutdown();
    }
}
