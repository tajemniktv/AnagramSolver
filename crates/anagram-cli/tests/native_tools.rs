use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};
fn scratch() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.codex/temp");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("native-tools-")
        .tempdir_in(root)
        .unwrap()
}

#[test]
fn dictionary_preparation_uses_lossy_corpus_decoding() {
    let temp = scratch();
    let base = temp.path().join("base.txt");
    let one = temp.path().join("one.txt");
    let output = temp.path().join("dictionary.txt");
    fs::write(&base, b"ca\xfft\ndog\n").unwrap();
    fs::write(&one, b"cat 5\ndog 5\n").unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_anagram-cli"))
        .arg("prepare-dictionary")
        .args([&base, &one, &output])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stdout)
    );
    assert!(
        fs::read_to_string(output)
            .unwrap()
            .lines()
            .any(|line| line == "cat")
    );
}
fn run(args: &[&str], input: &serde_json::Value) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_anagram-cli"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.to_string().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn parallel_phrase_batches_preserve_counts_and_fail_without_publication() {
    use anagram_cli::phrase_builder::build;
    use anagram_core::control::Control;
    let temp = scratch();
    let plain = temp.path().join("titles.txt");
    fs::write(
        &plain,
        "page_title\nwe_we_are_home\ndon't_stop_now\none\na_b_c_d_e_f_g_h_i\n".repeat(4000),
    )
    .unwrap();
    let compressed = temp.path().join("more.gz");
    let mut gzip = flate2::write::GzEncoder::new(
        fs::File::create(&compressed).unwrap(),
        flate2::Compression::default(),
    );
    gzip.write_all(b"we_we_are_home\n").unwrap();
    gzip.finish().unwrap();
    let sources = vec![plain.clone(), compressed];
    let a = temp.path().join("serial.db");
    let b = temp.path().join("parallel.db");
    let serial = build(&a, &sources, 1, &Control::default()).unwrap();
    let parallel = build(&b, &sources, 4, &Control::default()).unwrap();
    assert_eq!(serial["accepted_titles"], 8001);
    assert_eq!(serial["rows"], parallel["rows"]);
    let rows = |path| {
        let db = rusqlite::Connection::open(path).unwrap();
        let mut query = db
            .prepare("SELECT text,n,count FROM ngrams ORDER BY text")
            .unwrap();
        query
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    let expected = rows(&a);
    assert_eq!(expected, rows(&b));
    assert!(expected.contains(&("we are".into(), 2, 4001)));
    assert!(expected.contains(&("we we are home".into(), 4, 4001)));
    assert!(parallel["database_updates"].as_u64().unwrap() < 8001);
    let before = fs::read(&b).unwrap();
    assert!(build(&b, &sources, 4, &Control::default()).is_err());
    assert_eq!(before, fs::read(&b).unwrap());
    // A late invalid line fails after valid batches were already written to staging.
    use std::fs::OpenOptions;
    OpenOptions::new()
        .append(true)
        .open(&plain)
        .unwrap()
        .write_all(&vec![b'a'; 65537])
        .unwrap();
    let failed = temp.path().join("failed.db");
    assert!(build(&failed, &sources, 4, &Control::default()).is_err());
    assert!(!failed.exists());
    let cancelled = Control::default();
    cancelled.cancel();
    assert!(build(&failed, &sources, 4, &cancelled).is_err());
    assert!(!failed.exists());
}

#[test]
fn phrase_and_dictionary_builds_are_readable_and_never_clobber() {
    let temp = scratch();
    let titles = temp.path().join("titles.gz");
    let mut gzip = flate2::write::GzEncoder::new(
        fs::File::create(&titles).unwrap(),
        flate2::Compression::default(),
    );
    gzip.write_all(b"page_title\nwe_are_home\nwe_are_home\ndont_stop\n")
        .unwrap();
    gzip.finish().unwrap();
    let output = temp.path().join("phrases.db");
    let args = [
        "build-phrases",
        output.to_str().unwrap(),
        titles.to_str().unwrap(),
    ];
    let result = run(&args, &serde_json::Value::Null);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    use anagram_core::corpus_ranking::PhraseCorpus;
    let index = anagram_core::phrase_index::PhraseIndex::open(&output).unwrap();
    let counts = index
        .counts(&["we are home".into(), "we are".into()])
        .unwrap();
    assert_eq!(counts["we are home"], 2);
    assert_eq!(counts["we are"], 2);
    index.identity().unwrap();
    drop(index);
    let before = fs::read(&output).unwrap();
    assert!(!run(&args, &serde_json::Value::Null).status.success());
    assert_eq!(before, fs::read(&output).unwrap());
    let base = temp.path().join("base");
    let one = temp.path().join("one");
    let dictionary = temp.path().join("dictionary");
    fs::write(&base, "we\nhome\n").unwrap();
    fs::write(&one, "we\t1000\nhome\t100\n").unwrap();
    let result = run(
        &[
            "prepare-dictionary",
            base.to_str().unwrap(),
            one.to_str().unwrap(),
            dictionary.to_str().unwrap(),
        ],
        &serde_json::Value::Null,
    );
    assert!(result.status.success());
    let metadata: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(metadata["dictionary"].as_str(), dictionary.to_str());
    let words = fs::read_to_string(dictionary).unwrap();
    assert!(words.lines().any(|s| s == "dont"));
    assert!(words.lines().any(|s| s == "home"));
}
#[test]
fn configurable_training_pool_applies_phrase_weight_and_validates_options() {
    let temp = scratch();
    for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
        fs::write(temp.path().join(name), "").unwrap();
    }
    let phrase = temp.path().join("phrases.db");
    rusqlite::Connection::open(&phrase).unwrap().execute_batch("CREATE TABLE ngrams(text TEXT PRIMARY KEY,n INTEGER,count INTEGER); INSERT INTO ngrams VALUES('we are home',3,100);").unwrap();
    let build = |bonus, retained| {
        let input = serde_json::json!({"cases":[{"answer":"we are home"}],"options":{"retained_orders":retained,"phrase_bonus_max":bonus}});
        run(
            &[
                "ranker-build",
                temp.path().to_str().unwrap(),
                phrase.to_str().unwrap(),
            ],
            &input,
        )
    };
    let a = build(0.0, 6);
    let b = build(10.0, 6);
    assert!(a.status.success() && b.status.success());
    let a: serde_json::Value = serde_json::from_slice(&a.stdout).unwrap();
    let b: serde_json::Value = serde_json::from_slice(&b.stdout).unwrap();
    let score = |v: &serde_json::Value| {
        v["groups"][0]["items"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["key"] == "we are home")
            .unwrap()["baseline_score"]
            .as_f64()
            .unwrap()
    };
    assert!(score(&b) > score(&a));
    assert_eq!(b["build_options"]["retained_orders"], 6);
    assert_eq!(b["groups"][0]["items"].as_array().unwrap().len(), 6);
    assert!(!build(10.0, 1).status.success());
}

#[test]
fn native_training_build_fit_and_rank_round_trip() {
    let temp = scratch();
    for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
        fs::write(temp.path().join(name), "").unwrap();
    }
    let cases = serde_json::json!([{"answer":"we are home"},{"answer":"you are here"},{"answer":"they can go"},{"answer":"i will stay"},{"answer":"he is happy"},{"answer":"she was ready"}]);
    let result = run(&["ranker-build", temp.path().to_str().unwrap()], &cases);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let groups: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(groups["groups"].as_array().unwrap().len(), 6);
    let model = temp.path().join("model.json");
    let fit = run(&["ranker-train", model.to_str().unwrap()], &groups);
    assert!(
        fit.status.success(),
        "{}",
        String::from_utf8_lossy(&fit.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&fit.stdout).unwrap();
    assert_eq!(report["held_out"]["groups"], 6);
    assert_eq!(report["model_path"].as_str(), model.to_str());
    let items = &groups["groups"][0]["items"];
    let ranked = run(&["ranker-rank", model.to_str().unwrap()], items);
    assert!(ranked.status.success());
    let ranked: serde_json::Value = serde_json::from_slice(&ranked.stdout).unwrap();
    assert_eq!(
        ranked["indices"].as_array().unwrap().len(),
        items.as_array().unwrap().len()
    );
    assert!(ranked["model_identity"]["sha256"].as_str().unwrap().len() == 64);
    assert!(
        !run(&["ranker-train", model.to_str().unwrap()], &groups)
            .status
            .success()
    );
}
