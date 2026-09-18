use crate::{
    jobs::Desktop,
    settings::{self, Settings},
};
use std::{
    fs,
    path::PathBuf,
    thread,
    time::{Duration, Instant},
};

fn scratch() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.codex/temp");
    fs::create_dir_all(&root).unwrap();
    tempfile::Builder::new()
        .prefix("desktop-test-")
        .tempdir_in(root)
        .unwrap()
}

#[test]
fn experimental_training_publishes_small_model_without_enabling_it_and_joins_on_exit() {
    let temp = scratch();
    let data = temp.path().join("groups.json");
    let groups: Vec<_> = (0..12).map(|n| {
        let a = format!("a{}", char::from(b'a'+n)); let b = format!("b{}", char::from(b'a'+n));
        serde_json::json!({"key":format!("{a} {b}"),"items":[
            {"key":format!("{a} {b}"),"features":vec![1.0;18],"baseline_score":0.0,"positive":true},
            {"key":format!("{b} {a}"),"features":vec![0.0;18],"baseline_score":1.0,"positive":false}
        ]})
    }).collect();
    fs::write(
        &data,
        serde_json::to_vec(&serde_json::json!({"groups":groups})).unwrap(),
    )
    .unwrap();
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    assert!(
        app.train_model(data.to_string_lossy().into_owned(), 201, 5)
            .is_err()
    );
    app.train_model_options(
        data.to_string_lossy().into_owned(),
        anagram_core::training::Options {
            epochs: 5,
            folds: 5,
            learning_rate: 0.2,
            l2: 0.01,
        },
    )
    .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().experiment.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let inner = app.inner.lock().unwrap();
    assert!(
        inner.experiment.error.is_none(),
        "{:?}",
        inner.experiment.error
    );
    let report = inner.experiment.report.as_ref().unwrap();
    let model = PathBuf::from(report["model_path"].as_str().unwrap());
    assert!(model.starts_with(app.data.join("models")));
    assert!(fs::metadata(&model).unwrap().len() < 65536);
    assert!(inner.settings.model.is_empty());
    assert_eq!(report["held_out"]["groups"], 12);
    assert_eq!(report["options"]["learning_rate"], 0.2);
    assert_eq!(
        crate::read_report_file(report["report_path"].as_str().unwrap().into(), &app.data).unwrap(),
        *report
    );
    anagram_core::learned::Model::load(&model).unwrap();
    drop(inner);
    // A cancelled control before the worker can read must never publish a model.
    let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
    app.inner.lock().unwrap().worker_gate = Some(gate.clone());
    app.train_model(data.to_string_lossy().into_owned(), 200, 10)
        .unwrap();
    {
        let inner = app.inner.lock().unwrap();
        inner.control.cancel();
    }
    gate.wait();
    app.shutdown();
    let inner = app.inner.lock().unwrap();
    assert!(!inner.experiment.active);
    assert!(inner.experiment.report.is_none());
    assert!(inner.experiment.error.is_some());
    assert_eq!(fs::read_dir(app.data.join("models")).unwrap().count(), 2);
}

#[test]
fn invalid_optimizer_limits_are_rejected_before_starting_work() {
    let temp = scratch();
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    for (learning_rate, l2) in [
        (0.0, 0.0),
        (1.01, 0.0),
        (f64::NAN, 0.0),
        (0.08, -1.0),
        (0.08, 1.01),
        (0.08, f64::INFINITY),
    ] {
        let error = app
            .train_model_options(
                "does-not-exist.json".into(),
                anagram_core::training::Options {
                    learning_rate,
                    l2,
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(error.contains("learning rate"));
        let inner = app.inner.lock().unwrap();
        assert!(!inner.experiment.active);
        assert!(inner.worker.is_none());
        assert!(!app.data.join("models").exists());
    }
}
#[test]
fn installed_corpora_defaults_and_legacy_migration_preserve_custom_paths() {
    let temp = scratch();
    let corpora = temp.path().join("installation/corpora");
    let load = || settings::load_with_corpora(temp.path(), &corpora);
    let (mut settings, notice) = load();
    assert!(notice.is_none());
    let managed = corpora.join("dictionary/normal_user_v2.txt");
    assert_eq!(PathBuf::from(&settings.corpora.dictionary), managed);
    let legacy = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .join(".anagram_data/dictionary/normal_user_v2.txt");
    settings.corpora.dictionary = legacy.to_string_lossy().into_owned();
    settings.corpora.unigrams = "user-selected-unigrams.txt".into();
    settings.theme = "light".into();
    settings::save(temp.path(), &settings).unwrap();
    // No migration until a provisioned destination actually exists.
    assert_eq!(load().0.corpora.dictionary, settings.corpora.dictionary);
    fs::create_dir_all(managed.parent().unwrap()).unwrap();
    fs::write(&managed, "ate\n").unwrap();
    let (loaded, notice) = load();
    assert!(notice.is_none());
    assert_eq!(PathBuf::from(&loaded.corpora.dictionary), managed);
    assert_eq!(loaded.corpora.unigrams, "user-selected-unigrams.txt");
    assert_eq!(loaded.theme, "light");
    let saved: Settings =
        serde_json::from_slice(&fs::read(temp.path().join("settings.json")).unwrap()).unwrap();
    assert_eq!(saved.corpora.dictionary, loaded.corpora.dictionary);
    settings.corpora.dictionary = temp
        .path()
        .join("corpora/dictionary/normal_user_v2.txt")
        .to_string_lossy()
        .into_owned();
    settings::save(temp.path(), &settings).unwrap();
    assert_eq!(PathBuf::from(load().0.corpora.dictionary), managed);
}
#[test]
fn settings_replace_atomically_and_corruption_remains_recoverable() {
    let temp = scratch();
    let mut settings = Settings::default();
    settings::save(temp.path(), &settings).unwrap();
    settings.theme = "light".into();
    settings.corpora.dictionary = "chosen by user".into();
    settings::save(temp.path(), &settings).unwrap();
    let (loaded, warning) = settings::load(temp.path());
    assert!(warning.is_none());
    assert_eq!(loaded.theme, "light");
    assert_eq!(loaded.corpora.dictionary, "chosen by user");
    fs::write(temp.path().join("settings.json"), b"corrupt").unwrap();
    assert!(settings::load(temp.path()).1.is_some());
    assert_eq!(
        fs::read(temp.path().join("settings.json")).unwrap(),
        b"corrupt"
    );
    let oversized = vec![b' '; 2 * 1024 * 1024];
    fs::write(temp.path().join("settings.json"), &oversized).unwrap();
    assert!(settings::load(temp.path()).1.is_some());
    assert_eq!(
        fs::metadata(temp.path().join("settings.json"))
            .unwrap()
            .len(),
        oversized.len() as u64
    );
}
#[test]
fn generation_only_needs_no_ranking_corpora_and_exports_all_bags() {
    let temp = scratch();
    let dictionary = temp.path().join("words.txt");
    fs::write(&dictionary, "ate\neat\ntea\n").unwrap();
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    let mut inner = app.inner.lock().unwrap();
    inner.settings.corpora.dictionary = dictionary.to_string_lossy().into_owned();
    inner.settings.corpora.unigrams = temp
        .path()
        .join("missing-unigrams")
        .to_string_lossy()
        .into_owned();
    let mut request = inner.settings.request.clone();
    request["generation"]["text"] = "ate".into();
    request["generation"]["min_zipf"] = 0.into();
    request["generation"]["max_words"] = 11.into();
    drop(inner);
    let mut invalid = request.clone();
    invalid["result_limit_per_group"] = 0.into();
    assert!(app.start_with_mode(invalid, false, false, true).is_err());
    let id = app.start_with_mode(request, false, false, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let inner = app.inner.lock().unwrap();
    assert!(inner.job.error.is_none(), "{:?}", inner.job.error);
    assert_eq!(
        inner.job.result.as_ref().unwrap()["bags"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let status = inner.job.status.as_ref().unwrap();
    status.validate().unwrap();
    assert_eq!(status.counts.deep_analyzed, 0);
    assert!(!status.cache.hit);
    drop(inner);
    assert_eq!(crate::result_text(&app, id).unwrap(), "ate\r\neat\r\ntea");
    let mut inner = app.inner.lock().unwrap();
    inner.settings.runtime.custom_limits = true;
    inner.settings.runtime.limits.timeout_ms = Some(0);
    let request = inner.settings.request.clone();
    drop(inner);
    app.start_with_mode(request, false, false, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        app.inner.lock().unwrap().job.status.as_ref().unwrap().state,
        anagram_core::contracts::JobState::TimedOut
    );
    assert!(app.inner.lock().unwrap().job.result.is_none());
    app.shutdown();
}

#[test]
fn renamed_install_relocates_saved_corpus_model_cache_and_report_paths() {
    let temp = scratch();
    let install = temp.path().join("TajsAnagrams");
    let data = install.join("user-data");
    fs::create_dir_all(&data).unwrap();
    let actual = data.join("model.json");
    fs::write(&actual, "{}").unwrap();
    let old = temp
        .path()
        .join("AnagramSolver/user-data/model.json")
        .to_string_lossy()
        .into_owned();
    let mut migrated = old.clone();
    assert!(settings::relocate_path(&mut migrated, &data));
    assert_eq!(PathBuf::from(migrated), actual);
    let report = data.join("report.json");
    fs::write(
        &report,
        serde_json::json!({"model_path":old,"corpora":{"phrase":old}}).to_string(),
    )
    .unwrap();
    let value = crate::read_report_file(report.to_string_lossy().into_owned(), &data).unwrap();
    assert_eq!(PathBuf::from(value["model_path"].as_str().unwrap()), actual);
    assert_eq!(value["model_path"], value["corpora"]["phrase"]);
    let mut outside = temp
        .path()
        .join("custom/model.json")
        .to_string_lossy()
        .into_owned();
    assert!(!settings::relocate_path(&mut outside, &data));
}

#[test]
fn desktop_worker_completes_real_core_work_and_retains_no_queue() {
    let temp = scratch();
    for (name, text) in [
        ("dictionary", "ate\neat\ntea\n"),
        ("one", "ate\t100\neat\t100\ntea\t100\n"),
        ("two", ""),
        ("index.noun", ""),
        ("index.verb", ""),
        ("index.adj", ""),
        ("index.adv", ""),
    ] {
        fs::write(temp.path().join(name), text).unwrap();
    }
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    let mut inner = app.inner.lock().unwrap();
    let p = |name| temp.path().join(name).to_string_lossy().into_owned();
    inner.settings.corpora = settings::Corpora {
        dictionary: p("dictionary"),
        unigrams: p("one"),
        bigrams: p("two"),
        wordnet: p(""),
        phrase: String::new(),
    };
    inner.settings.runtime.cache_path = p("custom-cache.sqlite");
    inner.settings.runtime.cache_entries = 2;
    let mut request = inner.settings.request.clone();
    request["generation"]["text"] = "ate".into();
    request["generation"]["min_zipf"] = 0.into();
    drop(inner);
    let id = app.start(request.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let inner = app.inner.lock().unwrap();
    assert_eq!(inner.job.id, id);
    assert!(inner.job.error.is_none());
    assert_eq!(inner.job.result.as_ref().unwrap()["generated"], 3);
    assert!(temp.path().join("custom-cache.sqlite").is_file());
    drop(inner);
    assert_eq!(
        crate::result_text(&app, id + 1).unwrap_err(),
        "That result has expired."
    );
    let text = crate::result_text(&app, id).unwrap();
    let mut phrases: Vec<_> = text.lines().collect();
    phrases.sort_unstable();
    assert_eq!(phrases, ["ate", "eat", "tea"]);
    let restarted = Desktop::new(temp.path().join("settings")).unwrap();
    assert_eq!(
        restarted.inner.lock().unwrap().settings.request["generation"]["text"],
        "ate"
    );
    let mut invalid = request.clone();
    invalid["result_limit_per_group"] = 101.into();
    assert!(app.start(invalid).unwrap_err().contains("100 rows"));
    // Extended mode is explicit and not silently enabled by candidate_budget=0.
    let mut exhaustive = request.clone();
    exhaustive["result_limit_per_group"] = 101.into();
    exhaustive["generation"]["candidate_budget"] = 0.into();
    assert!(app.start(exhaustive.clone()).is_err());
    app.start_with_options(exhaustive, true, true).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(
        app.inner.lock().unwrap().job.result.as_ref().unwrap()["status"]["cache"]["rebuilt"]
            .as_bool()
            .unwrap()
    );
    let settings_before = fs::read(app.data.join("settings.json")).unwrap();
    app.clear_cache().unwrap();
    assert_eq!(
        settings_before,
        fs::read(app.data.join("settings.json")).unwrap()
    );
    app.start(request.clone()).unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        app.inner.lock().unwrap().job.result.as_ref().unwrap()["status"]["cache"]["hit"],
        false
    );
    app.inner.lock().unwrap().settings.corpora.dictionary = p("missing");
    assert!(
        app.start(request.clone())
            .unwrap_err()
            .contains("Choose its path in Settings")
    );
    app.shutdown();
    assert!(app.start(request).is_err());
}

#[test]
fn presentation_preserves_letters_without_guessing_ambiguous_contractions() {
    let words: Vec<_> = ["dont", "cant", "wont", "well", "ill"]
        .map(str::to_owned)
        .into();
    let rendered = anagram_core::format_phrase(&words);
    assert_eq!(rendered, "don't can't won't well ill");
    assert_eq!(anagram_core::normalize_letters(&rendered), words.join(""));
}

#[test]
fn panic_after_progress_publishes_failed_status_and_allows_retry() {
    let temp = scratch();
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    let request = {
        let mut inner = app.inner.lock().unwrap();
        inner.panic_after_progress = true;
        let mut request = inner.settings.request.clone();
        request["generation"]["text"] = "ate".into();
        request
    };
    app.start_with_mode(request.clone(), false, false, true)
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while app.inner.lock().unwrap().job.active {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    let inner = app.inner.lock().unwrap();
    let status = inner.job.status.as_ref().unwrap();
    status.validate().unwrap();
    assert_eq!(status.state, anagram_core::contracts::JobState::Failed);
    assert_eq!(status.error.as_ref().unwrap().code, "worker_panic");
    assert!(inner.job.result.is_none());
    drop(inner);
    app.start_with_mode(request, false, false, true).unwrap();
    app.shutdown();
}

#[test]
fn text_export_sorts_word_counts_numerically() {
    let temp = scratch();
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    app.inner.lock().unwrap().job.result = Some(serde_json::json!({"buckets": {
        "10": [{"best_order": ["ten"]}], "2": [{"best_order": ["two"]}], "1": [{"best_order": ["one"]}]
    }}));
    assert_eq!(crate::result_text(&app, 0).unwrap(), "one\r\ntwo\r\nten");
}

#[test]
fn shutdown_cancels_and_joins_an_active_worker() {
    let temp = scratch();
    // Gate worker admission deterministically rather than relying on slow I/O.
    fs::write(temp.path().join("dictionary"), "ate\neat\ntea\n").unwrap();
    for name in [
        "one",
        "two",
        "index.noun",
        "index.verb",
        "index.adj",
        "index.adv",
    ] {
        fs::write(temp.path().join(name), "").unwrap();
    }
    let app = Desktop::new(temp.path().join("settings")).unwrap();
    let path = |name| temp.path().join(name).to_string_lossy().into_owned();
    let mut inner = app.inner.lock().unwrap();
    inner.settings.corpora = settings::Corpora {
        dictionary: path("dictionary"),
        unigrams: path("one"),
        bigrams: path("two"),
        wordnet: path(""),
        phrase: String::new(),
    };
    let mut request = inner.settings.request.clone();
    request["generation"]["text"] = "ate".into();
    request["generation"]["min_zipf"] = 0.into();
    let gate = std::sync::Arc::new(std::sync::Barrier::new(2));
    inner.worker_gate = Some(gate.clone());
    drop(inner);
    app.start(request.clone()).unwrap();
    assert!(app.start(request).unwrap_err().contains("already running"));
    let started = Instant::now();
    thread::scope(|scope| {
        let shutdown = scope.spawn(|| app.shutdown());
        while !app.inner.lock().unwrap().closing {
            assert!(started.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(1));
        }
        gate.wait();
        shutdown.join().unwrap();
    });
    assert!(started.elapsed() < Duration::from_secs(5));
    let inner = app.inner.lock().unwrap();
    assert!(!inner.job.active);
    assert!(inner.worker.is_none());
    assert!(inner.job.result.is_none());
    assert_eq!(
        inner.job.status.as_ref().unwrap().state,
        anagram_core::contracts::JobState::Cancelled
    );
}
