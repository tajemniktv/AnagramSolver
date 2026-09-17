use anagram_core::{
    contracts::{Exhaustion, JobState, JobStatus, Stage},
    control::Control,
    policy::DeploymentLimits,
    solve,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[test]
fn parallel_refinement_model_and_cache_remain_controlled_and_identified() {
    let corpora = Corpora::new();
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let mut request: solve::Request = serde_json::from_value(
        fixtures
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["schema"] == "SolveRequest" && f["valid"] == true)
            .unwrap()["value"]
            .clone(),
    )
    .unwrap();
    request.generation.text = "ateate".into();
    request.generation.allow_repeat = true;
    request.generation.candidate_budget = 0;
    request.generation.min_words = 2;
    request.generation.max_words = 2;
    request.generation.min_zipf = 0.0;
    request.deep_all = true;
    let cache_path = corpora.0.join("enhanced.sqlite");
    let config = anagram_core::cache::Config {
        path: &cache_path,
        limits: anagram_core::cache::StorageLimits {
            max_entries: 8,
            max_payload_bytes: 1000000,
        },
        rebuild: false,
    };
    let run = |request: &solve::Request,
               model: Option<&std::path::Path>,
               control: &Control,
               observer: &mut dyn FnMut(&JobStatus)| {
        solve::solve_cached_observed(
            request,
            solve::Paths {
                dictionary: &corpora.0.join("dictionary"),
                unigrams: &corpora.0.join("one"),
                bigrams: &corpora.0.join("two"),
                wordnet: &corpora.0,
                phrase: None,
                model,
            },
            control,
            &DeploymentLimits::default(),
            ("enhanced", observer),
            Some(&config),
        )
    };
    let serial = run(&request, None, &Control::default(), &mut |_| {}).unwrap();
    request.workers = 0;
    let automatic = run(&request, None, &Control::default(), &mut |_| {}).unwrap();
    assert_eq!(
        serde_json::to_value(&serial.buckets).unwrap(),
        serde_json::to_value(&automatic.buckets).unwrap()
    );
    request.workers = 2;
    let mut counts = Vec::new();
    let parallel = run(&request, None, &Control::default(), &mut |s| {
        counts.push((s.counts.deep_analyzed, s.counts.orders_evaluated))
    })
    .unwrap();
    assert!(parallel.deep_analyzed > 1);
    assert_eq!(
        serde_json::to_value(&serial.buckets).unwrap(),
        serde_json::to_value(&parallel.buckets).unwrap()
    );
    assert_eq!(serial.orders_evaluated, parallel.orders_evaluated);
    assert!(
        counts
            .windows(2)
            .all(|p| p[0].0 <= p[1].0 && p[0].1 <= p[1].1)
    );
    request.refine = true;
    let refined = run(&request, None, &Control::default(), &mut |_| {}).unwrap();
    assert!(refined.orders_evaluated > parallel.orders_evaluated);
    for row in refined.buckets.values().flatten() {
        assert_eq!(
            anagram_core::normalize_letters(&row.display_phrase).len(),
            6
        );
    }
    let model_path = corpora.0.join("model.json");
    fs::write(
        &model_path,
        serde_json::to_vec(&anagram_core::learned::Model::new([0.25; 18]).unwrap()).unwrap(),
    )
    .unwrap();
    let modeled = run(
        &request,
        Some(&model_path),
        &Control::default(),
        &mut |_| {},
    )
    .unwrap();
    assert!(!modeled.status.cache.hit);
    assert!(
        modeled
            .status
            .versions
            .data
            .iter()
            .any(|d| d.role == "learned_model")
    );
    assert!(
        run(
            &request,
            Some(&model_path),
            &Control::default(),
            &mut |_| {}
        )
        .unwrap()
        .status
        .cache
        .hit
    );
    fs::write(
        &model_path,
        serde_json::to_vec(&anagram_core::learned::Model::new([0.5; 18]).unwrap()).unwrap(),
    )
    .unwrap();
    assert!(
        !run(
            &request,
            Some(&model_path),
            &Control::default(),
            &mut |_| {}
        )
        .unwrap()
        .status
        .cache
        .hit
    );
    request.generation.text = "ateateate".into();
    request.generation.max_words = 3;
    let control = Control::default();
    let cancelled = run(&request, None, &control, &mut |s| {
        if s.counts.deep_analyzed > 0 {
            control.cancel();
        }
    });
    assert_eq!(cancelled.err().unwrap().code, "cancelled");
    // A deliberately adverse model must actually affect selected orders, not
    // merely appear in provenance. It is not installed as a quality model.
    fs::write(corpora.0.join("dictionary"), "we\nare\nhome\n").unwrap();
    fs::write(corpora.0.join("one"), "we\t100\nare\t100\nhome\t100\n").unwrap();
    request.generation.text = "wearehome".into();
    request.generation.required = vec!["we".into(), "are".into(), "home".into()];
    request.generation.min_words = 3;
    request.refine = false;
    let baseline = run(&request, None, &Control::default(), &mut |_| {}).unwrap();
    fs::write(
        &model_path,
        serde_json::to_vec(&anagram_core::learned::Model::new([-1.0; 18]).unwrap()).unwrap(),
    )
    .unwrap();
    let adverse = run(
        &request,
        Some(&model_path),
        &Control::default(),
        &mut |_| {},
    )
    .unwrap();
    assert_ne!(
        baseline.buckets[&3][0].best_order,
        adverse.buckets[&3][0].best_order
    );
}

#[test]
fn fully_required_answer_is_ranked_with_repeated_word_multiplicity() {
    let corpora = Corpora::new();
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let mut request: solve::Request = serde_json::from_value(
        fixtures
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["schema"] == "SolveRequest" && f["valid"] == true)
            .unwrap()["value"]
            .clone(),
    )
    .unwrap();
    request.generation.text = "ateate".into();
    request.generation.required = vec!["ate".into(), "ate".into()];
    request.generation.min_words = 2;
    request.generation.max_words = 2;
    let result = solve::solve(
        &request,
        solve::Paths {
            dictionary: &corpora.0.join("dictionary"),
            unigrams: &corpora.0.join("one"),
            bigrams: &corpora.0.join("two"),
            wordnet: &corpora.0,
            phrase: None,
            model: None,
        },
    )
    .unwrap();
    assert_eq!(
        (result.generated, result.deep_analyzed, result.shown),
        (1, 1, 1)
    );
    assert_eq!(result.buckets[&2][0].best_order, ["ate", "ate"]);
    assert_eq!(result.status.state, JobState::Succeeded);
    assert_eq!(result.status.exhaustion, Exhaustion::Exhausted);
}

#[test]
fn cancellation_preserves_completed_corpus_rows() {
    let corpora = Corpora::new();
    let control = Control::default();
    let mut events = Vec::new();
    let result = corpora.run(&DeploymentLimits::default(), &control, &mut |status| {
        events.push(status.clone());
        if status.state == JobState::Running && status.counts.corpus_rescored == 1 {
            control.cancel();
        }
    });
    assert_eq!(result.err().unwrap().code, "cancelled");
    let terminal = events.last().unwrap();
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(terminal.counts.corpus_rescored, 1);
    assert_eq!(terminal.counts.shown, 0);
}
struct Corpora(PathBuf);
impl Corpora {
    fn new() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.codex/temp");
        fs::create_dir_all(&root).unwrap();
        let path = root.canonicalize().unwrap().join(format!(
            "progress-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("dictionary"), "ate\neat\ntea\n").unwrap();
        fs::write(path.join("one"), "ate\t100\neat\t100\ntea\t100\n").unwrap();
        fs::write(path.join("two"), "").unwrap();
        for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
            fs::write(path.join(name), "").unwrap();
        }
        Self(path)
    }
    fn run(
        &self,
        limits: &DeploymentLimits,
        control: &Control,
        observer: &mut dyn FnMut(&JobStatus),
    ) -> Result<solve::Result, anagram_core::request::Error> {
        self.run_cached(limits, control, observer, None)
    }
    fn run_cached(
        &self,
        limits: &DeploymentLimits,
        control: &Control,
        observer: &mut dyn FnMut(&JobStatus),
        cache: Option<&anagram_core::cache::Config<'_>>,
    ) -> Result<solve::Result, anagram_core::request::Error> {
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
        let request: solve::Request = serde_json::from_value(
            fixtures
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["schema"] == "SolveRequest" && f["valid"] == true)
                .unwrap()["value"]
                .clone(),
        )
        .unwrap();
        solve::solve_cached_observed(
            &request,
            solve::Paths {
                dictionary: &self.0.join("dictionary"),
                unigrams: &self.0.join("one"),
                bigrams: &self.0.join("two"),
                wordnet: &self.0,
                phrase: None,
                model: None,
            },
            control,
            limits,
            ("test-job", observer),
            cache,
        )
    }
}

#[test]
fn ranked_cache_reuses_only_results_and_rechecks_live_policy_and_data() {
    use anagram_core::cache::{Config, StorageLimits};
    let corpora = Corpora::new();
    let path = corpora.0.join("cache.sqlite");
    let mut config = Config {
        path: &path,
        limits: StorageLimits {
            max_entries: 4,
            max_payload_bytes: 1_000_000,
        },
        rebuild: false,
    };
    let control = Control::default();
    let limits = DeploymentLimits::default();
    let cold = corpora
        .run_cached(&limits, &control, &mut |_| {}, Some(&config))
        .unwrap();
    assert!(!cold.status.cache.hit);
    let mut events = Vec::new();
    let warm = corpora
        .run_cached(
            &limits,
            &control,
            &mut |s| events.push(s.clone()),
            Some(&config),
        )
        .unwrap();
    assert!(warm.status.cache.hit);
    assert_eq!(
        serde_json::to_value(&cold.buckets).unwrap(),
        serde_json::to_value(&warm.buckets).unwrap()
    );
    assert!(events.iter().all(|s| s.validate().is_ok()));
    assert!(!events.iter().any(|s| s.stage == Stage::DeepRanking));
    assert!(!events.iter().any(|s| s.stage == Stage::Generating));
    assert_eq!(warm.generation_stop, cold.generation_stop);
    assert_eq!(warm.generated, cold.generated);
    let denied = DeploymentLimits {
        max_deep_analyzed: Some(1),
        ..Default::default()
    };
    assert_eq!(
        corpora
            .run_cached(&denied, &control, &mut |_| {}, Some(&config))
            .err()
            .unwrap()
            .code,
        "deployment_limit_exceeded"
    );
    config.rebuild = true;
    let rebuilt = corpora
        .run_cached(&limits, &control, &mut |_| {}, Some(&config))
        .unwrap();
    assert!(rebuilt.status.cache.rebuilt && !rebuilt.status.cache.hit);
    config.rebuild = false;
    fs::write(corpora.0.join("dictionary"), "ate\neat\n").unwrap();
    let changed = corpora
        .run_cached(&limits, &control, &mut |_| {}, Some(&config))
        .unwrap();
    assert!(!changed.status.cache.hit);
    assert_eq!(changed.generated, 2);
    let writer = rusqlite::Connection::open(&path).unwrap();
    writer
        .execute("UPDATE native_cache_v1 SET payload=x'00'", [])
        .unwrap();
    drop(writer);
    let recovered = corpora
        .run_cached(&limits, &control, &mut |_| {}, Some(&config))
        .unwrap();
    assert!(recovered.status.cache.corrupt_entry_ignored && !recovered.status.cache.hit);
    let cancel = Control::default();
    let mut terminal = None;
    let interrupted = corpora.run_cached(
        &limits,
        &cancel,
        &mut |status| {
            if status.stage == Stage::Finalizing {
                cancel.cancel();
            }
            terminal = Some(status.clone());
        },
        Some(&config),
    );
    assert_eq!(interrupted.err().unwrap().code, "cancelled");
    assert_eq!(terminal.unwrap().counts.shown, 0);
    fs::write(&path, b"not a database").unwrap();
    let unavailable = corpora
        .run_cached(&limits, &control, &mut |_| {}, Some(&config))
        .unwrap();
    assert!(!unavailable.status.cache.hit);
    assert_eq!(fs::read(path).unwrap(), b"not a database");
}
impl Drop for Corpora {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn real_execution_emits_monotone_counts_and_exact_terminal_result() {
    let corpora = Corpora::new();
    let mut events = Vec::new();
    let result = corpora
        .run(
            &DeploymentLimits::default(),
            &Control::default(),
            &mut |event| events.push(event.clone()),
        )
        .unwrap();
    assert_eq!(events.first().unwrap().state, JobState::Queued);
    assert_eq!(events.last().unwrap().state, JobState::Succeeded);
    assert_eq!(
        serde_json::to_value(events.last().unwrap()).unwrap(),
        serde_json::to_value(&result.status).unwrap()
    );
    assert_eq!(result.status.counts.generated, 3);
    assert_eq!(result.status.counts.deep_selected, 3);
    assert_eq!(result.status.counts.deep_analyzed, result.deep_analyzed);
    assert_eq!(
        result.status.counts.orders_evaluated,
        result.orders_evaluated
    );
    assert_eq!(result.status.counts.shown, result.shown);
    assert_eq!(result.status.exhaustion, Exhaustion::Exhausted);
    assert_eq!(result.status.stage, Stage::Complete);
    assert!(result.status.versions.data_complete);
    assert_eq!(result.status.versions.data.len(), 10);
    assert!(
        result
            .status
            .versions
            .data
            .iter()
            .any(|d| d.role == "wordnet/noun.exc" && !d.present && d.bytes == 0)
    );
    assert!(
        result
            .status
            .versions
            .data
            .iter()
            .any(|d| d.role == "wordnet/index.noun" && d.present && d.bytes == 0)
    );
    for pair in events.windows(2) {
        assert!(pair[0].counts.deep_analyzed <= pair[1].counts.deep_analyzed);
        assert!(pair[0].counts.orders_evaluated <= pair[1].counts.orders_evaluated);
    }
}

#[test]
fn interrupted_first_bag_preserves_order_evaluations_in_terminal_status() {
    let corpora = Corpora::new();
    fs::write(
        corpora.0.join("dictionary"),
        "dog\ncat\npig\nhen\ncow\nfox\n",
    )
    .unwrap();
    fs::write(
        corpora.0.join("one"),
        "dog\t100\ncat\t100\npig\t100\nhen\t100\ncow\t100\nfox\t100\n",
    )
    .unwrap();
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let mut value = fixtures
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["schema"] == "SolveRequest" && f["valid"] == true)
        .unwrap()["value"]
        .clone();
    value["generation"]["text"] = "dogcatpighencowfox".into();
    value["generation"]["min_words"] = 6.into();
    value["generation"]["max_words"] = 6.into();
    value["generation"]["allow_repeat"] = false.into();
    value["generation"]["candidate_budget"] = 1.into();
    value["order_mode"] = "exact".into();
    let request = serde_json::from_value(value).unwrap();
    let control = Control::default();
    let mut events = Vec::new();
    let result = solve::solve_observed(
        &request,
        solve::Paths {
            dictionary: &corpora.0.join("dictionary"),
            unigrams: &corpora.0.join("one"),
            bigrams: &corpora.0.join("two"),
            wordnet: &corpora.0,
            phrase: None,
            model: None,
        },
        &control,
        &DeploymentLimits::default(),
        "partial-bag",
        &mut |event| {
            events.push(event.clone());
            if event.counts.orders_evaluated == 256 {
                control.cancel();
            }
        },
    );
    assert_eq!(result.err().unwrap().code, "cancelled");
    let terminal = events.last().unwrap();
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(terminal.counts.orders_evaluated, 256);
    assert_eq!(terminal.counts.deep_analyzed, 0);
    assert_eq!(terminal.counts.shown, 0);
    assert!(terminal.validate().is_ok());
}

#[test]
fn cancellation_keeps_completed_deep_work_and_known_generation_exhaustion() {
    let corpora = Corpora::new();
    let control = Control::default();
    let mut events = Vec::new();
    let error = corpora
        .run(&DeploymentLimits::default(), &control, &mut |event| {
            events.push(event.clone());
            if event.stage == Stage::DeepRanking && event.counts.deep_analyzed == 1 {
                control.cancel();
            }
        })
        .err()
        .unwrap();
    assert_eq!(error.code, "cancelled");
    let terminal = events.last().unwrap();
    assert_eq!(terminal.state, JobState::Cancelled);
    assert_eq!(terminal.stage, Stage::DeepRanking);
    assert_eq!(terminal.counts.deep_analyzed, 1);
    assert_eq!(terminal.counts.shown, 0);
    assert_eq!(terminal.exhaustion, Exhaustion::Exhausted);
}

#[test]
fn queued_timeout_and_running_policy_failure_are_distinct() {
    let corpora = Corpora::new();
    let mut events = Vec::new();
    let error = corpora
        .run(
            &DeploymentLimits {
                timeout_ms: Some(0),
                ..Default::default()
            },
            &Control::default(),
            &mut |e| events.push(e.clone()),
        )
        .err()
        .unwrap();
    assert_eq!(error.code, "timed_out");
    assert_eq!(events.len(), 2);
    assert_eq!(events[1].state, JobState::TimedOut);
    assert_eq!(events[1].exhaustion, Exhaustion::Unknown);
    events.clear();
    let error = corpora
        .run(
            &DeploymentLimits {
                max_deep_analyzed: Some(1),
                ..Default::default()
            },
            &Control::default(),
            &mut |e| events.push(e.clone()),
        )
        .err()
        .unwrap();
    assert_eq!(error.code, "deployment_limit_exceeded");
    let terminal = events.last().unwrap();
    assert_eq!(terminal.state, JobState::Failed);
    assert_eq!(terminal.stage, Stage::Preparing);
    assert_eq!(terminal.counts.deep_selected, 3);
    assert_eq!(terminal.counts.deep_analyzed, 0);
}

#[test]
fn cancellation_during_finalization_never_claims_published_rows() {
    let corpora = Corpora::new();
    let control = Control::default();
    let mut terminal = None;
    let error = corpora
        .run(&DeploymentLimits::default(), &control, &mut |event| {
            if event.stage == Stage::Finalizing {
                control.cancel();
            }
            terminal = Some(event.clone());
        })
        .err()
        .unwrap();
    assert_eq!(error.code, "cancelled");
    let status = terminal.unwrap();
    assert_eq!(status.state, JobState::Cancelled);
    assert_eq!(status.counts.deep_analyzed, 3);
    assert_eq!(status.counts.shown, 0);
}

#[test]
fn repeated_corpus_consumers_use_the_same_captured_bytes() {
    let corpora = Corpora::new();
    let baseline = corpora
        .run(
            &DeploymentLimits::default(),
            &Control::default(),
            &mut |_| {},
        )
        .unwrap();
    let mut changed = false;
    let actual = corpora
        .run(
            &DeploymentLimits::default(),
            &Control::default(),
            &mut |event| {
                if event.stage == Stage::DeepRanking && !changed {
                    fs::write(corpora.0.join("one"), "replacement corpus\n").unwrap();
                    fs::write(corpora.0.join("two"), "replacement corpus\n").unwrap();
                    changed = true;
                }
            },
        )
        .unwrap();
    assert!(changed);
    assert_eq!(
        serde_json::to_value(&actual.buckets).unwrap(),
        serde_json::to_value(&baseline.buckets).unwrap()
    );
    assert_eq!(actual.status.versions.data, baseline.status.versions.data);
}
