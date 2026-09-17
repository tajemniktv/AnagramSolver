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
        solve::solve_observed(
            &request,
            solve::Paths {
                dictionary: &self.0.join("dictionary"),
                unigrams: &self.0.join("one"),
                bigrams: &self.0.join("two"),
                wordnet: &self.0,
                phrase: None,
            },
            control,
            limits,
            "test-job",
            observer,
        )
    }
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
    for pair in events.windows(2) {
        assert!(pair[0].counts.deep_analyzed <= pair[1].counts.deep_analyzed);
        assert!(pair[0].counts.orders_evaluated <= pair[1].counts.orders_evaluated);
    }
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
