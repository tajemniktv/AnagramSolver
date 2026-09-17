//! Per-execution status owner. No scheduler, globals, or background work.
use crate::{
    contracts::*, control::Control, policy::DeploymentLimits, request::Error, solve::Request,
};
use std::time::Instant;

pub struct Progress<'a> {
    pub status: JobStatus,
    observer: &'a mut dyn FnMut(&JobStatus),
}

impl<'a> Progress<'a> {
    pub fn new(
        request: &Request,
        limits: &DeploymentLimits,
        control: &Control,
        job_id: &str,
        observer: &'a mut dyn FnMut(&JobStatus),
    ) -> Self {
        let status = JobStatus {
            schema_version: 1,
            job_id: job_id.to_owned(),
            state: JobState::Queued,
            stage: Stage::Queued,
            counts: Counts::default(),
            budgets: EffectiveBudgets {
                candidate_limit: (request.generation.candidate_budget != 0)
                    .then_some(request.generation.candidate_budget),
                deep_per_group: request.deep_per_group,
                deep_limit: limits.max_deep_analyzed,
                result_limit_per_group: request.result_limit_per_group,
                deadline_ms: control.deadline().map(|d| {
                    d.saturating_duration_since(Instant::now())
                        .as_millis()
                        .min(u64::MAX as u128) as u64
                }),
            },
            versions: Versions {
                engine: env!("CARGO_PKG_VERSION").to_owned(),
                ranking: "native-ranking-v2".to_owned(),
                data: vec![],
                data_complete: false,
            },
            cache: CacheFlags {
                hit: false,
                rebuilt: false,
                corrupt_entry_ignored: false,
            },
            exhaustion: Exhaustion::Unknown,
            error: None,
        };
        let mut progress = Self { status, observer };
        progress.emit();
        progress
    }
    pub fn emit(&mut self) {
        debug_assert!(
            self.status.validate().is_ok(),
            "invalid internal job status"
        );
        (self.observer)(&self.status);
    }
    pub fn start(&mut self) {
        self.transition(JobState::Running);
        self.stage(Stage::LoadingCorpora);
    }
    pub fn stage(&mut self, stage: Stage) {
        self.status.stage = stage;
        self.emit();
    }
    fn transition(&mut self, state: JobState) {
        assert!(self.status.state.can_transition_to(state));
        self.status.state = state;
    }
    pub fn finish(&mut self, error: Option<&Error>) {
        let state = match error.map(|error| error.code) {
            None => JobState::Succeeded,
            Some("cancelled") => JobState::Cancelled,
            Some("timed_out") => JobState::TimedOut,
            Some(_) => JobState::Failed,
        };
        self.transition(state);
        if error.is_none() {
            self.status.stage = Stage::Complete;
        }
        self.status.error = error.map(|e| Failure {
            code: e.code.to_owned(),
            message: e.message.clone(),
        });
        self.emit();
    }
}
