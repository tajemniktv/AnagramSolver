//! Transport-neutral progress contract; no scheduler or HTTP implementation.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
pub const MAX_WIRE_INTEGER: u64 = 9_007_199_254_740_991;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Cancelled,
    TimedOut,
    Failed,
}
impl JobState {
    pub fn can_transition_to(self, next: Self) -> bool {
        matches!(
            (self, next),
            (
                Self::Queued,
                Self::Running | Self::Cancelled | Self::TimedOut | Self::Failed
            ) | (
                Self::Running,
                Self::Succeeded | Self::Cancelled | Self::TimedOut | Self::Failed
            )
        )
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Queued,
    LoadingCorpora,
    Generating,
    Preparing,
    DeepRanking,
    CorpusRanking,
    Finalizing,
    Complete,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Exhaustion {
    Exhausted,
    Truncated,
    Unknown,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    #[schemars(range(max = 9007199254740991_u64))]
    pub generated: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub deep_selected: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub deep_analyzed: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub orders_evaluated: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub corpus_rescored: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub shown: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EffectiveBudgets {
    /// None means unbounded, unlike a displayed result limit.
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub candidate_limit: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub deep_per_group: usize,
    /// Hard work cap, separate from semantic family-expanding shortlist size.
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub deep_limit: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub result_limit_per_group: usize,
    #[schemars(range(max = 9007199254740991_u64))]
    pub deadline_ms: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DataRepresentation {
    FileBytes,
    PhraseRowsV1,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataIdentity {
    #[schemars(length(min = 1))]
    pub role: String,
    pub representation: DataRepresentation,
    pub present: bool,
    #[schemars(length(min = 64, max = 64))]
    pub sha256: String,
    #[schemars(range(max = 9007199254740991_u64))]
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Versions {
    #[schemars(length(min = 1))]
    pub engine: String,
    #[schemars(length(min = 1))]
    pub ranking: String,
    pub data: Vec<DataIdentity>,
    pub data_complete: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CacheFlags {
    pub hit: bool,
    pub rebuilt: bool,
    pub corrupt_entry_ignored: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Failure {
    pub code: String,
    pub message: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobStatus {
    #[schemars(range(min = 1, max = 1))]
    pub schema_version: u32,
    #[schemars(length(min = 1))]
    pub job_id: String,
    pub state: JobState,
    pub stage: Stage,
    pub counts: Counts,
    pub budgets: EffectiveBudgets,
    pub versions: Versions,
    pub cache: CacheFlags,
    pub exhaustion: Exhaustion,
    pub error: Option<Failure>,
}

impl JobStatus {
    /// Semantic invariants that structural JSON Schema cannot express.
    pub fn validate(&self) -> Result<(), &'static str> {
        let c = &self.counts;
        let b = &self.budgets;
        if self.schema_version != 1
            || self.job_id.is_empty()
            || self.versions.engine.is_empty()
            || self.versions.ranking.is_empty()
        {
            return Err("invalid status identity or version");
        }
        if [
            c.generated,
            c.deep_selected,
            c.deep_analyzed,
            c.orders_evaluated,
            c.corpus_rescored,
            c.shown,
            b.deep_per_group,
            b.result_limit_per_group,
        ]
        .into_iter()
        .any(|n| n as u128 > MAX_WIRE_INTEGER as u128)
            || b.deep_per_group == 0
            || b.result_limit_per_group == 0
            || [b.candidate_limit, b.deep_limit]
                .into_iter()
                .flatten()
                .any(|n| n == 0 || n as u128 > MAX_WIRE_INTEGER as u128)
            || b.deadline_ms.is_some_and(|n| n > MAX_WIRE_INTEGER)
        {
            return Err("invalid status numeric bounds");
        }
        if c.deep_selected > c.generated
            || c.deep_analyzed > c.deep_selected
            || c.shown > c.generated
            || c.corpus_rescored > c.generated
            || b.candidate_limit.is_some_and(|n| c.generated > n)
            || b.deep_limit.is_some_and(|n| c.deep_analyzed > n)
        {
            return Err("inconsistent completed-work counts");
        }
        if (self.exhaustion == Exhaustion::Truncated && b.candidate_limit != Some(c.generated))
            || (matches!(self.stage, Stage::Queued | Stage::LoadingCorpora)
                && (c.generated != 0
                    || c.orders_evaluated != 0
                    || self.exhaustion != Exhaustion::Unknown))
            || (matches!(self.stage, Stage::Generating | Stage::Preparing)
                && (c.deep_analyzed != 0 || c.orders_evaluated != 0 || c.corpus_rescored != 0))
            || (self.stage == Stage::Generating && c.deep_selected != 0)
        {
            return Err("counts or exhaustion do not match the execution stage");
        }
        let mut roles = std::collections::HashSet::new();
        for data in &self.versions.data {
            if data.role.is_empty()
                || !roles.insert(&data.role)
                || data.bytes > MAX_WIRE_INTEGER
                || data.sha256.len() != 64
                || !data
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
                || (!data.present
                    && (data.bytes != 0 || data.representation != DataRepresentation::FileBytes))
                || (data.bytes == 0
                    && data.sha256
                        != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            {
                return Err("invalid data identity");
            }
        }
        let error = self.error.as_ref().map(|e| e.code.as_str());
        let valid = match self.state {
            JobState::Queued => {
                self.stage == Stage::Queued
                    && error.is_none()
                    && c.generated == 0
                    && c.orders_evaluated == 0
                    && self.exhaustion == Exhaustion::Unknown
            }
            JobState::Running => {
                !matches!(self.stage, Stage::Queued | Stage::Complete)
                    && error.is_none()
                    && c.shown == 0
            }
            JobState::Succeeded => {
                self.stage == Stage::Complete
                    && error.is_none()
                    && self.exhaustion != Exhaustion::Unknown
                    && self.versions.data_complete
            }
            JobState::Cancelled => {
                self.stage != Stage::Complete && error == Some("cancelled") && c.shown == 0
            }
            JobState::TimedOut => {
                self.stage != Stage::Complete && error == Some("timed_out") && c.shown == 0
            }
            JobState::Failed => {
                self.stage != Stage::Complete
                    && error.is_some_and(|e| !e.is_empty() && e != "cancelled" && e != "timed_out")
                    && c.shown == 0
            }
        };
        if !valid {
            return Err("inconsistent lifecycle state, stage or error");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::JobState::*;
    #[test]
    fn terminal_jobs_cannot_restart_or_change_outcome() {
        let states = [Queued, Running, Succeeded, Cancelled, TimedOut, Failed];
        for terminal in [Succeeded, Cancelled, TimedOut, Failed] {
            for state in states {
                assert!(!terminal.can_transition_to(state));
            }
        }
        assert!(Queued.can_transition_to(Running));
        assert!(Queued.can_transition_to(Cancelled));
        assert!(!Queued.can_transition_to(Succeeded));
        assert!(Running.can_transition_to(Succeeded));
        assert!(!Running.can_transition_to(Queued));
    }

    #[test]
    fn status_rejects_false_completion_and_inconsistent_counts() {
        use super::*;
        let fixtures: serde_json::Value =
            serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
        let valid: JobStatus = serde_json::from_value(
            fixtures
                .as_array()
                .unwrap()
                .iter()
                .find(|f| f["schema"] == "JobStatus" && f["valid"] == true)
                .unwrap()["value"]
                .clone(),
        )
        .unwrap();
        assert!(valid.validate().is_ok());
        let mut bad = valid.clone();
        bad.error = None;
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.counts.deep_analyzed = 21;
        assert!(bad.validate().is_err());
        let mut bad = valid.clone();
        bad.state = JobState::Succeeded;
        bad.stage = Stage::Complete;
        bad.error = None;
        bad.exhaustion = Exhaustion::Exhausted;
        assert!(bad.validate().is_err()); // Missing complete provenance.
        let mut bad = valid;
        bad.budgets.candidate_limit = None;
        bad.exhaustion = Exhaustion::Truncated;
        assert!(bad.validate().is_err());
    }
}
