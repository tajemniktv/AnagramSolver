//! Transport-neutral progress contract; no scheduler or HTTP implementation.
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

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
    pub generated: usize,
    pub deep_selected: usize,
    pub deep_analyzed: usize,
    pub orders_evaluated: usize,
    pub corpus_rescored: usize,
    pub shown: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct EffectiveBudgets {
    /// None means unbounded, unlike a displayed result limit.
    pub candidate_limit: Option<usize>,
    pub deep_per_group: usize,
    /// Hard work cap, separate from semantic family-expanding shortlist size.
    pub deep_limit: Option<usize>,
    pub result_limit_per_group: usize,
    pub deadline_ms: Option<u64>,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DataIdentity {
    pub role: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Versions {
    pub engine: String,
    pub ranking: String,
    pub data: Vec<DataIdentity>,
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
    pub schema_version: u32,
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
}
