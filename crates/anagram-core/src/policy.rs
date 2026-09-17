//! Adapter-owned deployment policy, separate from semantic search requests.
use crate::{control::Control, normalize_letters, request::Error, solve::Request};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct DeploymentLimits {
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_input_bytes: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_normalized_letters: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_candidates: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_deep_analyzed: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_beam_width: Option<usize>,
    #[schemars(range(min = 1, max = 9007199254740991_u64))]
    pub max_retained_orders: Option<usize>,
    #[schemars(range(max = 9007199254740991_u64))]
    pub timeout_ms: Option<u64>,
}

impl DeploymentLimits {
    pub fn validate(&self) -> Result<(), Error> {
        let values = [
            self.max_input_bytes,
            self.max_normalized_letters,
            self.max_candidates,
            self.max_deep_analyzed,
            self.max_beam_width,
            self.max_retained_orders,
        ];
        if values
            .into_iter()
            .flatten()
            .any(|n| n == 0 || n as u128 > 9_007_199_254_740_991)
            || self.timeout_ms.is_some_and(|n| n > 9_007_199_254_740_991)
        {
            return Err(Error::new(
                "invalid_deployment_limits",
                "Limits must be positive exact JSON integers; timeout may be zero",
            ));
        }
        Ok(())
    }

    /// Reject rather than silently narrowing a requested semantic workload.
    pub fn admit(&self, request: &Request) -> Result<(), Error> {
        self.admit_generation(&request.generation)?;
        check(request.beam_width, self.max_beam_width, "beam width")?;
        check(
            request.retained_orders,
            self.max_retained_orders,
            "retained orders",
        )
    }

    pub fn admit_generation(&self, request: &crate::request::GenerateRequest) -> Result<(), Error> {
        self.validate()?;
        for (actual, maximum, name) in [
            (request.text.len(), self.max_input_bytes, "input bytes"),
            (
                normalize_letters(&request.text).len(),
                self.max_normalized_letters,
                "normalized letters",
            ),
        ] {
            check(actual, maximum, name)?;
        }
        if self.max_candidates.is_some() && request.candidate_budget == 0 {
            return Err(Error::new(
                "deployment_limit_exceeded",
                "Unbounded generation is not admitted by this deployment",
            ));
        }
        check(
            request.candidate_budget,
            self.max_candidates,
            "candidate budget",
        )
    }

    /// Morphology expansion happens before this check; it cannot bypass the cap.
    pub fn admit_deep(&self, selected: usize) -> Result<(), Error> {
        self.validate()?;
        check(
            selected,
            self.max_deep_analyzed,
            "family-expanded deep shortlist",
        )
    }

    pub fn control(&self, control: &Control) -> Result<Control, Error> {
        self.validate()?;
        let Some(millis) = self.timeout_ms else {
            return Ok(control.clone());
        };
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(millis))
            .ok_or_else(|| {
                Error::new(
                    "invalid_deployment_limits",
                    "Deadline exceeds platform clock range",
                )
            })?;
        Ok(control.with_earlier_deadline(deadline))
    }
}

fn check(actual: usize, maximum: Option<usize>, name: &str) -> Result<(), Error> {
    if maximum.is_some_and(|max| actual > max) {
        Err(Error::new(
            "deployment_limit_exceeded",
            format!("{name} exceeds deployment limit"),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_policy_deadline_never_extends_or_detaches_existing_control() {
        let control = Control::with_deadline(Instant::now());
        let policy = DeploymentLimits {
            timeout_ms: Some(60_000),
            ..Default::default()
        };
        let limited = policy.control(&control).unwrap();
        assert_eq!(limited.deadline(), control.deadline());
        control.cancel();
        assert_eq!(limited.check(), Err("cancelled"));
    }

    #[test]
    fn family_expansion_is_checked_against_a_hard_work_limit() {
        let policy = DeploymentLimits {
            max_deep_analyzed: Some(3),
            ..Default::default()
        };
        assert!(policy.admit_deep(3).is_ok());
        assert_eq!(
            policy.admit_deep(4).unwrap_err().code,
            "deployment_limit_exceeded"
        );
    }
}
