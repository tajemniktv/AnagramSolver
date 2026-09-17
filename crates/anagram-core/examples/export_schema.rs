use anagram_core::{request, solve};
use schemars::schema_for;
fn main() {
    println!(
        "{}",
        serde_json::json!({
            "GenerateRequest":schema_for!(request::GenerateRequest),
            "Generated":schema_for!(request::Generated),
            "SolveRequest":schema_for!(solve::Request),
            "SolveResult":schema_for!(solve::Result),
            "SolverError":schema_for!(request::Error),
            "ErrorResponse":schema_for!(request::ErrorResponse),
        "JobStatus":schema_for!(anagram_core::contracts::JobStatus)
        ,"DeploymentLimits":schema_for!(anagram_core::policy::DeploymentLimits)
        })
    );
}
