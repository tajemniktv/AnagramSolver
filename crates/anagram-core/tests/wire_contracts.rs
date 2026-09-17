use anagram_core::{contracts::JobStatus, request::GenerateRequest, solve::Request};
use serde_json::Value;

#[test]
fn shared_request_and_progress_fixtures_match_rust_deserialization() {
    let fixtures: Vec<Value> =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let mut checked = 0;
    for fixture in fixtures {
        let value = fixture["value"].clone();
        let accepted = match fixture["schema"].as_str().unwrap() {
            "GenerateRequest" => serde_json::from_value::<GenerateRequest>(value).is_ok(),
            "SolveRequest" => serde_json::from_value::<Request>(value).is_ok(),
            "JobStatus" => serde_json::from_value::<JobStatus>(value).is_ok(),
            // Output-only types are validated from actual CLI serialization by
            // the Python contract consumer, not deserialized by the engine.
            _ => continue,
        };
        assert_eq!(accepted, fixture["valid"].as_bool().unwrap(), "{fixture}");
        checked += 1;
    }
    assert!(checked >= 4);
}
