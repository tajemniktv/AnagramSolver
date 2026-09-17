use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn invoke(request: Value) -> (bool, Value) {
    invoke_with_args(request, &[])
}
fn invoke_with_args(request: Value, args: &[&str]) -> (bool, Value) {
    let dictionary =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/dictionary.txt");
    let mut child = Command::new(env!("CARGO_BIN_EXE_anagram-cli"))
        .args(["generate", dictionary.to_str().unwrap()])
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    (
        output.status.success(),
        serde_json::from_slice(&output.stdout).unwrap(),
    )
}

#[test]
fn expired_deadline_is_not_reported_as_corpus_failure() {
    let (ok, output) = invoke_with_args(request(), &["--timeout-ms", "0"]);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "timed_out");
    let (ok, output) = invoke_with_args(request(), &["--timeout-ms", "bad"]);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "invalid_timeout");
    let mut zero_residual = request();
    zero_residual["required"] = json!(["ate"]);
    let (ok, output) = invoke_with_args(zero_residual, &["--timeout-ms", "0"]);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "timed_out");
}

fn request() -> Value {
    json!({"schema_version":1,"text":"ate", "required":[],"hints":[],"excluded":[],
        "min_words":1,"max_words":3,"min_word_length":1,"max_word_length":20,
        "min_zipf":0,"candidate_budget":1,"allow_repeat":true,"strategy":"prefix","hint_mode":"any"})
}

#[test]
fn cli_proves_cap_and_labels_unranked_output() {
    let (ok, output) = invoke(request());
    assert!(ok);
    assert_eq!(output["kind"], "generation_only");
    assert_eq!(output["stop"], "candidate_cap");
    assert_eq!(output["bags"], json!([["ate"]]));
}

#[test]
fn explicit_errors_and_required_multiplicity() {
    let mut input = request();
    input["schema_version"] = json!(2);
    let (ok, output) = invoke(input);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "unsupported_version");
    let mut input = request();
    input["required"] = json!(["ate"]);
    input["excluded"] = json!(["ate"]);
    let (ok, output) = invoke(input);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "conflicting_constraints");
    let mut input = request();
    input["text"] = json!("ateate");
    input["required"] = json!(["ate", "ate"]);
    let (ok, output) = invoke(input);
    assert!(ok);
    assert_eq!(output["bags"], json!([["ate", "ate"]]));
    assert_eq!(output["stop"], "exhausted");
}

#[test]
fn validation_precedes_corpus_io_and_rejects_unsafe_wire_integers() {
    for (field, value, code) in [
        ("schema_version", json!(2), "unsupported_version"),
        (
            "candidate_budget",
            json!(9_007_199_254_740_992_u64),
            "invalid_limits",
        ),
        ("text", json!("!!!"), "empty_input"),
    ] {
        let mut input = request();
        input[field] = value;
        let mut child = Command::new(env!("CARGO_BIN_EXE_anagram-cli"))
            .args(["generate", "nonexistent-corpus-for-validation-test.txt"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(!output.status.success());
        let result: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(result["error"]["code"], code);
    }
}

#[test]
fn adapter_policy_rejects_over_budget_work_without_clamping_semantics() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/fixtures/deployment-bounded.json");
    let args = ["--limits", path.to_str().unwrap()];
    let (ok, output) = invoke_with_args(request(), &args);
    assert!(ok);
    assert_eq!(output["candidate_budget"], 1);
    for budget in [0, 2] {
        let mut input = request();
        input["candidate_budget"] = json!(budget);
        let (ok, output) = invoke_with_args(input, &args);
        assert!(!ok);
        assert_eq!(output["error"]["code"], "deployment_limit_exceeded");
    }
}
