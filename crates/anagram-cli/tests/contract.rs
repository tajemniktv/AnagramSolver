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
