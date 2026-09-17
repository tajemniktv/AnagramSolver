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
fn regex_exclusions_filter_and_validate_constraints() {
    let mut input = request();
    input["candidate_budget"] = json!(0);
    input["min_words"] = json!(1);
    input["max_words"] = json!(1);
    input["exclude_regex"] = json!(["^AT", "EA$"]);
    let (ok, result) = invoke(input.clone());
    assert!(ok, "{result}");
    assert_eq!(result["bags"], json!([["eat"]]));

    input["required"] = json!(["ate"]);
    assert_eq!(
        invoke(input.clone()).1["error"]["code"],
        "conflicting_constraints"
    );
    input["required"] = json!([]);
    input["hints"] = json!(["tea"]);
    assert_eq!(invoke(input.clone()).1["error"]["code"], "impossible_hints");
    input["hints"] = json!([]);
    for patterns in [
        json!(["["]),
        json!(["(?=a)"]),
        json!(["a".repeat(16_385)]),
        json!(vec!["a"; 65]),
    ] {
        input["exclude_regex"] = patterns;
        assert_eq!(invoke(input.clone()).1["error"]["code"], "invalid_regex");
    }
}

#[test]
fn cli_cache_controls_reuse_rebuild_and_reject_invalid_options() {
    for args in [
        vec!["--rebuild-cache"],
        vec!["--cache"],
        vec!["--cache", "unused", "--cache-max-bytes", "0"],
        vec!["--cache", "unused"],
        vec!["--cache", "one", "--cache", "two"],
    ] {
        assert!(!invoke_with_args(request(), &args).0);
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.codex/temp");
    std::fs::create_dir_all(&root).unwrap();
    let directory = root.join(format!("cli-cache-{}", std::process::id()));
    std::fs::create_dir(&directory).unwrap();
    let dictionary = directory.join("dictionary");
    let one = directory.join("one");
    let two = directory.join("two");
    let cache = directory.join("native.sqlite");
    std::fs::write(&dictionary, "ate\neat\ntea\n").unwrap();
    std::fs::write(&one, "ate\t100\neat\t100\ntea\t100\n").unwrap();
    std::fs::write(&two, "").unwrap();
    for name in ["index.noun", "index.verb", "index.adj", "index.adv"] {
        std::fs::write(directory.join(name), "").unwrap();
    }
    let fixtures: Value =
        serde_json::from_str(include_str!("../../../contracts/fixtures.json")).unwrap();
    let request = &fixtures
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["schema"] == "SolveRequest" && f["valid"] == true)
        .unwrap()["value"];
    let run = |rebuild: bool| {
        let mut command = Command::new(env!("CARGO_BIN_EXE_anagram-cli"));
        command
            .arg("solve")
            .args([&dictionary, &one, &two, &directory])
            .arg("--cache")
            .arg(&cache)
            .args([
                "--cache-max-entries",
                "2",
                "--cache-max-bytes",
                "1000000",
                "--progress",
            ]);
        if rebuild {
            command.arg("--rebuild-cache");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .take()
            .unwrap()
            .write_all(request.to_string().as_bytes())
            .unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stdout)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        let events: Vec<Value> = String::from_utf8(output.stderr)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(events.last().unwrap(), &value["status"]);
        value
    };
    let cold = run(false);
    let warm = run(false);
    let rebuilt = run(true);
    assert_eq!(cold["status"]["cache"]["hit"], false);
    assert_eq!(warm["status"]["cache"]["hit"], true);
    assert_eq!(rebuilt["status"]["cache"]["hit"], false);
    assert_eq!(rebuilt["status"]["cache"]["rebuilt"], true);
    assert_eq!(cold["buckets"], warm["buckets"]);
    assert_eq!(cold["buckets"], rebuilt["buckets"]);
    assert_eq!(
        cold["timings"]["ranking_computation_ms"],
        warm["timings"]["ranking_computation_ms"]
    );
    assert!(
        cold["timings"]["execution_ms"].as_u64().unwrap()
            >= cold["timings"]["ranking_computation_ms"].as_u64().unwrap()
    );
    assert!(warm["timings"]["execution_ms"].is_u64());
    std::fs::remove_dir_all(directory).unwrap();
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
fn forbidden_letters_are_normalized_and_cannot_be_forced() {
    let mut input = request();
    input["forbid_chars"] = json!("É!");
    let (ok, output) = invoke(input.clone());
    assert!(ok);
    assert_eq!(output["bags"], json!([]));
    input["required"] = json!(["ate"]);
    let (ok, output) = invoke(input.clone());
    assert!(!ok);
    assert_eq!(output["error"]["code"], "conflicting_constraints");
    input["required"] = json!([]);
    input["hints"] = json!(["tea"]);
    let (ok, output) = invoke(input);
    assert!(!ok);
    assert_eq!(output["error"]["code"], "impossible_hints");
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
