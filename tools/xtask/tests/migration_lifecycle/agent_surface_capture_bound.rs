//! Actual controller checks for the maximum supported soak capture and evidence bounds.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, RawCaptureOptions, Readiness,
    Value,
};
use std::{
    collections::BTreeMap, ffi::OsString, fs, num::NonZeroUsize, path::Path, time::Duration,
};

fn invoke(root: &Path, arguments: Vec<OsString>) -> (bool, String, String) {
    let mut args = vec!["automation".into()];
    args.extend(arguments);
    let environment: BTreeMap<_, _> = std::env::var_os("PATH")
        .map(|value| ("PATH".into(), Value::Public(value)))
        .into_iter()
        .collect();
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: args.into_iter().map(Value::Public).collect(),
            cwd: root.to_owned(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(30),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(4096),
            stderr: NonZeroUsize::new(4096),
        },
    )
    .unwrap();
    assert_eq!(result.process.outcome, Outcome::Exited);
    assert!(result.process.failure.is_none() && result.process.cleanup.complete);
    (
        result.process.status.unwrap().success(),
        String::from_utf8_lossy(result.stdout.as_ref().unwrap().as_bytes()).into_owned(),
        String::from_utf8_lossy(result.stderr.as_ref().unwrap().as_bytes()).into_owned(),
    )
}

#[test]
fn largest_accepted_soak_document_remains_valid_surface_capture_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let request = root.join("maximum request.json");
    let (ok, stdout, stderr) = invoke(
        &root,
        vec![
            "agent-fixture-inputs".into(),
            "soak".into(),
            "fixture".into(),
            "8388608".into(),
            request.clone().into_os_string(),
        ],
    );
    assert!(ok && stdout.is_empty(), "{stderr}");
    let mut body: serde_json::Value = serde_json::from_slice(&fs::read(request).unwrap()).unwrap();
    body["tools"] = serde_json::json!([{"type":"function","function":{"name":"fixture","parameters":{"type":"object"}}}]);
    body["tool_choice"] = serde_json::json!("auto");
    body["parallel_tool_calls"] = serde_json::json!(true);
    body["messages"].as_array_mut().unwrap().extend([
        serde_json::json!({"role":"assistant","tool_calls":[{"id":"fixture"}]}),
        serde_json::json!({"role":"tool","content":"fixture"}),
    ]);
    let mut streaming = body.clone();
    streaming["stream"] = serde_json::json!(true);
    let rows = [
        serde_json::json!({"method":"GET","path":"/v1/models","body":null}),
        serde_json::json!({"method":"POST","path":"/v1/chat/completions","body":body}),
        serde_json::json!({"method":"POST","path":"/v1/chat/completions","body":streaming}),
    ];
    let capture = root.join("maximum capture.jsonl");
    fs::write(
        &capture,
        rows.iter()
            .map(serde_json::Value::to_string)
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();
    assert!(fs::metadata(&capture).unwrap().len() > 8 * 1024 * 1024);
    let (ok, stdout, stderr) = invoke(
        &root,
        vec![
            "agent-fixture-evidence".into(),
            "surface".into(),
            capture.into_os_string(),
            "8388608".into(),
        ],
    );
    assert!(ok, "{stderr}");
    assert!(stdout.contains("captured requests: models=1 chat=2"));
}

#[test]
fn surplus_surface_and_generic_evidence_keep_distinct_actual_cli_limits() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let path = root.join("oversized evidence");
    for (verb, size, argument, expected) in [
        ("surface", 64_u64 * 1024 * 1024 + 1, "0", "64 MiB"),
        ("probe", 8_u64 * 1024 * 1024 + 1, "fixture", "8 MiB"),
    ] {
        fs::File::create(&path).unwrap().set_len(size).unwrap();
        let (ok, stdout, stderr) = invoke(
            &root,
            vec![
                "agent-fixture-evidence".into(),
                verb.into(),
                path.clone().into_os_string(),
                argument.into(),
            ],
        );
        assert!(!ok && stdout.is_empty());
        assert!(stderr.contains(expected), "{stderr}");
    }
}
