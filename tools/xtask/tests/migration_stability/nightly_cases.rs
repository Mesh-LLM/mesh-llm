//! Nightly plan, reporting, and failure evidence through the real CLI.
use super::*;

#[test]
fn stability_cli_nightly_plan_is_repeatable_and_skip_streaming_is_explicit() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("uncreated evidence");
    let mut args = arguments("nightly", "https://fixture.invalid/service", &output);
    args.extend(
        [
            "--models",
            "auto,mesh",
            "--attempts",
            "2",
            "--agent-smokes",
            "opencode,pi,goose",
            "--skip-streaming",
            "--print-plan",
        ]
        .map(str::to_owned),
    );
    let first = fixture::invoke(directory.path(), &args);
    let second = fixture::invoke(directory.path(), &args);
    assert_eq!(first.process.status.unwrap().code(), Some(0));
    assert_eq!(second.process.status.unwrap().code(), Some(0));
    assert_eq!(
        first.stdout.as_ref().unwrap().as_bytes(),
        second.stdout.as_ref().unwrap().as_bytes()
    );
    let plan: Value = serde_json::from_slice(first.stdout.unwrap().as_bytes()).unwrap();
    assert_eq!(plan["models"], json!(["auto", "mesh"]));
    assert_eq!(plan["steps"][0]["phases"], json!(["models", "chat"]));
    let checks = plan["steps"][1]["plan"]["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 4);
    for check in checks {
        assert_eq!(check["phases"], json!(["tool_call", "tool_result"]));
    }
    let names = plan["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|step| step["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            "openai-surface-probe",
            "tool-call-reliability",
            "opencode-agent-smoke",
            "pi-agent-smoke",
            "goose-agent-smoke"
        ]
    );
    assert!(!output.exists());
}

#[test]
fn stability_cli_nightly_surface_failures_retain_metadata_and_continue_cohort() {
    for http_failure in [false, true] {
        let status = if http_failure { 503 } else { 202 };
        let chat = if http_failure {
            json!({"error":"fixture unavailable"})
        } else {
            answer("WRONG")
        };
        let stream = if http_failure {
            json!({"error":"fixture unavailable"}).to_string()
        } else {
            sse(&[
                json!({"model":"stream-resolved","choices":[{"delta":{"content":"WRONG"}}]}),
                json!({"usage":{"completion_tokens":7}}),
            ])
        };
        let server = fixture::Server::new(vec![
            (200, json!({"data":[{"id":"fixture"}]}).to_string(), false),
            (status, chat.to_string(), false),
            (status, stream, true),
            (200, call().to_string(), false),
            (200, answer("signal-7429").to_string(), false),
            (200, sse(&stream_call()), true),
            (
                200,
                sse(&[json!({"choices":[{"delta":{"content":"signal-7429"}}]})]),
                true,
            ),
        ]);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evidence");
        let result = fixture::invoke(
            directory.path(),
            &arguments("nightly", &server.base, &output),
        );
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let rows = fixture::rows(&output.join("results.jsonl"));
        assert_eq!(rows.len(), 3);
        for row in &rows[1..] {
            assert_eq!(row["ok"], false);
            assert_eq!(row["status_code"], status);
            if http_failure {
                assert!(row["actual_model"].is_null());
                assert!(row["tok_per_sec"].is_null());
            } else {
                assert!(row["actual_model"].is_string());
                assert!(row["tok_per_sec"].as_f64().unwrap() > 0.0);
            }
        }
        if !http_failure {
            assert!(rows[2]["ttft_ms"].is_u64());
        }
        let requests = server.requests.lock().unwrap();
        assert_eq!(requests.len(), 7);
        for (_, payload) in &requests[1..] {
            assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], false);
        }
        let summary: Value =
            serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["ok"], false);
        assert_eq!(summary["failed"], 2);
    }
}

#[test]
fn stability_cli_nightly_skip_streaming_retains_timing_and_only_nonstreaming_requests() {
    let server = fixture::Server::new(vec![
        (200, json!({"data":[{"id":"fixture"}]}).to_string(), false),
        (200, answer("STABILITY_OK").to_string(), false),
        (200, call().to_string(), false),
        (200, answer("signal-7429").to_string(), false),
    ]);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("evidence");
    let mut args = arguments("nightly", &server.base, &output);
    args.push("--skip-streaming".into());
    let result = fixture::invoke(directory.path(), &args);
    assert_eq!(result.process.status.unwrap().code(), Some(0));
    let requests = server.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    for (_, payload) in &requests[1..] {
        assert_eq!(payload["stream"], false);
        assert_eq!(payload["chat_template_kwargs"]["enable_thinking"], false);
    }
    let probes = fixture::rows(&output.join("results.jsonl"));
    assert_eq!(
        probes
            .iter()
            .map(|row| row["phase"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["models", "chat"]
    );
    let tools = fixture::rows(&output.join("agent-tool-call-reliability/results.jsonl"));
    assert_eq!(
        tools
            .iter()
            .map(|row| row["phase"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["tool_call", "tool_result"]
    );
    let summary: Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    let markdown = fs::read_to_string(output.join("summary.md")).unwrap();
    assert!(markdown.contains("## Timing snapshot"));
    for (name, key) in [
        ("OpenAI surface probes", "probes"),
        ("Command probes", "commands"),
        ("Release attestation", "attestation"),
    ] {
        let counts = &summary[key];
        let expected = format!(
            "| {name} | {} | {} | {} | {} |",
            counts["passed"], counts["failed"], counts["prereq"], counts["elapsed_ms"]
        );
        assert!(
            markdown.contains(&expected),
            "missing timing evidence: {expected}"
        );
    }
}

#[test]
fn stability_cli_scoped_help_documents_evidence_without_network_or_output() {
    let directory = tempfile::tempdir().unwrap();
    for mode in [None, Some("nightly"), Some("tool-call")] {
        let mut args = vec!["automation".into(), "stability".into()];
        if let Some(mode) = mode {
            args.push(mode.into());
        }
        args.push("--help".into());
        let result = fixture::invoke(directory.path(), &args);
        assert_eq!(result.process.status.unwrap().code(), Some(0));
        let bytes = result.stdout.unwrap();
        let text = std::str::from_utf8(bytes.as_bytes()).unwrap();
        for required in [
            "--base-url",
            "--models",
            "--attempts",
            "--agent-smokes",
            "--output-dir",
            "--print-plan",
            "commands.jsonl",
            "results.jsonl",
            "summary.json",
        ] {
            assert!(
                text.contains(required),
                "missing documented stability contract: {required}"
            );
        }
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }
}

#[test]
fn stability_cli_unknown_agent_is_usage_error_before_any_evidence_write() {
    let directory = tempfile::tempdir().unwrap();
    let args = [
        "automation",
        "stability",
        "nightly",
        "--agent-smokes",
        "unknown",
        "--print-plan",
    ]
    .map(str::to_owned);
    let result = fixture::invoke(directory.path(), &args);
    assert_eq!(result.process.status.unwrap().code(), Some(2));
    assert!(
        std::str::from_utf8(result.stderr.unwrap().as_bytes())
            .unwrap()
            .contains("unknown agent smoke")
    );
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
}
